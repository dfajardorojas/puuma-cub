"""Per-fold and pooled evaluation metrics for trained run directories.


Metrics per fold and pooled over folds, on test predictions and on validation
predictions at each run's selected epoch: MSE on the standardised scale,
MAE +/- SD in weeks, R2 (sklearn `r2_score` on GA at delivery in weeks — see
R2_STATEMENT), Dice per foreground channel, and the 37-week
accuracy/sensitivity/specificity (preterm, < 37 weeks, is the positive class).

Usage::

    python scripts/evaluate.py --runs-root outputs/runs --config cl_plus_image
    python scripts/evaluate.py --runs-root outputs/runs --config cl_plus_image \
        --split validation --out metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as _sklearn_r2_score

#: Preterm threshold, weeks. Preterm is the positive class.
PRETERM_WEEKS = 37.0

#: Rows with no brain segmentation channel: an empty reference scores 1.0
#: against an empty prediction, so channel-2 Dice there is the convention, not
#: brain Dice, and is excluded.
NO_CHANNEL_2 = {"no_brain_seg"}

#: Module-level run root, set from --runs-root in main() so the verbatim
#: loaders below keep their original bodies.
RUNS = Path("outputs/runs")

R2_STATEMENT = (
    "R2 is sklearn.metrics.r2_score(y_true, y_pred) computed on GA at delivery "
    "IN WEEKS over the set being summarised  "
    "THE SCALE MATTERS FOR POOLED VALUES: the standardisation constants are "
    "per-fold, so pooled R2 in weeks and pooled R2 on the standardised scale are "
    "NOT the same number (cl_plus_image test: 0.377 weeks vs 0.381 standardised). "
    "They coincide only within a single fold, where one affine map applies."
)

# ===========================================================================
# completeness, discovered from the repository's own code
# ===========================================================================


def load_test(cfg: str, it: int) -> pd.DataFrame:
    return pd.read_csv(RUNS / f"{cfg}_iteration_{it}" / "test_predictions.csv")


def selected_epoch(cfg: str, it: int) -> tuple[int, int, int]:
    h = json.loads((RUNS / f"{cfg}_iteration_{it}" / "training_history.json").read_text())
    return (int(h["saved_checkpoint_epoch"]), int(h["argmin_epoch_ungated"]),
            int(h.get("config", {}).get("total_epochs", -1)))


def load_val(cfg: str, it: int) -> pd.DataFrame:
    sel, _, _ = selected_epoch(cfg, it)
    v = pd.read_csv(RUNS / f"{cfg}_iteration_{it}" / "validation_predictions.csv")
    return v[v["epoch"].astype(int) == sel].reset_index(drop=True)


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R2 exactly as the reference notebook computes it.

    Provenance -- the notebook's cell, verbatim:

        from sklearn.metrics import r2_score
        r2 = r2_score(y_true_test, y_pred_test)

    where ``y_true_test`` / ``y_pred_test`` are GA at delivery and its
    prediction **in weeks** (the notebook prints them as 28.28, 41.14, ...),
    for the set being summarised.  This calls the same library function on the
    same scale rather than re-deriving the formula, so the definition cannot
    drift from the notebook's.

    sklearn's default is 1 - SSres/SStot with SStot about the mean of ``true``
    within the array passed and no ddof correction.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if float(((true - true.mean()) ** 2).sum()) <= 0:
        return float("nan")
    return float(_sklearn_r2_score(true, pred))


def metrics(df: pd.DataFrame, row: str) -> dict:
    sp = df["ga_del_std_pred"].to_numpy(float)
    st = df["ga_del_std_true"].to_numpy(float)
    wp = df["ga_del_pred_weeks"].to_numpy(float)
    wt = df["ga_del_true_weeks"].to_numpy(float)
    abs_err = np.abs(wp - wt)
    out = {
        "n": int(len(df)),
        "MSE_std": float(np.mean((sp - st) ** 2)),
        "MAE_weeks": float(abs_err.mean()),
        "MAE_SD_weeks": float(abs_err.std(ddof=1)) if len(df) > 1 else float("nan"),
        "R2": r2(wt, wp),
    }
    for ch in (1, 2):
        col = f"dice_channel_{ch}"
        if col in df.columns and not (ch == 2 and row in NO_CHANNEL_2):
            v = pd.to_numeric(df[col], errors="coerce")
            out[f"dice_ch{ch}"] = float(v.mean())
            out[f"dice_ch{ch}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        else:
            out[f"dice_ch{ch}"] = float("nan")
            out[f"dice_ch{ch}_sd"] = float("nan")
    yt, yp = wt < PRETERM_WEEKS, wp < PRETERM_WEEKS
    tp, fn = int((yt & yp).sum()), int((yt & ~yp).sum())
    fp, tn = int((~yt & yp).sum()), int((~yt & ~yp).sum())
    n = len(df)
    out.update({"TP": tp, "FN": fn, "FP": fp, "TN": tn,
                "accuracy": (tp + tn) / n if n else float("nan"),
                "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
                "specificity": tn / (tn + fp) if (tn + fp) else float("nan")})
    return out


# ===========================================================================
# assembly


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("outputs/runs"),
                    help="directory holding <config>_iteration_<t>/")
    ap.add_argument("--config", required=True,
                    help="row name, e.g. cl_plus_image")
    ap.add_argument("--iterations", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", choices=["test", "validation"], default="test",
                    help="test predictions, or validation at the selected epoch")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional CSV for the per-fold + pooled rows")
    args = ap.parse_args()

    global RUNS
    RUNS = args.runs_root

    load = load_test if args.split == "test" else load_val
    frames, rows = [], []
    for it in args.iterations:
        frame = load(args.config, it)
        frames.append(frame)
        rows.append({"config": args.config, "split": args.split,
                     "fold": it, "n": len(frame),
                     "selected_epoch": selected_epoch(args.config, it)[0],
                     **metrics(frame, args.config)})
    pooled = pd.concat(frames, ignore_index=True)
    rows.append({"config": args.config, "split": args.split, "fold": "pooled",
                 "n": len(pooled), "selected_epoch": "-",
                 **metrics(pooled, args.config)})

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    if args.out is not None:
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
