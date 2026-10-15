"""The three reference rows over every iteration. No GPU, no training.

    python scripts/run_reference_rows.py --results-root smoke

Writes one per-subject CSV and one per-fold CSV under the chosen results root, in a
trained row's schema so the analysis reports all nine rows in one table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from puuma.nnunet_paths import ResultsRoot, bind_nnunet_paths  # noqa: E402

_ROOTS = {"campaign": ResultsRoot.CAMPAIGN, "smoke": ResultsRoot.SMOKE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, choices=sorted(_ROOTS))
    ap.add_argument("--iterations", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--evaluate-on", choices=("val", "test"), default="test")
    args = ap.parse_args()
    paths = bind_nnunet_paths(_ROOTS[args.results_root])

    import pandas as pd

    from puuma.cohort import split_for_iteration
    from puuma.reference_rows import REFERENCE_ROWS, run_reference_rows

    cohort = pd.read_csv(REPO / "cohort" / "cohort.csv").merge(
        pd.read_csv(REPO / "cohort" / "case_id_mapping.csv")[["scan_id", "case_identifier"]],
        on="scan_id", validate="one_to_one")
    subjects, folds = run_reference_rows(cohort, args.iterations, split_for_iteration,
                                         evaluate_on=args.evaluate_on)
    out = Path(paths.results) / "reference_rows"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subjects).to_csv(out / f"{args.evaluate_on}_predictions.csv", index=False)
    per_fold = pd.DataFrame(folds)
    per_fold.to_csv(out / f"{args.evaluate_on}_per_fold.csv", index=False)

    print("=" * 78)
    print(f"REFERENCE ROWS — evaluated on {args.evaluate_on}, fitted on TRAIN only")
    print("=" * 78)
    print(f"  {'row':<12} {'fold':>4} {'n_train':>8} {'n_eval':>7} {'MSE_std':>9} {'MAE wk':>8}")
    for _, r in per_fold.iterrows():
        print(f"  {r['row']:<12} {int(r['iteration']):>4} {int(r['n_train']):>8} "
              f"{int(r['n']):>7} {r['val_standardised_mse']:>9.4f} {r['val_mae_weeks']:>8.3f}")
    print(f"\n  {'row':<12} {'pooled MSE_std':>15} {'pooled MAE wk':>14}   (mean over folds)")
    for row in REFERENCE_ROWS:
        sub = per_fold[per_fold["row"] == row]
        print(f"  {row:<12} {sub['val_standardised_mse'].mean():>15.4f} "
              f"{sub['val_mae_weeks'].mean():>14.3f}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
