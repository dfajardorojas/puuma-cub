"""Test-time inference: one fold's `checkpoint_best.pth` to `test_predictions.csv`.

    python scripts/predict_test_fold.py --results-root smoke --tag smoke_inference

**`--results-root` has no default**, for the same reason `run_one.py` has none:
a process must not be able to pick between the campaign and the
smoke destinations silently.

**There is no `--config` argument either, and for the same reason.** The row comes
from `run_manifest.json`'s `config_row`, as `--iteration` already defaults to the
iteration the run recorded. 

**There is no `--checkpoint` argument, deliberately.** The checkpoint rule makes
`checkpoint_best.pth` the result and `checkpoint_final.pth` a diagnostic that is
never reported.

Nothing predicts until the inference gate (h)–(k) has passed. 
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The results root is chosen from the command line BEFORE anything from nnunetv2
# is imported: a single stray `from nnunetv2...` above this point
# captures the shipped paths and the run writes into the read-only clone.
from puuma.nnunet_paths import ResultsRoot, bind_nnunet_paths  # noqa: E402

_ROOTS = {"campaign": ResultsRoot.CAMPAIGN, "smoke": ResultsRoot.SMOKE}


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, choices=sorted(_ROOTS),
                    help="CAMPAIGN or SMOKE. No default: the campaign/smoke separation depends on a "
                         "process being unable to pick silently.")
    ap.add_argument("--tag", type=str, required=True,
                    help="the run's subdirectory under the results root")
    ap.add_argument("--iteration", type=int, default=None,
                    help="the fold iteration t; defaults to the one the run recorded")
    return ap.parse_args()


ARGS = _parse()
PATHS = bind_nnunet_paths(_ROOTS[ARGS.results_root])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

from puuma.checkpoint import (  # noqa: E402
    assert_checkpoint_matches_model, checkpoint_for_test_predictions,
    load_weights_for_inference,
)
from puuma.config import load_row  # noqa: E402
from puuma.dataloader import LOCAL_PATCH_SIZE, Standardiser  # noqa: E402
from puuma.inference import (  # noqa: E402
    TEST_MANIFEST, TEST_PREDICTIONS, assert_weeks_round_trip, predict_test_fold,
    rows_for_test_fold, write_test_manifest, write_test_predictions,
)
from puuma.model import build_row_model  # noqa: E402
from puuma.plans import CONFIGURATION, load_plans_with_puuma_configuration  # noqa: E402
from puuma.preflight import run_inference_preflight  # noqa: E402

DIV = "=" * 78


def banner(t: str) -> None:
    print(f"\n{DIV}\n{t}\n{DIV}", flush=True)


def main() -> None:
    args = ARGS
    out_dir = Path(PATHS.results) / args.tag
    cohort_csv = REPO / "cohort" / "cohort.csv"
    mapping_csv = REPO / "cohort" / "case_id_mapping.csv"
    manifest_json = out_dir / "run_manifest.json"

    banner("CHECKPOINT")
    # Resolved, never globbed and never named on the command line.
    checkpoint = checkpoint_for_test_predictions(out_dir)
    if not manifest_json.is_file():
        raise SystemExit(f"no run_manifest.json in {out_dir}; the inference gate's (i), (j) and "
                         "(k) are all checked against it")
    manifest = json.loads(manifest_json.read_text())
    iteration = args.iteration if args.iteration is not None else int(manifest["iteration"])
    # The row the run TRAINED, never the default. `image_only`, `global_only` and
    # `unet` all blend three inputs where `cl_plus_image` blends four, and
    # `no_brain_seg` trains on a different dataset entirely.
    if "config_row" not in manifest:
        raise SystemExit(f"{manifest_json} has no `config_row`; this script builds the "
                         "row the run trained and will not guess it")
    row = load_row(str(manifest["config_row"]))
    pre = Path(PATHS.preprocessed) / row.dataset
    print(f"  run            {out_dir}")
    print(f"  config row     {row.name}  (from run_manifest.json, not a flag)")
    print(f"  dataset        {row.dataset}  blend {list(row.blend_inputs)}  "
          f"decoder {row.decoder_out_channels}ch  backbone {row.backbone}")
    print(f"  weights        {checkpoint.name}  (never {checkpoint.name[:11]}final.pth)")
    print(f"  iteration      {iteration}"
          f"{'' if args.iteration is None else '  (from --iteration)'}")

    plans = load_plans_with_puuma_configuration(pre / "nnUNetPlans.json")
    dataset_json = json.loads((pre / "dataset.json").read_text())
    pm = PlansManager(plans)
    cm = pm.get_configuration(CONFIGURATION)
    device = torch.device("cuda")
    model = build_row_model(row, pm, dataset_json, cm, LOCAL_PATCH_SIZE, device=device)
    # Named rather than bare: a shape mismatch here means the wrong row, and the torch
    # error says so only in tensor shapes.
    assert_checkpoint_matches_model(checkpoint, model=model, built_from=row.name)
    meta = load_weights_for_inference(checkpoint, model=model, map_location=device)
    print(f"  trained by     {meta['entry_point']}")
    print(f"  selected epoch {meta['saved_checkpoint_epoch']} "
          f"(best EMA {meta['best_ema']})")
    print(f"  mirror axes recorded in the checkpoint: "
          f"{meta['inference_allowed_mirroring_axes']} — recorded, not applied; "
          "no test-time augmentation is used")

    banner("PRE-FLIGHT GATE (h)–(k) — nothing is predicted if any of this fails")
    checks = run_inference_preflight(cohort_csv, mapping_csv,
                                     pre / "splits_final.json", manifest_json,
                                     checkpoint, meta["standardisation"], iteration)
    for c in checks:
        print(f"  PASS  {c.name:48s} {c.detail}")

    # The constants come from the checkpoint, not from a refit here — (k) has just
    # proved they are this fold's. Refitting would make the check circular.
    recorded = manifest.get("standardisation") or {}
    standardiser = Standardiser(means=dict(meta["standardisation"]["means"]),
                                stds=dict(meta["standardisation"]["stds"]),
                                fitted_on=tuple(recorded.get("fitted_on") or ()))

    banner("INFERENCE — via the validation pass")
    rows = rows_for_test_fold(cohort_csv, mapping_csv, iteration)
    print(f"  {len(rows)} test mothers, one scan each")
    t0 = time.perf_counter()
    predictions = predict_test_fold(model, rows, pre / cm.data_identifier, standardiser,
                                    device, LOCAL_PATCH_SIZE, list(cm.patch_size),
                                    iteration, compute_dice=True)
    seconds = time.perf_counter() - t0
    print(f"  {seconds:.1f} s total, {seconds/len(rows):.2f} s per mother")

    round_trip = assert_weeks_round_trip(predictions, standardiser)
    print(f"  inverse transform round-trips to {round_trip['max_abs_delta']:.2e} "
          f"(tolerance {round_trip['tolerance']:g})")

    path = write_test_predictions(predictions, out_dir / TEST_PREDICTIONS)
    write_test_manifest(out_dir / TEST_MANIFEST, checkpoint=checkpoint,
                        checkpoint_meta=meta, standardisation=standardiser.as_dict(),
                        iteration=iteration, checks=checks, round_trip=round_trip,
                        n_mothers=len(predictions))

    banner("WRITTEN")
    flagged = [r for r in predictions if r["gate_failed"]]
    errors = np.array([r["ga_del_pred_weeks"] - r["ga_del_true_weeks"] for r in predictions])
    print(f"  {path}")
    print(f"  {len(predictions)} rows, {len(predictions[0])} columns")
    print(f"  gate-failed subjects: {len(flagged)}"
          f"{' — ' + ', '.join(r['mother_id'] for r in flagged) if flagged else ''}")
    print(f"  MAE {np.abs(errors).mean():.2f} weeks   "
          f"Dice {np.nanmean([r['dice_channel_1'] for r in predictions]):.3f} / "
          f"{np.nanmean([r['dice_channel_2'] for r in predictions]):.3f}")
    print("\n  These numbers are from ONE fold's checkpoint. The analysis pools over all five "
          "folds, and validation comes before test.")


if __name__ == "__main__":
    main()
