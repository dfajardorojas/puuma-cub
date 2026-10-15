"""Test time: `checkpoint_best.pth` to per-subject predictions.

Implements no prediction rule of its own: the whole prediction path — global
windows and their unweighted mean, gated 20³ placental patches and their mean,
`preds_std`, the gate fallback and its flag, per-channel Dice — is
:func:`puuma.validation.validate_fold`, called here unchanged, so validation
and test measure the same quantity. This module adds the test fold's rows (one
scan per mother), the join back to the cohort columns the output carries, and
a hard stop asserting the weeks prediction inverts with this fold's
standardisation constants (`preflight` check (k)).

No test-time augmentation is applied; `inference_allowed_mirroring_axes` is
recorded in the output manifest, never acted on. Imports `nnunetv2`
transitively through `puuma.validation` — the shim must be bound first.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from puuma.cohort import split_for_iteration
from puuma.dataloader import Standardiser
from puuma.validation import MotherPrediction, validate_fold

#: The per-subject output file name.
TEST_PREDICTIONS = "test_predictions.csv"
#: Provenance for the file above: which weights, which epoch, which constants.
TEST_MANIFEST = "test_predictions_manifest.json"
#: The weeks prediction is `z x std + mean`; recomputing it from the row's own
#: standardised value must return the number written beside it.
ROUND_TRIP_TOL = 1e-9


class InferenceError(RuntimeError):
    """Raised for a condition the inference rules make a hard stop at test time."""


def rows_for_test_fold(cohort_csv: Path, mapping_csv: Path,
                       iteration: int) -> list[dict]:
    """The test fold's mothers, one scan each.

    The rows carry the five keys :func:`~puuma.validation.validate_fold` reads
    plus the cohort context required in the output. `validate_fold` ignores the
    extra keys, so the same dictionaries serve both.
    """
    cohort = pd.read_csv(cohort_csv)
    mapping = pd.read_csv(mapping_csv)[["scan_id", "case_identifier"]]
    merged = cohort.merge(mapping, on="scan_id", validate="one_to_one")
    rows = [{"case_identifier": r.case_identifier, "mother_id": r.mother_id,
             "scan_id": r.scan_id, "fold": int(r.fold), "category": r.category,
             "ga_scan": float(r.tag_ga), "cl": float(r.cl_value),
             "ga_del": float(r.tag_gadel)}
            for r in split_for_iteration(merged, iteration)["test"].itertuples()]
    if not rows:
        raise InferenceError(f"iteration {iteration} has no test mothers")
    return rows


def _joined(row: dict, prediction: MotherPrediction, iteration: int) -> dict:
    """One output row: the cohort context, then the prediction, in that order."""
    predicted = prediction.as_row()
    return {"iteration": int(iteration), "fold": row["fold"],
            "category": row["category"], "mother_id": prediction.mother_id,
            # Which scan it came from, for the prediction and for the CL.
            "scan_id": row["scan_id"], "case_identifier": prediction.case_identifier,
            "ga_scan": row["ga_scan"], "cl": row["cl"], "cl_from_scan": row["scan_id"],
            **{k: v for k, v in predicted.items()
               if k not in ("mother_id", "case_identifier")}}


def predict_test_fold(model, rows: Sequence[dict], preprocessed_folder: Path,
                      standardiser: Standardiser, device, local_patch: Sequence[int],
                      global_patch: Sequence[int], iteration: int,
                      compute_dice: bool = True) -> list[dict]:
    """Per-subject test predictions, from the validation pass and nothing else.

    `validate_fold` is the whole prediction; it also enforces the rule of one scan per
    mother, raising rather than averaging two. Everything here is the join.
    """
    predictions = validate_fold(model, rows, Path(preprocessed_folder), standardiser,
                                device, local_patch, global_patch,
                                compute_dice=compute_dice)
    by_case = {r["case_identifier"]: r for r in rows}
    return [_joined(by_case[p.case_identifier], p, iteration) for p in predictions]


def assert_weeks_round_trip(rows: Sequence[dict], standardiser: Standardiser,
                            tol: float = ROUND_TRIP_TOL) -> dict:
    """The inverse transform, checked against the constants actually loaded.

    Both directions, because they fail differently. `ga_del_pred_weeks` must be
    the standardised prediction pushed back through these constants, and
    `ga_del_std_true` must be the observed weeks pushed forward through them. A
    file whose two scales were produced by different constants is internally
    inconsistent in exactly this way and in no other visible one.
    """
    if not rows:
        raise InferenceError("no rows to check the inverse transform against")
    worst = 0.0
    for row in rows:
        back = float(standardiser.invert("ga_del", np.array([[row["ga_del_std_pred"]]])))
        forward = float(standardiser.apply("ga_del", np.array([[row["ga_del_true_weeks"]]])))
        for written, recomputed, name in ((row["ga_del_pred_weeks"], back, "ga_del_pred_weeks"),
                                          (row["ga_del_std_true"], forward, "ga_del_std_true")):
            delta = abs(float(written) - recomputed)
            worst = max(worst, delta)
            if delta > tol:
                raise InferenceError(
                    f"{row['case_identifier']}: {name} is {written!r} but the loaded "
                    f"constants give {recomputed!r} (delta {delta:.3e} > {tol:g}). The "
                    "two scales in this file were not produced by the same constants.")
    return {"max_abs_delta": float(worst), "tolerance": float(tol), "n_rows": len(rows)}


def write_test_predictions(rows: Sequence[dict], path: Path) -> Path:
    """The per-subject test predictions, one row per mother."""
    path = Path(path)
    if not rows:
        raise InferenceError("refusing to write an empty test-predictions file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_test_manifest(path: Path, *, checkpoint: Path, checkpoint_meta: dict,
                        standardisation: dict, iteration: int, checks: Sequence,
                        round_trip: dict, n_mothers: int) -> Path:
    """Which weights, which epoch, which constants, and which gates passed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": int(iteration),
        "n_mothers": int(n_mothers),
        "checkpoint": str(checkpoint),
        "checkpoint_is": "checkpoint_best.pth — the selected weights, never final",
        "saved_checkpoint_epoch": checkpoint_meta.get("saved_checkpoint_epoch"),
        "entry_point_that_trained_it": checkpoint_meta.get("entry_point"),
        # Recorded, not applied: no test-time mirroring is used, and applying it
        # here would make validation and test disagree by construction.
        "inference_allowed_mirroring_axes":
            checkpoint_meta.get("inference_allowed_mirroring_axes"),
        "test_time_augmentation": "none (parity with the validation pass)",
        "standardisation": standardisation,
        "inverse_transform_check": round_trip,
        "preflight": [{"name": c.name, "detail": c.detail} for c in checks],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
