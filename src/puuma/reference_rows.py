"""The three reference rows: `constant`, `cl_only`, `cl_plus_ga`. No GPU.

Computed from `cohort.csv` and the fold structure alone; they are the baselines
the trained rows are compared against. Every fit uses that fold's TRAINING rows
and nothing else — a statistic fitted across the split would leak even though no
mother crosses a fold — and :attr:`ReferenceFit.fitted_on` records the exact
case identifiers used. The output schema matches a trained row's per-subject
predictions, with raw standardised MSE and MAE in weeks on that fold's
standardisation constants, so one table can report every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

#: The three rows, and the predictors each is allowed. `constant` has none.
#: The cohort's own column names: `tag_gadel` is GA at delivery, `cl_value` the
#: cervical length, `tag_ga` GA at scan (`cohort/cohort.csv`).
TARGET = "tag_gadel"
REFERENCE_ROWS = {"constant": (), "cl_only": ("cl_value",),
                  "cl_plus_ga": ("cl_value", "tag_ga")}


class ReferenceError(RuntimeError):
    """Raised for a fit that would see rows it must not."""


@dataclass(frozen=True)
class ReferenceFit:
    """One row, one fold. `coefficients` is empty for `constant`."""

    row: str
    iteration: int
    intercept: float
    coefficients: tuple[float, ...]
    predictors: tuple[str, ...]
    #: The case identifiers the fit saw. Recorded, not asserted.
    fitted_on: tuple[str, ...] = field(default_factory=tuple)
    #: The training-fold mean and sd of the target, the per-fold standardisation.
    ga_del_mean: float = float("nan")
    ga_del_std: float = float("nan")

    def predict_weeks(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.full(len(frame), float(self.intercept))
        for name, coefficient in zip(self.predictors, self.coefficients):
            out = out + float(coefficient) * frame[name].to_numpy(dtype=float)
        return out


def fit_reference(row: str, train: pd.DataFrame, iteration: int) -> ReferenceFit:
    """Fit one reference row on training rows only.

    `train` must be `split_for_iteration(...)["train"]`. Nothing here looks at the
    other two splits, and the caller cannot pass them by accident without it showing
    up in `fitted_on`.
    """
    if row not in REFERENCE_ROWS:
        raise ReferenceError(f"{row!r} is not one of the reference rows "
                             f"{sorted(REFERENCE_ROWS)}")
    predictors = REFERENCE_ROWS[row]
    needed = [TARGET, *predictors]
    usable = train.dropna(subset=needed)
    if usable.empty:
        raise ReferenceError(f"{row} iteration {iteration}: no training rows with "
                             f"{needed} present")
    target = usable[TARGET].to_numpy(dtype=float)
    mean, sd = float(target.mean()), float(target.std(ddof=0))
    if not predictors:
        # `constant` predicts the training-fold mean for every subject.
        return ReferenceFit(row, iteration, mean, (), (), tuple(usable["case_identifier"]),
                            mean, sd)
    design = np.column_stack([np.ones(len(usable))]
                             + [usable[p].to_numpy(dtype=float) for p in predictors])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return ReferenceFit(row, iteration, float(solution[0]), tuple(float(c) for c in solution[1:]),
                        predictors, tuple(usable["case_identifier"]), mean, sd)


def score_reference(fit: ReferenceFit, evaluate: pd.DataFrame) -> tuple[list[dict], dict]:
    """Per-subject rows in a trained row's schema, and the two reported metrics.

    The standardisation is the training fold's, exactly as a trained row's is, so
    `MSE_std` means the same thing across all nine rows: 1.0 is the training-fold mean
    predictor evaluated on this split.
    """
    usable = evaluate.dropna(subset=[TARGET, *fit.predictors])
    predicted = fit.predict_weeks(usable)
    truth = usable[TARGET].to_numpy(dtype=float)
    sd = fit.ga_del_std if fit.ga_del_std > 0 else float("nan")
    predicted_std = (predicted - fit.ga_del_mean) / sd
    truth_std = (truth - fit.ga_del_mean) / sd
    rows = [
        {"case_identifier": case, "mother_id": mother, "ga_del_std_pred": float(ps),
         "ga_del_std_true": float(ts), "ga_del_pred_weeks": float(pw),
         "ga_del_true_weeks": float(tw), "n_global_windows": 0, "n_patches": 0,
         "preds_std": float("nan"), "gate_failed": False,
         "best_gate_fill_on_grid": float("nan"), "dice_1": float("nan"),
         "dice_2": float("nan")}
        for case, mother, ps, ts, pw, tw in zip(
            usable["case_identifier"], usable["mother_id"], predicted_std, truth_std,
            predicted, truth)
    ]
    metrics = {
        "row": fit.row, "iteration": fit.iteration, "n": len(rows),
        "val_standardised_mse": float(np.mean((predicted_std - truth_std) ** 2)),
        "val_mae_weeks": float(np.mean(np.abs(predicted - truth))),
        "n_train": len(fit.fitted_on),
    }
    return rows, metrics


def run_reference_rows(cohort: pd.DataFrame, iterations: Sequence[int],
                       split: "callable", evaluate_on: str = "test"
                       ) -> tuple[list[dict], list[dict]]:
    """Every reference row over every iteration. Returns (per-subject, per-fold)."""
    subjects, folds = [], []
    for iteration in iterations:
        parts = split(cohort, iteration)
        for row in REFERENCE_ROWS:
            fit = fit_reference(row, parts["train"], iteration)
            seen = set(fit.fitted_on)
            for other in ("val", "test"):
                overlap = seen & set(parts[other]["case_identifier"])
                if overlap:
                    raise ReferenceError(
                        f"{row} iteration {iteration}: the fit saw {len(overlap)} "
                        f"{other} case(s) — {sorted(overlap)[:3]}. The reference rows "
                        "are fitted on training rows only, and no leakage assertion covers "
                        "this because those check mothers, not statistics.")
            rows, metrics = score_reference(fit, parts[evaluate_on])
            for r in rows:
                r.update(row=row, iteration=iteration)
            subjects.extend(rows)
            folds.append(metrics)
    return subjects, folds
