"""The selection EMA, the checkpoint rule, and the detectors.

nnU-Net selects its best checkpoint at
`training/nnUNetTrainer/nnUNetTrainer.py:1047` with `>`, because pseudo-Dice is
maximised. The selection metric here is the standardised validation MSE, which
is minimised, so the comparison in :class:`CheckpointSelector` is `<`, written
deliberately: a copy of the framework's line with only the metric swapped
would select the worst checkpoint in the run, silently.

The EMA is the framework's, reproduced rather than reused: the hook in
`nnunet_logger.py:48-52` fires on the literal key `'mean_fg_dice'`, so a
renamed series gets no EMA. The arithmetic is α = 0.1, `prev × 0.9 + 0.1 ×
new`, initialised to the first observed value with no bias correction —
reproduced exactly in :func:`update_ema`. No `nnunetv2` import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

#: `nnunet_logger.py:50`. Not a tunable: inherited unchanged.
EMA_ALPHA = 0.1
#: Detector 1. Worse than predicting the training-fold mean on its own
#: validation set.
DETECTOR_1_THRESHOLD = 1.0


class SelectionError(RuntimeError):
    """Raised for a condition that must stop rather than be worked around."""


def update_ema(previous: float | None, value: float, alpha: float = EMA_ALPHA) -> float:
    """`nnunet_logger.py:50–51`, reproduced.

    Initialised to the first observed value, with no bias correction — which is
    what the framework does and what the checkpoint rule inherits.
    """
    if not np.isfinite(value):
        raise SelectionError(f"cannot update the selection EMA with a non-finite value ({value})")
    if previous is None:
        return float(value)
    return float(previous * (1.0 - alpha) + alpha * value)


@dataclass
class CheckpointSelector:
    """The `checkpoint_best.pth` rule, with the comparison inverted.

    `best_ema` is the running best of the selection EMA and `saved_checkpoint_epoch`
    the epoch at which it last improved — the record's second epoch number.
    """

    best_ema: float | None = None
    saved_checkpoint_epoch: int | None = None
    history: list[tuple[int, float, bool]] = field(default_factory=list)

    def consider(self, epoch: int, ema: float) -> bool:
        """Return whether this epoch improves the selection metric.

        `<`, not `>`: the metric is the standardised validation MSE and is
        minimised. See this module's docstring for the line this inverts.
        """
        if not np.isfinite(ema):
            raise SelectionError(
                f"epoch {epoch}: non-finite selection EMA ({ema}). Selecting on it would "
                "pick an arbitrary checkpoint.")
        improved = self.best_ema is None or ema < self.best_ema
        if improved:
            self.best_ema = float(ema)
            self.saved_checkpoint_epoch = int(epoch)
        self.history.append((int(epoch), float(ema), bool(improved)))
        return improved


def argmin_epoch_ungated(raw_metric: Sequence[float]) -> int:
    """The record's first epoch number: the argmin of the raw, un-smoothed trace.

    Both this and `saved_checkpoint_epoch` are recorded, under distinct names —
    the two differ whenever the EMA and the raw trace disagree on the best
    epoch.
    """
    values = np.asarray(raw_metric, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        raise SelectionError("no finite raw metric values to take an argmin over")
    return int(np.nanargmin(values))


def detector_1_raw_mse_at_saved_checkpoint(raw_metric: Sequence[float],
                                           saved_checkpoint_epoch: int,
                                           threshold: float = DETECTOR_1_THRESHOLD) -> dict:
    """Detector 1, which reads the RAW metric, not the EMA.

    It reads the raw standardised validation MSE at
    `saved_checkpoint_epoch`, not the EMA. The two differ by construction and the
    threshold is stated against the raw value. A test fails if this reads the EMA.
    """
    values = np.asarray(raw_metric, dtype=np.float64)
    if saved_checkpoint_epoch is None:
        raise SelectionError("detector 1 needs a saved_checkpoint_epoch; none was recorded")
    if not 0 <= saved_checkpoint_epoch < values.size:
        raise SelectionError(
            f"saved_checkpoint_epoch {saved_checkpoint_epoch} outside the recorded trace "
            f"of {values.size} epochs")
    value = float(values[saved_checkpoint_epoch])
    return {"detector": "raw_standardised_val_mse_at_saved_checkpoint",
            "reads": "raw", "epoch": int(saved_checkpoint_epoch),
            "value": value, "threshold": threshold, "triggered": bool(value >= threshold)}


def detector_2_preds_std_all_zero(preds_std: Sequence[float]) -> dict:
    """Detector 2: the model has become invariant to its placental patch."""
    given = list(preds_std)
    values = np.asarray([v for v in given if np.isfinite(v)], dtype=np.float64)
    dropped = len(given) - int(values.size)
    triggered = bool(values.size > 0 and np.all(values == 0.0))
    return {"detector": "preds_std_all_zero", "n_subjects": int(values.size),
            # Subjects whose preds_std is not a number — the gate-failed
            # subjects predict from ONE patch, so they have no spread. They are
            # dropped rather than counted as zeros, which would fire this detector
            # spuriously. The count is reported so the detector cannot fall silent
            # unnoticed: if it equals the cohort, `n_subjects` is 0 and `triggered`
            # is False for want of data, not for want of a problem.
            "n_not_measured": dropped,
            "max": float(values.max()) if values.size else float("nan"),
            "triggered": triggered}


def assert_detector_2_not_triggered(preds_std: Sequence[float], epoch: int) -> dict:
    """Detector 2 as a hard stop, evaluated every epoch.

    A detector that only speaks after the run it was meant to stop is not a
    detector: this one runs on the epoch's own validation output and raises.
    """
    result = detector_2_preds_std_all_zero(preds_std)
    if result["triggered"]:
        raise SelectionError(
            f"detector 2 at epoch {epoch}: preds_std is exactly 0.0 for all "
            f"{result['n_subjects']} validation subjects. The model has become "
            "invariant to its placental patch — every patch of every subject "
            "gives the same prediction. Training stops: the run cannot recover, "
            "and continuing would produce an uninterpretable checkpoint.")
    return result


# --- Detector 7: the blend collapsing onto CL -----------------------------
#: The ratio below which the image pathway is judged starved:
#: `(|w_global| + |w_local|) / |w_cl|`. Calibrated as the geometric midpoint
#: of the highest collapsing and the lowest recovering trajectory, read on a
#: log scale.
BLEND_IMAGE_SHARE_THRESHOLD = 0.259
#: Detector 7 is a WARNING, not a stop: healthy runs cross the threshold
#: routinely, so a stop would end runs that are fine — a detector that stops a
#: healthy run is worse than none. The record carries `blend_weights` and
#: `blend_image_share` every epoch regardless, so nothing is lost by not
#: raising.
BLEND_COLLAPSE_IS_A_STOP = False
#: Three consecutive epochs, i.e. 750 steps at 250 per epoch: a one-epoch dip
#: must not warn; a sustained collapse must.
BLEND_COLLAPSE_WINDOW_EPOCHS = 3


def blend_image_share(weights: Sequence[float], blend_inputs: Sequence[str]) -> float:
    """`(|w_global| + |w_local|) / |w_cl|` for one epoch's blend weights.

    Returns NaN when the row has no CL input, which makes the ratio meaningless
    rather than infinite — `global_only` and `image_only` are not at risk of this
    failure because they have no CL to collapse onto.
    """
    by_name = dict(zip(blend_inputs, [float(w) for w in weights]))
    if "cl" not in by_name:
        return float("nan")
    image = abs(by_name.get("global_regression", 0.0)) + abs(by_name.get("local_regression", 0.0))
    cl = abs(by_name["cl"])
    return float("inf") if cl == 0.0 else image / cl


class BlendCollapseMonitor:
    """Detector 7: the blend layer starving its own image pathway.

    What it is for: the optimiser can move the blend off the image inputs while
    validation MSE keeps improving, because CL genuinely predicts `GA_del` — a
    model in that state predicts from CL alone, looks healthy, and no other
    recorded quantity would show it.

    What it is NOT: a guard against the finding that `cl_plus_image` is no
    better than `cl_only` — a null there is a real, reportable result. This
    detector fires on the optimiser starving the image branch during training,
    which is different from the image branch having nothing to add: a genuine
    null has a live head with a non-zero regression gradient; this failure has
    the blend weights collapsing while the heads still carry signal. Do not
    read a firing as evidence for that reading, or a silent run as evidence
    against it.

    Evaluated per epoch on the epoch's own blend weights, like detector 2 and
    for the same reason.
    """

    def __init__(self, blend_inputs: Sequence[str],
                 threshold: float = BLEND_IMAGE_SHARE_THRESHOLD,
                 window_epochs: int = BLEND_COLLAPSE_WINDOW_EPOCHS,
                 stop: bool = BLEND_COLLAPSE_IS_A_STOP) -> None:
        self.blend_inputs = tuple(blend_inputs)
        self.threshold = float(threshold)
        self.window_epochs = int(window_epochs)
        #: False by default — see `BLEND_COLLAPSE_IS_A_STOP`. A caller that wants the
        #: hard stop has to ask for it explicitly.
        self.stop = bool(stop)
        self.consecutive = 0
        self.warnings: list[str] = []
        self.history: list[float] = []

    def observe(self, weights: Sequence[float], epoch: int) -> float:
        """Record the share, warn if it is sustained below the threshold, return it.

        Rows without a CL input cannot trigger this and must not: `image_only`
        has no `cl` in its blend, so `blend_image_share` is NaN there — there is
        nothing for the pathway to be starved in favour of, and a run of that
        row must never warn. The NaN is returned unchanged and the counter is
        left alone.
        """
        share = blend_image_share(weights, self.blend_inputs)
        self.history.append(share)
        if not np.isfinite(share):
            return share
        if share >= self.threshold:
            self.consecutive = 0
            return share
        self.consecutive += 1
        if self.consecutive >= self.window_epochs:
            message = (
                f"detector 7 at epoch {epoch}: the blend's image inputs have "
                f"carried less than {self.threshold:g} of the CL weight for "
                f"{self.consecutive} consecutive epochs — the last {self.consecutive} "
                f"values are {[round(v, 4) for v in self.history[-self.consecutive:]]}. "
                "The optimiser may be starving the image pathway, in which case "
                "the run converges on a CL-only predictor and validation MSE "
                "will not show it, because CL itself predicts GA_del. This is a "
                "warning, not a stop; it cannot distinguish that failure from "
                "the image inputs genuinely adding nothing, so read it as "
                "something to inspect rather than something that has happened.")
            self.warnings.append(message)
            if self.stop:
                raise SelectionError(message)
        return share
