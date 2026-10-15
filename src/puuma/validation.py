"""The per-mother validation pass.

The framework's own validation is 50 random crops scored by a pooled
pseudo-Dice; this builds instead one full inference pass per mother, every
epoch: her predicted `GA_del`, her `preds_std`, her per-channel Dice.

* global branch — unweighted mean of the per-window regression outputs over
  nnU-Net's own sliding windows;
* local branch — mean over the 20³ patches that pass the >33% gate, with the
  per-patch spread retained as `preds_std`;
* segmentation — nnU-Net's own voxel-wise accumulation with Gaussian importance
  weighting, unchanged.

Imports `nnunetv2` at module level — the shim must be bound first.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian, compute_steps_for_sliding_window,
)
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import (
    DownsampleSegForDSTransform2,
)

from puuma.dataloader import FOREGROUND_THRESHOLD, PLACENTA_LABEL, Standardiser

#: Fallback blend order for a model that does not declare one.
BLEND_ORDER = ("global_regression", "local_regression", "ga_scan", "cl")

#: `predict_from_raw_data.py:40`; the trainer's own predictor uses the same (`:1124`).
TILE_STEP_SIZE = 0.5
LOCAL_STRIDE = (10, 10, 10)
#: Placenta and brain. Background is excluded.
FOREGROUND_CHANNELS = (1, 2)
#: How many 20³ patches are stacked into one forward.
PATCH_CHUNK = 64
#: The preprocessor writes -1 into the cropped region; training removes it
#: (`nnUNetTrainer.py:734`). Dice must see the same label map the loss did.
IGNORE_LABEL = -1


class ValidationError(RuntimeError):
    """Raised for a condition the validation or inference rules make a hard stop."""


@dataclass
class MotherPrediction:
    """One mother's validation result, written to the run record in long format."""

    case_identifier: str
    mother_id: str
    ga_del_std_pred: float
    ga_del_std_true: float
    ga_del_pred_weeks: float
    ga_del_true_weeks: float
    n_global_windows: int
    n_patches: int
    preds_std: float
    dice: dict[int, float]
    #: This subject had no patch position above the gate, so the prediction
    #: comes from the single best-filled position. Recorded per subject, not
    #: only as a count, so a later analysis path can exclude or examine these
    #: subjects without re-running anything.
    gate_failed: bool = False
    #: The highest fill any position on the grid achieved — not the fill of
    #: the patch the prediction came from. On a flagged subject the two
    #: coincide, because the prediction is that one position. On a passing
    #: subject the prediction is the mean over every position above the gate,
    #: so this is the best of them and describes the grid rather than the
    #: prediction.
    best_gate_fill_on_grid: float = float("nan")

    def as_row(self) -> dict:
        row = {k: getattr(self, k) for k in
               ("case_identifier", "mother_id", "ga_del_std_pred", "ga_del_std_true",
                "ga_del_pred_weeks", "ga_del_true_weeks", "n_global_windows",
                "n_patches", "preds_std",
                # The gate flag is written PER SUBJECT, so a later analysis
                # path can exclude or examine those subjects without re-running.
                "gate_failed", "best_gate_fill_on_grid")}
        row.update({f"dice_channel_{c}": self.dice.get(c, float("nan"))
                    for c in FOREGROUND_CHANNELS})
        return row


def padded_shape(shape: Sequence[int], patch: Sequence[int]) -> tuple[int, ...]:
    """The shape nnU-Net slides over: at least the patch on every axis."""
    return tuple(max(int(s), int(p)) for s, p in zip(shape, patch))


def global_windows(shape: Sequence[int], patch: Sequence[int]) -> list[tuple[slice, ...]]:
    """nnU-Net's own window list, on the padded shape.

    `predict_sliding_window_return_logits:606–608` pads the image to at least the
    patch *before* slicing, so a case smaller than the patch on an axis yields one
    window there. The slices these return are therefore indices into the padded
    array, not the original — :func:`pad_to_patch` produces that array, and
    slicing the unpadded one instead yields short windows whose skip connections
    fail to concatenate in the decoder.
    """
    padded = padded_shape(shape, patch)
    steps = compute_steps_for_sliding_window(padded, tuple(patch), TILE_STEP_SIZE)
    return [tuple(slice(s, s + p) for s, p in zip((x, y, z), patch))
            for x in steps[0] for y in steps[1] for z in steps[2]]


def pad_to_patch(volume: np.ndarray, patch: Sequence[int]) -> np.ndarray:
    """Zero-pad a (C, X, Y, Z) volume to at least the patch on every spatial axis.

    Padding is appended at the end of each axis, so index 0 of the padded array is
    index 0 of the original and cropping back is a plain prefix slice.
    """
    target = padded_shape(volume.shape[1:], patch)
    pad = [(0, 0)] + [(0, t - s) for t, s in zip(target, volume.shape[1:])]
    if all(p == (0, 0) for p in pad):
        return volume
    return np.pad(volume, pad, mode="constant", constant_values=0)


def gated_patch_slices(mask: np.ndarray, patch: Sequence[int],
                       stride: Sequence[int] = LOCAL_STRIDE,
                       threshold: float = FOREGROUND_THRESHOLD,
                       ) -> tuple[list[tuple[slice, ...]], bool, float]:
    """Sliding placental patches that pass the gate.

    The gate tests the MASK, strictly greater than the threshold, as a ratio of
    two integers — a float32 mean widens to a double that can exceed the threshold
    at exact equality.

    :returns: ``(slices, gate_failed, best_fill)``. When at least one position
        passes, `slices` are those positions, `gate_failed` is False, and
        `best_fill` is the highest fill among them. When none passes, `slices`
        holds the single highest-filled position on the grid and `gate_failed`
        is True — the same shape of rule as the training-side fallback: one
        position, chosen for being the best available, not several chosen by
        lowering the threshold.

    Raises for the degenerate case — no placenta in the volume at all — which
    is a different failure and stays distinct. There is then no "best" position to
    fall back to, and absorbing it would hide a broken mask.
    """
    patch = np.asarray(patch, dtype=int)
    shape = np.asarray(mask.shape, dtype=int)
    if (shape < patch).any():
        raise ValidationError(f"volume {tuple(shape)} smaller than the local patch {tuple(patch)}")
    if int(mask.sum()) == 0:
        raise ValidationError(
            "no placenta voxels in the volume; no placental patch position exists. This is "
            "a hard stop and is NOT the same condition as the gate fallback, which needs a "
            "best position to fall back to.")
    hi = shape - patch
    out, best_fill, best_sl = [], -1.0, None
    for x in range(0, hi[0] + 1, stride[0]):
        for y in range(0, hi[1] + 1, stride[1]):
            for z in range(0, hi[2] + 1, stride[2]):
                sl = (slice(x, x + patch[0]), slice(y, y + patch[1]), slice(z, z + patch[2]))
                sub = mask[sl]
                fill = int(sub.sum()) / sub.size
                if fill > best_fill:
                    best_fill, best_sl = fill, sl
                if fill > threshold:
                    out.append(sl)
    if out:
        return out, False, best_fill
    return [best_sl], True, best_fill


def dice_per_channel(prediction: np.ndarray, reference: np.ndarray,
                     channels: Sequence[int] = FOREGROUND_CHANNELS) -> dict[int, float]:
    """Dice, background excluded, per channel, on the preprocessed grid.

    An empty reference *and* an empty prediction is a perfect agreement and scores
    1.0; an empty reference against a non-empty prediction scores 0.0. Returning
    NaN for the first would quietly drop the case from a mean.
    """
    out = {}
    for c in channels:
        p, r = prediction == c, reference == c
        denom = int(p.sum()) + int(r.sum())
        out[c] = 1.0 if denom == 0 else 2.0 * int((p & r).sum()) / denom
    return out


@torch.no_grad()
def validate_mother(model, volume: np.ndarray, segmentation: np.ndarray,
                    case_identifier: str, mother_id: str,
                    ga_scan: float, cl: float, ga_del: float,
                    standardiser: Standardiser, device: torch.device,
                    local_patch: Sequence[int], global_patch: Sequence[int],
                    compute_dice: bool = True,
                    patch_chunk: int = PATCH_CHUNK) -> MotherPrediction:
    """One mother, one full pass. Called once per mother per epoch."""
    was_training = model.training
    model.eval()
    try:
        label = segmentation[0].copy()
        label[label == IGNORE_LABEL] = 0
        mask = label == PLACENTA_LABEL

        windows = global_windows(volume.shape[1:], global_patch)
        original_shape = volume.shape[1:]
        padded = pad_to_patch(volume, global_patch)
        gvol = torch.as_tensor(padded).to(device)

        seg_logits = n_pred = gaussian = None
        if compute_dice:
            # The row's declared count, carried on the model; introspecting the
            # backbone for it assumes a decoder attribute not every backbone has.
            n_channels = int(model.decoder_out_channels)
            seg_logits = torch.zeros((n_channels, *padded.shape[1:]),
                                     dtype=torch.float32, device=device)
            n_pred = torch.zeros(padded.shape[1:], dtype=torch.float32, device=device)
            gaussian = compute_gaussian(tuple(global_patch), sigma_scale=1. / 8,
                                        value_scaling_factor=10, device=device).float()

        global_outputs = []
        for sl in windows:
            window = gvol[(slice(None), *sl)][None]
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                regression, logits = model.global_window_forward(window, compute_dice)
                global_outputs.append(regression.float())
                if compute_dice:
                    logits = logits.float()
            if compute_dice:
                seg_logits[(slice(None), *sl)] += logits[0] * gaussian
                n_pred[sl] += gaussian
                del logits
            del window
        # Unweighted mean over windows, an inference rule only.
        global_regression = torch.cat(global_outputs, dim=0).mean()

        dice = {}
        if compute_dice:
            seg_logits /= n_pred
            crop = (slice(None), *[slice(0, s) for s in original_shape])
            predicted = seg_logits[crop].argmax(0).cpu().numpy()
            dice = dice_per_channel(predicted, label)
            del seg_logits, n_pred, predicted
        del gvol

        # A row without a local branch has no placental patches to gate, no
        # per-patch spread, and a blend that never had a slot for them. Read from
        # the model rather than assumed, so the two paths cannot disagree.
        has_local = getattr(model, "has_local_branch", True)
        slices, gate_failed, best_fill_on_grid = (
            gated_patch_slices(mask, local_patch) if has_local
            else ([], False, float("nan")))

        local_outputs = []
        for i in range(0, len(slices), patch_chunk):
            chunk = slices[i:i + patch_chunk]
            stack = torch.stack([
                torch.as_tensor(volume[(slice(None), *sl)]) * torch.as_tensor(mask[sl]).float()
                for sl in chunk]).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                features = model.local_branch(stack)
                features = features[-1] if isinstance(features, (list, tuple)) else features
                local_outputs.append(model.local_head(features).float())
            del stack, features
        local_all = torch.cat(local_outputs, dim=0) if local_outputs else None
        local_regression = local_all.mean() if local_all is not None else None

        ga = torch.as_tensor(standardiser.apply("ga_scan", np.array([[ga_scan]])),
                             dtype=torch.float32, device=device)
        clv = torch.as_tensor(standardiser.apply("cl", np.array([[cl]])),
                              dtype=torch.float32, device=device)
        g = global_regression.reshape(1, 1)
        available = {"global_regression": g, "ga_scan": ga, "cl": clv,
                     "local_regression": (local_regression.reshape(1, 1)
                                          if local_regression is not None else None)}
        order = tuple(getattr(model, "blend_inputs", BLEND_ORDER))
        blended = model.blend(torch.cat([available[k] for k in order], dim=-1))

        # preds_std: the spread of the per-patch prediction with the global term
        # held fixed. Detector 2 reads this as "has the model become invariant
        # to its placental patch".
        if local_all is None:
            # No local branch: the per-patch spread detector 2 reads is not a
            # quantity this row has. NaN says "not measured"; 0.0 is that
            # detector's failure signal and would fire it on every subject.
            per_patch = None
        else:
            n_patch = local_all.shape[0]
            expanded = {"global_regression": g.expand(n_patch, 1),
                        "local_regression": local_all,
                        "ga_scan": ga.expand(n_patch, 1),
                        "cl": clv.expand(n_patch, 1)}
            per_patch = model.blend(torch.cat([expanded[k] for k in order], dim=-1))
        # With one patch there is no spread, and 0.0 would be a lie: detector 2
        # reads exactly that value as "the model has become invariant to its
        # placental patch". NaN says "not measured" instead, and detector 2 drops
        # non-finite entries rather than counting them as zeros.
        preds_std = (float(per_patch.std(unbiased=False))
                     if per_patch is not None and per_patch.shape[0] > 1
                     else float("nan"))

        std_pred = float(blended)
        return MotherPrediction(
            case_identifier=case_identifier, mother_id=mother_id,
            ga_del_std_pred=std_pred,
            ga_del_std_true=float(standardiser.apply("ga_del", np.array([[ga_del]]))),
            ga_del_pred_weeks=float(standardiser.invert("ga_del", np.array([[std_pred]]))),
            ga_del_true_weeks=float(ga_del),
            n_global_windows=len(windows), n_patches=len(slices),
            preds_std=preds_std, dice=dice,
            gate_failed=gate_failed, best_gate_fill_on_grid=best_fill_on_grid)
    finally:
        if was_training:
            model.train()


def validate_fold(model, mothers: Iterable[dict], preprocessed_folder: Path,
                  standardiser: Standardiser, device: torch.device,
                  local_patch: Sequence[int], global_patch: Sequence[int],
                  compute_dice: bool = True) -> list[MotherPrediction]:
    """Every mother in the validation fold, exactly once.

    ``mothers`` carries one row per mother — the cohort keeps a single evaluation scan
    per mother in validation, so one row is one mother is one scan. A duplicate
    raises rather than being averaged away.
    """
    rows = list(mothers)
    seen = [r["mother_id"] for r in rows]
    if len(set(seen)) != len(seen):
        raise ValidationError(
            "the validation split contains more than one scan for some mother. "
            "Validation uses exactly one evaluation scan per mother; averaging "
            "two would silently change the unit of analysis.")
    out = []
    for row in rows:
        data = np.load(preprocessed_folder / f"{row['case_identifier']}.npz")
        out.append(validate_mother(
            model, data["data"], data["seg"], row["case_identifier"], row["mother_id"],
            row["ga_scan"], row["cl"], row["ga_del"], standardiser, device,
            local_patch, global_patch, compute_dice=compute_dice))
    return out


def standardised_mse(predictions: Sequence[MotherPrediction]) -> float:
    """The selection metric and detector 1's quantity, on the standardised scale.

    Every mother counts: there is no exclusion filter, and a subject whose gate
    fails carries a real prediction and enters this mean like any other.
    """
    if not predictions:
        raise ValidationError("no validation predictions; there is no metric")
    err = np.array([p.ga_del_std_pred - p.ga_del_std_true for p in predictions],
                   dtype=np.float64)
    return float((err ** 2).mean())


def mae_weeks(predictions: Sequence[MotherPrediction]) -> float:
    """The read-out, in weeks, by inverting the standardisation."""
    if not predictions:
        return float("nan")
    return float(np.abs([p.ga_del_pred_weeks - p.ga_del_true_weeks
                         for p in predictions]).mean())


# ---------------------------------------------------------------------------
# The two smoke-test assertions, both hard stops
# ---------------------------------------------------------------------------
#: Smoke-test assertion 2: a validation pass must cost less than this fraction of the
#: wall time of `TRAIN_STEPS_PER_EPOCH` training steps.
VALIDATION_TIME_FRACTION = 0.25
#: `nnUNetTrainer.py:145`: an epoch is exactly 250 iterations.
TRAIN_STEPS_PER_EPOCH = 250


def assert_validation_file_readable(path: Path, expected_mothers: int) -> int:
    """Smoke-test assertion 1: a run that does not produce a readable per-mother file fails.

    Readable means it parses *and* holds one row per mother — a file that exists
    and is empty would pass a mere existence check and is exactly the failure this
    guards against.
    """
    import csv
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"smoke-test assertion 1: no per-mother validation file at {path}")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValidationError(f"smoke-test assertion 1: {path} is empty")
    if "case_identifier" not in rows[0]:
        raise ValidationError(
            f"smoke-test assertion 1: {path} has no case_identifier column; columns were "
            f"{list(rows[0])}")
    latest = max(int(r["epoch"]) for r in rows)
    n = len({r["mother_id"] for r in rows if int(r["epoch"]) == latest})
    if n != expected_mothers:
        raise ValidationError(
            f"smoke-test assertion 1: {path} holds {n} mothers at epoch {latest}, expected "
            f"{expected_mothers}. Every validation mother must appear exactly once.")
    return n


def assert_validation_within_time_budget(validation_seconds: float,
                                         mean_train_step_seconds: float,
                                         steps: int = TRAIN_STEPS_PER_EPOCH,
                                         fraction: float = VALIDATION_TIME_FRACTION) -> dict:
    """Smoke-test assertion 2, as a hard stop.

    One validation pass must take less than a quarter of the wall time of 250
    training steps; otherwise the patch batching is wrong or the validation
    cadence needs revisiting.
    """
    budget = fraction * steps * mean_train_step_seconds
    result = {"validation_seconds": float(validation_seconds), "budget_seconds": float(budget),
              "mean_train_step_seconds": float(mean_train_step_seconds),
              "steps": int(steps), "fraction": float(fraction),
              "ratio": float(validation_seconds / budget) if budget > 0 else float("inf")}
    if validation_seconds >= budget:
        raise ValidationError(
            f"smoke-test assertion 2: a validation pass took {validation_seconds:.2f} s against a "
            f"budget of {budget:.2f} s ({fraction:.0%} of {steps} training steps at "
            f"{mean_train_step_seconds:.3f} s each). The patch batching is wrong "
            "or the validation cadence needs revisiting.")
    return result


# ---------------------------------------------------------------------------
# The second validation quantity: the full training objective on validation data
# ---------------------------------------------------------------------------
@torch.no_grad()
def validation_objective(model, loss_fn, mothers: Iterable[dict], preprocessed_folder: Path,
                         standardiser: Standardiser, device: torch.device,
                         local_patch: Sequence[int], global_patch: Sequence[int],
                         deep_supervision_scales: Sequence) -> dict:
    """The training objective, evaluated on validation data.

    This is a second path over the same mothers and deliberately not a
    modification of the per-mother pass: the per-mother pass aggregates windows
    and patches into one prediction per subject, which is the right thing for
    the regression metric and the wrong thing for a number meant to sit beside
    the training total. This one mimics what `train_step` computes — one
    window, the deep-supervision target list, both terms at `w_reg = 2` and
    `w_seg = 1` — so the result is comparable.

    Two things a reader must know, or the gap will be misread:

    1. Validation windows are NOT augmented and training windows are, so part
       of any train/validation gap here is the augmentation, not
       generalisation.
    2. The window and the patch are the FIRST of their deterministic lists,
       not a random draw, so the same window and the same patch are used for a
       given mother at every epoch; the point of this number is its trend.

    The loss is evaluated inside autocast, as `train_step` does: outside it,
    nnU-Net's cross-entropy reduces fp16 logits in fp16 over the
    highest-resolution level and returns NaN, and nothing detects it at the
    call site.
    """
    was_training = model.training
    model.eval()
    downsample = DownsampleSegForDSTransform2(deep_supervision_scales, 0,
                                              input_key="target", output_key="target")
    totals, regs, segs, skipped = [], [], [], []
    try:
        for row in mothers:
            data = np.load(preprocessed_folder / f"{row['case_identifier']}.npz")
            volume, seg = data["data"], data["seg"]
            label = seg[0].copy()
            label[label == IGNORE_LABEL] = 0
            mask = label == PLACENTA_LABEL

            # Same fallback as the per-mother pass; a gate failure does not
            # remove the subject from this average. A row with no local branch
            # has no patches at all.
            has_local = getattr(model, "has_local_branch", True)
            patches = (gated_patch_slices(mask, local_patch)[0] if has_local else [None])

            windows = global_windows(volume.shape[1:], global_patch)
            padded_volume = pad_to_patch(volume, global_patch)
            padded_label = pad_to_patch(label[None], global_patch)[0]
            wsl = windows[0]                              # deterministic: the first
            window = torch.as_tensor(padded_volume[(slice(None), *wsl)])[None].to(device)
            # The downsampler is a numpy transform — in the training chain it runs
            # before `NumpyToTensor` (`nnUNetTrainer.py:762` then `:764`), so the
            # targets are built in numpy here and converted afterwards, in that order.
            window_label = padded_label[wsl].astype(np.float32)[None, None]
            targets = [torch.as_tensor(t).to(device)
                       for t in downsample(target=window_label)["target"]]

            psl = patches[0]                              # deterministic: the first
            patch_tensor = None
            if psl is not None:
                patch_tensor = (torch.as_tensor(volume[(slice(None), *psl)])
                                * torch.as_tensor(mask[psl]).float())[None].to(device)

            ga = torch.as_tensor(standardiser.apply("ga_scan", np.array([[row["ga_scan"]]])),
                                 dtype=torch.float32, device=device)
            cl = torch.as_tensor(standardiser.apply("cl", np.array([[row["cl"]]])),
                                 dtype=torch.float32, device=device)
            y = torch.as_tensor(standardiser.apply("ga_del", np.array([[row["ga_del"]]])),
                                dtype=torch.float32, device=device)

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(window, patch_tensor, ga, cl)
                terms = loss_fn(prediction, targets, y)
            totals.append(float(terms.total))
            regs.append(float(terms.regression))
            segs.append(float(terms.segmentation))
            del window, window_label, targets, patch_tensor, prediction, terms
    finally:
        if was_training:
            model.train()
    if not totals:
        raise ValidationError(
            "no validation mother yielded a gated patch; the comparable objective has "
            "nothing to average over")
    return {"total": float(np.mean(totals)), "regression": float(np.mean(regs)),
            "segmentation": float(np.mean(segs)), "n_mothers": len(totals),
            "n_skipped": len(skipped), "skipped": skipped}
