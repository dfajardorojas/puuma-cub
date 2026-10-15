"""Training dataloader: nnU-Net's global window plus the 20³ placental patch.

* :class:`ScalarTable` — case identifier → (GA at scan, CL, `GA_del`), built
  from ``cohort.csv``; a missing scalar raises, never defaults.
* :class:`Standardiser` — per-fold z-scoring constants for the scalars.
* :class:`PlacentaPatchTransform` — the 20³ crop, appended to the end of the
  transform chain.

The crop is a transform because the augmenter wraps the dataloader
(`nnUNetTrainer.py:631`, `:634`): the chain runs after ``generate_train_batch``
returns, so a patch cropped in the dataloader would come from the un-augmented
volume while ``SpatialTransform`` deforms the global window. At the chain's end
the ignore label is gone and the full-resolution label survives as
``target[0]``; both are asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform

#: The label map. Brain is 2 and is not eligible for the local patch.
PLACENTA_LABEL = 1
LOCAL_PATCH_SIZE: tuple[int, int, int] = (20, 20, 20)
#: A per-patch content threshold, strict `>`, tested on the MASK — not
#: nnU-Net's `oversample_foreground_percent`, which happens to share the value.
FOREGROUND_THRESHOLD = 0.33
#: Not a tuning parameter: on exhaustion the crop falls back to the best
#: centre found (see :class:`PlacentaPatchTransform`), it does not lower the
#: gate.
MAX_PATCH_ATTEMPTS = 10_000
#: Keys the crop transform adds to the batch, for the run record. `patch_gate_failed` marks the
#: items that fell back; `patch_fill` carries every item's achieved placenta
#: fraction, so the record has the distribution and not only the count.
GATE_KEYS = ("patch_gate_failed", "patch_fill")

SCALAR_KEYS = ("ga_scan", "cl", "ga_del")


class DataloaderError(RuntimeError):
    """Raised for any condition that must stop a run rather than be worked around."""


@dataclass(frozen=True)
class Scalars:
    """The three per-scan scalars the blend layer and the loss need."""

    ga_scan: float
    cl: float
    ga_del: float


class ScalarTable:
    """Case identifier → :class:`Scalars`, built from ``cohort.csv``.

    ``cohort.csv`` is the single source of truth for the scalars, as it already
    is for folds and the tie-break.
    """

    def __init__(self, scalars: dict[str, Scalars]) -> None:
        self._scalars = dict(scalars)

    @classmethod
    def from_cohort(cls, cohort_csv: Path, mapping_csv: Path) -> "ScalarTable":
        """Build from ``cohort.csv`` keyed by nnU-Net's case identifier.

        The scan-id → case-identifier mapping is read from the artefact the
        dataset builder already wrote (``cohort/case_id_mapping.csv``); it is not
        re-derived here, so the two cannot drift.
        """
        cohort = pd.read_csv(cohort_csv)
        mapping = pd.read_csv(mapping_csv)[["scan_id", "case_identifier"]]
        merged = cohort.merge(mapping, on="scan_id", how="inner", validate="one_to_one")
        if len(merged) != len(cohort):
            raise DataloaderError(
                f"case-identifier mapping covers {len(merged)} of {len(cohort)} cohort rows; "
                "the mapping and the cohort disagree")
        table: dict[str, Scalars] = {}
        for row in merged.itertuples():
            for field in ("tag_ga", "cl_value", "tag_gadel"):
                value = getattr(row, field)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    raise DataloaderError(
                        f"{row.case_identifier}: {field} is missing. A scalar must never "
                        "default; the cohort is wrong or this scan should not be here.")
            table[row.case_identifier] = Scalars(
                ga_scan=float(row.tag_ga), cl=float(row.cl_value), ga_del=float(row.tag_gadel))
        return cls(table)

    def __len__(self) -> int:
        return len(self._scalars)

    def __contains__(self, case_identifier: str) -> bool:
        return case_identifier in self._scalars

    def __getitem__(self, case_identifier: str) -> Scalars:
        try:
            return self._scalars[case_identifier]
        except KeyError:
            raise DataloaderError(
                f"no scalars for case {case_identifier!r}. A missing scalar raises; it is "
                "never defaulted.") from None

    def columns(self, case_identifiers: Sequence[str],
                standardiser: "Standardiser | None" = None) -> dict[str, np.ndarray]:
        """The three scalars for a batch, shaped (B, 1) as the blend layer wants.

        With a :class:`Standardiser` the values are z-scored; without one they are
        raw. Training always passes one.
        """
        rows = [self[c] for c in case_identifiers]
        out = {name: np.array([[getattr(r, name)] for r in rows], dtype=np.float32)
               for name in SCALAR_KEYS}
        if standardiser is not None:
            out = {k: standardiser.apply(k, v).astype(np.float32) for k, v in out.items()}
        return out

    def fit_standardiser(self, training_case_identifiers: Sequence[str]) -> "Standardiser":
        """Z-scoring constants from the training rows of one fold, and no others.

        Every identifier must be present; a missing one raises rather than being
        skipped, because a silently smaller fitting set is exactly the failure this
        is meant to make impossible.
        """
        keys = list(training_case_identifiers)
        if not keys:
            raise DataloaderError("cannot fit standardisation constants on an empty split")
        if len(set(keys)) != len(keys):
            raise DataloaderError("the training split contains duplicate case identifiers")
        rows = [self[c] for c in keys]          # raises on anything unknown
        means, stds = {}, {}
        for name in SCALAR_KEYS:
            values = np.array([getattr(r, name) for r in rows], dtype=np.float64)
            mean, std = float(values.mean()), float(values.std(ddof=0))
            if not np.isfinite(mean) or not np.isfinite(std):
                raise DataloaderError(f"{name}: non-finite standardisation constant")
            if std <= 0.0:
                raise DataloaderError(
                    f"{name} has zero variance across the {len(keys)} training rows; "
                    "z-scoring would divide by zero. This is a cohort problem, not a "
                    "numerical one, and must not be papered over with an epsilon.")
            means[name], stds[name] = mean, std
        return Standardiser(means=means, stds=stds, fitted_on=tuple(keys))


@dataclass(frozen=True)
class Standardiser:
    """Per-fold z-scoring constants for the three scalars.

    The scalars are z-scored per training fold, with the constants stored in
    the run config; standardisation is applied here, in the scalar path, so
    there is one mechanism and not two.

    The constants come from the training rows of that fold and nothing else:
    fitting them over the whole cohort would leak the validation and test
    distributions into training, and no leakage assertion would catch it —
    those check that mothers do not cross folds, not that statistics do not.
    :attr:`fitted_on` records exactly which cases were used.
    """

    means: dict[str, float]
    stds: dict[str, float]
    fitted_on: tuple[str, ...]

    def apply(self, name: str, values: np.ndarray) -> np.ndarray:
        return (values - self.means[name]) / self.stds[name]

    def invert(self, name: str, values: np.ndarray) -> np.ndarray:
        """Back to weeks. The budget check and detector 1 are defined on the
        standardised scale; MAE is reported in weeks by inverting."""
        return values * self.stds[name] + self.means[name]

    def as_dict(self) -> dict:
        """What the run record is required to carry."""
        return {"means": dict(self.means), "stds": dict(self.stds),
                "n_fitted_on": len(self.fitted_on), "fitted_on": list(self.fitted_on)}


class PlacentaPatchTransform(AbstractTransform):
    """Crop the 20³ placental patch from the augmented global window.

    Per item: mask the augmented window to the placenta channel of the
    augmented label map, draw a centre uniformly from the mask's non-zero
    voxels, crop, and test the gate on the mask — strictly more than
    :data:`FOREGROUND_THRESHOLD` of the patch's voxels are placenta. Redraw on
    failure up to :data:`MAX_PATCH_ATTEMPTS`. Near a volume edge the box is
    clamped inward, never padded and never rejected.

    On exhaustion the crop falls back to the best centre found and flags the
    item — it does not raise and does not lower the gate. The one condition
    that still raises is a window with no placenta in it at all.

    The gate tests the mask and not the masked image: an intensity test calls
    a voxel background when it is close to the background value, and on
    nnU-Net's per-case z-scored grid "zero" is the case mean, not absence of
    signal.
    """

    def __init__(self, patch_size: Sequence[int] = LOCAL_PATCH_SIZE,
                 threshold: float = FOREGROUND_THRESHOLD,
                 max_attempts: int = MAX_PATCH_ATTEMPTS,
                 label: int = PLACENTA_LABEL,
                 data_key: str = "data", target_key: str = "target",
                 output_key: str = "patch",
                 rng: np.random.Generator | None = None,
                 record_attempts: list[int] | None = None,
                 record_slices: list[tuple[int, int, int]] | None = None) -> None:
        self.patch_size = np.array(patch_size, dtype=int)
        self.threshold = float(threshold)
        self.max_attempts = int(max_attempts)
        self.label = int(label)
        self.data_key, self.target_key, self.output_key = data_key, target_key, output_key
        #: ``None`` means the process-global NumPy RNG, which is what every stock
        #: transform uses and what batchgenerators seeds per worker. An explicit
        #: generator is for tests, where a shared seed across workers would be wrong.
        self.rng = rng
        self.record_attempts = record_attempts
        #: Lower corners of the boxes actually cropped. Tests use this to assert the
        #: patch against the augmented arrays; production leaves it ``None``.
        self.record_slices = record_slices

    def _randint(self, high: int) -> int:
        if self.rng is not None:
            return int(self.rng.integers(high))
        return int(np.random.randint(high))

    def _full_resolution_label(self, target) -> torch.Tensor:
        if isinstance(target, (list, tuple)):
            if len(target) == 0:
                raise DataloaderError("deep-supervision target list is empty")
            return target[0]
        return target

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = self._full_resolution_label(data_dict[self.target_key])
        if tuple(label.shape[2:]) != tuple(data.shape[2:]):
            raise DataloaderError(
                f"target[0] has spatial shape {tuple(label.shape[2:])} against data's "
                f"{tuple(data.shape[2:])}. The crop needs the full-resolution label; the "
                "first deep-supervision scale is no longer [1, 1, 1].")
        patches, failed, fills = [], [], []
        for b in range(data.shape[0]):
            patch, gate_failed, fill = self._crop_one(data[b], label[b, 0])
            patches.append(patch); failed.append(gate_failed); fills.append(fill)
        data_dict[self.output_key] = torch.stack(patches, dim=0)
        # Loud over silent: the fallback is
        # recorded per item, never absorbed.
        data_dict["patch_gate_failed"] = np.array(failed, dtype=bool)
        data_dict["patch_fill"] = np.array(fills, dtype=np.float32)
        return data_dict

    def _crop_one(self, volume: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        shape = np.array(label.shape, dtype=int)
        if (shape < self.patch_size).any():
            raise DataloaderError(
                f"global window {tuple(shape)} is smaller than the local patch "
                f"{tuple(self.patch_size)} on some axis")
        mask = label == self.label
        centres = torch.nonzero(mask, as_tuple=False)
        if centres.shape[0] == 0:
            # Distinct from the gate fallback below: that one handles "the
            # placenta is present but too thin for a 20³ box"; this is "there is
            # no placenta in this window at all", so no centre is even eligible
            # and there is no best patch to fall back to. If it fires, something
            # upstream is wrong and absorbing it would hide that.
            raise DataloaderError(
                "no placenta voxels in the augmented window; a local patch cannot be drawn. "
                "This is a hard stop, not a skipped sample, and not the same condition as "
                "the gate fallback.")
        centres_np = centres.numpy() if centres.device.type == "cpu" else centres.cpu().numpy()

        half = self.patch_size // 2
        upper = shape - (self.patch_size - half)
        valid = ((centres_np >= half) & (centres_np <= upper)).all(axis=1)
        pool = centres_np[valid] if valid.any() else centres_np

        low_max = shape - self.patch_size
        best_fill, best_sl, best_lo = -1.0, None, None
        for attempt in range(1, self.max_attempts + 1):
            centre = pool[self._randint(pool.shape[0])]
            lo = np.clip(centre - half, 0, low_max)          # clamp inward, never pad
            hi = lo + self.patch_size
            sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
            patch_mask = mask[sl]
            # The fraction is computed as an exact integer ratio, not as a
            # float32 tensor mean: at 20³ a patch that is exactly 33% placenta
            # (2640 of 8000) gives a float32 mean of 0.33000001311302185 and
            # would pass a strictly-greater test it should fail. The ratio of
            # two integers in float64 is exact here.
            fill = int(patch_mask.sum()) / patch_mask.numel()
            if fill > best_fill:
                best_fill, best_sl, best_lo = fill, sl, lo
            if fill > self.threshold:
                if self.record_attempts is not None:
                    self.record_attempts.append(attempt)
                if self.record_slices is not None:
                    self.record_slices.append((int(lo[0]), int(lo[1]), int(lo[2])))
                return (volume[(slice(None), *sl)] * patch_mask.to(volume.dtype),
                        False, fill)

        # Exhausted. Crop at the BEST centre seen, not the last one, and flag
        # it. The failure mode is "this placenta is too thin for a 20³ box at
        # any position in this window", not "the crop missed the anatomy", so
        # training on the best available patch is a near miss; raising here
        # would end the run, because `NonDetMultiThreadedAugmenter` cannot
        # survive a worker exception.
        if self.record_attempts is not None:
            self.record_attempts.append(self.max_attempts)
        if self.record_slices is not None:
            self.record_slices.append((int(best_lo[0]), int(best_lo[1]), int(best_lo[2])))
        return (volume[(slice(None), *best_sl)] * mask[best_sl].to(volume.dtype),
                True, best_fill)


def append_patch_transform(transforms, **kwargs) -> None:
    """Append :class:`PlacentaPatchTransform` to a composed nnU-Net chain.

    Appending rather than inserting is deliberate: at the end of the chain the
    ignore label has been removed (`nnUNetTrainer.py:734`) and the full-resolution
    label is still reachable as ``target[0]``.
    """
    transforms.transforms.append(PlacentaPatchTransform(**kwargs))
