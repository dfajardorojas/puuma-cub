"""Loading and validating the placenta and brain segmentation masks.

Two guards live here, both hard stops:

* Validity: a mask must be integer-valued, non-negative, and its unique values
  must be a subset of the expected set; anything else is rejected by rule,
  never by filename.
* Dimensionality: some images and masks are 4-D, `(X, Y, Z, echoes)`, and 4-D
  masks are not constant along the echo axis — they are per-echo segmentations.
  Volume 0 is taken, per file, and every case where the rule fires is logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

#: Where a file is 4-D, this volume is the one used.
VOLUME_INDEX: int = 0

#: What a binary segmentation mask may contain.
BINARY_MASK_VALUES: frozenset[int] = frozenset({0, 1})


class MaskValidationError(RuntimeError):
    """A mask file does not contain a usable segmentation. Always fatal."""


@dataclass(frozen=True)
class VolumeAudit:
    """What was found in one file, and whether the volume-0 rule fired."""

    path: str
    ndim: int
    shape_on_disk: tuple[int, ...]
    shape_used: tuple[int, ...]
    took_volume_0: bool
    n_unique: int
    minimum: float
    maximum: float

    def describe(self) -> str:
        rule = f" -> volume {VOLUME_INDEX}" if self.took_volume_0 else ""
        return (f"{Path(self.path).name}  ndim {self.ndim} {self.shape_on_disk}{rule}"
                f"  values [{self.minimum:g}, {self.maximum:g}] "
                f"({self.n_unique} unique)")


def load_volume(path: str | Path) -> tuple[np.ndarray, VolumeAudit]:
    """Load a NIfTI volume, taking volume 0 if it is 4-D.

    :returns: the array actually used, and an audit of what was on disk.
    :raises MaskValidationError: if the file is neither 3-D nor 4-D.
    """
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    on_disk = tuple(int(n) for n in array.shape)
    if array.ndim == 4:
        array = array[..., VOLUME_INDEX]
        took = True
    elif array.ndim == 3:
        took = False
    else:
        raise MaskValidationError(
            f"{path}: expected a 3-D or 4-D volume, found ndim {array.ndim} "
            f"with shape {on_disk}")
    unique = np.unique(array)
    audit = VolumeAudit(
        path=str(path), ndim=len(on_disk), shape_on_disk=on_disk,
        shape_used=tuple(int(n) for n in array.shape), took_volume_0=took,
        n_unique=int(unique.size), minimum=float(unique.min()),
        maximum=float(unique.max()))
    return array, audit


def validate_binary_mask(array: np.ndarray, audit: VolumeAudit,
                         allowed: frozenset[int] = BINARY_MASK_VALUES) -> None:
    """Require `array` to be a segmentation mask over `allowed`.

    Rejects by rule, never by filename: non-integer values, negatives, or any
    value outside `allowed`.

    :raises MaskValidationError: naming the failing values.
    """
    unique = np.unique(array)
    if not np.all(np.isfinite(unique)):
        raise MaskValidationError(
            f"{audit.path}: mask contains non-finite values")
    if not np.allclose(unique, np.round(unique)):
        raise MaskValidationError(
            f"{audit.path}: mask is not integer-valued — {audit.n_unique} unique "
            f"values spanning [{audit.minimum:g}, {audit.maximum:g}]; a "
            f"segmentation cannot be continuous")
    if audit.minimum < 0:
        raise MaskValidationError(
            f"{audit.path}: mask contains negative values (minimum "
            f"{audit.minimum:g}); a segmentation label cannot be negative")
    found = {int(round(float(v))) for v in unique}
    unexpected = sorted(found - set(allowed))
    if unexpected:
        raise MaskValidationError(
            f"{audit.path}: mask contains value(s) {unexpected} outside the "
            f"expected set {sorted(allowed)}")


def image_spatial_shape(path: str | Path) -> tuple[int, ...]:
    """The three spatial dimensions of an image, ignoring any echo axis."""
    shape = tuple(int(n) for n in nib.load(str(path)).shape)
    if len(shape) == 4:
        return shape[:3]
    if len(shape) == 3:
        return shape
    raise MaskValidationError(
        f"{path}: expected a 3-D or 4-D image, found shape {shape}")


# ---------------------------------------------------------------------------
# The cohort-level mask guard `scripts/build_cohort.py` applies.
# ---------------------------------------------------------------------------
def nib_shape(path: str) -> tuple[int, ...]:
    """Shape of a NIfTI file without reading its data."""
    import nibabel as nib
    return tuple(int(n) for n in nib.load(str(path)).shape)


def validate_masks(cohort, out_dir):
    """Structural guard: both masks must be usable segmentations.

    Applies the volume-0 rule per file where a volume is 4-D, requires both
    masks to be binary over `{0, 1}`, and requires their geometry to match the
    image's spatial shape once the rule has been applied.

    Rejection is by rule, never by filename. Every exclusion is logged with the
    failing values, and every case where the volume-0 rule fired is logged too.

    :returns: `(surviving cohort, excluded scans, volume-0 firings)`.
    """
    import pandas as pd
    from pathlib import Path as _Path
    out_dir = _Path(out_dir)
    print("B1b — masks are usable segmentations (guard)")
    keep, excluded, fired = [], [], []
    for _, row in cohort.iterrows():
        spatial = image_spatial_shape(row["image_path"])
        image_ndim = len(nib_shape(row["image_path"]))
        if image_ndim == 4:
            fired.append({"scan_id": row["scan_id"], "file": "image",
                          "path": row["image_path"],
                          "shape_on_disk": str(nib_shape(row["image_path"]))})
        failure: str | None = None
        for column, label in (("mask_path", "placenta"), ("brain_mask_path", "brain")):
            # Narrow: only a mask that fails validation is an exclusion. Anything
            # else — a missing file, an unreadable one — is a bug and propagates.
            try:
                array, audit = load_volume(row[column])
                validate_binary_mask(array, audit)
            except MaskValidationError as exc:
                failure = f"{label}: {exc}"
                break
            if audit.took_volume_0:
                fired.append({"scan_id": row["scan_id"], "file": label,
                              "path": row[column],
                              "shape_on_disk": str(audit.shape_on_disk)})
            if audit.shape_used != spatial:
                failure = (f"{label}: geometry {audit.shape_used} does not match "
                           f"the image's spatial shape {spatial}")
                break
        if failure is None:
            keep.append(row["scan_id"])
        else:
            excluded.append({"scan_id": row["scan_id"],
                             "mother_id": row["mother_id"], "reason": failure})

    excluded_frame = pd.DataFrame(excluded, columns=["scan_id", "mother_id", "reason"])
    fired_frame = pd.DataFrame(fired, columns=["scan_id", "file", "path", "shape_on_disk"])
    print(f"  volume-0 rule fired on {len(fired_frame)} file(s) across "
          f"{fired_frame['scan_id'].nunique() if len(fired_frame) else 0} scan(s)")
    if len(fired_frame):
        by_kind = fired_frame["file"].value_counts().to_dict()
        print(f"    by kind: {by_kind}")
    print(f"  scans excluded by the mask guard: {len(excluded_frame)}")
    for entry in excluded:
        print(f"    {entry['scan_id']} ({entry['mother_id']}): {entry['reason']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded_frame.to_csv(out_dir / "excluded_scans.csv", index=False)
    fired_frame.to_csv(out_dir / "volume0_rule_applied.csv", index=False)
    return cohort[cohort["scan_id"].isin(keep)], excluded_frame, fired_frame
