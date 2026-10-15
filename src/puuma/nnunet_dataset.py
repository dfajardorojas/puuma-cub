"""The seam between `cohort.csv` and everything nnU-Net sees.

Three things live here because more than one consumer needs them: the
case-identifier rule, the label combination, and the split derivation. The
dataloader's scalar lookup and `splits_final.json` both key on the identifier,
so it is written to disk as an artefact rather than recomputed from a
convention. The label map is `{background: 0, placenta: 1, brain: 2}`, and
brain wins where the two masks overlap.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Plain integers, so `has_regions` is False and
#: `num_segmentation_heads` is 3 — background, placenta, brain.
LABELS: dict[str, int] = {"background": 0, "placenta": 1, "brain": 2}
LABEL_PLACENTA: int = LABELS["placenta"]
LABEL_BRAIN: int = LABELS["brain"]

#: The `no_brain_seg` row is a second dataset, not a runtime special case.
#: Same volumes, same case identifiers, placenta mask only — so the row is the
#: same model trained on a placenta-only target, with no label remapping, no
#: config branch and nothing to get wrong at batch time. `cohort.csv` and
#: `splits_final.json` apply unchanged because the identifiers are identical.
LABELS_PLACENTA_ONLY: dict[str, int] = {"background": 0, "placenta": 1}
DATASET_ID_PLACENTA_ONLY: int = 2
DATASET_NAME_PLACENTA_ONLY: str = "Dataset002_PlacentaCL"


#: The two datasets, as data. `DATASET_VARIANTS[name]` gives
#: `(dataset id, dataset name, label map)`.
DATASET_VARIANTS: dict[str, tuple[int, str, dict[str, int]]] = {}
ALLOWED_LABEL_VALUES: frozenset[int] = frozenset(LABELS.values())

FILE_ENDING: str = ".nii.gz"
DATASET_ID: int = 1
DATASET_NAME: str = "Dataset001_PlacentaBrainCL"
DATASET_VARIANTS.update({
    "placenta_and_brain": (DATASET_ID, DATASET_NAME, LABELS),
    "placenta_only": (DATASET_ID_PLACENTA_ONLY, DATASET_NAME_PLACENTA_ONLY,
                      LABELS_PLACENTA_ONLY),
})

#: A case identifier ending in `_XXXX` would be stripped as a channel index by
#: `nnunetv2/utilities/utils.py:29`, which crops `len(file_ending) + 5`.
_TRAILING_CHANNEL = re.compile(r"_\d{4}$")
_LEGAL_CHARS = re.compile(r"[A-Za-z0-9_.\-]+")


class DatasetBuildError(RuntimeError):
    """A dataset invariant was violated. Always fatal."""


# ---------------------------------------------------------------------------
# B1 — case identifiers
# ---------------------------------------------------------------------------
def case_identifier(scan_id: str) -> str:
    """The nnU-Net case identifier for a `cohort.csv` scan.

    The identity: `scan_id` is already unique, legal, and readable, and keeping
    it means `splits_final.json` and the scalar lookup stay auditable by eye.
    """
    return str(scan_id)


def check_case_identifier(identifier: str, file_ending: str = FILE_ENDING) -> None:
    """Require `identifier` to be legal for nnU-Net.

    Legality is read off the clone, not assumed:

    * `utilities/utils.py:29` derives the identifier by cropping
      `len(file_ending) + 5` characters, i.e. `_XXXX` plus the ending — so an
      identifier that itself ends in `_XXXX` would be truncated wrongly.
    * `utilities/utils.py:46` matches `re.escape(id) + r"_\\d\\d\\d\\d" + ending`
      with `fullmatch`, so the identifier must not contain the file ending.
    * `training/dataloading/utils.py:123` skips anything containing
      `segFromPrevStage`.

    :raises DatasetBuildError: naming the rule that was broken.
    """
    if not identifier:
        raise DatasetBuildError("empty case identifier")
    if _TRAILING_CHANNEL.search(identifier):
        raise DatasetBuildError(
            f"{identifier!r} ends in '_XXXX'; utilities/utils.py:29 would crop it "
            f"as a channel index")
    if file_ending in identifier:
        raise DatasetBuildError(
            f"{identifier!r} contains the file ending {file_ending!r}")
    if "segFromPrevStage" in identifier:
        raise DatasetBuildError(
            f"{identifier!r} contains 'segFromPrevStage', which "
            f"training/dataloading/utils.py:123 filters out")
    if not _LEGAL_CHARS.fullmatch(identifier):
        raise DatasetBuildError(
            f"{identifier!r} contains characters outside [A-Za-z0-9_.-]")
    # The round trip nnU-Net itself performs must be the identity.
    crop = len(file_ending) + 5
    if (identifier + "_0000" + file_ending)[:-crop] != identifier:
        raise DatasetBuildError(
            f"{identifier!r} does not survive nnU-Net's own identifier extraction")


def build_case_mapping(scan_ids: Sequence[str]) -> pd.DataFrame:
    """`scan_id` → `case_identifier`, asserted bijective and legal.

    :raises DatasetBuildError: on a duplicate on either side, or an illegal id.
    """
    frame = pd.DataFrame({"scan_id": [str(s) for s in scan_ids]})
    frame["case_identifier"] = frame["scan_id"].map(case_identifier)
    for identifier in frame["case_identifier"]:
        check_case_identifier(identifier)
    if frame["scan_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["scan_id"].duplicated(), "scan_id"])
        raise DatasetBuildError(f"duplicate scan_id: {duplicates}")
    if frame["case_identifier"].duplicated().any():
        duplicates = sorted(frame.loc[frame["case_identifier"].duplicated(),
                                      "case_identifier"])
        raise DatasetBuildError(f"duplicate case_identifier: {duplicates}")
    if len(set(frame["scan_id"])) != len(set(frame["case_identifier"])):
        raise DatasetBuildError("mapping is not bijective")
    return frame.sort_values("scan_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# B2 — the label volume
# ---------------------------------------------------------------------------
def placenta_only_label(placenta: np.ndarray) -> np.ndarray:
    """The `no_brain_seg` dataset's label volume: placenta, nothing else.

    Deliberately not `combine_masks(...)` with the brain dropped afterwards:
    that would depend on precedence having fired the right way and then be undone.
    The brain mask is never read for this dataset.
    """
    label = np.zeros(placenta.shape, dtype=np.uint8)
    label[placenta > 0] = LABEL_PLACENTA
    return label


def combine_masks(placenta: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """One label volume from the two binary masks.

    Placenta is written first, then brain over it, so a voxel claimed by both
    becomes `brain`.

    :raises DatasetBuildError: if the two masks disagree on shape.
    """
    if placenta.shape != brain.shape:
        raise DatasetBuildError(
            f"mask shapes differ: placenta {placenta.shape}, brain {brain.shape}")
    label = np.zeros(placenta.shape, dtype=np.uint8)
    label[placenta > 0] = LABEL_PLACENTA
    label[brain > 0] = LABEL_BRAIN            # brain wins
    return label


def check_label_volume(label: np.ndarray, spatial_shape: tuple[int, ...],
                       case: str, allowed: frozenset[int] | None = None) -> None:
    """Require the combined label to match its image and carry only `{0, 1, 2}`.

    :raises DatasetBuildError: naming the case and what was wrong.
    """
    if tuple(label.shape) != tuple(spatial_shape):
        raise DatasetBuildError(
            f"{case}: label geometry {label.shape} does not match the image's "
            f"spatial shape {spatial_shape}")
    permitted = ALLOWED_LABEL_VALUES if allowed is None else frozenset(allowed)
    found = {int(v) for v in np.unique(label)}
    unexpected = sorted(found - permitted)
    if unexpected:
        raise DatasetBuildError(
            f"{case}: label contains value(s) {unexpected} outside "
            f"{sorted(permitted)}")


# ---------------------------------------------------------------------------
# B4 — splits_final.json
# ---------------------------------------------------------------------------
def build_splits(cohort: pd.DataFrame, mapping: pd.DataFrame,
                 n_folds: int = 5) -> list[dict[str, list[str]]]:
    """`splits_final.json` content, one entry per iteration *t*.

    For iteration *t*: `train` is every scan of the three training folds, `val`
    is one scan per mother of fold (*t*−1) mod 5 — the one marked
    `is_evaluation_scan` — and the test fold appears in neither list. nnU-Net
    does not require every case to appear.

    The two lists are deliberately asymmetric in scans per mother.
    """
    from puuma.cohort import fold_roles

    # Only the identifier is taken from the mapping; `fold` and
    # `is_evaluation_scan` come from cohort.csv, which is their single source of
    # truth. Merging the whole mapping would suffix them and silently split the
    # authority for a column in two.
    merged = cohort.merge(mapping[["scan_id", "case_identifier"]],
                          on="scan_id", how="left")
    if merged["case_identifier"].isna().any():
        missing = sorted(merged.loc[merged["case_identifier"].isna(), "scan_id"])
        raise DatasetBuildError(f"scans absent from the mapping: {missing}")

    splits = []
    for iteration in range(n_folds):
        test_fold, val_fold, train_folds = fold_roles(iteration, n_folds)
        train = merged[merged["fold"].isin(train_folds)]
        val = merged[(merged["fold"] == val_fold) & merged["is_evaluation_scan"]]
        test = merged[merged["fold"] == test_fold]
        overlap = set(train["case_identifier"]) & set(val["case_identifier"])
        if overlap:
            raise DatasetBuildError(f"t={iteration}: train and val share {overlap}")
        leaked = (set(test["case_identifier"])
                  & (set(train["case_identifier"]) | set(val["case_identifier"])))
        if leaked:
            raise DatasetBuildError(
                f"t={iteration}: the test fold appears in train or val: {leaked}")
        splits.append({"train": sorted(train["case_identifier"]),
                       "val": sorted(val["case_identifier"])})
    return splits
