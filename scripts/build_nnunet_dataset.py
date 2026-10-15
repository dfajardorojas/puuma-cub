"""Build the nnU-Net dataset from `cohort/cohort.csv`, then plan and preprocess.

    python scripts/build_nnunet_dataset.py

Four stages: the case-identifier mapping (B1), the raw dataset (B2), planning and
preprocessing through the API (B3), and `splits_final.json` (B4).

**The shim is imported first.** nnU-Net's three data roots are
hardcoded into the clone, 43 modules bind them at import time, and binding must
precede trainer *construction*, not merely the training call. Nothing from
`nnunetv2` may be imported above `bind_nnunet_paths`.

**Never the console scripts**: `nnUNetv2_plan_and_preprocess`
does not import the shim and would write into the read-only clone. The API is
called directly instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# --- the shim, before anything from nnunetv2 --------------------------------
from puuma.nnunet_paths import (  # noqa: E402
    ResultsRoot,
    bind_nnunet_paths,
    verify_bindings,
)

PATHS = bind_nnunet_paths(ResultsRoot.CAMPAIGN)

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from puuma.masks import load_volume, validate_binary_mask, image_spatial_shape  # noqa: E402
from puuma.nnunet_dataset import (  # noqa: E402
    DATASET_VARIANTS,
    FILE_ENDING,
    DatasetBuildError,
    build_case_mapping,
    build_splits,
    check_label_volume,
    combine_masks,
    placenta_only_label,
)

import argparse  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--variant", default="placenta_and_brain", choices=sorted(DATASET_VARIANTS),
                 help="which dataset to build. `placenta_only` is the no_brain_seg "
                      "row: the same volumes with the placenta mask only, so the row "
                      "is the same model on a placenta-only target rather than a "
                      "runtime special case.")
_ap.add_argument("--skip-preprocess", action="store_true",
                 help="run B1/B2/B4 only. Used to regenerate the raw dataset and its "
                      "audit without re-running the planner.")
_ap.add_argument("--skip-mapping", action="store_true",
                 help="do not rewrite cohort/case_id_mapping.csv. The identifiers are "
                      "identical across variants, so the second build must not touch "
                      "the artefact the first one wrote.")
ARGS = _ap.parse_args()
DATASET_ID, DATASET_NAME, LABELS = DATASET_VARIANTS[ARGS.variant]

COHORT_CSV = REPO / "cohort" / "cohort.csv"
MAPPING_CSV = REPO / "cohort" / "case_id_mapping.csv"
RAW = Path(PATHS.raw) / DATASET_NAME
PREPROCESSED = Path(PATHS.preprocessed) / DATASET_NAME
CONFIG = "3d_fullres"
PLACENTA_ONLY = ARGS.variant == "placenta_only"
DIVIDER = "=" * 78


def banner(title: str) -> None:
    print(f"\n{DIVIDER}\n{title}\n{DIVIDER}")


def under_write_root(path: Path) -> bool:
    return path.resolve().is_relative_to(REPO.resolve())


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------
def stage_mapping(cohort: pd.DataFrame) -> pd.DataFrame:
    banner("B1 — case-identifier mapping")
    mapping = build_case_mapping(cohort["scan_id"])
    mapping = mapping.merge(cohort[["scan_id", "mother_id", "fold",
                                    "is_evaluation_scan"]], on="scan_id")
    mapping.to_csv(MAPPING_CSV, index=False)
    print(f"  {len(mapping)} cases, bijective and legal for nnU-Net")
    print(f"  written to {MAPPING_CSV.relative_to(REPO)}")
    print(f"  example: {mapping['scan_id'].iloc[0]} -> "
          f"{mapping['case_identifier'].iloc[0]}")
    return mapping


# ---------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------
def stage_raw(cohort: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    banner("B2 — raw dataset")
    images_dir, labels_dir = RAW / "imagesTr", RAW / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    identifier_of = dict(zip(mapping["scan_id"], mapping["case_identifier"]))
    rows, n_symlinked, n_extracted = [], 0, 0

    for _, row in cohort.iterrows():
        case = identifier_of[row["scan_id"]]
        spatial = image_spatial_shape(row["image_path"])

        # Images are REFERENCED, never copied. A symlink suffices
        # wherever the source is already 3-D. Where it is 4-D the volume-0 rule
        # has to be materialised, because nnU-Net reads one 3-D volume per
        # channel file and would otherwise see the echo axis.
        target = images_dir / f"{case}_0000{FILE_ENDING}"
        if target.is_symlink() or target.exists():
            target.unlink()
        source = nib.load(row["image_path"])
        if len(source.shape) == 4:
            volume = np.asarray(source.dataobj)[..., 0]
            nib.save(nib.Nifti1Image(volume, source.affine, source.header), target)
            n_extracted += 1
        else:
            target.symlink_to(row["image_path"])
            n_symlinked += 1

        placenta, placenta_audit = load_volume(row["mask_path"])
        validate_binary_mask(placenta, placenta_audit)
        if PLACENTA_ONLY:
            # The brain mask is never read for this dataset — not read and then
            # dropped, which would depend on precedence firing and being undone.
            brain = np.zeros_like(placenta)
            label = placenta_only_label(placenta)
        else:
            brain, brain_audit = load_volume(row["brain_mask_path"])
            validate_binary_mask(brain, brain_audit)
            label = combine_masks(placenta, brain)
        check_label_volume(label, spatial, case, allowed=frozenset(LABELS.values()))

        reference = nib.load(row["mask_path"])
        affine = reference.affine
        nib.save(nib.Nifti1Image(label, affine, dtype=np.uint8),
                 labels_dir / f"{case}{FILE_ENDING}")

        rows.append({
            "case_identifier": case, "scan_id": row["scan_id"],
            "spatial_shape": str(spatial),
            "image_was_4d": len(source.shape) == 4,
            "placenta_before": int((placenta > 0).sum()),
            "brain_before": int((brain > 0).sum()),
            "overlap": int(((placenta > 0) & (brain > 0)).sum()),
            "placenta_after": int((label == LABELS["placenta"]).sum()),
            "brain_after": int((label == LABELS["brain"]).sum()) if "brain" in LABELS else 0,
        })

    audit = pd.DataFrame(rows)
    if len(audit) != len(cohort):
        raise DatasetBuildError(
            f"{len(cohort) - len(audit)} case(s) were skipped; no case may be "
            f"skipped silently")
    print(f"  {len(audit)} cases written — no case skipped")
    print(f"  images: {n_symlinked} symlinked (referenced, not copied), "
          f"{n_extracted} volume-0 extracts written because the source is 4-D")

    dataset_json = {
        "channel_names": {"0": "T2s"},
        "labels": dict(LABELS),
        "numTraining": len(audit),
        "file_ending": FILE_ENDING,
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    (RAW / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"  dataset.json labels: {dataset_json['labels']}  "
          f"(plain ints -> has_regions False, {len(LABELS)} segmentation heads)")

    banner("B2 — precedence rule, demonstrated")
    contested = audit[audit["overlap"] > 0]
    print(f"  cases where the two masks overlap: {len(contested)}")
    if len(contested):
        print(contested[["case_identifier", "placenta_before", "brain_before",
                         "overlap", "placenta_after", "brain_after"]].to_string(index=False))
        for _, entry in contested.iterrows():
            lost = entry["placenta_before"] - entry["placenta_after"]
            kept = entry["brain_after"] - entry["brain_before"]
            if lost != entry["overlap"]:
                raise DatasetBuildError(
                    f"{entry['case_identifier']}: placenta lost {lost} voxels but "
                    f"{entry['overlap']} were contested; precedence did not fire "
                    f"as specified")
            if kept != 0:
                raise DatasetBuildError(
                    f"{entry['case_identifier']}: brain gained {kept} voxels; "
                    f"brain should be unchanged by precedence")
            print(f"  {entry['case_identifier']}: placenta "
                  f"{entry['placenta_before']} -> {entry['placenta_after']} "
                  f"(lost {lost} = the {entry['overlap']} contested voxels); "
                  f"brain unchanged at {entry['brain_after']}")
    uncontested = audit[audit["overlap"] == 0]
    if not (uncontested["placenta_before"] == uncontested["placenta_after"]).all():
        raise DatasetBuildError("placenta changed on a case with no overlap")
    print(f"  the other {len(uncontested)} cases: placenta unchanged, as expected")
    # The variant name is part of the audit filename so that building a second
    # variant cannot overwrite the first variant's audit. The default variant
    # keeps the unsuffixed name.
    suffix = "" if ARGS.variant == "placenta_and_brain" else f"_{ARGS.variant}"
    audit.to_csv(REPO / "cohort" / f"nnunet_label_audit{suffix}.csv", index=False)
    return audit


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------
def stage_preprocess() -> dict:
    banner("B3 — planning and preprocessing (API, never the console script)")
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess,
    )
    extract_fingerprints([DATASET_ID], num_processes=8, check_dataset_integrity=True)
    plan_experiments([DATASET_ID])
    preprocess([DATASET_ID], configurations=[CONFIG], num_processes=[8])

    plans = json.loads((PREPROCESSED / "nnUNetPlans.json").read_text())
    chosen = plans["configurations"][CONFIG]
    print("\n  what the PLANNER chose — NOT authoritative:")
    for key in ("patch_size", "batch_size", "spacing", "UNet_class_name"):
        if key in chosen:
            print(f"    {key:18s} {chosen[key]}")
    print("    ^ The planner's patch and batch are for a PlainConvUNet on a")
    print("      planner-derived grid and are not used.")
    return chosen


# ---------------------------------------------------------------------------
# B4
# ---------------------------------------------------------------------------
def stage_splits(cohort: pd.DataFrame, mapping: pd.DataFrame) -> Path:
    banner("B4 — splits_final.json")
    splits = build_splits(cohort, mapping)
    path = PREPROCESSED / "splits_final.json"
    path.write_text(json.dumps(splits, indent=2))
    print(f"  written to {path.relative_to(REPO)}")
    merged = cohort.merge(mapping[["scan_id", "case_identifier"]], on="scan_id")
    for iteration, entry in enumerate(splits):
        val_fold = (iteration - 1) % 5
        train_mothers = merged[merged["case_identifier"].isin(entry["train"])]["mother_id"].nunique()
        val_mothers = merged[merged["case_identifier"].isin(entry["val"])]["mother_id"].nunique()
        print(f"    t={iteration}  test=fold {iteration}  val=fold {val_fold}  "
              f"train {len(entry['train']):3d} scans / {train_mothers:3d} mothers   "
              f"val {len(entry['val']):3d} scans / {val_mothers:3d} mothers")
    print("  train/val asymmetry in scans per mother is intended")
    return path


def main() -> None:
    cohort = pd.read_csv(COHORT_CSV)
    print(f"variant: {ARGS.variant} -> {DATASET_NAME} (id {DATASET_ID}), labels {LABELS}")
    print(f"cohort: {len(cohort)} scans / {cohort['mother_id'].nunique()} mothers")
    if ARGS.skip_mapping:
        banner("B1 — case-identifier mapping (REUSED, not rewritten)")
        mapping = pd.read_csv(MAPPING_CSV)
        rebuilt = build_case_mapping(cohort["scan_id"])
        if sorted(rebuilt["case_identifier"]) != sorted(mapping["case_identifier"]):
            raise DatasetBuildError(
                "the existing mapping does not match a recomputation; the two datasets "
                "would not share case identifiers and splits_final.json would not apply")
        print(f"  {len(mapping)} identifiers reused and verified against a recomputation")
    else:
        mapping = stage_mapping(cohort)
    stage_raw(cohort, mapping)
    if not ARGS.skip_preprocess:
        stage_preprocess()
    stage_splits(cohort, mapping)

    banner("E — containment and bindings")
    verify_bindings(PATHS)
    for name, path in PATHS.as_dict().items():
        inside = under_write_root(Path(path))
        print(f"  {name:22s} {path}  under write root: {inside}")
        if not inside:
            raise DatasetBuildError(f"{name} escaped the write root")
    print("\n  BUILD COMPLETE")


if __name__ == "__main__":
    main()
