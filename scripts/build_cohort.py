"""Build `cohort/cohort.csv` from a user-supplied scan manifest.

**The manifest** is one CSV with these columns (one row per scan; files
present and visually inspected before they get here):

    scan_id, mother_id, image_path, placenta_mask_path, brain_mask_path,
    ga_scan, ga_del, cl

An optional `study` column is carried through; absent, `study` is set to
"unspecified". `cl` may be empty where no cervical-length measurement exists;
a value of 0.0 is a real measurement, not a missing one. Gestational ages are
in weeks.

Usage::

    python scripts/build_cohort.py --manifest my_manifest.csv

The defaults reproduce the paper's configuration: 5 folds,
`StratifiedKFold(shuffle=True, random_state=1)` over mothers stratified on the
delivery category, and the SHA-256 tie-break seeded with 20260818
(`puuma/cohort.py` `N_FOLDS`, `FOLD_SEED`, `TIE_BREAK_SEED`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from puuma.cohort import (  # noqa: E402
    COLUMNS,
    DET_PRETERM_CAT_TO_CATEGORY,
    FOLD_SEED,
    GA_SCAN_MAX,
    N_FOLDS,
    TIE_BREAK_SEED,
    CohortError,
    add_n_scans_mother,
    apply_ga_plausibility_guard,
    assert_every_stratum_present,
    assert_gadel_constant_within_mother,
    assign_folds,
    category_of,
    select_evaluation_scans,
)
from puuma.masks import validate_masks  # noqa: E402

#: The manifest's required columns.
MANIFEST_COLUMNS = ["scan_id", "mother_id", "image_path", "placenta_mask_path",
                    "brain_mask_path", "ga_scan", "ga_del", "cl"]

#: The category -> det_preterm_cat code, inverted from the library's mapping so
#: the emitted column cannot disagree with `category`.
CATEGORY_TO_DET_CODE = {name: code
                        for code, name in DET_PRETERM_CAT_TO_CATEGORY.items()}

DIVIDER = "=" * 78


def banner(title: str) -> None:
    print(f"\n{DIVIDER}\n{title}\n{DIVIDER}")


def load_manifest(path: Path) -> pd.DataFrame:
    banner("A — manifest")
    frame = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing:
        raise CohortError(f"manifest lacks required column(s): {missing}")
    if "study" not in frame.columns:
        frame["study"] = "unspecified"
    if frame["scan_id"].duplicated().any():
        raise CohortError("scan_id is not unique across the manifest")
    print(f"  {len(frame)} scans, {frame['mother_id'].nunique()} mothers")

    banner("A2 — plausible-range guard")
    frame = apply_ga_plausibility_guard(frame, "ga_scan", "ga_del", cl_col="cl")
    return frame


def inclusion_filters(cohort: pd.DataFrame, ga_scan_max: float) -> pd.DataFrame:
    banner("B1 — inclusion filters")

    def report(label: str, frame: pd.DataFrame, before: int) -> None:
        print(f"    {label:<32s} scans {len(frame):4d}  mothers "
              f"{frame['mother_id'].nunique():4d}  removed {before - len(frame):3d}")

    n = len(cohort); report("as loaded", cohort, n)
    n = len(cohort); cohort = cohort[cohort["ga_scan"] < ga_scan_max]
    report(f"GA at scan < {ga_scan_max:g}", cohort, n)
    n = len(cohort); cohort = cohort[cohort["ga_del"] >= cohort["ga_scan"]]
    report("GA at delivery >= GA at scan", cohort, n)
    n = len(cohort)
    present = (cohort["image_path"].map(os.path.exists)
               & cohort["placenta_mask_path"].map(os.path.exists)
               & cohort["brain_mask_path"].notna()
               & cohort["brain_mask_path"].map(os.path.exists))
    cohort = cohort[present]
    report("images/masks on disk", cohort, n)

    n = len(cohort); cohort = cohort[cohort["cl"].notna()]
    report("cervical length present", cohort, n)
    return cohort.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="CSV with the columns in the module docstring")
    ap.add_argument("--out-dir", type=Path, default=REPO / "cohort",
                    help="where cohort.csv and the guard logs are written")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS,
                    help=f"default {N_FOLDS}, the paper's value (puuma/cohort.py)")
    ap.add_argument("--fold-seed", type=int, default=FOLD_SEED,
                    help=f"StratifiedKFold random_state; default {FOLD_SEED}, "
                         "the paper's value")
    ap.add_argument("--tie-break-seed", type=int, default=TIE_BREAK_SEED,
                    help=f"evaluation-scan tie-break seed; default "
                         f"{TIE_BREAK_SEED}, the paper's value")
    ap.add_argument("--ga-scan-max", type=float, default=GA_SCAN_MAX,
                    help=f"inclusion ceiling on GA at scan, weeks; default "
                         f"{GA_SCAN_MAX:g}, the paper's value")
    args = ap.parse_args()

    cohort = load_manifest(args.manifest)
    cohort = inclusion_filters(cohort, args.ga_scan_max)

    # the mask guard expects the published column names
    cohort = cohort.rename(columns={"placenta_mask_path": "mask_path"})
    cohort, _excluded, _fired = validate_masks(cohort, args.out_dir)
    cohort = cohort.reset_index(drop=True)

    banner("B4 — categories")
    assert_gadel_constant_within_mother(cohort, "ga_del")
    cohort["category"] = cohort["ga_del"].map(lambda g: category_of(float(g)))
    cohort["det_preterm_cat"] = cohort["category"].map(CATEGORY_TO_DET_CODE)
    per_mother = cohort.groupby("mother_id")["category"].first()
    print(per_mother.value_counts().to_string())

    banner("C1 — folds")
    per_mother = (cohort.groupby("mother_id")["category"].first()
                  .sort_index().reset_index())
    fold_of = assign_folds(per_mother["mother_id"], per_mother["category"],
                           n_folds=args.n_folds, seed=args.fold_seed)
    cohort["fold"] = cohort["mother_id"].map(fold_of).astype(int)
    print(f"  StratifiedKFold(n_splits={args.n_folds}, shuffle=True, "
          f"random_state={args.fold_seed}) over {len(per_mother)} mothers")
    assert_every_stratum_present(cohort)

    banner("C2 — evaluation-scan tie-break")
    cohort = cohort.rename(columns={"cl": "cl_value"})
    cohort["has_cl"] = cohort["cl_value"].notna()
    cohort, n_tied = select_evaluation_scans(cohort, "ga_scan",
                                             seed=args.tie_break_seed)
    print(f"  mothers with a tie at earliest GA: {n_tied}")
    print(f"  evaluation scans (one per mother): "
          f"{int(cohort['is_evaluation_scan'].sum())}")
    cohort = add_n_scans_mother(cohort)

    banner("D — write cohort.csv")
    out = (cohort.rename(columns={"ga_scan": "tag_ga", "ga_del": "tag_gadel"})
           [COLUMNS]
           .sort_values(["mother_id", "scan_id"])
           .reset_index(drop=True))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "cohort.csv"
    out.to_csv(path, index=False)
    print(f"  {path}  {len(out)} rows x {len(COLUMNS)} columns")
    print(f"  columns: {', '.join(COLUMNS)}")


if __name__ == "__main__":
    main()
