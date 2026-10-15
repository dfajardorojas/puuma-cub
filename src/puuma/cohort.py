"""Cohort rules: categories, folds, the tie-break and per-iteration roles.

This module is the single place each rule is expressed:
`scripts/build_cohort.py` orchestrates, and consumers recompute against these
functions. No `nnunetv2` import, so the paths shim does not apply to the
cohort build.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# --- seeds, both recorded in the run config ---------------------------------
TIE_BREAK_SEED: int = 20260818
#: `StratifiedKFold(n_splits=5, shuffle=True, random_state=1)`.
FOLD_SEED: int = 1
N_FOLDS: int = 5

# --- CL sentinel semantics --------------------------------------------------
#: Values in `data_all` that mean "missing". `tag_cervix_length` has no NaN of
#: its own — missingness is encoded entirely by these, and in practice only -1
#: ever appears in the CL column.
MISSING_SENTINELS: tuple[int, int] = (-1, -5)

# --- the four categories, left-closed and right-open ------------------------
CATEGORY_TERM = "term"
CATEGORY_LATE = "late preterm"
CATEGORY_VERY = "very preterm"
CATEGORY_EXTREME = "extremely preterm"

#: `(lower, upper, name)`, evaluated as `lower <= GA_del < upper`.
CATEGORY_BANDS: tuple[tuple[float, float, str], ...] = (
    (-np.inf, 28.0, CATEGORY_EXTREME),
    (28.0, 32.0, CATEGORY_VERY),
    (32.0, 37.0, CATEGORY_LATE),
    (37.0, np.inf, CATEGORY_TERM),
)
CATEGORY_ORDER: tuple[str, ...] = (
    CATEGORY_TERM, CATEGORY_LATE, CATEGORY_VERY, CATEGORY_EXTREME)

#: The `det_preterm_cat` codes and their category names.
DET_PRETERM_CAT_TO_CATEGORY: dict[int, str] = {
    0: CATEGORY_TERM, 1: CATEGORY_LATE, 2: CATEGORY_VERY, 3: CATEGORY_EXTREME}


class CohortError(RuntimeError):
    """A cohort rule or expected figure was violated. Always fatal."""


def category_of(ga_del: float) -> str:
    """The category for a delivery GA.

    Left-closed and right-open, so `GA_del == 37.0` is term and
    `GA_del == 32.0` is late preterm.

    :raises CohortError: if `ga_del` is missing or falls through every band.
    """
    if ga_del is None or (isinstance(ga_del, float) and np.isnan(ga_del)):
        raise CohortError("GA_del is missing; the category is undefined")
    for lower, upper, name in CATEGORY_BANDS:
        if lower <= ga_del < upper:
            return name
    raise CohortError(f"GA_del {ga_del!r} fell through every category band")


# ---------------------------------------------------------------------------
# The tie-break
# ---------------------------------------------------------------------------
def tie_break_rng(mother_id: str, seed: int = TIE_BREAK_SEED) -> np.random.Generator:
    """A generator seeded from `(seed, mother_id)` and nothing else.

    SHA-256 rather than `hash()`: Python salts string hashing per process, so a
    `hash()`-seeded draw would differ between runs and the determinism
    assertion could never pass.
    """
    digest = hashlib.sha256(f"{seed}:{mother_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def choose_evaluation_scan(mother_id: str, candidate_scan_ids: Sequence[str],
                           seed: int = TIE_BREAK_SEED) -> str:
    """Draw one scan uniformly from a mother's tied earliest-GA scans.

    Candidates are sorted before the draw, so the result depends on the *set* of
    tied scans and not on the order they arrived in: it does not depend on row
    order in the source table.

    :raises CohortError: if there are no candidates.
    """
    ordered = sorted(candidate_scan_ids)
    if not ordered:
        raise CohortError(f"{mother_id}: no candidate scan to draw from")
    if len(ordered) == 1:
        return ordered[0]
    return ordered[int(tie_break_rng(mother_id, seed).integers(len(ordered)))]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def assign_folds(mother_ids: Sequence[str], categories: Sequence[str],
                 n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    """One fold in `[0, n_folds)` per mother, stratified on category.

    Mothers are sorted by `mother_id` first so the split does not depend on the
    order rows happened to be read in.

    :raises CohortError: if any category holds fewer mothers than folds.
    """
    if len(mother_ids) != len(categories):
        raise CohortError("mother_ids and categories differ in length")
    frame = (pd.DataFrame({"mother_id": list(mother_ids),
                           "category": list(categories)})
             .sort_values("mother_id")
             .reset_index(drop=True))
    counts = frame["category"].value_counts()
    too_small = {c: int(n) for c, n in counts.items() if n < n_folds}
    if too_small:
        raise CohortError(
            f"category/categories {too_small} hold fewer mothers than the "
            f"{n_folds} folds requested, so at least one fold would receive none")

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of: dict[str, int] = {}
    placeholder = np.zeros(len(frame))
    for fold, (_, held_out) in enumerate(splitter.split(placeholder, frame["category"])):
        for index in held_out:
            fold_of[frame.at[int(index), "mother_id"]] = fold
    if len(fold_of) != len(frame):
        raise CohortError("StratifiedKFold did not assign every mother exactly once")
    return fold_of


def fold_roles(iteration: int, n_folds: int = N_FOLDS) -> tuple[int, int, list[int]]:
    """The cyclical scheme for iteration *t*.

    :returns: `(test_fold, validation_fold, training_folds)`.
    """
    if not 0 <= iteration < n_folds:
        raise CohortError(f"iteration {iteration} outside [0, {n_folds})")
    test = iteration
    validation = (iteration - 1) % n_folds
    training = [f for f in range(n_folds) if f not in (test, validation)]
    return test, validation, training


def role_of(fold: int, iteration: int, n_folds: int = N_FOLDS) -> str:
    """`'train'`, `'val'` or `'test'` for a mother's fold at iteration *t*.

    Roles are derived, never stored: five stored role columns would be five
    chances to drift from the one `fold` column that defines them.
    """
    test, validation, _ = fold_roles(iteration, n_folds)
    if fold == test:
        return "test"
    if fold == validation:
        return "val"
    return "train"


def split_for_iteration(cohort: pd.DataFrame, iteration: int,
                        n_folds: int = N_FOLDS) -> dict[str, pd.DataFrame]:
    """The three row sets for iteration *t*, applying the evaluation-scan deduplication.

    Training keeps every surviving scan of a training mother; validation and
    test keep only the scan marked `is_evaluation_scan`. The two are
    deliberately not symmetric in scans per mother.
    """
    roles = cohort["fold"].map(lambda f: role_of(int(f), iteration, n_folds))
    return {
        "train": cohort[roles == "train"].copy(),
        "val": cohort[(roles == "val") & cohort["is_evaluation_scan"]].copy(),
        "test": cohort[(roles == "test") & cohort["is_evaluation_scan"]].copy(),
    }


# ---------------------------------------------------------------------------
# Cohort-build steps `scripts/build_cohort.py` applies, in its order.
# ---------------------------------------------------------------------------
#: Plausible gestational-age ranges, weeks.
GA_RANGE: tuple[float, float] = (10.0, 45.0)
GADEL_RANGE: tuple[float, float] = (15.0, 46.0)
#: Inclusion ceiling on GA at scan, weeks.
GA_SCAN_MAX: float = 37.0
#: Tolerance when comparing gestational ages.
GA_TOL: float = 1e-4

#: `cohort.csv`'s emitted schema.
COLUMNS = ["scan_id", "mother_id", "study", "image_path", "mask_path",
           "brain_mask_path", "tag_ga", "tag_gadel", "cl_value", "has_cl",
           "category", "det_preterm_cat", "fold", "is_evaluation_scan",
           "n_scans_mother"]


def apply_ga_plausibility_guard(frame, ga_col: str, gadel_col: str,
                                cl_col: str | None = None):
    """Plausible-range guard.

    Values outside the ranges are reported and set to NaN; an excluded row that
    carries a CL value is a hard stop when `cl_col` is given.
    """
    import numpy as np
    bad_ga = frame[ga_col].notna() & ~frame[ga_col].between(*GA_RANGE)
    bad_gadel = frame[gadel_col].notna() & ~frame[gadel_col].between(*GADEL_RANGE)
    excluded = frame[bad_ga | bad_gadel]
    print(f"  {ga_col} outside {GA_RANGE}: {int(bad_ga.sum())} rows; "
          f"{gadel_col} outside {GADEL_RANGE}: {int(bad_gadel.sum())} rows")
    if len(excluded):
        print(excluded.to_string(index=False))
        if cl_col is not None and excluded[cl_col].notna().any():
            raise CohortError("a range-guard exclusion carries a CL value; "
                              "inspect before proceeding")
    frame = frame.copy()
    frame.loc[bad_ga, ga_col] = np.nan
    frame.loc[bad_gadel, gadel_col] = np.nan
    return frame


def assert_gadel_constant_within_mother(cohort, gadel_col: str) -> None:
    """One pregnancy, one delivery date."""
    spread = cohort.groupby("mother_id")[gadel_col].nunique()
    if (spread > 1).any():
        raise CohortError(f"{gadel_col} varies within mother(s): "
                          f"{sorted(spread[spread > 1].index)}")


def assert_every_stratum_present(cohort) -> None:
    """Every category in every fold."""
    table = (cohort.groupby(["fold", "category"])["mother_id"].nunique()
             .unstack(fill_value=0).reindex(columns=list(CATEGORY_ORDER), fill_value=0))
    print(table.to_string())
    missing = [(int(f), c) for f in table.index for c in CATEGORY_ORDER
               if table.at[f, c] == 0]
    if missing:
        raise CohortError(f"stratum absent from a fold: {missing} — assertion (d)")
    print("  OK  every stratum present in every fold")


def select_evaluation_scans(cohort, ga_col: str,
                            seed: int = TIE_BREAK_SEED,
                            ga_tol: float = GA_TOL):
    """Per-mother evaluation scan at the earliest GA, with the seeded tie-break.

    :returns: `(cohort with is_evaluation_scan set, number of tied mothers)`.
    """
    import numpy as np
    cohort = cohort.copy()
    cohort["is_evaluation_scan"] = False
    n_tied = 0
    for mother, group in cohort.groupby("mother_id"):
        earliest = group[ga_col].min()
        tied = group[np.abs(group[ga_col] - earliest) < ga_tol]
        if tied["cl_value"].nunique() > 1:
            raise CohortError(
                f"{mother}: tied earliest-GA scans carry different CL values "
                f"{sorted(tied['cl_value'].unique())}; they must share one CL value")
        if len(tied) > 1:
            n_tied += 1
        chosen = choose_evaluation_scan(str(mother), tied["scan_id"].tolist(),
                                        seed=seed)
        cohort.loc[cohort["scan_id"] == chosen, "is_evaluation_scan"] = True
    return cohort, n_tied


def add_n_scans_mother(cohort):
    """Scans per mother."""
    cohort = cohort.copy()
    cohort["n_scans_mother"] = cohort.groupby("mother_id")["scan_id"].transform("size")
    return cohort
