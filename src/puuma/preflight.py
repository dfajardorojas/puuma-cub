"""The gate that runs before a fold trains.

Every leakage assertion (a)–(g) is here as a function that raises, plus the
stated cohort figures; `run_preflight` executes all of them, reports each, and
raises if any fails. Nothing may train if it does.

Inference has its own gate, (h)–(k), in :func:`run_inference_preflight`.
Training's gate does not cover it: (a)–(g) are about the cohort and the split
lists, and say nothing about which weights are being read or which constants
invert the prediction — a `test_predictions.csv` produced with standardisation
constants fitted on anything other than that fold's training rows looks exactly
like one produced correctly, and every number in it is contaminated.

No `nnunetv2` import, so import-order safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from puuma.checkpoint import BEST_CHECKPOINT, FINAL_CHECKPOINT, LATEST_CHECKPOINT
from puuma.cohort import (
    FOLD_SEED, N_FOLDS, TIE_BREAK_SEED, assign_folds, category_of,
    choose_evaluation_scan, fold_roles, split_for_iteration,
)
from puuma.dataloader import ScalarTable

#: A mismatch is a hard stop, not a warning.
EXPECTED_SCANS = 190
EXPECTED_MOTHERS = 120
EXPECTED_MOTHER_CATEGORIES = {"term": 85, "late preterm": 13,
                              "very preterm": 8, "extremely preterm": 14}
#: The tolerance for "the same gestational age".
GA_TOL = 1e-6
#: (k) compares the checkpoint's constants against a refit of the same values in
#: the same order, so they agree to the last bit when they agree at all. The
#: tolerance is here only so that a re-derivation which sums in a different order
#: does not read as leakage; it is far below any difference a changed fitting set
#: could produce.
STANDARDISATION_TOL = 1e-12


class PreflightError(RuntimeError):
    """Raised when the cohort or the splits fail a pre-flight check."""


@dataclass(frozen=True)
class Check:
    name: str
    detail: str


def _cohort_counts(cohort: pd.DataFrame) -> Check:
    n_scans, n_mothers = len(cohort), cohort["mother_id"].nunique()
    if (n_scans, n_mothers) != (EXPECTED_SCANS, EXPECTED_MOTHERS):
        raise PreflightError(
            f"cohort is {n_scans} scans / {n_mothers} mothers; expected "
            f"{EXPECTED_SCANS} / {EXPECTED_MOTHERS} (EXPECTED_SCANS and "
            "EXPECTED_MOTHERS in puuma/preflight.py). Training does not start.")
    per_mother = cohort.groupby("mother_id")["category"].first()
    counts = per_mother.value_counts().to_dict()
    if counts != EXPECTED_MOTHER_CATEGORIES:
        raise PreflightError(
            f"mother-level categories {counts} against the expected {EXPECTED_MOTHER_CATEGORIES}")
    return Check("cohort counts", f"{n_scans} scans / {n_mothers} mothers, {counts}")


def _a_each_mother_in_one_fold(cohort: pd.DataFrame) -> Check:
    per_mother = cohort.groupby("mother_id")["fold"].nunique()
    bad = per_mother[per_mother != 1]
    if len(bad):
        raise PreflightError(f"(a) {len(bad)} mothers span more than one fold: {list(bad.index)[:5]}")
    return Check("(a) each mother in exactly one fold", f"{len(per_mother)} mothers")


def _b_one_row_per_mother(cohort: pd.DataFrame, iteration: int) -> Check:
    splits = split_for_iteration(cohort, iteration)
    for role in ("val", "test"):
        counts = splits[role].groupby("mother_id").size()
        if (counts != 1).any():
            raise PreflightError(
                f"(b) {role} holds more than one row for "
                f"{list(counts[counts != 1].index)[:5]}")
    return Check("(b) val and test hold one row per mother",
                 f"val {len(splits['val'])}, test {len(splits['test'])}")


def _c_train_keeps_every_scan(cohort: pd.DataFrame, iteration: int) -> Check:
    splits = split_for_iteration(cohort, iteration)
    train_mothers = set(splits["train"]["mother_id"])
    expected = cohort[cohort["mother_id"].isin(train_mothers)]
    if len(splits["train"]) != len(expected):
        raise PreflightError(
            f"(c) train holds {len(splits['train'])} rows for {len(train_mothers)} mothers "
            f"but those mothers have {len(expected)} scans")
    return Check("(c) train keeps every scan of its mothers",
                 f"{len(splits['train'])} scans / {len(train_mothers)} mothers")


def _d_every_stratum_in_every_fold(cohort: pd.DataFrame) -> Check:
    strata = set(EXPECTED_MOTHER_CATEGORIES)
    for fold in range(N_FOLDS):
        present = set(cohort[cohort["fold"] == fold]["category"])
        if present != strata:
            raise PreflightError(f"(d) fold {fold} is missing {sorted(strata - present)}")
    return Check("(d) every stratum present in every fold", f"{len(strata)} strata x {N_FOLDS} folds")


def _e_fold_column_recomputes(cohort: pd.DataFrame) -> Check:
    per_mother = (cohort.groupby("mother_id")["category"].first().sort_index().reset_index())
    recomputed = assign_folds(per_mother["mother_id"], per_mother["category"],
                              n_folds=N_FOLDS, seed=FOLD_SEED)
    stored = cohort.groupby("mother_id")["fold"].first().to_dict()
    if recomputed != stored:
        differing = [m for m in stored if recomputed.get(m) != stored[m]]
        raise PreflightError(
            f"(e) the stored fold column disagrees with a recomputation for "
            f"{len(differing)} mothers, first few {differing[:5]}")
    categories = cohort["tag_gadel"].map(lambda g: category_of(float(g)))
    if not (categories == cohort["category"]).all():
        raise PreflightError("(e) the category column does not recompute from tag_gadel")
    return Check("(e) fold column matches a recomputation", f"{len(stored)} mothers")


def _f_tie_break_reproduces(cohort: pd.DataFrame) -> Check:
    n_tied = 0
    for mother, group in cohort.groupby("mother_id"):
        earliest = group["tag_ga"].min()
        tied = group[(group["tag_ga"] - earliest).abs() < GA_TOL]
        if len(tied) > 1:
            n_tied += 1
        recomputed = choose_evaluation_scan(str(mother), tied["scan_id"].tolist(),
                                            seed=TIE_BREAK_SEED)
        stored = group[group["is_evaluation_scan"]]["scan_id"].tolist()
        if stored != [recomputed]:
            raise PreflightError(
                f"(f) {mother}: stored evaluation scan {stored} but the tie-break "
                f"recomputes to {recomputed}")
    return Check("(f) tie-break reproduces from the seed", f"{n_tied} mothers tied")


def _g_splits_match(cohort: pd.DataFrame, mapping: pd.DataFrame,
                    splits_json: Path, iteration: int) -> Check:
    if not Path(splits_json).is_file():
        raise PreflightError(
            f"(g) no splits_final.json at {splits_json}. The recorded failure mode: "
            "nnU-Net silently generates its own with KFold(shuffle=True, "
            "random_state=12345) over case identifiers — ungrouped and unstratified — "
            "and trains on it with nothing in the logs to show it.")
    splits = json.loads(Path(splits_json).read_text())
    if len(splits) != N_FOLDS:
        raise PreflightError(f"(g) splits_final.json holds {len(splits)} folds, expected {N_FOLDS}")
    merged = cohort.merge(mapping[["scan_id", "case_identifier"]], on="scan_id",
                          validate="one_to_one")
    test_fold, val_fold, train_folds = fold_roles(iteration)
    entry = splits[iteration]
    expected_train = set(merged[merged["fold"].isin(train_folds)]["case_identifier"])
    expected_val = set(merged[(merged["fold"] == val_fold)
                              & merged["is_evaluation_scan"]]["case_identifier"])
    if set(entry["train"]) != expected_train:
        raise PreflightError(
            f"(g) iteration {iteration}: train list disagrees with cohort.csv on "
            f"{len(set(entry['train']) ^ expected_train)} cases")
    if set(entry["val"]) != expected_val:
        raise PreflightError(
            f"(g) iteration {iteration}: val list disagrees with cohort.csv on "
            f"{len(set(entry['val']) ^ expected_val)} cases")
    test_cases = set(merged[merged["fold"] == test_fold]["case_identifier"])
    leaked = test_cases & (set(entry["train"]) | set(entry["val"]))
    if leaked:
        raise PreflightError(
            f"(g) iteration {iteration}: {len(leaked)} test-fold cases appear in the "
            f"split lists, first few {sorted(leaked)[:5]}")
    return Check("(g) splits_final.json matches cohort.csv",
                 f"iteration {iteration}: train {len(entry['train'])}, val {len(entry['val'])}, "
                 f"test fold {test_fold} withheld")


def run_preflight(cohort_csv: Path, mapping_csv: Path, splits_json: Path,
                  iteration: int) -> list[Check]:
    """Every gate, in order. Raises :class:`PreflightError` on the first failure."""
    cohort = pd.read_csv(cohort_csv)
    mapping = pd.read_csv(mapping_csv)
    return [
        _cohort_counts(cohort),
        _a_each_mother_in_one_fold(cohort),
        _b_one_row_per_mother(cohort, iteration),
        _c_train_keeps_every_scan(cohort, iteration),
        _d_every_stratum_in_every_fold(cohort),
        _e_fold_column_recomputes(cohort),
        _f_tie_break_reproduces(cohort),
        _g_splits_match(cohort, mapping, splits_json, iteration),
    ]


# ---------------------------------------------------------------------------
# The inference gate — (h)–(k), continuing the training gate's lettering
# ---------------------------------------------------------------------------
def _h_checkpoint_is_the_selected_one(checkpoint_path: Path) -> Check:
    """(h) The weights are `checkpoint_best.pth` and they exist.

    `checkpoint_best.pth` produces the test predictions.
    `checkpoint_final.pth` is retained as a diagnostic artefact and is never
    reported as a result. The two are one character apart in a glob, and a run
    that reached its budget has both on disk, so the wrong one is always
    available and never signals that it is wrong.
    """
    path = Path(checkpoint_path)
    if path.name != BEST_CHECKPOINT:
        role = ("the last epoch's weights regardless of quality"
                if path.name == FINAL_CHECKPOINT else
                "a periodic resume artefact" if path.name == LATEST_CHECKPOINT
                else "not one of the three checkpoint files")
        raise PreflightError(
            f"(h) test predictions may come only from {BEST_CHECKPOINT}; this is "
            f"{path.name}, which is {role}. The checkpoint rule makes the selected checkpoint the "
            "result and everything else a diagnostic.")
    if not path.is_file():
        raise PreflightError(f"(h) no {BEST_CHECKPOINT} at {path}")
    return Check("(h) weights are the selected checkpoint", str(path))


def _i_test_mothers_match_cohort(cohort: pd.DataFrame, mapping: pd.DataFrame,
                                 manifest: dict, iteration: int) -> Check:
    """(i) The fold's test mothers are the ones this run withheld.

    The run recorded its own test set in `run_manifest.json` when it started. If
    `cohort.csv` has moved since — a rebuilt cohort, a changed tie-break, a
    different iteration passed on the command line — the two disagree and the
    predictions would be attributed to the wrong split.
    """
    recorded_iteration = manifest.get("iteration")
    if recorded_iteration != iteration:
        raise PreflightError(
            f"(i) the run recorded iteration {recorded_iteration} and this invocation "
            f"asks for {iteration}; the test fold differs between them")
    merged = cohort.merge(mapping[["scan_id", "case_identifier"]], on="scan_id",
                          validate="one_to_one")
    expected = {(r.mother_id, r.case_identifier)
                for r in split_for_iteration(merged, iteration)["test"].itertuples()}
    recorded = {(e["mother_id"], e["case_identifier"])
                for e in manifest.get("evaluation_scan_selection", {}).get("test", [])}
    if recorded != expected:
        raise PreflightError(
            f"(i) the run's recorded test set and cohort.csv disagree on "
            f"{len(recorded ^ expected)} (mother, scan) pairs; e.g. "
            f"{sorted(recorded ^ expected)[:3]}")
    if not expected:
        raise PreflightError(f"(i) iteration {iteration} has no test mothers")
    return Check("(i) test mothers match cohort.csv",
                 f"{len(expected)} mothers, one scan each")


def _j_no_test_mother_in_train_or_val(cohort: pd.DataFrame, mapping: pd.DataFrame,
                                      splits_json: Path, manifest: dict,
                                      iteration: int) -> Check:
    """(j) No test mother appears in that fold's training or validation lists.

    Checked at the level of the mother, not the case. Assertion (g)
    already compares the case lists; this asks the question leakage is actually
    about — whether any scan of a woman who is being tested was trained on. A
    second scan of a test mother in the training list is a different case
    identifier and would pass every case-level check there is.
    """
    merged = cohort.merge(mapping[["scan_id", "case_identifier"]], on="scan_id",
                          validate="one_to_one")
    test_fold, _, _ = fold_roles(iteration)
    test_mothers = set(merged[merged["fold"] == test_fold]["mother_id"])
    by_case = dict(zip(merged["case_identifier"], merged["mother_id"]))

    splits = json.loads(Path(splits_json).read_text())
    listed = set(splits[iteration]["train"]) | set(splits[iteration]["val"])
    listed |= {s["case_identifier"] for s in manifest.get("training_scans", [])}
    leaked = {c for c in listed if by_case.get(c) in test_mothers}
    if leaked:
        raise PreflightError(
            f"(j) {len(leaked)} scans of test-fold mothers appear in the training or "
            f"validation lists: {sorted(leaked)[:5]}. Every number in the test file "
            "would be reported on data the model was fitted to.")
    return Check("(j) no test mother in train or val",
                 f"{len(test_mothers)} test mothers against {len(listed)} listed scans")


def _k_standardisation_is_the_folds(cohort_csv: Path, mapping_csv: Path,
                                    cohort: pd.DataFrame, mapping: pd.DataFrame,
                                    manifest: dict, standardisation: dict,
                                    iteration: int) -> Check:
    """(k) The constants that invert the prediction are that fold's training constants.

    This is the leakage a later reader could not detect from the output: the
    weeks prediction is `z x std + mean`; fit those two numbers on the whole
    cohort and every row of `test_predictions.csv` is contaminated, while the
    file itself looks exactly the same. They are fixed to the training rows of
    the fold.

    Two comparisons, because either alone is weak. The set of cases the
    constants were fitted on is compared with the fold's training cases, which is
    the property that matters; and the numbers themselves are compared with a
    refit, which catches a manifest that records one thing and a checkpoint that
    carries another.
    """
    if not standardisation:
        raise PreflightError(
            "(k) the checkpoint carries no standardisation constants, so the weeks "
            "prediction cannot be shown to invert with the fold's own")
    merged = cohort.merge(mapping[["scan_id", "case_identifier"]], on="scan_id",
                          validate="one_to_one")
    train_keys = sorted(split_for_iteration(merged, iteration)["train"].case_identifier)
    refit = ScalarTable.from_cohort(cohort_csv, mapping_csv).fit_standardiser(train_keys)

    recorded = manifest.get("standardisation") or {}
    fitted_on = set(recorded.get("fitted_on") or [])
    if fitted_on and fitted_on != set(train_keys):
        extra, missing = fitted_on - set(train_keys), set(train_keys) - fitted_on
        raise PreflightError(
            f"(k) the constants were fitted on {len(fitted_on)} cases and this fold has "
            f"{len(train_keys)} training cases; {len(extra)} do not belong to it "
            f"(e.g. {sorted(extra)[:3]}) and {len(missing)} are missing. The standardisation "
            "constants are computed from the training rows of the fold and from nothing else.")

    for name, source in (("means", standardisation.get("means")),
                         ("stds", standardisation.get("stds"))):
        theirs, ours = dict(source or {}), getattr(refit, name)
        if set(theirs) != set(ours):
            raise PreflightError(f"(k) the checkpoint's {name} cover {sorted(theirs)}, "
                                 f"the refit covers {sorted(ours)}")
        for key in ours:
            if abs(float(theirs[key]) - float(ours[key])) > STANDARDISATION_TOL:
                raise PreflightError(
                    f"(k) {name}[{key}]: the checkpoint carries {theirs[key]!r} and a "
                    f"refit on this fold's {len(train_keys)} training rows gives "
                    f"{ours[key]!r}. The constants inverting the weeks prediction are "
                    "not this fold's, and no column in the output would show it.")
    return Check("(k) standardisation is this fold's training constants",
                 f"{len(train_keys)} training rows, "
                 f"mean GA_del {refit.means['ga_del']:.4f}")


def run_inference_preflight(cohort_csv: Path, mapping_csv: Path, splits_json: Path,
                            manifest_json: Path, checkpoint_path: Path,
                            standardisation: dict, iteration: int) -> list[Check]:
    """The inference gate, in order. Raises :class:`PreflightError` on the first failure.

    Runs before a single subject is predicted. (h) is checked first and again
    here: the script that loads the weights must not be the only thing standing
    between `checkpoint_final.pth` and a reported result.
    """
    manifest_json = Path(manifest_json)
    if not manifest_json.is_file():
        raise PreflightError(
            f"no run_manifest.json at {manifest_json}. The run is required to record "
            "which scan was selected per mother for validation and test, and (i), (j) "
            "and (k) are all checked against it.")
    manifest = json.loads(manifest_json.read_text())
    cohort = pd.read_csv(cohort_csv)
    mapping = pd.read_csv(mapping_csv)
    return [
        _h_checkpoint_is_the_selected_one(checkpoint_path),
        _i_test_mothers_match_cohort(cohort, mapping, manifest, iteration),
        _j_no_test_mother_in_train_or_val(cohort, mapping, splits_json, manifest, iteration),
        _k_standardisation_is_the_folds(cohort_csv, mapping_csv, cohort, mapping,
                                        manifest, standardisation, iteration),
    ]
