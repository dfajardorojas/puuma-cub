# PUUMA-CuB — fetal T2* MRI gestational-age-at-birth regression

Code for a multi-task 3D network that predicts gestational age at birth from
fetal T2* MRI: a U-Mamba global branch over the whole field of view, a 20³
placental-patch branch, auxiliary placenta/brain segmentation, and a linear
blend over the two image predictions, gestational age at scan and cervical
length. Trained and evaluated under five-fold, mother-disjoint cyclical
cross-validation on the paper's cohort of 190 scans from 120 pregnancies.

> Paper reference: [placeholder — citation to be added on publication].

## What is NOT included, and why

- **Data.** The MRI volumes, segmentation masks and all participant-derived
  tables (the cohort CSV, the preprocessed nnU-Net datasets) are not released.
- **Trained weights.** The per-fold checkpoints are not part of this
  repository.
- **The cohort assembly resolver.** The paper's cohort was assembled from
  institutional metadata by a subject-identifier resolver that is specific to
  that archive; `scripts/build_cohort.py` here replaces it with a user-supplied
  manifest and applies the same downstream rules through the same library code.
- **Figure and table rendering.** The scripts that typeset the paper's tables
  and figures read the completed campaign tree and are not released;
  `scripts/evaluate.py` carries the metric computation verbatim.
- **The campaign launcher.** The multi-day serial launcher is specific to the
  original tree's bookkeeping. The per-fold commands it executed are below.

## Installation

1. `conda env create -f environment.yml` (environment name `MambaTest2`).
2. Install U-Mamba — which vendors nnU-Net v2.1.1 — in editable mode at the
   pinned commit:

   ```
   git clone https://github.com/bowang-lab/U-Mamba
   cd U-Mamba && git checkout 28459e33ca03769800dd35e23c6e62491d1925b5
   pip install -e umamba/
   ```

3. **The paths shim.** nnU-Net as vendored hardcodes its three working roots
   (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) to directories
   inside its own clone; the environment variables are commented out, so
   setting them does nothing. `src/puuma/nnunet_paths.py` rebinds all three
   under this repository before anything else in `nnunetv2` is imported, and
   every entry point here imports it first. Two consequences:
   - run the pipeline only through the scripts in `scripts/`;
   - **never invoke nnU-Net's own console scripts** (`nnUNetv2_train`,
     `nnUNetv2_plan_and_preprocess`, `nnUNetv2_predict`, …) — they do not
     import the shim and would write into the U-Mamba clone.

## Pipeline

1. **Manifest → cohort.** Prepare one CSV with columns
   `scan_id, mother_id, image_path, placenta_mask_path, brain_mask_path,
   ga_scan, ga_del, cl` (optional `study`; gestational ages in weeks; `cl` in
   mm, empty where unmeasured — 0.0 is a real measurement). Then:

   ```
   python scripts/build_cohort.py --manifest my_manifest.csv
   ```

   Defaults reproduce the paper's configuration: 5 folds,
   `StratifiedKFold(shuffle=True, random_state=1)` stratified on the delivery
   category, tie-break seed 20260818. `cohort/` is git-ignored by design, so
   the built `cohort.csv` stays local and participant tables are never
   committed.

2. **Cohort → nnU-Net dataset.**

   ```
   python scripts/build_nnunet_dataset.py
   ```

3. **Training, one fold at a time.** Config names:
   `cl_plus_image` (the reference model, PUUMA-CuB), `image_only`,
   `no_brain_seg`, `global_only`, `unet`, `no_sampler`, `no_skips`.
   For config `<c>` and fold `<i>` in 0..4, the campaign's exact command:

   ```
   python scripts/run_one.py --results-root campaign --config <c> \
       --iteration <i> --total-epochs 250 --plot-every 10
   ```

4. **Test predictions for the fold, from the selected checkpoint:**

   ```
   python scripts/predict_test_fold.py --results-root campaign \
       --tag <c>_iteration_<i>
   ```

5. **Reference rows** (constant / CL-only / CL+GA linear baselines, no GPU):

   ```
   python scripts/run_reference_rows.py
   ```

6. **Metrics** (per fold and pooled; test, or validation at the selected
   epoch):

   ```
   python scripts/evaluate.py --runs-root outputs/runs --config cl_plus_image
   ```

## The checkpoint rule

During training, validation runs every epoch and an EMA (alpha 0.1) of the
standardised validation MSE is tracked; `checkpoint_best.pth` is written
whenever that EMA reaches a new minimum. `predict_test_fold.py` loads
`checkpoint_best.pth` and nothing else — `checkpoint_final.pth` (the last
epoch) is a different model and is never used for test predictions.

## The shipped nnUNetPlans.json

The two files under `data/nnUNet_preprocessed/*/nnUNetPlans.json` are the
architecture configuration the paper trained (widths, strides, patch
[80, 160, 192], spacing [2.0, 1.8605, 1.8605]). `build_nnunet_dataset.py`
regenerates plans for new data, and regenerated plans can differ with the
dataset; **to reproduce the paper's network, use the shipped plans.**

## Divergences from the code that produced the paper

This repository is a staging copy of the working tree that produced the paper,
with redactions (absolute paths, participant identifiers), a generic cohort
builder replacing the institutional one, an extracted `evaluate.py`, and
licence headers added. `DIVERGENCE_LOG.md` lists every differing line against
the source tree, every moved code block with its original line range, and the
items that remain deliberately unresolved.
