"""One fold, end to end.

    python scripts/run_one.py --results-root smoke --iteration 0 --total-epochs 250

**`--results-root` has no default, on purpose.** It is the shim's mandatory typed
argument: a process cannot silently pick between the campaign and
the smoke destinations, which is what the separation depends on. Campaign runs
are the author's to launch.

The first thing it does is the pre-flight gate: the leakage assertions (a)–(g) and the
cohort figures. They are required before training, and a test suite is
not the same as a check inside the job.

The test-time inference is `scripts/predict_test_fold.py`, which reads this run's
`checkpoint_best.pth` and `run_manifest.json` afterwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The results root is chosen from the command line BEFORE anything from nnunetv2
# is imported: binding precedes trainer construction, and a single
# stray `from nnunetv2...` above this point captures the shipped paths.
from puuma.nnunet_paths import ResultsRoot, bind_nnunet_paths  # noqa: E402

_ROOTS = {"campaign": ResultsRoot.CAMPAIGN, "smoke": ResultsRoot.SMOKE}


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, choices=sorted(_ROOTS),
                    help="CAMPAIGN or SMOKE. No default: the campaign/smoke separation depends on a "
                         "process being unable to pick silently.")
    ap.add_argument("--config", type=str, default="cl_plus_image",
                    help="configuration row, by name; see configs/. A row that is "
                         "declared but not runnable refuses here, before anything trains.")
    ap.add_argument("--iteration", type=int, required=True, help="the fold iteration t")
    ap.add_argument("--total-epochs", type=int, required=True,
                    help="the full budget; PolyLR is defined over it")
    ap.add_argument("--epochs", type=int, default=None,
                    help="stop after this many (for smoke runs); defaults to the budget")
    ap.add_argument("--initial-lr", type=float, default=None,
                    help="the BACKBONE initial learning rate, the value PolyLR "
                         "anneals from. Default 1e-2. A diagnostic flag, "
                         "like --no-amp and --rotation-degrees: the adopted rate is "
                         "unchanged and this does not alter it.")
    ap.add_argument("--head-lr", type=float, default=None,
                    help="the HEAD-GROUP initial rate, default 1e-3. Left alone when "
                         "--initial-lr moves, because 1e-3 was measured for the head's "
                         "own dynamics and not as a ratio; pass this to hold the 10:1 "
                         "ratio instead (e.g. --initial-lr 4e-3 --head-lr 4e-4).")
    ap.add_argument("--amp", action="store_true",
                    help="turn AMP back ON. **fp32 is the adopted default**: "
                         "all three completed folds ran fp32, and AMP's fp16 ceiling "
                         "is what turned the Mamba growth into a NaN. Costs ~17% step "
                         "time and is kept as a flag, not a default.")
    ap.add_argument("--no-amp", action="store_true",
                    help="accepted and redundant — fp32 is now the default. Kept so "
                         "the recorded diagnostic commands still run.")
    ap.add_argument("--mamba-ceiling", type=float, default=None,
                    help="detector 6's ceiling. Default 549.3, the geometric "
                         "midpoint of the measured normal maximum (16.4) and the "
                         "lowest divergent one (1.84e4); detector 3 is unaffected.")
    ap.add_argument("--rotation-degrees", type=float, default=None,
                    help="augmentation half-range in degrees (the magnitude question is open; "
                         "nnU-Net's own value was 30, the reference notebook's 2.865)")
    ap.add_argument("--scale-range", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="augmentation scale range (nnU-Net hardcodes 0.7 1.4)")
    ap.add_argument("--steps-per-epoch", type=int, default=250,
                    help="an nnU-Net epoch is exactly 250 iterations")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--tag", type=str, default=None, help="results subdirectory")
    ap.add_argument("--resume-from", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="augmentation workers; defaults to puuma.runtime's value. "
                         "The count affects throughput only, not results.")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not hold the training cases in RAM (see puuma.nnunet_cache)")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run the pre-flight gate for this config and iteration, then exit 0. "
                         "Creates no run directory. The launcher uses this before a "
                         "config's FIRST fold, so a failing gate costs seconds rather "
                         "than being discovered fourteen hours in.")
    ap.add_argument("--plot-every", type=int, default=10,
                    help="rewrite the training curve every n epochs. The per-epoch "
                         "JSONL is always on and free; replotting is not, which is why "
                         "this one has a flag. 0 disables it until the end of the run.")
    ap.add_argument("--save-every", type=int, default=None,
                    help="periodic checkpoint cadence; defaults to 50")
    return ap.parse_args()


ARGS = _parse()
PATHS = bind_nnunet_paths(_ROOTS[ARGS.results_root])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import (  # noqa: E402
    LimitedLenWrapper,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size  # noqa: E402
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter  # noqa: E402

from puuma.checkpoint import (  # noqa: E402
    BEST_CHECKPOINT, FINAL_CHECKPOINT, LATEST_CHECKPOINT, SAVE_EVERY, CheckpointWriter,
    load_checkpoint,
)
from puuma.cohort import split_for_iteration  # noqa: E402
from puuma.config import balanced_sampling_probabilities, load_row  # noqa: E402
from puuma.dataloader import LOCAL_PATCH_SIZE, PlacentaPatchTransform, ScalarTable  # noqa: E402
from puuma.loss import PuumaLoss, W_REG, W_SEG  # noqa: E402
from puuma.model import build_row_model  # noqa: E402
from puuma.nnunet_cache import build_dataset  # noqa: E402
from puuma.nnunet_dataloader import PuumaDataLoader3D  # noqa: E402
from puuma.plans import CONFIGURATION, load_plans_with_puuma_configuration  # noqa: E402
from puuma.preflight import run_preflight  # noqa: E402
from puuma.record import EpochRecord, RunRecord  # noqa: E402
from puuma.runtime import RuntimeConfig, physical_cores  # noqa: E402
from puuma.selection import (  # noqa: E402
    BlendCollapseMonitor, CheckpointSelector, argmin_epoch_ungated,
    assert_detector_2_not_triggered,
    detector_1_raw_mse_at_saved_checkpoint, detector_2_preds_std_all_zero, update_ema,
)
from puuma.training import (  # noqa: E402
    BACKBONE_LR, HEAD_LR, INITIAL_LR, MAMBA_ACTIVATION_CEILING, GateFailureMonitor,
    MambaActivationMonitor,
    assert_remove_label_transform,
    build_optimiser, train_step,
)
from puuma.validation import (  # noqa: E402
    assert_validation_file_readable, assert_validation_within_time_budget,
    mae_weeks, standardised_mse, validate_fold, validation_objective,
)

#: The dataset is the ROW's, not a constant: `no_brain_seg` is a second
#: dataset built from the same volumes with the placenta mask only, so the row is
#: the same model on a placenta-only target rather than a runtime special case.
MIRROR_AXES = (0, 1, 2)
#: Execution knobs — batch size, workers, prefetch, pinning, the RAM cache — all
#: from `puuma.runtime` so there is one copy of each. `BATCH_SIZE` was written out
#: here and in `smoke_validate.py`, which is two places to forget.
RUNTIME = RuntimeConfig()
BATCH_SIZE = RUNTIME.batch_size
DIV = "=" * 78


def banner(t: str) -> None:
    print(f"\n{DIV}\n{t}\n{DIV}", flush=True)


def main() -> None:
    args = ARGS
    # Diagnostic rate overrides. The rates live in `puuma.training`, not `RuntimeConfig`: that module
    # is deliberately torch-free and runs at import (it captures the parent CPU
    # affinity before any fork), so importing `training` there would drag `nnunetv2`
    # in ahead of the paths shim — exactly the forbidden import ordering.
    initial_lr = BACKBONE_LR if args.initial_lr is None else float(args.initial_lr)
    head_lr = HEAD_LR if args.head_lr is None else float(args.head_lr)
    # The augmentation magnitudes are provisional, so they are settable per invocation
    # without editing anything. Rebound before the first use of RUNTIME below.
    global RUNTIME
    if args.rotation_degrees is not None or args.scale_range is not None:
        RUNTIME = dataclasses.replace(
            RUNTIME,
            rotation_degrees=(RUNTIME.rotation_degrees if args.rotation_degrees is None
                              else float(args.rotation_degrees)),
            scale_range=(tuple(RUNTIME.scale_range) if args.scale_range is None
                         else tuple(args.scale_range)))
    epochs = args.epochs or args.total_epochs
    save_every = args.save_every or SAVE_EVERY
    row = load_row(args.config)
    row.require_implemented()          # before the pre-flight, before any GPU work
    tag = args.tag or f"{row.name}_iteration_{args.iteration}"
    out_dir = Path(PATHS.results) / tag
    pre = Path(PATHS.preprocessed) / row.dataset

    banner("PRE-FLIGHT GATE — nothing trains if any of this fails")
    print(f"  results root   {args.results_root.upper()}  ->  {PATHS.results}")
    print(f"  config row     {row.describe()}")
    checks = run_preflight(REPO / "cohort" / "cohort.csv",
                           REPO / "cohort" / "case_id_mapping.csv",
                           pre / "splits_final.json", args.iteration)
    for c in checks:
        print(f"  PASS  {c.name:46s} {c.detail}")
    if args.preflight_only:
        # Before out_dir is created, so this leaves nothing behind.
        print(f"\n  PRE-FLIGHT ONLY: gate passed for {row.name} iteration "
              f"{args.iteration}; nothing trained, no run directory created.")
        return

    banner("SETUP")
    plans = load_plans_with_puuma_configuration(pre / "nnUNetPlans.json")
    dataset_json = json.loads((pre / "dataset.json").read_text())
    pm = PlansManager(plans)
    cm = pm.get_configuration(CONFIGURATION)
    lm = pm.get_label_manager(dataset_json)
    patch = np.array(cm.patch_size)
    workers = RUNTIME.workers if args.num_workers is None else args.num_workers
    RUNTIME.apply_sharing_strategy()

    cohort = pd.read_csv(REPO / "cohort" / "cohort.csv").merge(
        pd.read_csv(REPO / "cohort" / "case_id_mapping.csv")[["scan_id", "case_identifier"]],
        on="scan_id", validate="one_to_one")
    splits = split_for_iteration(cohort, args.iteration)
    train_keys = sorted(splits["train"].case_identifier)
    table = ScalarTable.from_cohort(REPO / "cohort" / "cohort.csv",
                                    REPO / "cohort" / "case_id_mapping.csv")
    standardiser = table.fit_standardiser(train_keys)
    val_rows = [{"case_identifier": r.case_identifier, "mother_id": r.mother_id,
                 "ga_scan": float(r.tag_ga), "cl": float(r.cl_value),
                 "ga_del": float(r.tag_gadel)} for r in splits["val"].itertuples()]
    print(f"  iteration {args.iteration}: train {len(train_keys)} scans, "
          f"val {len(val_rows)} mothers, test {len(splits['test'])}")
    print(f"  epochs {epochs} of a {args.total_epochs}-epoch budget, "
          f"{args.steps_per_epoch} steps each, batch {BATCH_SIZE}")
    print(f"  per-epoch record appended to {RunRecord.EPOCH_LOG} as each epoch ends; "
          f"curve rewritten every {args.plot_every or 'never until the end'} epochs")
    print(f"  runtime: {RUNTIME.describe()}")
    print(f"  augmentation workers: {workers} (machine has {physical_cores()} physical "
          f"cores; get_allowed_n_proc_DA would say {get_allowed_n_proc_DA()}, which is a "
          "hardcoded DKFZ lookup table, not a measurement of this host)")

    # The augmentation magnitudes, from the runtime config. nnU-Net
    # leaves `rotation_for_DA` open as a parameter; the scale it hardcodes, so that
    # one is applied to the built transform below.
    rot = RUNTIME.rotation_for_DA()
    initial_patch = get_patch_size(patch, *rot.values(), tuple(RUNTIME.scale_range))
    ds_scales = list(list(i) for i in 1/np.cumprod(np.vstack(cm.pool_op_kernel_sizes), axis=0))[:-1]
    # Built BEFORE the augmenter, so its forked workers inherit the cache
    # copy-on-write rather than each decompressing the archives themselves.
    dataset = build_dataset(pre / cm.data_identifier, train_keys,
                            cache=RUNTIME.cache_cases_in_ram and not args.no_cache,
                            verbose=True)
    # The category-balanced sampler is per-case probabilities, never row
    # replication — so it cannot change training volume, which is fixed at 250
    # iterations. `no_sampler` is the row that passes None.
    probabilities = None
    if row.balanced_sampler:
        by_case = dict(zip(cohort.case_identifier, cohort.category))
        probabilities = balanced_sampling_probabilities([by_case[k] for k in train_keys])
    print(f"  sampler: {'category-balanced' if probabilities else 'none (uniform)'}")
    dl = PuumaDataLoader3D(dataset, BATCH_SIZE, initial_patch, patch, lm,
                           oversample_foreground_percent=0.33,
                           sampling_probabilities=probabilities,
                           pad_sides=None, scalar_table=table, standardiser=standardiser)
    transforms = nnUNetTrainer.get_training_transforms(
        patch, rot, ds_scales, MIRROR_AXES, False, order_resampling_data=3,
        order_resampling_seg=1, use_mask_for_norm=cm.use_mask_for_norm, is_cascaded=False,
        foreground_labels=lm.foreground_labels, regions=None, ignore_label=None)
    # nnU-Net hardcodes `scale=(0.7, 1.4)` INSIDE get_training_transforms — it is not
    # a parameter — so the only way to set it without forking the builder is on the
    # constructed transform. Composition, order and every probability are untouched.
    spatial = next(t for t in transforms.transforms
                   if type(t).__name__ == "SpatialTransform")
    spatial.scale = tuple(RUNTIME.scale_range)
    print(f"  augmentation: rotation ±{RUNTIME.rotation_degrees:g}° on x/y/z, "
          f"scale {tuple(RUNTIME.scale_range)}, mirror axes {MIRROR_AXES}; "
          f"initial patch {list(initial_patch)} against final {list(patch)}")
    assert_remove_label_transform(transforms)          # detector 4
    transforms.transforms.append(PlacentaPatchTransform())
    if workers == 0:
        generator = SingleThreadedAugmenter(dl, transforms)
    else:
        generator = LimitedLenWrapper(args.steps_per_epoch, data_loader=dl,
                                      transform=transforms, num_processes=workers,
                                      num_cached=RUNTIME.prefetch_depth, seeds=None,
                                      pin_memory=RUNTIME.pin_memory, wait_time=0.02)

    device = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build_row_model(row, pm, dataset_json, cm, LOCAL_PATCH_SIZE, device=device)
    report = model.parameter_report()
    loss_fn = PuumaLoss(lm, cm.batch_dice, ds_scales).to(device)
    optimiser, scheduler = build_optimiser(model, num_epochs=args.total_epochs,
                                           initial_lr=initial_lr, head_lr=head_lr)
    print(f"  optimiser: backbone lr {optimiser.param_groups[0]['lr']:.3e}, head lr "
          f"{optimiser.param_groups[1]['lr']:.3e} (ratio "
          f"{optimiser.param_groups[0]['lr'] / optimiser.param_groups[1]['lr']:.1f}:1), "
          f"PolyLR over {args.total_epochs} epochs")
    # fp32 when asked. `train_step` branches on `scaler is None` — it
    # disables autocast, skips the scale/unscale and steps the optimiser directly —
    # and `save_checkpoint`/`load_checkpoint` already handle None. Detector 3 is
    # untouched and is the backstop once fp16's ceiling is out of the picture.
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    print(f"  precision: {'fp32 (adopted default)' if scaler is None else 'AMP fp16 (--amp)'}")
    print(f"  model {report.total:,} parameters; widths {model.widths()}")

    ceiling = (MAMBA_ACTIVATION_CEILING if args.mamba_ceiling is None
               else float(args.mamba_ceiling))
    init_args = {"configuration": CONFIGURATION, "config_row": row.name,
                 "row": row.describe(), "iteration": args.iteration,
                 "seed": args.seed, "total_epochs": args.total_epochs,
                 "steps_per_epoch": args.steps_per_epoch, "batch_size": BATCH_SIZE,
                 "local_patch": list(LOCAL_PATCH_SIZE), "results_root": args.results_root,
                 "runtime": RUNTIME.describe(),
                 # The curve labels the blend series from this, so the panel names
                 # the inputs rather than numbering them.
                 "blend_inputs": list(row.blend_inputs),
                 # Recorded so a reader cannot mistake a diagnostic run
                 # for a campaign one. AMP on and ceiling 1000 are the campaign values.
                 "amp": bool(args.amp),
                 "precision": "amp_fp16" if args.amp else "fp32",
                 "mamba_activation_ceiling": ceiling,
                 # PolyLR is defined by the budget AND these two rates, so
                 # a resumed run that changes either gets a different schedule.
                 "initial_lr": initial_lr, "head_lr": head_lr,
                 "loss_weights": {"w_reg": W_REG, "w_seg": W_SEG}}
    record = RunRecord(output_dir=out_dir, config=init_args)
    selector = CheckpointSelector()
    writer = CheckpointWriter(out_dir, total_epochs=args.total_epochs, save_every=save_every)
    ema, start_epoch = None, 0
    gate_monitor = GateFailureMonitor()          # detector 5
    mamba_monitor = MambaActivationMonitor(ceiling=ceiling)   # detector 6
    print(f"  detector 6 ceiling {ceiling:g}; detector 3 (non-finite loss) armed, "
          "unchanged")
    blend_monitor = BlendCollapseMonitor(row.blend_inputs)   # detector 7

    if args.resume_from:
        meta = load_checkpoint(Path(args.resume_from), model=model, optimiser=optimiser,
                               scaler=scaler, scheduler=scheduler, record=record,
                               selector=selector)
        start_epoch, ema = meta["next_epoch"], meta["ema"]
        if meta["init_args"].get("total_epochs") not in (None, args.total_epochs):
            raise SystemExit(
                f"the checkpoint was written under a "
                f"{meta['init_args']['total_epochs']}-epoch budget and this invocation "
                f"uses {args.total_epochs}; PolyLR is defined over it")
        # The same hazard, for the same reason: PolyLR scales each group from its own
        # initial rate over the budget, so resuming at a different one silently trains
        # the rest of the run on a different schedule. A mismatch on the budget
        # already raises; these raise identically rather than being caught later by a
        # curve nobody can explain.
        for field, now in (("initial_lr", initial_lr), ("head_lr", head_lr)):
            was = meta["init_args"].get(field)
            if was is not None and float(was) != float(now):
                raise SystemExit(
                    f"the checkpoint was written with {field}={was:g} and this "
                    f"invocation uses {now:g}. PolyLR anneals from the initial rate "
                    "over the budget, so resuming at a different one changes "
                    "the schedule for every remaining epoch. Pass "
                    f"--{field.replace('_', '-')} {was:g} to continue this run.")
        banner(f"RESUMED from {args.resume_from} at epoch {start_epoch}")
        print(f"  history carried: {len(record.epochs)} epochs; EMA {ema}; "
              f"best {selector.best_ema} at epoch {selector.saved_checkpoint_epoch}")

    # The run record requires the CL value used and which scan it came from, and which scan was
    # selected per mother for validation and test. They live in cohort.csv, but the record
    # asks the RUN to write them, so the run is self-describing years later.
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed, "iteration": args.iteration,
        "results_root": args.results_root, "configuration": CONFIGURATION,
        "config_row": row.name, "dataset": row.dataset,
        "entry_point": "scripts/run_one.py",
        "inference_allowed_mirroring_axes": list(MIRROR_AXES),
        "standardisation": standardiser.as_dict(),
        "evaluation_scan_selection": {
            role: [{"mother_id": r.mother_id, "case_identifier": r.case_identifier,
                    "cl_value": float(r.cl_value), "cl_from_scan": r.scan_id,
                    "ga_scan": float(r.tag_ga), "ga_del": float(r.tag_gadel)}
                   for r in splits[role].itertuples()]
            for role in ("val", "test")},
        "training_scans": [{"case_identifier": r.case_identifier, "mother_id": r.mother_id,
                            "cl_value": float(r.cl_value), "cl_from_scan": r.scan_id}
                           for r in splits["train"].itertuples()],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    ck = dict(model=model, optimiser=optimiser, scaler=scaler, scheduler=scheduler,
              record=record, standardisation={"means": standardiser.means,
                                              "stds": standardiser.stds},
              init_args=init_args, entry_point="scripts/run_one.py",
              inference_allowed_mirroring_axes=MIRROR_AXES)

    banner(f"TRAINING epochs {start_epoch}..{epochs - 1}")
    step_times, wait_times, val_times = [], [], []
    step = 0
    try:
        for epoch in range(start_epoch, epochs):
            t_epoch = time.perf_counter()
            model.train()
            tot = reg = seg = 0.0
            gate_failed, gate_fills = 0, []
            gnorms = {"regression": float("nan"), "segmentation": float("nan")}
            for i in range(args.steps_per_epoch):
                step += 1
                t0 = time.perf_counter()
                batch = next(generator)
                wait = time.perf_counter() - t0
                rec = train_step(model, loss_fn, optimiser, scaler, batch, device, step,
                                 seconds_data=wait, collect_grad_norms=(i == 0),
                                 gate_monitor=gate_monitor)
                wait_times.append(wait)
                step_times.append(wait + rec.seconds_compute)
                tot += rec.total; reg += rec.regression; seg += rec.segmentation
                gate_failed += rec.gate_failed; gate_fills.extend(rec.gate_fills)
                # Detector 6, every step, on the magnitudes the forward already
                # recorded. Free: the tracker computes them for the activation penalty,
                # which is off, and this is a comparison.
                mamba_monitor.observe(getattr(model, "mamba", None) and model.mamba.magnitudes
                                      or {}, step)
                if rec.grad_norms:
                    gnorms = rec.grad_norms
            n = args.steps_per_epoch
            scheduler.step(epoch)

            t_val = time.perf_counter()
            preds = validate_fold(model, val_rows, pre / cm.data_identifier, standardiser,
                                  device, LOCAL_PATCH_SIZE, list(patch), compute_dice=True)
            objective = validation_objective(model, loss_fn, val_rows, pre / cm.data_identifier,
                                             standardiser, device, LOCAL_PATCH_SIZE,
                                             list(patch), ds_scales)
            val_seconds = time.perf_counter() - t_val
            val_times.append(val_seconds)

            # Detector 2, per epoch and as a hard stop. Placed here, on this
            # epoch's own validation output, because evaluating it after the
            # training loop meant it never ran on the fold it was written for.
            assert_detector_2_not_triggered([p.preds_std for p in preds], epoch)
            # Detector 7, on this epoch's own blend weights, before the metric
            # and before any checkpoint can be selected — same placement and the
            # same reason as detector 2.
            blend = model.blend.weight.detach().float().flatten().tolist()
            image_share = blend_monitor.observe(blend, epoch)
            # Detector 7 warns rather than stops (see BLEND_COLLAPSE_IS_A_STOP).
            # Printed here so it lands in the run log next to the epoch it describes.
            if blend_monitor.warnings and blend_monitor.consecutive == \
                    blend_monitor.window_epochs:
                print(f"  WARNING {blend_monitor.warnings[-1]}", flush=True)
            mse = standardised_mse(preds)
            ema = update_ema(ema, mse)
            improved = selector.consider(epoch, ema)
            d1 = float(np.nanmean([p.dice[1] for p in preds]))
            d2 = float(np.nanmean([p.dice[2] for p in preds]))

            record.add_validation(epoch, preds)
            record.add_epoch(EpochRecord(
                epoch=epoch, train_loss_total=tot/n, train_loss_regression=reg/n,
                train_loss_segmentation=seg/n,
                val_weighted_regression_term=W_REG*mse,
                val_objective_total=objective["total"],
                val_objective_regression=objective["regression"],
                val_objective_segmentation=objective["segmentation"],
                val_standardised_mse=mse, val_standardised_mse_ema=ema,
                val_mae_weeks=mae_weeks(preds),
                grad_norm_regression=gnorms["regression"],
                grad_norm_segmentation=gnorms["segmentation"],
                learning_rate=float(optimiser.param_groups[0]["lr"]),
                epoch_seconds=time.perf_counter() - t_epoch, validation_seconds=val_seconds,
                dice_channel_1=d1, dice_channel_2=d2,
                n_gate_failed_mothers=sum(1 for p in preds if p.gate_failed),
                gate_failed_items=gate_failed,
                gate_failed_rate=gate_failed / (n * BATCH_SIZE),
                gate_failed_fill_min=float(np.min(gate_fills)) if gate_fills else float("nan"),
                gate_failed_fill_median=float(np.median(gate_fills)) if gate_fills else float("nan"),
                blend_weights=blend, blend_image_share=image_share,
                **dict(zip(("mamba_activation_max", "mamba_activation_layer"),
                           mamba_monitor.take_epoch_max()))))

            # The write is DERIVED from the selector's verdict, never re-evaluated.
            written = writer.on_epoch_end(epoch, selector=selector, improved=improved,
                                          ema=ema, **ck)
            record.write(saved_checkpoint_epoch=selector.saved_checkpoint_epoch,
                         argmin_epoch_ungated=argmin_epoch_ungated(record.raw_metric_trace()),
                         standardisation=standardiser.as_dict(), parameters=report.as_dict())
            # The curve, periodically: a run interrupted at hour two otherwise
            # leaves no picture at all, because plot() ran only at the end.
            if args.plot_every and (epoch + 1) % args.plot_every == 0:
                record.plot(saved_checkpoint_epoch=selector.saved_checkpoint_epoch)
            print(f"  epoch {epoch}  train {tot/n:7.4f}  val obj {objective['total']:7.4f}  "
                  f"MSE_std {mse:7.4f}  EMA {ema:7.4f}  MAE {mae_weeks(preds):5.2f} wk  "
                  f"Dice {d1:.3f}/{d2:.3f}  {val_seconds:5.1f} s val"
                  f"{'  wrote ' + ', '.join(written) if written else ''}", flush=True)

        if epochs >= args.total_epochs:
            # Only a run that reached its budget is finished. An invocation that
            # stops early leaves checkpoint_latest.pth on disk — that file's
            # presence is precisely the signal that a resume is needed.
            final = writer.on_train_end(epochs - 1, selector=selector, ema=ema, **ck)
            print(f"\n  end of training: {', '.join(final)}")
        else:
            print(f"\n  stopped at epoch {epochs - 1} of a {args.total_epochs}-epoch "
                  f"budget; {LATEST_CHECKPOINT} left in place for resume")
    finally:
        if hasattr(generator, "_finish"):
            generator._finish()

    banner("RECORD AND DETECTORS")
    raw = record.raw_metric_trace()
    det1 = detector_1_raw_mse_at_saved_checkpoint(raw, selector.saved_checkpoint_epoch)
    det2 = detector_2_preds_std_all_zero([r["preds_std"] for r in record.validation_rows])
    record.write(saved_checkpoint_epoch=selector.saved_checkpoint_epoch,
                 argmin_epoch_ungated=argmin_epoch_ungated(raw),
                 detectors={"detector_1": det1, "detector_2": det2},
                 standardisation=standardiser.as_dict(), parameters=report.as_dict())
    curve = record.plot(saved_checkpoint_epoch=selector.saved_checkpoint_epoch)
    print(f"  saved_checkpoint_epoch {selector.saved_checkpoint_epoch}   "
          f"argmin_epoch_ungated {argmin_epoch_ungated(raw)}")
    print(f"  detector 1 (RAW at saved epoch) {det1['value']:.4f} -> triggered {det1['triggered']}")
    print(f"  detector 2 (preds_std all zero) max {det2['max']:.5f} -> triggered {det2['triggered']}")
    for name in (BEST_CHECKPOINT, FINAL_CHECKPOINT, LATEST_CHECKPOINT):
        path = out_dir / name
        print(f"  {name:24s} {'present' if path.is_file() else 'absent'}")

    n_mothers = assert_validation_file_readable(out_dir / "validation_predictions.csv",
                                                expected_mothers=len(val_rows))
    mean_step = float(np.mean(step_times))
    budget = assert_validation_within_time_budget(float(np.mean(val_times)), mean_step)
    banner("COST")
    print(f"  smoke-test assertion 1 PASSED: {n_mothers} mothers in the per-mother file")
    print(f"  smoke-test assertion 2 PASSED: validation {budget['validation_seconds']:.2f} s "
          f"of a {budget['budget_seconds']:.2f} s budget, ratio {budget['ratio']:.4f}")
    print(f"  per step: {mean_step*1000:.0f} ms total, of which "
          f"{np.mean(wait_times)*1000:.0f} ms waiting for data ({workers} workers)")
    print(f"  projected: {args.steps_per_epoch} steps + validation = "
          f"{args.steps_per_epoch*mean_step + float(np.mean(val_times)):.0f} s per epoch, "
          f"{(args.steps_per_epoch*mean_step + float(np.mean(val_times)))*args.total_epochs/3600:.1f} h "
          f"for {args.total_epochs} epochs")
    print("\n  This does NOT close the open items that need the full epoch budget.")
    print(f"\n  written to {out_dir}\n  curve: {curve.name}")


if __name__ == "__main__":
    main()
