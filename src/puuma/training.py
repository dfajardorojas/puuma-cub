"""One training step, and the pieces required around it.

Scope is deliberately narrow: this is what a training step needs; the
per-mother validation pass, the checkpointing and the inference are not here
and are not started.

Imports `nnunetv2` at module level — the shim must be bound first, before
trainer construction, not merely the training call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from puuma.dataloader import FOREGROUND_THRESHOLD as FOREGROUND_GATE
from puuma.loss import LossError, LossTerms, PuumaLoss, per_task_gradient_norms

#: Read from `nnUNetTrainer.py:142-143, 471-473, 907/912`. nnU-Net's own
#: initial rate, retained as a named reference. Not what trained runs use —
#: see `BACKBONE_LR`.
INITIAL_LR = 1e-2
#: The backbone initial rate. nnU-Net's inherited 1e-2 does not converge
#: inside the 250-epoch budget on this problem; 4e-3 does, without approaching
#: the activation ceiling.
BACKBONE_LR = 4e-3
#: What nnU-Net ships (`nnUNetTrainer.py:143`). Retained as a named reference,
#: and still the head group's value.
WEIGHT_DECAY = 3e-5
#: The backbone weight decay, on the encoder and decoder only; it bounds the
#: Mamba activation growth. The head group keeps the inherited 3e-5: one group,
#: one number.
BACKBONE_WEIGHT_DECAY = 3e-4
#: What nnU-Net ships (`nnUNetTrainer.py:471`). Kept as a named reference.
MOMENTUM = 0.99
#: 0.90 against nnU-Net's 0.99, on the encoder and decoder: at 0.99 the Mamba
#: activations grow without bound and the run dies on detector 3; at 0.90 the
#: growth is flat. Reducing the learning rate instead costs segmentation badly
#: and is self-defeating under PolyLR, whose integrated rate is `lr0 * T / 1.9`.
BACKBONE_MOMENTUM = 0.90
GRAD_CLIP = 12.0

#: The second parameter group: the two regression heads (each including its
#: `InstanceNorm3d`) and the blend layer. These modules are not part of the
#: U-Mamba architecture, so nnU-Net's settings were never chosen for them, and
#: at the backbone's step size the heads' units die together early in training.
#: Normalising the head's input does not substitute: `InstanceNorm3d` scales
#: the squeeze's weight gradient up by the same factor it scales the input,
#: making the step larger rather than better conditioned. The step size is the
#: thing, and it is set here; 4e-4 holds a 10:1 ratio to the backbone rate.
HEAD_LR = 4e-4
HEAD_MOMENTUM = 0.9
#: The attributes that make up that group. Both `PuumaNet` and `UnetRowNet` expose
#: these names; `local_head` is `None` on a row without a local branch and is then
#: absent from the group rather than empty in it.
HEAD_GROUP_MODULES = ("global_head", "local_head", "blend")

# --- Detector 5: the gate-fallback rate -----------------------------------
#: Fraction of items in the window that may fall back before the run stops.
#: Set well above the healthy fallback rate so it cannot fire on noise; it
#: exists to catch something wrong upstream — a fold whose placentas are
#: unusually thin, a preprocessing change, an augmentation range widened — not
#: to police the known rate.
GATE_FAILURE_RATE_THRESHOLD = 0.01
#: Evaluated over a rolling window of steps, not per step or per epoch: 2,500
#: steps is ten epochs at 250 steps each. Large enough that a run at the
#: healthy rate never trips it, small enough to notice a real change within 4%
#: of the budget.
GATE_FAILURE_WINDOW_STEPS = 2500


class TrainingError(RuntimeError):
    """Raised for a startup or step condition this design makes a hard stop."""


@dataclass
class StepRecord:
    """What the run record wants written per step or at intervals."""

    step: int
    total: float
    regression: float
    segmentation: float
    shares: dict[str, float]
    grad_norms: dict[str, float] | None
    seconds_data: float
    seconds_compute: float
    lr: float
    #: The activation penalty, unweighted, or None when it is off, with what
    #: it contributes to the whole objective. The record keeps it beside the two
    #: task terms and not inside `shares` — see `LossTerms.shares`.
    activation: float | None = None
    activation_fraction: float = 0.0
    #: How many items in this batch fell back to the best centre.
    gate_failed: int = 0
    #: The achieved fill of those items, so the record carries the distribution
    #: and not only the count.
    gate_fills: list = field(default_factory=list)


class GateFailureMonitor:
    """Detector 5: the gate-fallback rate over a rolling window of steps.

    The fallback itself is benign — a near miss at the gate, not a missed
    placenta. What is not benign is the rate changing, because that means the
    data or the augmentation has moved. This raises when it does.

    It only evaluates once a full window has accumulated, so it cannot fire on the
    first few steps of a run.
    """

    def __init__(self, threshold: float = GATE_FAILURE_RATE_THRESHOLD,
                 window_steps: int = GATE_FAILURE_WINDOW_STEPS) -> None:
        self.threshold = float(threshold)
        self.window_steps = int(window_steps)
        self._items: list[int] = []
        self._failed: list[int] = []

    @property
    def window_full(self) -> bool:
        return len(self._items) >= self.window_steps

    def rate(self) -> float:
        total = sum(self._items)
        return (sum(self._failed) / total) if total else 0.0

    def observe(self, n_items: int, n_failed: int) -> None:
        self._items.append(int(n_items))
        self._failed.append(int(n_failed))
        if len(self._items) > self.window_steps:
            self._items.pop(0)
            self._failed.pop(0)
        if self.window_full and self.rate() > self.threshold:
            raise TrainingError(
                f"detector 5: {sum(self._failed)} of {sum(self._items)} items in the "
                f"last {self.window_steps} steps fell back from the {FOREGROUND_GATE:.0%} "
                f"gate — a rate of {self.rate():.3%} against a threshold of "
                f"{self.threshold:.1%}. A rate this high means something upstream "
                "changed; the fallback is not the problem, and lowering the "
                "gate would hide it. Training stops.")


# --- Detector 6: the Mamba activation ceiling ----------------------------
#: Stop when any `MambaLayer`'s output magnitude exceeds this. Calibrated as
#: the geometric midpoint of the highest healthy per-epoch maximum and the
#: lowest divergent one, symmetric on the log scale the quantity lives on.
MAMBA_ACTIVATION_CEILING = 549.3
#: The fp32 analogue, for runs with AMP off. The ceiling sits one window's
#: growth below the magnitude at which the convolution after a Mamba layer
#: overflows; in fp32 that is far above anything observable, so detector 6
#: stops being a useful early warning there and detector 3, the non-finite
#: hard stop, is the real backstop.
MAMBA_ACTIVATION_CEILING_FP32 = 6.9262e36
#: Three consecutive steps, so one anomalous batch cannot stop a healthy run while a
#: genuine runaway — monotone over ten steps or more — satisfies it immediately.
MAMBA_ACTIVATION_WINDOW = 3


class MambaActivationMonitor:
    """Detector 6, evaluated every training step.

    A short stable window cannot rule out a slow climb: a run can hold a small
    per-step growth factor for several epochs and still extrapolate to overflow
    later, indistinguishable from a healthy run over the window observed. It
    reports which layer, because the leading layer is not stable across runs.

    The quantity is already computed — `MambaActivationTracker` records it on
    every forward for the activation penalty — so this costs a comparison per
    step and is free with the penalty switched off.
    """

    def __init__(self, ceiling: float = MAMBA_ACTIVATION_CEILING,
                 window: int = MAMBA_ACTIVATION_WINDOW) -> None:
        self.ceiling = float(ceiling)
        self.window = int(window)
        self.consecutive = 0
        #: Lifetime worst, for the run's own record.
        self.worst = 0.0
        self.worst_layer = ""
        #: Per epoch, because the run record carries it per epoch and the
        #: lifetime figure cannot show whether the growth is bounded or
        #: climbing. Reset by :meth:`take_epoch_max`, which the training loop
        #: calls at the epoch end.
        self.epoch_worst = 0.0
        self.epoch_worst_layer = ""

    def take_epoch_max(self) -> tuple[float, str]:
        """This epoch's maximum and the layer that produced it, then reset.

        The leading layer is not a constant across runs, so recording the value
        without the layer would lose the thing that moves.
        """
        value, layer = self.epoch_worst, self.epoch_worst_layer
        self.epoch_worst, self.epoch_worst_layer = 0.0, ""
        return value, layer

    def observe(self, magnitudes: dict, step: int) -> None:
        """`magnitudes` is `model.mamba.magnitudes`: layer name -> max |output|."""
        if not magnitudes:
            return
        layer, value = max(((k, float(v)) for k, v in magnitudes.items()),
                           key=lambda kv: kv[1])
        if value > self.worst:
            self.worst, self.worst_layer = value, layer
        if value > self.epoch_worst:
            self.epoch_worst, self.epoch_worst_layer = value, layer
        if value <= self.ceiling:
            self.consecutive = 0
            return
        self.consecutive += 1
        if self.consecutive >= self.window:
            raise TrainingError(
                f"detector 6 at step {step}: the maximum Mamba activation has "
                f"exceeded {self.ceiling:g} on {self.consecutive} consecutive steps, "
                f"reaching {value:.4g} at {layer!r}. This is the growth that ends "
                "in an fp16 overflow in the convolution after a Mamba layer. "
                "Training stops here so that a readable checkpoint remains; "
                "detector 3 would stop too late to leave one.")


#: `polylr.py:5` — the exponent nnU-Net uses.
POLY_LR_EXPONENT = 0.9


class PolyLR:
    """PolyLR, replicated because the clone's class cannot be constructed.

    `training/lr_scheduler/polylr.py:11` calls
    ``super().__init__(optimizer, current_step if current_step is not None else -1, False)``
    — three positional arguments. `torch.optim.lr_scheduler.LRScheduler.__init__`
    in the installed PyTorch (2.10.0) takes ``(optimizer, last_epoch=-1)``; the
    third argument was the long-deprecated ``verbose``, now removed. Constructing
    it raises ``TypeError``. The clone is read-only, so the
    arithmetic is reproduced here instead of patched there.

    This is exact, not an approximation: `polylr.py:14-20`'s ``step`` uses none
    of the base class's machinery — it computes
    ``initial_lr * (1 - step / max_steps) ** exponent`` and assigns it to every
    param group. That is what this does, with the same default exponent.

    One schedule over two parameter groups: with the second group
    (:func:`build_optimiser`), each group's learning rate is its own base lr
    times the one poly factor ``(1 - step/max_steps) ** exponent``, read from the
    group's ``initial_lr`` key — PyTorch's standard convention for exactly this.
    The groups therefore share the schedule and the budget and differ only in
    base lr; there is no second schedule and no ``ReduceLROnPlateau``.

    This is not a further deviation from the framework's arithmetic:
    ``polylr.py:19`` assigns one value to every group because a stock trainer has
    exactly one group, where "assign the same lr" and "scale each group's own base"
    are the same operation. On a single-group optimiser with no ``initial_lr`` key
    this class still does precisely what it did before.
    """

    def __init__(self, optimiser: torch.optim.Optimizer, initial_lr: float,
                 max_steps: int, exponent: float = POLY_LR_EXPONENT) -> None:
        if max_steps <= 0:
            raise TrainingError("PolyLR needs a positive max_steps; the schedule is "
                                "defined over the budget actually run")
        self.optimiser = optimiser
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        #: One base lr per group, so a second group keeps its own through the decay.
        self.base_lrs = [float(g.get("initial_lr", initial_lr))
                         for g in optimiser.param_groups]
        self.ctr = 0
        self.step(0)

    def factor(self, current_step: int) -> float:
        """The decay itself, shared by every group (`polylr.py:19`)."""
        return (1 - current_step / self.max_steps) ** self.exponent

    def step(self, current_step: int | None = None) -> float:
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        factor = self.factor(current_step)
        for group, base in zip(self.optimiser.param_groups, self.base_lrs):
            group["lr"] = base * factor
        return float(self.optimiser.param_groups[0]["lr"])


@dataclass(frozen=True)
class ParameterGroups:
    """The optimiser partition, with the names, so a test can check it rather than trust it."""

    backbone: list
    head: list
    backbone_names: list
    head_names: list

    def counts(self) -> dict[str, int]:
        return {"backbone_tensors": len(self.backbone), "head_tensors": len(self.head),
                "backbone_parameters": sum(p.numel() for p in self.backbone),
                "head_parameters": sum(p.numel() for p in self.head)}


def partition_parameters(model: nn.Module) -> ParameterGroups:
    """Split the model into the two optimiser groups, and prove the split is a partition.

    Group 2 is `global_head`, `local_head` and `blend` — both regression heads
    including their `InstanceNorm3d`, because the norm is inside the head's own
    `Sequential` and is as much a part of it as the squeeze. Group 1 is
    everything else: the global branch's encoder and decoder, and the local
    branch's encoder.

    The two failure modes are not symmetric and neither is silent here:

    * a parameter in neither group does not train at all, and nothing downstream
      would raise — the run would simply be wrong;
    * a parameter in both would be stepped twice per iteration. PyTorch itself
      rejects that (``some parameters appear in more than one parameter group``), but
      only for tensors it can see as duplicates, so it is checked here as well.

    A renamed head attribute would empty group 2 rather than error, so an empty
    group 2 is a hard stop too.

    The partition is over distinct tensors, not over names: nnU-Net registers
    each convolution twice — `ConvDropoutNormReLU` keeps `self.conv` and then puts
    the same object into `self.all_modules`, so
    `global_branch.encoder.stem.1.conv1.conv.weight` and
    `...conv1.all_modules.0.weight` are one tensor under two names. That is module
    aliasing, not weight tying, and `named_parameters()` deduplicates it by default.
    Counting names instead of tensors reports roughly twice the model's parameters
    and turns a normal encoder into an apparent tying error.
    """
    head_ids: dict[int, str] = {}
    for attribute in HEAD_GROUP_MODULES:
        module = getattr(model, attribute, None)
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) in head_ids:
                raise TrainingError(
                    f"a parameter is shared between {head_ids[id(parameter)]!r} and "
                    f"{attribute!r}; the second group would step it twice")
            head_ids[id(parameter)] = attribute
    if not head_ids:
        raise TrainingError(
            f"none of {HEAD_GROUP_MODULES} exists on {type(model).__name__}, so the "
            "second parameter group would be empty and the heads would silently "
            "train at the encoder's learning rate.")

    backbone, head, backbone_names, head_names = [], [], [], []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        seen.add(id(parameter))
        if id(parameter) in head_ids:
            head.append(parameter)
            head_names.append(name)
        else:
            backbone.append(parameter)
            backbone_names.append(name)

    if len(head) != len(head_ids):
        raise TrainingError(
            f"{len(head_ids)} head-group tensors were collected from "
            f"{HEAD_GROUP_MODULES} but {len(head)} of them are reachable from "
            "model.named_parameters(); a head module is not registered on the model")
    if len(backbone) + len(head) != len(seen):
        raise TrainingError("the two groups do not partition the model's parameters")
    total = sum(p.numel() for p in model.parameters())
    grouped = sum(p.numel() for p in backbone) + sum(p.numel() for p in head)
    if grouped != total:
        raise TrainingError(
            f"the two groups hold {grouped:,d} parameters and the model has "
            f"{total:,d}; a parameter in neither group would not train at all")
    return ParameterGroups(backbone, head, backbone_names, head_names)


def build_optimiser(model: nn.Module, num_epochs: int,
                    initial_lr: float = BACKBONE_LR, head_lr: float = HEAD_LR,
                    head_momentum: float = HEAD_MOMENTUM
                    ) -> tuple[torch.optim.Optimizer, PolyLR]:
    """The optimiser and schedule: one SGD, two parameter groups.

    Group 1 — encoder and decoder — carries the inherited settings with exactly
    two changed numbers: momentum 0.90 against nnU-Net's 0.99
    (`BACKBONE_MOMENTUM`) and weight decay 3e-4 against 3e-5
    (`BACKBONE_WEIGHT_DECAY`). Group 2 — the two regression heads and the blend
    — differs in learning rate and momentum only, for the reason recorded on
    `HEAD_LR`: those modules are not part of the U-Mamba architecture and the
    backbone's step size kills them within one epoch.

    Nesterov, the gradient clip and the schedule are shared, and so is the head
    group's weight decay, which stays at the inherited 3e-5. It is one
    optimiser class with two learning rates, not two optimisers.

    PolyLR is defined over `num_epochs` (`nnUNetTrainer.py:473`), so the schedule is
    only correct if the number given here is the number actually run.
    """
    groups = partition_parameters(model)
    optimiser = torch.optim.SGD(
        [{"params": groups.backbone, "lr": initial_lr, "initial_lr": initial_lr,
          "momentum": BACKBONE_MOMENTUM, "weight_decay": BACKBONE_WEIGHT_DECAY,
          "name": "backbone"},
         {"params": groups.head, "lr": head_lr, "initial_lr": head_lr,
          "momentum": head_momentum, "weight_decay": WEIGHT_DECAY, "name": "head"}],
        lr=initial_lr, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, nesterov=True)
    return optimiser, PolyLR(optimiser, initial_lr, num_epochs)


def assert_remove_label_transform(transforms) -> None:
    """Detector 4, asserted at startup.

    The preprocessor writes `-1` into the segmentation for the cropped region.
    Stock nnU-Net removes it before the loss (`nnUNetTrainer.py:734`); skipping it
    faults the CUDA context with "unspecified launch failure" rather than raising,
    and the reported stack points at the wrong line.
    """
    names = [type(t).__name__ for t in getattr(transforms, "transforms", transforms)]
    if "RemoveLabelTransform" not in names:
        raise TrainingError(
            "RemoveLabelTransform(-1, 0) is absent from the transform chain. "
            "Detector 4 asserts this at startup because its absence does not raise — "
            "it faults the CUDA context and blames the wrong line. Chain was: "
            f"{names}")


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move one dataloader batch onto the device, converting the scalars."""
    out = {}
    out["data"] = batch["data"].to(device, non_blocking=True)
    out["patch"] = batch["patch"].to(device, non_blocking=True)
    out["target"] = [t.to(device, non_blocking=True) for t in batch["target"]]
    for key in ("ga_scan", "cl", "ga_del"):
        value = batch[key]
        tensor = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
        out[key] = tensor.to(device, non_blocking=True).float()
    out["keys"] = batch["keys"]
    for key in ("patch_gate_failed", "patch_fill"):
        if key in batch:
            out[key] = np.asarray(batch[key])
    return out


def train_step(model: nn.Module, loss_fn: PuumaLoss, optimiser: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler | None, batch: dict, device: torch.device,
               step: int, *, seconds_data: float = 0.0,
               collect_grad_norms: bool = False,
               gate_monitor: "GateFailureMonitor | None" = None) -> StepRecord:
    """One optimisation step on one batch.

    The gradient-norm probe runs only when asked, because it costs two extra
    backward traversals of the shared encoder. It uses `torch.autograd.grad`, which
    does not touch `.grad`, so the real backward below is unaffected.
    """
    t0 = time.perf_counter()
    b = batch_to_device(batch, device)
    optimiser.zero_grad(set_to_none=True)

    amp = scaler is not None
    with torch.autocast("cuda", enabled=amp) if device.type == "cuda" else _null():
        prediction = model(b["data"], b["patch"], b["ga_scan"], b["cl"])
        terms = loss_fn(prediction, b["target"], b["ga_del"])

    grad_norms = None
    if collect_grad_norms:
        # The AMP loss scale must be passed through: without it the fp16 backward
        # underflows the regression term's gradient to zero or overflows it to NaN.
        grad_norms = per_task_gradient_norms(
            terms, model.shared_encoder_parameters(),
            scale=float(scaler.get_scale()) if scaler is not None else 1.0)

    if amp:
        scaler.scale(terms.total).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimiser)
        scaler.update()
    else:
        terms.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    flagged = b.get("patch_gate_failed")
    fills = b.get("patch_fill")
    n_failed = int(np.count_nonzero(flagged)) if flagged is not None else 0
    failed_fills = ([float(f) for f, g in zip(fills, flagged) if g]
                    if (fills is not None and flagged is not None) else [])
    if gate_monitor is not None:
        gate_monitor.observe(len(flagged) if flagged is not None else 0, n_failed)
    return StepRecord(
        step=step, total=float(terms.total.detach()),
        regression=float(terms.regression.detach()),
        segmentation=float(terms.segmentation.detach()),
        shares=terms.shares(), grad_norms=grad_norms,
        activation=(None if terms.activation is None
                    else float(terms.activation.detach())),
        activation_fraction=terms.activation_fraction(),
        seconds_data=seconds_data, seconds_compute=time.perf_counter() - t0,
        lr=float(optimiser.param_groups[0]["lr"]),
        gate_failed=n_failed, gate_fills=failed_fills)


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False
