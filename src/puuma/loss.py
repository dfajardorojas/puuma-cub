"""The training objective, and the instrumentation for it.

    L  =  w_reg · MSE(GA_del_std)  +  w_seg · L_seg          w_reg = 2, w_seg = 1

`L_seg` is nnU-Net's own loss, unchanged: `DC_and_CE_loss` with `weight_ce=1`,
`weight_dice=1`, `do_bg=False`, `smooth=1e-5` and `MemoryEfficientSoftDiceLoss`,
wrapped in `DeepSupervisionWrapper` with weights `1/2**i`, the last zeroed and
the rest normalised to sum to 1, replicated from `nnUNetTrainer.py:350-374`.
The weights are fixed and identical for every configuration row.

This loss must be evaluated INSIDE the autocast context, as `train_step` does:
called on fp16 logits outside autocast, nnU-Net's cross-entropy reduces in
fp16 over the highest-resolution level and returns NaN. Imports `nnunetv2` at
module level — the shim must be bound first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

#: Fixed; not swept, not scheduled, not learned.
W_REG = 2.0
W_SEG = 1.0


class LossError(RuntimeError):
    """Raised on a condition the detectors require to be a hard stop."""


@dataclass(frozen=True)
class LossTerms:
    """One step's terms, and the shares the run record wants written at intervals."""

    total: torch.Tensor
    regression: torch.Tensor
    segmentation: torch.Tensor
    #: The activation penalty, unweighted, or None when it is switched off.
    #: `alpha` is the coefficient it entered `total` with, so the two together give
    #: the contribution without the reader having to know the constant.
    activation: torch.Tensor | None = None
    alpha: float = 0.0

    @property
    def weighted_activation(self) -> float:
        return 0.0 if self.activation is None else float(self.alpha * self.activation.detach())

    @property
    def weighted_regression(self) -> float:
        return float(W_REG * self.regression.detach())

    @property
    def weighted_segmentation(self) -> float:
        return float(W_SEG * self.segmentation.detach())

    def shares(self) -> dict[str, float]:
        """Fraction of the total each term contributes — a proxy; the gradient
        norms are the better quantity.

        The activation penalty is deliberately NOT in this denominator: the
        share is a ratio between the two task terms (the segmentation term can
        go negative, so the ratio is not bounded), and adding a third, strictly
        positive term would change what the share means, for a quantity that is
        not a task at all. It is reported instead as `activation_fraction`:
        what the penalty is against the whole objective, on its own scale.
        """
        total = self.weighted_regression + self.weighted_segmentation
        if total == 0.0:
            return {"regression": float("nan"), "segmentation": float("nan")}
        return {"regression": self.weighted_regression / total,
                "segmentation": self.weighted_segmentation / total}

    def activation_fraction(self) -> float:
        """The penalty against the whole objective — 0.0 when it is switched off."""
        whole = self.weighted_regression + self.weighted_segmentation + self.weighted_activation
        if whole == 0.0 or not np.isfinite(whole):
            return float("nan")
        return self.weighted_activation / whole


def deep_supervision_weights(n_levels: int) -> np.ndarray:
    """`nnUNetTrainer.py:366-374`, replicated exactly.

    Its comment says "we don't use the lowest 2 outputs" while the code zeroes only
    `weights[-1]`. The comment is stale; the code is what runs.
    """
    weights = np.array([1 / (2 ** i) for i in range(n_levels)])
    weights[-1] = 0
    return weights / weights.sum()


def build_segmentation_loss(label_manager, batch_dice: bool,
                            deep_supervision_scales: Sequence | None) -> nn.Module:
    """nnU-Net's own segmentation loss, built with nnU-Net's own arguments."""
    if label_manager.has_regions:
        loss = DC_and_BCE_loss(
            {}, {"batch_dice": batch_dice, "do_bg": True, "smooth": 1e-5, "ddp": False},
            use_ignore_label=label_manager.ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss)
    else:
        loss = DC_and_CE_loss(
            {"batch_dice": batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": False},
            {}, weight_ce=1, weight_dice=1, ignore_label=label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)
    if deep_supervision_scales is not None:
        loss = DeepSupervisionWrapper(loss, deep_supervision_weights(len(deep_supervision_scales)))
    return loss


#: The activation penalty, off unless a coefficient is supplied. Jamba-1.5
#: (arXiv:2408.12570 §3.2) reports the same phenomenon in a hybrid
#: Transformer-Mamba model — "the output of the last Mamba layers" growing "as
#: high as 4e6", harmless in BF16 and breaking fp16's 64K range — and fixes it
#: with an auxiliary loss "proportional to the mean-square of activations in
#: the forward pass, with a configurable alpha factor". Their alpha was chosen
#: for a language model in BF16 and does not transfer; it is not assumed here.
ACTIVATION_ALPHA = 0.0


class PuumaLoss(nn.Module):
    """The fixed-weight scalarization, and the optional activation penalty.

    The penalty is a third term and it is not a task: the two task weights keep
    their values and their ratio, and what the penalty constrains is a
    direction the task loss cannot see, so it competes with neither task term.
    It is excluded from `shares()`'s denominator for the reason given there and
    reported separately.
    """

    def __init__(self, label_manager, batch_dice: bool,
                 deep_supervision_scales: Sequence | None,
                 activation_alpha: float = ACTIVATION_ALPHA) -> None:
        super().__init__()
        self.segmentation_loss = build_segmentation_loss(
            label_manager, batch_dice, deep_supervision_scales)
        self.regression_loss = nn.MSELoss()
        self.activation_alpha = float(activation_alpha)
        if self.activation_alpha < 0:
            raise LossError("the activation penalty's coefficient must not be negative")
        self.n_deep_supervision = (len(deep_supervision_scales)
                                   if deep_supervision_scales is not None else 1)

    def forward(self, prediction: dict, targets: Sequence[torch.Tensor],
                ga_del_std: torch.Tensor) -> LossTerms:
        segmentation = prediction["segmentation"]
        if self.n_deep_supervision > 1:
            if not isinstance(segmentation, (list, tuple)) or not isinstance(targets, (list, tuple)):
                raise LossError("deep supervision is on but the outputs or targets are not a list")
            if len(segmentation) != self.n_deep_supervision or len(targets) != self.n_deep_supervision:
                raise LossError(
                    f"deep supervision expects {self.n_deep_supervision} outputs and targets; "
                    f"got {len(segmentation)} outputs and {len(targets)} targets. Every level "
                    "must be consumed — a silently dropped level reweights the loss.")
        seg = self.segmentation_loss(segmentation, targets)
        reg = self.regression_loss(prediction["ga_del"].float(), ga_del_std.float())
        total = W_REG * reg + W_SEG * seg
        activation = prediction.get("mamba_activation")
        if self.activation_alpha > 0 and activation is not None:
            total = total + self.activation_alpha * activation
        else:
            activation = None
        checked = [("regression", reg), ("segmentation", seg), ("total", total)]
        if activation is not None:
            checked.insert(2, ("activation", activation))
        # Detector 3: a hard stop, checked every step, never a warning.
        for name, value in checked:
            if not torch.isfinite(value):
                raise LossError(
                    f"non-finite {name} loss ({float(value)}). Detector 3 makes this a "
                    "hard stop: a run that continues past a NaN produces a checkpoint nobody "
                    "can interpret.")
        return LossTerms(total=total, regression=reg, segmentation=seg,
                         activation=activation, alpha=self.activation_alpha)


def per_task_gradient_norms(terms: LossTerms, shared_parameters: Sequence[nn.Parameter],
                            *, scale: float = 1.0) -> dict[str, float]:
    """The recorded per-task gradient norms into the shared encoder.

    Loss share is a proxy; this is the quantity the "segmentation dominates"
    concern is actually about. Each term's gradient is taken separately with
    `torch.autograd.grad`, which does not touch `.grad`, so the caller's own
    backward pass is unaffected. The graph is retained because both terms and
    then the real backward all traverse it. Weighted by the objective's `w_reg`
    and `w_seg`, because what steers the encoder is the gradient of the term as
    it enters the total, not of the bare term.

    `scale` must be the AMP loss scale when AMP is on: under autocast the
    backward runs in fp16, and an unscaled per-term gradient can underflow to
    zero or overflow to NaN while the loss itself stays finite. This probe
    bypasses `GradScaler` by construction, so it does the scaling itself and
    divides the norm back out.
    """
    params = [p for p in shared_parameters if p.requires_grad]
    if not params:
        raise LossError("no shared encoder parameters to take gradient norms over")
    if not np.isfinite(scale) or scale <= 0:
        raise LossError(f"AMP loss scale must be finite and positive, got {scale}")
    out = {}
    for name, weight, value in (("regression", W_REG, terms.regression),
                                ("segmentation", W_SEG, terms.segmentation)):
        out[name] = _scaled_norm(weight * value, params, scale)
    return out


def _scaled_norm(term: torch.Tensor, params: Sequence[nn.Parameter], scale: float) -> float:
    """The gradient norm of one term, found at a scale that neither under- nor overflows.

    The two terms can need different scales, which is why one shared factor
    does not work: at one scale the regression gradient is finite and the
    segmentation gradient overflows, at another the reverse. The ladder below
    tries the caller's scale first and then walks down and up, taking the
    first finite non-zero answer, and divides it back out — so the number
    reported is the true norm regardless of which rung produced it.
    """
    ladder = [scale] + [scale / (16 ** k) for k in range(1, 5)] + [1.0, 256.0, 65536.0]
    saw_finite = False
    for rung in ladder:
        if not np.isfinite(rung) or rung <= 0:
            continue
        grads = torch.autograd.grad(rung * term, params, retain_graph=True, allow_unused=True)
        if all(g is None for g in grads):
            # This term does not reach the shared encoder at all — a real answer,
            # not a numerical one. It happens when a branch is disconnected.
            return 0.0
        total = torch.zeros((), device=params[0].device, dtype=torch.float32)
        for g in grads:
            if g is not None:
                total = total + g.detach().float().pow(2).sum()
        norm = float(total.sqrt())
        if np.isfinite(norm):
            saw_finite = True
            if norm > 0.0:
                return norm / rung
    # Finite at some rung but zero everywhere: smaller than fp16 can represent.
    # Non-finite at every rung: report NaN rather than invent a number.
    return 0.0 if saw_finite else float("nan")
