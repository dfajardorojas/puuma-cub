"""Deep supervision for the `unet` row, by hooks and 1×1×1 heads.

`UMambaEnc` returns five outputs and the objective weights them `1/2**i` with
the last zeroed; the dataloader emits five targets regardless. MONAI's
`UNet.forward` returns a single tensor, so without this the `unet` row would
train with no deep supervision while `global_only` trains with it.

MONAI's `UNet` is never subclassed. This wraps it: forward hooks capture the
output of each intermediate decoder level, and a 1×1×1 convolution per level
projects that level to the segmentation channels, as nnU-Net's own
`seg_layers` do. The resolutions line up without resampling: the decoder
levels emerge at the five scales `DownsampleSegForDSTransform2` produces.

Imports `nnunetv2` nowhere — MONAI and torch only, so import-order safe.
"""

from __future__ import annotations

from typing import Sequence

import torch
from monai.networks.nets import UNet
from torch import nn

#: The `unet` row. Channels are the U-Mamba widths; the stride ladder is
#: nnU-Net's own pooling on this grid, and `(2,)×5` does not survive it — the
#: bottleneck rounds 2.5 up to 3 and the skip concatenation fails.
UNET_CHANNELS: tuple[int, ...] = (16, 32, 64, 128, 160, 160)
UNET_STRIDES: tuple[tuple[int, int, int], ...] = (
    (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2))
#: Background, placenta, brain. Two foreground classes need three channels.
UNET_OUT_CHANNELS = 3


class UnetBaselineError(RuntimeError):
    """Raised when the module tree is not the shape this wrapper assumes."""


def bottleneck_module(unet: UNet) -> nn.Module:
    """MONAI's deepest block, found by walking rather than by a hardcoded path.

    The `unet` row's regression head sits on the bottleneck. MONAI exposes no
    module named `bottleneck`, `encoder` or `decoder` — the deepest block sits at
    `model.1.submodule.1.submodule.1.submodule.1.submodule.1.submodule`, and that
    string is a silent breakage the first time the channel list changes length.
    This walks the same nesting :func:`decoder_up_modules` does and stops where
    the recursion bottoms out.
    """
    block = unet.model
    while True:
        children = list(block.children())
        if len(children) != 3:
            return block                            # the bottom layer
        submodule = getattr(children[1], "submodule", None)
        if submodule is None:
            raise UnetBaselineError(
                "expected a SkipConnection with a .submodule while walking to the "
                "bottleneck; MONAI's UNet structure is not what this wrapper assumes")
        block = submodule


def decoder_up_modules(unet: UNet) -> list[nn.Module]:
    """MONAI's decoder levels, shallowest first, found by walking the nesting.

    `UNet._create_block` returns `Sequential(down, SkipConnection(submodule), up)`
    at every level, so the up layer is index 2 of each Sequential and the next
    level down is `submodule` of index 1. The paths are derived by walking
    rather than written out, because a hardcoded
    `model.1.submodule.1.submodule...` string is a silent breakage the first time
    the channel list changes length.
    """
    ups: list[nn.Module] = []
    block = unet.model
    while True:
        children = list(block.children())
        if len(children) != 3:
            break                                  # the bottom layer: no up
        down, skip, up = children
        ups.append(up)
        submodule = getattr(skip, "submodule", None)
        if submodule is None:
            raise UnetBaselineError(
                f"expected a SkipConnection with a .submodule at depth {len(ups)}; "
                f"MONAI's UNet structure is not what this wrapper assumes")
        block = submodule
    if not ups:
        raise UnetBaselineError("found no decoder levels; the UNet is not nested")
    return ups


class MonaiUNetWithDeepSupervision(nn.Module):
    """MONAI `UNet` returning five outputs, highest resolution first.

    The list is ordered to match `DownsampleSegForDSTransform2`'s targets and
    `puuma.loss.deep_supervision_weights`, both of which put the full-resolution
    level first. `DeepSupervisionWrapper` consumes it unchanged.
    """

    def __init__(self, in_channels: int = 1,
                 out_channels: int = UNET_OUT_CHANNELS,
                 channels: Sequence[int] = UNET_CHANNELS,
                 strides: Sequence[Sequence[int]] = UNET_STRIDES,
                 num_res_units: int = 1, norm: str = "instance",
                 deep_supervision: bool = True) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.unet = UNet(spatial_dims=3, in_channels=in_channels,
                         out_channels=out_channels, channels=tuple(channels),
                         strides=tuple(tuple(s) for s in strides),
                         num_res_units=num_res_units, norm=norm)
        ups = decoder_up_modules(self.unet)
        # ups[0] is the top level and already emits `out_channels`; the deeper
        # ones emit feature maps and need projecting, exactly as nnU-Net's
        # `seg_layers` do.
        self._intermediate = ups[1:]
        widths = [int(channels[i]) for i in range(len(self._intermediate))]
        self.seg_layers = nn.ModuleList(
            [nn.Conv3d(w, out_channels, kernel_size=1) for w in widths])
        self._captured: list[torch.Tensor | None] = [None] * len(self._intermediate)
        for i, module in enumerate(self._intermediate):
            module.register_forward_hook(self._capture(i))
        # The bottleneck, for the row's regression head. Same walk, not a path string.
        self._bottleneck: torch.Tensor | None = None
        bottleneck_module(self.unet).register_forward_hook(self._capture_bottleneck)

    def _capture_bottleneck(self, _module, _inputs, output):
        self._bottleneck = output

    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            self._captured[index] = output
        return hook

    def forward(self, volume: torch.Tensor):
        self._captured = [None] * len(self._intermediate)
        self._bottleneck = None
        top = self.unet(volume)
        if not self.deep_supervision:
            return top
        missing = [i for i, t in enumerate(self._captured) if t is None]
        if missing:
            raise UnetBaselineError(
                f"decoder levels {missing} produced no output; the hooks did not fire, "
                "so the module tree walked at construction is not the one that ran")
        return [top] + [layer(t) for layer, t in zip(self.seg_layers, self._captured)]

    @property
    def bottleneck(self) -> torch.Tensor:
        """The deepest block's output — the bottleneck — captured on the last forward."""
        if self._bottleneck is None:
            raise UnetBaselineError(
                "no bottleneck captured; call forward() before reading it")
        return self._bottleneck

    @torch.no_grad()
    def bottleneck_latent(self, patch_size: Sequence[int]) -> int:
        """Flattened bottleneck width, from a real forward pass.

        Probed rather than derived: the count is read
        from the constructed network, and the ceil-division the ladder performs on
        axis 0 is exactly the arithmetic a hand derivation gets wrong.
        """
        device = next(self.parameters()).device
        was = self.deep_supervision
        self.deep_supervision = False
        try:
            self.unet(torch.zeros(1, 1, *patch_size, device=device))
        finally:
            self.deep_supervision = was
        return int(self.bottleneck.numel())

    def deep_supervision_parameters(self) -> int:
        """What the added heads cost, read off the module rather than quoted."""
        return sum(p.numel() for p in self.seg_layers.parameters())


class UnetRowNet(nn.Module):
    """The `unet` row, with the same interface as :class:`~puuma.model.PuumaNet`.

    Global branch only — it is the backbone swap of `global_only`, which has no
    local branch — so the blend is `Linear(3, 1)` over global regression, GA at
    scan and CL, in that order. The regression head is the same `Sequential`
    the U-Mamba rows use, on a bottleneck that happens to be the same size
    (160 × 5 × 5 × 6 = 24,000), so the two rows' heads match exactly and the
    capacity difference between them is entirely in the backbone.

    `forward` returns the dict `train_step` and the validation pass already read,
    with `local_regression` None.
    """

    def __init__(self, global_patch_size, blend_inputs=("global_regression", "ga_scan", "cl"),
                 out_channels: int = UNET_OUT_CHANNELS,
                 deep_supervision: bool = True,
                 device: torch.device | str = "cuda",
                 normalise_head: bool = True) -> None:
        super().__init__()
        from puuma.model import BLEND_INPUTS, ModelError, _regression_head
        device = torch.device(device)
        self.blend_inputs = tuple(blend_inputs)
        if "local_regression" in self.blend_inputs:
            raise ModelError(
                "the unet row is the backbone swap of global_only and has no local "
                "branch; a local_regression blend input has nothing to fill it")
        unknown = [b for b in self.blend_inputs if b not in BLEND_INPUTS]
        if unknown:
            raise ModelError(f"unknown blend input(s) {unknown}")
        self.global_patch_size = tuple(int(p) for p in global_patch_size)
        #: The row's declared count, as on `PuumaNet` — this row is exactly why the
        #: validation pass must not introspect the backbone for it.
        self.decoder_out_channels = int(out_channels)
        self.has_local_branch = False
        self.local_patch_size = None
        self.global_branch = MonaiUNetWithDeepSupervision(
            out_channels=out_channels, deep_supervision=deep_supervision).to(device)
        self.global_latent = self.global_branch.bottleneck_latent(self.global_patch_size)
        self.global_channels = int(self.global_branch.bottleneck.shape[1])
        self.local_latent = 0
        self.normalise_head = bool(normalise_head)
        self.global_head = _regression_head(self.global_latent, self.global_channels,
                                            normalise=self.normalise_head).to(device)
        self.local_head = None
        self.local_branch = None
        self.blend = nn.Linear(len(self.blend_inputs), 1).to(device)
        nn.init.zeros_(self.blend.bias)

    def forward(self, volume: torch.Tensor, patch=None, ga_scan=None, cl=None) -> dict:
        from puuma.model import ModelError
        if tuple(volume.shape[2:]) != self.global_patch_size:
            raise ModelError(
                f"global input is {tuple(volume.shape[2:])}, expected "
                f"{self.global_patch_size}")
        segmentation = self.global_branch(volume)
        global_regression = self.global_head(self.global_branch.bottleneck)
        available = {"global_regression": global_regression, "local_regression": None,
                     "ga_scan": ga_scan, "cl": cl}
        blended = self.blend(torch.cat([available[k] for k in self.blend_inputs], dim=-1))
        # No Mamba layers in this backbone, so the activation penalty has nothing
        # to penalise and is None rather than 0.0 — an explicit "not applicable".
        return {"mamba_activation": None, "ga_del": blended, "segmentation": segmentation,
                "global_regression": global_regression, "local_regression": None}

    def global_window_forward(self, volume: torch.Tensor, want_segmentation: bool):
        """The validation pass's seam, for a backbone with no encoder/decoder split.

        MONAI's `UNet` is one module; the bottleneck is captured by a hook during the
        forward, so the regression head reads it after the call rather than from a
        skip list. `want_segmentation` is accepted and ignored: this backbone computes
        the segmentation on the way to the bottleneck either way, so there is nothing
        to skip.
        """
        segmentation = self.global_branch(volume)
        logits = segmentation[0] if isinstance(segmentation, (list, tuple)) else segmentation
        return self.global_head(self.global_branch.bottleneck), logits

    def shared_encoder_parameters(self) -> list[nn.Parameter]:
        """The shared trunk.

        Not the same object as `PuumaNet`'s: there, the encoder is a named
        module the decoder and the head
        both read. MONAI builds encoder and decoder as one recursive block with no
        seam, so the whole backbone is returned — the gradient-norm probe covers
        more of the network here than it does for the U-Mamba rows, and the two
        rows' norms are not directly comparable.
        """
        return [p for p in self.global_branch.unet.parameters() if p.requires_grad]

    def parameter_report(self):
        from puuma.model import ParameterReport
        n = lambda m: sum(p.numel() for p in m.parameters())
        return ParameterReport(
            total=n(self), global_branch=n(self.global_branch), local_branch=0,
            global_head=n(self.global_head), local_head=0, blend=n(self.blend),
            global_latent=self.global_latent, local_latent=0)

    @torch.no_grad()
    def widths(self) -> list[int]:
        return list(UNET_CHANNELS)
