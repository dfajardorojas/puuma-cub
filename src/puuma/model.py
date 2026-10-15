"""The composite model: global branch, local branch, blend layer.

Imports `nnunetv2` at module level, so the paths shim must be bound first —
before trainer construction, not just the training call.

* Global branch — `UMambaEnc` at the native patch [80, 160, 192], widths
  16/160 from `puuma.plans`' configuration, decoder emitting three channels
  with deep supervision; the regression head sits on the bottleneck, outside
  the deep-supervision wrapper.
* Local branch — the three-stage encoder [16, 32, 64] at 20³.
* Blend layer — `Linear(4, 1)` over, in this order: global regression output,
  local regression output, GA at scan, CL.

No parameter count is quoted; :func:`parameter_report` reads it off the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from nnunetv2.nets.UMambaEnc_3d import ResidualMambaEncoder, get_umamba_enc_3d_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

#: The local branch. Divisibility by 4 is a hard constraint.
LOCAL_STAGES = 3
LOCAL_FEATURES = (16, 32, 64)
LOCAL_STRIDES = ((1, 1, 1), (2, 2, 2), (2, 2, 2))
#: The head's nonlinearity: `LeakyReLU` at nnU-Net's own slope. A leaky unit
#: with a negative pre-activation still passes gradient, so the head cannot be
#: knocked permanently silent — the dead-ReLU failure a plain `ReLU` head is
#: exposed to. nnU-Net uses `LeakyReLU(0.01)` throughout its own architecture,
#: including the local encoder below. It adds no parameters.
HEAD_NEGATIVE_SLOPE = 0.01
#: The order the blend layer reads its four inputs in. Stated, because a silent
#: permutation here is invisible in the loss and changes what the layer means.
BLEND_INPUTS = ("global_regression", "local_regression", "ga_scan", "cl")


class ModelError(RuntimeError):
    """Raised for a configuration this design does not permit."""


@dataclass(frozen=True)
class ParameterReport:
    """Counts read from a constructed model, never from a document."""

    total: int
    global_branch: int
    local_branch: int
    global_head: int
    local_head: int
    blend: int
    global_latent: int
    local_latent: int

    def as_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in
                ("total", "global_branch", "local_branch", "global_head",
                 "local_head", "blend", "global_latent", "local_latent")}


def _regression_head(latent_dim: int, channels: int,
                     normalise: bool = True) -> nn.Sequential:
    """InstanceNorm3d → Flatten → Linear(latent, 16) → LeakyReLU → Linear(16, 1).

    `normalise=False` drops the `InstanceNorm3d` and gives the bare head; the
    default is normalised. The final bias is initialised to 0, i.e. the
    standardised mean.

    The `InstanceNorm3d` exists because the bottleneck can arrive at a small
    scale, compressing the whole pre-ReLU distribution into a band where all
    sixteen units sit a hair from zero and can die together. It is nnU-Net's
    own operator — `UMambaEnc_3d.py` hardcodes `InstanceNorm` throughout — and
    costs `2 × channels` parameters rather than the ~48,000 a `LayerNorm` over
    the flattened vector would; `BatchNorm1d` is excluded because at batch 2
    its statistics come from two samples and it behaves differently at eval.
    It is placed before `Flatten` so the normalisation is per channel over the
    spatial extent; after flattening there are no channels left to normalise
    over.
    """
    layers = [nn.InstanceNorm3d(channels, affine=True)] if normalise else []
    layers += [nn.Flatten(), nn.Linear(latent_dim, 16),
               nn.LeakyReLU(negative_slope=HEAD_NEGATIVE_SLOPE), nn.Linear(16, 1)]
    head = nn.Sequential(*layers)
    nn.init.zeros_(head[-1].bias)
    return head


def squeeze_of(head: nn.Sequential) -> nn.Linear:
    """The 24000 → 16 layer, found by type so the index cannot go stale."""
    return next(m for m in head if isinstance(m, nn.Linear))


def build_local_encoder(patch_size: Sequence[int]) -> ResidualMambaEncoder:
    """The local branch. Raises if the size violates the divisibility rule."""
    patch = tuple(int(p) for p in patch_size)
    total_stride = 1
    for s in LOCAL_STRIDES:
        total_stride *= s[0]
    bad = [p for p in patch if p % total_stride]
    if bad:
        raise ModelError(
            f"local patch {patch} is not divisible by {total_stride} on every axis. "
            "This is a hard constraint: a size that is not makes "
            "UMambaEnc_3d.py:176's floor-division bookkeeping disagree with what the "
            "convolution produces, and the network constructs and then dies on the "
            "first forward.")
    return ResidualMambaEncoder(
        input_size=patch, input_channels=1, n_stages=LOCAL_STAGES,
        features_per_stage=list(LOCAL_FEATURES), conv_op=nn.Conv3d,
        kernel_sizes=[(3, 3, 3)] * LOCAL_STAGES, strides=[tuple(s) for s in LOCAL_STRIDES],
        n_blocks_per_stage=[1] * LOCAL_STAGES, conv_bias=False,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
        return_skips=False, stem_channels=LOCAL_FEATURES[0], pool_type="conv")


class MambaActivationTracker:
    """The activation penalty: the mean-square of every `MambaLayer`'s output.

    Every `MambaLayer` in both branches is hooked and the per-layer magnitudes
    are recorded; a rule hooking only the last layers (where Jamba-1.5,
    arXiv:2408.12570 §3.2, penalised) would assume where the growth is rather
    than measure it.

    The mean-square is taken in fp32: a runaway activation squared in fp16
    overflows at 65,504, so a penalty computed in the autocast dtype would
    itself become the first non-finite value it exists to prevent. The
    per-layer mean is used rather than the sum, so the coefficient means the
    same thing on a row with a local branch and on one without.
    """

    def __init__(self, model: nn.Module) -> None:
        from nnunetv2.nets.UMambaEnc_3d import MambaLayer
        self.layers = [(name, module) for name, module in model.named_modules()
                       if isinstance(module, MambaLayer)]
        self.squares: dict[str, torch.Tensor] = {}
        self.magnitudes: dict[str, torch.Tensor] = {}
        self.handles = [module.register_forward_hook(self._make(name))
                        for name, module in self.layers]

    def _make(self, name: str):
        def hook(_module, _inputs, output):
            wide = output.float()
            self.squares[name] = wide.pow(2).mean()
            self.magnitudes[name] = wide.detach().abs().max()
        return hook

    def reset(self) -> None:
        self.squares.clear()
        self.magnitudes.clear()

    def penalty(self) -> torch.Tensor | None:
        if not self.squares:
            return None
        return torch.stack(list(self.squares.values())).mean()


class PuumaNet(nn.Module):
    """The composite architecture. Returns the blended prediction and the segmentation list."""

    def __init__(self, plans_manager: PlansManager, dataset_json: dict,
                 configuration_manager: ConfigurationManager,
                 local_patch_size: Sequence[int],
                 num_input_channels: int = 1,
                 deep_supervision: bool = True,
                 device: torch.device | str = "cuda",
                 blend_inputs: Sequence[str] = BLEND_INPUTS,
                 normalise_head: bool = True,
                 decoder_out_channels: int | None = None) -> None:
        super().__init__()
        device = torch.device(device)
        # The configuration rows are data: the blend's width, its order, and whether a local
        # branch exists at all follow from this tuple and from nothing else.
        self.blend_inputs = tuple(blend_inputs)
        unknown = [b for b in self.blend_inputs if b not in BLEND_INPUTS]
        if unknown:
            raise ModelError(f"unknown blend input(s) {unknown}; valid inputs are {BLEND_INPUTS}")
        if "global_regression" not in self.blend_inputs:
            raise ModelError("every row keeps the global branch")
        self.has_local_branch = "local_regression" in self.blend_inputs
        self.local_patch_size = tuple(int(p) for p in local_patch_size)
        self.global_patch_size = tuple(int(p) for p in configuration_manager.patch_size)
        self.deep_supervision = deep_supervision
        #: The declared segmentation channel count, from the row. The validation
        #: pass needs it to size its accumulator and must not introspect the
        #: backbone for it — `MonaiUNetWithDeepSupervision` has no `.decoder`, so
        #: reading `decoder.seg_layers[-1].out_channels` fails for every `unet`
        #: row. `build_row_model` already checks the row against
        #: `len(dataset_json["labels"])`, so the two sources agree by
        #: construction; the row is the one that is declared.
        self.decoder_out_channels = int(
            decoder_out_channels if decoder_out_channels is not None
            else len(dataset_json["labels"]))

        self.global_branch = get_umamba_enc_3d_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels=num_input_channels,
            deep_supervision=deep_supervision).to(device)
        self.local_branch = (build_local_encoder(self.local_patch_size).to(device)
                             if self.has_local_branch else None)

        # Latents come from a real forward pass; the Mamba kernels are CUDA-only
        # and the bottleneck shape is not worth deriving by hand.
        (self.global_latent, self.local_latent,
         self.global_channels, self.local_channels) = self._probe_latents(device)
        self.normalise_head = bool(normalise_head)
        self.global_head = _regression_head(self.global_latent, self.global_channels,
                                            normalise=self.normalise_head).to(device)
        self.local_head = (_regression_head(self.local_latent, self.local_channels,
                                            normalise=self.normalise_head).to(device)
                           if self.has_local_branch else None)

        self.blend = nn.Linear(len(self.blend_inputs), 1).to(device)
        nn.init.zeros_(self.blend.bias)
        # Registered last, so it sees the finished module tree.
        self.mamba = MambaActivationTracker(self)

    @torch.no_grad()
    def _probe_latents(self, device: torch.device) -> tuple[int, int, int, int]:
        """Flattened width and channel count for each bottleneck.

        The channel count is needed because the head now normalises before
        flattening, and like the width it is probed from a real forward rather
        than derived — the ceil-division the stride ladder performs is exactly
        what a hand derivation gets wrong.
        """
        g = self.global_branch.encoder(
            torch.zeros(1, 1, *self.global_patch_size, device=device))[-1]
        local_numel, local_channels = 0, 0
        if self.has_local_branch:
            l = self.local_branch(torch.zeros(1, 1, *self.local_patch_size, device=device))
            l = l[-1] if isinstance(l, (list, tuple)) else l
            local_numel, local_channels = int(l.numel()), int(l.shape[1])
            del l
        out = int(g.numel()), local_numel, int(g.shape[1]), local_channels
        del g
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out

    def forward(self, volume: torch.Tensor, patch: torch.Tensor | None,
                ga_scan: torch.Tensor, cl: torch.Tensor) -> dict[str, object]:
        if tuple(volume.shape[2:]) != self.global_patch_size:
            raise ModelError(
                f"global input is {tuple(volume.shape[2:])}, expected "
                f"{self.global_patch_size}. This model does not resize; a silent "
                "interpolation here would change what the network sees.")
        self.mamba.reset()
        skips = self.global_branch.encoder(volume)
        segmentation = self.global_branch.decoder(skips)
        global_regression = self.global_head(skips[-1])

        local_regression = None
        if self.has_local_branch:
            if patch is None:
                raise ModelError("this row has a local branch and was given no patch")
            if tuple(patch.shape[2:]) != self.local_patch_size:
                raise ModelError(
                    f"local patch is {tuple(patch.shape[2:])}, expected "
                    f"{self.local_patch_size}")
            local_features = self.local_branch(patch)
            if isinstance(local_features, (list, tuple)):
                local_features = local_features[-1]
            local_regression = self.local_head(local_features)

        # Assembled FROM `blend_inputs`, not from a literal list, so the width
        # and the order cannot disagree.
        available = {"global_regression": global_regression,
                     "local_regression": local_regression,
                     "ga_scan": ga_scan, "cl": cl}
        blended = self.blend(torch.cat([available[k] for k in self.blend_inputs], dim=-1))
        return {"ga_del": blended, "segmentation": segmentation,
                "global_regression": global_regression,
                "local_regression": local_regression,
                "mamba_activation": self.mamba.penalty()}

    def global_window_forward(self, window: torch.Tensor, want_segmentation: bool
                              ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One global window → (regression, highest-resolution seg logits or None).

        The seam the validation pass uses instead of the backbone's internals.
        `PuumaNet` splits into `encoder`/`decoder`; `UnetRowNet` does not — it is one
        MONAI `UNet` with a bottleneck hook. Both implement this, so the validation window
        loop works for every row without knowing which it has.
        """
        skips = self.global_branch.encoder(window)
        regression = self.global_head(skips[-1])
        logits = None
        if want_segmentation:
            logits = self.global_branch.decoder(skips)
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
        del skips
        return regression, logits

    def shared_encoder_parameters(self) -> list[nn.Parameter]:
        """The parameters both tasks push gradients into.

        The global encoder is what the segmentation decoder and the global
        regression head have in common; the decoder and the heads are each used by
        one task only, so a gradient norm taken over them would not answer the
        question being asked.
        """
        return [p for p in self.global_branch.encoder.parameters() if p.requires_grad]

    def parameter_report(self) -> ParameterReport:
        """Counts read off this instance."""
        n = lambda m: sum(p.numel() for p in m.parameters())
        zero = lambda m: n(m) if m is not None else 0
        return ParameterReport(
            total=n(self), global_branch=n(self.global_branch),
            local_branch=zero(self.local_branch), global_head=n(self.global_head),
            local_head=zero(self.local_head), blend=n(self.blend),
            global_latent=self.global_latent, local_latent=self.local_latent)

    @torch.no_grad()
    def widths(self) -> list[int]:
        """Encoder widths read off the running model, not off the JSON.

        The channel dimension of each skip is the width of that stage, so a probe
        forward reports what the network actually built rather than what the
        plans asked for.
        """
        device = next(self.parameters()).device
        skips = self.global_branch.encoder(
            torch.zeros(1, 1, *self.global_patch_size, device=device))
        out = [int(s.shape[1]) for s in skips]
        del skips
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out


def build_row_model(row, plans_manager: PlansManager, dataset_json: dict,
                    configuration_manager: ConfigurationManager,
                    local_patch_size: Sequence[int],
                    device: torch.device | str = "cuda",
                    deep_supervision: bool = True,
                    normalise_head: bool = True) -> nn.Module:
    """The one place a configuration row becomes a network.

    A row is data (`puuma.config`); this reads its dimensions and builds. There is
    no `if row.name == ...` anywhere — the blend width and order come from
    `row.blend_inputs`, the local branch exists iff that tuple asks for it, and the
    backbone is a switch on a declared field: `monai_unet` builds `UnetRowNet`,
    `umamba` builds `PuumaNet`, and `umamba_noskip` builds `PuumaNet` and then
    swaps the global decoder for `NoSkipUNetResDecoder` (the `no_skips` row;
    the encoder, both heads and the blend are untouched, so the regression head
    still reads `self.global_branch.encoder(window)[-1]`).

    The decoder width is checked rather than set: nnU-Net derives it from
    `len(dataset_json["labels"])`, so a row asking for a different number is
    asking for a label set this dataset does not carry, and that must fail here
    rather than as an out-of-bounds scatter index later.
    """
    row.require_implemented()
    declared = len(dataset_json["labels"])
    if row.decoder_out_channels != declared:
        raise ModelError(
            f"row {row.name!r} wants {row.decoder_out_channels} decoder channels but "
            f"dataset.json declares {declared} labels {sorted(dataset_json['labels'])}. "
            "nnU-Net derives the decoder width from the label set; supply the "
            "row's label set to the LabelManager rather than overriding the width, "
            "or the target's highest label becomes an out-of-bounds scatter index "
            "and the Dice loss faults the CUDA context instead of raising.")
    if row.backbone == "monai_unet":
        from puuma.unet_baseline import UnetRowNet
        return UnetRowNet(configuration_manager.patch_size,
                          blend_inputs=row.blend_inputs,
                          out_channels=row.decoder_out_channels,
                          deep_supervision=deep_supervision, device=device,
                          normalise_head=normalise_head)
    model = PuumaNet(plans_manager, dataset_json, configuration_manager,
                     local_patch_size, deep_supervision=deep_supervision,
                     device=device, blend_inputs=row.blend_inputs,
                     normalise_head=normalise_head,
                     decoder_out_channels=row.decoder_out_channels)
    if row.backbone == "umamba_noskip":
        # After construction, deliberately: `_probe_latents` and the
        # `MambaActivationTracker` have already walked the tree, and neither
        # reads the decoder — the probe uses the encoder's bottleneck and the
        # tracker hooks `MambaLayer`s, of which no decoder has any.
        from puuma.noskip_decoder import swap_in_noskip_decoder
        swap_in_noskip_decoder(model.global_branch, device)
    return model
