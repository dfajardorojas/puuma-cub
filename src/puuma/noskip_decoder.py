# Modified copy of nnunetv2/nets/UMambaEnc_3d.py (lines 307-395) from
# U-Mamba, commit 28459e33ca03769800dd35e23c6e62491d1925b5, licensed
# under the Apache License, Version 2.0 (see LICENSE).
# Changes: `:349` — the first block of each stage takes `input_features_skip`
# channels, not `2 * input_features_skip`; `:382` — `x = torch.cat((x,
# skips[-(s+2)]), 1)` is deleted from `forward`.
"""The `no_skips` row's decoder: `UNetResDecoder` with the skip path removed.

A copy of `nnunetv2/nets/UMambaEnc_3d.py:307-395` with exactly two changes:

* `:349` — the first block of each stage takes `input_features_skip` channels,
  not `2 * input_features_skip`, because there is no concatenated skip to widen
  its input;
* `:382` — `x = torch.cat((x, skips[-(s+2)]), 1)` is deleted from `forward`.

Everything else is the clone's: the `forward(skips)` signature is kept (the
decoder consumes `skips[-1]` only), so callers handing over the full skip list
need no change, and the five deep-supervision `seg_layers` are unchanged in
count, channels and resolution. `compute_conv_feature_map_size` is not copied;
nothing in this repository calls it.
"""
from __future__ import annotations

import torch
from torch import nn

from nnunetv2.nets.UMambaEnc_3d import BasicBlockD, BasicResBlock, UpsampleLayer
from nnunetv2.utilities.network_initialization import InitWeights_He


class NoSkipUNetResDecoder(nn.Module):
    """`UNetResDecoder` without skip connections. See the module docstring."""

    def __init__(self, encoder, num_classes,
                 n_conv_per_stage, deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, \
            "n_conv_per_stage must have as many entries as we have " \
            "resolution stages - 1 (n_stages in encoder - 1), " \
            "here: %d" % n_stages_encoder

        stages = []
        upsample_layers = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=input_features_skip,   # PUUMA: was 2 * input_features_skip
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s - 1] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes,
                                              1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            # PUUMA: the clone concatenates skips[-(s+2)] here; removed.
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


def swap_in_noskip_decoder(global_branch: nn.Module,
                           device: torch.device | str) -> nn.Module:
    """Replace a built `UMambaEnc`'s decoder with the no-skip variant, in place.

    Every number is read off the built module, never retyped: `num_classes`
    from the last `seg_layer`'s out-channels, `deep_supervision` from the flag the
    builder set, and the effective per-stage conv counts from the built stages —
    `UMambaEnc.__init__` forces the deeper half of the plans'
    `n_conv_per_stage_decoder` to 1 before `UNetResDecoder` sees it, so the plans'
    own list is the wrong source ([2, 2, 2, 1, 1] effective against [2, 2, 2, 2, 2]
    declared).

    The new decoder is re-initialised with `InitWeights_He(1e-2)`, the same
    initialiser `get_umamba_enc_3d_from_plans` applied to the whole network — the
    builder's pass initialised the decoder being discarded, not this one.
    """
    old = global_branch.decoder
    effective_n_conv = [len(stage) for stage in old.stages]
    decoder = NoSkipUNetResDecoder(
        global_branch.encoder,
        num_classes=old.seg_layers[-1].out_channels,
        n_conv_per_stage=effective_n_conv,
        deep_supervision=old.deep_supervision,
    )
    decoder.apply(InitWeights_He(1e-2))
    global_branch.decoder = decoder.to(device)
    return global_branch
