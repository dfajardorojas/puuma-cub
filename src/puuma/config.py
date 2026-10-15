"""The configuration rows, as data rather than as branches in code.

One YAML per row under `configs/`. A row names what it removes and every
dimension that follows — the blend inputs in order, the decoder's output
channels, the label map, which backbone, and whether the sampler is
category-balanced; the validation below re-derives what it can rather than
trusting the file.

The blend order is load-bearing: a silent permutation is invisible in the loss
and changes what the layer means, so the order in the file is the order the
model concatenates in. A row that is declared but not implemented refuses at
load (:meth:`RowConfig.require_implemented`), not hours into a run.

No `nnunetv2` import, so import-order safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

#: Every blend input this design knows, in the fixed canonical order. A row's
#: `blend_inputs` must be a subsequence of this: dropping one is a configuration,
#: reordering them is a mistake.
CANONICAL_BLEND_INPUTS: tuple[str, ...] = (
    "global_regression", "local_regression", "ga_scan", "cl")
#: The named backbones. `monai_unet` is the `unet` row's external baseline;
#: `umamba_noskip` is `umamba` with the global decoder's skip connections removed
#: (the `no_skips` row; `puuma/noskip_decoder.py`).
BACKBONES = ("umamba", "monai_unet", "umamba_noskip")
TIERS = ("primary", "secondary", "exploratory")
#: Keys a row file may carry. An unknown key raises rather than being ignored —
#: a typo'd dimension that is silently dropped is the failure this guards.
_ALLOWED = {"name", "tier", "removes", "question", "backbone", "blend_inputs",
            "decoder_out_channels", "label_map", "balanced_sampler", "implemented",
            "unimplemented_reason", "dataset"}
_REQUIRED = {"name", "tier", "backbone", "blend_inputs", "decoder_out_channels",
             "label_map", "balanced_sampler", "implemented", "dataset"}

#: Where the rows live.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class ConfigError(RuntimeError):
    """Raised for a row that is malformed, unknown, or not runnable."""


@dataclass(frozen=True)
class RowConfig:
    """One row of the configuration table, with every dimension it changes."""

    name: str
    tier: str
    backbone: str
    blend_inputs: tuple[str, ...]
    decoder_out_channels: int
    label_map: dict[str, int]
    balanced_sampler: bool
    implemented: bool
    dataset: str = "Dataset001_PlacentaBrainCL"
    removes: str | None = None
    question: str = ""
    unimplemented_reason: str = ""

    # -- derived, so no caller re-derives them differently ------------------
    @property
    def has_local_branch(self) -> bool:
        """The local branch exists iff its output is blended.

        Not a separate flag: a local branch whose output goes nowhere would be
        trained by no gradient and is not a state this design has.
        """
        return "local_regression" in self.blend_inputs

    @property
    def has_global_branch(self) -> bool:
        return "global_regression" in self.blend_inputs

    @property
    def blend_width(self) -> int:
        return len(self.blend_inputs)

    @property
    def branches(self) -> tuple[str, ...]:
        out = []
        if self.has_global_branch:
            out.append("global")
        if self.has_local_branch:
            out.append("local")
        return tuple(out)

    def require_implemented(self) -> None:
        """Refuse a declared-but-unbuilt row before anything trains."""
        if not self.implemented:
            raise ConfigError(
                f"config row {self.name!r} is declared but is not runnable.\n"
                f"{self.unimplemented_reason.strip()}\n"
                "Refusing at startup rather than at the first batch: the failure it "
                "would otherwise produce is a CUDA context fault, not an exception.")

    def describe(self) -> str:
        return (f"{self.name} [{self.tier}] dataset={self.dataset} "
                f"backbone={self.backbone} "
                f"branches={'+'.join(self.branches)} "
                f"blend=Linear({self.blend_width},1){list(self.blend_inputs)} "
                f"out_channels={self.decoder_out_channels} "
                f"balanced_sampler={self.balanced_sampler}")


def _validate(raw: dict, source: Path) -> RowConfig:
    keys = set(raw)
    unknown = keys - _ALLOWED
    if unknown:
        raise ConfigError(f"{source.name}: unknown key(s) {sorted(unknown)}; "
                          f"allowed are {sorted(_ALLOWED)}")
    missing = _REQUIRED - keys
    if missing:
        raise ConfigError(f"{source.name}: missing required key(s) {sorted(missing)}")
    if raw["name"] != source.stem:
        raise ConfigError(
            f"{source.name}: declares name {raw['name']!r}; the file name is the row "
            "name and the two must agree, or `--config <name>` reaches the wrong file")
    if raw["tier"] not in TIERS:
        raise ConfigError(f"{source.name}: tier {raw['tier']!r} not one of {TIERS}")
    if raw["backbone"] not in BACKBONES:
        raise ConfigError(f"{source.name}: backbone {raw['backbone']!r} not one of {BACKBONES}")

    inputs = tuple(raw["blend_inputs"])
    unknown_inputs = [i for i in inputs if i not in CANONICAL_BLEND_INPUTS]
    if unknown_inputs:
        raise ConfigError(f"{source.name}: unknown blend input(s) {unknown_inputs}")
    if len(set(inputs)) != len(inputs):
        raise ConfigError(f"{source.name}: duplicate blend inputs {inputs}")
    order = [CANONICAL_BLEND_INPUTS.index(i) for i in inputs]
    if order != sorted(order):
        raise ConfigError(
            f"{source.name}: blend inputs {list(inputs)} are out of the canonical "
            f"order {list(CANONICAL_BLEND_INPUTS)}. Dropping one is a configuration; "
            "reordering them silently changes what the layer means.")
    if "global_regression" not in inputs:
        raise ConfigError(f"{source.name}: every row keeps the global branch")

    labels = dict(raw["label_map"])
    if raw["decoder_out_channels"] != len(labels):
        raise ConfigError(
            f"{source.name}: decoder_out_channels is {raw['decoder_out_channels']} but "
            f"the label map has {len(labels)} entries {sorted(labels)}. nnU-Net derives "
            "the decoder width from len(labels), so a disagreement here is a "
            "two-channel decoder fed a three-value target — an out-of-bounds scatter "
            "index and a CUDA fault, not an exception.")
    if sorted(labels.values()) != list(range(len(labels))):
        raise ConfigError(
            f"{source.name}: label values {sorted(labels.values())} must be contiguous "
            "from 0; nnU-Net indexes output channels by label value")
    if labels.get("background") != 0:
        raise ConfigError(f"{source.name}: background must be label 0")
    if not raw["implemented"] and not raw.get("unimplemented_reason", "").strip():
        raise ConfigError(
            f"{source.name}: implemented is false with no unimplemented_reason. A row "
            "that refuses must say what is undecided, or the refusal is unactionable.")

    return RowConfig(
        name=raw["name"], tier=raw["tier"], backbone=raw["backbone"],
        blend_inputs=inputs, decoder_out_channels=int(raw["decoder_out_channels"]),
        label_map=labels, balanced_sampler=bool(raw["balanced_sampler"]),
        implemented=bool(raw["implemented"]), dataset=str(raw["dataset"]),
        removes=raw.get("removes"),
        question=raw.get("question", "") or "",
        unimplemented_reason=raw.get("unimplemented_reason", "") or "")


def load_row(name: str, config_dir: Path | None = None) -> RowConfig:
    """One row by name. An unknown name lists what exists rather than guessing."""
    directory = Path(config_dir or CONFIG_DIR)
    path = directory / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.yaml"))
        raise ConfigError(
            f"no config row {name!r} in {directory}. Available: {available}")
    return _validate(yaml.safe_load(path.read_text()) or {}, path)


def load_all_rows(config_dir: Path | None = None) -> dict[str, RowConfig]:
    """Every row, validated. Used by the tests and by `--list-configs`."""
    directory = Path(config_dir or CONFIG_DIR)
    return {p.stem: _validate(yaml.safe_load(p.read_text()) or {}, p)
            for p in sorted(directory.glob("*.yaml"))}


def balanced_sampling_probabilities(categories: Iterable[str]) -> list[float]:
    """The category-balanced sampler, as the per-case probability vector.

    `nnUNetDataLoaderBase` forwards `sampling_probabilities` to batchgenerators,
    which uses it at `data_loader.py:188` as `np.random.choice(..., p=...)`. So
    balancing is per-case probabilities and never row replication — the sampler
    cannot change training volume: 250 iterations per epoch is fixed regardless.

    Each category receives equal total mass, split evenly among its cases.
    """
    cats = list(categories)
    if not cats:
        raise ConfigError("cannot build a sampling vector over zero cases")
    counts: dict[str, int] = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1
    share = 1.0 / len(counts)
    weights = [share / counts[c] for c in cats]
    total = sum(weights)
    return [w / total for w in weights]
