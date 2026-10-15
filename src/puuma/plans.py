"""The named plans configuration.

Builds a copy of the planner's plans dict carrying one extra configuration,
``3d_fullres_puuma``, which inherits from ``3d_fullres`` and overrides only the
two U-Net width keys; ``nnUNetPlans.json`` is never edited. ``data_identifier``
is inherited, so the configuration reads the existing preprocessed folder and
triggers no re-preprocessing. No `nnunetv2` import, so import-order safe.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

#: The configuration name every run trains under.
CONFIGURATION = "3d_fullres_puuma"
PARENT_CONFIGURATION = "3d_fullres"
#: The fixed widths. The only two keys this configuration sets.
UNET_BASE_NUM_FEATURES = 16
UNET_MAX_NUM_FEATURES = 160


class PlansError(RuntimeError):
    """Raised when the plans lack the expected structure."""


def add_puuma_configuration(plans: dict) -> dict:
    """Return a copy of ``plans`` carrying the fixed configuration.

    The input is not mutated, so a caller holding the planner's own dict keeps it.
    """
    out = copy.deepcopy(plans)
    configurations = out["configurations"]
    if PARENT_CONFIGURATION not in configurations:
        raise PlansError(
            f"the plans have no {PARENT_CONFIGURATION!r} to inherit from; "
            f"available: {sorted(configurations)}")
    if CONFIGURATION in configurations:
        raise PlansError(
            f"{CONFIGURATION!r} already exists in these plans. This function adds it; "
            "it does not silently overwrite a configuration someone else defined.")
    parent = configurations[PARENT_CONFIGURATION]
    for key in ("data_identifier", "patch_size", "spacing"):
        if key not in parent:
            raise PlansError(f"{PARENT_CONFIGURATION!r} has no {key!r}")
    configurations[CONFIGURATION] = {
        "inherits_from": PARENT_CONFIGURATION,
        "UNet_base_num_features": UNET_BASE_NUM_FEATURES,
        "unet_max_num_features": UNET_MAX_NUM_FEATURES,
    }
    return out


def load_plans_with_puuma_configuration(plans_file: Path) -> dict:
    """Read the planner's plans and return a copy carrying the fixed configuration."""
    return add_puuma_configuration(json.loads(Path(plans_file).read_text()))
