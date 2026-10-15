"""``nnUNetDataLoader3D`` with the per-scan scalars attached.

Imports `nnunetv2` at module level, so the paths shim must be bound before this
module is imported; the transform and the scalar table live in
:mod:`puuma.dataloader`, which has no `nnunetv2` dependency.

The global window is nnU-Net's own: ``get_bbox`` places it with the stock
foreground oversampling (`base_data_loader.py:41-49`, `:63-80`, called from
`data_loader_3d.py:17` and `:26`), and the augmenter applies the transform
chain afterwards. The only addition is three scalars per item, which ride the
batch dictionary through the chain unchanged.
"""

from __future__ import annotations

from typing import Sequence

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D

from puuma.dataloader import SCALAR_KEYS, DataloaderError, ScalarTable, Standardiser
from puuma.runtime import lift_fork_affinity


class PuumaDataLoader3D(nnUNetDataLoader3D):
    """``nnUNetDataLoader3D`` plus GA at scan, CL and ``GA_del``.

    The scalars are looked up from :class:`~puuma.dataloader.ScalarTable` by the
    case identifiers the base class already reports in ``batch['keys']``. A
    missing scalar raises.
    """

    def __init__(self, *args, scalar_table: ScalarTable,
                 standardiser: Standardiser | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: The per-fold z-scoring, fitted on the TRAINING rows of this fold. The
        #: validation loader is given the SAME object, never one refitted on its
        #: own split.
        self.standardiser = standardiser
        if not isinstance(scalar_table, ScalarTable):
            raise DataloaderError("scalar_table must be a ScalarTable, not a bare mapping")
        self.scalar_table = scalar_table
        self._affinity_lifted = False
        missing = [k for k in self.indices if k not in scalar_table]
        if missing:
            raise DataloaderError(
                f"{len(missing)} of {len(self.indices)} cases in this split have no scalars, "
                f"first few: {missing[:5]}. Checked at construction so a run cannot start "
                "and fail hours later on one batch.")

    def generate_train_batch(self) -> dict:
        # Once per process, in the worker: libiomp's fork handler pins each
        # forked augmentation worker to one physical core. Lifted here because
        # this is the first project code to run inside the worker, and an
        # environment variable would depend on import order.
        if not self._affinity_lifted:
            self._affinity_lifted = True
            lift_fork_affinity()
        batch = super().generate_train_batch()
        batch.update(self.scalar_table.columns(batch["keys"], self.standardiser))
        return batch


__all__ = ["PuumaDataLoader3D", "SCALAR_KEYS"]
