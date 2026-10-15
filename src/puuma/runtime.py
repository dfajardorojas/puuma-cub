"""The knobs that decide how a run is executed, in one place.

Nothing here is science: this module fixes how many processes shovel data,
whether the cases are held in RAM, and how batches reach the GPU.
:data:`BATCH_SIZE` is re-exported from here so that no script carries its own
copy.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

#: The mask the parent holds, captured at import — before any fork, and before
#: libiomp has had the chance to narrow a child. Workers restore this.
_PARENT_AFFINITY = frozenset(os.sched_getaffinity(0))

#: The planner's own value for this dataset and configuration. An epoch is
#: fixed at 250 iterations regardless, so this changes the work per step and
#: the gradient's sample size, not the number of optimiser updates.
BATCH_SIZE = 2

#: Augmentation worker processes. `get_allowed_n_proc_DA()` is a hardcoded
#: hostname lookup table for DKFZ's cluster (`utilities/default_n_proc_DA.py`)
#: that falls through to a default of 12 for any host it does not recognise —
#: not a measurement of the local machine — so the count is pinned here.
AUGMENTATION_WORKERS = 8

#: Queue depth shared across all workers (`num_cached`).
PREFETCH_DEPTH = 6

#: Pinning happens in the augmenter's single `results_loop` thread, on the
#: whole batch, before the training loop can see it. Off.
PIN_MEMORY = False

#: Hold every training case decompressed in RAM. See :mod:`puuma.nnunet_cache`.
#: Built in the parent process before the augmenter forks, so the workers inherit
#: it copy-on-write and it is paid for once.
CACHE_CASES_IN_RAM = True

#: Rotation half-range in degrees for the training augmentation, on all three
#: axes; nnU-Net's own value.
ROTATION_DEGREES = 30.0

#: The scale range for the same transform. nnU-Net hardcodes (0.7, 1.4) inside
#: `get_training_transforms` — it is not a parameter — so this is applied to
#: the constructed `SpatialTransform`.
SCALE_RANGE = (0.7, 1.4)

#: `torch.multiprocessing`'s tensor sharing strategy, or None to leave the
#: default (`file_descriptor`). `file_system` leaks `/dev/shm` entries if a
#: worker dies without cleaning up, so it is not switched on by default.
SHARING_STRATEGY: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """One resolved set of execution knobs, so a script passes one object."""

    batch_size: int = BATCH_SIZE
    workers: int = AUGMENTATION_WORKERS
    prefetch_depth: int = PREFETCH_DEPTH
    pin_memory: bool = PIN_MEMORY
    cache_cases_in_ram: bool = CACHE_CASES_IN_RAM
    sharing_strategy: str | None = SHARING_STRATEGY
    #: The augmentation magnitudes, here so a run records them and so they can be
    #: changed without touching model or transform code.
    rotation_degrees: float = ROTATION_DEGREES
    scale_range: tuple = SCALE_RANGE

    def rotation_for_DA(self) -> dict:
        """The dict nnU-Net's `get_training_transforms` takes as a parameter.

        The framework does not fix these: `nnUNetTrainer.py:387-431` derives them
        from the plans in `configure_rotation_dummyDA_mirroring_and_inital_patch_size`
        and passes them in. Setting them fills a parameter the framework leaves open.
        """
        import numpy as np
        half = float(np.deg2rad(self.rotation_degrees))
        return {axis: (-half, half) for axis in ("x", "y", "z")}

    def describe(self) -> str:
        return (f"batch {self.batch_size}, {self.workers} workers, "
                f"prefetch {self.prefetch_depth}, "
                f"pin_memory {'on' if self.pin_memory else 'off'}, "
                f"RAM cache {'on' if self.cache_cases_in_ram else 'off'}, "
                f"sharing {self.sharing_strategy or 'default (file_descriptor)'}, "
                f"rotation ±{self.rotation_degrees:g}°, "
                f"scale {tuple(self.scale_range)}")

    def apply_sharing_strategy(self) -> None:
        """Set the torch sharing strategy, if one was asked for.

        Must run before the augmenter forks its workers.
        """
        if self.sharing_strategy is not None:
            import torch.multiprocessing as torch_mp
            torch_mp.set_sharing_strategy(self.sharing_strategy)


def lift_fork_affinity() -> tuple[int, int]:
    """Undo libiomp's fork-time CPU pinning. Call from inside the worker.

    Importing torch loads Intel's `libiomp5`, which installs a `pthread_atfork`
    handler that pins every forked child to one physical core — the same one
    for every worker — so augmentation workers time-share a single core.
    Restoring the parent's mask inside the worker does not depend on import
    order, unlike `OMP_PROC_BIND=FALSE`, which libiomp reads only when it
    initialises.

    `KMP_AFFINITY=disabled` is NOT an equivalent and must not be used: it kills
    every worker with `OMP: Error #13: Assertion failure at
    kmp_affinity.cpp(4314)` under real load.

    :returns: `(before, after)` CPU counts, so a caller can assert the lift.
    """
    before = len(os.sched_getaffinity(0))
    os.sched_setaffinity(0, _PARENT_AFFINITY)
    return before, len(os.sched_getaffinity(0))


def suppress_bytecode_in_readonly_trees() -> None:
    """Stop Python writing `__pycache__` into the read-only clone.

    Nothing may be written outside the write root — caches included. `nnunetv2`
    is installed editable, so it imports from the clone, and the interpreter
    writes `.pyc` files next to its sources whenever a module is imported
    without a current cache.

    Both halves are needed: `sys.dont_write_bytecode` stops this interpreter,
    and the environment variable stops the forked augmentation workers and any
    subprocess. Called by the paths shim, which every entry point imports first
    and before `nnunetv2`.
    """
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def physical_cores() -> int:
    """Physical cores, not the SMT count `os.cpu_count()` reports.

    `get_allowed_n_proc_DA` caps its lookup-table answer with `os.cpu_count()`,
    which counts hyperthreads.
    """
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
        pairs = set()
        physical = core = None
        for line in text.splitlines():
            if line.startswith("physical id"):
                physical = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
                pairs.add((physical, core))
        return len(pairs) if pairs else (os.cpu_count() or 1)
    except OSError:
        return os.cpu_count() or 1
