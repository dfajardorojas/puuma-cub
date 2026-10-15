"""An in-RAM case cache for `nnUNetDataset`, using the framework's own hook.

`load_case` consults `'open_data_file'` and `'open_seg_file'` in each dataset
entry before touching the filesystem; filling those keys with decompressed
arrays removes the per-draw decompression of the `.npz` archives without
overriding any framework code and without writing `.npy` files into the
preprocessed store. Build the cache in the parent process, before the augmenter
forks its workers, so the arrays are shared copy-on-write.

Imports `nnunetv2` at module level — the shim must be bound first.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class CacheError(RuntimeError):
    """Raised when the cache cannot be built as asked."""


#: Refuse to build a cache larger than this fraction of currently available
#: RAM, so an oversized cohort fails with an exception rather than driving the
#: machine into swap.
MAX_FRACTION_OF_AVAILABLE_RAM = 0.5


def available_ram_bytes() -> int:
    """`MemAvailable` from `/proc/meminfo`, in bytes."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise CacheError("could not read MemAvailable from /proc/meminfo")


def cache_size_bytes(folder: str, case_identifiers: Sequence[str]) -> int:
    """Decompressed size of the cases, without holding them."""
    total = 0
    for key in case_identifiers:
        with np.load(f"{folder}/{key}.npz") as archive:
            total += archive["data"].nbytes + archive["seg"].nbytes
    return total


class CachedNnunetDataset(nnUNetDataset):
    """`nnUNetDataset` with every case decompressed once, up front.

    Build it before the augmenter, in the parent process, so the arrays are
    inherited by every forked worker copy-on-write.

    The arrays are marked read-only: a transform that wrote through to a cached
    array would corrupt every later draw of that case, and the flag turns that
    into an exception at the point of the write.
    """

    def __init__(self, *args, verbose: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache_bytes = 0
        self.cache_seconds = 0.0
        keys = list(self.dataset.keys())
        folder = str(self.dataset[keys[0]]["data_file"]).rsplit("/", 1)[0]
        needed = cache_size_bytes(folder, keys)
        available = available_ram_bytes()
        if needed > MAX_FRACTION_OF_AVAILABLE_RAM * available:
            raise CacheError(
                f"caching {len(keys)} cases needs {needed/1e9:.2f} GB and only "
                f"{available/1e9:.2f} GB is available; refusing to build a cache larger "
                f"than {MAX_FRACTION_OF_AVAILABLE_RAM:.0%} of it. Run without the cache, "
                "or unpack the dataset to .npy and rely on the page cache instead.")
        started = time.perf_counter()
        for key in keys:
            with np.load(self.dataset[key]["data_file"]) as archive:
                data, seg = archive["data"], archive["seg"]
            data.flags.writeable = False
            seg.flags.writeable = False
            # The framework's own keys: `load_case` consults these before it goes
            # to the filesystem, so nothing there is overridden.
            self.dataset[key]["open_data_file"] = data
            self.dataset[key]["open_seg_file"] = seg
            self.cache_bytes += data.nbytes + seg.nbytes
        self.cache_seconds = time.perf_counter() - started
        if verbose:
            print(f"  cached {len(keys)} cases, {self.cache_bytes/1e9:.2f} GB, "
                  f"in {self.cache_seconds:.1f} s", flush=True)

    @property
    def cached(self) -> bool:
        return all("open_data_file" in v for v in self.dataset.values())


def build_dataset(folder: str, case_identifiers: Sequence[str], *, cache: bool,
                  verbose: bool = False) -> nnUNetDataset:
    """The one place a dataset is constructed, cached or not.

    Both branches pass `folder_with_segs_from_previous_stage=None` and
    `num_images_properties_loading_threshold=0`, so the only difference between
    them is the cache.
    """
    cls = CachedNnunetDataset if cache else nnUNetDataset
    kwargs = {"verbose": verbose} if cache else {}
    return cls(str(folder), list(case_identifiers),
               folder_with_segs_from_previous_stage=None,
               num_images_properties_loading_threshold=0, **kwargs)
