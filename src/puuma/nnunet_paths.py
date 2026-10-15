"""Rebind nnU-Net's three hardcoded data roots inside the write root.

U-Mamba's vendored `nnunetv2/paths.py:44-48` hardcodes `nnUNet_raw`,
`nnUNet_preprocessed` and `nnUNet_results` to `<clone>/data/...` and never
reads the environment — the `os.environ.get(...)` calls are commented out. 43
modules bind those names at import time and none looks them up late, so a
rebind only reaches modules not yet imported; nnU-Net writes to these roots.

Import order is the entire mechanism: call :func:`bind_nnunet_paths` before
anything else in `nnunetv2` is imported. A single stray `from nnunetv2...`
above it captures the shipped values, and :func:`assert_no_stale_bindings`
turns that silent failure into a stop. Importing this module is itself safe:
`nnunetv2/__init__.py` is empty and `nnunetv2/paths.py` only computes strings.
The guards raise (never `assert`); stale bindings are reported, not repaired.
"""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

#: Resolved so the containment check cannot be defeated by a
#: symlink or a `..` component.
WRITE_ROOT: Path = Path(__file__).resolve().parents[2]

#: The three names `nnunetv2/paths.py` defines and 43 modules copy.
NNUNET_PATH_NAMES: tuple[str, str, str] = (
    "nnUNet_raw",
    "nnUNet_preprocessed",
    "nnUNet_results",
)


class NnunetPathsError(RuntimeError):
    """A paths guard failed. Always fatal — never caught and continued from."""


class ResultsRoot(enum.Enum):
    """Which results tree a process writes to.

    There is deliberately no default. Smoke runs stay out of
    the CAMPAIGN results tree because a 2-epoch run there would be skipped as
    "already complete" and would enter the final results. A shim that picked one
    silently would destroy that separation, so the caller must name it and
    :func:`bind_nnunet_paths` accepts nothing else.
    """

    CAMPAIGN = "campaign"
    SMOKE = "smoke"


@dataclass(frozen=True)
class NnunetPaths:
    """The three roots this process bound, and which results tree it chose."""

    raw: Path
    preprocessed: Path
    results: Path
    results_root: ResultsRoot

    def as_dict(self) -> dict[str, Path]:
        """Mapping from nnU-Net's own variable names to the bound paths."""
        return {
            "nnUNet_raw": self.raw,
            "nnUNet_preprocessed": self.preprocessed,
            "nnUNet_results": self.results,
        }


def _suppress_bytecode() -> None:
    """Nothing is written outside the write root, caches included.

    Here rather than in an entry point because this module is the one thing every
    entry point imports first and before `nnunetv2` — which is the import that
    causes the interpreter to write `.pyc` files into the read-only clone.
    """
    from puuma.runtime import suppress_bytecode_in_readonly_trees
    suppress_bytecode_in_readonly_trees()


def intended_paths(results_root: ResultsRoot) -> NnunetPaths:
    """The paths this project intends, without touching `nnunetv2` or the disk.

    `nnUNet_raw` and `nnUNet_preprocessed` are shared: they hold the dataset and
    nnU-Net's preprocessed arrays, which do not differ between a smoke run and a
    full run. Only `nnUNet_results` differs.
    """
    if not isinstance(results_root, ResultsRoot):
        raise TypeError(
            f"results_root must be a ResultsRoot, not {type(results_root).__name__}. "
            f"Pass ResultsRoot.CAMPAIGN or ResultsRoot.SMOKE explicitly."
        )
    data = WRITE_ROOT / "data"
    results = (WRITE_ROOT / "outputs" / "runs" if results_root is ResultsRoot.CAMPAIGN
               else WRITE_ROOT / "outputs" / "smoke")
    return NnunetPaths(
        raw=data / "nnUNet_raw",
        preprocessed=data / "nnUNet_preprocessed",
        results=results,
        results_root=results_root,
    )


def assert_all_under_write_root(paths: NnunetPaths) -> None:
    """Guard 2: all three roots resolve under the write root.

    Runs before any directory is created, so a rejected path is never
    brought into existence.

    :raises NnunetPathsError: if any root resolves outside the write root.
    """
    root = WRITE_ROOT.resolve()
    outside = []
    for name, path in paths.as_dict().items():
        # `resolve()` on both sides, then a path-relationship test. A string
        # prefix test would accept `<write root>_elsewhere`.
        if not path.resolve().is_relative_to(root):
            outside.append(f"{name} -> {path.resolve()}")
    if outside:
        raise NnunetPathsError(
            "nnU-Net path(s) resolve outside the write root "
            f"{root}, which the write-root rule forbids: " + "; ".join(outside)
        )


def _module_globals(module: ModuleType | None) -> dict[str, object] | None:
    """A module's `__dict__`, or `None` if it has none.

    Read through `__dict__` rather than `getattr` so a module defining
    `__getattr__` cannot run code, or synthesise an attribute, during the sweep.
    """
    if module is None:
        return None
    namespace = getattr(module, "__dict__", None)
    return namespace if isinstance(namespace, dict) else None


def assert_no_stale_bindings(paths: NnunetPaths) -> None:
    """Guard 1: no module holds a stale copy of the three names.

    Walks `sys.modules` for any module holding `nnUNet_raw`,
    `nnUNet_preprocessed` or `nnUNet_results` and requires every copy to equal
    the intended value. A module that imported `nnunetv2.paths` before
    :func:`bind_nnunet_paths` ran holds the shipped value, and nothing else can
    see that — the run would simply write to the clone.

    :raises NnunetPathsError: listing every module and name that disagrees.
    """
    expected = {name: str(path) for name, path in paths.as_dict().items()}
    stale: list[str] = []
    for module_name in list(sys.modules):
        namespace = _module_globals(sys.modules.get(module_name))
        if namespace is None:
            continue
        for name in NNUNET_PATH_NAMES:
            if name not in namespace:
                continue
            found = namespace[name]
            if str(found) != expected[name]:
                stale.append(f"{module_name}.{name} = {found!r} (expected {expected[name]!r})")
    if stale:
        raise NnunetPathsError(
            "stale nnU-Net path binding(s) found — something imported nnunetv2 "
            "before bind_nnunet_paths() ran, so it captured the shipped values "
            "and would write outside the write root. Move the shim import above "
            "every nnunetv2 import in this entry point. Offenders: "
            + "; ".join(sorted(stale))
        )


def bind_nnunet_paths(results_root: ResultsRoot, *, create: bool = True) -> NnunetPaths:
    """Rebind nnU-Net's three roots, then run both guards.

    Must be called before anything else in `nnunetv2` is imported.

    :param results_root: `ResultsRoot.CAMPAIGN` or `ResultsRoot.SMOKE`. No
        default — see :class:`ResultsRoot`.
    :param create: create the three directories. The containment guard runs
        first either way, so a rejected path is never created.
    :returns: the paths that were bound.
    :raises NnunetPathsError: if either guard fails.
    :raises TypeError: if `results_root` is not a :class:`ResultsRoot`.
    """
    # Before anything from nnunetv2 is imported: that import is what makes the
    # interpreter write .pyc files into the read-only clone.
    _suppress_bytecode()

    paths = intended_paths(results_root)

    # Guard 2 first: never create a directory we are about to reject.
    assert_all_under_write_root(paths)

    if create:
        for path in paths.as_dict().values():
            path.mkdir(parents=True, exist_ok=True)

    # Safe to import: nnunetv2/__init__.py is empty and paths.py only builds
    # strings, so this pulls in nothing that binds the names.
    import nnunetv2.paths as nnunet_paths

    nnunet_paths.nnUNet_raw = str(paths.raw)
    nnunet_paths.nnUNet_preprocessed = str(paths.preprocessed)
    nnunet_paths.nnUNet_results = str(paths.results)

    assert_no_stale_bindings(paths)
    return paths


def verify_bindings(paths: NnunetPaths) -> None:
    """Re-run both guards. Call once more after all imports, before any run.

    :func:`bind_nnunet_paths` checks the state at bind time; this checks it
    again once the entry point has finished importing, which is when a late
    stray import would have done its damage.

    :raises NnunetPathsError: if either guard fails.
    """
    assert_all_under_write_root(paths)
    assert_no_stale_bindings(paths)
