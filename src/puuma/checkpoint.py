"""Saving and resuming a run.

:class:`~puuma.record.RunRecord` is serialised into the checkpoint directly,
beside the model, optimiser, scaler, scheduler counter, selector and RNG
streams. The full history survives a save/load cycle; the EMA resumes with its
state, not reinitialised to the first post-resume value; the selector resumes
knowing the current best and its epoch, so a resumed run cannot re-select a
worse checkpoint; and the detectors see the whole history rather than the
fragment since resume.

No `nnunetv2` import, so import-order safe.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

#: Bumped when the payload's shape changes, so an old checkpoint fails loudly
#: rather than deep inside `RunRecord.load_state_dict`.
CHECKPOINT_FORMAT = 4


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be resumed as written."""


#: The three checkpoint files. Named as constants so a downstream script
#: asks for the artefact it means rather than retyping a filename.
LATEST_CHECKPOINT = "checkpoint_latest.pth"
#: This one produces the test predictions.
BEST_CHECKPOINT = "checkpoint_best.pth"
#: Diagnostic only. Best against final says whether the model was still
#: improving at the budget; it is never reported as a result.
FINAL_CHECKPOINT = "checkpoint_final.pth"
#: `nnUNetTrainer.py:183` — every 50 epochs.
SAVE_EVERY = 50


def checkpoint_for_test_predictions(output_dir: Path) -> Path:
    """The one and only checkpoint inference may read.

    A helper rather than a filename, so a later script cannot reach for
    `checkpoint_final.pth` by accident: the checkpoint rule makes `best` the result and `final`
    a diagnostic, and the two are one character apart in a glob.
    """
    path = Path(output_dir) / BEST_CHECKPOINT
    if not path.is_file():
        raise CheckpointError(
            f"no {BEST_CHECKPOINT} in {output_dir}. Test predictions come from the "
            f"selected checkpoint and from nothing else — {FINAL_CHECKPOINT} is a "
            "diagnostic and must not be substituted.")
    return path


def save_checkpoint(path: Path, *, model, optimiser, scaler, scheduler, record,
                    selector, ema: float | None, next_epoch: int,
                    standardisation: dict, init_args: dict | None = None,
                    entry_point: str | None = None,
                    inference_allowed_mirroring_axes=None,
                    extra: dict | None = None) -> Path:
    """Write everything a resumed run needs, including the RNG streams.

    The RNG states travel because a resumed run that draws a different data stream
    is not the same run, and the resume test compares against an uninterrupted one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "next_epoch": int(next_epoch),
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler_ctr": int(getattr(scheduler, "ctr", 0)),
        "record": record.state_dict(),
        "selector": {"best_ema": selector.best_ema,
                     "saved_checkpoint_epoch": selector.saved_checkpoint_epoch,
                     "history": list(selector.history)},
        # The selection EMA's running value. Reinitialising it after a resume would
        # restart the smoothing and make the selection depend on where the run broke.
        "ema": None if ema is None else float(ema),
        # Carried so a resumed run cannot refit the standardisation on a different
        # split — the constants are fixed to the training rows of that fold.
        "standardisation": standardisation,
        # The entry point, its arguments and the mirroring axes are carried
        # explicitly rather than left implied.
        "init_args": init_args or {},
        "entry_point": entry_point,
        "inference_allowed_mirroring_axes": (
            None if inference_allowed_mirroring_axes is None
            else list(inference_allowed_mirroring_axes)),
        "rng": {"torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate()},
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, *, model, optimiser, scaler, scheduler, record,
                    selector, map_location="cuda", restore_rng: bool = True) -> dict:
    """Restore a run in a fresh process. Returns the resume metadata."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"checkpoint format {payload.get('format')} but this code writes "
            f"{CHECKPOINT_FORMAT}; resuming across formats is not supported and would "
            "silently drop fields")

    model.load_state_dict(payload["model"])
    optimiser.load_state_dict(payload["optimiser"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None:
        scheduler.ctr = int(payload.get("scheduler_ctr", 0))

    record.load_state_dict(payload["record"])
    sel = payload["selector"]
    selector.best_ema = sel["best_ema"]
    selector.saved_checkpoint_epoch = sel["saved_checkpoint_epoch"]
    selector.history = [tuple(h) for h in sel["history"]]

    if restore_rng and payload.get("rng"):
        rng = payload["rng"]
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s
                                          for s in rng["torch_cuda"]])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    return {"next_epoch": int(payload["next_epoch"]), "ema": payload["ema"],
            "standardisation": payload.get("standardisation"),
            "init_args": payload.get("init_args", {}),
            "entry_point": payload.get("entry_point"),
            "inference_allowed_mirroring_axes":
                payload.get("inference_allowed_mirroring_axes"),
            "extra": payload.get("extra", {})}


def assert_checkpoint_matches_model(path: Path, *, model, built_from: str,
                                   map_location="cpu") -> None:
    """Raise, naming both rows, if the checkpoint was not trained on this model.

    A checkpoint loaded into a model built from the wrong row dies on a bare
    ``size mismatch for blend.weight`` that names neither row. The blend width
    is the giveaway — the rows differ in how many inputs they blend — so a
    mismatch there is reported as what it is: the wrong row.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    saved = payload.get("model") or {}
    current = model.state_dict()
    recorded_row = (payload.get("init_args") or {}).get("config_row")
    missing = sorted(set(current) - set(saved))
    unexpected = sorted(set(saved) - set(current))
    mismatched = [(k, tuple(saved[k].shape), tuple(current[k].shape))
                  for k in sorted(set(saved) & set(current))
                  if hasattr(saved[k], "shape") and tuple(saved[k].shape) != tuple(current[k].shape)]
    if not (missing or unexpected or mismatched):
        return
    lines = [f"the checkpoint at {path.name} was not trained on the model this built.",
             f"  model built from row : {built_from!r}",
             f"  checkpoint recorded  : {recorded_row!r}"]
    if mismatched:
        lines.append(f"  {len(mismatched)} parameter(s) differ in shape, first three:")
        for name, was, now in mismatched[:3]:
            lines.append(f"    {name}: checkpoint {was} against built {now}")
        if any(name.startswith("blend.") for name, _, _ in mismatched):
            lines.append("  `blend.weight` differs, which is how the rows differ from "
                         "one another — this is the wrong row, not a corrupt file.")
    if missing:
        lines.append(f"  {len(missing)} tensor(s) absent from the checkpoint, e.g. {missing[:3]}")
    if unexpected:
        lines.append(f"  {len(unexpected)} tensor(s) in the checkpoint the model has no "
                     f"place for, e.g. {unexpected[:3]}")
    lines.append("  The row comes from run_manifest.json's `config_row`; if that is "
                 "wrong the manifest is wrong, and no command-line override exists "
                 "deliberately.")
    raise CheckpointError("\n".join(lines))


def load_weights_for_inference(path: Path, *, model, map_location="cuda") -> dict:
    """Weights and provenance only, for the test-time path.

    :func:`load_checkpoint` restores a run: optimiser, grad scaler, scheduler
    counter, the whole record, the selector, and the RNG streams. Inference needs
    none of that, and restoring it would make a read-only path mutate global
    state — the process-wide RNG streams above all. It also would not typecheck:
    there is no optimiser at inference time to hand it, and fabricating one to
    satisfy a signature is how a diagnostic artefact ends up looking like a run.

    So this reads the same payload, applies the same format check, loads the model
    and returns what inference needs to describe where the weights came from.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"checkpoint format {payload.get('format')} but this code writes "
            f"{CHECKPOINT_FORMAT}; reading across formats is not supported and would "
            "silently drop fields")
    model.load_state_dict(payload["model"])
    selector = payload.get("selector") or {}
    return {"next_epoch": int(payload["next_epoch"]),
            "saved_checkpoint_epoch": selector.get("saved_checkpoint_epoch"),
            "best_ema": selector.get("best_ema"),
            "standardisation": payload.get("standardisation"),
            "init_args": payload.get("init_args", {}),
            "entry_point": payload.get("entry_point"),
            # This is load-bearing: inference has to know which flips are
            # permitted, and a checkpoint that does not say cannot be used
            # without guessing. It is carried and recorded; no test-time
            # mirroring is applied, so nothing here acts on it.
            "inference_allowed_mirroring_axes":
                payload.get("inference_allowed_mirroring_axes"),
            "extra": payload.get("extra", {})}


class CheckpointWriter:
    """The three checkpoint files, with the best-write derived from the selector.

    The selector decides improved; the writer decides write. Expressed as two
    comparisons they can drift, and a run would report a
    `saved_checkpoint_epoch` whose weights were never written — or write
    weights at an epoch it did not select. Neither shows up in a loss curve.

    So the writer performs no comparison of its own. It reads
    `selector.saved_checkpoint_epoch` — the selector's own record of its
    verdict — and writes `checkpoint_best.pth` exactly when that equals the
    epoch just finished. When the caller also passes the boolean the selector
    returned, the two are cross-checked and a disagreement raises rather than
    being resolved.

    The periodic rule is `nnUNetTrainer.py:1042`: every `SAVE_EVERY` epochs,
    skipped on the final epoch — `(epoch + 1) % save_every == 0 and
    epoch != num_epochs - 1`. The deletion is `on_train_end` `:857–859`.
    """

    def __init__(self, output_dir: Path, total_epochs: int, save_every: int = SAVE_EVERY):
        self.output_dir = Path(output_dir)
        self.total_epochs = int(total_epochs)
        self.save_every = int(save_every)
        self.written: list[tuple[int, str]] = []

    def _save(self, name: str, epoch: int, **kwargs) -> Path:
        path = save_checkpoint(self.output_dir / name, next_epoch=epoch + 1, **kwargs)
        self.written.append((epoch, name))
        return path

    def on_epoch_end(self, epoch: int, *, selector, improved: bool | None = None,
                     **kwargs) -> list[str]:
        """Write whatever this epoch calls for. Returns the filenames written."""
        derived = selector.saved_checkpoint_epoch == epoch
        if improved is not None and bool(improved) != derived:
            raise CheckpointError(
                f"epoch {epoch}: the selector returned improved={improved} but its "
                f"saved_checkpoint_epoch is {selector.saved_checkpoint_epoch}. The write "
                "and the selection have drifted; refusing to write either way.")
        names = []
        if (epoch + 1) % self.save_every == 0 and epoch != self.total_epochs - 1:
            self._save(LATEST_CHECKPOINT, epoch, selector=selector, **kwargs)
            names.append(LATEST_CHECKPOINT)
        if derived:
            self._save(BEST_CHECKPOINT, epoch, selector=selector, **kwargs)
            names.append(BEST_CHECKPOINT)
        return names

    def on_train_end(self, epoch: int, *, selector, **kwargs) -> list[str]:
        """`checkpoint_final.pth`, then delete `checkpoint_latest.pth`.

        The deletion is what makes the presence of `checkpoint_latest.pth` mean
        "this run did not finish" — which is the signal a resume looks for.
        """
        self._save(FINAL_CHECKPOINT, epoch, selector=selector, **kwargs)
        names = [FINAL_CHECKPOINT]
        latest = self.output_dir / LATEST_CHECKPOINT
        if latest.is_file():
            latest.unlink()
            names.append(f"deleted {LATEST_CHECKPOINT}")
        return names
