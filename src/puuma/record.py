"""The run record: the training history and the curve.

Covers what a fold needs while it is running — the per-epoch history, the
per-mother validation predictions, and the curve. Test predictions and the
selected-checkpoint file are the inference path's, and are not here.

Loss share is derived, never stored as if bounded: `DC_and_CE` goes negative
once the segmentation is good, so the "share" is a ratio to a denominator that
can approach zero and change sign. The raw weighted terms are what the history
stores; :func:`derive_share` computes the ratio and returns NaN where the
denominator makes it meaningless.

No `nnunetv2` import, so import-order safe.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

#: Every field required per epoch. Asserted by a test, so adding a new
#: requirement without adding it here fails rather than being forgotten.
REQUIRED_EPOCH_FIELDS = (
    "epoch", "train_loss_total", "train_loss_regression", "train_loss_segmentation",
    # The two validation quantities, named so neither can be mistaken for the other.
    # `val_weighted_regression_term` is w_reg x MSE_std — the regression term alone,
    # comparable to `train_loss_regression` x w_reg and to nothing else.
    # `val_objective_*` is the FULL training objective on validation data, both terms
    # at the same weights, comparable to `train_loss_total`.
    "val_weighted_regression_term", "val_objective_total",
    "val_objective_regression", "val_objective_segmentation",
    "val_standardised_mse", "val_standardised_mse_ema",
    "val_mae_weeks", "grad_norm_regression", "grad_norm_segmentation",
    "learning_rate", "epoch_seconds", "validation_seconds",
    "dice_channel_1", "dice_channel_2",
    # The training-side gate fallback, made visible.
    # The count alone would not say whether the fallback was benign; the fills do.
    "gate_failed_items", "gate_failed_rate",
    "gate_failed_fill_min", "gate_failed_fill_median",
    # The inference-side counterpart. The per-SUBJECT flag lives in
    # `validation_predictions.csv`; this is the aggregate for the curve.
    "n_gate_failed_mothers",
)


#: The objective's regression weight, used only to put the training term on the same axis
#: as the validation one. The authoritative constant lives in `puuma.loss`; this
#: module deliberately does not import it, to stay free of `nnunetv2`.
W_REG_FOR_PLOT = 2.0


class RecordError(RuntimeError):
    """Raised when the record would be incomplete or self-inconsistent."""


@dataclass
class EpochRecord:
    """One epoch of the run history. Raw terms only; shares are derived."""

    epoch: int
    train_loss_total: float
    train_loss_regression: float
    train_loss_segmentation: float
    val_weighted_regression_term: float
    val_objective_total: float
    val_objective_regression: float
    val_objective_segmentation: float
    val_standardised_mse: float
    val_standardised_mse_ema: float
    val_mae_weeks: float
    grad_norm_regression: float
    grad_norm_segmentation: float
    learning_rate: float
    epoch_seconds: float
    validation_seconds: float
    dice_channel_1: float
    dice_channel_2: float
    gate_failed_items: int = 0
    gate_failed_rate: float = 0.0
    gate_failed_fill_min: float = float("nan")
    gate_failed_fill_median: float = float("nan")
    n_gate_failed_mothers: int = 0
    #: The blend layer's weights, in `row.blend_inputs` order, as their own
    #: series — the only per-epoch quantity that shows detector 7's failure: a
    #: run converging on a CL-only predictor keeps improving its validation
    #: MSE, because CL genuinely predicts `GA_del`.
    blend_weights: list = field(default_factory=list)
    #: `(|w_global| + |w_local|) / |w_cl|`, the ratio detector 7 watches. Stored
    #: rather than recomputed so the trace carries what the detector saw.
    blend_image_share: float = float("nan")
    #: The maximum `MambaLayer` output magnitude over the epoch, and the layer
    #: that produced it — already computed every step for detector 6. NaN on a
    #: row with no Mamba layers.
    mamba_activation_max: float = float("nan")
    mamba_activation_layer: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def derive_share(weighted_regression: float, weighted_segmentation: float) -> dict[str, float]:
    """The loss share, derived and explicitly unbounded.

    Returns NaN when the denominator is zero or when either term is negative — in
    that regime the ratio is not a fraction and reporting a number invites it to
    be read as one.
    """
    total = weighted_regression + weighted_segmentation
    if total == 0.0 or weighted_regression < 0.0 or weighted_segmentation < 0.0:
        return {"regression": float("nan"), "segmentation": float("nan"),
                "well_defined": False}
    return {"regression": weighted_regression / total,
            "segmentation": weighted_segmentation / total, "well_defined": True}


@dataclass
class RunRecord:
    """The accumulating record for one fold."""

    output_dir: Path
    config: dict = field(default_factory=dict)
    epochs: list[EpochRecord] = field(default_factory=list)
    validation_rows: list[dict] = field(default_factory=list)

    #: One JSON object per epoch, appended and flushed as the epoch completes.
    EPOCH_LOG = "training_epochs.jsonl"

    def epoch_log_path(self) -> Path:
        return Path(self.output_dir) / self.EPOCH_LOG

    def add_epoch(self, record: EpochRecord) -> None:
        """Record an epoch, and append it to the JSONL immediately.

        Always on, no flag: one line per epoch costs nothing.
        `training_history.json` is a whole-file rewrite — truncate, serialise,
        close — so a run killed inside that window leaves a truncated JSON; an
        append of one line, flushed and `fsync`ed, cannot be half-written into
        the previous epochs. The JSONL is the durable record and the JSON the
        convenient one.

        After a resume the file may repeat an epoch number: a run killed after
        an epoch was appended but before the next checkpoint resumes from the
        checkpoint's epoch, and those epochs are appended a second time.
        :func:`read_epoch_log` keeps the last line for each epoch, which is the
        one that ran.
        """
        missing = [f for f in REQUIRED_EPOCH_FIELDS if not hasattr(record, f)]
        if missing:
            raise RecordError(f"epoch record is missing required fields: {missing}")
        self.epochs.append(record)
        path = self.epoch_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record.as_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def add_validation(self, epoch: int, predictions: Sequence) -> None:
        """Per-subject validation predictions, every validation epoch, long format."""
        for p in predictions:
            row = p.as_row() if hasattr(p, "as_row") else dict(p)
            row["epoch"] = int(epoch)
            self.validation_rows.append(row)

    def raw_metric_trace(self) -> list[float]:
        return [e.val_standardised_mse for e in self.epochs]

    def write(self, saved_checkpoint_epoch: int | None = None,
              argmin_epoch_ungated: int | None = None,
              detectors: dict | None = None,
              standardisation: dict | None = None,
              parameters: dict | None = None) -> None:
        """Write the history, the per-subject predictions and the run header."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            # Both are recorded, under distinct names: the two differ whenever
            # the EMA and the raw trace disagree on the best epoch.
            "argmin_epoch_ungated": argmin_epoch_ungated,
            "saved_checkpoint_epoch": saved_checkpoint_epoch,
            "standardisation": standardisation,
            "parameters": parameters,
            "detectors": detectors or {},
            "epochs": [e.as_dict() for e in self.epochs],
        }
        (self.output_dir / "training_history.json").write_text(json.dumps(payload, indent=2))
        _write_csv(self.output_dir / "validation_predictions.csv", self.validation_rows)

    def plot(self, saved_checkpoint_epoch: int | None = None) -> Path:
        """The run's training curve. `progress.png` does not cover this.

        Its selection panel plots pseudo-Dice, which this design drops, and its logger
        records only the total losses — so the metric the checkpoint is chosen on
        and the per-task gradient norms would both be absent.
        """
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from puuma.selection import BLEND_IMAGE_SHARE_THRESHOLD

        if not self.epochs:
            raise RecordError("no epochs recorded; there is no curve to draw")
        e = [r.epoch for r in self.epochs]
        fig, ax = plt.subplots(8, 1, figsize=(11, 34), sharex=True)

        # Panel 0: the two totals, which ARE comparable — same terms, same weights.
        ax[0].plot(e, [r.train_loss_total for r in self.epochs], label="train objective")
        ax[0].plot(e, [r.val_objective_total for r in self.epochs],
                   label="val objective (same terms and weights)")
        ax[0].set_ylabel("objective"); ax[0].legend()
        ax[0].set_title("training curve — validation windows are NOT augmented, "
                        "so part of any gap is augmentation, not generalisation")

        # Panel 1: the regression term against the regression term. The validation
        # quantity here is w_reg x MSE_std and is comparable to nothing else.
        ax[1].plot(e, [W_REG_FOR_PLOT * r.train_loss_regression for r in self.epochs],
                   label="train w_reg x MSE")
        ax[1].plot(e, [r.val_weighted_regression_term for r in self.epochs],
                   label="val w_reg x MSE_std")
        ax[1].set_ylabel("regression term"); ax[1].legend()

        ax[2].plot(e, [r.val_standardised_mse for r in self.epochs], ls=":",
                   label="val MSE_std (raw)")
        ax[2].plot(e, [r.val_standardised_mse_ema for r in self.epochs],
                   label="val MSE_std (EMA, selection)")
        if saved_checkpoint_epoch is not None:
            ax[2].axvline(saved_checkpoint_epoch, color="k", ls="--",
                          label=f"selected epoch {saved_checkpoint_epoch}")
        ax[2].set_ylabel("standardised MSE"); ax[2].legend()

        ax[3].plot(e, [r.train_loss_regression for r in self.epochs], label="regression")
        ax[3].plot(e, [r.train_loss_segmentation for r in self.epochs], label="segmentation")
        ax[3].axhline(0.0, color="grey", lw=0.8)
        ax[3].set_ylabel("train term (raw)"); ax[3].legend()
        ax[3].set_title("terms separately — share is not a fraction")

        ax[4].semilogy(e, [max(r.grad_norm_regression, 1e-12) for r in self.epochs],
                       label="|g| regression")
        ax[4].semilogy(e, [max(r.grad_norm_segmentation, 1e-12) for r in self.epochs],
                       label="|g| segmentation")
        ax[4].set_ylabel("grad norm into shared encoder")
        ax[4].legend()

        # The gate-fallback count as its own series, with the fills beside it —
        # a rising count is only alarming if the fills fall with it.
        ax[5].bar(e, [r.gate_failed_items for r in self.epochs], color="tab:red",
                  alpha=0.6, label="items using the gate fallback")
        ax[5].set_ylabel("fallback count"); ax[5].set_xlabel("epoch")
        twin = ax[5].twinx()
        twin.plot(e, [r.gate_failed_fill_min for r in self.epochs], "k.",
                  label="min achieved fill")
        twin.axhline(0.33, color="grey", ls="--", lw=0.8)
        twin.set_ylabel("achieved fill (gate 0.33)")
        h1, l1 = ax[5].get_legend_handles_labels(); h2, l2 = twin.get_legend_handles_labels()
        ax[5].legend(h1 + h2, l1 + l2, loc="upper left")

        # The blend's weights as their own series, with detector 7's
        # threshold drawn on the twin axis. A weight going to zero is the thing to
        # see, so the zero line is drawn rather than left implicit.
        names = list(self.config.get("blend_inputs") or [])
        series = [r.blend_weights for r in self.epochs]
        width = max((len(w) for w in series), default=0)
        for index in range(width):
            label = names[index] if index < len(names) else f"w{index}"
            ax[6].plot(e, [w[index] if index < len(w) else float("nan") for w in series],
                       label=label)
        ax[6].axhline(0.0, color="grey", lw=0.8)
        ax[6].set_ylabel("blend weight"); ax[6].set_xlabel("epoch")
        share = ax[6].twinx()
        share.semilogy(e, [max(r.blend_image_share, 1e-4) for r in self.epochs], "k:",
                       label="image / CL share")
        share.axhline(BLEND_IMAGE_SHARE_THRESHOLD, color="k", ls="--", lw=0.8)
        share.set_ylabel("image / CL weight (detector 7)")
        h1, l1 = ax[6].get_legend_handles_labels(); h2, l2 = share.get_legend_handles_labels()
        ax[6].legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=8)

        # The Mamba activation maximum, log scale, with detector 6's ceiling.
        maxima = [r.mamba_activation_max for r in self.epochs]
        if any(np.isfinite(v) and v > 0 for v in maxima):
            ax[7].semilogy(e, [max(v, 1e-12) if np.isfinite(v) else np.nan
                               for v in maxima], label="max |MambaLayer output|")
            ceiling = self.config.get("mamba_activation_ceiling")
            if ceiling and float(ceiling) < 1e6:
                ax[7].axhline(float(ceiling), color="r", ls="--", lw=0.8,
                              label=f"detector 6 ceiling {float(ceiling):g}")
            leaders = {r.mamba_activation_layer for r in self.epochs
                       if r.mamba_activation_layer}
            ax[7].set_ylabel("max Mamba activation")
            ax[7].set_title(f"leading layer(s): {', '.join(sorted(leaders)) or 'n/a'}",
                            fontsize=8, loc="left")
            ax[7].legend(loc="upper left", fontsize=8)
        else:
            ax[7].text(0.5, 0.5, "no Mamba layers in this row",
                       ha="center", va="center", transform=ax[7].transAxes)
        ax[7].set_xlabel("epoch")

        fig.tight_layout()
        path = self.output_dir / "training_curve.png"
        fig.savefig(path, dpi=110); plt.close(fig)
        return path


    def state_dict(self) -> dict:
        """Everything a resumed run needs to continue this record.

        The whole history travels, not the fragment since the last save — the
        detectors read the full trace and would otherwise see only the tail.
        """
        return {"config": dict(self.config),
                "epochs": [e.as_dict() for e in self.epochs],
                "validation_rows": [dict(r) for r in self.validation_rows]}

    def load_state_dict(self, state: dict) -> None:
        self.config = dict(state.get("config", {}))
        self.epochs = [EpochRecord(**e) for e in state.get("epochs", [])]
        self.validation_rows = [dict(r) for r in state.get("validation_rows", [])]


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    import csv
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_epoch_log(path: Path) -> list[dict]:
    """Read a `training_epochs.jsonl`, keeping the LAST line for each epoch.

    A resumed run may append an epoch twice (see :meth:`RunRecord.add_epoch`); the
    later line is the one that ran. A trailing partial line — the process died
    mid-write — is dropped rather than raising, because the point of this file is
    to be readable from outside the session while the run is still going.
    """
    path = Path(path)
    if not path.is_file():
        raise RecordError(f"no epoch log at {path}")
    by_epoch: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue                      # a partial trailing line; the run is live
        by_epoch[int(entry["epoch"])] = entry
    return [by_epoch[k] for k in sorted(by_epoch)]
