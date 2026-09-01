"""D7 representation-preservation diagnostics (plan §43-§48, §77).

``D7DriftTracker`` is attached to a Trainer after the init overlay and before
``train()``.  It captures the post-overlay backbone state and the animal-teacher
probe representations as the BASELINE (for S1/S2/S3/B1 the overlay IS the
pretrained backbone, so epoch-0 drift is exactly 0 by construction), then per
epoch records:

- backbone_param_drift  ||theta_t - theta_init|| / (||theta_init|| + eps) over
  all backbone tensors, early blocks (layers 0-3) and late blocks (layers 4-7)
- feature_drift         mean(1 - cos(h_t, h_init)) over a FIXED human3 TRAIN
  probe set (first N unique sorted sample_ids — never validation), computed
  only on the configured epochs (default 0/5/10/15/19)

Rows are appended to ``<save_path>/diagnostics/d7_representation_drift.csv``.
The human3_rmse column is joined by the aggregator from epoch_summary.csv so
this module stays decoupled from the metric plumbing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

BACKBONE_PREFIX = "encoder.backbone."
LAYERS_MARKER = BACKBONE_PREFIX + "layers."
EARLY_LAYERS = frozenset(range(0, 4))
LATE_LAYERS = frozenset(range(4, 8))


def parse_feature_drift_epochs(raw: str) -> set[int]:
    epochs = {int(token) for token in str(raw).split(",") if str(token).strip() != ""}
    if any(value < 0 for value in epochs):
        raise ValueError(f"feature_drift_epochs must be non-negative: {raw!r}")
    return epochs


def select_probe_ids(candidate_ids, limit: int) -> list[str]:
    """Plan §47: fixed probe set = first `limit` unique sample_ids in sorted
    order, drawn from human3 TRAIN only."""

    unique = sorted({str(value) for value in candidate_ids})
    if not unique:
        raise ValueError("probe selection found no human3 train sample ids")
    return unique[: int(limit)]


def collect_probe_batches(train_loaders: dict, probe_ids, collator=None):
    """Iterate the given TRAIN loader dict only, returning the minimal set of
    collated batches whose union covers the probe ids.  Val/calibration/test
    loaders are never passed in, so they cannot be touched."""

    wanted = set(probe_ids)
    seen: set[str] = set()
    batches = []
    for task in sorted(train_loaders):
        loader = train_loaders.get(task)
        if loader is None:
            continue
        for batch in loader:
            if getattr(batch, "get", lambda *_: None)("is_empty", False):
                continue
            batch_ids = [str(value) for value in (getattr(batch, "sample_id", None) or [])]
            if not any(sample_id in wanted and sample_id not in seen for sample_id in batch_ids):
                continue
            batches.append(batch)
            seen.update(sample_id for sample_id in batch_ids if sample_id in wanted)
            if wanted <= seen:
                return batches
    missing = sorted(wanted - seen)
    if missing:
        raise RuntimeError(
            f"probe set incomplete: missing {len(missing)} human3 train ids "
            f"(e.g. {missing[:3]})"
        )
    return batches


def pooled_representations(model, batches, device) -> dict[str, torch.Tensor]:
    """CLS-pooled backbone representation per sample id (no grad, eval)."""

    was_training = model.training
    model.eval()
    representations: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                hidden = model.encoder.backbone(batch)[:, 0, :].detach().float().cpu()
                for position, sample_id in enumerate(batch.sample_id):
                    representations[str(sample_id)] = hidden[position]
    finally:
        if was_training:
            model.train()
    return representations


def backbone_parameter_drift(state: dict, init_state: dict) -> tuple[float, float, float]:
    """Plan §43-§44: relative L2 drift over all backbone tensors, early blocks
    (layers 0-3) and late blocks (layers 4-7)."""

    sums = {"all": 0.0, "early": 0.0, "late": 0.0}
    base = {"all": 0.0, "early": 0.0, "late": 0.0}
    for key, tensor in state.items():
        if not key.startswith(BACKBONE_PREFIX) or key not in init_state:
            continue
        delta = tensor.detach().float() - init_state[key].detach().float()
        norm_sq = float(delta.norm() ** 2)
        anchor_sq = float(init_state[key].detach().float().norm() ** 2)
        sums["all"] += norm_sq
        base["all"] += anchor_sq
        if key.startswith(LAYERS_MARKER):
            layer_index = int(key.split(".")[3])
            bucket = "early" if layer_index in EARLY_LAYERS else "late"
            sums[bucket] += norm_sq
            base[bucket] += anchor_sq
    eps = 1e-12
    return tuple(
        (sums[bucket] ** 0.5) / ((base[bucket] ** 0.5) + eps) for bucket in ("all", "early", "late")
    )


def feature_drift(current: dict, reference: dict) -> float:
    """Plan §46: mean over the probe set of 1 - cos(h_t, h_init)."""

    ids = sorted(set(current) & set(reference))
    if not ids:
        raise RuntimeError("feature drift: no overlapping probe representations")
    scores = []
    for sample_id in ids:
        h_t = current[sample_id]
        h_0 = reference[sample_id]
        cosine = float(
            torch.nn.functional.cosine_similarity(
                h_t.unsqueeze(0), h_0.unsqueeze(0), dim=1, eps=1e-8
            )
        )
        scores.append(1.0 - cosine)
    return sum(scores) / len(scores)


DRIFT_FIELDS = (
    "epoch",
    "backbone_param_drift",
    "early_block_drift",
    "late_block_drift",
    "feature_drift",
)


class D7DriftTracker:
    """Attached as ``trainer.d7_drift_tracker``; call ``log_epoch`` at each
    epoch boundary from the training loop."""

    def __init__(
        self,
        *,
        model,
        train_loaders: dict,
        device,
        output_dir,
        probe_size: int = 128,
        feature_drift_epochs=None,
    ):
        self.device = device
        self.feature_drift_epochs = (
            set(feature_drift_epochs)
            if feature_drift_epochs is not None
            else {0, 5, 10, 15, 19}
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        init_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        self.init_state = init_state
        self.probe_ids = select_probe_ids(
            (
                sample_id
                for loader in train_loaders.values()
                if loader is not None
                for batch in loader
                if not getattr(batch, "get", lambda *_: None)("is_empty", False)
                for sample_id in (getattr(batch, "sample_id", None) or [])
            ),
            probe_size,
        )
        self.probe_batches = collect_probe_batches(train_loaders, self.probe_ids)
        self.reference_representations = pooled_representations(model, self.probe_batches, device)
        self.rows: list[dict] = []
        # The trainer logs at the START of each epoch: an "epoch t" row is the
        # state after t completed epochs, so epoch 0 is the untouched
        # post-overlay baseline (drift exactly 0, plan §96).

    def log_epoch(self, epoch: int, model) -> dict:
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        all_drift, early_drift, late_drift = backbone_parameter_drift(state, self.init_state)
        feature = ""
        if int(epoch) in self.feature_drift_epochs:
            current = pooled_representations(model, self.probe_batches, self.device)
            feature = f"{feature_drift(current, self.reference_representations):.8f}"
        row = {
            "epoch": int(epoch),
            "backbone_param_drift": f"{all_drift:.8f}",
            "early_block_drift": f"{early_drift:.8f}",
            "late_block_drift": f"{late_drift:.8f}",
            "feature_drift": feature,
        }
        self.rows.append(row)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "d7_representation_drift.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(DRIFT_FIELDS))
                writer.writeheader()
                writer.writerows(self.rows)
        return row
