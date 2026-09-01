"""D7 representation-preservation diagnostics (plan §43-§48, §77; fifth review P1-3/P1-4).

``D7DriftTracker`` is attached to a Trainer after the init overlay.  It
captures the post-overlay backbone state and the animal-teacher probe
representations as the BASELINE (for S1/S2/S3/B1 the overlay IS the pretrained
backbone, so the baseline drift is exactly 0 by construction), then:

- baseline row:  epoch=-1, state=baseline — logged at construction
- per-epoch row: logged AFTER each epoch's validation (state=post_epoch), so
  drift row epoch t and validation row epoch t refer to the SAME model state

For each row it records backbone parameter drift ||theta_t - theta_init|| /
(||theta_init|| + eps) over all backbone tensors, early blocks (layers 0-3)
and late blocks (layers 4-7), plus feature drift mean(1 - cos(h_t, h_init))
over a FIXED human3 TRAIN probe set (first N unique sorted sample_ids — never
validation) on the configured epochs.  ``drift_anchor_type`` records whether
the baseline is the animal-pretrained init (S-family/B1) or a counterfactual
modified init (O4/O5) so absolute drift magnitudes are never compared across
anchor types (fifth review P2-2).

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


def train_sample_ids(train_loaders: dict) -> list[str]:
    """Deterministic human3 train sample ids, read straight from the task
    datasets (never by iterating a shuffled DataLoader).  For the formal
    ToxAcuteTaskDataset ``get_sample_id`` reads the store index only — it does
    not open LMDB graphs."""

    sample_ids: set[str] = set()
    for task in sorted(train_loaders):
        loader = train_loaders.get(task)
        if loader is None:
            continue
        dataset = loader.dataset
        for index in range(len(dataset)):
            if hasattr(dataset, "get_sample_id"):
                sample_id = dataset.get_sample_id(index)
            else:
                sample_id = getattr(dataset[index], "sample_id")
            sample_ids.add(str(sample_id))
    if not sample_ids:
        raise ValueError("D7 probe selection found no Human3 train sample IDs")
    return sorted(sample_ids)


def collect_probe_batches(train_loaders: dict, probe_ids, *, batch_size: int = 64):
    """Sixth-review P1 (§17-§20): build EXACT probe batches containing ONLY
    the requested molecules — no shuffled-batch neighbours.  Items are pulled
    by sample_id straight from the datasets and re-collated in probe order
    with the loaders' own collate_fn, so training loader generators are never
    advanced and the probe set stays bit-identical across runs."""

    wanted_order = [str(value) for value in probe_ids]
    wanted = set(wanted_order)
    items: dict[str, object] = {}
    collate_fn = None
    for task in sorted(train_loaders):
        loader = train_loaders.get(task)
        if loader is None:
            continue
        if collate_fn is None:
            collate_fn = getattr(loader, "collate_fn", None)
        dataset = loader.dataset
        for index in range(len(dataset)):
            if hasattr(dataset, "get_sample_id"):
                sample_id = str(dataset.get_sample_id(index))
            else:
                sample_id = str(getattr(dataset[index], "sample_id"))
            if sample_id not in wanted or sample_id in items:
                continue
            # Backbone diagnostics do not use task label semantics; any human3
            # task copy of the molecule is sufficient.
            items[sample_id] = dataset[index]
            if len(items) == len(wanted):
                break
        if len(items) == len(wanted):
            break
    missing = [sample_id for sample_id in wanted_order if sample_id not in items]
    if missing:
        raise RuntimeError(
            f"D7 probe set incomplete: missing {len(missing)} IDs; "
            f"examples={missing[:3]}"
        )
    if collate_fn is None:
        raise RuntimeError("D7 probe collection found no collator")
    ordered_items = [items[sample_id] for sample_id in wanted_order]
    return [
        collate_fn(ordered_items[start : start + batch_size])
        for start in range(0, len(ordered_items), batch_size)
    ]


def pooled_representations(model, batches, device, expected_ids=None) -> dict[str, torch.Tensor]:
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
    if expected_ids is not None:
        # Sixth-review P1 (§22): the probe statistics must cover EXACTLY the
        # selected molecules — a stray batch neighbour would pollute the
        # mechanism measurement.
        expected = {str(value) for value in expected_ids}
        observed = set(representations)
        if observed != expected:
            raise RuntimeError(
                "D7 probe representation ID mismatch: "
                f"expected={len(expected)} observed={len(observed)} "
                f"extra={sorted(observed - expected)[:3]} "
                f"missing={sorted(expected - observed)[:3]}"
            )
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
    "state",
    "drift_anchor_type",
    "backbone_param_drift",
    "early_block_drift",
    "late_block_drift",
    "feature_drift",
)


class D7DriftTracker:
    """Attached as ``trainer.d7_drift_tracker``.

    Epoch alignment (fifth review P1-4): the constructor logs the BASELINE row
    (epoch=-1, state=baseline — the untouched post-overlay state) and the
    trainer calls ``log_epoch`` AFTER each epoch's validation, so

        drift row epoch t  ==  validation row epoch t

    refer to the SAME model state.  Feature-drift epochs (§48) are therefore
    interpreted as "after completing these training epochs".
    """

    def __init__(
        self,
        *,
        model,
        train_loaders: dict,
        device,
        output_dir,
        probe_size: int = 128,
        feature_drift_epochs=None,
        anchor_type: str = "animal_pretrained",
    ):
        if anchor_type not in ("animal_pretrained", "counterfactual_init"):
            raise ValueError(f"unknown drift anchor type: {anchor_type!r}")
        self.device = device
        self.anchor_type = anchor_type
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
        self.probe_ids = select_probe_ids(train_sample_ids(train_loaders), probe_size)
        self.probe_batches = collect_probe_batches(train_loaders, self.probe_ids)
        self.reference_representations = pooled_representations(
            model, self.probe_batches, device, expected_ids=self.probe_ids
        )
        self.rows: list[dict] = []
        # Baseline row: the untouched post-overlay state, epoch=-1 (§38).
        self._append_row(-1, "baseline", model, compute_feature=False)

    def _append_row(self, epoch: int, state: str, model, *, compute_feature: bool) -> dict:
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        all_drift, early_drift, late_drift = backbone_parameter_drift(state_dict, self.init_state)
        if state == "baseline":
            # The baseline IS the reference — its feature drift is 0 by
            # definition (fifth-review P1-4 alignment).
            feature = "0.00000000"
        elif compute_feature and int(epoch) in self.feature_drift_epochs:
            current = pooled_representations(
                model, self.probe_batches, self.device, expected_ids=self.probe_ids
            )
            feature = f"{feature_drift(current, self.reference_representations):.8f}"
        else:
            feature = ""
        row = {
            "epoch": int(epoch),
            "state": state,
            "drift_anchor_type": self.anchor_type,
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

    def log_epoch(self, epoch: int, model) -> dict:
        """Log the POST-epoch state so it aligns with validation epoch t."""
        return self._append_row(epoch, "post_epoch", model, compute_feature=True)
