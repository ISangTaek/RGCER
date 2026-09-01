"""D7-0 CLST direct-forward audit (plan §30-§37, §71-§72).

D6 Stage A recorded CLST layer scores of exactly 0 for the matched
real/shuffle teacher pair while their parameter deltas were clearly non-zero —
logically suspicious.  This audit deliberately does NOT reuse the old CLST
hook-aggregation path: it is a fresh, direct-forward implementation over a
FIXED human3 TRAIN probe batch (never validation), reporting per-layer
real-vs-shuffle representation statistics (plan §72), the final pooled
difference (§34), and a weight-perturbation sanity check (§35): adding a
deterministic +1e-4 to one layer-0 weight tensor of the real teacher MUST
produce a non-zero layer difference, otherwise the audit itself is broken
(§36 CLST_DIAGNOSTIC_BROKEN).

Outputs:
- D7_0_CLST_DIRECT_AUDIT.csv   layer, n_samples, real_norm_mean,
  shuffle_norm_mean, difference_norm_mean, difference_norm_max, cosine_mean,
  allclose_fraction  (plus a "final_pooled" row, §34)
- D7_0_CLST_WEIGHT_SANITY.json perturbation evidence (§35)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from main import build_parser, validate_params

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from d6_prep_inits import (  # noqa: E402  (checkpoint loader only — NOT the old CLST scorer)
    _build_model,
    _human_loaders,
    _load_teacher_checkpoint,
    _verify_teacher_pair,
)


def verify_audit_teachers(real_dir, shuffle_dir, expected_epoch: int, expected_model_seed: int) -> dict:
    """Fifth-review P1-5 (§40-§43): the audit MUST consume a matched
    real/shuffle teacher pair — otherwise initialization/split noise could be
    misread as a semantic difference.  Reuses the full D6 pair contract."""

    return _verify_teacher_pair(
        Path(real_dir),
        Path(shuffle_dir),
        expected_epoch,
        expected_model_seed=expected_model_seed,
    )


def _fixed_probe_batches(train_loaders: dict, probe_size: int):
    """Plan §32/§47: one fixed set of human3 TRAIN batches covering the first
    `probe_size` unique sorted sample_ids."""

    from d7_diagnostics import collect_probe_batches, select_probe_ids

    seen = set()
    candidate = []
    for loader in train_loaders.values():
        if loader is None:
            continue
        for batch in loader:
            if getattr(batch, "get", lambda *_: None)("is_empty", False):
                continue
            for sample_id in (getattr(batch, "sample_id", None) or []):
                if str(sample_id) not in seen:
                    seen.add(str(sample_id))
                    candidate.append(str(sample_id))
    probe_ids = select_probe_ids(candidate, probe_size)
    return collect_probe_batches(train_loaders, probe_ids), probe_ids


def _layer_outputs(model, batches, device):
    """Direct forward capture: run the backbone once per batch and record each
    layer's CLS-pooled output plus the final pooled representation."""

    layers = model.encoder.backbone.layers
    stores: list[dict[int, list[torch.Tensor]]] = [dict() for _ in layers]
    final: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(store, index):
        def hook(_module, _inputs, output):
            store.setdefault(index, []).append(output[:, 0, :].detach().float().cpu())
        return hook

    was_training = model.training
    model.eval()
    try:
        for index, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(stores[index], index)))
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                hidden = model.encoder.backbone(batch)[:, 0, :].detach().float().cpu()
                for position, sample_id in enumerate(batch.sample_id):
                    final[str(sample_id)] = hidden[position]
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()
    per_layer = []
    for index in range(len(layers)):
        tensors = stores[index][index]
        per_layer.append(torch.cat(tensors, dim=0))
    return per_layer, final


def _row(layer, real: torch.Tensor, shuffle: torch.Tensor) -> dict:
    difference = real - shuffle
    cosine = torch.nn.functional.cosine_similarity(real, shuffle, dim=1, eps=1e-8)
    return {
        "layer": layer,
        "n_samples": int(real.shape[0]),
        "real_norm_mean": float(real.norm(dim=1).mean()),
        "shuffle_norm_mean": float(shuffle.norm(dim=1).mean()),
        "difference_norm_mean": float(difference.norm(dim=1).mean()),
        "difference_norm_max": float(difference.norm(dim=1).max()),
        "cosine_mean": float(cosine.mean()),
        "allclose_fraction": float(
            torch.isclose(real, shuffle, atol=1e-6, rtol=0.0).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_real_dir", required=True)
    parser.add_argument("--teacher_shuffle_dir", required=True)
    parser.add_argument("--expected_teacher_epoch", type=int, required=True)
    parser.add_argument("--human_seed", type=int, default=42)
    parser.add_argument("--probe_size", type=int, default=128)
    parser.add_argument("--gpu_id", default="cpu")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if str(args.gpu_id) == "cpu"
        else torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    )
    real_path, real_payload = _load_teacher_checkpoint(
        Path(args.teacher_real_dir), args.expected_teacher_epoch
    )
    shuffle_path, shuffle_payload = _load_teacher_checkpoint(
        Path(args.teacher_shuffle_dir), args.expected_teacher_epoch
    )
    # §40-§43: prove the pair is matched (init hash / seed / manifest /
    # datastore / sanity table) BEFORE any representation comparison.
    matched = verify_audit_teachers(
        args.teacher_real_dir, args.teacher_shuffle_dir, args.expected_teacher_epoch, args.human_seed
    )
    template_config = matched["teacher_configuration"]

    model_real = _build_model(args.human_seed, "animal56", device, template_config)[0]
    model_real.load_state_dict(real_payload["model_state"], strict=True)
    model_real.to(device)
    model_shuffle = _build_model(args.human_seed, "animal56", device, template_config)[0]
    model_shuffle.load_state_dict(shuffle_payload["model_state"], strict=True)
    model_shuffle.to(device)

    params = build_parser().parse_args([])
    for key in (
        "data_store_dir", "split_seed", "splitting", "vs", "calibration_size", "ts",
        "hidden_dim", "mid_dim", "a_layers", "a_heads", "edge_bias_mode",
        "spatial_pos_clip", "max_nodes_filter", "prediction_mode",
        "head_hidden_dim", "head_dropout",
    ):
        if key in template_config:
            setattr(params, key, template_config[key])
    setattr(params, "arch", "Graphormer")
    setattr(params, "toxacute_task_scope", "human3")
    setattr(params, "seed", args.human_seed)
    setattr(params, "fit_conformal", False)
    setattr(params, "train_eval_scope", "validation_only")
    setattr(params, "card_lambda_delta", 0.0)
    validate_params(params)
    loaders = _human_loaders(params)
    batches, probe_ids = _fixed_probe_batches(loaders["train"], args.probe_size)
    print(f"D7-0 audit on {len(probe_ids)} fixed human3 train molecules")

    real_layers, real_final = _layer_outputs(model_real, batches, device)
    shuffle_layers, shuffle_final = _layer_outputs(model_shuffle, batches, device)

    rows = [
        _row(index, real_layers[index], shuffle_layers[index])
        for index in range(len(real_layers))
    ]
    final_diff_norms = {
        sample_id: float((real_final[sample_id] - shuffle_final[sample_id]).norm())
        for sample_id in sorted(real_final)
    }
    final_row = _row(
        "final_pooled",
        torch.stack([real_final[key] for key in sorted(real_final)]),
        torch.stack([shuffle_final[key] for key in sorted(real_final)]),
    )
    rows.append(final_row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_csv}")

    # §35 weight-perturbation sanity: the audit MUST see a deterministic
    # +1e-4 change on one layer-0 weight tensor.
    model_perturbed = _build_model(args.human_seed, "animal56", device, template_config)[0]
    model_perturbed.load_state_dict(real_payload["model_state"], strict=True)
    model_perturbed.to(device)
    layer0 = model_perturbed.encoder.backbone.layers[0]
    perturbed_tensor_name = None
    for name, parameter in layer0.named_parameters():
        if parameter.dim() >= 2:
            with torch.no_grad():
                parameter.add_(1e-4)
            perturbed_tensor_name = f"encoder.backbone.layers.0.{name}"
            break
    perturbed_layers, _ = _layer_outputs(model_perturbed, batches, device)
    sanity_rows = []
    for index in range(len(real_layers)):
        difference = float(
            (perturbed_layers[index] - real_layers[index]).norm(dim=1).mean()
        )
        sanity_rows.append({"layer": index, "difference_norm_mean": difference})
    layer0_difference = sanity_rows[0]["difference_norm_mean"] if sanity_rows else 0.0
    detected = bool(layer0_difference > 0.0)
    sanity = {
        "perturbed_tensor": perturbed_tensor_name,
        "perturbation": 1e-4,
        "per_layer_difference_norm_mean": sanity_rows,
        "layer0_difference_norm_mean": layer0_difference,
        "detected": detected,
        "probe_ids_first5": probe_ids[:5],
        "teacher_real_checkpoint": str(real_path),
        "teacher_shuffle_checkpoint": str(shuffle_path),
        "max_final_pooled_difference": max(final_diff_norms.values()),
        # §43: matched-teacher provenance evidence.
        "matched_teacher_provenance": {
            "teacher_initial_model_sha256": matched.get("teacher_initial_model_sha256"),
            "split_manifest_hash": matched.get("split_manifest_hash"),
            "datastore_fingerprint": matched.get("datastore_fingerprint"),
            "feature_schema_version": matched.get("feature_schema_version"),
            "animal_shuffle_mapping_sha256": matched.get("animal_shuffle_mapping_sha256"),
            "animal_shuffle_seed": matched.get("animal_shuffle_seed"),
            "teacher_real_epoch": matched.get("teacher_real_epoch"),
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print(f"wrote {output_json}")

    if not detected:
        # §36 CLST_DIAGNOSTIC_BROKEN: the audit cannot see a real weight
        # change, so its zero-difference verdicts carry no information.
        print("CLST_DIAGNOSTIC_BROKEN: perturbation sanity was NOT detected")
        raise SystemExit(1)
    nonzero_layers = [
        row["layer"] for row in rows if row["difference_norm_mean"] > 0.0
    ]
    print(
        "D7-0 audit complete: perturbation sanity detected; "
        f"{len(nonzero_layers)}/{len(rows)} layers show non-zero real-vs-shuffle difference"
    )


if __name__ == "__main__":
    main()
