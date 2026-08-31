"""D6 candidate initialisation utilities (plan §7-§27, §63-§67).

Modes:
- csdt : theta_init = theta0 + alpha * (theta_real - theta_shuffle), shared
         Graphormer backbone only; heads stay seed-matched fresh (§64).
- b1   : standard sequential transfer initialisation - copy the real
         teacher's shared backbone into theta0 (diagnostic baseline).
- clst : layer-selective transfer - rank backbone blocks by the
         real-vs-shuffled representation discrepancy S_l computed on human3
         TRAIN molecules only (§11-§12, §66) and initialise the top-k blocks
         from the real teacher.
- card_table : precompute the per-sample pooled representation delta
         h_real(x) - h_shuffle(x) for human3 train molecules (§15, §67) and
         save it as the CARD distillation table.

The two teachers must share their initialisation hash with the theta0 build
seed (§63); this script verifies that and records per-layer delta norms.
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

import numpy as np
import torch

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS
from config import prepare_args
from dataset import DataCollator
from main import (
    _build_model_components,
    _loaders,
    _resolve_data_store,
    build_parser,
    task_names_for_params,
    validate_params,
)
from reproducibility import seed_everything, state_dict_sha256

BACKBONE_PREFIX = "encoder.backbone."


def _build_model(seed: int, task_scope: str, device):
    params = build_parser().parse_args([])
    setattr(params, "arch", "Graphormer")
    setattr(params, "toxacute_task_scope", task_scope)
    setattr(params, "seed", seed)
    validate_params(params)
    seed_everything(seed)
    task_names = task_names_for_params(params)
    kwargs, _ = prepare_args(params)
    encoder_class, architecture_class, decoders = _build_model_components(params, task_names, device)
    model = architecture_class(
        task_names, encoder_class, decoders, device, params, **kwargs.get("arch_args", {})
    )
    return model, task_names


def _teacher_state_path(run_dir: Path) -> Path:
    """Stage A/B teachers take the FIXED last-epoch parameters (plan §24/§38)."""

    last = sorted(Path(run_dir).glob("*_last.pt"))
    if not last:
        raise FileNotFoundError(f"no *_last.pt under {run_dir}")
    return last[0]


def _load_teacher_state(run_dir: Path) -> dict:
    payload = torch.load(_teacher_state_path(run_dir), map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model_state_dict"))
    if state is None:
        raise KeyError(f"checkpoint under {run_dir} has no model_state")
    return state


def _verify_matched_teacher_init(teacher_real_dir: Path, teacher_shuffle_dir: Path) -> str:
    """§63: real/shuffle teachers must share their initialisation hash."""

    hashes = {}
    for label, run_dir in (("real", teacher_real_dir), ("shuffle", teacher_shuffle_dir)):
        metadata_path = Path(run_dir) / "run_metadata.json"
        if not metadata_path.exists():
            raise SystemExit(f"missing run_metadata.json under {run_dir}")
        hashes[label] = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "initial_model_sha256"
        )
    if not hashes["real"] or hashes["real"] != hashes["shuffle"]:
        raise SystemExit(
            f"teacher initialisation mismatch: real={hashes['real']} shuffle={hashes['shuffle']}"
        )
    return hashes["real"]


def apply_csdt(theta0: dict, real: dict, shuffle: dict, alpha: float) -> dict:
    """theta_init = theta0 + alpha * (real - shuffle) on backbone params (§64)."""

    merged = {key: value.detach().clone() for key, value in theta0.items()}
    for key in merged:
        if key.startswith(BACKBONE_PREFIX) and key in real and key in shuffle:
            merged[key] = theta0[key] + alpha * (real[key].float() - shuffle[key].float())
    return merged


def apply_b1(theta0: dict, real: dict) -> dict:
    """Sequential transfer: copy the real teacher's whole shared backbone."""

    merged = {key: value.detach().clone() for key, value in theta0.items()}
    for key in merged:
        if key.startswith(BACKBONE_PREFIX) and key in real:
            merged[key] = real[key].detach().clone()
    return merged


@torch.no_grad()
def clst_layer_scores(
    model_real,
    model_shuffle,
    loader,
    collator,
    device,
) -> tuple[list[dict], int]:
    """S_l = E||h_real - h_shuffle|| / (E||h_real|| + eps) per block (§11, §66)."""

    layers = model_real.encoder.backbone.layers
    n_layers = len(layers)
    hooks_real, hooks_shuffle = [], []
    acts = {"real": {}, "shuffle": {}}

    def make_hook(store, layer_index):
        def hook(_module, _inputs, output):
            store[layer_index] = output[:, 0, :].detach().float().cpu()
        return hook

    for index, layer in enumerate(layers):
        hooks_real.append(layer.register_forward_hook(make_hook(acts["real"], index)))
        hooks_shuffle.append(layer.register_forward_hook(make_hook(acts["shuffle"], index)))

    diff_sums = np.zeros(n_layers)
    real_norm_sums = np.zeros(n_layers)
    count = 0
    model_real.eval()
    model_shuffle.eval()
    for batch in loader:
        if getattr(batch, "get", lambda *_: None)("is_empty", False):
            continue
        batch = batch.to(device)
        model_real.encoder.backbone(batch)
        model_shuffle.encoder.backbone(batch)
        for layer_index in range(n_layers):
            real_pooled = acts["real"][layer_index]
            shuffle_pooled = acts["shuffle"][layer_index]
            diff_sums[layer_index] += float((real_pooled - shuffle_pooled).norm(dim=-1).sum())
            real_norm_sums[layer_index] += float(real_pooled.norm(dim=-1).sum())
        count += real_pooled.size(0)
    for hook in hooks_real + hooks_shuffle:
        hook.remove()

    rows = []
    for layer_index in range(n_layers):
        denominator = real_norm_sums[layer_index] / max(count, 1) + 1e-8
        score = (diff_sums[layer_index] / max(count, 1)) / denominator
        rows.append(
            {
                "layer": layer_index,
                "real_minus_shuffle_norm_mean": float(diff_sums[layer_index] / max(count, 1)),
                "real_rep_norm_mean": float(real_norm_sums[layer_index] / max(count, 1)),
                "semantic_layer_score": float(score),
            }
        )
    rows.sort(key=lambda row: row["semantic_layer_score"], reverse=True)
    for rank, row in enumerate(rows):
        row["rank"] = rank
    return rows, count


@torch.no_grad()
def card_delta_table(model_real, model_shuffle, loaders, task_names, collator, device) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample pooled-representation delta over human3 train molecules (§15)."""

    model_real.eval()
    model_shuffle.eval()
    ids: list[str] = []
    deltas: list[np.ndarray] = []
    for task in task_names:
        loader = loaders["train"].get(task)
        if loader is None:
            continue
        for batch in loader:
            if getattr(batch, "get", lambda *_: None)("is_empty", False):
                continue
            batch = batch.to(device)
            sample_ids = [str(value) for value in (getattr(batch, "sample_id", None) or [])]
            real_repr = model_real.encoder(batch)
            shuffle_repr = model_shuffle.encoder(batch)
            delta = (real_repr - shuffle_repr).detach().float().cpu().numpy()
            for index, sample_id in enumerate(sample_ids):
                if sample_id in set(ids):
                    continue
                ids.append(sample_id)
                deltas.append(delta[index])
    return np.asarray(ids), np.stack(deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["csdt", "clst", "b1", "card_table"]
    )
    parser.add_argument("--teacher_real_dir", required=True)
    parser.add_argument("--teacher_shuffle_dir", default=None)
    parser.add_argument("--human_seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--clst_top_k", type=int, default=2)
    parser.add_argument("--gpu_id", default="cpu")
    parser.add_argument("--output", required=True, help="init state .pt (or CARD npz table)")
    parser.add_argument("--delta_csv", default=None, help="CSDT/CLST per-layer diagnostic CSV")
    args = parser.parse_args()

    device = torch.device(args.gpu_id if args.gpu_id != "cpu" and torch.cuda.is_available() else "cpu")
    teacher_real_dir = Path(args.teacher_real_dir)
    teacher_shuffle_dir = Path(args.teacher_shuffle_dir) if args.teacher_shuffle_dir else None

    if args.mode == "card_table":
        model_real, task_names = _build_model(args.human_seed, "animal56", device)
        state_real = _load_teacher_state(teacher_real_dir)
        model_real.load_state_dict(state_real, strict=True)
        model_real.to(device).eval()
        model_shuffle, _ = _build_model(args.human_seed, "animal56", device)
        model_shuffle.load_state_dict(_load_teacher_state(teacher_shuffle_dir), strict=True)
        model_shuffle.to(device).eval()

        params = build_parser().parse_args([])
        setattr(params, "arch", "Graphormer")
        setattr(params, "toxacute_task_scope", "human3")
        setattr(params, "seed", args.human_seed)
        validate_params(params)
        store = _resolve_data_store(params, HUMAN_TARGET_TASKS, required=True)
        collator = DataCollator(spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None)
        loaders = _loaders(params, list(HUMAN_TARGET_TASKS), collator)
        ids, deltas = card_delta_table(model_real, model_shuffle, loaders, list(HUMAN_TARGET_TASKS), collator, device)
        np.savez(args.output, ids=ids, delta=deltas)
        print(f"card delta table: {len(ids)} samples x {deltas.shape[1]} dims -> {args.output}")
        return

    # theta0: fresh seed-matched human3 model (same construction as main()).
    model_human, _ = _build_model(args.human_seed, "human3", torch.device("cpu"))
    theta0 = {key: value.detach().clone() for key, value in model_human.state_dict().items()}
    matched_init_hash = None

    if args.mode != "b1":
        matched_init_hash = _verify_matched_teacher_init(teacher_real_dir, teacher_shuffle_dir)
        print(f"teacher matched initialisation hash: {matched_init_hash}")

    model_real, _ = _build_model(args.human_seed, "animal56", torch.device("cpu"))
    state_real = _load_teacher_state(teacher_real_dir)
    model_real.load_state_dict(state_real, strict=True)
    state_real = model_real.state_dict()

    if args.mode == "csdt":
        model_shuffle, _ = _build_model(args.human_seed, "animal56", torch.device("cpu"))
        model_shuffle.load_state_dict(_load_teacher_state(teacher_shuffle_dir), strict=True)
        state_shuffle = model_shuffle.state_dict()
        merged = apply_csdt(theta0, state_real, state_shuffle, args.alpha)

        if args.delta_csv:
            rows = []
            for key in sorted(theta0):
                if not key.startswith(BACKBONE_PREFIX):
                    continue
                theta0_norm = float(theta0[key].float().norm())
                real_norm = float(state_real[key].float().norm())
                shuffle_norm = float(state_shuffle[key].float().norm())
                delta_norm = float((state_real[key].float() - state_shuffle[key].float()).norm())
                rows.append(
                    {
                        "layer": key,
                        "parameter_group": key,
                        "theta0_norm": theta0_norm,
                        "real_norm": real_norm,
                        "shuffle_norm": shuffle_norm,
                        "semantic_delta_norm": delta_norm,
                        "semantic_delta_ratio": (delta_norm / theta0_norm) if theta0_norm > 0 else float("nan"),
                        "alpha": args.alpha,
                    }
                )
            _write_rows(args.delta_csv, rows)
            anomalous = [row for row in rows if row["semantic_delta_ratio"] > 1.0]
            print(f"csdt: {len(rows)} backbone tensors; anomalous ratio>1: {len(anomalous)}")
    elif args.mode == "b1":
        merged = apply_b1(theta0, state_real)
    elif args.mode == "clst":
        if not teacher_shuffle_dir:
            raise SystemExit("clst requires --teacher_shuffle_dir")
        model_shuffle, _ = _build_model(args.human_seed, "animal56", torch.device("cpu"))
        model_shuffle.load_state_dict(_load_teacher_state(teacher_shuffle_dir), strict=True)

        params = build_parser().parse_args([])
        setattr(params, "arch", "Graphormer")
        setattr(params, "toxacute_task_scope", "human3")
        setattr(params, "seed", args.human_seed)
        validate_params(params)
        seed_everything(args.human_seed)
        store = _resolve_data_store(params, HUMAN_TARGET_TASKS, required=True)
        collator = DataCollator(spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None)
        loaders = _loaders(params, list(HUMAN_TARGET_TASKS), collator)
        device = torch.device(args.gpu_id if args.gpu_id != "cpu" and torch.cuda.is_available() else "cpu")
        model_real.to(device)
        model_shuffle.to(device)
        score_rows, n = clst_layer_scores(model_real, model_shuffle, loaders, collator, device)
        selected = {row["layer"] for row in score_rows[: args.clst_top_k]}

        merged = {key: value.detach().clone() for key, value in theta0.items()}
        for key in merged:
            if key.startswith(BACKBONE_PREFIX + "layers."):
                layer_index = int(key.split(".")[3])
                if layer_index in selected:
                    merged[key] = state_real[key].detach().clone()
        for row in score_rows:
            row["selected"] = int(row["layer"] in selected)
        if args.delta_csv:
            _write_rows(
                args.delta_csv,
                [
                    {
                        "seed": args.human_seed,
                        "layer": row["layer"],
                        "real_rep_norm": row["real_rep_norm_mean"],
                        "shuffle_rep_norm": float("nan"),
                        "real_minus_shuffle_norm": row["real_minus_shuffle_norm_mean"],
                        "semantic_layer_score": row["semantic_layer_score"],
                        "rank": row["rank"],
                        "selected": row["selected"],
                    }
                    for row in score_rows
                ],
            )
        print(f"clst: scored {len(score_rows)} layers on {n} train samples; selected {sorted(selected)}")

    model_human.load_state_dict(merged, strict=True)
    torch.save(model_human.state_dict(), args.output)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "human_seed": args.human_seed,
                "alpha": args.alpha,
                "teacher_matched_init_hash": matched_init_hash if args.mode != "b1" else None,
                "init_state_sha256": state_dict_sha256(model_human),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def _write_rows(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
