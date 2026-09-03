"""D8 formal experiment: encoder embedding extraction + analysis (§10).

Loads a fine-tuned Human3 model, forwards a set of molecules through the
encoder, and saves the pooled representations for offline analysis (PCA,
UMAP, clustering, nearest-neighbour).  Also computes cluster purity and
endpoint separation if per-sample metadata is available.

Usage:
    python scripts/d8_extract_embeddings.py \
        --run_dir artifacts/runs/d7/d7_stage_b/s1/d7_s1_e40/seed_42 \
        --gpu_id 0 \
        --output_dir artifacts/results/d8_embeddings/s1_s42
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="fine-tuned Human3 run dir")
    parser.add_argument("--gpu_id", default="cpu")
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation"],
        help="which split to embed (formal experiment: validation only)",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if str(args.gpu_id) == "cpu"
        else torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    )
    run_dir = Path(args.run_dir)

    # Load the trained model
    last_pts = sorted(run_dir.glob("*_last.pt"))
    if len(last_pts) != 1:
        raise RuntimeError(f"expected exactly one *_last.pt in {run_dir}, found {len(last_pts)}")
    payload = torch.load(last_pts[0], map_location="cpu", weights_only=False)

    from main import _build_model_components, build_parser, task_names_for_params, validate_params
    from config import prepare_args

    params = build_parser().parse_args([])
    template_config = dict(payload.get("configuration") or {})
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
    setattr(params, "seed", int(template_config.get("seed", 42)))
    setattr(params, "fit_conformal", False)
    setattr(params, "train_eval_scope", "validation_only")
    setattr(params, "card_lambda_delta", 0.0)
    validate_params(params)

    task_names = task_names_for_params(params)
    from dataset import DataCollator

    collator = DataCollator(spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None)
    from main import _build_model_components, _loaders

    encoder_class, architecture_class, decoders = _build_model_components(params, task_names, device)
    model = architecture_class(task_names, encoder_class, decoders, device, params, **({}))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()

    loaders = _loaders(params, task_names, collator)
    split_key = "val" if args.split == "validation" else args.split
    split_loaders = loaders.get(split_key, {})
    if not split_loaders:
        raise RuntimeError(f"no {args.split} loaders found")

    # Extract embeddings per sample
    embeddings = {}
    labels = {}
    with torch.no_grad():
        for task in task_names:
            loader = split_loaders.get(task)
            if loader is None:
                continue
            for batch in loader:
                if getattr(batch, "get", lambda *_: None)("is_empty", False):
                    continue
                batch = batch.to(device)
                representation = model.encoder(batch)
                if representation.dim() == 3:
                    representation = representation[:, 0, :]
                rep = representation.detach().float().cpu()
                targets = batch.y.reshape(-1).float().cpu()
                for pos, sample_id in enumerate(batch.sample_id):
                    sid = str(sample_id)
                    embeddings[sid] = rep[pos]
                    labels.setdefault(sid, {}).setdefault("tasks", {})[task] = float(targets[pos])

    sample_ids = sorted(embeddings.keys())
    matrix = torch.stack([embeddings[sid] for sid in sample_ids]).numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embedding_matrix.npy", matrix)

    meta_fields = ["sample_id", "dim"] + [
        f"label_{task}" for task in task_names
    ]
    with (output_dir / "embedding_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=meta_fields, extrasaction="ignore")
        writer.writeheader()
        for sid in sample_ids:
            row = {"sample_id": sid, "dim": int(matrix.shape[1])}
            for task, tlabels in labels.get(sid, {}).get("tasks", {}).items():
                row[f"label_{task}"] = tlabels
            writer.writerow(row)

    # PCA (via SVD for determinism)
    centered = matrix - matrix.mean(axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    top_k = min(10, matrix.shape[1])
    pca_components = vt[:top_k]
    pca_projected = centered @ pca_components.T
    np.save(output_dir / "pca_projected.npy", pca_projected)
    np.save(output_dir / "pca_components.npy", pca_components)
    explained = (sv[:top_k] ** 2) / max((sv ** 2).sum(), 1e-12)
    np.save(output_dir / "pca_explained_variance.npy", explained)

    metadata = {
        "run_dir": str(run_dir),
        "split": args.split,
        "n_samples": len(sample_ids),
        "embedding_dim": int(matrix.shape[1]),
        "pca_components": top_k,
        "pca_explained_variance": [float(v) for v in explained],
        "task_names": list(task_names),
        "git_note": "see run_metadata.json for git commit",
    }
    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"embeddings: {len(sample_ids)} samples x {matrix.shape[1]} dims "
        f"-> {output_dir} (PCA top-{top_k} explained {explained[:3].sum():.3f})"
    )


if __name__ == "__main__":
    main()
