"""Fresh-process model initialization hash (review §12 Test C, §42).

Run the same command twice with the same seed: the printed SHA256 must be
identical.  A different seed must produce a different hash.  This is the
regression guard for the P0 bug where ``--seed`` was applied only after
prediction-head construction, so identical seeds produced different
initial weights across processes.

Example:
    python scripts/hash_initial_model.py --arch Graphormer_rgcer --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from main import _build_model_components, effective_prediction_mode
from reproducibility import seed_everything, state_dict_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash initial model state for reproducibility audit")
    parser.add_argument("--dataset", default="toxacute")
    parser.add_argument(
        "--arch",
        choices=["Graphormer", "Graphormer_prompt", "Graphormer_rgcer"],
        default="Graphormer_rgcer",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--head_hidden_dim", type=int, default=96)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    args = parser.parse_args()

    # Hash must come from a fresh process on a quiet RNG: nothing else may
    # consume global RNG before this point.
    seed_everything(args.seed)

    params = SimpleNamespace(
        dataset=args.dataset,
        arch=args.arch,
        hidden_dim=args.hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        prediction_mode="quantile",
    )
    params.prediction_mode = effective_prediction_mode(params)

    tasks = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
    device = torch.device("cpu")
    encoder_class, architecture_class, decoders = _build_model_components(params, tasks, device)
    arch_args = SimpleNamespace(
        prediction_mode=params.prediction_mode,
        hidden_dim=args.hidden_dim,
        a_heads=4,
        a_layers=2,
        t_layers=1,
        mid_dim=192,
        prompt_heads=4,
        prompt_dropout=0.1,
        adapter_ratio=0.25,
        use_factorized_prompt=False,
        task_residual_scale=0.1,
        router_dim=None,
        router_top_k=8,
        router_temperature=1.0,
        exclude_target_from_sources=True,
        response_hidden_dim=96,
        edge_bias_mode="path",
        spatial_pos_clip=20,
        auxiliary_metadata_overrides=None,
        rgcer_fallback_space="prediction",
        rgcer_transfer_mechanism="endpoint_router",
        rgcer_use_source_response=True,
        rgcer_use_target_response=True,
        rgcer_use_molecule_query=True,
        rgcer_use_sparse_routing=True,
        rgcer_use_null_route=True,
        rgcer_use_film=True,
        rgcer_use_adapter=True,
    )
    model = architecture_class(tasks, encoder_class, decoders, device, arch_args).to(device)

    digest = state_dict_sha256(model)
    print(f"seed={args.seed}")
    print(f"initial_model_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
