"""Official GROVER commit adapter and safe pretrained-state loading."""

from __future__ import annotations

import importlib
import subprocess
import sys
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import torch

from ..constants import GROVER_BASE_SHA256, GROVER_COMMIT
from ..utils import sha256_file


def activate_grover(source_root: str | Path) -> Path:
    root = Path(source_root).resolve()
    if not (root / "grover" / "model" / "models.py").exists():
        raise FileNotFoundError(f"Official GROVER source tree not found: {root}")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != GROVER_COMMIT:
        raise RuntimeError(f"Expected GROVER source commit {GROVER_COMMIT}, found {commit}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def grover_args(device: torch.device, dropout: float = 0.1) -> Namespace:
    return Namespace(
        embedding_output_type="both", backbone="dualtrans", hidden_size=800, bias=False, depth=6,
        activation="PReLU", undirected=False, dense=False, dropout=dropout, num_attn_head=4,
        num_mt_block=1, cuda=device.type == "cuda", self_attention=False, attn_hidden=128,
        attn_out=4, features_only=False, features_size=0, features_dim=0, ffn_num_layers=2,
        ffn_hidden_size=300, output_size=3, dataset_type="regression", dist_coff=0.1,
        no_cache=True, bond_drop_rate=0.0, input_layer="fc",
    )


def create_grover(source_root: str | Path, device: torch.device, *, dropout: float = 0.1):
    activate_grover(source_root)
    from grover.model.models import GroverFinetuneTask

    args = grover_args(device, dropout)
    return GroverFinetuneTask(args).to(device), args


def validate_grover_encoder_state(state: dict, own_state: dict) -> None:
    expected = {key for key in own_state if key.startswith("grover.")}
    supplied = {key for key in state if key.startswith("grover.")}
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    shape_mismatch = sorted(
        key for key in expected & supplied if tuple(state[key].shape) != tuple(own_state[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "GROVER encoder state is incomplete or incompatible: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={shape_mismatch}"
        )


def build_grover(source_root: str | Path, weights: str | Path, device: torch.device, *, dropout: float = 0.1):
    actual_sha256 = sha256_file(weights)
    if actual_sha256 != GROVER_BASE_SHA256:
        raise ValueError(
            f"GROVER_base SHA-256 mismatch: {actual_sha256} != {GROVER_BASE_SHA256}"
        )
    model, args = create_grover(source_root, device, dropout=dropout)
    namespace_context = nullcontext()
    try:
        from torch.serialization import safe_globals
        namespace_context = safe_globals([Namespace])
    except ImportError:
        pass
    with namespace_context:
        checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("GROVER weight file is not the expected official checkpoint structure")
    state = checkpoint["state_dict"]
    own = model.state_dict()
    validate_grover_encoder_state(state, own)
    encoder_state = {key: state[key] for key in own if key.startswith("grover.")}
    result = model.load_state_dict(encoder_state, strict=False)
    loaded_grover = sorted(encoder_state)
    allowed_missing_prefixes = (
        "mol_atom_from_atom_ffn.",
        "mol_atom_from_bond_ffn.",
        "readout.cached_zero_vector",
    )
    disallowed_missing = sorted(
        key for key in result.missing_keys if not key.startswith(allowed_missing_prefixes)
    )
    if disallowed_missing or result.unexpected_keys:
        raise ValueError(
            "GROVER pretrained load produced non-head state drift: "
            f"missing={disallowed_missing}, unexpected={sorted(result.unexpected_keys)}"
        )
    audit = {
        "source_commit": GROVER_COMMIT,
        "weights_sha256": actual_sha256,
        "approved_weights_sha256": GROVER_BASE_SHA256,
        "loaded_key_count": len(encoder_state),
        "loaded_encoder_key_count": len(loaded_grover),
        "missing_keys": sorted(result.missing_keys),
        "allowed_missing_prefixes": list(allowed_missing_prefixes),
        "disallowed_missing_keys": disallowed_missing,
        "unexpected_keys": sorted(result.unexpected_keys),
        "shape_mismatch": [],
        "features_dim": 0,
    }
    return model, args, audit


def grover_batch(smiles: list[str], args: Namespace, device: torch.device):
    from grover.data.molgraph import mol2graph

    batch = mol2graph(smiles, {}, args).get_components()
    return tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)


def optimizer_with_coverage(model, *, init_lr: float, weight_decay: float, fine_tune_coff: float = 1.0):
    base, head, seen = [], [], set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"Duplicate trainable parameter: {name}")
        seen.add(id(parameter))
        (base if name.startswith("grover.") else head).append(parameter)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected or not base or not head:
        raise RuntimeError("GROVER optimizer groups do not cover encoder and head exactly once")
    optimizer = torch.optim.Adam(
        [
            {"params": base, "lr": init_lr * fine_tune_coff},
            {"params": head, "lr": init_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer, {"base_parameters": len(base), "head_parameters": len(head), "unique_parameters": len(seen)}
