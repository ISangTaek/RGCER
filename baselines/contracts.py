"""Stable model, task, and feature contracts stored in every checkpoint."""

from __future__ import annotations

from .constants import CHEMPROP_COMMIT, GROVER_COMMIT, HUMAN3_TASKS, TOXACOL_COMMIT, TOXACUTE_TASKS
from .features import afp_schema
from .models.toxacol import endpoint_feature_matrix
from .utils import canonical_sha256


MODEL_TYPES = {
    "rf": "sklearn.RandomForestRegressor.per_human3_task",
    "afp": "torch_geometric.AttentiveFPRegressor",
    "dmpnn": "chemprop.v1.6.1.MPNEncoder.native_2layer_FFN",
    "grover": "grover.GroverFinetuneTask.base",
    "toxacol": "Acute_Toxicity_FSL.CorrelationNet.joint59",
}


def expected_task_names(method: str) -> tuple[str, ...]:
    return TOXACUTE_TASKS if method == "toxacol" else HUMAN3_TASKS


def expected_feature_schema(method: str) -> dict:
    if method == "rf":
        payload = {
            "name": "Morgan",
            "radius": 2,
            "n_bits": 2048,
            "include_chirality": True,
        }
    elif method == "afp":
        schema = afp_schema()
        payload = {
            "name": "toxacute_categorical_concatenated_one_hot",
            "atom_dim": schema.atom_dim,
            "bond_dim": schema.bond_dim,
            "categorical_schema_hash": schema.schema_hash,
        }
    elif method == "dmpnn":
        payload = {
            "name": "chemprop_native_molgraph",
            "source_commit": CHEMPROP_COMMIT,
            "atom_messages": False,
            "hidden_size": 300,
            "depth": 3,
            "ffn_num_layers": 2,
            "ffn_hidden_size": 300,
        }
    elif method == "grover":
        payload = {
            "name": "grover_native_molgraph",
            "source_commit": GROVER_COMMIT,
            "backbone": "dualtrans",
            "hidden_size": 800,
            "depth": 6,
            "features_dim": 0,
            "ffn_num_layers": 2,
            "ffn_hidden_size": 300,
        }
    elif method == "toxacol":
        _, endpoint_schema = endpoint_feature_matrix()
        payload = {
            "name": "Avalon1024_plus_endpoint26",
            "fingerprint_bits": 1024,
            "endpoint_schema_hash": endpoint_schema["schema_hash"],
            "source_commit": TOXACOL_COMMIT,
        }
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    return {**payload, "schema_hash": canonical_sha256(payload)}
