"""Independent checkpoint loading and authorized validation prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from torch_geometric.data import Batch

from .checkpoint import validate_checkpoint_contract
from .constants import HUMAN3_TASKS, TOXACUTE_TASKS
from .data import BaselinePartition, load_authorized_validation
from .features import afp_graph, avalon_matrix, morgan_matrix
from .models.afp import AttentiveFPRegressor
from .models.dmpnn import ChempropDMPNN
from .models.grover import create_grover, grover_batch
from .models.toxacol import ToxACoLNet


def load_checkpoint(path: str | Path, *, expected_method: str | None = None):
    checkpoint = Path(path)
    if checkpoint.suffix == ".joblib":
        payload = joblib.load(checkpoint)
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    scaler = validate_checkpoint_contract(payload, expected_method=expected_method)
    return payload, scaler


def _batches(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield range(start, min(start + batch_size, length))


def _torch_predictions(model, partition, scaler, device, batch_size, method, source_args=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for indices in _batches(len(partition.sample_ids), batch_size):
            index_list = list(indices)
            if method == "afp":
                graphs = [afp_graph(partition.graph_records[index]) for index in index_list]
                output = model(Batch.from_data_list(graphs).to(device))
            elif method == "grover":
                smiles = [partition.smiles[index] for index in index_list]
                output = model(grover_batch(smiles, source_args, device), [None] * len(smiles))
            else:
                smiles = [partition.smiles[index] for index in index_list]
                output = model(smiles)
            rows.append(output.detach().cpu().numpy())
    return scaler.inverse_transform(np.concatenate(rows, axis=0))


def predict_partition(
    checkpoint: str | Path,
    partition: BaselinePartition,
    *,
    device: str = "cpu",
    source_root: str | Path | None = None,
    expected_method: str | None = None,
) -> tuple[np.ndarray, dict, object]:
    payload, scaler = load_checkpoint(checkpoint, expected_method=expected_method)
    method = payload["method"]
    config = payload["config"]
    training = config["training"]
    torch_device = torch.device(device)
    batch_size = int(training.get("batch_size") or 64)
    if method == "rf":
        features = morgan_matrix(partition.smiles, include_chirality=True)
        predictions = np.column_stack([model.predict(features) for model in payload["models"]])
    elif method == "afp":
        schema = config["feature_schema"]
        model = AttentiveFPRegressor(
            int(schema["atom_dim"]), int(schema["bond_dim"]), dropout=float(training["dropout"])
        ).to(torch_device)
        model.load_state_dict(payload["model_state_dict"])
        predictions = _torch_predictions(model, partition, scaler, torch_device, batch_size, method)
    elif method == "dmpnn":
        if source_root is None:
            raise ValueError("D-MPNN checkpoint inference requires --source-root for Chemprop 1.6.1")
        model = ChempropDMPNN(source_root, torch_device, dropout=float(training["dropout"])).to(torch_device)
        model.load_state_dict(payload["model_state_dict"])
        predictions = _torch_predictions(model, partition, scaler, torch_device, batch_size, method)
    elif method == "grover":
        if source_root is None:
            raise ValueError("GROVER checkpoint inference requires --source-root for the locked source commit")
        model, args = create_grover(source_root, torch_device, dropout=float(training["dropout"]))
        model.load_state_dict(payload["model_state_dict"])
        predictions = _torch_predictions(model, partition, scaler, torch_device, batch_size, method, args)
    elif method == "toxacol":
        state = payload["model_state_dict"]
        if "adjacency" not in state or "endpoint_features" not in state:
            raise ValueError("TOXACol checkpoint is missing train-derived graph buffers")
        model = ToxACoLNet(
            state["adjacency"].cpu().numpy(),
            state["endpoint_features"].cpu().numpy(),
            dropout=float(training["dropout"]),
        ).to(torch_device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            scaled = model(torch.as_tensor(avalon_matrix(partition.smiles), device=torch_device)).cpu().numpy()
        all_predictions = scaler.inverse_transform(scaled)
        columns = [TOXACUTE_TASKS.index(task) for task in HUMAN3_TASKS]
        predictions = all_predictions[:, columns]
    else:
        raise ValueError(f"Unknown checkpoint method: {method}")
    return np.asarray(predictions, dtype=np.float64), payload, scaler


def predict_validation(
    checkpoint: str | Path,
    datastore: str | Path,
    *,
    split: str = "validation",
    device: str = "cpu",
    source_root: str | Path | None = None,
    expected_method: str | None = None,
):
    payload, _ = load_checkpoint(checkpoint, expected_method=expected_method)
    partition = load_authorized_validation(datastore, payload["method"], split=split)
    try:
        predictions, payload, scaler = predict_partition(
            checkpoint,
            partition,
            device=device,
            source_root=source_root,
            expected_method=expected_method,
        )
        return partition, predictions, payload, scaler
    except Exception:
        close = getattr(partition.graph_records, "close", None)
        if close is not None:
            close()
        raise
