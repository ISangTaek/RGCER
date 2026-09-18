"""TOXACol correlation network, transcribed from the locked upstream commit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from architecture.toxacute_tasks import parse_toxacute_task_name

from ..constants import TOXACUTE_TASKS, TOXACOL_COMMIT
from ..utils import canonical_sha256

SPECIES = (
    "mouse", "mammal (species unspecified)", "guinea pig", "rat", "rabbit", "dog", "cat",
    "bird - wild", "quail", "duck", "chicken", "man", "women", "human", "frog",
)
ROUTES = ("intraperitoneal", "intravenous", "oral", "unreported", "skin", "subcutaneous", "intramuscular", "parenteral")
MEASUREMENTS = ("LD50", "LDLo", "TDLo")


def endpoint_feature_matrix(task_names=TOXACUTE_TASKS, *, extend_vocabulary=False) -> tuple[np.ndarray, dict]:
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("task names must be nonempty and unique")
    parts = [task.rsplit("_", 2) for task in task_names]
    if any(len(p) != 3 or not all(p) for p in parts):
        raise ValueError("expected species_route_measurement")
    parts = [("bird - wild" if p[0] == "bird-wild" else p[0], p[1], p[2]) for p in parts]
    vocab = [list(SPECIES), list(ROUTES), list(MEASUREMENTS)]
    if extend_vocabulary:
        for i in range(3):
            vocab[i].extend(sorted({p[i] for p in parts} - set(vocab[i])))
    rows = []
    for task, part in zip(task_names, parts):
        if not extend_vocabulary:
            parse_toxacute_task_name(task)  # retain legacy task validation
        vector = np.zeros(sum(map(len, vocab)), dtype=np.float32)
        offset = 0
        for category, value in zip(vocab, part):
            vector[offset + category.index(value)] = 1.0
            offset += len(category)
        rows.append(vector)
    matrix = np.stack(rows)
    schema = dict(zip(("species", "routes", "measurements"), vocab))
    schema["schema_hash"] = canonical_sha256(schema)
    return matrix, schema


def task_adjacency(train_labels: np.ndarray, *, min_shared: int = 15, pcc_threshold: float = 0.75) -> tuple[np.ndarray, dict]:
    labels = np.asarray(train_labels, dtype=np.float64)
    width = labels.shape[1]
    adjacency = np.zeros((width, width), dtype=np.float32)
    constant_pairs = 0
    for left in range(width):
        for right in range(left + 1, width):
            mask = np.isfinite(labels[:, left]) & np.isfinite(labels[:, right])
            shared = int(mask.sum())
            if shared < min_shared:
                continue
            x, y = labels[mask, left], labels[mask, right]
            if np.std(x) <= 0 or np.std(y) <= 0:
                constant_pairs += 1
                continue
            pcc = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(pcc) and pcc >= pcc_threshold:
                adjacency[left, right] = adjacency[right, left] = 1.0
    with_identity = adjacency + np.eye(width, dtype=np.float32)
    degrees = np.sum(with_identity, axis=1)
    normalized = with_identity * (degrees ** -0.5)[:, None] * (degrees ** -0.5)[None, :]
    audit = {
        "min_shared": min_shared,
        "pcc_threshold": pcc_threshold,
        "undirected_edges": int(np.sum(adjacency) // 2),
        "constant_or_undefined_pairs": constant_pairs,
        "normalization": "D^-1/2(A+I)D^-1/2",
    }
    return normalized.astype(np.float32), audit


class FCL(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return F.relu(self.dropout(self.batch_norm(self.linear(x))))


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(0.1 * (torch.rand(input_dim, output_dim) - 0.5))

    def forward(self, x, adjacency):
        return adjacency @ (x @ self.weight)


class CorrelationLayer(nn.Module):
    def __init__(self, num_tasks: int, feature_dim: int):
        super().__init__()
        self.weight = nn.Parameter(0.1 * (torch.rand(num_tasks, feature_dim) - 0.5))

    def forward(self, molecule, task_features):
        bridge = molecule @ F.leaky_relu(task_features.t(), negative_slope=0.1)
        return bridge @ self.weight + molecule


class ToxACoLNet(nn.Module):
    source_commit = TOXACOL_COMMIT

    def __init__(self, adjacency: np.ndarray, endpoint_features: np.ndarray, *, dropout: float = 0.1):
        super().__init__()
        adjacency = np.asarray(adjacency)
        endpoint_features = np.asarray(endpoint_features)
        if (endpoint_features.ndim != 2 or min(endpoint_features.shape) < 1
                or adjacency.shape != (len(endpoint_features), len(endpoint_features))
                or not np.isfinite(adjacency).all() or not np.isfinite(endpoint_features).all()
                or not np.allclose(adjacency, adjacency.T, rtol=0, atol=1e-7)
                or np.any(adjacency < 0) or np.any(np.diag(adjacency) <= 0)):
            raise ValueError("invalid task adjacency/endpoint features")
        num_tasks, feature_width = endpoint_features.shape
        dimensions = (1024, 768, 512, 384, 64)
        task_dimensions = (feature_width, 768, 512, 384, 64)
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.register_buffer("endpoint_features", torch.as_tensor(endpoint_features, dtype=torch.float32))
        self.dnn = nn.ModuleList(FCL(dimensions[i], dimensions[i + 1], dropout) for i in range(4))
        self.gcn = nn.ModuleList(GCNLayer(task_dimensions[i], task_dimensions[i + 1]) for i in range(4))
        self.correlation = nn.ModuleList(CorrelationLayer(num_tasks, dimensions[i + 1]) for i in range(4))
        self.tail_weight = nn.Parameter(0.1 * torch.rand(64, num_tasks))

    def forward(self, fingerprints):
        molecule = fingerprints
        tasks = self.endpoint_features
        for layer_index, (dnn, gcn, correlation) in enumerate(zip(self.dnn, self.gcn, self.correlation)):
            molecule = dnn(molecule)
            tasks = gcn(tasks, self.adjacency)
            molecule = correlation(molecule, tasks)
            if layer_index < len(self.gcn) - 1:
                tasks = F.leaky_relu(tasks, negative_slope=0.1)
        return molecule @ tasks.t() + molecule @ self.tail_weight


def toxacol_learning_rate(epoch: int, bounds: list[int], values: list[float]) -> float:
    if len(values) != len(bounds) + 1:
        raise ValueError("LUT requires one more learning-rate value than bounds")
    for index, bound in enumerate(bounds):
        if epoch < bound:  # zero-based epoch; first strict upper bound wins
            return float(values[index])
    return float(values[-1])
