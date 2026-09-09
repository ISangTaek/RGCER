"""Baseline-specific molecular features; no learned model substitutes live here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

from molecular_features import schema_metadata

from .utils import canonical_sha256


def morgan_matrix(
    smiles: list[str] | tuple[str, ...],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    include_chirality: bool = True,
) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )
    matrix = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for row, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {text!r}")
        matrix[row] = np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.float32)
    return matrix


def avalon_matrix(smiles: list[str] | tuple[str, ...], *, n_bits: int = 1024) -> np.ndarray:
    from rdkit.Avalon import pyAvalonTools

    matrix = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for row, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {text!r}")
        bitvect = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        matrix[row] = np.asarray(bitvect, dtype=np.float32)
    return matrix


@dataclass(frozen=True)
class AFPSchema:
    atom_widths: tuple[int, ...]
    bond_widths: tuple[int, ...]
    atom_dim: int
    bond_dim: int
    schema_hash: str


def afp_schema() -> AFPSchema:
    metadata = schema_metadata()
    atom_widths = tuple(int(value) + 1 for value in metadata["atom_cardinalities"])
    bond_widths = tuple(int(value) + 1 for value in metadata["bond_cardinalities"])
    payload = {**metadata, "encoding": "concatenated_one_hot", "zero_reserved": "PAD_not_observed"}
    return AFPSchema(atom_widths, bond_widths, sum(atom_widths), sum(bond_widths), canonical_sha256(payload))


def _one_hot_fields(values: torch.Tensor, widths: tuple[int, ...]) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != len(widths):
        raise ValueError("categorical feature tensor does not match schema")
    encoded = []
    for column, width in enumerate(widths):
        ids = values[:, column].long()
        if bool(torch.any(ids <= 0)) or bool(torch.any(ids >= width)):
            raise ValueError("Actual atoms/bonds must use schema ids 1..max; 0 is reserved for padding")
        encoded.append(F.one_hot(ids, num_classes=width).float())
    return torch.cat(encoded, dim=1)


def afp_graph(record: dict, labels: np.ndarray | None = None, mask: np.ndarray | None = None) -> Data:
    schema = afp_schema()
    atom_ids = record["x"].long()
    spatial = record["spatial_pos"].long()
    directed = (spatial == 1).nonzero(as_tuple=False)
    if directed.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, schema.bond_dim), dtype=torch.float32)
    else:
        edge_index = directed.t().contiguous()
        bond_ids = record["attn_edge_type"][directed[:, 0], directed[:, 1], :].long()
        edge_attr = _one_hot_fields(bond_ids, schema.bond_widths)
    data = Data(x=_one_hot_fields(atom_ids, schema.atom_widths), edge_index=edge_index, edge_attr=edge_attr)
    if labels is not None:
        data.y = torch.as_tensor(labels, dtype=torch.float32).view(1, -1)
    if mask is not None:
        data.label_mask = torch.as_tensor(mask, dtype=torch.float32).view(1, -1)
    return data
