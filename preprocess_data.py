"""Preprocess toxacute molecules into versioned Graphormer inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import algos
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

from experiment_config import (
    TOXACUTE_PREPROCESSED_DIR,
    TOXACUTE_RAW_CSV,
)
from architecture.toxacute_tasks import TOXACUTE_TASKS
from molecular_features import (
    BOND_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    atom_features,
    bond_features,
    canonicalize_smiles,
    scaffold_from_mol,
)
from split_manifest import create_split_manifest, write_manifest


def one_of_k_encoding_unk(x, allowable_set):
    """Retained for compatibility with older external callers."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == item for item in allowable_set]


def convert_to_single_emb_offline(x, offset=1):
    """Compatibility shim; categorical fields are now encoded explicitly."""
    del offset
    return x.long()


def _generate_scaffold(smiles, include_chirality=False):
    del include_chirality
    mol, _ = canonicalize_smiles(smiles)
    return scaffold_from_mol(mol)


def _is_valid_smiles(value) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _is_valid_label(value) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _records_from_dataframe(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    records = []
    errors = []
    for row_index, row in df.iterrows():
        raw_smiles = row.get("smiles")
        if not _is_valid_smiles(raw_smiles):
            errors.append({"row_index": int(row_index), "error": "missing_smiles"})
            continue
        try:
            mol, canonical_smiles = canonicalize_smiles(str(raw_smiles))
            scaffold = scaffold_from_mol(mol)
        except Exception as exc:
            errors.append(
                {
                    "row_index": int(row_index),
                    "smiles": str(raw_smiles),
                    "error": str(exc),
                }
            )
            continue
        records.append(
            {
                "sample_id": f"row_{int(row_index)}",
                "row_index": int(row_index),
                "raw_smiles": str(raw_smiles),
                "canonical_smiles": canonical_smiles,
                "scaffold": scaffold,
            }
        )
    return records, errors


def _build_edge_tensors(mol: Chem.Mol):
    num_atoms = mol.GetNumAtoms()
    feature_dim = len(BOND_FEATURE_NAMES)
    adjacency = np.zeros((num_atoms, num_atoms), dtype=np.int64)
    edge_feature_matrix = np.zeros((num_atoms, num_atoms, feature_dim), dtype=np.int64)
    attn_edge_type = torch.zeros((num_atoms, num_atoms, feature_dim), dtype=torch.long)
    edge_index_values = []
    edge_attr_values = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        # edge_attr/attn_edge_type use 1-based categories; zero is reserved
        # for no edge and padding. gen_edge_input uses zero-based categories.
        zero_based_features = np.asarray(features, dtype=np.int64) - 1
        for source, target in ((begin, end), (end, begin)):
            adjacency[source, target] = 1
            edge_feature_matrix[source, target, :] = zero_based_features
            attn_edge_type[source, target, :] = torch.tensor(features, dtype=torch.long)
            edge_index_values.append((source, target))
            edge_attr_values.append(features)

    if edge_index_values:
        edge_index = torch.tensor(edge_index_values, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_values, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, feature_dim), dtype=torch.long)
    return adjacency, edge_index, edge_attr, attn_edge_type, edge_feature_matrix


def get_graph_data_from_smiles(
    smiles_string,
    label_val,
    convert_x_fn=None,
    *,
    sample_id="inference_sample",
    task_name=None,
    max_path_distance=8,
):
    """Create one graph with atom, direct-bond, and multi-hop bond features."""
    del convert_x_fn
    mol, canonical_smiles = canonicalize_smiles(str(smiles_string))
    if mol.GetNumAtoms() == 0:
        raise ValueError("SMILES produced an empty molecule")

    atom_matrix = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.long)
    adjacency, edge_index, edge_attr, attn_edge_type, edge_feature_matrix = _build_edge_tensors(mol)
    spatial_pos_np, path_np = algos.floyd_warshall(np.ascontiguousarray(adjacency))
    # The bundled Cython implementation assumes max_dist is at least the
    # longest finite path and writes without a bounds check.  Call it with a
    # safe per-molecule capacity, then keep the configured compact prefix.
    finite_distances = spatial_pos_np[spatial_pos_np < 510]
    required_path_distance = int(finite_distances.max()) if finite_distances.size else 0
    cython_path_distance = max(int(max_path_distance), required_path_distance)
    edge_input_full = algos.gen_edge_input(
        cython_path_distance,
        np.ascontiguousarray(path_np),
        np.ascontiguousarray(edge_feature_matrix),
    )
    edge_input_np = edge_input_full[:, :, : int(max_path_distance), :]

    degree = torch.from_numpy(adjacency.sum(axis=1).astype(np.int64))
    return Data(
        x=atom_matrix,
        edge_index=edge_index,
        edge_attr=edge_attr,
        in_degree=degree,
        out_degree=degree.clone(),
        spatial_pos=torch.from_numpy(spatial_pos_np.astype(np.int64)),
        attn_edge_type=attn_edge_type,
        edge_input=torch.from_numpy(edge_input_np.astype(np.int64)),
        y=torch.tensor([float(label_val)], dtype=torch.float),
        label=float(label_val),
        smiles=str(smiles_string),
        raw_smiles=str(smiles_string),
        canonical_smiles=canonical_smiles,
        scaffold_smiles=scaffold_from_mol(mol),
        scaffold=scaffold_from_mol(mol),
        sample_id=str(sample_id),
        task_name=task_name,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def preprocess_and_save(
    raw_csv_path,
    task_list_str,
    output_dir_base,
    *,
    splitting="scaffold",
    valid_size=0.1,
    calibration_size=0.1,
    test_size=0.1,
    seed=42,
    max_path_distance=8,
):
    raw_csv_path = Path(raw_csv_path)
    output_dir_base = Path(output_dir_base)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV data file not found: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    if task_list_str == "all":
        tasks = [column for column in df.columns[6:]]
    else:
        tasks = [task.strip() for task in task_list_str.split(",") if task.strip()]
    if not tasks:
        raise ValueError("No tasks provided")
    missing_tasks = [task for task in tasks if task not in df.columns]
    if missing_tasks:
        raise KeyError(f"Tasks not found in CSV: {missing_tasks}")

    records, errors = _records_from_dataframe(df)
    if not records:
        raise ValueError("No valid SMILES records found")
    manifest = create_split_manifest(
        records,
        splitting=splitting,
        ratios={
            "train": 1.0 - valid_size - calibration_size - test_size,
            "validation": valid_size,
            "calibration": calibration_size,
            "test": test_size,
        },
        seed=seed,
    )
    output_dir_base.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir_base / "split_manifest.json"
    write_manifest(manifest, manifest_path)
    if errors:
        (output_dir_base / "preprocess_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    record_by_index = {record["row_index"]: record for record in records}
    print(f"Reading raw data from: {raw_csv_path}")
    print(f"Saving preprocessed data to: {output_dir_base.resolve()}")
    print(f"Global split manifest: {manifest_path}")
    print(f"Tasks: {tasks}")

    for task_name in tasks:
        task_output_dir = output_dir_base / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in task_output_dir.glob("data_*.pt"):
            stale_file.unlink()
        processed = 0
        skipped = 0
        for row_index, record in record_by_index.items():
            label_value = df.at[row_index, task_name]
            if not _is_valid_label(label_value):
                skipped += 1
                continue
            try:
                graph_data = get_graph_data_from_smiles(
                    record["raw_smiles"],
                    float(label_value),
                    sample_id=record["sample_id"],
                    task_name=task_name,
                    max_path_distance=max_path_distance,
                )
                torch.save(graph_data, task_output_dir / f"data_{row_index}.pt")
                processed += 1
            except Exception as exc:
                skipped += 1
                print(f"Error processing row {row_index} for {task_name}: {exc}")
        print(f"{task_name}: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline molecular Graphormer preprocessing")
    parser.add_argument("--raw_csv_path", type=str, default=TOXACUTE_RAW_CSV)
    parser.add_argument("--task_list", type=str, default=",".join(TOXACUTE_TASKS))
    parser.add_argument("--output_dir", type=str, default=TOXACUTE_PREPROCESSED_DIR)
    parser.add_argument("--splitting", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--calibration_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_path_distance", type=int, default=8)
    args = parser.parse_args()
    preprocess_and_save(
        args.raw_csv_path,
        args.task_list,
        args.output_dir,
        splitting=args.splitting,
        valid_size=args.valid_size,
        calibration_size=args.calibration_size,
        test_size=args.test_size,
        seed=args.seed,
        max_path_distance=args.max_path_distance,
    )
