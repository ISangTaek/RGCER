"""Dataset, global split, and CPU-only Graphormer collation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Data

from split_manifest import load_manifest
from molecular_features import BOND_FEATURE_NAMES


class PreprocessedDatasetWrapper(Dataset):
    def __init__(self, task_data_dir):
        self.task_data_dir = Path(task_data_dir)
        if not self.task_data_dir.is_dir():
            raise FileNotFoundError(f"Preprocessed data directory not found: {self.task_data_dir}")
        self.processed_files = sorted(
            self.task_data_dir.glob("data_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not self.processed_files:
            print(f"Warning: no preprocessed .pt files found in {self.task_data_dir}")
        self._sample_ids = {}

    def __len__(self):
        return len(self.processed_files)

    def __getitem__(self, index):
        file_path = self.processed_files[index]
        try:
            return torch.load(file_path, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Error loading preprocessed file {file_path}: {exc}") from exc

    def get_sample_id(self, index):
        if index not in self._sample_ids:
            self._sample_ids[index] = str(self[index].sample_id)
        return self._sample_ids[index]


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """Backward-compatible sampler retained for external inference callers."""

    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DataCollator:
    """Pad graph tensors on CPU; device transfer belongs in Trainer."""

    def __init__(self, spatial_pos_max_clip=20, max_node_filter=None):
        self.spatial_pos_max_clip = int(spatial_pos_max_clip)
        self.max_node_filter = max_node_filter

    @staticmethod
    def _pad_node_features(x, padlen):
        padded = x.new_zeros((padlen, x.size(-1)))
        padded[: x.size(0)] = x
        return padded

    @staticmethod
    def _pad_degree(x, padlen):
        padded = x.new_zeros((padlen,))
        padded[: x.size(0)] = x + 1
        return padded

    def _pad_spatial_pos(self, x, padlen):
        padded = x.new_zeros((padlen, padlen))
        values = x.clamp_min(-1).clamp_max(self.spatial_pos_max_clip) + 1
        padded[: x.size(0), : x.size(0)] = values
        return padded

    @staticmethod
    def _pad_attn_edge_type(x, padlen):
        padded = x.new_zeros((padlen, padlen, x.size(-1)))
        padded[: x.size(0), : x.size(0)] = x
        return padded

    @staticmethod
    def _pad_edge_input(x, padlen, max_path_distance):
        padded = x.new_zeros((padlen, padlen, max_path_distance, x.size(-1)))
        values = x.clamp_min(-1) + 1
        path_length = min(x.size(2), max_path_distance)
        padded[: x.size(0), : x.size(1), :path_length] = values[:, :, :path_length]
        return padded

    def _empty_batch(self):
        bond_fields = len(BOND_FEATURE_NAMES)
        return Data(
            x=torch.empty((0, 0, 0), dtype=torch.long),
            in_degree=torch.empty((0, 0), dtype=torch.long),
            out_degree=torch.empty((0, 0), dtype=torch.long),
            spatial_pos=torch.empty((0, 0, 0), dtype=torch.long),
            attn_bias=torch.empty((0, 1, 1), dtype=torch.float),
            attn_edge_type=torch.empty((0, 0, 0, bond_fields), dtype=torch.long),
            edge_input=torch.empty((0, 0, 0, 1, bond_fields), dtype=torch.long),
            y=torch.empty((0, 1), dtype=torch.float),
            padding_mask=torch.empty((0, 0), dtype=torch.bool),
            node_mask=torch.empty((0, 0), dtype=torch.bool),
            smiles=[],
            sample_id=[],
            canonical_smiles=[],
            feature_schema_version=None,
            is_empty=True,
        )

    def __call__(self, data_list):
        data_list = [item for item in data_list if item is not None]
        if self.max_node_filter is not None:
            data_list = [item for item in data_list if item.x.size(0) <= self.max_node_filter]
        if not data_list:
            return self._empty_batch()

        required_fields = (
            "x",
            "in_degree",
            "out_degree",
            "spatial_pos",
            "attn_edge_type",
            "edge_input",
            "y",
        )
        for item in data_list:
            missing = [field for field in required_fields if not hasattr(item, field)]
            if missing:
                raise ValueError(f"Preprocessed graph is missing required fields: {missing}")
            node_count = item.x.size(0)
            if item.spatial_pos.shape != (node_count, node_count):
                raise ValueError("spatial_pos must have shape [num_nodes, num_nodes]")
            if item.attn_edge_type.shape[:2] != (node_count, node_count):
                raise ValueError("attn_edge_type must have shape [num_nodes, num_nodes, num_bond_fields]")
            if item.edge_input.shape[:2] != (node_count, node_count):
                raise ValueError("edge_input must have shape [num_nodes, num_nodes, path, num_bond_fields]")

        batch_size = len(data_list)
        max_nodes = max(item.x.size(0) for item in data_list)
        max_path_distance = max(item.edge_input.size(2) for item in data_list)
        atom_feature_dim = data_list[0].x.size(-1)
        bond_feature_dim = data_list[0].attn_edge_type.size(-1)

        x = torch.stack([self._pad_node_features(item.x.long(), max_nodes) for item in data_list])
        in_degree = torch.stack([self._pad_degree(item.in_degree.long(), max_nodes) for item in data_list])
        out_degree = torch.stack([self._pad_degree(item.out_degree.long(), max_nodes) for item in data_list])
        spatial_pos = torch.stack([self._pad_spatial_pos(item.spatial_pos.long(), max_nodes) for item in data_list])
        attn_edge_type = torch.stack(
            [self._pad_attn_edge_type(item.attn_edge_type.long(), max_nodes) for item in data_list]
        )
        edge_input = torch.stack(
            [self._pad_edge_input(item.edge_input.long(), max_nodes, max_path_distance) for item in data_list]
        )
        y = torch.cat([item.y.reshape(-1, 1).float() for item in data_list], dim=0)

        padding_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool)
        attn_bias = torch.zeros((batch_size, max_nodes + 1, max_nodes + 1), dtype=torch.float)
        for batch_index, item in enumerate(data_list):
            node_count = item.x.size(0)
            padding_mask[batch_index, :node_count] = False
            long_distance = item.spatial_pos >= self.spatial_pos_max_clip
            attn_bias[batch_index, 1 : node_count + 1, 1 : node_count + 1][long_distance] = float("-inf")
            if node_count < max_nodes:
                # Valid queries cannot attend to padded keys. Padded query rows
                # are masked after each transformer block.
                attn_bias[batch_index, :, node_count + 1 :] = float("-inf")
                attn_bias[batch_index, node_count + 1 :, :] = 0.0

        return Data(
            x=x,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos,
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            edge_input=edge_input,
            padding_mask=padding_mask,
            node_mask=~padding_mask,
            y=y,
            smiles=[getattr(item, "smiles", "") for item in data_list],
            sample_id=[str(getattr(item, "sample_id", "")) for item in data_list],
            canonical_smiles=[getattr(item, "canonical_smiles", "") for item in data_list],
            feature_schema_version=getattr(data_list[0], "feature_schema_version", None),
            is_empty=False,
        )


class DataloaderWrapper:
    def __init__(
        self,
        task_list,
        preprocessed_data_base_dir,
        batch_size,
        splitting,
        valid_size,
        test_size,
        num_workers=0,
        collate_fn_for_loader=None,
        calibration_size=0.1,
        split_seed=42,
        manifest_path=None,
    ):
        if splitting not in {"random", "scaffold"}:
            raise ValueError(f"Unsupported splitting type: {splitting}")
        self.task_list = list(task_list)
        self.batch_size = int(batch_size)
        self.preprocessed_data_base_dir = Path(preprocessed_data_base_dir)
        self.splitting = splitting
        self.valid_size = float(valid_size)
        self.calibration_size = float(calibration_size)
        self.test_size = float(test_size)
        if self.valid_size < 0 or self.calibration_size < 0 or self.test_size < 0:
            raise ValueError("split sizes must be non-negative")
        if self.valid_size + self.calibration_size + self.test_size >= 1.0:
            raise ValueError("validation + calibration + test ratios must be less than 1")
        self.num_workers = int(num_workers)
        self.collate_fn_for_loader = collate_fn_for_loader
        self.split_seed = int(split_seed)
        self.manifest_path = Path(manifest_path) if manifest_path else self.preprocessed_data_base_dir / "split_manifest.json"

    def _load_split_lookup(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Global split manifest not found: {self.manifest_path}. "
                "Run preprocess_data.py before creating DataLoaders."
            )
        manifest = load_manifest(self.manifest_path)
        if manifest.get("splitting") != self.splitting:
            raise ValueError(
                f"Manifest splitting={manifest.get('splitting')!r} does not match requested {self.splitting!r}"
            )
        expected_ratios = {
            "train": 1.0 - self.valid_size - self.calibration_size - self.test_size,
            "validation": self.valid_size,
            "calibration": self.calibration_size,
            "test": self.test_size,
        }
        actual_ratios = manifest.get("ratios", {})
        for name, expected in expected_ratios.items():
            if not np.isclose(float(actual_ratios.get(name, float("nan"))), expected):
                raise ValueError(
                    f"Manifest ratio for {name!r} does not match the requested split configuration: "
                    f"manifest={actual_ratios.get(name)!r}, requested={expected!r}"
                )
        if int(manifest.get("seed")) != self.split_seed:
            raise ValueError(
                f"Manifest seed={manifest.get('seed')!r} does not match requested split_seed={self.split_seed}"
            )
        return {record["sample_id"]: record["split"] for record in manifest["records"]}

    @staticmethod
    def _loader(dataset, indices, split_name, batch_size, num_workers, collate_fn):
        subset = Subset(dataset, list(indices))
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=split_name == "train" and len(indices) > 0,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def _build_task_loaders(self, dataset, split_lookup):
        split_indices = {name: [] for name in ("train", "validation", "calibration", "test")}
        for index in range(len(dataset)):
            sample_id = dataset.get_sample_id(index)
            if sample_id not in split_lookup:
                raise KeyError(f"sample_id {sample_id!r} is missing from {self.manifest_path}")
            split_indices[split_lookup[sample_id]].append(index)
        return {
            "train": self._loader(
                dataset, split_indices["train"], "train", self.batch_size, self.num_workers, self.collate_fn_for_loader
            ),
            "val": self._loader(
                dataset,
                split_indices["validation"],
                "validation",
                self.batch_size,
                self.num_workers,
                self.collate_fn_for_loader,
            ),
            "calibration": self._loader(
                dataset,
                split_indices["calibration"],
                "calibration",
                self.batch_size,
                self.num_workers,
                self.collate_fn_for_loader,
            ),
            "test": self._loader(
                dataset, split_indices["test"], "test", self.batch_size, self.num_workers, self.collate_fn_for_loader
            ),
        }

    def get_data_loaders(self):
        split_lookup = self._load_split_lookup()
        all_task_loaders = {}
        for task_name in self.task_list:
            task_dir = self.preprocessed_data_base_dir / task_name
            dataset = PreprocessedDatasetWrapper(task_dir)
            all_task_loaders[task_name] = self._build_task_loaders(dataset, split_lookup)
            counts = {name: len(loader.dataset) for name, loader in all_task_loaders[task_name].items()}
            print(f"{task_name} split counts: {counts}")
        return all_task_loaders

    def get_train_validation_data_loaders(self, dataset, task_name):
        del task_name
        split_lookup = self._load_split_lookup()
        loaders = self._build_task_loaders(dataset, split_lookup)
        return loaders["train"], loaders["val"], loaders["calibration"], loaders["test"]
