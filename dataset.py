"""Dataset, global split, and CPU-only Graphormer collation utilities."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Data

from split_manifest import load_manifest
from molecular_features import BOND_FEATURE_NAMES
from reproducibility import loader_generator
from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset


class PreprocessedDatasetWrapper(Dataset):
    def __init__(self, task_data_dir):
        warnings.warn(
            "PreprocessedDatasetWrapper is legacy V1 storage; use ToxAcuteDataStore for formal runs.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            # ``spatial_pos_max_clip`` is the largest *visible* hop: distances
            # at the clip (and its clamped representation in the embedding
            # index) still attend. Only strictly farther hops — including the
            # 510 marker for disconnected pairs — are masked out.
            long_distance = item.spatial_pos > self.spatial_pos_max_clip
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
        preprocessed_data_base_dir=None,
        batch_size=64,
        splitting="scaffold",
        valid_size=0.1,
        test_size=0.1,
        num_workers=0,
        collate_fn_for_loader=None,
        calibration_size=0.1,
        split_seed=42,
        manifest_path=None,
        *,
        data_store=None,
        max_nodes_filter=None,
        label_providers=None,
        loader_seed=None,
    ):
        if data_store is None and isinstance(preprocessed_data_base_dir, ToxAcuteDataStore):
            data_store = preprocessed_data_base_dir
            preprocessed_data_base_dir = None
        if data_store is not None and not isinstance(data_store, ToxAcuteDataStore):
            data_store = ToxAcuteDataStore.resolve(data_store)
        self.data_store = data_store
        if splitting not in {"random", "scaffold"}:
            raise ValueError(f"Unsupported splitting type: {splitting}")
        self.task_list = list(task_list)
        self.batch_size = int(batch_size)
        self.preprocessed_data_base_dir = Path(preprocessed_data_base_dir) if preprocessed_data_base_dir else None
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
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else (self.preprocessed_data_base_dir / "split_manifest.json" if self.preprocessed_data_base_dir else None)
        )
        self.max_nodes_filter = max_nodes_filter
        self.label_providers = dict(label_providers or {})
        # Review §8-9: train shuffling must draw from a per-task generator
        # derived from the base seed (never the global RNG), so paired-seed
        # runs see identical batch orders regardless of model-init RNG use.
        # ``loader_seed`` defaults to split_seed so legacy callers stay
        # reproducible too.
        self.loader_seed = int(loader_seed) if loader_seed is not None else self.split_seed

        if self.data_store is None and self.preprocessed_data_base_dir is None:
            raise ValueError("Either data_store or preprocessed_data_base_dir is required")

    @property
    def is_v2(self):
        return self.data_store is not None

    def _load_split_lookup(self):
        if self.manifest_path is None:
            raise ValueError("A legacy split manifest path is required for V1 data")
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
    def _loader(dataset, indices, split_name, batch_size, num_workers, collate_fn, task_name=None, loader_seed=None):
        subset = Subset(dataset, list(indices))
        generator = None
        if split_name == "train" and len(indices) > 0:
            generator = loader_generator(int(loader_seed or 42), task_name or "", 0)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=split_name == "train" and len(indices) > 0,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=False,
            generator=generator,
        )

    def _build_task_loaders(self, dataset, split_lookup, task_name=None):
        split_indices = {name: [] for name in ("train", "validation", "calibration", "test")}
        for index in range(len(dataset)):
            sample_id = dataset.get_sample_id(index)
            if sample_id not in split_lookup:
                raise KeyError(f"sample_id {sample_id!r} is missing from {self.manifest_path}")
            split_indices[split_lookup[sample_id]].append(index)
        return {
            "train": self._loader(
                dataset, split_indices["train"], "train", self.batch_size, self.num_workers, self.collate_fn_for_loader, task_name=task_name, loader_seed=self.loader_seed
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

    def _v2_loader(self, dataset, split_name, task_name=None):
        generator = None
        if split_name == "train" and len(dataset) > 0:
            generator = loader_generator(self.loader_seed, task_name or "", 0)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=split_name == "train" and len(dataset) > 0,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collate_fn_for_loader,
            drop_last=False,
            # Review-2 §27 (plan A): persistent workers make iteration order
            # depend on worker lifecycle, breaking epoch-boundary resume
            # determinism when num_workers > 0.  Formal runs also pin
            # --num_loader_workers 0.
            persistent_workers=False,
            generator=generator,
        )

    def _get_v2_data_loaders(self):
        real_tasks = [task for task in self.task_list if task in self.data_store.task_names]
        self.data_store.validate(strict=False, expected_task_names=real_tasks)
        metadata = self.data_store.metadata
        if metadata.get("splitting") != self.splitting:
            raise ValueError("DataStore splitting does not match requested splitting")
        if int(metadata.get("split_seed")) != self.split_seed:
            raise ValueError("DataStore split_seed does not match requested split_seed")
        expected_ratios = {
            "train": 1.0 - self.valid_size - self.calibration_size - self.test_size,
            "validation": self.valid_size,
            "calibration": self.calibration_size,
            "test": self.test_size,
        }
        actual_ratios = metadata.get("split_ratios", {})
        for name, expected in expected_ratios.items():
            if not np.isclose(float(actual_ratios.get(name, np.nan)), expected):
                raise ValueError(f"DataStore split ratio for {name!r} does not match requested configuration")
        all_task_loaders = {}
        for task_name in self.task_list:
            datasets = {
                split: ToxAcuteTaskDataset(
                    self.data_store,
                    task_name,
                    split=None if split == "all" else split,
                    max_nodes=self.max_nodes_filter,
                    label_provider=self.label_providers.get(task_name),
                )
                for split in ("train", "validation", "calibration", "test")
            }
            all_task_loaders[task_name] = {
                "train": self._v2_loader(datasets["train"], "train", task_name=task_name),
                "val": self._v2_loader(datasets["validation"], "validation"),
                "calibration": self._v2_loader(datasets["calibration"], "calibration"),
                "test": self._v2_loader(datasets["test"], "test"),
            }
            counts = {name: len(loader.dataset) for name, loader in all_task_loaders[task_name].items()}
            print(f"{task_name} split counts: {counts}")
        return all_task_loaders

    def get_data_loaders(self):
        if self.is_v2:
            return self._get_v2_data_loaders()
        split_lookup = self._load_split_lookup()
        all_task_loaders = {}
        for task_name in self.task_list:
            task_dir = self.preprocessed_data_base_dir / task_name
            dataset = PreprocessedDatasetWrapper(task_dir)
            all_task_loaders[task_name] = self._build_task_loaders(dataset, split_lookup, task_name=task_name)
            counts = {name: len(loader.dataset) for name, loader in all_task_loaders[task_name].items()}
            print(f"{task_name} split counts: {counts}")
        return all_task_loaders

    def get_train_validation_data_loaders(self, dataset, task_name):
        if self.is_v2:
            del dataset
            datasets = {
                split: ToxAcuteTaskDataset(
                    self.data_store,
                    task_name,
                    split=split,
                    max_nodes=self.max_nodes_filter,
                    label_provider=self.label_providers.get(task_name),
                )
                for split in ("train", "validation", "calibration", "test")
            }
            return (
                self._v2_loader(datasets["train"], "train", task_name=task_name),
                self._v2_loader(datasets["validation"], "validation"),
                self._v2_loader(datasets["calibration"], "calibration"),
                self._v2_loader(datasets["test"], "test"),
            )
        split_lookup = self._load_split_lookup()
        loaders = self._build_task_loaders(dataset, split_lookup, task_name=task_name)
        return loaders["train"], loaders["val"], loaders["calibration"], loaders["test"]
