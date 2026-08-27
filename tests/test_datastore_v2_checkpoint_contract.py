from types import SimpleNamespace

import torch
from torch import nn

from loss import MSELoss
from tests.test_checkpoint_ablation_config import _DummyArchitecture, _args
from tests.test_datastore_v2_contract import _write_raw
from toxacute_datastore import ToxAcuteDataStore, build_datastore_v2
from trainer import Trainer
from weighting.EW import EW


def _formal_args(tmp_path, metadata, root, load_path=None):
    args = _args(tmp_path)
    args.data_store_dir = str(root)
    args.preprocessed_data_dir = None
    args.datastore_metadata = metadata
    args.split_seed = 42
    args.max_nodes_filter = 20
    args.spatial_pos_clip = 13
    args.max_path_distance = int(metadata["max_path_distance"])
    args.load_path = load_path
    return args


def _trainer(args):
    return Trainer(
        task_dict={"task_a": {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]}},
        weighting=EW,
        architecture=_DummyArchitecture,
        encoder_class=nn.Identity,
        decoders=nn.ModuleDict(),
        optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        args=args,
        save_path=args.save_path,
        load_path=args.load_path,
    )


def test_formal_checkpoint_records_and_validates_datastore_identity(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        split_seed=42,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(root)
    source_args = _formal_args(tmp_path / "source", store.metadata, root)
    source = _trainer(source_args)
    checkpoint_path = source._save_checkpoint(6, "model_best.pt")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == 5
    assert payload["data_config"]["datastore_fingerprint"] == store.fingerprint
    assert payload["data_config"]["max_path_distance"] == 6

    loaded_args = _formal_args(tmp_path / "loaded", store.metadata, root, checkpoint_path)
    loaded = _trainer(loaded_args)
    assert loaded.loaded_epoch == 6
    store.close()
