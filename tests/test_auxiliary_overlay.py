import numpy as np
import torch
from types import SimpleNamespace

from auxiliary_labels import ShuffledEndpointOverlay
from architecture.response_guided_router import RGCERTaskConditioner
from trainer import Trainer
from test_datastore_v2_contract import _write_raw
from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset, build_datastore_v2


def test_shuffled_overlay_is_split_local_and_auxiliary_only(tmp_path):
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
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(root)
    overlay = ShuffledEndpointOverlay(store, "task_a", seed=7)
    assert overlay.spec.include_in_macro is False
    assert overlay.spec.fit_conformal is False
    assert overlay.spec.auxiliary_only is True

    dataset = ToxAcuteTaskDataset(
        store,
        overlay.task_name,
        split="train",
        label_provider=overlay,
    )
    assert len(dataset) > 0
    labels = [dataset[index].y.item() for index in range(len(dataset))]
    source_labels = [store.get_label(int(index), "task_a") for index in dataset.indices]
    assert all(np.isfinite(labels))
    assert sorted(labels) == sorted(source_labels)
    store.close()


def test_auxiliary_metadata_override_keeps_factorized_prompt_usable():
    fake = "shuffled_human_oral_TDLo"
    conditioner = RGCERTaskConditioner(
        ["human_oral_TDLo", fake],
        hidden_dim=8,
        use_factorized_prompt=True,
        router_dim=8,
        router_top_k=1,
        dropout=0.0,
        metadata_overrides={
            fake: {
                "task_name": fake,
                "organism": "human",
                "route": "oral",
                "measurement": "TDLo",
                "population": "general",
            }
        },
    )
    representation, diagnostics = conditioner(
        torch.randn(2, 8),
        fake,
        torch.randn(2, 2, 1),
        return_aux=True,
    )
    assert representation.shape == (2, 8)
    assert diagnostics.target_task == fake


def test_auxiliary_endpoint_is_excluded_from_interval_report():
    trainer = Trainer.__new__(Trainer)
    trainer.task_name = ["formal", "fake"]
    trainer.task_dict = {
        "formal": {"metrics": ["RMSE"], "fit_conformal": True},
        "fake": {"metrics": ["RMSE"], "fit_conformal": False},
    }
    trainer.args = SimpleNamespace(conformal_alpha=0.1)
    trainer._prediction_mode = lambda: "quantile"
    trainer._is_regression = lambda task: True
    trainer._collect_predictions = lambda *args, **kwargs: (
        {},
        {
            "formal": {
                "lower": [torch.tensor([[0.0]])],
                "upper": [torch.tensor([[1.0]])],
                "target": [torch.tensor([[0.5]])],
            },
            "fake": {
                "lower": [torch.tensor([[0.0]])],
                "upper": [torch.tensor([[100.0]])],
                "target": [torch.tensor([[50.0]])],
            },
        },
        {},
    )
    trainer._score_buffers = lambda buffers: {"tasks": {}, "score": 0.0}
    trainer._print_metrics = lambda *args, **kwargs: None

    result = trainer._evaluate({}, mode="validation")

    assert result["interval"]["MeanWidth"] == 1.0
