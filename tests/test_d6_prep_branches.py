"""D6 prep branch-level smokes (review P0-1/§55-§56).

These tests drive ``d6_prep_inits.main()`` through monkeypatched teachers,
model construction and loaders, so branch wiring errors (undefined collator,
wrong task list, missing provenance) can no longer escape helper-level tests.
No LMDB / datastore access happens.
"""

import json
import sys

import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace

import scripts.d6_prep_inits as prep
from architecture.toxacute_tasks import HUMAN_TARGET_TASKS


class _StubLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, *args, **kwargs):
        return x * self.alpha


class _StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_StubLayer() for _ in range(2)])

    def forward(self, batch):
        return torch.ones(batch.batch_size, 3, 4)


class _StubEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _StubBackbone()

    def forward(self, batch):
        return self.backbone(batch)[:, 0, :]


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _StubEncoder()
        self.decoders = nn.ModuleDict(
            {task: nn.Linear(4, 3) for task in HUMAN_TARGET_TASKS}
        )


def _stub_params():
    return SimpleNamespace(
        spatial_pos_clip=20,
        seed=42,
        hidden_dim=4,
        a_heads=1,
    )


def _install_stubs(monkeypatch, tmp_path, clst_rows=None, card_rows=(["s1", "s2"], np.ones((2, 4)))):
    """Monkeypatch the prep module so main() runs without LMDB/datastore."""

    real_dir = tmp_path / "teacher_real"
    shuffle_dir = tmp_path / "teacher_shuffle"
    for run_dir in (real_dir, shuffle_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"initial_model_sha256": "teacher-init-hash"}), encoding="utf-8"
        )
        # _sha256_file hashes the checkpoint file itself even though loading
        # is stubbed.
        torch.save({"stub": True}, run_dir / "teacher_last.pt")

    stub_payload = {
        "checkpoint_version": 6,
        "epoch": 29,
        "model_state": _StubModel().state_dict(),
        "task_names": list(HUMAN_TARGET_TASKS),
        "configuration": {"arch": "Graphormer", "seed": 42},
        "architecture_config": {},
        "split_manifest_hash": "m",
        "feature_schema_version": "s",
        "data_config": {"datastore_fingerprint": "f"},
        "reproducibility": {"base_seed": 42},
    }

    recorded = {"loaders_args": [], "clst_args": [], "card_args": [], "built": []}

    def _stub_build_model(seed, task_scope, device, template_config=None):
        recorded["built"].append((seed, task_scope))
        return _StubModel(), _stub_params(), list(HUMAN_TARGET_TASKS)

    def _stub_load_checkpoint(run_dir, expected_epoch):
        assert int(expected_epoch) == 29
        return str(run_dir / "teacher_last.pt"), dict(stub_payload)

    def _stub_verify(real_dir, shuffle_dir, expected_epoch, expected_model_seed=None):
        return {
            "teacher_configuration": {"hidden_dim": 4},
            "teacher_initial_model_sha256": "teacher-init-hash",
            "teacher_real_checkpoint_sha256": "real-sha",
            "teacher_shuffle_checkpoint_sha256": "shuffle-sha",
            "animal_shuffle_seed": 20260831,
            "animal_shuffle_mapping_sha256": "map-hash",
        }

    fake_train_loaders = {
        task: [_FakeBatchLike([f"{task}_{index}"])] for index, task in enumerate(HUMAN_TARGET_TASKS)
    }

    def _stub_loaders(params, task_list, collator):
        recorded["loaders_args"].append(list(task_list))
        return _stub_human_loaders(params)

    def _stub_human_loaders(params):
        recorded["loaders_args"].append(list(HUMAN_TARGET_TASKS))
        return {"train": fake_train_loaders, "val": {}, "calibration": {}, "test": {}}

    def _stub_clst_scores(model_real, model_shuffle, train_loaders, device):
        recorded["clst_args"].append(train_loaders)
        rows = clst_rows if clst_rows is not None else [
            {"layer": index, "semantic_layer_score": float(index), "rank": index}
            for index in range(2)
        ]
        return rows, 3

    def _stub_card_table(model_real, model_shuffle, train_loaders, task_names, device):
        recorded["card_args"].append(list(task_names))
        ids, deltas = card_rows
        return np.asarray(ids), np.asarray(deltas)

    monkeypatch.setattr(prep, "_load_teacher_checkpoint", _stub_load_checkpoint)
    monkeypatch.setattr(prep, "_verify_teacher_pair", _stub_verify)
    monkeypatch.setattr(prep, "_b1_teacher_contract", lambda *a, **k: None)
    monkeypatch.setattr(prep, "_build_model", _stub_build_model)
    monkeypatch.setattr(prep, "_human_loaders", _stub_human_loaders)
    monkeypatch.setattr(prep, "_loaders", _stub_loaders)
    monkeypatch.setattr(prep, "clst_layer_scores", _stub_clst_scores)
    monkeypatch.setattr(prep, "card_delta_table", _stub_card_table)
    monkeypatch.setattr(prep, "validate_params", lambda params: None)
    monkeypatch.setattr(
        prep, "_regenerate_teacher_anchor", lambda seed, template, expected: _StubModel()
    )
    return recorded, real_dir, shuffle_dir


class _FakeBatchLike(dict):
    """Dict batch with the attribute surface the stubs expect."""

    sample_id = []

    def __init__(self, ids):
        super().__init__()
        self.sample_id = list(ids)

    def to(self, device):
        return self

    def get(self, key, default=None):
        return default


def _run_prep_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["d6_prep_inits.py"] + argv)
    prep.main()


def test_clst_main_branch_smoke(monkeypatch, tmp_path):
    recorded, real_dir, shuffle_dir = _install_stubs(monkeypatch, tmp_path)
    output = tmp_path / "o2_init.pt"
    _run_prep_main(
        monkeypatch,
        [
            "clst",
            "--teacher_real_dir", str(real_dir),
            "--teacher_shuffle_dir", str(shuffle_dir),
            "--expected_teacher_epoch", "29",
            "--human_seed", "42",
            "--clst_top_k", "2",
            "--output", str(output),
        ],
    )
    # Review P0-1 regression: the branch must not hit an undefined collator,
    # must build the human3 loaders, and must score the train loaders only.
    assert recorded["loaders_args"] and recorded["loaders_args"][0] == list(HUMAN_TARGET_TASKS)
    assert recorded["clst_args"] and set(recorded["clst_args"][0]) == set(HUMAN_TARGET_TASKS)
    assert output.exists()
    assert output.with_name(output.name + ".provenance.json").exists()


def test_card_table_main_branch_uses_human3_train_tasks(monkeypatch, tmp_path):
    recorded, real_dir, shuffle_dir = _install_stubs(
        monkeypatch, tmp_path, card_rows=(["h1", "h2", "h3"], np.ones((3, 4)))
    )
    output = tmp_path / "card_table.npz"
    _run_prep_main(
        monkeypatch,
        [
            "card_table",
            "--teacher_real_dir", str(real_dir),
            "--teacher_shuffle_dir", str(shuffle_dir),
            "--expected_teacher_epoch", "29",
            "--human_seed", "42",
            "--output", str(output),
        ],
    )
    # Review P0-2 regression: the human3 train loader keys are the three human
    # endpoints — never the 56 animal task names.
    assert recorded["card_args"] and recorded["card_args"][0] == list(HUMAN_TARGET_TASKS)
    with np.load(output) as table:
        assert table["ids"].size == 3
        assert table["delta"].shape == (3, 4)
    assert output.with_name(output.name + ".provenance.json").exists()
    provenance = json.loads(output.with_name(output.name + ".provenance.json").read_text())
    assert provenance["teacher_real_checkpoint_sha256"] == "real-sha"
    assert provenance["animal_shuffle_mapping_sha256"] == "map-hash"


def test_csdt_main_branch_smoke(monkeypatch, tmp_path):
    recorded, real_dir, shuffle_dir = _install_stubs(monkeypatch, tmp_path)
    output = tmp_path / "o1_init.pt"
    _run_prep_main(
        monkeypatch,
        [
            "csdt",
            "--teacher_real_dir", str(real_dir),
            "--teacher_shuffle_dir", str(shuffle_dir),
            "--expected_teacher_epoch", "29",
            "--human_seed", "42",
            "--alpha", "1.0",
            "--output", str(output),
        ],
    )
    assert output.exists()
    assert output.with_name(output.name + ".provenance.json").exists()


def test_anchor_main_branch_smoke(monkeypatch, tmp_path):
    recorded, real_dir, shuffle_dir = _install_stubs(monkeypatch, tmp_path)
    output = tmp_path / "b0a_init.pt"
    _run_prep_main(
        monkeypatch,
        [
            "anchor",
            "--teacher_real_dir", str(real_dir),
            "--expected_teacher_epoch", "29",
            "--human_seed", "42",
            "--output", str(output),
        ],
    )
    assert output.exists()
    assert output.with_name(output.name + ".provenance.json").exists()


def test_b1_main_branch_smoke(monkeypatch, tmp_path):
    recorded, real_dir, shuffle_dir = _install_stubs(monkeypatch, tmp_path)
    output = tmp_path / "b1_init.pt"
    _run_prep_main(
        monkeypatch,
        [
            "b1",
            "--teacher_real_dir", str(real_dir),
            "--expected_teacher_epoch", "29",
            "--human_seed", "42",
            "--output", str(output),
        ],
    )
    assert output.exists()
    assert output.with_name(output.name + ".provenance.json").exists()
