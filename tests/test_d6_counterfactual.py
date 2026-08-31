"""D6 counterfactual-transfer contracts (plan §61-§67, §78).

Covers: matched teacher initialisation hash, shuffle integrity (label
multiset/sample count/real validation labels/cross-seed identical mapping),
the CSDT parameter-delta formula, CLST layer scoring on train molecules only,
and CARD (teacher target carries no gradient while the adapter does).
"""

import numpy as np
import pytest
import torch
from torch import nn

from reproducibility import state_dict_sha256
from scripts.d6_prep_inits import apply_b1, apply_csdt, _build_model
from shuffled_animal_labels import ShuffledAnimalTrainLabels


class _StubStore:
    def __init__(self, labels):
        self._labels = labels

    def get_task_indices(self, task, split=None):
        assert split == "train"
        return np.array(sorted(self._labels[task]), dtype=np.int64)

    def get_label(self, index, task):
        return self._labels[task][int(index)]


LABELS = {"mouse_oral_LD50": {0: 1.0, 2: 3.0, 5: 5.0}, "rat_oral_LD50": {1: 2.0, 3: 4.0}}


def test_shuffled_labels_preserve_multiset_and_are_deterministic():
    store = _StubStore(LABELS)
    provider_a = ShuffledAnimalTrainLabels(store, list(LABELS), shuffle_seed=20260831)
    provider_b = ShuffledAnimalTrainLabels(store, list(LABELS), shuffle_seed=20260831)

    for task, mapping in provider_a.mappings.items():
        original = sorted(LABELS[task].values())
        shuffled = sorted(mapping.values())
        assert shuffled == original  # multiset unchanged (§78 shuffle integrity)
        assert len(mapping) == len(LABELS[task])  # sample count unchanged
        assert mapping == provider_b.mappings[task]  # §62: mapping identical across instances


def test_shuffled_labels_leave_validation_real_and_human_absent():
    store = _StubStore(LABELS)
    provider = ShuffledAnimalTrainLabels(store, list(LABELS), shuffle_seed=20260831)
    for index, label in LABELS["mouse_oral_LD50"].items():
        assert provider.get_label(index, "mouse_oral_LD50", split="validation") == pytest.approx(label)
    assert provider.get_label(0, "man_oral_TDLo", split="train") is None  # human never touched


def test_main_writes_shuffle_sanity_csv(tmp_path):
    # Smoke-gate regression (2026-08-31): the shuffled teacher crashed with
    # NameError because main.py's D6_SHUFFLE_SANITY.csv write referenced a
    # _write_csv helper that did not exist in that module.
    import main as main_module

    target = tmp_path / "run" / "D6_SHUFFLE_SANITY.csv"
    rows = [
        {
            "task": "mouse_oral_LD50",
            "n": 3,
            "mean_before": 1.5,
            "mean_after": 1.5,
            "std_before": 0.5,
            "std_after": 0.5,
            "label_multiset_equal": True,
            "mapping_hash": "h",
        }
    ]
    main_module._write_csv(target, list(rows[0].keys()), rows)
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == (
        "task,n,mean_before,mean_after,std_before,std_after,label_multiset_equal,mapping_hash"
    )
    assert lines[1].startswith("mouse_oral_LD50,3,")
    assert lines[1].endswith(",True,h")


def test_csdt_formula_is_exact():
    theta0 = {"encoder.backbone.a": torch.tensor([1.0, 2.0]), "decoders.head": torch.tensor([0.5])}
    real = {"encoder.backbone.a": torch.tensor([2.0, 1.0]), "decoders.head": torch.tensor([9.0])}
    shuffle = {"encoder.backbone.a": torch.tensor([0.0, 0.0]), "decoders.head": torch.tensor([7.0])}
    merged = apply_csdt(theta0, real, shuffle, alpha=0.5)
    assert torch.allclose(merged["encoder.backbone.a"], torch.tensor([2.0, 2.5]))
    assert torch.equal(merged["decoders.head"], theta0["decoders.head"])  # heads stay fresh (§64)


def test_b1_copies_only_backbone():
    theta0 = {"encoder.backbone.a": torch.tensor([1.0]), "decoders.head": torch.tensor([0.5])}
    real = {"encoder.backbone.a": torch.tensor([4.0]), "decoders.head": torch.tensor([9.0])}
    merged = apply_b1(theta0, real)
    assert torch.equal(merged["encoder.backbone.a"], torch.tensor([4.0]))
    assert torch.equal(merged["decoders.head"], theta0["decoders.head"])


def test_matched_teachers_share_initialisation_hash():
    model_a, _, _ = _build_model(42, "animal56", torch.device("cpu"))
    model_b, _, _ = _build_model(42, "animal56", torch.device("cpu"))
    model_c, _, _ = _build_model(43, "animal56", torch.device("cpu"))
    assert state_dict_sha256(model_a) == state_dict_sha256(model_b)
    assert state_dict_sha256(model_a) != state_dict_sha256(model_c)


def test_card_zero_init_residual_and_gradient_flow(tmp_path):
    np.savez(
        tmp_path / "delta.npz",
        ids=np.array(["s1", "s2"]),
        delta=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    from card_adapter import CardAdapter

    card = CardAdapter(hidden_dim=2, bottleneck=3, lambda_delta=0.1, delta_table_path=str(tmp_path / "delta.npz"))
    representation = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    out = card(representation, ["s1", "s2"])
    # Zero-init output layer: the residual starts exactly at the HPS baseline.
    assert torch.allclose(out, representation)
    # Review P0-1 Test B: unit-direction MSE at zero init is order one.
    assert 0.5 <= float(card.last_loss) <= 1.5
    loss = card.last_loss * card.lambda_delta
    loss.backward()
    assert card.up.weight.grad is not None and card.up.weight.grad.abs().sum() > 0
    assert card.delta_table.grad is None  # teacher delta is a frozen target (§78)


def test_card_zero_init_gradient_is_finite_and_bounded(tmp_path):
    # Review P0-1 Test A / P1-6 (§41-§42): the regression must bound the
    # CARD DISTILLATION gradient itself (backward on last_loss), not the
    # prediction-residual path — the old cosine-on-zero-vector loss produced
    # a ~1e10 gradient here; the unit-direction MSE must stay ~O(1).
    from card_adapter import CardAdapter

    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["s1", "s2"]), delta=np.array([[1.0, 0.0], [0.0, 1.0]]))
    card = CardAdapter(hidden_dim=2, bottleneck=3, lambda_delta=0.1, delta_table_path=str(table_path))
    representation = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    card(representation, ["s1", "s2"])
    loss = card.last_loss * card.lambda_delta
    loss.backward()
    for parameter in card.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    up_grad_norm = float(card.up.weight.grad.norm())
    assert up_grad_norm < 100, f"card up grad norm exploded: {up_grad_norm}"


# ----------------------------------------------------------------------
# Review §53: teacher pair contract, CLST train-only/dedup, CARD alignment.
# ----------------------------------------------------------------------
import json

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS
from scripts.d6_prep_inits import (
    _load_teacher_checkpoint,
    _verify_single_real_teacher,
    _verify_teacher_pair,
    clst_layer_scores,
)


def _teacher_payload(shuffle, epoch=29, seed=42, manifest="m", fingerprint="f"):
    return {
        "checkpoint_version": 6,
        "epoch": epoch,
        "model_state": {"w": torch.zeros(1)},
        "task_names": sorted(ANIMAL_SOURCE_TASKS),
        "configuration": {
            "arch": "Graphormer",
            "shuffle_animal_train_labels": shuffle,
            "animal_shuffle_seed": 20260831 if shuffle else None,
            "train_eval_scope": "validation_only",
            "fit_conformal": False,
        },
        "architecture_config": {"hidden_dim": 96},
        "split_manifest_hash": manifest,
        "feature_schema_version": "atom_v2_bond_v1_pathavg_v1",
        "data_config": {"datastore_fingerprint": fingerprint},
        "reproducibility": {"base_seed": seed},
    }


def _write_shuffle_audit(run_dir, mapping_hash="map-hash", label_multiset_equal="True"):
    """Review P1-2: every formal shuffle teacher carries its manifest +
    sanity table; prep must refuse the pair without them."""
    manifest = {
        "animal_shuffle_seed": 20260831,
        "animal_shuffle_mapping_sha256": mapping_hash,
        "tasks": sorted(ANIMAL_SOURCE_TASKS),
    }
    (run_dir / "D6_ANIMAL_SHUFFLE_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with (run_dir / "D6_SHUFFLE_SANITY.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "task,n,mean_before,mean_after,std_before,std_after,label_multiset_equal,mapping_hash\n"
        )
        for task in sorted(ANIMAL_SOURCE_TASKS):
            handle.write(
                f"{task},4,1.5,1.5,0.5,0.5,{label_multiset_equal},{mapping_hash}\n"
            )


def _write_teacher(root, name, payload, init_hash="init-hash"):
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, run_dir / "teacher_last.pt")
    metadata = {"initial_model_sha256": init_hash}
    if payload.get("configuration", {}).get("shuffle_animal_train_labels") is True:
        metadata["animal_shuffle_seed"] = 20260831
        metadata["animal_shuffle_mapping_sha256"] = "map-hash"
        _write_shuffle_audit(run_dir)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return run_dir


def test_teacher_pair_requires_same_initial_hash(tmp_path):
    real = _write_teacher(tmp_path, "real", _teacher_payload(False), init_hash="hash-a")
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True), init_hash="hash-b")
    with pytest.raises(SystemExit):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_pair_requires_same_epoch(tmp_path):
    real = _write_teacher(tmp_path, "real", _teacher_payload(False, epoch=29))
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True, epoch=39))
    with pytest.raises(SystemExit):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_pair_requires_same_manifest(tmp_path):
    real = _write_teacher(tmp_path, "real", _teacher_payload(False, manifest="m1"))
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True, manifest="m2"))
    with pytest.raises(SystemExit):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_pair_requires_real_labels_for_real_teacher(tmp_path):
    real = _write_teacher(tmp_path, "real", _teacher_payload(shuffle=True))
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(shuffle=True))
    with pytest.raises(SystemExit):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_load_rejects_wrong_fixed_epoch(tmp_path):
    run_dir = _write_teacher(tmp_path, "real", _teacher_payload(False, epoch=39))
    with pytest.raises(SystemExit):
        _load_teacher_checkpoint(run_dir, expected_epoch=29)


def test_b1_requires_matching_model_seed(tmp_path):
    run_dir = _write_teacher(tmp_path, "real", _teacher_payload(False, seed=43))
    payload = torch.load(run_dir / "teacher_last.pt", weights_only=False)
    with pytest.raises(SystemExit, match="base seed"):
        _verify_single_real_teacher(
            payload, run_dir, expected_model_seed=42, expected_epoch=29
        )


def test_teacher_pair_rejects_missing_base_seed(tmp_path):
    # Review P1-3 (§34-§35): a v6 teacher without reproducibility.base_seed
    # must FAIL the pair contract, never pass silently.
    real_payload = _teacher_payload(False)
    real_payload["reproducibility"] = {}
    real = _write_teacher(tmp_path, "real", real_payload)
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True))
    with pytest.raises(SystemExit, match="base_seed"):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_pair_requires_shuffle_manifest_file(tmp_path):
    # Review P1-2 (§30-§31): the shuffle manifest is mandatory for a formal
    # counterfactual pair.
    real = _write_teacher(tmp_path, "real", _teacher_payload(False))
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True))
    (shuffle / "D6_ANIMAL_SHUFFLE_MANIFEST.json").unlink()
    with pytest.raises(SystemExit, match="missing D6_ANIMAL_SHUFFLE_MANIFEST"):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_pair_requires_shuffle_sanity_csv(tmp_path):
    # Review P1-2 (§32): the shuffle sanity table is mandatory too.
    real = _write_teacher(tmp_path, "real", _teacher_payload(False))
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True))
    (shuffle / "D6_SHUFFLE_SANITY.csv").unlink()
    with pytest.raises(SystemExit, match="missing D6_SHUFFLE_SANITY"):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_pair_rejects_label_multiset_mismatch(tmp_path):
    # Review P1-2 (§32): every task must prove its shuffled label multiset
    # equals the original one.
    real = _write_teacher(tmp_path, "real", _teacher_payload(False))
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True))
    _write_shuffle_audit(shuffle, label_multiset_equal="False")
    with pytest.raises(SystemExit, match="label_multiset_equal"):
        _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)


def test_teacher_pair_passes_when_matched(tmp_path):
    real = _write_teacher(tmp_path, "real", _teacher_payload(False))
    shuffle = _write_teacher(tmp_path, "shuffle", _teacher_payload(True))
    matched = _verify_teacher_pair(real, shuffle, expected_epoch=29, expected_model_seed=42)
    assert matched["teacher_real_epoch"] == 29
    assert matched["teacher_initial_model_sha256"] == "init-hash"
    # Review P1-1 (§52): the pair sidecar must carry the full data identity.
    for key in (
        "split_manifest_hash", "feature_schema_version", "datastore_fingerprint",
        "teacher_shuffle_checkpoint_sha256", "animal_shuffle_seed",
        "animal_shuffle_mapping_sha256",
    ):
        assert matched.get(key) not in (None, ""), f"pair provenance lacks {key}"


class _StubLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, *args, **kwargs):
        return x * self.scale


class _StubBackbone(nn.Module):
    def __init__(self, scales, offset):
        super().__init__()
        self.layers = nn.ModuleList([_StubLayer(scale) for scale in scales])
        self.offset = offset

    def forward(self, batch):
        x = torch.ones(batch.batch_size, 3, 4) + self.offset
        for layer in self.layers:
            x = layer(x)
        return x


class _StubModel(nn.Module):
    def __init__(self, scales, offset):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = _StubBackbone(scales, offset)


class _FakeBatch:
    batch_size = 2
    sample_id = []

    def to(self, device):
        return self

    def get(self, key, default=None):
        return default


class _CountingLoader:
    def __init__(self, sample_id):
        self.sample_id = sample_id
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        batch = _FakeBatch()
        batch.sample_id = list(self.sample_id)
        # Review P1-4 guard: batch_size must equal the sample-id count.
        batch.batch_size = len(batch.sample_id)
        return iter([batch])

    def __len__(self):
        return 1


def test_clst_deduplicates_and_scores_train_molecules_only():
    from types import SimpleNamespace

    model_real = _StubModel(scales=[1.0, 3.0], offset=0.0)
    model_shuffle = _StubModel(scales=[1.0, 3.0], offset=0.5)
    loader_a = _CountingLoader(["s1", "s2"])  # HUMAN_TASKS[0]
    loader_b = _CountingLoader(["s2", "s3"])  # HUMAN_TASKS[1] shares s2
    empty = _CountingLoader([])
    train_loaders = {
        HUMAN_TARGET_TASKS[0]: loader_a,
        HUMAN_TARGET_TASKS[1]: loader_b,
        HUMAN_TARGET_TASKS[2]: empty,
    }
    device = torch.device("cpu")
    rows, unique = clst_layer_scores(model_real, model_shuffle, train_loaders, device)

    assert unique == 3  # s1, s2, s3 deduplicated across endpoints
    assert len(rows) == 2
    assert all(row["rank"] in (0, 1) for row in rows)
    assert all(row["semantic_layer_score"] == row["semantic_layer_score"] for row in rows)  # finite
    assert loader_a.iterations == 1 and loader_b.iterations == 1 and empty.iterations == 1


def test_clst_uses_only_the_loaders_it_is_given():
    # Structural contract (review §78): the scorer receives exactly one dict
    # and only touches its HUMAN_TARGET_TASKS entries; val/calibration/test
    # loaders can never be iterated because they are not passed in.
    from types import SimpleNamespace

    model_real = _StubModel(scales=[1.0], offset=0.0)
    model_shuffle = _StubModel(scales=[1.0], offset=0.5)
    poisoned = _CountingLoader(["x"])

    class _PoisonedDict(dict):
        def get(self, key, default=None):
            if key not in HUMAN_TARGET_TASKS:
                return poisoned
            return super().get(key, default)

    train_loaders = _PoisonedDict({HUMAN_TARGET_TASKS[0]: _CountingLoader(["s1"])})
    rows, unique = clst_layer_scores(model_real, model_shuffle, train_loaders, torch.device("cpu"))
    assert unique == 1
    assert poisoned.iterations == 0


def test_card_alignment_uses_exact_batch_positions(tmp_path):
    from torch.nn import functional as F

    from card_adapter import CardAdapter

    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["a", "c"]), delta=np.array([[1.0, 0.0], [0.0, 1.0]]))
    card = CardAdapter(hidden_dim=2, bottleneck=2, lambda_delta=0.1, delta_table_path=str(table_path))
    card.eval()

    representation = torch.tensor([[1.0, 2.0], [5.0, 5.0], [3.0, 3.0]])
    out = card(representation, ["a", "missing", "c"])  # middle id missing

    # Expected loss pairs positions [0, 2] with table rows [a, c] — the old
    # slice-based implementation would pair position 1 with c's delta.
    with torch.no_grad():
        aligned = card.up(F.gelu(card.down(representation)))[[0, 2]]
        deltas = card.delta_table[[0, 1]]
        expected = (aligned - deltas).pow(2).sum(dim=-1).mean()
    assert card.last_loss.detach() == pytest.approx(expected.detach(), abs=1e-6)
    assert torch.allclose(out, representation)  # zero-init residual invariant


def test_card_missing_training_id_fails_fast(tmp_path):
    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["a", "c"]), delta=np.array([[1.0, 0.0], [0.0, 1.0]]))
    from card_adapter import CardAdapter

    card = CardAdapter(hidden_dim=2, bottleneck=2, lambda_delta=0.1, delta_table_path=str(table_path))
    card.train()
    representation = torch.zeros(3, 2)
    with pytest.raises(KeyError, match="missing"):
        card(representation, ["a", "missing", "c"])


def test_card_eval_allows_missing_ids(tmp_path):
    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["a", "c"]), delta=np.array([[1.0, 0.0], [0.0, 1.0]]))
    from card_adapter import CardAdapter

    card = CardAdapter(hidden_dim=2, bottleneck=2, lambda_delta=0.1, delta_table_path=str(table_path))
    card.eval()
    out = card(torch.ones(2, 2), ["missing", "also_missing"])
    assert torch.isfinite(card.last_loss)
    assert torch.allclose(out, torch.ones(2, 2))  # zero-init residual unchanged


def test_card_table_validation_rejects_bad_tables(tmp_path):
    from card_adapter import CardAdapter

    bad_ids = tmp_path / "dup.npz"
    np.savez(bad_ids, ids=np.array(["a", "a"]), delta=np.ones((2, 2)))
    with pytest.raises(ValueError, match="duplicate"):
        CardAdapter(2, 2, 0.1, str(bad_ids))

    wrong_width = tmp_path / "width.npz"
    np.savez(wrong_width, ids=np.array(["a"]), delta=np.ones((1, 5)))
    with pytest.raises(ValueError, match="hidden_dim"):
        CardAdapter(2, 2, 0.1, str(wrong_width))

    non_finite = tmp_path / "nan.npz"
    np.savez(non_finite, ids=np.array(["a"]), delta=np.array([[float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        CardAdapter(2, 2, 0.1, str(non_finite))


def test_card_records_raw_teacher_delta_norm(tmp_path):
    from card_adapter import CardAdapter

    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["a"]), delta=np.array([[3.0, 4.0]]))  # norm 5
    card = CardAdapter(hidden_dim=2, bottleneck=2, lambda_delta=0.1, delta_table_path=str(table_path))
    card.train()
    card(torch.ones(1, 2), ["a"])
    assert card.epoch_stats["teacher_delta_norm_sum"] == pytest.approx(5.0)


def test_card_eval_does_not_accumulate_training_epoch_stats(tmp_path):
    from card_adapter import CardAdapter

    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["a"]), delta=np.array([[3.0, 4.0]]))
    card = CardAdapter(hidden_dim=2, bottleneck=2, lambda_delta=0.1, delta_table_path=str(table_path))
    card.train()
    card(torch.ones(1, 2), ["a"])
    card.pop_epoch_stats()
    card.eval()
    card(torch.ones(1, 2), ["a"])
    assert card.epoch_stats["loss_count"] == 0  # review P1-1


def test_card_zero_target_is_safe(tmp_path):
    # Review P0-1 Test C / §10: a zero teacher delta must not enter the
    # distillation loss; everything stays finite and the zero targets are
    # counted.
    from card_adapter import CardAdapter

    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["z1", "z2"]), delta=np.zeros((2, 2)))
    card = CardAdapter(hidden_dim=2, bottleneck=2, lambda_delta=0.1, delta_table_path=str(table_path))
    card.train()
    out = card(torch.ones(2, 2), ["z1", "z2"])
    assert torch.isfinite(out).all()
    assert torch.isfinite(card.last_loss)
    stats = card.pop_epoch_stats()
    assert stats["card_valid_target_fraction"] == pytest.approx(0.0)
    assert stats["card_zero_target_count"] == 2


def test_card_training_requires_full_sample_id_alignment(tmp_path):
    # Review P1-6: a training batch whose sample-id count does not match the
    # representation batch size must fail fast instead of silently disabling
    # the distillation loss.
    from card_adapter import CardAdapter

    table_path = tmp_path / "delta.npz"
    np.savez(table_path, ids=np.array(["a", "c"]), delta=np.ones((2, 2)))
    card = CardAdapter(hidden_dim=2, bottleneck=2, lambda_delta=0.1, delta_table_path=str(table_path))
    card.train()
    with pytest.raises(RuntimeError, match="one sample_id per representation row"):
        card(torch.ones(3, 2), ["a", "c"])  # 3 rows, 2 ids


def test_same_seed_human3_and_animal56_native_backbones_differ():
    # Review P0-4 premise: same seed does NOT give the same backbone across
    # task scopes (56 vs 3 decoders consume different RNG first).  This test
    # documents the premise so nobody re-assumes coordinate equality.
    model_human, _, _ = _build_model(42, "human3", torch.device("cpu"))
    model_animal, _, _ = _build_model(42, "animal56", torch.device("cpu"))
    human_backbone = {k: v for k, v in model_human.state_dict().items() if k.startswith("encoder.backbone.")}
    animal_backbone = {k: v for k, v in model_animal.state_dict().items() if k.startswith("encoder.backbone.")}
    assert any(
        not torch.equal(human_backbone[key], animal_backbone[key])
        for key in human_backbone
    )


def _write_initial_state_snapshot(run_dir, model, sha=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            },
            "initial_model_sha256": sha or state_dict_sha256(model),
            "seed": 42,
        },
        run_dir / "initial_model.pt",
    )
    return run_dir / "initial_model.pt"


def test_teacher_anchor_snapshot_roundtrip(tmp_path):
    # D6 P0-4 root fix: the anchor is loaded from the teacher run's recorded
    # initial-state snapshot, verified on both payload hash and actual weights.
    from scripts.d6_prep_inits import _load_teacher_anchor_state, _state_dict_sha256

    model, _, _ = _build_model(42, "animal56", torch.device("cpu"))
    run_dir = tmp_path / "teacher"
    _write_initial_state_snapshot(run_dir, model)
    expected = state_dict_sha256(model)
    state = _load_teacher_anchor_state(run_dir, expected)
    assert _state_dict_sha256(state) == expected


def test_teacher_anchor_snapshot_rejects_missing_file(tmp_path):
    from scripts.d6_prep_inits import _load_teacher_anchor_state

    with pytest.raises(SystemExit, match="missing initial_model.pt"):
        _load_teacher_anchor_state(tmp_path, "some-hash")


def test_teacher_anchor_snapshot_rejects_wrong_hash(tmp_path):
    from scripts.d6_prep_inits import _load_teacher_anchor_state

    model, _, _ = _build_model(42, "animal56", torch.device("cpu"))
    run_dir = tmp_path / "teacher"
    _write_initial_state_snapshot(run_dir, model, sha="wrong-hash")
    with pytest.raises(SystemExit, match="hash mismatch"):
        _load_teacher_anchor_state(run_dir, "expected-hash")


def test_teacher_anchor_snapshot_rejects_tampered_weights(tmp_path):
    # A snapshot claiming hash X whose weights hash to Y must be refused.
    from scripts.d6_prep_inits import _load_teacher_anchor_state

    model, _, _ = _build_model(42, "animal56", torch.device("cpu"))
    run_dir = tmp_path / "teacher"
    run_dir.mkdir(parents=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "initial_model_sha256": "claimed",
            "seed": 42,
        },
        run_dir / "initial_model.pt",
    )
    with pytest.raises(SystemExit, match="do not match the recorded hash"):
        _load_teacher_anchor_state(run_dir, "claimed")


def test_anchor_only_preserves_human_heads():
    from scripts.d6_prep_inits import build_anchor_state

    model_human, _, _ = _build_model(42, "human3", torch.device("cpu"))
    # Build a genuine animal initial model with the same seed.
    model_animal, _, _ = _build_model(42, "animal56", torch.device("cpu"))
    anchor = build_anchor_state(model_human, model_animal.state_dict())
    human_sd = model_human.state_dict()
    animal_sd = model_animal.state_dict()
    for key, value in anchor.items():
        if key.startswith("encoder.backbone."):
            assert torch.equal(value, animal_sd[key])
        else:
            assert torch.equal(value, human_sd[key])


def test_csdt_alpha_zero_equals_anchor():
    from scripts.d6_prep_inits import apply_csdt

    anchor = {"encoder.backbone.a": torch.tensor([1.0, 1.0]), "decoders.head": torch.tensor([0.5])}
    real = {"encoder.backbone.a": torch.tensor([3.0, 2.0])}
    shuffle = {"encoder.backbone.a": torch.tensor([1.0, 0.0])}
    merged = apply_csdt(anchor, real, shuffle, alpha=0.0)
    for key, value in anchor.items():
        assert torch.equal(merged[key], value)


def test_csdt_alpha_one_exact_delta():
    from scripts.d6_prep_inits import apply_csdt

    anchor = {"encoder.backbone.a": torch.tensor([1.0, 1.0])}
    real = {"encoder.backbone.a": torch.tensor([3.0, 2.0])}
    shuffle = {"encoder.backbone.a": torch.tensor([1.0, 0.0])}
    merged = apply_csdt(anchor, real, shuffle, alpha=1.0)
    assert torch.allclose(merged["encoder.backbone.a"], torch.tensor([3.0, 3.0]))


# ----------------------------------------------------------------------
# Review P0-2 (§17-§26) / P1-7 (§43-§44): the unified artifact provenance
# validator must bind init states and CARD tables to the CURRENT run's
# manifest/datastore/feature-schema identity before any training step.
# ----------------------------------------------------------------------
import hashlib
from pathlib import Path

from main import _d6_artifact_contract, _validate_d6_artifact_provenance

RUN_DATA_METADATA = {
    "split_manifest_hash": "m",
    "datastore_fingerprint": "f",
    "feature_schema_version": "atom_v2_bond_v1_pathavg_v1",
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_artifact_with_sidecar(
    tmp_path,
    filename,
    mode,
    *,
    seed=42,
    manifest="m",
    fingerprint="f",
    schema="atom_v2_bond_v1_pathavg_v1",
    omit_sha=False,
    tamper_artifact=False,
):
    artifact = tmp_path / filename
    torch.save({"model_state": {"w": torch.zeros(1)}}, artifact)
    provenance = {
        "mode": mode,
        "human_seed": seed,
        "expected_teacher_epoch": 29,
        "split_manifest_hash": manifest,
        "datastore_fingerprint": fingerprint,
        "feature_schema_version": schema,
    }
    if not omit_sha:
        provenance["output_sha256"] = _digest(artifact)
    if tamper_artifact:
        # Mutate the artifact AFTER the sha was recorded — a post-hoc swap
        # must be caught by the output_sha256 check (review §21).
        torch.save({"model_state": {"w": torch.ones(1)}}, artifact)
    sidecar = artifact.with_name(artifact.name + ".provenance.json")
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    return artifact, sidecar


def test_o3_accepts_matching_card_table_provenance(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(tmp_path, "card_table.pt", "card_table")
    provenance = _validate_d6_artifact_provenance(
        provenance_path=sidecar,
        artifact_path=artifact,
        candidate="o3",
        seed=42,
        data_metadata=RUN_DATA_METADATA,
    )
    contract = _d6_artifact_contract(provenance)
    assert contract["mode"] == "card_table"
    assert contract["human_seed"] == 42
    assert contract["split_manifest_hash"] == "m"
    assert contract["datastore_fingerprint"] == "f"
    assert contract["expected_teacher_epoch"] == 29


def test_o3_rejects_card_table_from_wrong_seed(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "card_table.pt", "card_table", seed=44
    )
    with pytest.raises(ValueError, match="human_seed"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o3",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_o3_rejects_card_table_from_wrong_manifest(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "card_table.pt", "card_table", manifest="other-split"
    )
    with pytest.raises(ValueError, match="split_manifest_hash"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o3",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_o3_rejects_card_table_from_wrong_datastore(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "card_table.pt", "card_table", fingerprint="other-store"
    )
    with pytest.raises(ValueError, match="datastore_fingerprint"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o3",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_o3_rejects_non_card_table_provenance(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(tmp_path, "card_table.pt", "clst")
    with pytest.raises(ValueError, match="mode"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o3",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_o3_rejects_tampered_card_table(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "card_table.pt", "card_table", tamper_artifact=True
    )
    with pytest.raises(ValueError, match="output_sha256"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o3",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_o3_rejects_sidecar_without_artifact_sha(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "card_table.pt", "card_table", omit_sha=True
    )
    with pytest.raises(ValueError, match="output_sha256"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o3",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_o1_rejects_init_from_wrong_manifest(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "o1_init.pt", "csdt", manifest="other-split"
    )
    with pytest.raises(ValueError, match="split_manifest_hash"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o1",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_o2_rejects_init_from_wrong_datastore(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "o2_init.pt", "clst", fingerprint="other-store"
    )
    with pytest.raises(ValueError, match="datastore_fingerprint"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="o2",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_b1_rejects_init_from_wrong_feature_schema(tmp_path):
    artifact, sidecar = _write_artifact_with_sidecar(
        tmp_path, "b1_init.pt", "b1", schema="other-schema"
    )
    with pytest.raises(ValueError, match="feature_schema_version"):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="b1",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )


def test_b0a_rejects_missing_sidecar(tmp_path):
    artifact, _ = _write_artifact_with_sidecar(tmp_path, "b0a_init.pt", "anchor")
    sidecar = artifact.with_name(artifact.name + ".provenance.json")
    sidecar.unlink()
    with pytest.raises(FileNotFoundError):
        _validate_d6_artifact_provenance(
            provenance_path=sidecar, artifact_path=artifact, candidate="b0a",
            seed=42, data_metadata=RUN_DATA_METADATA,
        )
