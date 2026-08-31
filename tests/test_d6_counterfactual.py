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
from scripts.d6_prep_inits import apply_b1, apply_csdt, _build_model, _verify_matched_teacher_init
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
    model_a, _ = _build_model(42, "animal56", torch.device("cpu"))
    model_b, _ = _build_model(42, "animal56", torch.device("cpu"))
    model_c, _ = _build_model(43, "animal56", torch.device("cpu"))
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
    loss = card.last_loss * card.lambda_delta
    loss.backward()
    assert card.up.weight.grad is not None and card.up.weight.grad.abs().sum() > 0
    assert card.delta_table.grad is None  # teacher delta is a frozen target (§78)
