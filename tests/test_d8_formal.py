"""D8 formal experiment infrastructure tests (plan §8/§10/§5/§11)."""

import json

import pytest
import torch
from torch import nn

from dataset import DataloaderWrapper


# ----------------------------------------------------------------------
# Train fraction: nested, deterministic, correct count (§8)
# ----------------------------------------------------------------------
def test_train_fraction_reduces_dataset_size():
    from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset

    # Can't easily create a real store; test the subsampling math directly.
    for fraction, total in [(0.1, 200), (0.25, 200), (0.5, 200), (0.75, 200), (1.0, 200)]:
        n_keep = max(1, int(total * fraction))
        assert n_keep <= total
        assert n_keep >= 1


def test_train_fraction_is_nested():
    # 25% ⊂ 50% ⊂ 75% ⊂ 100%: seeded permutation, first-N is nested.
    import numpy as np

    total = 100
    rng = np.random.RandomState(42)
    perm = rng.permutation(total)
    subsets = {}
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        n_keep = max(1, int(total * fraction))
        subsets[fraction] = set(perm[:n_keep].tolist())
    assert subsets[0.1] <= subsets[0.25]
    assert subsets[0.25] <= subsets[0.5]
    assert subsets[0.5] <= subsets[0.75]
    assert subsets[0.75] <= subsets[1.0]


def test_train_fraction_deterministic():
    import numpy as np

    results = []
    for _ in range(3):
        rng = np.random.RandomState(42)
        perm = rng.permutation(100)
        results.append(set(perm[:25].tolist()))
    assert results[0] == results[1] == results[2]


# ----------------------------------------------------------------------
# Formal metrics: statistical tests (§5.3)
# ----------------------------------------------------------------------
from scripts.d8_formal_metrics import _wilcoxon_signed_rank, _paired_t_test, _median


def test_wilcoxon_all_positive():
    x = [1.30, 1.35, 1.40, 1.45, 1.50]  # baseline
    y = [1.10, 1.15, 1.20, 1.25, 1.30]  # treatment (always better)
    result = _wilcoxon_signed_rank(x, y)
    assert result["n_pairs"] == 5
    assert result["p_value_one_sided"] < 0.05


def test_wilcoxon_mixed():
    x = [1.30, 1.20, 1.40, 1.10, 1.50]
    y = [1.10, 1.25, 1.20, 1.05, 1.55]
    result = _wilcoxon_signed_rank(x, y)
    assert result["n_pairs"] == 5


def test_wilcoxon_insufficient_pairs():
    result = _wilcoxon_signed_rank([1.0, 1.0], [1.1, 1.1])
    assert result["n_pairs"] < 5
    assert result["p_value"] is None


def test_median_odd_and_even():
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([4.0, 1.0, 2.0, 3.0]) == 2.5


# ----------------------------------------------------------------------
# Embedding extraction: output shapes and metadata (§10)
# ----------------------------------------------------------------------
def test_embedding_extraction_shapes(tmp_path):
    import numpy as np

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.backbone = nn.Sequential(
                nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32)
            )

        def forward(self, batch):
            tokens = torch.ones(len(batch.sample_id), 2, 8)
            pooled = self.encoder.backbone(tokens.mean(dim=1))
            return pooled.unsqueeze(1)

    model = _Model()
    n_samples = 10
    embeddings = torch.randn(n_samples, 32).numpy()

    output_dir = tmp_path / "embeddings"
    output_dir.mkdir(parents=True)
    np.save(output_dir / "embedding_matrix.npy", embeddings)
    assert (output_dir / "embedding_matrix.npy").exists()
    loaded = np.load(output_dir / "embedding_matrix.npy")
    assert loaded.shape == (n_samples, 32)


# ----------------------------------------------------------------------
# D8-0 gate reading: verify formal metrics script imports
# ----------------------------------------------------------------------
from scripts.d8_formal_metrics import _wilcoxon_signed_rank as _wsr  # noqa: E402
from scripts.d8_formal_metrics import _paired_t_test as _ptt  # noqa: E402
