import subprocess

import pytest
import torch

from baselines.constants import CHEMPROP_COMMIT, GROVER_BASE_SHA256, GROVER_COMMIT, TOXACOL_COMMIT
from baselines.models.grover import build_grover, validate_grover_encoder_state
from baselines.utils import sha256_file


def _commit(path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def test_exact_official_source_commits_and_grover_weight(chemprop_source, grover_source, grover_pretrained, toxacol_source):
    assert _commit(chemprop_source) == CHEMPROP_COMMIT
    assert _commit(grover_source) == GROVER_COMMIT
    assert _commit(toxacol_source) == TOXACOL_COMMIT
    assert sha256_file(grover_pretrained) == GROVER_BASE_SHA256
    _, _, audit = build_grover(grover_source, grover_pretrained, torch.device("cpu"))
    assert audit["loaded_encoder_key_count"] == 106
    assert audit["shape_mismatch"] == []


def test_grover_wrong_sha_is_rejected_before_source_loading(tmp_path):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the approved checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_grover(tmp_path / "missing-source", wrong, torch.device("cpu"))


def test_grover_encoder_validator_rejects_missing_key_and_shape():
    own = {"grover.a": torch.zeros(2, 3), "grover.b": torch.zeros(1), "head": torch.zeros(1)}
    with pytest.raises(ValueError, match="missing"):
        validate_grover_encoder_state({"grover.a": torch.zeros(2, 3)}, own)
    with pytest.raises(ValueError, match="shape_mismatch"):
        validate_grover_encoder_state({"grover.a": torch.zeros(3, 2), "grover.b": torch.zeros(1)}, own)
