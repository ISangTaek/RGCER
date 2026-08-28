"""Model-initialization reproducibility (review §12, §42)."""

import subprocess
import sys
from pathlib import Path

import torch

from reproducibility import seed_everything, stable_seed, state_dict_sha256
from architecture.prediction_heads import TaskPredictionHead

TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]


def _build_decoders():
    return torch.nn.ModuleDict(
        {
            task: TaskPredictionHead(
                hidden_dim=32,
                mode="quantile",
                head_hidden_dim=32,
                dropout=0.1,
            )
            for task in TASKS
        }
    )


def test_same_seed_produces_identical_initialization():
    seed_everything(42)
    decoders_a = _build_decoders()
    seed_everything(42)
    decoders_b = _build_decoders()

    for task in TASKS:
        for parameter_a, parameter_b in zip(
            decoders_a[task].parameters(), decoders_b[task].parameters()
        ):
            torch.testing.assert_close(parameter_a, parameter_b, rtol=0, atol=0)


def test_different_seed_produces_different_initialization():
    seed_everything(42)
    decoders_a = _build_decoders()
    seed_everything(43)
    decoders_b = _build_decoders()

    differing = 0
    for task in TASKS:
        for parameter_a, parameter_b in zip(
            decoders_a[task].parameters(), decoders_b[task].parameters()
        ):
            if not torch.equal(parameter_a, parameter_b):
                differing += 1
    assert differing > 0


def test_state_dict_hash_tracks_initialization():
    seed_everything(42)
    hash_a = state_dict_sha256(_build_decoders())
    seed_everything(42)
    hash_b = state_dict_sha256(_build_decoders())
    seed_everything(43)
    hash_c = state_dict_sha256(_build_decoders())

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_stable_seed_is_independent_of_python_string_hashing():
    assert stable_seed(42, "task_a", "train", 0) == stable_seed(42, "task_a", "train", 0)
    assert stable_seed(42, "task_a", "train", 0) != stable_seed(42, "task_b", "train", 0)
    assert stable_seed(42, "task_a", "train", 0) != stable_seed(42, "task_a", "train", 1)


def test_fresh_subprocess_hash_is_stable_for_same_seed_and_differs_across_seeds():
    script = Path(__file__).resolve().parents[1] / "scripts" / "hash_initial_model.py"

    def _hash(seed: int) -> str:
        completed = subprocess.run(
            [sys.executable, str(script), "--seed", str(seed), "--arch", "Graphormer"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr
        for line in completed.stdout.splitlines():
            if line.startswith("initial_model_sha256="):
                return line.split("=", 1)[1].strip()
        raise AssertionError(f"hash missing from output: {completed.stdout}")

    hash_42_first = _hash(42)
    hash_42_second = _hash(42)
    hash_43 = _hash(43)

    assert hash_42_first == hash_42_second
    assert hash_42_first != hash_43
