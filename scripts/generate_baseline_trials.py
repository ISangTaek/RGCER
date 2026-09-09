#!/usr/bin/env python
"""Regenerate the deterministic protocol-approved HPO trial lists."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.constants import METHODS
from baselines.trials import finalize_learning_rates, generate_trials
from baselines.utils import canonical_sha256, write_json


def main() -> int:
    target = ROOT / "configs" / "baselines" / "trials"
    for method in METHODS:
        trials = finalize_learning_rates(method, generate_trials(method, seed=42))
        numbered = [{"trial_id": index, **parameters} for index, parameters in enumerate(trials)]
        write_json(target / f"{method}.json", {
            "method": method,
            "seed": 42,
            "sampler": "numpy.random.RandomState(42)",
            "without_replacement": method == "rf",
            "trial_count": len(numbered),
            "trials_sha256": canonical_sha256(numbered),
            "trials": numbered,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
