#!/usr/bin/env python
"""Run one full train/validation baseline trial from the frozen configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.config import resolve_training_config
from baselines.constants import METHODS
from baselines.runner import run_training


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method", required=True, choices=METHODS + ("attentivefp",))
    value.add_argument("--config", help="Method config JSON; defaults to configs/baselines/<method>.json")
    value.add_argument("--trials", help="Trial-list JSON; defaults to configs/baselines/trials/<method>.json")
    value.add_argument("--trial", type=int, required=True, help="Zero-based frozen trial index")
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--datastore", required=True)
    value.add_argument("--output", "--output-dir", dest="output_dir", required=True)
    value.add_argument("--device", default="cpu")
    value.add_argument("--source-root", help="Exact official source tree for D-MPNN or GROVER")
    value.add_argument("--pretrained", help="Approved official GROVER_base checkpoint; training only")
    return value


def main() -> int:
    args = parser().parse_args()
    method = "afp" if args.method == "attentivefp" else args.method
    config = resolve_training_config(
        method,
        role="formal",
        seed=args.seed,
        device=args.device,
        config_path=args.config,
        trials_path=args.trials,
        trial_index=args.trial,
    )
    result = run_training(
        method,
        args.datastore,
        args.output_dir,
        config=config,
        source_root=args.source_root,
        pretrained=args.pretrained,
    )
    print(f"PASS {result['role']} {result['method']} -> {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
