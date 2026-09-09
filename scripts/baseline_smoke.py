#!/usr/bin/env python
"""Run one protocol-locked local Human3 smoke baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.constants import METHODS
from baselines.runner import run_smoke


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method", required=True, choices=METHODS + ("attentivefp",))
    value.add_argument("--datastore", required=True)
    value.add_argument("--output", "--output-dir", dest="output_dir", required=True)
    value.add_argument("--device", default="cpu")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--config", help="Method config JSON; defaults to configs/baselines/<method>.json")
    value.add_argument("--source-root", help="Exact official source tree for dmpnn/grover")
    value.add_argument("--pretrained", help="Official GROVER_base checkpoint")
    value.add_argument("--split", default="validation", choices=("validation", "test", "calibration"))
    return value


def main() -> int:
    args = parser().parse_args()
    if args.split != "validation":
        raise SystemExit("Human3 smoke is locked to train/validation; calibration and test are forbidden")
    result = run_smoke(
        ("afp" if args.method == "attentivefp" else args.method), args.datastore, args.output_dir, device=args.device, seed=args.seed,
        source_root=args.source_root, pretrained=args.pretrained,
        config_path=args.config,
    )
    print(f"PASS {result['method']} -> {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
