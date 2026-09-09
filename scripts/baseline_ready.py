#!/usr/bin/env python
"""Fail-closed preflight for all real data, source, and GROVER weight assets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.constants import CHEMPROP_COMMIT, GROVER_BASE_SHA256, GROVER_COMMIT, TOXACOL_COMMIT
from baselines.data import load_human3_smoke
from baselines.models.dmpnn import activate_chemprop
from baselines.models.grover import build_grover
from baselines.utils import sha256_file, write_json


def _commit(path: str | Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datastore", required=True)
    parser.add_argument("--chemprop-source", required=True)
    parser.add_argument("--grover-source", required=True)
    parser.add_argument("--grover-pretrained", required=True)
    parser.add_argument("--toxacol-source", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    smoke = load_human3_smoke(args.datastore)
    activate_chemprop(args.chemprop_source)
    if _commit(args.grover_source) != GROVER_COMMIT:
        raise RuntimeError("GROVER source commit mismatch")
    if _commit(args.toxacol_source) != TOXACOL_COMMIT:
        raise RuntimeError("TOXACol source commit mismatch")
    if _commit(args.chemprop_source) != CHEMPROP_COMMIT:
        raise RuntimeError("Chemprop source commit mismatch")
    if sha256_file(args.grover_pretrained) != GROVER_BASE_SHA256:
        raise RuntimeError("GROVER_base weight SHA-256 mismatch")
    _, _, grover_audit = build_grover(args.grover_source, args.grover_pretrained, torch.device("cpu"))
    result = {
        "status": "READY",
        "real_assets_required": True,
        "skips_accepted_as_ready": False,
        "datastore": smoke.datastore_root,
        "chemprop_commit": CHEMPROP_COMMIT,
        "grover_commit": GROVER_COMMIT,
        "grover_weights_sha256": GROVER_BASE_SHA256,
        "grover_loaded_encoder_key_count": grover_audit["loaded_encoder_key_count"],
        "toxacol_commit": TOXACOL_COMMIT,
    }
    if args.output_json:
        write_json(args.output_json, result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
