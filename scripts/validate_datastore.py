"""Strict offline validator for a ToxAcute DataStore V2 build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toxacute_datastore import ToxAcuteDataStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ToxAcute DataStore V2")
    parser.add_argument("--root", "--data_store_dir", dest="root", required=True)
    parser.add_argument("--task_names", default=None, help="Comma-separated expected task names")
    parser.add_argument("--max_path_distance", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    task_names = None if args.task_names is None else [item for item in args.task_names.split(",") if item]
    store = ToxAcuteDataStore.resolve(args.root)
    try:
        store.validate(
            strict=bool(args.strict),
            expected_task_names=task_names,
            expected_max_path_distance=args.max_path_distance,
        )
        print(json.dumps({"status": "OK", **store.context.__dict__}, indent=2, sort_keys=True))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
