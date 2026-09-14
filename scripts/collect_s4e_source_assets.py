"""CLI for bounded, CPU-only S4E source-asset collection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s4e_source_assets import S4ECollectionError, collect_source_assets  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect SHA-bound source/init/teacher metadata for S4E without "
            "constructing a model, running inference, or reading predictions."
        )
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--low-policy", required=True, type=Path)
    parser.add_argument("--core-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        code = collect_source_assets(
            arguments.repo_root,
            arguments.low_policy,
            arguments.core_lock,
            arguments.output,
        )
    except (OSError, S4ECollectionError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "execution_status": "ERROR",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "acceptance_status": "PENDING_REVIEW",
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "execution_status": "COMPLETE" if code == 0 else "COMPLETE_WITH_ISSUES",
                "exit_code": code,
                "output": str(arguments.output.resolve()),
                "acceptance_status": "PENDING_REVIEW",
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
