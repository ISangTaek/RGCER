"""Remove old, inactive DataStore V2 builds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ready_builds(root: Path) -> list[Path]:
    builds = root / "builds"
    if not builds.exists():
        return []
    return sorted(
        [path for path in builds.iterdir() if path.is_dir() and (path / "READY").exists()],
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )


def collect_garbage(root: str | Path, keep: int = 2) -> list[Path]:
    """Delete inactive READY builds, never the CURRENT build."""

    if int(keep) < 1:
        raise ValueError("keep must be at least 1")
    root = Path(root).resolve()
    current_path = root / "CURRENT"
    current = current_path.read_text(encoding="utf-8").strip() if current_path.exists() else None
    builds = _ready_builds(root)
    retained = {path.name for path in builds[: int(keep)]}
    if current:
        retained.add(current)
    removed = []
    for path in builds:
        if path.name in retained:
            continue
        import shutil

        shutil.rmtree(path)
        removed.append(path)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Garbage collect inactive DataStore V2 builds")
    parser.add_argument("--root", required=True)
    parser.add_argument("--keep", type=int, default=2)
    args = parser.parse_args()
    for path in collect_garbage(args.root, args.keep):
        print(f"removed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
