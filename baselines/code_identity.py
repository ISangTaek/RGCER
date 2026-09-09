"""Content identity for the uncommitted baseline implementation under test."""

from __future__ import annotations

from pathlib import Path

from .utils import canonical_sha256, sha256_file


def implementation_identity(repository_root: str | Path) -> dict:
    root = Path(repository_root).resolve()
    files = []
    patterns = (
        "baselines/**/*.py",
        "configs/baselines/**/*.json",
        "scripts/baseline_*.py",
        "scripts/generate_baseline_trials.py",
        "scripts/verify_baseline_smoke.py",
        "tests/baselines/**/*.py",
    )
    selected = set()
    for pattern in patterns:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        })
    if not files:
        raise ValueError("No baseline implementation files found for code identity")
    return {
        "algorithm": "sha256(canonical_json(path+sha256))",
        "implementation_sha256": canonical_sha256(files),
        "file_count": len(files),
        "files": files,
    }
