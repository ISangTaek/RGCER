"""Formal reproducibility contract (review §4-11, §31-34).

Every random decision in a run must be derivable from the base ``--seed``
plus stable identifiers (task, split, epoch) instead of process history.
``seed_everything`` must be called in ``main()`` before any ``nn.Module``
construction; Trainer keeps a defensive re-seed only.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping

import numpy as np
import torch

SEED_POLICY_VERSION = 2
EPOCH_SEED_SCHEME = "sha256(base_seed|train_epoch|epoch)"
LOADER_SEED_SCHEME = "sha256(base_seed|task|train|epoch)"
# Review-2 §27: persistent workers make DataLoader iteration order depend on
# worker lifecycle, so strict resume requires them disabled; formal runs
# additionally fix --num_loader_workers 0.
PERSISTENT_WORKER_POLICY = "disabled_for_strict_resume"


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> int:
    """Seed python/numpy/torch (CPU + CUDA) global RNGs; return the seed."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    return seed


def stable_seed(base_seed: int, *parts) -> int:
    """Deterministic 31-bit seed derived from the base seed and labels.

    Uses SHA-256, never Python's ``hash()`` (which is per-process
    randomized for strings).
    """

    payload = "|".join([str(int(base_seed)), *map(str, parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**31 - 1)


def state_dict_sha256(module: torch.nn.Module) -> str:
    """Content hash of a module's parameters/buffers, order-stable."""

    hasher = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def loader_generator(base_seed: int, task_name: str, epoch: int) -> torch.Generator:
    """Independent torch.Generator for one (task, epoch) train shuffle."""

    generator = torch.Generator()
    generator.manual_seed(stable_seed(base_seed, task_name, "train", int(epoch)))
    return generator


def reseed_train_loader(loader, base_seed: int, task_name: str, epoch: int) -> bool:
    """Reset a train DataLoader's shuffle generator for ``epoch``.

    Returns False when the loader has no dedicated generator (eval loaders
    run with ``shuffle=False`` and need nothing).  Decoupling shuffle from
    the global RNG keeps paired-seed runs (HPS vs RGCER) on identical batch
    orders even though the two models consume different amounts of global
    RNG during initialization.
    """

    generator = getattr(loader, "generator", None)
    if generator is None:
        return False
    generator.manual_seed(stable_seed(base_seed, task_name, "train", int(epoch)))
    return True


def reseed_train_loaders(
    train_dataloaders_dict: Mapping[str, object], base_seed: int, epoch: int
) -> dict[str, bool]:
    """Reseed every task's train loader for ``epoch``; report what happened."""

    return {
        task: reseed_train_loader(loader, base_seed, task, epoch)
        for task, loader in train_dataloaders_dict.items()
        if loader is not None
    }


def consume_loader_order(loader, limit: int | None = None) -> list:
    """Draw the raw shuffle order of a loader (test helper)."""

    items: list = []
    for batch in loader:
        items.append(batch)
        if limit is not None and len(items) >= limit:
            break
    return items


def scheduled_batches(
    schedule: Iterator[str] | list[str],
    loaders: Mapping[str, object],
) -> Iterator[tuple[str, object]]:
    """Yield ``(task, batch)`` honoring the planned schedule exactly.

    A task may appear in the schedule more times than its loader has
    batches (``human_target_floor`` extra passes).  The iterator restarts
    only while scheduled-but-unconsumed occurrences remain, so extra passes
    are really consumed and no infinite cycling is possible.
    """

    planned: dict[str, int] = {}
    for task in schedule:
        planned[task] = planned.get(task, 0) + 1
    iterators: dict[str, object] = {}
    consumed: dict[str, int] = {}
    for task in schedule:
        if task not in iterators:
            iterators[task] = iter(loaders[task])
            consumed[task] = 0
        batch = next(iterators[task], None)
        exhausted = batch is None
        if exhausted and consumed[task] < planned[task]:
            iterators[task] = iter(loaders[task])
            batch = next(iterators[task], None)
        if batch is None:
            continue
        consumed[task] += 1
        yield task, batch
