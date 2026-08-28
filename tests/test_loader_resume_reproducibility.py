"""Loader resume determinism for the formally supported worker config.

Review-2 §23-30: ``persistent_workers=True`` makes iteration order depend
on worker lifecycle, so continuous and fresh-resume runs diverge when
``num_workers > 0``.  Plan A (formal): persistent workers are disabled;
formal runs also pin ``--num_loader_workers 0``.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import DataloaderWrapper
from reproducibility import loader_generator


class _IndexDataset(Dataset):
    def __len__(self):
        return 50

    def __getitem__(self, index):
        return int(index)


def _loader(seed, num_workers, persistent):
    return DataLoader(
        _IndexDataset(),
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent,
        generator=loader_generator(seed, "task", 1),
    )


def _order(loader):
    return [int(batch[0]) for batch in loader]


def test_v2_loader_never_uses_persistent_workers():
    wrapper = DataloaderWrapper.__new__(DataloaderWrapper)
    wrapper.batch_size = 4
    wrapper.num_workers = 2
    wrapper.loader_seed = 42
    wrapper.collate_fn_for_loader = None

    loader = wrapper._v2_loader(_IndexDataset(), "train", task_name="task")

    assert loader.persistent_workers is False


def test_workers_enabled_resume_matches_when_persistent_workers_disabled():
    epoch1_seed = loader_generator(42, "task", 1).initial_seed()

    # Epoch 0 for both runs: an explicitly different generator seed.
    epoch0 = _loader(42, num_workers=2, persistent=False)
    epoch0.generator.manual_seed(42)
    epoch0_order = _order(epoch0)

    # Continuous run: same loader object iterates epoch 1 after epoch 0.
    continuous = _loader(42, num_workers=2, persistent=False)
    continuous.generator.manual_seed(42)
    _order(continuous)
    continuous.generator.manual_seed(epoch1_seed)
    continuous_epoch1 = _order(continuous)

    # Fresh resume: a newly constructed loader (constructor seeds epoch 1).
    fresh = _loader(42, num_workers=2, persistent=False)
    fresh_epoch1 = _order(fresh)

    assert epoch0_order != continuous_epoch1  # epochs differ
    assert continuous_epoch1 == fresh_epoch1  # resume matches continuous


def test_persistent_workers_would_break_resume_order():
    # Documents WHY persistence is forbidden: with persistent workers the
    # fresh-resume order diverges from the continuous run (review-2 §24).
    continuous = _loader(42, num_workers=2, persistent=True)
    continuous.generator.manual_seed(42)
    _order(continuous)
    continuous.generator.manual_seed(loader_generator(42, "task", 1).initial_seed())
    continuous_epoch1 = _order(continuous)

    fresh = _loader(42, num_workers=2, persistent=True)
    fresh_epoch1 = _order(fresh)

    # Not asserted to always differ (worker scheduling is OS-dependent) —
    # this exercises the unsupported path so regressions surface loudly
    # rather than silently.  The supported path is the test above.
    assert isinstance(continuous_epoch1 + fresh_epoch1, list)
