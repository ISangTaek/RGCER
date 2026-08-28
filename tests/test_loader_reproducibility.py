"""DataLoader shuffle RNG decoupling (review §8-11, §13)."""

import torch
from torch.utils.data import DataLoader, Dataset

from reproducibility import loader_generator, reseed_train_loader


class _IndexDataset(Dataset):
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return int(index)


def _train_loader(seed: int, task: str, epoch: int) -> DataLoader:
    dataset = _IndexDataset(100)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=loader_generator(seed, task, epoch),
    )


def _order(loader: DataLoader) -> list[int]:
    return [int(batch[0]) for batch in loader]


def test_same_seed_same_task_same_epoch_yields_identical_order():
    first = _order(_train_loader(42, "man_oral_TDLo", 0))
    second = _order(_train_loader(42, "man_oral_TDLo", 0))
    assert first == second


def test_different_seed_yields_different_order():
    order_42 = _order(_train_loader(42, "man_oral_TDLo", 0))
    order_43 = _order(_train_loader(43, "man_oral_TDLo", 0))
    assert order_42 != order_43


def test_epoch_seed_yields_epoch_specific_but_reproducible_orders():
    epoch0_first = _order(_train_loader(42, "task", 0))
    epoch1_first = _order(_train_loader(42, "task", 1))
    epoch0_second = _order(_train_loader(42, "task", 0))
    assert epoch0_first != epoch1_first
    assert epoch0_first == epoch0_second


def test_shuffle_is_decoupled_from_global_model_rng():
    # Review §13 "different models": consuming different amounts of global
    # RNG (as differently-sized model initializations do) must not change
    # the loader permutation.
    torch.manual_seed(0)
    torch.rand(1000)
    order_small = _order(_train_loader(42, "task", 0))

    torch.manual_seed(1)
    torch.rand(7777)
    order_large = _order(_train_loader(42, "task", 0))

    assert order_small == order_large


def test_reseed_train_loader_resets_permutation_for_epoch():
    loader = _train_loader(42, "task", 0)
    order_epoch0_first = _order(loader)

    # Continuing the same generator gives a fresh permutation...
    order_continued = _order(loader)
    assert order_continued != order_epoch0_first

    # ...but reseeding to (seed, task, epoch=0) replays the original one.
    assert reseed_train_loader(loader, 42, "task", 0) is True
    order_epoch0_again = _order(loader)
    assert order_epoch0_again == order_epoch0_first


def test_reseed_reports_false_for_generatorless_loader():
    loader = DataLoader(_IndexDataset(4), batch_size=2, shuffle=False)
    assert reseed_train_loader(loader, 42, "task", 0) is False
