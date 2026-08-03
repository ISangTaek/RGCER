from types import SimpleNamespace

from trainer import Trainer


class SizedLoader:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_schedule_uses_each_task_batch_once():
    trainer = Trainer.__new__(Trainer)
    trainer.task_name = ["a", "b", "c"]
    trainer.seed = 5
    schedule = trainer._task_schedule({"a": SizedLoader(5), "b": SizedLoader(2), "c": SizedLoader(1)}, 0)
    assert {task: schedule.count(task) for task in trainer.task_name} == {"a": 5, "b": 2, "c": 1}
