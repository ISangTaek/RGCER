from types import SimpleNamespace

import torch

from conformal import ConformalCalibrator
from trainer import Trainer


def test_quantile_output_is_denormalized_before_interval_conversion():
    trainer = Trainer.__new__(Trainer)
    trainer.task_scalers = {"task": {"mean": 10.0, "std": 2.0}}
    trainer.task_dict = {"task": {"metrics": ["RMSE"]}}
    trainer.args = SimpleNamespace(prediction_mode="quantile")
    trainer.conformal_calibrator = ConformalCalibrator(alpha=0.1, min_calibration_size=0)
    decoded = trainer.decode_task_output("task", torch.tensor([[1.5, 0.0, 0.0]]), apply_conformal=False)
    assert torch.allclose(decoded["median"], torch.tensor([[13.0]]))
    assert decoded["lower"].item() < 13.0 < decoded["upper"].item()
