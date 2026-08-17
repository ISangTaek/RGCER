from unittest.mock import patch

import torch

from tests.test_rgcer_contract import build_model
from dataset import DataCollator
from preprocess_data import get_graph_data_from_smiles


def test_forced_null_is_exact_hps_fallback():
    model, _ = build_model()
    model.train()
    # Ensure the assertion also covers the two decoder calls in train mode.
    model.decoders["task_a"].network[3].p = 0.5
    batch = DataCollator()([get_graph_data_from_smiles("CCO", 1.0)])
    router = model.encoder.task_conditioner.router
    original = router.forward

    def forced_null(*args, **kwargs):
        context, source, joint, _null, entropy = original(*args, **kwargs)
        return context, source, torch.zeros_like(joint), torch.ones_like(_null), entropy * 0.0

    with patch.object(router, "forward", side_effect=forced_null):
        _, diagnostics = model(batch, task_name="task_a", return_aux=True)
    assert diagnostics["final_representation"] is None
    assert torch.equal(diagnostics["final_raw"], diagnostics["base_raw"])
    assert torch.equal(diagnostics["final_prediction"], diagnostics["base_prediction"])
