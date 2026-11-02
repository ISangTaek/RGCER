import numpy as np
import torch

from architecture.toxacute_tasks import TOXACUTE_TASKS
from pgt_v2.chemistry import graph_from_input, prepare_molecule
from pgt_v2.config import candidate
from pgt_v2.data import PGTCollator
from pgt_v2.endpoint_contrast import endpoint_design
from pgt_v2.evaluation import load_model_checkpoint
from pgt_v2.models import build_model, task_metadata
from pgt_v2.training import save_checkpoint


def test_reference_balances_changed_fields_and_excludes_missing_conditions():
    tasks = ['rat_oral_LD50', 'mouse_oral_LD50', 'rabbit_oral_LD50', 'rat_skin_LD50',
             'rat_oral_LDLo', 'rat_unreported_LD50']
    metadata = task_metadata(tasks)
    _, reference = endpoint_design([metadata[t] for t in tasks], [20, 10, 100, 2, 9, 40])
    torch.testing.assert_close(reference[0, 1:3].sum(), torch.tensor(1 / 3))
    assert reference[0, 2] > reference[0, 1]
    torch.testing.assert_close(reference[0, 3:5], torch.full((2,), 1 / 3))
    assert not reference[5].any() and not reference[:, 5].any()


def test_all59_nested_model_can_learn_and_reload_condition_contrasts(tmp_path):
    tasks = list(TOXACUTE_TASKS)
    config = dict(candidate('M2', 'C2'), hidden_dim=16, ffn_dim=32, layers=1)
    counts = np.arange(1, len(tasks) + 1)
    batch = PGTCollator()([graph_from_input(prepare_molecule(s), sample_id=s, raw_row_index=i)
                           for i, s in enumerate(('C', 'CC(O)C', 'c1ccncc1'))])
    def create(method):
        torch.manual_seed(42)
        return build_model(method, tasks, config, counts=counts).eval()
    base = create('M2')
    expected_rng = torch.get_rng_state()
    model = create('N5')
    assert torch.equal(expected_rng, torch.get_rng_state())
    shuffled = create('N5-Shuffle')
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in shuffled.parameters())
    assert not torch.equal(model.endpoint_contrast.features, shuffled.endpoint_contrast.features)
    reference = base(batch).detach()
    torch.testing.assert_close(reference, model(batch), atol=0, rtol=0)
    prediction, aux = model(batch, return_aux=True)
    weights = aux['target_attention']
    assert weights.shape[:2] == (3, 59)
    assert not torch.allclose(weights[1, -1], weights[1, -2])
    assert not weights.masked_select(batch.padding_mask[:, None]).any()
    # One synthetic update checks that zero initial correction is trainable.
    loss = (prediction - torch.linspace(-1, 1, 59)).square().mean()
    loss.backward()
    assert model.endpoint_contrast.value.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.endpoint_contrast.value.weight.add_(model.endpoint_contrast.value.weight.grad, alpha=-0.1)
    model.zero_grad()
    pred, aux = model(batch, return_aux=True)
    assert aux['prompt_delta'][1:].abs().max() > 0
    torch.testing.assert_close(aux['prompt_delta'][0], torch.zeros(59), atol=1e-6, rtol=0)
    torch.testing.assert_close(pred, reference + aux['prompt_atom_contribution'].sum(-1))
    torch.testing.assert_close(model(batch, prompt_enabled=False), reference, atol=0, rtol=0)
    pred[1, -1].backward()
    for layer in (model.endpoint_contrast.query, model.endpoint_contrast.key):
        assert layer.weight.grad.isfinite().all() and layer.weight.grad.abs().sum() > 0
    path = tmp_path / 'best.pt'
    save_checkpoint(path, dict(kind='best', identity=dict(method_id='N5', task_names=tasks, config=config),
                               model=model.state_dict(), stats=dict(count=counts, adjacency=np.eye(59))))
    restored, _ = load_model_checkpoint(path)
    torch.testing.assert_close(restored(batch), pred.detach(), atol=0, rtol=0)
