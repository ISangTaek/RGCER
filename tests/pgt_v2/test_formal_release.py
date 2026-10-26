"""Bounded checks for the two C06F interface changes, without external assets."""
import numpy as np
import pytest
import torch

from pgt_v2.fragment_explanations import matched_pairs, pair_concordance
from pgt_v2.training import train_model, read_checkpoint, seed_everything


def test_fixed_pairs_keep_ties_zero_responses_and_missing_scope():
    fragments = [dict(size_bin=1, aromatic=False, a_count=2, b_count=1) for _ in range(3)]
    fragments += [dict(size_bin=3, aromatic=False, a_count=2, b_count=1),
                  dict(size_bin=1, aromatic=False, a_count=0, b_count=1)]
    pairs = matched_pairs(fragments)
    assert pairs == [(0,1), (0,2), (1,2)]
    assert pair_concordance([1,2,3,99,99], [1,2,3,0,0], pairs) == 1.
    assert pair_concordance([1,2,3,99,99], [3,2,1,0,0], pairs) == -1.
    assert pair_concordance([1,1,3,99,99], [0,0,2,0,0], pairs) == pytest.approx(2/3)
    assert pair_concordance([1,2,3,99,99], [0,0,0,0,0], pairs) == 0.
    assert pair_concordance([1,None,3,99,99], [1,2,3,0,0], pairs) is None
    assert pair_concordance([1], [1], []) is None


def test_auxiliary_checkpoint_diverges_without_changing_main_and_survives_resume(tmp_path, monkeypatch):
    from pgt_v2 import training
    class Inputs:
        records = {'group_id':np.array(['a','b','c','d'])}
        def __len__(self): return 4
        def batch(self, indices): return torch.ones(len(indices), 1)
    inputs = Inputs(); labels = np.zeros((4,2))
    stats = dict(mean=np.zeros(2), scale=np.ones(2), count=np.array([4,4]))
    base = dict(lr=.01, weight_decay=0., effective_batch=4, micro_batch=2, grad_clip=5.,
                max_epochs=3, min_epochs=3, patience=30)
    cfg = {**base, 'auxiliary_selection_tasks':['sparse']}
    identity = dict(task_names=['sparse','dense'], config=cfg)
    reports = [np.tile(v,(4,1)) for v in ([1.,9.], [2.,2.], [3.,3.])]
    def set_predictions(values):
        stream = iter(values)
        monkeypatch.setattr(training, 'predict_array', lambda *a, **kw: next(stream))
    seed_everything(42); model = torch.nn.Linear(1,2)
    set_predictions(reports[:1])
    train_model(model, inputs, labels, inputs, labels, stats, config=cfg, identity=identity,
                output=tmp_path/'aux', method='N1', stop_after_epoch=1)
    set_predictions(reports[1:])
    resumed = torch.nn.Linear(1,2)
    result = train_model(resumed, inputs, labels, inputs, labels, stats, config=cfg, identity=identity,
                output=tmp_path/'aux', method='N1', resume=True)
    assert result['best_epoch'] == 2 and result['selection_score'] == 2.
    assert result['auxiliary_selection']['epoch'] == 1
    assert result['auxiliary_selection']['score'] == 1.
    aux = read_checkpoint(tmp_path/'aux/best_s21.pt')
    assert aux['selection_role'] == 'AUXILIARY_S21' and aux['best_epoch'] == 1
    main = read_checkpoint(tmp_path/'aux/best.pt')
    seed_everything(42); baseline = torch.nn.Linear(1,2)
    set_predictions(reports)
    train_model(baseline, inputs, labels, inputs, labels, stats, config=base,
                identity=dict(task_names=['sparse','dense'], config=base), output=tmp_path/'main', method='N1')
    without_aux = read_checkpoint(tmp_path/'main/best.pt')
    assert main['best_epoch'] == without_aux['best_epoch'] == 2
    for key in main['model']:
        assert torch.equal(main['model'][key], without_aux['model'][key])
