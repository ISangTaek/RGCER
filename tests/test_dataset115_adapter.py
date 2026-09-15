from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dataset115_adapter import LabelView, TrainOnlyScaler, GraphTaskView, Dataset115Table
from dataset115_contract import ContractError, PRIMARY
from dataset115_model import build_human5_model
from tests.test_dataset115_contract import audit_inputs


def test_loading_and_views_use_frozen_routes(audit_inputs):
    from dataset115_contract import digest
    c, s, t = audit_inputs
    table = Dataset115Table.load(c, s, t, expected_tox_sha=digest(t))
    for split in ('train', 'validation', 'calibration', 'test'):
        a = table.view('A', 'target', split); b = table.view('B', 'target', split)
        assert len(a.sample_ids) == 1 and len(b.sample_ids) == 0
        assert a.labels[0, 0] == 0 and np.isnan(a.labels[0, 2])
        assert not set(table.view('A', 'source', split).tasks) & set(PRIMARY)
    with pytest.raises(ContractError):
        table.view('B', 'source', 'train')


def test_readonly_labels_and_bad_namespace():
    v = view()
    with pytest.raises(ValueError):
        v.labels[0, 0] = 123
    with pytest.raises(ContractError, match='namespace'):
        replace(v, sample_ids=('toxacute:row_0', 'dataset115:row_1'))


def view(split='train', values=None):
    return LabelView('B', 'target', split, PRIMARY,
        ('dataset115:row_0', 'dataset115:row_1'), ('CC', 'CCC'), ('CC', 'CCC'), ('CC', 'CCC'),
        np.array([[0, 2, 3, 4, 5], [2, 4, 5, 6, 7]]) if values is None else values, 'a' * 64)


def test_scaler_uses_only_training_and_roundtrips():
    train = view(); scaler = TrainOnlyScaler.fit(train)
    evaluation = view('validation', np.full((2, 5), 100000.0))
    scaled, mask = scaler.transform(evaluation)
    assert np.all(scaler.scaler.means == np.array([1, 3, 4, 5, 6]))
    np.testing.assert_allclose(scaler.scaler.inverse_transform(scaled), evaluation.labels)
    assert mask.sum() == 10
    assert scaler.trainer_scalers()[PRIMARY[0]] == dict(mean=1.0, std=1.0, count=2)


@pytest.mark.parametrize('split', ['validation', 'calibration', 'test'])
def test_nontrain_fit_rejected(split):
    with pytest.raises(ContractError, match='train-only'):
        TrainOnlyScaler.fit(view(split))


def test_missing_not_zero_and_constant_column():
    v = view(values=[[0, 2, 3, 4, 5], [np.nan, 2, 5, 6, 7]])
    scaler = TrainOnlyScaler.fit(v); z, mask = scaler.transform(v)
    assert mask[:, 0].tolist() == [1, 0]
    assert scaler.scaler.counts[0] == 1 and scaler.scaler.stds[1] == 1e-6
    assert z[1, 0] == 0


def test_empty_task_not_silent():
    v = view(values=np.full((2, 5), np.nan))
    with pytest.raises(ValueError, match='No finite training'):
        TrainOnlyScaler.fit(v)


@pytest.mark.parametrize('scale', ['mg/kg', '-log10(mg/kg)', None])
def test_unit_conversion_not_implicit(scale):
    with pytest.raises(ContractError, match='scale'):
        replace(view(), scale=scale)


@pytest.mark.parametrize('changes', [dict(route='A'), dict(input_identity='b' * 64)])
def test_scaler_cross_route_or_data_forbidden(changes):
    s = TrainOnlyScaler.fit(view())
    with pytest.raises(ContractError, match='identity'):
        s.transform(replace(view('test'), **changes))


def test_task_order_is_not_silently_reinterpreted():
    with pytest.raises(ContractError, match='order'):
        replace(view(), tasks=tuple(reversed(PRIMARY)))


def test_source_cannot_include_human():
    with pytest.raises(ContractError, match='human source'):
        replace(view(), role='source', route='A')


def test_infinite_label_rejected():
    with pytest.raises(ContractError, match='nonfinite'):
        view(values=np.full((2, 5), np.inf))


def test_raw_graph_y_not_double_standardized():
    from dataset import DataCollator
    v = view(); ds = GraphTaskView(v, PRIMARY[1]); batch = DataCollator()([ds[0], ds[1]])
    assert batch.y.flatten().tolist() == [2.0, 4.0]
    assert ds.get_sample_id(0) == 'dataset115:row_0'


def test_graph_canonical_mismatch_rejected():
    v = replace(view(), canonical=('C', 'CCC'))
    with pytest.raises(ContractError, match='canonicalization'):
        GraphTaskView(v, PRIMARY[0])[0]


def args():
    return SimpleNamespace(a_heads=2, a_layers=1, hidden_dim=16, mid_dim=32,
        head_hidden_dim=8, head_dropout=0.1, edge_bias_mode='path', spatial_pos_clip=20)


def test_paired_heads_source_copy_and_freeze_with_backward():
    from dataset import DataCollator
    torch.manual_seed(99); before = torch.random.get_rng_state().clone()
    b0 = build_human5_model(args(), method='B0', seed=42)
    assert torch.equal(before, torch.random.get_rng_state())
    source = {k: v.clone() for k, v in b0.encoder.state_dict().items()}
    b1 = build_human5_model(args(), method='B1', seed=42, source_encoder_state=source)
    rpt = build_human5_model(args(), method='RPT', seed=42, source_encoder_state=source)
    for key, value in b1.decoders.state_dict().items():
        assert torch.equal(value, rpt.decoders.state_dict()[key])
        assert torch.equal(value, b0.decoders.state_dict()[key])
    batch = DataCollator()([GraphTaskView(view(), PRIMARY[0])[0]])
    for model in (b0, b1, rpt):
        model.train(); outputs = model(batch, return_all_tasks=True)
        assert list(outputs) == list(PRIMARY)
        assert all(v.shape == (1, 3) and torch.isfinite(v).all() for v in outputs.values())
        sum(v.sum() for v in outputs.values()).backward()
        assert all(p.grad is not None for p in model.decoders.parameters())
    assert all(p.grad is None and not p.requires_grad for p in rpt.encoder.parameters())
    assert any(p.grad is not None for p in b1.encoder.parameters())
    assert rpt.encoder.training  # Preserve established gradient-only freeze semantics.
    for key, value in source.items():
        assert torch.equal(value, rpt.encoder.state_dict()[key])


@pytest.mark.parametrize('method,source', [('B0', {}), ('B1', None), ('RPT', {}), ('wrong', None)])
def test_invalid_source_or_method_rejected(method, source):
    with pytest.raises(ContractError):
        build_human5_model(args(), method=method, seed=42, source_encoder_state=source)


def test_nonfinite_source_rejected():
    source = build_human5_model(args(), method='B0', seed=42).encoder.state_dict()
    key = next(iter(source)); source[key] = torch.full_like(source[key], float('nan'))
    with pytest.raises(ContractError, match='tensor'):
        build_human5_model(args(), method='B1', seed=42, source_encoder_state=source)


def test_cli_existing_evidence_not_overwritten(tmp_path):
    from scripts.verify_dataset115_adapter import main
    out = tmp_path / 'result.json'; out.write_text('original')
    with pytest.raises(SystemExit) as exc:
        main(['--csv', 'missing', '--split-manifest', 'missing', '--tox-manifest', 'missing',
              '--expected-tox-file-sha256', '0'*64, '--output', str(out)])
    assert exc.value.code == 2 and out.read_text() == 'original'


def test_cli_bad_input_produces_no_success(tmp_path):
    from scripts.verify_dataset115_adapter import main
    out = tmp_path / 'result.json'
    with pytest.raises(SystemExit) as exc:
        main(['--csv', str(tmp_path/'missing'), '--split-manifest', 'missing', '--tox-manifest', 'missing',
              '--expected-tox-file-sha256', '0'*64, '--output', str(out)])
    assert exc.value.code == 2 and not out.exists()
