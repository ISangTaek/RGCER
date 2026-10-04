from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import d2_data as mod
import d2_control as core
from dataset115_contract import semantic_digest
from dataset115_smoke import state_digest
from tests.test_dataset115_adapter import args
from tests.test_d2_control import data_factory, one_thread


@pytest.mark.parametrize('seed', range(42, 47))
def test_route_a_actual_checkpoint_schema_uses_identity_seed_and_tasks(tmp_path, seed):
    a = args()
    enc = core.encoder(a, 99).state_dict()
    tasks = [f'animal{i}_oral_LD50' for i in range(104)]
    identity = dict(seed=seed, task_names=tasks, scaler=dict(scaler=dict(task_names=tasks, counts=[3]*104)))
    payload = dict(epoch=39, identity=identity, identity_sha256=semantic_digest(identity),
                   model_state={'encoder.'+k: v for k, v in enc.items()})
    path = tmp_path/'epoch_039.pt'
    torch.save(payload, path)
    contract = dict(initial_encoder=state_digest(enc), source_identity=dict(seed=seed,
                    teacher_sha256=mod.sha(path), source_identity_sha256=payload['identity_sha256']))
    state, names, counts, bound = mod.locked_source(SimpleNamespace(_output=tmp_path), 'A', seed,
                                                 tmp_path, None, a, contract)
    assert names == tasks and counts == dict.fromkeys(tasks, 3) and bound['seed'] == seed
    assert state_digest(state) == state_digest(enc)


@pytest.mark.parametrize('seed', range(42, 47))
def test_animal_checkpoint_binding_uses_requested_seed(tmp_path, monkeypatch, seed):
    import dataset115_source
    import s4e_mechanism_smoke
    a = args()
    enc = core.encoder(a, 99).state_dict()
    tasks = [f'animal{i}_oral_LD50' for i in range(56)]
    teacher = dict(task_names=tasks, sha256='e'*64)
    payload = dict(configuration=dict(seed=seed), epoch=39, task_scalers={t: dict(count=3) for t in tasks})
    state = {'encoder.'+k: v for k, v in enc.items()}
    seen = []
    def binding(path, actual_seed):
        seen.append(actual_seed)
        return teacher, {}
    monkeypatch.setattr(dataset115_source, 'load_binding', binding)
    monkeypatch.setattr(s4e_mechanism_smoke, 'state_from_asset', lambda *a: (payload, state, {}))
    source_lock = tmp_path/'lock.json'
    source_lock.write_text('{}')
    values, _, counts, _ = mod.locked_source(None, 'B', seed, tmp_path, source_lock, a,
                                            dict(initial_encoder=state_digest(enc)))
    assert seen == [seed] and len(counts) == 56 and state_digest(values) == state_digest(enc)


@pytest.mark.parametrize('damage', ['sha_scheme', 'missing', 'nan', 'shape'])
def test_extract_source_encoder_fails_closed(damage):
    a = args()
    enc = core.encoder(a, 99)
    expected = state_digest(enc.state_dict())
    state = {'encoder.'+k: v.clone() for k, v in enc.state_dict().items()}
    key = next(iter(state))
    if damage == 'sha_scheme':
        expected = mod.state_dict_sha256(enc)
    elif damage == 'missing':
        del state[key]
    elif damage == 'nan':
        state[key].fill_(float('nan'))
    else:
        state[key] = torch.zeros(1)
    with pytest.raises((ValueError, RuntimeError)):
        mod.extract_encoder(state, a, expected)


@pytest.mark.parametrize('split', ['test', 'calibration'])
def test_adapter_has_no_holdout_batch_api(data_factory, split):
    data = data_factory('PJ')
    with pytest.raises(ValueError, match='scope'):
        data.batch('target', split, data.target_tasks[0], [0], 'cpu')


@pytest.mark.parametrize('damage', ['label', 'canonical', 'sample_id'])
def test_graph_batch_checked_against_trusted_observations(data_factory, damage):
    data = data_factory('PJ')
    task = data.target_tasks[0]
    row = data.by_task['train'][task][0]
    row[damage] = 100. if damage == 'label' else 'wrong'
    with pytest.raises(ValueError):
        data.batch('target', 'train', task, [0], 'cpu')


@pytest.mark.parametrize('seed', range(42, 47))
def test_legacy_source_extension_forwards_seed_without_changing_default(monkeypatch, seed):
    import dataset115_source
    import v9_joint_source
    seen = []
    def binding(path, supplied):
        seen.append(supplied)
        raise RuntimeError('stop after seed observation')
    monkeypatch.setattr(dataset115_source, 'load_binding', binding)
    with pytest.raises(RuntimeError, match='seed observation'):
        v9_joint_source.load(None, SimpleNamespace(model=None), 'B', Path('.'), None, seed=seed)
    assert seen == [seed]


@pytest.mark.parametrize('seed', [True, 41, 47, '42'])
def test_invalid_source_seed_rejected_before_assets(seed):
    import v9_joint_source
    with pytest.raises(ValueError, match='seed'):
        v9_joint_source.load(None, None, 'B', Path('.'), None, seed=seed)
