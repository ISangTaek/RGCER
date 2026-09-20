"""Synthetic CPU-only factory/adapter checks; no approved source is trained."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

import p1d_routes as mod
from dataset115_adapter import Dataset115Table
from dataset115_contract import ContractError, PRIMARY, digest
from dataset115_model import build_human5_model
from dataset115_route_a_training import RouteATrainer, SourceTrainer
from dataset115_smoke import state_digest
from dataset115_training import RouteBTrainer, task_batches
from tests.test_dataset115_adapter import args, view
from tests.test_dataset115_route_a_smoke import TASKS
from tests.test_dataset115_training import validation_view


@pytest.fixture
def setup_factory(monkeypatch):
    # Actual Dataset115Table.view and actual Graphormer/engines; source loader
    # boundaries alone are substituted with a small synthetic reviewed asset.
    table = Dataset115Table()
    table.identity = view().input_identity
    table.tasks = PRIMARY + tuple(TASKS)
    table.source_tasks = tuple(TASKS)
    table.overlaps = frozenset()
    table.records = tuple(dict(row_index=i, raw_smiles=s, canonical_smiles=s,
                               split_group=s, split='train' if i < 2 else 'validation')
                          for i, s in enumerate(('CC', 'CCC', 'CO', 'CN')))
    target = np.concatenate([view().labels, validation_view().labels])
    table.values = np.concatenate([target, np.tile([[0.], [2.], [0.], [2.]], (1, 104))], axis=1)
    table.values.setflags(write=False)
    encoder = build_human5_model(args(), method='B0', seed=123).encoder.state_dict()
    calls = []

    def forbid_source_run(*a, **kw):
        pytest.fail('source retraining is forbidden')

    monkeypatch.setattr(SourceTrainer, 'run', forbid_source_run)
    monkeypatch.setattr(mod, 'ARCH', vars(args()))

    def make(route='B', seed=42):
        source = dict(seed=seed, teacher_sha256='1'*64, init_sha256='2'*64)
        if route == 'A':
            source.update(route='A', selection='fixed_final_epoch39',
                          init_sha256=state_digest(encoder), source_identity_sha256='3'*64)

        def load_b(repo, lock, actual_seed):
            calls.append(('B', repo, lock, actual_seed))
            assert actual_seed == seed
            return deepcopy(encoder), args(), dict(teacher={'sha256': source['teacher_sha256']},
                                                   init={'sha256': source['init_sha256']})

        def load_a(output, trainer):
            calls.append(('A', output, trainer))
            assert type(trainer) is SourceTrainer
            assert trainer.config == mod.CONFIG and trainer.seed == seed
            assert trainer.train.split == 'train' and trainer.train.role == 'source'
            return deepcopy(encoder), deepcopy(source), {'formal_source': True}

        monkeypatch.setattr(mod, 'load_route_b_encoder', load_b)
        monkeypatch.setattr(mod, 'load_source', load_a)
        engine = RouteATrainer if route == 'A' else RouteBTrainer
        reference = engine(args(), encoder, source, table.view(route, 'target', 'train'),
                           table.view(route, 'target', 'validation'), method='B1', seed=seed,
                           config=mod.CONFIG, device='cpu')
        expected = mod.expected_identity_from_record(reference.identity,
                    initial_encoder=reference.initial_encoder, initial_heads=reference.initial_heads)
        kwargs = dict(route=route, seed=seed, expected_identity=expected)
        kwargs.update(dict(source_output='synthetic-source') if route == 'A'
                      else dict(source_repo='synthetic-repo', source_lock='synthetic-lock'))
        return mod.RouteFactory(table, **kwargs), reference, kwargs

    return make, table, calls


@pytest.mark.parametrize('route', ['A', 'B'])
def test_original_and_controlled_exact_paired_updates(setup_factory, route):
    make, _, calls = setup_factory
    factory, reference, _ = make(route)
    assert set(factory.expected_identity) == set(mod.EXPECTED_FIELDS)
    original = factory.make_trainer(original=True)
    controlled = factory.make_trainer()
    assert original.optimization is None
    assert controlled.optimization.spec.arm == 'B1_high'
    assert original.identity == reference.identity
    assert original.config == controlled.config == mod.CONFIG
    assert asdict(mod.CONFIG) == dict(epochs=40, batch_size=32, learning_rate=.001,
                                     weight_decay=1e-5, grad_clip=1.)
    assert original.initial_heads == controlled.initial_heads
    assert original.initial_encoder == controlled.initial_encoder
    from loss import QuantileRegressionLoss
    loss_fn = QuantileRegressionLoss()
    controlled.optimization.begin_epoch(0)
    counts = {t: len(ds) for t, ds in original.datasets['train'].items()}
    for task, indices in task_batches(counts, 32, 42, 0)[:2]:
        rng = torch.get_rng_state().clone()
        outputs = []
        for trainer in (original, controlled):
            torch.set_rng_state(rng)
            trainer.model.train()
            batch = trainer._batch('train', task, indices)
            scaler = trainer.scalers[task]
            trainer.optimizer.zero_grad(set_to_none=True)
            loss = loss_fn.compute_loss(trainer.model(batch, task_name=task)[task],
                    (batch.y.reshape(-1, 1)-scaler['mean'])/scaler['std'])
            loss.backward()
            grads = {k: None if p.grad is None else p.grad.clone()
                     for k, p in trainer.model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1., error_if_nonfinite=True)
            trainer.optimizer.step()
            outputs.append((loss.detach(), grads, deepcopy(trainer.optimizer.state_dict())))
        assert torch.equal(outputs[0][0], outputs[1][0])
        assert_nested_equal(outputs[0][1], outputs[1][1])
        assert_nested_equal(outputs[0][2], outputs[1][2])
        assert_nested_equal(original.model.state_dict(), controlled.model.state_dict())
    assert len(calls) == 2


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert type(a) is type(b) and a == b


@pytest.mark.parametrize('field', mod.EXPECTED_FIELDS)
def test_every_expected_field_bound(setup_factory, field):
    make, table, _ = setup_factory
    _, _, kwargs = make()
    expected = kwargs['expected_identity']
    if field in ('source_identity', 'architecture', 'scaler'):
        expected[field]['unknown'] = True
    elif field == 'task_names':
        expected[field] = list(reversed(PRIMARY))
    else:
        expected[field] = 'f'*64
    with pytest.raises(ContractError):
        mod.RouteFactory(table, **kwargs).make_trainer()


@pytest.mark.parametrize('change', ['missing', 'unknown', 'null', 'seed_bool', 'float_dimension', 'bool_count'])
def test_contract_schema_and_nested_types_fail_closed(setup_factory, change):
    make, table, _ = setup_factory
    _, _, kwargs = make()
    expected = kwargs['expected_identity']
    if change == 'missing':
        del expected['initial_heads']
    elif change == 'unknown':
        expected['test'] = False
    elif change == 'null':
        expected['scaler'] = None
    elif change == 'seed_bool':
        kwargs['seed'] = True
    elif change == 'float_dimension':
        expected['architecture']['a_layers'] = 1.0
    else:
        expected['scaler']['scaler']['counts'][0] = True
    with pytest.raises(ContractError):
        mod.RouteFactory(table, **kwargs).make_trainer()


def test_reference_copy_and_no_scope_or_config_overrides(setup_factory):
    make, _, _ = setup_factory
    factory, _, kwargs = make()
    kwargs['expected_identity']['initial_heads'] = 'f'*64
    factory.expected_identity['initial_encoder'] = 'f'*64
    factory.make_trainer()
    for extra in (dict(config=mod.CONFIG), dict(epochs=1), dict(split='test'),
                  dict(method='RPT'), dict(resume='x')):
        with pytest.raises(TypeError):
            factory.make_trainer(**extra)
    for extra in (dict(original=1), dict(original=True, arm='HF_low'), dict(device='cuda:1')):
        with pytest.raises(ValueError):
            factory.make_trainer(**extra)


def test_mutated_table_population_fails_before_source_load(setup_factory):
    make, table, calls = setup_factory
    factory, _, _ = make()
    table.values = table.values.copy()
    table.values[0, 0] += 1
    with pytest.raises(ContractError, match='train_observations'):
        factory.make_trainer()
    assert not calls


def test_from_files_calls_real_audit_boundary(setup_factory, monkeypatch):
    make, table, _ = setup_factory
    _, _, kwargs = make()
    calls = []
    def load(*a, **kw):
        calls.append((a, kw))
        return table
    monkeypatch.setattr(Dataset115Table, 'load', load)
    mod.RouteFactory.from_files(csv_path='csv', split_path='split', tox_path='tox',
                                expected_tox_sha='e'*64, **kwargs).make_trainer()
    assert calls == [(('csv', 'split', 'tox'), {'expected_tox_sha': 'e'*64})]


@pytest.mark.parametrize('route,arm', [('B', 'B1_high'), ('B', 'B1_low'),
                                      ('B', 'HF_high'), ('B', 'HF_low'), ('A', 'HF_low')])
def test_single_run_and_fresh_independent_verification(setup_factory, tmp_path, route, arm):
    make, _, calls = setup_factory
    factory, _, _ = make(route)
    adapter = mod.SingleRunAdapter(factory, arm=arm)
    receipt = adapter.run(tmp_path/'run')
    assert receipt['route'] == route and receipt['arm'] == arm
    assert receipt['epochs_checked'] == 40 and receipt['optimization_checked'] is True
    assert receipt['acceptance_status'] == 'PENDING_REVIEW'
    assert receipt['test_predictions_accessed'] is False
    assert receipt['calibration_predictions_accessed'] is False
    assert len(calls) == 2  # Train and fresh verifier reload the source independently.
    with pytest.raises(ContractError, match='already used'):
        adapter.run(tmp_path/'second')
    assert not (tmp_path/'second').exists()


@pytest.fixture
def synthetic_run(setup_factory, tmp_path):
    make, _, _ = setup_factory
    factory, _, _ = make()
    adapter = mod.SingleRunAdapter(factory, arm='HF_low')
    output = tmp_path/'run'
    factory.controlled_trainer(arm='HF_low').run(output)
    return adapter, output


@pytest.mark.parametrize('tamper', ['lr', 'step', 'record', 'frozen', 'label'])
def test_self_consistent_artifact_tampering_rejected(synthetic_run, tamper):
    adapter, output = synthetic_run
    if tamper == 'label':
        path = output/'validation_epoch_000.json'
        rows = json.loads(path.read_text())
        rows[0]['label'] += 1
        path.write_text(json.dumps(rows))
    else:
        # Last epoch has no descendant histories to update. Frozen drift uses
        # epoch0, with all descendant histories unchanged (only tensors change).
        epoch = 0 if tamper == 'frozen' else 39
        path = output/f'epoch_{epoch:03d}.pt'
        payload = torch.load(path, weights_only=True)
        if tamper == 'lr':
            payload['optimizer_state']['param_groups'][0]['lr'] = .9
        elif tamper == 'step':
            next(iter(payload['optimizer_state']['state'].values()))['step'] += 1
        elif tamper == 'record':
            payload['history'][-1]['optimization']['frozen'] = True
        else:
            key = next(k for k in payload['model_state'] if k.startswith('encoder.backbone.'))
            payload['model_state'][key] += 1
            payload['model_digest'] = state_digest(payload['model_state'])
            payload['best_model_state'] = deepcopy(payload['model_state'])
            payload['best_model_digest'] = payload['model_digest']
        torch.save(payload, path)
        rp = output/f'epoch_{epoch:03d}.receipt.json'
        receipt = json.loads(rp.read_text())
        receipt.update(sha256=digest(path), size_bytes=path.stat().st_size)
        rp.write_text(json.dumps(receipt))
        sp = output/'training_summary.json'
        summary = json.loads(sp.read_text())
        summary['checkpoints'][epoch] = receipt
        if tamper == 'record':
            summary['history'] = payload['history']
        sp.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        adapter.verify(output)


def test_failed_attempt_consumed_without_retry(setup_factory, tmp_path, monkeypatch):
    make, _, _ = setup_factory
    factory, _, _ = make()
    def fail(**kw):
        raise ContractError('missing reviewed source')
    monkeypatch.setattr(factory, 'controlled_trainer', fail)
    adapter = mod.SingleRunAdapter(factory)
    with pytest.raises(ContractError, match='missing reviewed source'):
        adapter.run(tmp_path/'run')
    with pytest.raises(ContractError, match='already used'):
        adapter.run(tmp_path/'retry')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('route', ['A', 'B'])
def test_route_runtime_smoke_and_live_zero_updates(setup_factory, tmp_path, route):
    """Real route numerics through the parent's smoke and live replay APIs."""
    from dataset import DataCollator
    from p1d_runtime import route_step, verify_live_gradients
    from p1d_smoke import SmokeAdapter, capture_rng, model_digest, run_setting_smoke

    make, _, source_calls = setup_factory
    factory, reference, _ = make(route, seed=42)
    requested_arms, requested_batches, optimizer_calls = [], [], []
    live_phase = False
    def before_step(optimizer, a, kw):
        # This observes actual AdamW calls, independently of either receipt.
        assert not live_phase, 'live gradient verification performed an optimizer update'

    def after_step(optimizer, a, kw):
        optimizer_calls.append(1)

    def make_trainer(arm):
        # Exactly the RouteFactory mapping used by p1d_runtime.prepare.
        requested_arms.append(arm)
        trainer = factory.make_trainer(original=arm is None, arm=arm or 'B1_high', device='cpu')
        trainer.optimizer.register_step_pre_hook(before_step)
        trainer.optimizer.register_step_post_hook(after_step)
        return trainer

    def get_batch(task, index):
        assert task in PRIMARY and type(index) is int and 0 <= index < 8
        ds = reference.datasets['train'][task]
        assert ds.view.split == 'train' and ds.view.route == route
        batch = DataCollator()([ds[j] for j in range(min(2, len(ds)))]).to('cpu')
        assert batch.y.device.type == 'cpu' and torch.isfinite(batch.y).all()
        assert list(batch.sample_id) == [ds.get_sample_id(j) for j in range(2)]
        requested_batches.append((task, index))
        return batch

    identity = dict(setting=route, seed=42, contract=factory.expected_identity,
                    batch_policy='first_two_train_records_per_task_in_fixed_view',
                    tasks=list(PRIMARY), test_accessed=False, calibration_accessed=False,
                    initial_model_sha256=model_digest(reference.model.state_dict()))
    adapter = SmokeAdapter(make_trainer, route_step, get_batch)
    output = tmp_path/route
    before_rng = capture_rng()
    receipt = run_setting_smoke(adapter, setting=route, task_names=list(PRIMARY),
                                output_dir=output, expected_identity=identity)
    assert_nested_equal(before_rng, capture_rng())
    assert receipt['observed_optimizer_updates'] == len(optimizer_calls) == 11
    assert receipt['updates_by_branch'] == dict(original=2, B1_high=2, HF_low=6, resume=1)
    assert receipt['comparison'] == 'exact'
    assert receipt['scope'] == 'SMOKE_ONLY_NOT_FORMAL_TRAINING'
    assert receipt['acceptance_status'] == 'PENDING_REVIEW'
    assert requested_arms == [None, 'B1_high', 'HF_low', 'HF_low']
    schedule = list(enumerate(receipt['pair_tasks'] + receipt['hf_tasks']))
    assert requested_batches == [(task, index) for index, task in schedule]

    live_phase = True
    live = verify_live_gradients(output, adapter, setting=route, expected_identity=identity)
    assert_nested_equal(before_rng, capture_rng())
    assert live['optimizer_updates'] == 0 and len(optimizer_calls) == 11
    assert live['scope'] == 'TRAIN_BATCH_LOSS_GRADIENT_REPLAY_NO_OPTIMIZER_STEPS'
    assert len(live['checks']) == 11 and all(row['gradients_equal'] for row in live['checks'])
    assert [row['artifact'] for row in live['checks']] == (
        [f'original_{i}.pt' for i in range(2)] + [f'B1_high_{i}.pt' for i in range(2)]
        + [f'HF_low_{i}.pt' for i in range(6)] + ['resumed_epoch5.pt'])
    assert requested_arms == [None, 'B1_high', 'HF_low', 'HF_low', None]
    assert len(source_calls) == 5
    assert requested_batches[8:] == (
        [(task, index) for index, task in schedule[:2]] * 2
        + [(task, index) for index, task in schedule[2:]] + [(schedule[7][1], 7)])
