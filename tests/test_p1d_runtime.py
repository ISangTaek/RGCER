import pytest
from scripts.run_p1d_smoke import parser


@pytest.mark.parametrize('option', ['--epochs','--seed','--test','--split','--budget','--resume'])
def test_cli_has_no_scope_expansion(option):
    with pytest.raises(SystemExit):
        parser().parse_args(['--expected-commit','a'*40,'--setting','A','--output','out',
                            '--split-manifest','split','--source-lock','lock',option,'1'])


def test_cli_verify_only_scope():
    a=parser().parse_args(['--expected-commit','a'*40,'--setting','ToxAcute','--output','out',
                          '--split-manifest','split','--source-lock','lock','--verify-only'])
    assert a.verify_only and a.setting=='ToxAcute'


def test_attempt_cannot_be_refunded_by_new_output(tmp_path):
    from p1d_runtime import claim_attempt
    for setting in ('A','B','ToxAcute'):
        claim_attempt(tmp_path,setting,'a'*40,tmp_path/setting)
        with pytest.raises(FileExistsError):claim_attempt(tmp_path,setting,'b'*40,tmp_path/(setting+'retry'))
    assert len(list((tmp_path/'.tmp/p1d3_attempts_20260920').glob('*.json')))==3


def test_batch_binding_rejects_equal_labels_wrong_ids():
    from p1d_runtime import batch_identity,same
    from tests.test_p1d_tox import fixture_factory,TASKS
    from p1d_optimization import OptimizationError
    f=fixture_factory();batch=f.batch(TASKS[0],0)
    expected=batch_identity(batch,TASKS[0],0)
    batch.sample_id=['different']*len(batch.sample_id)
    with pytest.raises(OptimizationError,match='batch'):
        same(batch_identity(batch,TASKS[0],0),expected,'batch')


def test_live_rejects_jointly_tampered_terminal_rng(tmp_path, monkeypatch):
    """Rehashed, mutually consistent terminal RNG must fail actual replay."""
    import json
    import torch
    from p1d_optimization import OptimizationError
    from p1d_runtime import same, verify_live_gradients
    from p1d_smoke import SmokeAdapter, capture_rng, run_setting_smoke, verify_smoke
    from tests.test_p1d_tox import fixture_factory, TASKS, tox_step, file_sha

    factory = fixture_factory()
    updates = []
    make = factory.make_trainer

    def observed_factory(arm):
        trainer = make(arm)
        trainer.optimizer.register_step_post_hook(lambda *_: updates.append(arm))
        return trainer

    adapter = SmokeAdapter(observed_factory, tox_step, factory.batch)
    identity = {'fixture': 'micro_tox_terminal_rng', 'task_names': list(TASKS)}
    output = tmp_path / 'smoke'
    receipt = run_setting_smoke(adapter, setting='ToxAcute', task_names=TASKS,
                                output_dir=output, expected_identity=identity)
    assert receipt['observed_optimizer_updates'] == len(updates) == 11

    def forbidden_step(*_args, **_kwargs):
        pytest.fail('verification must not execute optimizer updates')

    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden_step)
    # Establish that the unchanged fixture passes the same live path first.
    assert verify_live_gradients(output, adapter, setting='ToxAcute',
                                 expected_identity=identity)['optimizer_updates'] == 0
    alternate = torch.Generator(device='cpu').manual_seed(888).get_state()
    for name in ('HF_low_5.pt', 'resumed_epoch5.pt'):
        path = output / name
        envelope = torch.load(path, map_location='cpu', weights_only=True)
        assert not torch.equal(envelope['payload']['after']['rng']['torch'], alternate)
        envelope['payload']['after']['rng']['torch'] = alternate.clone()
        torch.save(envelope, path)
        receipt['artifacts'][name] = file_sha(path)
    (output / 'receipt.json').write_text(json.dumps(receipt, allow_nan=False), encoding='utf-8')

    # Both final states and their SHA entries agree: offline consistency alone
    # cannot prove that these bytes were produced by the numerical path.
    assert verify_smoke(output, identity)['observed_optimizer_updates'] == 11
    rng_before = capture_rng()
    with pytest.raises(OptimizationError, match='live RNG'):
        verify_live_gradients(output, adapter, setting='ToxAcute', expected_identity=identity)
    same(capture_rng(), rng_before, 'failed live replay preserves caller RNG')
    assert len(updates) == 11


def test_prepare_tox_wires_scaler_snapshot_restore_and_live(tmp_path, monkeypatch):
    """Exercise production prepare; only asset construction is substituted."""
    from copy import deepcopy
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    import p1d_runtime as runtime
    import toxacute_datastore as store_module
    from p1d_optimization import OptimizationError
    from p1d_smoke import capture_rng, run_setting_smoke, verify_smoke
    from tests.test_p1d_tox import fixture_factory, TASKS

    factory = fixture_factory()
    store_root = tmp_path / 'fake_store'
    manifest_sha = 'a' * 64
    lock = dict(schema='p1d3_smoke_lock_v1', seed=42,
                formal_training_authorized=False, test_authorized=False,
                maximum_smoke_updates=33, smoke_updates_per_setting=11,
                inputs=dict(datastore_relative='fake_store', tox_manifest_sha256=manifest_sha),
                tox=deepcopy(factory.contract))
    boundary_calls = []

    def resolve(path):
        assert path == store_root
        boundary_calls.append('store')
        return SimpleNamespace(root=store_root)

    def fixed_hash(path):
        assert path == store_root / 'split_manifest.json'
        boundary_calls.append('manifest')
        return manifest_sha

    def factory_fixture(**kwargs):
        assert kwargs == dict(repo=tmp_path.resolve(), datastore=store_root,
                              contract=lock['tox'], device='cpu')
        boundary_calls.append('factory')
        return factory

    monkeypatch.setattr(store_module.ToxAcuteDataStore, 'resolve', resolve)
    monkeypatch.setattr(runtime, 'file_sha', fixed_hash)
    monkeypatch.setattr(runtime, 'ToxFactory', factory_fixture)
    updates, arms = [], []
    live_phase = False
    make = factory.make_trainer

    def observed_factory(arm):
        arms.append(arm)
        trainer = make(arm)
        trainer.optimizer.register_step_post_hook(lambda *_: updates.append(arm))
        # Force restore to do real work, rather than merely retaining a freshly
        # reconstructed identical scaler: poison resume and live factory state.
        if (arm == 'HF_low' and arms.count('HF_low') == 2) or live_phase:
            trainer.task_scalers[TASKS[0]]['mean'] += 7.0
        return trainer

    monkeypatch.setattr(factory, 'make_trainer', observed_factory)
    adapter, tasks, identity, tox = runtime.prepare(
        tmp_path, lock, setting='ToxAcute', split_manifest=tmp_path / 'unused_split',
        source_lock=tmp_path / 'unused_source', device='cpu')
    assert boundary_calls == ['store', 'manifest', 'factory']
    assert tox is factory and tasks == identity['task_names'] == list(TASKS)
    assert arms == [None] and updates == []
    assert callable(adapter.capture_extra) and callable(adapter.restore_extra)
    assert len(identity['batch_evidence']) == 8
    schedule = list(TASKS[:2]) + [TASKS[i % len(TASKS)] for i in range(6)]
    for i, task in enumerate(schedule):
        runtime.same(runtime.batch_identity(adapter.batch_getter(task, i), task, i),
                     identity['batch_evidence'][i], 'prepared batch identity')

    restore_calls = []
    production_restore = adapter.restore_extra

    def observed_restore(trainer, state):
        before = deepcopy(trainer.task_scalers)
        production_restore(trainer, state)
        runtime.same(adapter.capture_extra(trainer), {'scalers': identity['scalers']}, 'restored extras')
        restore_calls.append(before)

    adapter = replace(adapter, restore_extra=observed_restore)
    output = tmp_path / 'prepared_smoke'
    rng_before = capture_rng()
    receipt = run_setting_smoke(adapter, setting='ToxAcute', task_names=tasks,
                                output_dir=output, expected_identity=identity)
    runtime.same(capture_rng(), rng_before, 'smoke caller RNG')
    assert receipt['observed_optimizer_updates'] == len(updates) == 11
    assert arms == [None, None, 'B1_high', 'HF_low', 'HF_low']
    assert len(restore_calls) == 1
    assert restore_calls[0][TASKS[0]]['mean'] == identity['scalers'][TASKS[0]]['mean'] + 7.0
    snapshot = torch.load(output / 'epoch4_snapshot.pt', map_location='cpu', weights_only=True)
    runtime.same(snapshot['identity'], identity, 'snapshot identity')
    runtime.same(snapshot['payload']['state']['extra'], {'scalers': identity['scalers']}, 'saved scalers')
    for name in ('HF_low_5.pt', 'resumed_epoch5.pt'):
        row = torch.load(output / name, map_location='cpu', weights_only=True)['payload']
        for side in ('before', 'after'):
            runtime.same(row[side]['extra'], {'scalers': identity['scalers']}, 'epoch5 scalers')

    def forbidden_step(*_args, **_kwargs):
        pytest.fail('live verification must not execute optimizer updates')

    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden_step)
    live_phase = True
    assert verify_smoke(output, identity)['observed_optimizer_updates'] == 11
    live = runtime.verify_live_gradients(output, adapter, setting='ToxAcute', expected_identity=identity)
    assert live['optimizer_updates'] == 0 and len(live['checks']) == 11
    assert all(row['gradients_equal'] for row in live['checks'])
    assert len(updates) == 11 and len(restore_calls) == 12
    assert restore_calls[1][TASKS[0]]['mean'] == identity['scalers'][TASKS[0]]['mean'] + 7.0
    runtime.same(capture_rng(), rng_before, 'live caller RNG')

    invalid = {'scalers': deepcopy(identity['scalers'])}
    invalid['scalers'][TASKS[0]]['std'] += 1.0
    with pytest.raises(OptimizationError, match='restored scaler'):
        production_restore(SimpleNamespace(task_scalers={}), invalid)
