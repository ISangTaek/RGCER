"""Asset-free CPU tests. Timings in fixtures are fabricated and never results."""
from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

import v9_cost_probe as mod
from tests.test_p1d_routes import setup_factory
from tests.test_p1d_tox import fixture_factory


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fake_rows(setting):
    tasks, counts = mod.tasks_and_counts(setting)
    return [dict(r, n=len(r['indices']), sample_ids_sha256='a'*64,
                 max_nodes=5, real_nodes=3*len(r['indices']),
                 batch_seconds=.02, update_seconds=.1, loss=.3)
            for r in mod.schedule(tasks, counts, mod.BATCH_SIZES[setting])]


@pytest.mark.parametrize('setting', mod.SETTINGS)
def test_real_identity_counts_and_batch_sizes_not_two_record_smoke(setting):
    tasks, counts = mod.tasks_and_counts(setting)
    rows = mod.schedule(tasks, counts, mod.BATCH_SIZES[setting])
    assert len(rows) == 12
    assert {r['task'] for r in rows if r['measured']} == set(tasks)
    assert max(len(r['indices']) for r in rows) == mod.BATCH_SIZES[setting]
    for task in tasks:
        own = [r['indices'] for r in rows if r['task'] == task]
        if counts[task] > mod.BATCH_SIZES[setting]:
            assert not set(own[0]) & set(own[1])
    report = mod.summarize(fake_rows(setting), tasks, counts, mod.BATCH_SIZES[setting])
    expected_steps = sum((n + mod.BATCH_SIZES[setting]-1) // mod.BATCH_SIZES[setting] for n in counts.values())
    assert report['train_epoch_seconds_rough'] == pytest.approx(expected_steps * .12)
    assert report['authorizes_formal_training'] is False
    assert report['is_upper_bound'] is False
    assert 'candidate_overhead' in report['excludes']


@pytest.mark.parametrize('change', ['missing', 'wrong_indices', 'bool_measured', 'nan', 'negative_time', 'dropped', 'unknown'])
def test_bad_timing_or_batch_evidence_cannot_pass(change):
    tasks, counts = mod.tasks_and_counts('A')
    rows = fake_rows('A')
    if change == 'missing': rows.pop()
    if change == 'wrong_indices': rows[4]['indices'][0] = 99999
    if change == 'bool_measured': rows[4]['measured'] = 1
    if change == 'nan': rows[4]['loss'] = float('nan')
    if change == 'negative_time': rows[4]['batch_seconds'] = -1
    if change == 'dropped': rows[4]['n'] -= 1
    if change == 'unknown': rows[4]['unknown'] = True
    with pytest.raises(ValueError):
        mod.summarize(rows, tasks, counts, 32)


@pytest.mark.parametrize('setting', mod.SETTINGS)
@pytest.mark.parametrize('mode', mod.MODES)
def test_real_graphormer_and_original_loss_update_scope(setting, mode, setup_factory):
    from reproducibility import state_dict_sha256
    from loss import QuantileRegressionLoss
    if setting == 'ToxAcute':
        factory = fixture_factory()
        trainer = factory.make_trainer()
        task = next(iter(factory.datasets['train']))
        batch = factory.collator([factory.datasets['train'][task][0], factory.datasets['train'][task][1]])
    else:
        factory = setup_factory[0](setting)[0]
        trainer = factory.make_trainer(original=True)
        task = next(iter(trainer.datasets['train']))
        batch = trainer._batch('train', task, list(range(len(trainer.datasets['train'][task]))))
    initial_encoder = state_dict_sha256(trainer.model.encoder)
    initial_heads = state_dict_sha256(trainer.model.decoders)
    optimizer = mod.configure_model(trainer.model, mode)
    optimizer.zero_grad(set_to_none=True)
    if setting == 'ToxAcute':
        loss, _ = trainer._training_step(batch, task, 0)
    else:
        scaler = trainer.scalers[task]
        loss = QuantileRegressionLoss().compute_loss(trainer.model(batch, task_name=task)[task],
                    (batch.y.reshape(-1, 1)-scaler['mean'])/scaler['std'])
    mod.checked_step(trainer.model, optimizer, loss, mode)
    assert (initial_encoder == state_dict_sha256(trainer.model.encoder)) == (mode == 'frozen')
    assert initial_heads != state_dict_sha256(trainer.model.decoders)
    if mode == 'frozen':
        assert all(p.grad is None and p not in optimizer.state for p in trainer.model.encoder.parameters())
    assert trainer.model.encoder.training
    assert str(next(trainer.model.parameters()).device) == 'cpu'


def test_failed_or_changed_output_does_not_allow_another_attempt(tmp_path):
    mod.claim(tmp_path, tmp_path/'first', 'a'*40)
    with pytest.raises(FileExistsError):
        mod.claim(tmp_path, tmp_path/'second', 'a'*40)


def test_schema_and_scope_reject_nonmatrix_jobs():
    assert len(mod.jobs()) == 6
    for job in mod.jobs(): mod.check_job(job)
    for key, value in [('seed', 43), ('mode', 'SRGT'), ('extra', True), ('setting', 'Human6')]:
        bad = dict(mod.jobs()[0], **{key: value})
        with pytest.raises(ValueError): mod.check_job(bad)


def suite_fixture(root):
    root.mkdir()
    mod.write(root/'launch.json', dict(task=mod.TASK, commit='a'*40, jobs=mod.jobs()))
    for job in mod.jobs():
        out = root/job['id']; out.mkdir()
        tasks, counts = mod.tasks_and_counts(job['setting'])
        rows = fake_rows(job['setting'])
        for row in rows:
            i = row['step']
            mod.write(out/f'step_{i:02d}.json', row)
            mod.write(out/f'update_{i:02d}.intent.json', dict(step=i, max_total=i+1))
        r = dict(schema='v9_cost_job_v1', task=mod.TASK, job=job, commit='a'*40,
                 contract_sha256=mod.digest(mod.contract(job['setting'])), counts=counts, rows=rows,
                 summary=mod.summarize(rows, tasks, counts, mod.BATCH_SIZES[job['setting']]),
                 initial_encoder='1'*64, final_encoder=('1' if job['mode']=='frozen' else '2')*64,
                 peak_allocated_bytes=1000, peak_reserved_bytes=2000,
                 scope='TRAIN_COST_ONLY_NO_CANDIDATES', acceptance='PENDING_REVIEW',
                 holdout_predictions_accessed=False, validation_evaluated=False, scientific_pass=False)
        mod.write(out/'receipt.json', r)
    return root


def test_readonly_verification_recomputes_all_six_jobs(tmp_path):
    root = suite_fixture(tmp_path/'suite')
    result = mod.verify(root, 'a'*40)
    assert result['optimizer_updates'] == 72
    assert result['completed_jobs'] == 6
    assert result['formal_training_authorized'] is False
    assert result['scientific_acceptance'] == 'NOT_ASSESSED'


@pytest.mark.parametrize('change', ['commit', 'contract', 'summary', 'holdout', 'encoder', 'extra_step', 'missing_job'])
def test_verifier_detects_tampering(tmp_path, change):
    root = suite_fixture(tmp_path/'suite')
    path = root/mod.jobs()[0]['id']/'receipt.json'
    r = mod.read(path)
    if change == 'commit': r['commit'] = 'b'*40
    if change == 'contract': r['contract_sha256'] = '0'*64
    if change == 'summary': r['summary']['train_only_40_epoch_hours_rough'] = 0
    if change == 'holdout': r['holdout_predictions_accessed'] = True
    if change == 'encoder': r['final_encoder'] = '2'*64
    if change == 'extra_step': mod.write(path.parent/'step_12.json', r['rows'][0])
    if change == 'missing_job': path.unlink()
    else: path.write_text(json.dumps(r), encoding='utf8')
    with pytest.raises((ValueError, FileNotFoundError)):
        mod.verify(root, 'a'*40)


@pytest.mark.parametrize('failure', ['exit', 'timeout'])
def test_supervisor_stops_immediately_without_retry(tmp_path, monkeypatch, failure):
    import p1d4_batch
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    monkeypatch.setattr(mod, 'check_code', lambda *a: None)
    monkeypatch.setattr(mod, 'check_gate', lambda *a: None)
    monkeypatch.setattr(mod, 'gpu_uuid', lambda *a: 'GPU-fixed-for-test')
    monkeypatch.setattr(mod, 'contract', lambda *a: {})
    monkeypatch.setattr(p1d4_batch, 'free_gpus', lambda x: x)
    wsl = tmp_path/'wsl'; wsl.mkdir()
    server = tmp_path/'server'; server.mkdir()
    calls = []
    def child(argv, **kw):
        calls.append(argv)
        assert 0 < kw['timeout'] <= mod.WALL_SECONDS
        assert kw['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-fixed-for-test'
        if failure == 'timeout': raise subprocess.TimeoutExpired(argv, 1)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(subprocess, 'run', child)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        mod.run(tmp_path/'out', 'a'*40, 0, tmp_path/'split', tmp_path/'source', wsl, server)
    assert len(calls) == 1
    assert (tmp_path/'out'/'failed.json').is_file()
    assert (tmp_path/mod.REGISTRY/'attempt.json').is_file()


def test_json_rejects_duplicate_and_nonfinite_fields(tmp_path):
    path = tmp_path/'bad.json'
    for value in ['{"x":1,"x":2}', '{"x":NaN}']:
        path.write_text(value, encoding='utf8')
        with pytest.raises(ValueError): mod.read(path)


def gate_fixture(path, role='wsl'):
    path.mkdir()
    cases = ''.join(f'<testcase classname="tests.{t.rsplit("/",1)[-1][:-3]}" name="synthetic"/>' for t in mod.TEST_FILES)
    (path/'tests.xml').write_text('<testsuites><testsuite>'+cases+'</testsuite></testsuites>', encoding='utf8')
    mod.write(path/'gate.json', dict(task=mod.TASK, role=role, commit='a'*40, tests=list(mod.TEST_FILES),
                                   junit_sha256=mod.sha(path/'tests.xml'), real_data_optimizer_updates=0))
    return path


def test_wsl_and_server_receipts_not_interchangeable(tmp_path):
    root = gate_fixture(tmp_path/'wsl')
    assert mod.check_gate(root, 'a'*40, 'wsl')['role'] == 'wsl'
    with pytest.raises(ValueError): mod.check_gate(root, 'a'*40, 'server')
    with pytest.raises(ValueError): mod.check_gate(root, 'b'*40, 'wsl')


def test_package_partial_failure_and_never_overwrite(tmp_path):
    root = tmp_path/'failed'; root.mkdir()
    mod.write(root/'launch.json', dict(task=mod.TASK, commit='a'*40, jobs=mod.jobs()))
    mod.write(root/'failed.json', dict(error_type='TimeoutExpired'))
    report = mod.package(root)
    assert report['scientific_acceptance'] == 'NOT_ASSESSED'
    assert report['sha256'] == mod.sha(tmp_path/'failed.zip')
    with pytest.raises(ValueError): mod.package(root)


def test_command_wrapper_keeps_nonzero_exit_and_prevents_duplicate_launch(tmp_path):
    import sys
    from scripts.v9_cost_capture import capture
    prefix = tmp_path/'capture'
    command = [sys.executable, '-c', 'import sys; print("cpu fixture"); sys.exit(7)']
    assert capture(prefix, command) == 7
    assert mod.read(tmp_path/'capture.exit.json')['exit_code'] == 7
    assert mod.read(tmp_path/'capture.started.json')['command'] == command
    assert (tmp_path/'capture.stdout.log').read_text().strip() == 'cpu fixture'
    with pytest.raises(FileExistsError): capture(prefix, command)
