from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import pytest

import d2_control as core
import d2_screen as mod


def test_gate_rejects_small_report_even_with_correct_class_names(tmp_path):
    root = ET.Element('testsuite')
    for name in mod.TESTS:
        ET.SubElement(root, 'testcase', classname='tests.'+Path(name).stem, name='fabricated')
    ET.ElementTree(root).write(tmp_path/'tests.xml')
    mod.write(tmp_path/'gate.json', dict(task=mod.TASK, commit='a'*40, role='wsl', tests=list(mod.TESTS),
              junit_sha256=mod.sha(tmp_path/'tests.xml'), exit_code=0, status='PASS', test_count=len(mod.TESTS)))
    with pytest.raises(ValueError, match='case count'):
        mod.gate_check(tmp_path, 'a'*40, 'wsl')


def fake_results():
    return [dict(job=j, identity=dict(initial_heads='h', initial_encoder='r' if j['condition'] in ('RT', 'RJ') else 'p'),
                 exposure=dict(target=dict(task=8)), schedule_sha256='s', selected=dict(metrics=dict(macro_rmse=1.)))
            for j in core.jobs()]


@pytest.mark.parametrize('damage', ['heads', 'encoder', 'schedule', 'exposure', 'missing'])
def test_pairing_rejects_mismatched_initialization_or_target_stream(damage):
    results = fake_results()
    mod.pairing_check(results)
    if damage == 'heads':
        results[0]['identity']['initial_heads'] = 'other'
    elif damage == 'encoder':
        results[0]['identity']['initial_encoder'] = 'other'
    elif damage == 'schedule':
        results[0]['schedule_sha256'] = 'other'
    elif damage == 'exposure':
        results[0]['exposure']['target']['task'] = 10
    else:
        results.pop()
    with pytest.raises(ValueError):
        mod.pairing_check(results)


def test_dispatch_stops_on_failure_preserves_partial_and_cannot_retry(tmp_path, monkeypatch):
    import p1d4_batch
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    monkeypatch.setattr(mod, 'check_code', lambda *a: None)
    monkeypatch.setattr(mod, 'gate_check', lambda *a: None)
    monkeypatch.setattr(mod, 'preflight_inputs', lambda *a: [])
    monkeypatch.setattr(p1d4_batch, 'free_gpus', lambda *a: [0])
    monkeypatch.setattr(mod.time, 'sleep', lambda *a: None)
    gate = tmp_path/'gate'
    gate.mkdir()
    for name in ('gate.json', 'tests.xml', 'tests.log'):
        (gate/name).write_text('{}')
    args = SimpleNamespace(commit='a'*40, output=tmp_path/'result', wsl_gate=gate, server_gate=gate,
                           gpus='0', source_lock=gate/'gate.json', split_manifest=gate/'gate.json')
    seen = []
    def fail(job, *args):
        seen.append(job)
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(mod, 'execute_child', fail)
    with pytest.raises(ValueError, match='partial failure'):
        mod.run(args)
    assert len(seen) == 1
    assert mod.read(args.output/'completion.json')['status'] == 'FAILED_PARTIAL'
    args.output = tmp_path/'different_result'
    with pytest.raises(ValueError, match='already claimed'):
        mod.run(args)


def test_package_keeps_tensor_files_on_server_and_checks_transport(tmp_path):
    root = tmp_path/'run'
    root.mkdir()
    mod.write(root/'completion.json', dict(status='FAILED_PARTIAL'))
    (root/'best.pt').write_bytes(b'remains on server')
    out = tmp_path/'facts.zip'
    result = mod.package(root, out)
    with zipfile.ZipFile(out) as z:
        assert 'best.pt' not in z.namelist() and set(z.namelist()) == {'completion.json', 'transport_manifest.json'}
    assert result['sha256'] == mod.sha(out) and (root/'best.pt').read_bytes() == b'remains on server'
    with pytest.raises(ValueError):
        mod.package(root, out)


def test_worker_command_changes_to_new_cpu_verification_process(tmp_path, monkeypatch):
    import v9_cost_probe
    monkeypatch.setattr(v9_cost_probe, 'gpu_uuid', lambda g: 'GPU-test-fixture')
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    for folder in ('runs', 'commands', 'logs'):
        (tmp_path/folder).mkdir()
    calls = []
    job = core.jobs()[0]
    def child(argv, **kw):
        calls.append((argv.copy(), kw['env'].copy()))
        if len(calls) == 1:
            output = tmp_path/'runs'/job['id']
            output.mkdir()
            mod.write(output/'training.json', dict(job=job))
        else:
            output = tmp_path/'runs'/job['id']
            mod.write(output/'verification.json', dict(status='CONTENT_PASS', job=job,
                        training_sha256=mod.sha(output/'training.json')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(mod.subprocess, 'run', child)
    mod.execute_child(job, tmp_path, 'a'*40, 0, Path('split.json'), Path('source.json'))
    assert [a[0][3] for a in calls] == ['worker', 'verify-worker']
    assert calls[0][1]['CUDA_VISIBLE_DEVICES'] == 'GPU-test-fixture'
    assert calls[1][1]['CUDA_VISIBLE_DEVICES'] == ''
