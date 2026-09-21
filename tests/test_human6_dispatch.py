"""Dispatcher checks without private assets, GPU or launching experiments."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

spec=importlib.util.spec_from_file_location('h6cli',Path(__file__).parents[1]/'scripts/run_human6.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def test_exact_matrix():
    assert len(cli.jobs('smoke'))==3 and {s for _,s in cli.jobs('smoke')}=={42}
    assert len(cli.jobs('formal'))==15 and len(set(cli.jobs('formal')))==15
    with pytest.raises(ValueError):cli.jobs('test')


def test_no_repeat_phase(tmp_path):
    (tmp_path/'smoke').mkdir()
    args=SimpleNamespace(command='smoke',gpus=[0],work=str(tmp_path))
    with pytest.raises(ValueError,match='already started'):cli.batch(args,cli.h.policy())


@pytest.mark.parametrize('memory,pids', [('1024\n',''), ('128\n','1234\n')])
def test_occupied_gpu_rejected(monkeypatch,memory,pids):
    def run(args,**kwargs):
        return SimpleNamespace(stdout=memory if '--query-gpu=memory.used' in args else pids)
    monkeypatch.setattr(cli.subprocess,'run',run)
    with pytest.raises(ValueError,match='GPU'):cli.gpu_free(0)


def test_content_not_self_report_pass(tmp_path):
    root=tmp_path/'formal';root.mkdir()
    for m,s in cli.jobs('formal'):(root/cli.name(m,s)).mkdir()
    first=root/'FROZEN_s42'
    (first/'result.json').write_text(json.dumps({'validation_status':'PASS'}))
    with pytest.raises((KeyError,FileNotFoundError,ValueError)):
        cli.verify(tmp_path,'formal',{},cli.h.policy())


def test_package_missing_evidence_is_not_success(tmp_path):
    with pytest.raises(ValueError,match='verify formal first'):cli.package(tmp_path,cli.h.policy())


def test_failure_stops_new_jobs_without_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'inputs',lambda *a: {})
    monkeypatch.setattr(cli.h,'source_asset',lambda *a,**k: (None,None))
    monkeypatch.setattr(cli,'gpu_free',lambda g: None)
    monkeypatch.setattr(cli,'verify',lambda *a,**k: {})
    calls=[]
    def failed(cmd,**kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(cli.subprocess,'run',failed)
    args=SimpleNamespace(command='formal',gpus=[0],work=str(tmp_path),split_manifest=str(tmp_path/'split.json'),source_root=None)
    with pytest.raises(ValueError,match='no automatic retry'):
        cli.batch(args,cli.h.policy())
    assert len(calls)==1
    result=json.loads((tmp_path/'formal_commands.json').read_text())
    assert result['commands'][0]['exit_code']==2 and len(result['not_started'])==14
