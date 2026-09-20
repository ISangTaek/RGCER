import copy,hashlib,json,math
from pathlib import Path
import pytest
import torch
import p1d5_test as m
from dataset115_smoke import state_digest

def rows(task='t',split='validation'):
    return [dict(task=task,sample_id='id'+str(i),split=split,label=float(i),prediction=float(i)+.2) for i in range(2)]

def lock_fixture():
    lock=dict(schema='p1d5_test_v1',task_id=m.TASK,training_performed=False,validation_atol=1e-6,validation_rtol=1e-6,rows=[])
    for setting in ('ToxAcute','A','B'):
        for family in ('B1','HF'):
            arm='B1_high' if setting=='B' and family=='B1' else family+'_low'
            for seed in range(42,47):
                r=dict(setting=setting,family=family,arm=arm,seed=seed,best_epoch=3,run_id=f'{setting}_{arm}_s{seed}',
                    action='REUSE' if setting=='B' and family=='B1' else 'EXPORT',checkpoint={'sha256':'0'*64},model_digest='1'*64,
                    validation_rows=rows(),reused_test_rows=rows(split='test'))
                lock['rows'].append(r)
    return lock

class FakeData:
    def population(self,setting,split):return [{k:v for k,v in r.items() if k!='prediction'} for r in rows(split=split)]
    def tasks(self,setting):return ['t']

@pytest.mark.parametrize('change',[lambda r:r.pop(),lambda r:r.append(r[0]),lambda r:r[0].update(arm='HF_high'),lambda r:r[0].update(seed=True),lambda r:r[0].update(best_epoch=True),lambda r:r[0].update(action='REUSE')])
def test_matrix_rejects_change(change):
    lock=lock_fixture();change(lock['rows'])
    with pytest.raises((ValueError,AssertionError,RuntimeError)):m.validate_lock(lock)

@pytest.mark.parametrize('change',[lambda r:r.pop(),lambda r:r.append(r[0]),lambda r:r[0].update(label=9.),lambda r:r[0].update(prediction=float('nan')),lambda r:r[0].update(prediction=True),lambda r:r[0].update(split='train'),lambda r:r[0].update(task='wrong'),lambda r:r[0].update(sample_id='foreign:id0'),lambda r:r[0].update(lower=3.,upper=1.)])
def test_metrics_rejects_bad_records(change):
    r=rows();change(r)
    with pytest.raises((ValueError,AssertionError,RuntimeError)):m.metrics(r,FakeData().population('A','validation'),['t'],'validation')

def test_metrics_correct_and_r2_null():
    r=rows();v=m.metrics(r,FakeData().population('A','validation'),['t'],'validation')
    assert v['macro_rmse']==pytest.approx(.2) and v['endpoints']['t']['r2']==pytest.approx(.84)
    r[1]['label']=0.;v=m.metrics(r,r,['t'],'validation');assert v['endpoints']['t']['r2'] is None

def test_all_validation_precedes_test_and_roundtrip(tmp_path):
    lock=lock_fixture();calls=[]
    def pred(state,row,split):calls.append((row['run_id'],split));return rows(split=split)
    out=tmp_path/'out';m.execute(lock,FakeData(),out,lambda row:None,pred)
    assert [s for _,s in calls]==['validation']*25+['test']*25
    assert m.verify(lock,FakeData(),out)['checked_runs']==30
    with pytest.raises((ValueError,RuntimeError)):m.execute(lock,FakeData(),out,lambda row:None,pred)

def test_last_validation_failure_never_accesses_test(tmp_path):
    lock=lock_fixture();calls=[]
    def pred(state,row,split):
        calls.append(split);r=rows(split=split)
        if row is lock['rows'][-1]:r[0]['prediction']=50.
        return r
    with pytest.raises((ValueError,RuntimeError)):m.execute(lock,FakeData(),tmp_path/'out',lambda row:None,pred)
    assert 'test' not in calls

@pytest.mark.parametrize('field,value',[('seed',True),('arm','HF_high'),('checkpoint_sha256','x'),('prediction_file','../evil'),('metrics',{})])
def test_saved_report_tamper(tmp_path,field,value):
    lock=lock_fixture();out=tmp_path/'out'
    m.execute(lock,FakeData(),out,lambda row:None,lambda state,row,split:rows(split=split))
    p=out/'test_summary.json';d=json.loads(p.read_bytes());d['runs'][0][field]=value;p.write_text(json.dumps(d))
    with pytest.raises((ValueError,RuntimeError,KeyError)):m.verify(lock,FakeData(),out)

def test_load_state_correct_and_wrong_epoch(tmp_path):
    state={'x':torch.tensor([1.])};identity={'seed':42};p=tmp_path/'best.pt'
    torch.save(dict(epoch=3,identity=identity,model_state=state),p)
    row=dict(checkpoint={'sha256':m.file_sha(p)},checkpoint_kind='route',best_epoch=3,setting='A',model_digest=state_digest(state),identity=identity)
    assert torch.equal(m.load_state(p,row)['x'],state['x'])
    row['best_epoch']=2
    with pytest.raises((ValueError,RuntimeError)):m.load_state(p,row)

def test_selected_path_rejects_escape(tmp_path):
    (tmp_path/'outside.pt').write_bytes(b'a');inner=tmp_path/'inner';inner.mkdir()
    with pytest.raises((ValueError,RuntimeError)):m.selected_path({'checkpoint':{'canonical_member':'../outside.pt'}},inner,inner)

def test_readonly_module_has_no_training_calls():
    import ast
    tree=ast.parse(Path(m.__file__).read_text(encoding='utf8'))
    prohibited={'step','backward','fit','train','make_trainer','controlled_trainer','_fit_task_scalers'}
    assert not [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in prohibited]

def test_lock_bytes_cannot_be_replaced(tmp_path):
    p=tmp_path/'lock.json';p.write_text(json.dumps(lock_fixture()))
    with pytest.raises((ValueError,RuntimeError)):m.load_lock(p)

def cli_module():
    import importlib.util
    p=Path(__file__).resolve().parents[1]/'scripts/run_p1d5_test.py'
    spec=importlib.util.spec_from_file_location('p1d5_cli_test',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_snapshot_rejects_weights(tmp_path):
    (tmp_path/'unexpected.pt').write_bytes(b'x')
    with pytest.raises((ValueError,RuntimeError)):cli_module().snapshot(tmp_path)

def test_gate_rejects_wrong_role(tmp_path):
    (tmp_path/'preflight.json').write_text(json.dumps(dict(role='server',commit='a'*40,exit_code=0)))
    with pytest.raises((ValueError,RuntimeError)):cli_module().gate_check(tmp_path,'a'*40,'wsl')

def test_package_changed_evidence_rejected(tmp_path):
    from types import SimpleNamespace
    module=cli_module();(tmp_path/'payload.json').write_text('{}')
    (tmp_path/'verification.json').write_text(json.dumps(dict(commit='a'*40,content_status='PASS',checked_runs=30,files={})))
    with pytest.raises((ValueError,RuntimeError)):module.archive(SimpleNamespace(output=tmp_path,commit='a'*40))

def test_load_wrong_sha_and_identity(tmp_path):
    state={'x':torch.tensor([1.])};p=tmp_path/'best.pt';torch.save(dict(epoch=3,identity={'seed':42},model_state=state),p)
    row=dict(checkpoint={'sha256':'0'*64},checkpoint_kind='route',best_epoch=3,setting='A',model_digest=state_digest(state),identity={'seed':43})
    with pytest.raises((ValueError,RuntimeError)):m.load_state(p,row)
    row['checkpoint']['sha256']=m.file_sha(p)
    with pytest.raises((ValueError,RuntimeError)):m.load_state(p,row)
