import json
import numpy as np
import pytest
from types import SimpleNamespace
from baselines import b115_test_export as m
from dataset115_adapter import LabelView
from dataset115_contract import PRIMARY


def view(route='A',split='test'):
    return LabelView(route,'target',split,PRIMARY,('dataset115:row_0','dataset115:row_1'),
        ('C','CC'),('C','CC'),('g1','g2'),np.arange(10,dtype=float).reshape(2,5),'a'*64)


def test_frozen_policy_current_engine_and_ten_selected():
    p=m.load_policy()
    assert len(p['jobs'])==10 and p['selected']=={'A':2,'B':0}
    assert {(j['route'],j['seed']) for j in p['jobs']}=={(r,s) for r in ('A','B') for s in range(42,47)}
    assert all(j['trial']==p['selected'][j['route']] for j in p['jobs'])


@pytest.mark.parametrize('kind',['policy','implementation'])
def test_policy_rejects_changes(monkeypatch,kind):
    if kind=='policy':monkeypatch.setattr(m,'POLICY_SHA','0'*64)
    else:monkeypatch.setattr(m,'implementation_identity',lambda:{})
    with pytest.raises(ValueError):m.load_policy()


def test_assets_rehash_each_use(tmp_path):
    (tmp_path/'campaign.json').write_text('{}')
    d=tmp_path/'job';d.mkdir();f=d/'weights.pt';f.write_bytes(b'original')
    p=dict(campaign_sha256=m.sha(tmp_path/'campaign.json'),jobs=[dict(job_id='job',files={'weights.pt':m.sha(f)})])
    m.check_assets(tmp_path,p)
    f.write_bytes(b'changed')
    with pytest.raises(ValueError):m.check_assets(tmp_path,p)


def test_input_rejects_before_dataset_read(tmp_path,monkeypatch):
    p=tmp_path/'input';p.write_bytes(b'original')
    policy={'inputs':{'csv_path':m.sha(p)}};p.write_bytes(b'changed')
    monkeypatch.setattr(m.Dataset115Table,'load',lambda *a,**k:pytest.fail('must not read changed inputs'))
    with pytest.raises(ValueError):m.load_table(policy,{'csv_path':str(p)})


@pytest.mark.parametrize('field',['sample_ids','canonical','groups','tasks','truth','prediction','extra'])
def test_export_content_tamper(tmp_path,field):
    v=view();prediction=v.labels+.1;data=m.arrays(v,prediction)
    if field in ('prediction','truth'):data[field]=data[field]+100
    elif field=='extra':data[field]=np.zeros(1)
    else:data[field]=data[field][::-1]
    file=tmp_path/'result.npz';np.savez(file,**data)
    with pytest.raises((ValueError,AssertionError)):m.check_export(file,v,prediction)


def test_export_replay_tolerance_preserves_stored_metrics(tmp_path):
    v=view();p=v.labels+.1;file=tmp_path/'p.npz';np.savez(file,**m.arrays(v,p))
    assert m.check_export(file,v,p+1e-7)==m.metrics(v.labels,p)


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_prediction_rejected(bad):
    v=view();p=v.labels.copy();p[0,0]=bad
    with pytest.raises(ValueError):m.metrics(v.labels,p)


@pytest.fixture
def flow(tmp_path,monkeypatch):
    p=dict(task_id='synthetic',training_commit='a'*40,jobs=[dict(job_id=f'{r}_{s}',route=r,seed=s,
        best_epoch=0,files={'training/epoch_000.pt':'b'*64}) for r in ('A','B') for s in range(42,47)])
    monkeypatch.setattr(m,'load_policy',lambda:p)
    monkeypatch.setattr(m,'check_assets',lambda *args:None)
    monkeypatch.setattr(m,'load_table',lambda *args:SimpleNamespace())
    monkeypatch.setattr(m,'preflight',lambda *args:({j['job_id']:(None,None) for j in p['jobs']},[]))
    monkeypatch.setattr(m,'active_view',lambda table,route,split:view(route,split))
    monkeypatch.setattr(m,'avalon_matrix',lambda smiles:np.zeros((len(smiles),1024)))
    monkeypatch.setattr(m,'predict',lambda *args:view().labels+.1)
    return tmp_path/'out'


def test_full_synthetic_export_verify_package(flow):
    result=m.run('test','unused',flow,{})
    (flow/'execution.json').write_text('{}')
    assert len(result['runs'])==10
    assert m.run('verify','unused',flow,{})==result
    archive=m.package(flow)
    assert m.sha(archive)==open(archive+'.sha256').read().split()[0]
    with pytest.raises(ValueError):m.run('test','unused',flow,{})


@pytest.mark.parametrize('change',['extra','summary','prediction'])
def test_verify_rejects_changed_delivery(flow,change):
    m.run('test','unused',flow,{});(flow/'execution.json').write_text('{}')
    if change=='extra':(flow/'extra.npz').write_bytes(b'bad')
    elif change=='summary':(flow/'test_summary.json').write_text('{}')
    else:np.savez(flow/'A_42.npz',**m.arrays(view(),view().labels+100))
    with pytest.raises((ValueError,AssertionError)):m.run('verify','unused',flow,{})


def test_no_test_view_when_validation_fails(flow,monkeypatch):
    def fail(*args):raise ValueError('validation failed')
    monkeypatch.setattr(m,'preflight',fail)
    monkeypatch.setattr(m,'active_view',lambda *args:pytest.fail('test touched'))
    with pytest.raises(ValueError):m.run('test','unused',flow,{})
    assert not flow.exists()
