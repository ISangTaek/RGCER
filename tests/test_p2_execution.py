from copy import deepcopy
import hashlib
import json
import pytest
from p2_contract import P2Error, PROTOCOL_SHA, MEMBERS_SHA
from p2_execution import write_new, smoke_matrix, audit_validation, load_release, APPROVAL_SHA


def release():
    return dict(schema='p2_release_v1',stage='SMOKE',commit='a'*40,protocol_sha256=PROTOCOL_SHA,
        members_sha256=MEMBERS_SHA,approval_sha256=APPROVAL_SHA,updates_max=112,runs=smoke_matrix(),
        test_authorized=False,calibration_authorized=False,retry_authorized=False)


def test_matrix():
    rows=smoke_matrix()
    assert len(rows)==7 and sum(r['updates'] for r in rows)==112
    assert len({r['run_id'] for r in rows})==7
    assert all(r['seed']==42 for r in rows)


@pytest.mark.parametrize('mutation',['nan','member','group','label','split','macro','mae','count','best','epoch'])
def test_independent_audit_negative(mutation):
    from tests.test_p2_engine import engine
    e=engine('HF_low')
    for _ in range(6):e.step()
    h=deepcopy(e.history);best=e.best_epoch
    assert audit_validation(h,e.view,best)['status']=='PASS'
    row=h[0]['validation_rows'][0]
    if mutation=='nan':row['prediction']=float('nan')
    elif mutation=='member':row['sample_id']='wrong'
    elif mutation=='group':row['group']='wrong'
    elif mutation=='label':row['label']+=1
    elif mutation=='split':row['split']='test'
    elif mutation=='macro':h[0]['validation']['macro_rmse']+=1
    elif mutation=='mae':h[0]['validation']['endpoints']['a']['mae']+=1
    elif mutation=='count':h[0]['validation']['endpoints']['a']['n']=True
    elif mutation=='best':best=True
    elif mutation=='epoch':h[0]['epoch']=True
    with pytest.raises(P2Error):audit_validation(h,e.view,best)


def test_p2_input_dropout_matches_protocol():
    from p2_model import build_model
    m=build_model('source',42)
    assert m.encoder.backbone.input_dropout.p==0.
    assert all(layer.attention.dropout.p==.1 for layer in m.encoder.backbone.layers)


@pytest.mark.parametrize('field,value',[('stage','FORMAL'),('updates_max',113),('test_authorized',True),('retry_authorized',0),('commit','short'),('members_sha256','b'*64),('runs',[])])
def test_release_negative(tmp_path,monkeypatch,field,value):
    import p2_execution
    original=p2_execution.bound_json
    monkeypatch.setattr(p2_execution,'bound_json',lambda p,s: {'budget_status':'APPROVED'} if p=='approval' else original(p,s))
    r=release();path=tmp_path/'good.json';sha=write_new(path,r)
    assert load_release(path,sha,'approval')==r
    r[field]=value;bad=tmp_path/'bad.json';sha=write_new(bad,r)
    with pytest.raises(P2Error):load_release(bad,sha,'approval')
