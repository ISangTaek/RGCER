"""Synthetic verifier tests: no external data, CUDA or scientific acceptance."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
from dataset115_contract import PRIMARY, semantic_digest
from baselines.b115_training import implementation_identity, predict, validation_metrics
from baselines.features import avalon_matrix
from baselines.models.toxacol import endpoint_feature_matrix, ToxACoLNet
from baselines.scaling import TaskScaler
import scripts.verify_b115_smoke as verifier


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    tasks = tuple(ANIMAL_SOURCE_TASKS)+PRIMARY
    endpoint,vocab = endpoint_feature_matrix(tasks,extend_vocabulary=True)
    scaler = TaskScaler.fit(np.arange(122,dtype=float).reshape(2,61),tasks,allow_empty=False)
    matrix=np.eye(61,dtype=np.float32)
    pairs=[(i,j) for i in range(56) for j in range(i+1,56)][:320]
    for i,j in pairs: matrix[i,j]=matrix[j,i]=1
    degree=matrix.sum(axis=1)
    adj=matrix/np.sqrt(degree[:,None]*degree[None,:])
    ids=['dataset115:row_1','dataset115:row_2']
    labels=np.arange(10,dtype=float).reshape(2,5)
    view=SimpleNamespace(sample_ids=tuple(ids),smiles=('C','CC'),labels=labels)
    monkeypatch.setattr(verifier.Dataset115Table,'load',lambda *a,**kw: SimpleNamespace(view=lambda *a:view))
    files=[]
    for name in ('csv','split','tox'):
        p=tmp_path/name
        p.write_text('{"records":[]}',encoding='utf8')
        files.append(p)
    audit=dict(input=[dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
        data_identity='a'*64,B=dict(tasks=list(tasks),scaler={t:dict(n=int(scaler.counts[j]),
          mean=float(scaler.means[j]),std=float(scaler.stds[j])) for j,t in enumerate(tasks)}))
    ap=tmp_path/'audit.json'; ap.write_text(json.dumps(audit),encoding='utf8')
    sha=hashlib.sha256(ap.read_bytes()).hexdigest()
    monkeypatch.setattr(verifier,'AUDIT_SHA',sha)
    contract=dict(tasks=list(tasks),input_identity='a'*64,vocabulary=vocab,audit_sha256=sha,
        rows=55670,observations=78324,scaler=scaler.to_dict(),adjacency=adj.tolist(),
        validation_ids_sha256=semantic_digest(ids),validation_labels_sha256=semantic_digest(labels.tolist()))
    out=tmp_path/'out'; out.mkdir()
    (out/'data_contract.json').write_text(json.dumps(contract),encoding='utf8')
    torch.manual_seed(7)
    model=ToxACoLNet(adj,endpoint)
    pred=predict(model,avalon_matrix(view.smiles),scaler,'cpu')
    torch.save(dict(role='SMOKE_NOT_FORMAL',state=model.state_dict(),contract=contract),out/'smoke.pt')
    np.savez(out/'validation_smoke.npz',sample_ids=np.asarray(ids),tasks=np.asarray(PRIMARY),truth=labels,prediction=pred)
    (out/'sample_ids.json').write_text(json.dumps(ids),encoding='utf8')
    report=dict(route='B',mode='one-step',active_rows=55670,observations=78324,tasks=61,
        feature_width=27,edges=320,source_target_edges=0,test_executed=False,updates=1,
        implementation=implementation_identity(),checkpoint_sha256=hashlib.sha256((out/'smoke.pt').read_bytes()).hexdigest(),
        validation_rows=2,metrics=validation_metrics(labels,pred),reload_max_abs=0,support_rows=2,loss=1.)
    (out/'report.json').write_text(json.dumps(report),encoding='utf8')
    yield dict(output=out,audit_path=ap,csv_path=files[0],split_path=files[1],tox_manifest=files[2]),report
    torch.set_num_threads(old_threads)


def test_verifier_synthetic_valid(bundle):
    kw,_=bundle
    assert verifier.verify(**kw)['content_status']=='PASS'


@pytest.mark.parametrize('kind',['member','truth','prediction','report','scope','graph'])
def test_verifier_rejects_self_consistent_wrong_content(bundle,kind):
    kw,report=bundle
    out=kw['output']
    if kind in ('member','truth','prediction'):
        with np.load(out/'validation_smoke.npz') as z: arrays={k:z[k] for k in z.files}
        if kind=='member': arrays['sample_ids'][0]='dataset115:row_9'
        if kind=='truth': arrays['truth'][0,0]+=1
        if kind=='prediction':
            arrays['prediction'][0,0]+=1
            report['metrics']=validation_metrics(arrays['truth'],arrays['prediction'])
        np.savez(out/'validation_smoke.npz',**arrays)
    elif kind=='report': report['active_rows']+=1
    elif kind=='scope': report['test_executed']=True
    else:
        c=json.loads((out/'data_contract.json').read_text())
        c['adjacency']=np.eye(61).tolist()
        (out/'data_contract.json').write_text(json.dumps(c),encoding='utf8')
        payload=torch.load(out/'smoke.pt',weights_only=True)
        payload['contract']=c
        payload['state']['adjacency']=torch.eye(61)
        torch.save(payload,out/'smoke.pt')
        report['checkpoint_sha256']=hashlib.sha256((out/'smoke.pt').read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report),encoding='utf8')
    with pytest.raises((ValueError,AssertionError)):
        verifier.verify(**kw)
