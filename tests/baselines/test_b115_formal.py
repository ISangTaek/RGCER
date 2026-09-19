import json
from pathlib import Path
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from baselines.b115_formal import choose,job_id,validate_history,read_campaign
from baselines.b115_training import configuration,LUT,validation_metrics
from baselines.models.toxacol import toxacol_learning_rate


def test_matrix_and_tie():
    assert choose({0:1.,1:1.,2:2.})==0
    assert choose({0:2.,1:1.,2:3.})==1
    jobs={job_id('screen',r,42,t) for r in ('A','B') for t in range(3)}
    jobs.update(job_id('replicate',r,s,1) for r in ('A','B') for s in range(43,47))
    assert len(jobs)==14


@pytest.mark.parametrize('args',[('screen','A',43,0),('replicate','B',42,1),('screen','C',42,0),
    ('screen','A',42,3),('test','A',42,0),('screen','A',True,0),('screen','A',42,True)])
def test_forbidden_jobs(args):
    with pytest.raises(ValueError):job_id(*args)


@pytest.mark.parametrize('scores',[{0:1,1:2},{0:1,1:2,2:float('nan')},{0:1,1:2,3:3}])
def test_incomplete_selection(scores):
    with pytest.raises(ValueError):choose(scores)


@pytest.fixture
def trajectory(tmp_path):
    truth=np.arange(10,dtype=float).reshape(2,5);config=configuration();history=[]
    for e in range(120):
        pred=truth+(.5 if e>=3 else 1.)
        np.save(tmp_path/f'validation_{e:03d}.npy',pred)
        m=validation_metrics(truth,pred)
        lr=toxacol_learning_rate(e,[r[0] for r in LUT[:-1]],[r[1] for r in LUT])
        history.append(dict(epoch=e,lr=lr,train_loss=1.,observations=100,epoch_updates=2,**m))
        (tmp_path/f'epoch_{e:03d}.json').write_text(json.dumps(dict(epoch=e,best_epoch=0 if e<3 else 3,**m)))
    run=dict(epochs_completed=120,start_epoch=0,resume_parent=None,test_executed=False,
             updates=240,history=history,best_epoch=3,best_validation_macro_rmse=.5)
    return run,truth,tmp_path,config,33,100


def test_complete_history(trajectory):
    assert validate_history(*trajectory)==(3,.5)


@pytest.mark.parametrize('kind',['steps','lr','best','missing_task','missing_epoch','prediction','observations','test'])
def test_tampered_history(trajectory,kind):
    run,truth,path,config,rows,obs=trajectory
    if kind=='steps':run['updates']-=1
    elif kind=='lr':run['history'][0]['lr']*=2
    elif kind=='best':run['best_epoch']=4
    elif kind=='missing_task':truth[:,0]=np.nan
    elif kind=='missing_epoch':run['history'].pop()
    elif kind=='prediction':np.save(path/'validation_000.npy',truth+100)
    elif kind=='observations':run['history'][0]['observations']-=1
    else:run['test_executed']=True
    with pytest.raises(ValueError):validate_history(run,truth,path,config,rows,obs)


def test_campaign_cannot_change_scope(tmp_path,monkeypatch):
    import baselines.b115_formal as m
    monkeypatch.setattr(m,'code_identity',lambda:{'code':'fixed'})
    p=dict(task_id=m.TASK_ID,commit='a'*40,code={'code':'fixed'},root=str(tmp_path.resolve()),
           max_runs=14,max_model_epochs=1680,allowed_splits=['train','validation'],input_sha256={},inputs={})
    (tmp_path/'campaign.json').write_text(json.dumps(p))
    assert read_campaign(tmp_path,'a'*40)==p
    p['allowed_splits'].append('test')
    (tmp_path/'campaign.json').write_text(json.dumps(p))
    with pytest.raises(ValueError):read_campaign(tmp_path,'a'*40)


@pytest.fixture
def verified_job_fixture(tmp_path):
    from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
    from dataset115_contract import PRIMARY
    from baselines.scaling import TaskScaler
    from baselines.features import avalon_matrix
    from baselines.models.toxacol import endpoint_feature_matrix
    from baselines.b115_training import train_engine
    from baselines.b115_formal import sha
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    tasks=tuple(ANIMAL_SOURCE_TASKS)+PRIMARY
    labels=np.random.RandomState(9).normal(size=(33,61))
    scaler=TaskScaler.fit(labels,tasks,allow_empty=False)
    features,_=endpoint_feature_matrix(tasks,extend_vocabulary=True);adj=np.eye(61,dtype=np.float32)
    val=SimpleNamespace(sample_ids=('v0','v1'),tasks=PRIMARY,smiles=('C','CC'),labels=np.ones((2,5)))
    train=SimpleNamespace(sample_ids=tuple(map(str,range(33))),labels=labels)
    contract={'tasks':list(tasks)}
    campaign=tmp_path/'campaign.json';campaign.write_text('{}')
    folder=tmp_path/'screen_A_t0_s42';folder.mkdir()
    path=folder/'training'
    train_engine(train_x=np.zeros((33,1024),dtype=np.float32),train_labels=labels,
        validation_x=avalon_matrix(val.smiles),validation_labels=val.labels,
        adjacency=adj,endpoint_features=features,scaler=scaler,contract=contract,output=path,
        config=configuration(lr_multiplier=.5),stop_after_epoch=1)
    # Simulated 120-record ledger for verifier tests, not real epochs or evidence.
    run=json.loads((path/'run.json').read_text());item=run['history'][0]
    pred=np.load(path/'validation_000.npy');payload=(path/'epoch_000.pt').read_bytes()
    meta=json.loads((path/'epoch_000.json').read_text());history=[]
    for epoch in range(120):
        lr=toxacol_learning_rate(epoch,[r[0] for r in LUT[:-1]],[r[1] for r in LUT])*.5
        history.append({**item,'epoch':epoch,'lr':lr})
        if epoch:
            (path/f'epoch_{epoch:03d}.pt').hardlink_to(path/'epoch_000.pt')
        (path/f'epoch_{epoch:03d}.json').write_text(json.dumps({**meta,'epoch':epoch}))
        np.save(path/f'validation_{epoch:03d}.npy',pred)
    run.update(history=history,epochs_completed=120,updates=240)
    (path/'run.json').write_text(json.dumps(run))
    np.savez(folder/'validation_truth.npz',sample_ids=np.asarray(val.sample_ids),tasks=np.asarray(PRIMARY),truth=val.labels)
    (folder/'execution.json').write_text(json.dumps(dict(exit_code=0,job_id=folder.name,trial=0,
        phase='screen',route='A',seed=42,campaign_sha256=sha(campaign))))
    yield tmp_path,(train,val,adj,features,scaler,contract)
    torch.set_num_threads(old_threads)


def test_full_verifier_synthetic_checkpoint_replay(verified_job_fixture):
    from baselines.b115_formal import verify_job
    root,loaded=verified_job_fixture
    result=verify_job(root,{},'screen','A',42,0,loaded)
    assert result['best_epoch']==0 and result['replay_max_abs']==0 and len(result['assets'])==120


@pytest.mark.parametrize('kind',['truth','weight','execution'])
def test_full_verifier_rejects(verified_job_fixture,kind):
    from baselines.b115_formal import verify_job
    root,loaded=verified_job_fixture;folder=root/'screen_A_t0_s42'
    if kind=='truth':
        with np.load(folder/'validation_truth.npz') as z: data={k:z[k] for k in z.files}
        data['truth'][0,0]+=1;np.savez(folder/'validation_truth.npz',**data)
    elif kind=='weight':
        with (folder/'training/epoch_003.pt').open('ab') as f:f.write(b'changed')
    else:
        p=folder/'execution.json';obj=json.loads(p.read_text());obj['trial']=1;p.write_text(json.dumps(obj))
    with pytest.raises((ValueError,AssertionError)):verify_job(root,{},'screen','A',42,0,loaded)
