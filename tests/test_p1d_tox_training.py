from copy import deepcopy
import json
import pytest
import shutil
import torch

from p1d_optimization import OptimizationError
from p1d_tox_training import ToxRunAdapter, metrics
from tests.test_p1d_tox import fixture_factory, TASKS
from p1d_tox import file_sha


@pytest.fixture(scope='module')
def full_run(tmp_path_factory):
    root=tmp_path_factory.mktemp('tox_formal')/'run'
    factory=fixture_factory()
    adapter=ToxRunAdapter(factory,'HF_low')
    result=adapter.run(root)
    return root,factory,result


def test_full_forty_epochs_and_replay(full_run):
    root,factory,result=full_run
    assert result['optimizer_updates']==120
    assert len(result['history'])==40 and result['acceptance_status']=='PENDING_REVIEW'
    assert len(list(root.glob('epoch_*.pt')))==40
    assert ToxRunAdapter(factory,'HF_low').verify(root)==result
    with pytest.raises(OptimizationError,match='already exists'):
        ToxRunAdapter(factory,'HF_low').run(root)


def test_wrong_seed_or_arm_rejects_existing_run(full_run):
    root,factory,_=full_run
    with pytest.raises(OptimizationError,match='configuration identity'):
        ToxRunAdapter(factory,'HF_high').verify(root)
    f=fixture_factory();f.contract['args']['seed']=43
    with pytest.raises(OptimizationError,match='configuration identity'):
        ToxRunAdapter(f,'HF_low').verify(root)


@pytest.mark.parametrize('fault',['adam','scaler','rng','bool_identity','prediction','best'])
def test_self_consistent_tampering_rejected(full_run,tmp_path,fault):
    source,factory,_=full_run
    root=tmp_path/'tampered';shutil.copytree(source,root)
    def read(name):return json.loads((root/name).read_text())
    def write(name,value):(root/name).write_text(json.dumps(value,allow_nan=False),encoding='utf8')
    if fault=='best':
        summary=read('summary.json');summary['best_epoch']=True;write('summary.json',summary)
    elif fault=='prediction':
        name='epoch_000_validation.json';rows=read(name);rows[0]['label']+=1;write(name,rows)
        r=read('epoch_000_receipt.json');r['predictions_sha256']=file_sha(root/name);write('epoch_000_receipt.json',r)
    else:
        path=root/'epoch_000.pt';q=torch.load(path,weights_only=True)
        if fault=='adam':next(iter(q['optimizer']['state'].values()))['step'].fill_(2)
        if fault=='scaler':q['scalers'][TASKS[0]]['mean']+=1
        if fault=='rng':q['rng']['torch']=torch.zeros(2,dtype=torch.uint8)
        if fault=='bool_identity':q['identity']['test_accessed']=0
        torch.save(q,path)
        r=read('epoch_000_receipt.json');r.update(checkpoint_sha256=file_sha(path),checkpoint_size=path.stat().st_size)
        write('epoch_000_receipt.json',r)
    with pytest.raises(OptimizationError):ToxRunAdapter(factory,'HF_low').verify(root)


def test_raw_metrics_and_undefined_r2():
    f=fixture_factory();rows=[]
    for task,ds in f.datasets['validation'].items():
        for i in range(len(ds)):
            y=float(ds[i].y.reshape(-1)[0])
            rows.append(dict(task=task,sample_id=ds.get_sample_id(i),split='validation',label=y,prediction=y))
    result=metrics(f,rows)
    assert result['macro_rmse']==result['macro_mae']==0
    assert all(v['r2'] in (None,1.) for v in result['endpoints'].values())
    bad=deepcopy(rows);bad[0]['prediction']=float('nan')
    with pytest.raises(OptimizationError):metrics(f,bad)
    with pytest.raises(OptimizationError):metrics(f,rows[:-1])


@pytest.mark.parametrize('seed',[42,43,44,45,46])
def test_seed_is_propagated_without_replacing_bound_init(seed):
    from reproducibility import state_dict_sha256,stable_seed
    f=fixture_factory();f.contract['args']['seed']=seed
    trainer=f.make_trainer('B1_low')
    assert trainer.args.seed==seed and f.identity()['seed']==seed
    assert state_dict_sha256(trainer.model)==f.contract['initial_full_model_digest']
    for t,loader in f.loaders['train'].items():
        # Fitting scalers consumes the loader but retains its configured seed.
        assert loader.generator.initial_seed()==stable_seed(seed,t,'train',0)


@pytest.mark.parametrize('seed',[True,41,47,'42'])
def test_invalid_seed_rejected(seed):
    f=fixture_factory();f.contract['args']['seed']=seed
    with pytest.raises(OptimizationError,match='Tox seed'):f.make_trainer('B1_low')
