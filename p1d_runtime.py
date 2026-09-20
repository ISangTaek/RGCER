"""Real-data P1D3 smoke wiring. No formal training/screening entry is exposed."""
from copy import deepcopy
from pathlib import Path
import json
import math
import torch

from dataset import DataCollator
from dataset115_adapter import GraphTaskView
from dataset115_contract import PRIMARY
from loss import QuantileRegressionLoss
from p1d_smoke import SmokeAdapter, model_digest, restore_rng, verify_smoke
from p1d_tox import ToxFactory, TASKS, tox_step, file_sha, replay_historical, validate_replay_rows
from p1d_optimization import require


def same(a,b,label):
    require(type(a) is type(b),label+' type')
    if isinstance(a,torch.Tensor):require(torch.equal(a.cpu(),b.cpu()),label)
    elif isinstance(a,dict):
        require(a.keys()==b.keys(),label+' keys')
        for k in a:same(a[k],b[k],label+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        require(len(a)==len(b),label+' length')
        for x,y in zip(a,b):same(x,y,label)
    else:require(a==b,label)


def batch_identity(batch,task,index):
    tensors={k:v for k,v in batch.to_dict().items() if isinstance(v,torch.Tensor)}
    require(torch.isfinite(batch.y).all(),'batch labels finite')
    return dict(task=task,index=index,sample_ids=list(batch.sample_id),
        labels=[float(v) for v in batch.y.reshape(-1).cpu()],tensor_digest=model_digest(tensors))


def claim_attempt(repo,setting,commit,output):
    require(setting in ('ToxAcute','A','B'),'attempt setting')
    folder=Path(repo)/'.tmp/p1d3_attempts_20260920';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(setting+'.json')
    # O_EXCL is the one-shot boundary, including concurrent invocations. Keep
    # this record after failures; no new output directory refunds the claim.
    with path.open('x',encoding='utf-8') as f:
        json.dump(dict(task_id='P1D3_REAL_SMOKE_20260920',setting=setting,
            commit=commit,output=str(Path(output).resolve()),reserved_updates=11),f,allow_nan=False)
    return path


def loss_for(trainer,task,batch,setting):
    if setting=='ToxAcute':return trainer._training_step(batch,task,0)[0]
    s=trainer.scalers[task]
    return QuantileRegressionLoss().compute_loss(trainer.model(batch,task_name=task)[task],
            (batch.y.reshape(-1,1)-s['mean'])/s['std'])


def route_step(trainer,task,batch):
    trainer.model.train();trainer.optimizer.zero_grad(set_to_none=True)
    loss=loss_for(trainer,task,batch,'route')
    require(torch.isfinite(loss),'nonfinite route smoke loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in trainer.model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    trainer.optimizer.step()
    return float(loss.detach())


def write_json(path,value):
    raw=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    p=Path(path)
    if p.exists():require(p.read_text(encoding='utf-8')==raw,'existing result differs: '+p.name)
    else:
        with p.open('x',encoding='utf-8') as f:f.write(raw)


def prepare(repo,lock,*,setting,split_manifest,source_lock,device):
    from dataset115_adapter import Dataset115Table
    from p1d_routes import RouteFactory
    from toxacute_datastore import ToxAcuteDataStore
    repo=Path(repo).resolve();inputs=lock['inputs']
    require(lock['schema']=='p1d3_smoke_lock_v1' and lock['seed']==42
            and lock['formal_training_authorized'] is False and lock['test_authorized'] is False,'smoke lock')
    require(type(lock['maximum_smoke_updates']) is int and lock['maximum_smoke_updates']==33
            and type(lock['smoke_updates_per_setting']) is int and lock['smoke_updates_per_setting']==11,'smoke budget')
    require(setting in ('ToxAcute','A','B'),'setting')
    store=ToxAcuteDataStore.resolve(repo/inputs['datastore_relative'])
    require(file_sha(store.root/'split_manifest.json')==inputs['tox_manifest_sha256'],'Tox manifest bytes')
    tox=None
    if setting=='ToxAcute':
        tox=ToxFactory(repo=repo,datastore=store.root,contract=lock['tox'],device=device)
        make=tox.make_trainer;get=tox.batch;step=tox_step;tasks=TASKS
        identity=tox.identity()
    else:
        require(file_sha(repo/inputs['csv_relative'])==inputs['csv_sha256'],'CSV bytes')
        require(file_sha(split_manifest)==inputs['split_sha256'],'dataset115 split bytes')
        require(file_sha(source_lock)==inputs['source_lock_sha256'],'source lock bytes')
        table=Dataset115Table.load(repo/inputs['csv_relative'],split_manifest,store.root/'split_manifest.json',
                                   expected_tox_sha=inputs['tox_manifest_sha256'])
        extra=(dict(source_repo=repo,source_lock=source_lock) if setting=='B' else
               dict(source_output=repo/inputs['route_a_source_relative']))
        factory=RouteFactory(table,route=setting,seed=42,expected_identity=lock['routes'][setting],**extra)
        make=lambda arm:factory.make_trainer(original=arm is None,arm=arm or 'B1_high',device=device)
        view=table.view(setting,'target','train')
        datasets={t:GraphTaskView(view,t) for t in PRIMARY}
        def get(t,i):
            require(t in PRIMARY and type(i) is int and 0<=i<8,'route smoke batch scope')
            ds=datasets[t]
            return DataCollator()([ds[j] for j in range(min(2,len(ds)))]).to(device)
        step=route_step;tasks=PRIMARY
        identity=dict(setting=setting,seed=42,contract=lock['routes'][setting],
                      batch_policy='first_two_train_records_per_task_in_fixed_view',tasks=list(tasks),
                      test_accessed=False,calibration_accessed=False)
    original=make(None)
    identity['task_names']=list(tasks)
    identity['initial_model_sha256']=model_digest(original.model.state_dict())
    scaler_attribute='task_scalers' if setting=='ToxAcute' else 'scalers'
    identity['scalers']=deepcopy(getattr(original,scaler_attribute))
    batch_evidence=[]
    for i,t in enumerate(list(tasks[:2])+[tasks[j%len(tasks)] for j in range(6)]):
        b=get(t,i)
        batch_evidence.append(batch_identity(b,t,i))
    identity['batch_evidence']=batch_evidence
    # No optimizer step is performed here. Fresh factory clones remain subject
    # to the same source/data checks in each branch.
    del original
    def capture(trainer):return dict(scalers=deepcopy(getattr(trainer,scaler_attribute)))
    def restore(trainer,value):
        same(value,dict(scalers=identity['scalers']),'restored scaler')
        setattr(trainer,scaler_attribute,deepcopy(value['scalers']))
    return SmokeAdapter(make,step,get,capture,restore),list(tasks),identity,tox


def verify_live_gradients(output,adapter,*,setting,expected_identity):
    """Recompute each loss and gradient using live TRAIN batches; zero updates."""
    from p1d_smoke import capture_rng
    saved_rng=capture_rng()
    verify_smoke(output,expected_identity)
    root=Path(output)
    receipt=json.loads((root/'receipt.json').read_bytes())
    tasks=receipt['pair_tasks']+receipt['hf_tasks']
    checks=[]
    try:
        trainer=adapter.factory(None)
        for name,task,index in ([(f'original_{i}.pt',tasks[i],i) for i in range(2)]+
                               [(f'B1_high_{i}.pt',tasks[i],i) for i in range(2)]+
                               [(f'HF_low_{i}.pt',tasks[2+i],2+i) for i in range(6)]+
                               [('resumed_epoch5.pt',tasks[7],7)]):
            raw=torch.load(root/name,map_location='cpu',weights_only=True)
            row=raw['payload'];before=row['before']
            trainer.model.load_state_dict(before['model'],strict=True)
            for n,p in trainer.model.named_parameters():p.requires_grad_(before['requires_grad'][n]);p.grad=None
            for n,m in trainer.model.named_modules():m.training=before['modes'][n]
            if adapter.restore_extra:adapter.restore_extra(trainer,deepcopy(before['extra']))
            batch=adapter.batch_getter(task,index)
            if 'batch_evidence' in expected_identity:
                same(batch_identity(batch,task,index),expected_identity['batch_evidence'][index],'live batch identity')
            restore_rng(before['rng'])
            trainer.model.train()
            loss=loss_for(trainer,task,batch,setting)
            require(torch.isfinite(loss) and float(loss.detach())==row['loss'],'live loss differs')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainer.model.parameters()),1.,error_if_nonfinite=True)
            same(capture_rng(),row['after']['rng'],'live RNG')
            for n,p in trainer.model.named_parameters():
                ref=row['gradients'][n]
                require((p.grad is None and ref is None) or
                        (p.grad is not None and ref is not None and torch.equal(p.grad.detach().cpu(),ref)),
                        'live gradient differs: '+n)
            checks.append(dict(artifact=name,task=task,loss=float(loss.detach()),gradients_equal=True))
        require(len(checks)==11,'live replay count')
        return dict(scope='TRAIN_BATCH_LOSS_GRADIENT_REPLAY_NO_OPTIMIZER_STEPS',checks=checks,optimizer_updates=0)
    finally:
        restore_rng(saved_rng)


def historical_replay(repo,tox):
    if tox is None:return None
    lock=json.loads((Path(repo)/'configs/p1d_accepted_s3_reuse.json').read_bytes())
    row=next(r for r in lock['runs'] if r['seed']==42)
    name='graphormer_d7_s3_e40_seed42_best.pt'
    return replay_historical(tox,path=Path(repo)/row['relative_path']/name,
        expected_sha=row['weight_sha256'][name],expected_epoch=row['best_epoch'],
        expected_rmse=row['best_validation_macro_rmse'])


def historical_check(repo,output,tox,*,replay):
    if tox is None:return None
    lock=json.loads((Path(repo)/'configs/p1d_accepted_s3_reuse.json').read_bytes())
    row=next(r for r in lock['runs'] if r['seed']==42)
    name='graphormer_d7_s3_e40_seed42_best.pt'
    path=Path(repo)/row['relative_path']/name
    target=Path(output)/'historical_validation.json'
    if replay:
        result=historical_replay(repo,tox)
        write_json(target,result)
    result=json.loads(target.read_bytes())
    require(file_sha(path)==result['checkpoint_sha256']==row['weight_sha256'][name],'historical binding')
    require(result['epoch']==row['best_epoch'] and result['rmse_tolerance']==1e-5,'historical replay contract')
    metrics=validate_replay_rows(tox,result['rows'])
    require(metrics==result['metrics'] and abs(metrics['macro_rmse']-row['best_validation_macro_rmse'])<=1e-5,'historical metrics')
    return dict(checkpoint_sha256=result['checkpoint_sha256'],epoch=result['epoch'],metrics=metrics)
