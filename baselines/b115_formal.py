"""Approved 14-run B115 campaign; verification re-reads outputs, not PASS flags."""
from __future__ import annotations
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from .b115_data import load_b115
from .b115_training import configuration, implementation_identity, LUT, predict, validation_metrics
from .features import avalon_matrix
from .models.toxacol import ToxACoLNet, toxacol_learning_rate
from .scaling import TaskScaler
from dataset115_contract import semantic_digest
from scripts.verify_b115_smoke import read_json

MULTIPLIERS = (.5, 1., 1.5)
TASK_ID = 'V5_B115_FORMAL_20260919'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def code_identity():
    root=Path(__file__).resolve().parents[1]
    result=implementation_identity()
    for name in ('baselines/b115_formal.py','scripts/run_b115_formal.py'):
        result[name]=sha(root/name)
    return result


def check_checkout(expected):
    if len(expected)!=40 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('full commit required')
    root=Path(__file__).resolve().parents[1]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=root,text=True).strip()
    if head!=expected or dirty:
        raise ValueError('checkout commit/worktree differs; no reset/clean permitted')


def job_id(phase,route,seed,trial):
    if route not in ('A','B') or type(seed) is not int or type(trial) is not int or trial not in range(3):
        raise ValueError('unknown route/seed/trial')
    if phase=='screen' and seed==42: return f'screen_{route}_t{trial}_s42'
    if phase=='replicate' and seed in range(43,47): return f'replicate_{route}_s{seed}'
    raise ValueError('job outside approved 6+8 matrix')


def choose(scores):
    if set(scores)!={0,1,2} or any(type(k) is not int or not math.isfinite(v) for k,v in scores.items()):
        raise ValueError('requires three finite trials')
    return min(scores,key=lambda t:(scores[t],t))


def load_route(campaign,route):
    inputs=campaign['inputs']
    return load_b115(route=route,allowlist=inputs[f'allowlist_{route}'],
        **{k:inputs[k] for k in ('audit_path','csv_path','split_path','tox_manifest','datastore')})


def read_campaign(root,commit):
    c=read_json(Path(root)/'campaign.json')
    if (c['task_id']!=TASK_ID or c['commit']!=commit or c['code']!=code_identity()
            or c['root']!=str(Path(root).resolve()) or c['max_runs']!=14 or c['max_model_epochs']!=1680
            or c['allowed_splits']!=['train','validation']):
        raise ValueError('campaign identity/scope mismatch')
    for name,value in c['input_sha256'].items():
        if sha(c['inputs'][name])!=value: raise ValueError('input changed: '+name)
    return c


def validate_history(run,truth,directory,config,rows,observations):
    history=run['history']
    if (run['epochs_completed']!=120 or run['start_epoch']!=0 or run['resume_parent'] is not None
            or run['test_executed'] is not False or len(history)!=120):
        raise ValueError('incomplete/unapproved trajectory')
    steps=math.ceil(rows/32)
    if run['updates']!=120*steps: raise ValueError('wrong update count')
    scores=[]
    for epoch,item in enumerate(history):
        lr=toxacol_learning_rate(epoch,[r[0] for r in LUT[:-1]],[r[1] for r in LUT])*config['lr_multiplier']
        if (type(item['epoch']) is not int or item['epoch']!=epoch or item['epoch_updates']!=steps
                or item['observations']!=observations or item['lr']!=lr or not math.isfinite(item['train_loss'])):
            raise ValueError('epoch history/cost/LR mismatch')
        predictions=np.load(Path(directory)/f'validation_{epoch:03d}.npy',allow_pickle=False)
        metrics=validation_metrics(truth,predictions)
        # Independent reduction, in addition to the common reporting API.
        errors=[np.sqrt(np.mean((predictions[np.isfinite(truth[:,j]),j]-truth[np.isfinite(truth[:,j]),j])**2)) for j in range(5)]
        if not np.isclose(metrics['macro_rmse'],np.mean(errors),rtol=1e-12,atol=1e-12):
            raise ValueError('independent RMSE mismatch')
        if any(item[k]!=v for k,v in metrics.items()): raise ValueError('history metrics mismatch')
        meta=read_json(Path(directory)/f'epoch_{epoch:03d}.json')
        if any(meta[k]!=v for k,v in metrics.items()) or meta['epoch']!=epoch:
            raise ValueError('epoch metadata mismatch')
        scores.append(metrics['macro_rmse'])
        if meta['best_epoch']!=min(range(epoch+1),key=lambda e:scores[e]):
            raise ValueError('intermediate selection mismatch')
    best=min(range(120),key=lambda e:scores[e])
    if run['best_epoch']!=best or run['best_validation_macro_rmse']!=scores[best]:
        raise ValueError('best selection mismatch')
    return best,scores[best]


def verify_job(root,campaign,phase,route,seed,trial,loaded=None):
    job=job_id(phase,route,seed,trial);folder=Path(root)/job;directory=folder/'training'
    record=read_json(folder/'execution.json')
    expected_config=configuration(seed=seed,lr_multiplier=MULTIPLIERS[trial])
    if (record['exit_code']!=0 or record['job_id']!=job or record['trial']!=trial
            or record['phase']!=phase or record['route']!=route or record['seed']!=seed
            or record['campaign_sha256']!=sha(Path(root)/'campaign.json')):
        raise ValueError('execution contract mismatch')
    train,val,adj,features,scaler,contract=loaded or load_route(campaign,route)
    with np.load(folder/'validation_truth.npz',allow_pickle=False) as archive:
        if archive['sample_ids'].tolist()!=list(val.sample_ids) or archive['tasks'].tolist()!=list(val.tasks):
            raise ValueError('exported validation membership mismatch')
        np.testing.assert_array_equal(archive['truth'],val.labels)
    resolved=read_json(directory/'resolved.json');run=read_json(directory/'run.json')
    if (resolved['data']!=contract or resolved['config']!=expected_config
            or resolved['scaler']!=scaler.to_dict() or resolved['adjacency']!=adj.tolist()
            or resolved['endpoint_features']!=features.tolist()
            or resolved['implementation']!=implementation_identity()
            or run['identity']!=semantic_digest(resolved)):
        raise ValueError('resolved data/config identity mismatch')
    best,score=validate_history(run,val.labels,directory,expected_config,len(train.sample_ids),int(np.isfinite(train.labels).sum()))
    assets=[]
    for epoch in range(120):
        path=directory/f'epoch_{epoch:03d}.pt';digest=sha(path)
        if read_json(directory/f'epoch_{epoch:03d}.json')['checkpoint_sha256']!=digest:
            raise ValueError('checkpoint SHA mismatch')
        assets.append(dict(epoch=epoch,path=str(path.resolve()),sha256=digest,bytes=path.stat().st_size))
    payload=torch.load(directory/f'epoch_{best:03d}.pt',map_location='cpu',weights_only=True)
    if (payload['format']!='B115_epoch_v1' or payload['epoch']!=best or payload['resolved']!=resolved
            or payload['identity']!=run['identity'] or payload['best_epoch']!=best or payload['best']!=score):
        raise ValueError('best checkpoint identity mismatch')
    model=ToxACoLNet(adj,features);model.load_state_dict(payload['model'],strict=True)
    torch.testing.assert_close(model.adjacency,torch.as_tensor(adj),rtol=0,atol=0)
    torch.testing.assert_close(model.endpoint_features,torch.as_tensor(features),rtol=0,atol=0)
    actual=predict(model,avalon_matrix(val.smiles),scaler,'cpu')
    saved=np.load(directory/f'validation_{best:03d}.npy',allow_pickle=False)
    np.testing.assert_allclose(actual,saved,rtol=1e-5,atol=1e-5)
    return dict(job_id=job,route=route,seed=seed,trial=trial,best_epoch=best,macro_rmse=score,
        run_sha256=sha(directory/'run.json'),best_checkpoint_sha256=assets[best]['sha256'],
        replay_max_abs=float(np.max(np.abs(actual-saved))),epochs=120,updates=run['updates'],assets=assets)


def screen_selection(root,campaign):
    results=[];selected={}
    for route in ('A','B'):
        loaded=load_route(campaign,route)
        entries=[verify_job(root,campaign,'screen',route,42,t,loaded) for t in range(3)]
        results.extend(entries)
        selected[route]=choose({e['trial']:e['macro_rmse'] for e in entries})
    return dict(task_id=TASK_ID,campaign_sha256=sha(Path(root)/'campaign.json'),
                selected=selected,screen=results)


def verify_selection(root,campaign):
    current=screen_selection(root,campaign)
    saved=read_json(Path(root)/'selection.json')
    if ({k:v for k,v in saved.items() if k!='screen'} !=
            {k:v for k,v in current.items() if k!='screen'}
            or len(saved['screen'])!=len(current['screen'])
            or not all(same_verified_job(a,b) for a,b in zip(saved['screen'],current['screen']))):
        raise ValueError('selection changed or stale')
    return current


def same_verified_job(saved,current):
    """Compare identities/results exactly, not the diagnostic CPU replay residual.

    Callers must obtain current from verify_job, which checks the actual arrays
    with the frozen allclose tolerance. A small residual is not itself a PASS.
    """
    for value in (saved,current):
        residual=value.get('replay_max_abs')
        if type(residual) not in (int,float) or not math.isfinite(residual) or residual<0:
            return False
    return ({k:v for k,v in saved.items() if k!='replay_max_abs'} ==
            {k:v for k,v in current.items() if k!='replay_max_abs'})
