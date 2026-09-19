"""059-accepted single-best inference. No training, selection or ensemble API."""
from __future__ import annotations

from pathlib import Path
import hashlib
import zipfile
import numpy as np
import torch

from .b115_formal import sha
from .b115_training import implementation_identity, predict, save_json
from .features import avalon_matrix
from .metrics import regression_metrics
from .models.toxacol import ToxACoLNet
from .scaling import TaskScaler
from dataset115_adapter import Dataset115Table, LabelView
from dataset115_contract import PRIMARY, semantic_digest
from scripts.verify_b115_smoke import read_json

POLICY_SHA='e910076598b38be032b03b02d304a1942acbf7a027eaecd336a6fe8b09523b45'
POLICY=Path(__file__).resolve().parents[1]/'protocols/b115_test_20260920.json'


def load_policy():
    if sha(POLICY)!=POLICY_SHA:
        raise ValueError('frozen policy bytes differ')
    policy=read_json(POLICY)
    if policy['implementation']!=implementation_identity():
        raise ValueError('inference engine differs from accepted training implementation')
    return policy


def check_assets(root,policy):
    root=Path(root)
    if sha(root/'campaign.json')!=policy['campaign_sha256']:
        raise ValueError('wrong accepted campaign')
    for job in policy['jobs']:
        for name,digest in job['files'].items():
            if sha(root/job['job_id']/name)!=digest:
                raise ValueError('accepted asset changed: '+job['job_id']+'/'+name)


def active_view(table,route,split):
    view=table.view(route,'target',split)
    keep=np.flatnonzero(np.isfinite(view.labels).any(axis=1))
    return LabelView(route,'target',split,PRIMARY,
        *(tuple(getattr(view,k)[i] for i in keep) for k in ('sample_ids','smiles','canonical','groups')),
        view.labels[keep],view.input_identity)


def load_table(policy,inputs):
    if set(inputs)!=set(policy['inputs']):
        raise ValueError('wrong input keys')
    for key,digest in policy['inputs'].items():
        if sha(inputs[key])!=digest:
            raise ValueError('frozen input changed: '+key)
    return Dataset115Table.load(inputs['csv_path'],inputs['split_path'],inputs['tox_manifest'],
                                expected_tox_sha=policy['inputs']['tox_manifest'])


def restore(root,job):
    folder=Path(root)/job['job_id'];directory=folder/'training'
    # Check bytes again at use time; never trust only an earlier receipt.
    for name,digest in job['files'].items():
        if sha(folder/name)!=digest:raise ValueError('asset changed during inference')
    resolved=read_json(directory/'resolved.json')
    payload=torch.load(directory/f"epoch_{job['best_epoch']:03d}.pt",map_location='cpu',weights_only=True)
    if (payload['format']!='B115_epoch_v1' or payload['resolved']!=resolved
            or payload['identity']!=semantic_digest(resolved)
            or payload['epoch']!=job['best_epoch'] or payload['best_epoch']!=job['best_epoch']
            or payload['best']!=job['validation_macro_rmse']
            or resolved['config']['seed']!=job['seed'] or resolved['data']['route']!=job['route']):
        raise ValueError('checkpoint payload mismatch')
    adj=np.asarray(resolved['adjacency'],dtype=np.float32)
    features=np.asarray(resolved['endpoint_features'],dtype=np.float32)
    scaler=TaskScaler.from_dict(resolved['scaler'])
    if scaler.task_names[-5:]!=PRIMARY:raise ValueError('wrong target order')
    model=ToxACoLNet(adj,features,dropout=resolved['config']['dropout'])
    model.load_state_dict(payload['model'],strict=True)
    torch.testing.assert_close(model.adjacency,torch.as_tensor(adj),rtol=0,atol=0)
    torch.testing.assert_close(model.endpoint_features,torch.as_tensor(features),rtol=0,atol=0)
    return model,scaler


def metrics(truth,prediction):
    if truth.shape!=prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('wrong shape/nonfinite prediction')
    result=regression_metrics(truth,prediction,PRIMARY)
    if result['macro_rmse'] is None:raise ValueError('undefined primary macro')
    direct=[float(np.sqrt(np.mean((prediction[np.isfinite(truth[:,j]),j]-truth[np.isfinite(truth[:,j]),j])**2))) for j in range(5)]
    np.testing.assert_allclose(result['macro_rmse'],np.mean(direct),rtol=1e-12,atol=1e-12)
    return result


def preflight(root,policy,table):
    check_assets(root,policy)
    loaded={};records=[]
    for route in ('A','B'):
        view=active_view(table,route,'validation');features=avalon_matrix(view.smiles)
        for job in policy['jobs']:
            if job['route']!=route:continue
            folder=Path(root)/job['job_id']
            with np.load(folder/'validation_truth.npz',allow_pickle=False) as z:
                if z['sample_ids'].tolist()!=list(view.sample_ids) or z['tasks'].tolist()!=list(PRIMARY):
                    raise ValueError('validation membership/order mismatch')
                np.testing.assert_array_equal(z['truth'],view.labels)
            model,scaler=restore(root,job)
            predictions=predict(model,features,scaler,'cpu')
            saved=np.load(folder/'training'/f"validation_{job['best_epoch']:03d}.npy",allow_pickle=False)
            np.testing.assert_allclose(predictions,saved,rtol=1e-5,atol=1e-5)
            if metrics(view.labels,saved)['macro_rmse']!=job['validation_macro_rmse']:
                raise ValueError('saved validation score differs from accepted score')
            loaded[job['job_id']]=(model,scaler)
            records.append(dict(job_id=job['job_id'],validation_macro_rmse=job['validation_macro_rmse'],
                replay_max_abs=float(np.abs(predictions-saved).max()),rows=len(view.sample_ids),
                observations=int(np.isfinite(view.labels).sum())))
    if len(records)!=10:raise ValueError('requires all ten validation replays')
    return loaded,records


def arrays(view,prediction):
    return dict(sample_ids=np.asarray(view.sample_ids),canonical=np.asarray(view.canonical),
        groups=np.asarray(view.groups),tasks=np.asarray(PRIMARY),truth=view.labels,prediction=prediction)


def check_export(path,view,replayed):
    expected=arrays(view,replayed)
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(expected):raise ValueError('wrong prediction fields')
        for key,value in expected.items():
            if key=='prediction':np.testing.assert_allclose(z[key],value,rtol=1e-5,atol=1e-5)
            else:np.testing.assert_array_equal(z[key],value)
        return metrics(z['truth'],z['prediction'])


def run(action,root,output,inputs):
    if action not in ('preflight','test','verify'):raise ValueError('unapproved action')
    policy=load_policy();check_assets(root,policy)
    output=Path(output)
    if action!='verify' and output.exists():raise ValueError('output exists; do not overwrite')
    table=load_table(policy,inputs)
    models,records=preflight(root,policy,table)
    if action!='verify':
        output.mkdir(parents=True,exist_ok=False)
        save_json(output/'validation_replay.json',dict(policy_sha256=POLICY_SHA,runs=records))
    if action=='preflight':return dict(validation_replays=10,test_executed=False)
    if action=='verify':
        expected={j['job_id']+'.npz' for j in policy['jobs']}|{'test_summary.json','validation_replay.json','execution.json'}
        if {p.name for p in output.iterdir()}!=expected:raise ValueError('unexpected/missing export members')
    # No test view or test forward call occurs until all ten validations pass.
    rows=[]
    for route in ('A','B'):
        view=active_view(table,route,'test');x=avalon_matrix(view.smiles)
        for job in policy['jobs']:
            if job['route']!=route:continue
            model,scaler=models[job['job_id']]
            prediction=predict(model,x,scaler,'cpu')
            file=output/(job['job_id']+'.npz')
            if action=='test':np.savez_compressed(file,**arrays(view,prediction))
            result=check_export(file,view,prediction)
            rows.append(dict(job_id=job['job_id'],route=route,seed=job['seed'],best_epoch=job['best_epoch'],
                checkpoint_sha256=job['files'][f"training/epoch_{job['best_epoch']:03d}.pt"],
                prediction_sha256=sha(file),rows=len(view.sample_ids),
                observations=int(np.isfinite(view.labels).sum()),groups=len(set(view.groups)),metrics=result))
    result=dict(task_id=policy['task_id'],policy_sha256=POLICY_SHA,mode='single_validation_best',
        training_commit=policy['training_commit'],test_executed=True,calibration_executed=False,
        new_training_epochs=0,runs=rows,acceptance_status='PENDING_REVIEW')
    if action=='test':save_json(output/'test_summary.json',result)
    elif read_json(output/'test_summary.json')!=result:raise ValueError('test summary differs from recomputed result')
    return result


def package(output):
    """Call only after fresh run('verify'); exclusive archive creation."""
    output=Path(output)
    files=sorted(p for p in output.iterdir() if p.is_file())
    if any(p.suffix not in ('.json','.npz') for p in files):raise ValueError('unexpected output file')
    hashes={p.name:sha(p) for p in files}
    target=output.with_suffix('.zip')
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.name)
        z.writestr('checksums.sha256',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('CRC mismatch')
        for name,digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('member mismatch')
    with target.with_suffix('.zip.sha256').open('x',encoding='utf8') as stream:
        stream.write(f'{sha(target)}  {target.name}\n')
    return str(target)
