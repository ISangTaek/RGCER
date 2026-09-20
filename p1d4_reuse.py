"""Import accepted histories, verifying actual files; never train a missing alias."""
from copy import deepcopy
import json
import math
from pathlib import Path

from p1d4_identity import contract_for
from p1d_optimization import require
from p1d_tox import file_sha,replay_historical
from p1d_tox_training import _same

LOCK=Path(__file__).resolve().parent/'configs/p1d4_reuse_lock.json'


def bound_history(alias):
    values=json.loads(LOCK.read_bytes())
    require(values['schema']=='p1d4_reuse_lock_v1' and values['execution_authorized'] is False,'reuse lock')
    require(type(alias) is str and alias in values['runs'],'unapproved alias')
    r=deepcopy(values['runs'][alias])
    c=contract_for(r['setting'],r['seed'])
    tasks=list(c['counts']['validation']) if r['setting']=='ToxAcute' else c['task_names']
    history=r['history']
    require(type(history) is list and len(history)==40,'inherited complete history')
    counts=None;errors=[]
    for e,row in enumerate(history):
        require(type(row['epoch']) is int and row['epoch']==e and row['split']=='validation','inherited epoch')
        ep=row['endpoints'];require(set(ep)==set(tasks),'inherited tasks')
        current={t:ep[t]['n'] for t in tasks}
        require(all(type(n) is int and n>0 for n in current.values()),'inherited counts')
        if counts is None:counts=current
        require(current==counts,'inherited population changed')
        require(all(type(ep[t]['rmse']) in (int,float) and math.isfinite(ep[t]['rmse']) and ep[t]['rmse']>=0
                    for t in tasks),'inherited metric')
        errors.append(math.fsum(ep[t]['rmse'] for t in tasks)/len(tasks))
    if r['setting']=='ToxAcute':require(counts==c['counts']['validation'],'inherited Human3 counts')
    best=min(range(40),key=errors.__getitem__)
    require(type(r['best_epoch']) is int and best==r['best_epoch'] and abs(errors[best]-r['best_rmse'])<1e-12,
            'inherited best rule')
    return r


def check_files(repo,record):
    repo=Path(repo).resolve();evidence=[]
    for item in record['files']:
        path=(repo/item['path']).resolve()
        require(path.is_relative_to(repo) and path.is_file(),'missing or out-of-repo reused asset; STOP_NOT_RETRAIN')
        require(path.stat().st_size==item['size_bytes'] and file_sha(path)==item['sha256'], 'reused asset identity')
        evidence.append(dict(item))
    return evidence


def import_alias(repo,alias,*,tox_factory=None):
    record=bound_history(alias)
    files=check_files(repo,record)
    replay=None
    if record['fresh_validation_replay']:
        require(tox_factory is not None and _same(tox_factory.contract,contract_for('ToxAcute',record['seed'])),
                'same-seed bound replay factory')
        replay=replay_historical(tox_factory,path=Path(repo)/record['best_path'],
                expected_sha=record['best_sha256'],expected_epoch=record['best_epoch'],expected_rmse=record['best_rmse'])
    return dict(alias=alias,setting=record['setting'],seed=record['seed'],arm=record['arm'],
                history=record['history'],files=files,provenance=record['provenance'],
                validation_replay=replay,metric_evidence='INHERITED_ACCEPTED_HISTORY',
                new_epochs=0,new_optimizer_updates=0,validation_status='PASS',acceptance_status='PENDING_REVIEW')
