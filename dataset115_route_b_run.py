"""Frozen S5B Route B training/validation contract and content verification."""
from pathlib import Path
import json
import math
import hashlib
import io

import numpy as np
import torch

from dataset115_contract import PRIMARY, semantic_digest
from dataset115_training import EpochConfig, require, task_batches, validation_metrics
from dataset115_smoke import state_digest

TASK_ID = 'S5B_ROUTEB_15_TRAIN_VALIDATION_20260915'
CONFIG = EpochConfig(40, 32, 0.001, 1e-5, 1.0)
SERVER_TOX_SHA = 'eeba19c20362e88d309aab9cd59200f3eff3d4b2a48e5f4c8dc5e57b712f6ae8'


def verify_training_run(output, trainer):
    """Recompute every epoch metric from IDs/labels/predictions and read tensors.

    trainer must be freshly bound to the actual frozen CSV/splits/source, not
    constructed from the result's self-declared contract. CPU verification;
    does not train or evaluate test/calibration and does not grant acceptance.
    """
    output = Path(output)
    def read(name):
        def bad(x): raise ValueError('nonfinite JSON '+x)
        def pairs(items):
            d={}
            for k,v in items:
                require(k not in d,'duplicate JSON field');d[k]=v
            return d
        return json.loads((output/name).read_bytes(),parse_constant=bad,object_pairs_hook=pairs)
    summary = read('training_summary.json')
    config = read('resolved_config.json')
    require(config == trainer.identity and summary['identity_sha256'] == trainer.identity_sha, 'result contract identity')
    require(summary['complete'] is True and summary['completed_epochs'] == summary['planned_epochs'] == trainer.config.epochs, 'incomplete run')
    require(summary['inherited_checkpoint_sha256'] is None, 'this batch does not authorize resumed assets')
    for k in ('test_predictions_accessed','calibration_predictions_accessed'):
        require(summary[k] is False,'result split scope')
    view = trainer.validation
    expected = {(t,view.sample_ids[i]):(float(view.labels[i,j]),view.canonical[i],view.groups[i])
        for j,t in enumerate(PRIMARY) for i in range(len(view.sample_ids)) if np.isfinite(view.labels[i,j])}
    def verify_rows(rows):
        keys=[(r['task'],r['sample_id']) for r in rows]
        require(len(keys)==len(set(keys)) and set(keys)==set(expected),'prediction population')
        for row in rows:
            ref=expected[(row['task'],row['sample_id'])]
            require((row['label'],row['canonical'],row['group'])==ref and row['split']=='validation','prediction raw identity')
        return validation_metrics(rows)
    epochs=trainer.config.epochs
    require(len(summary['history'])==len(summary['checkpoints'])==epochs,'epoch matrix')
    counts={t:len(d) for t,d in trainer.datasets['train'].items()}
    nsteps=len(task_batches(counts,trainer.config.batch_size,trainer.seed,0))
    best_epoch=None;best_score=None;best_rows=None;last=None
    for epoch,(history,receipt) in enumerate(zip(summary['history'],summary['checkpoints'])):
        require(history['epoch']==receipt['epoch']==epoch,'epoch order')
        require(history['exposure']==counts and history['total_steps']==nsteps*(epoch+1),'exposure/steps')
        require(set(history['mean_batch_loss'])==set(PRIMARY) and all(math.isfinite(v) for v in history['mean_batch_loss'].values()),'training loss')
        require(math.isfinite(history['max_grad_norm']),'gradient norm')
        rows=read(f'validation_epoch_{epoch:03d}.json');metrics=verify_rows(rows)
        require(metrics==history['validation'],'recomputed epoch metric')
        if best_score is None or metrics['macro_rmse']<best_score:
            best_epoch=epoch;best_score=metrics['macro_rmse'];best_rows=rows
        require(receipt==read(f'epoch_{epoch:03d}.receipt.json'),'receipt identity')
        name=f'epoch_{epoch:03d}.pt';require(receipt['path']==name,'checkpoint path')
        raw=(output/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==receipt['sha256'] and len(raw)==receipt['size_bytes'],'checkpoint file identity')
        p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        require(p['identity']==config and p['identity_sha256']==trainer.identity_sha and p['epoch']==epoch,'checkpoint contract')
        require(p['history']==summary['history'][:epoch+1] and p['total_steps']==history['total_steps'],'checkpoint history')
        require(p['initial_encoder']==trainer.initial_encoder and p['initial_heads']==trainer.initial_heads,'initial tensor identity')
        require(p['best_epoch']==best_epoch and p['best_rows']==best_rows,'checkpoint best rule')
        expected_state=trainer.model.state_dict()
        for field,digest_field in [('model_state','model_digest'),('best_model_state','best_model_digest')]:
            state=p[field]
            require(set(state)==set(expected_state),'model tensor keys')
            for k,v in state.items():
                require(v.shape==expected_state[k].shape and v.dtype==expected_state[k].dtype and torch.isfinite(v).all(),'model tensor spec')
            require(state_digest(state)==p[digest_field],'model tensor digest')
            if trainer.method=='RPT':
                require(state_digest({k[len('encoder.'):]:v for k,v in state.items() if k.startswith('encoder.')})==trainer.initial_encoder,'RPT source drift')
        if best_epoch==epoch:require(p['model_digest']==p['best_model_digest'],'selected weights differ')
        elif last is not None:require(p['best_model_digest']==last['best_model_digest'],'unselected best changed')
        for s in p['optimizer_state']['state'].values():
            require(all(torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor)),'optimizer finite')
        last=p
    require(summary['best_epoch']==best_epoch and summary['best_validation']==summary['history'][best_epoch]['validation'],'final best')
    require(read('best_validation.json')==best_rows,'final best predictions')
    # Validate actual selected tensors with a fresh forward in the same runtime.
    trainer.model.load_state_dict(last['best_model_state'],strict=True)
    replay_rows,replay_metrics=trainer.evaluate_validation()
    require(replay_rows==best_rows and replay_metrics==summary['best_validation'],'independent best forward')
    return dict(task_id=TASK_ID, method=trainer.method,seed=trainer.seed,epochs_checked=epochs,
        checkpoints_checked=epochs,validation_observations=len(expected),best_epoch=best_epoch,
        best_validation=summary['best_validation'],identity_sha256=trainer.identity_sha,
        initial_encoder=trainer.initial_encoder,initial_heads=trainer.initial_heads,
        content_status='PASS',acceptance_status='PENDING_REVIEW',test_predictions_accessed=False,
        model_epoch_files_remain_on_server=True)
