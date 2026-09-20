"""Fixed 40-epoch Human3 engine and independent artifact verifier.

Library only: invoking this is the responsibility of the budget-locked runner.
No resume, CLI, data reconstruction, holdout access or authorization inference.
"""
from copy import deepcopy
from pathlib import Path
import math
import time

import torch

from dataset115_training import _write_json
from p1d_optimization import OptimizationSpec, require
from p1d_smoke import capture_rng, restore_rng, _clone, _read_json, model_digest
from p1d_tox import TASKS, file_sha, validate_replay_rows
from p1d_tox_epoch import run_epoch


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys()==b.keys() and all(_same(a[k],b[k]) for k in a)
    if isinstance(a, (tuple,list)):
        return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    return a==b


def _verify_rng(state):
    require(type(state) is dict and set(state)=={'python','numpy','torch','cuda'}, 'RNG schema')
    require(isinstance(state['torch'],torch.Tensor) and state['torch'].dtype==torch.uint8
            and state['torch'].shape==torch.get_rng_state().shape, 'torch RNG')
    require(type(state['cuda']) is list and all(isinstance(v,torch.Tensor)
            and v.dtype==torch.uint8 and v.ndim==1 and v.numel()>0 for v in state['cuda']), 'CUDA RNG')
    current=capture_rng()
    try:
        # Verify Python/NumPy/CPU formats without consuming or changing caller RNG.
        restore_rng(dict(state,cuda=[]))
    finally:
        restore_rng(current)


def validation_rows(factory, trainer, epoch):
    trainer.model.eval()
    rows = []
    with torch.no_grad():
        for task in TASKS:
            for batch in factory.loaders['validation'][task]:
                require(not batch.is_empty, 'empty validation batch')
                batch = batch.to(factory.device)
                output, _ = trainer._forward_task(batch, task, epoch, return_aux=True)
                pred = trainer.decode_task_output(task, output[task], apply_conformal=False)['median'].reshape(-1)
                require(pred.numel() == batch.y.numel() == len(batch.sample_id)
                        and torch.isfinite(pred).all(), 'validation prediction shape/finite')
                rows.extend(dict(task=task, sample_id=str(sid), split='validation',
                                 label=float(y), prediction=float(p))
                            for sid, y, p in zip(batch.sample_id, batch.y.reshape(-1).cpu(), pred.cpu()))
    validate_replay_rows(factory, rows)
    return rows


def metrics(factory, rows):
    result = validate_replay_rows(factory, rows)
    for task, values in result['endpoints'].items():
        selected = [r for r in rows if r['task'] == task]
        n = len(selected)
        mean = math.fsum(r['label'] for r in selected) / n
        sst = math.fsum((r['label'] - mean)**2 for r in selected)
        sse = math.fsum((r['prediction'] - r['label'])**2 for r in selected)
        values.update(mae=math.fsum(abs(r['prediction'] - r['label']) for r in selected)/n,
                      r2=None if n < 2 or sst == 0 else 1 - sse/sst,
                      r2_reason='insufficient_or_constant_labels' if n < 2 or sst == 0 else None)
    result['macro_mae'] = math.fsum(result['endpoints'][t]['mae'] for t in TASKS)/3
    return result


def identity(factory, arm):
    return dict(schema='p1d4_tox_training_v1', setting='ToxAcute',
                seed=factory.contract['args']['seed'], contract=deepcopy(factory.contract),
                optimization=OptimizationSpec(arm).identity(), epochs=40,
                selection='validation_human3_macro_rmse_strict_min_first_tie',
                device=factory.device, torch_version=str(torch.__version__),
                test_accessed=False, calibration_accessed=False, resumed=False)


class ToxRunAdapter:
    def __init__(self, factory, arm):
        self.factory, self.arm = factory, arm
        self.identity = identity(factory, arm)
        self.used = False

    def run(self, output):
        require(not self.used, 'adapter already used')
        root = Path(output)
        require(not root.exists(), 'output already exists; no overwrite/resume')
        self.used = True
        root.mkdir(parents=True)
        _write_json(root/'configuration.json', self.identity)
        trainer = self.factory.make_trainer(self.arm)
        _write_json(root/'scalers.json', trainer.task_scalers)
        history = []
        best = None
        for epoch in range(40):
            start = time.monotonic()
            train = run_epoch(trainer, self.factory.loaders['train'], epoch)
            rows = validation_rows(self.factory, trainer, epoch)
            result = metrics(self.factory, rows)
            record = dict(epoch=epoch, split='validation', **result, train=train,
                          elapsed_seconds=time.monotonic()-start)
            name = f'epoch_{epoch:03d}'
            _write_json(root/(name+'_validation.json'), rows)
            _write_json(root/(name+'_metrics.json'), record)
            checkpoint = dict(schema='p1d4_tox_checkpoint_v1', identity=self.identity,
                              epoch=epoch, model_state=_clone(trainer.model.state_dict()),
                              optimizer=_clone(trainer.optimizer.state_dict()),
                              scalers=deepcopy(trainer.task_scalers), rng=capture_rng(),
                              loader_rng={t:l.generator.get_state().clone() for t,l in self.factory.loaders['train'].items()},
                              optimizer_updates=trainer.optimizer_updates)
            with (root/(name+'.pt')).open('xb') as stream:
                torch.save(checkpoint, stream)
            _write_json(root/(name+'_receipt.json'), dict(
                epoch=epoch, checkpoint_sha256=file_sha(root/(name+'.pt')),
                checkpoint_size=(root/(name+'.pt')).stat().st_size,
                model_digest=model_digest(checkpoint['model_state']),
                predictions_sha256=file_sha(root/(name+'_validation.json')),
                metrics_sha256=file_sha(root/(name+'_metrics.json'))))
            history.append(record)
            if best is None or result['macro_rmse'] < history[best]['macro_rmse']:
                best = epoch
        _write_json(root/'history.json', history)
        _write_json(root/'summary.json', dict(identity=self.identity, epochs_completed=40,
                    best_epoch=best, best_validation=metrics(self.factory,
                    _read_json(root/f'epoch_{best:03d}_validation.json')),
                    best_checkpoint_sha256=file_sha(root/f'epoch_{best:03d}.pt'),
                    optimizer_updates=trainer.optimizer_updates))
        return self.verify(root)

    def verify(self, output):
        root = Path(output)
        require(_same(_read_json(root/'configuration.json'), self.identity), 'configuration identity')
        trainer = self.factory.make_trainer(self.arm)
        require(_same(_read_json(root/'scalers.json'), trainer.task_scalers), 'train-only scalers')
        history = _read_json(root/'history.json')
        require(type(history) is list and len(history) == 40, 'complete history')
        checked = []
        reference_state = trainer.model.state_dict()
        for epoch in range(40):
            name = f'epoch_{epoch:03d}'
            receipt = _read_json(root/(name+'_receipt.json'))
            require(type(receipt['epoch']) is int and receipt['epoch'] == epoch, 'receipt epoch')
            for suffix, field in (('.pt','checkpoint_sha256'), ('_validation.json','predictions_sha256'),
                                  ('_metrics.json','metrics_sha256')):
                require(file_sha(root/(name+suffix)) == receipt[field], 'artifact SHA '+field)
            require((root/(name+'.pt')).stat().st_size == receipt['checkpoint_size'], 'checkpoint size')
            q = torch.load(root/(name+'.pt'), map_location='cpu', weights_only=True)
            require(set(q) == {'schema','identity','epoch','model_state','optimizer','scalers','rng','loader_rng','optimizer_updates'},
                    'checkpoint schema')
            require(q['schema']=='p1d4_tox_checkpoint_v1' and _same(q['identity'],self.identity)
                    and type(q['epoch']) is int and q['epoch']==epoch, 'checkpoint identity/epoch')
            state = q['model_state']
            require(set(state)==set(reference_state) and all(isinstance(v,torch.Tensor)
                    and v.shape==reference_state[k].shape and v.dtype==reference_state[k].dtype
                    and torch.isfinite(v).all() for k,v in state.items()), 'model contract')
            require(model_digest(state)==receipt['model_digest'], 'model digest')
            require(_same(q['scalers'],trainer.task_scalers), 'checkpoint scalers')
            _verify_rng(q['rng'])
            require(type(q['loader_rng']) is dict and set(q['loader_rng'])==set(TASKS), 'loader RNG tasks')
            for t,value in q['loader_rng'].items():
                require(isinstance(value,torch.Tensor) and value.dtype==torch.uint8
                        and value.shape==self.factory.loaders['train'][t].generator.get_state().shape, 'loader RNG')
                torch.Generator().set_state(value)
            self._verify_steps(trainer, q, epoch)
            rows = _read_json(root/(name+'_validation.json'))
            result = metrics(self.factory, rows)
            r = _read_json(root/(name+'_metrics.json'))
            require(_same(r,history[epoch]) and type(r['epoch']) is int and r['epoch']==epoch
                    and r['split']=='validation', 'history identity')
            require(all(_same(r[k],v) for k,v in result.items()), 'independent validation metrics')
            counts={t:len(self.factory.loaders['train'][t]) for t in TASKS}
            train=r['train']
            expected_counts=dict(epoch=epoch,updates=sum(counts.values()),cumulative_updates=q['optimizer_updates'],
                                 task_batches=counts,task_samples=self.factory.contract['counts']['train'])
            require(all(_same(train[k],v) for k,v in expected_counts.items()), 'training counts')
            require(set(train['loss'])==set(TASKS) and all(type(v) in (int,float) and math.isfinite(v)
                    for v in train['loss'].values()), 'finite train loss')
            # Reconstruct the control record from the checkpoint, not the report.
            trainer.model.load_state_dict(state, strict=True)
            trainer.optimizer.load_state_dict(q['optimizer'])
            trainer.optimization.epoch=epoch
            for _,p in trainer.optimization.backbone:
                p.requires_grad_(epoch>=trainer.optimization.spec.warmup_epochs)
            require(_same(train['optimization'],trainer.optimization.record()), 'optimization record')
            checked.append(dict(epoch=epoch, split='validation', endpoints=result['endpoints']))
        best=min(range(40),key=lambda i:history[i]['macro_rmse'])
        summary=_read_json(root/'summary.json')
        best_rows=_read_json(root/f'epoch_{best:03d}_validation.json')
        expected=dict(identity=self.identity,epochs_completed=40,best_epoch=best,
                      best_validation=metrics(self.factory,best_rows),
                      best_checkpoint_sha256=file_sha(root/f'epoch_{best:03d}.pt'),
                      optimizer_updates=q['optimizer_updates'])
        require(_same(summary,expected), 'summary/earliest best')
        q=torch.load(root/f'epoch_{best:03d}.pt',map_location='cpu',weights_only=True)
        trainer.model.load_state_dict(q['model_state'],strict=True)
        require(validation_rows(self.factory,trainer,best)==best_rows, 'fresh best prediction replay')
        return dict(identity=self.identity, history=checked, best_epoch=best,
                    best_validation=expected['best_validation'], optimizer_updates=expected['optimizer_updates'],
                    validation_status='PASS', acceptance_status='PENDING_REVIEW')

    def _verify_steps(self, trainer, q, epoch):
        control=trainer.optimization
        control.validate_optimizer_state(q['optimizer'],epoch,q['model_state'])
        counts={t:len(self.factory.loaders['train'][t]) for t in TASKS}
        updates=sum(counts.values())
        require(type(q['optimizer_updates']) is int and q['optimizer_updates']==updates*(epoch+1), 'update count')
        ordered=(list(trainer.model.named_parameters()) if self.arm=='B1_high' else control.backbone+control.heads)
        ids=[i for group in q['optimizer']['param_groups'] for i in group['params']]
        for index,(name,parameter) in zip(ids,ordered):
            item=q['optimizer']['state'].get(index)
            if name.startswith('decoders.'):
                task=name.split('.')[1]
                steps=counts[task]*(epoch+1)
            else:
                steps=updates*max(0,epoch+1-control.spec.warmup_epochs)
                # Path-mode Graphormer keeps these unused direct-mode parameters.
                if name.startswith('encoder.backbone.direct_bond_embeddings.'):
                    steps=0
            if steps==0:
                require(item is None and torch.equal(q['model_state'][name],control.initial_backbone[name]),
                        'inactive backbone state/tensor')
            else:
                require(item is not None and float(item['step'])==steps, 'continuous Adam step '+name)
