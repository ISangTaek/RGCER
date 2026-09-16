"""Route A source epochs and target adaptation; no holdout/test entry.

Source is selected by fixed final epoch, never a Human5 score. All executable
budget authorization lives in the release CLI/card, not in this engine.
"""
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import io
import json
import math

import numpy as np
import torch

from dataset115_adapter import GraphTaskView, TrainOnlyScaler
from dataset115_contract import digest, semantic_digest
from dataset115_route_a_smoke import build_source, check_source_tasks
from dataset115_smoke import state_digest
from dataset115_training import EpochConfig, RouteBTrainer, _cpu_state, _write_json, require

TASK_ID = 'S5F2_ROUTEA_20_TRAIN_VALIDATION_20260916'
CONFIG = EpochConfig(40, 32, .001, 1e-5, 1.)


def source_batches(tasks, counts, batch_size, seed, epoch):
    check_source_tasks(tasks)
    require(set(counts) == set(tasks) and all(type(n) is int and n > 0 for n in counts.values()), 'source counts')
    require(type(batch_size) is int and batch_size > 0, 'source batch size')
    require(type(seed) is int and seed in range(42, 47) and type(epoch) is int and epoch >= 0, 'source schedule identity')
    batches = {}
    for j, task in enumerate(tasks):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, j])).permutation(counts[task]).tolist()
        batches[task] = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
    schedule = [(t, k) for t in tasks for k in range(len(batches[t]))]
    np.random.default_rng(seed + epoch).shuffle(schedule)
    return [(t, batches[t][k]) for t, k in schedule]


def read_json(path):
    def bad(value): raise ValueError('nonfinite JSON '+value)
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    return json.loads(Path(path).read_bytes(), parse_constant=bad, object_pairs_hook=pairs)


class SourceTrainer:
    def __init__(self, args, train, *, seed, config=CONFIG, device='cpu'):
        require(train.route == 'A' and train.role == 'source' and train.split == 'train', 'Route A source train only')
        check_source_tasks(train.tasks)
        require(device in ('cpu', 'cuda:0') and isinstance(config, EpochConfig), 'source runtime/config')
        self.train, self.seed, self.config, self.device = train, seed, config, device
        self.tasks = train.tasks
        self.scaler = TrainOnlyScaler.fit(train)
        self.scalers = self.scaler.trainer_scalers()
        self.datasets = {t: GraphTaskView(train, t) for t in self.tasks}
        self.counts = {t: len(ds) for t, ds in self.datasets.items()}
        require(all(n > 0 for n in self.counts.values()), 'empty source task')
        self.model = build_source(args, self.tasks, seed).to(device)
        self.initial_digest = state_digest(self.model.state_dict())
        self.initial_encoder = state_digest(self.model.encoder.state_dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # Bounded cache: do not retain 75k per-observation graphs in each GPU worker.
        self.cache = OrderedDict()
        self.identity = dict(schema='dataset115_route_a_source_epoch_v1', seed=seed,
            config=asdict(config), architecture=vars(args), task_names=list(self.tasks),
            input_identity=train.input_identity, train_ids=semantic_digest(list(train.sample_ids)),
            train_observations=RouteBTrainer._view_digest(train), scaler=self.scaler.to_dict(),
            selection='fixed_final_epoch', sampling='one_pass_shuffled_task_batches_proportional',
            tasks_per_update=1, loss='QuantileRegressionLoss_EW_single_active_task',
            torch_version=str(torch.__version__), device=device, graph_cache_max_entries=256)
        self.identity_sha = semantic_digest(self.identity)
        self.used = False
        torch.manual_seed(seed)

    def batch(self, task, indices):
        from dataset import DataCollator
        ds = self.datasets[task]; graphs = []
        for i in indices:
            key = (task, i)
            if key not in self.cache: self.cache[key] = ds[i]
            self.cache.move_to_end(key); graphs.append(self.cache[key])
            while len(self.cache) > 256: self.cache.popitem(last=False)
        batch = DataCollator()(graphs)
        require(not batch.is_empty and batch.y.numel() == len(indices) and torch.isfinite(batch.y).all(), 'source graph/label dropped')
        return batch.to(self.device)

    def probe(self):
        """Reload check on fixed TRAIN observations, not a selection score."""
        self.model.eval(); rows = []
        with torch.no_grad():
            for task, ds in self.datasets.items():
                indices = list(range(min(2, len(ds))))
                pred = self.model(self.batch(task, indices), task_name=task)[task]
                require(tuple(pred.shape) == (len(indices), 3) and torch.isfinite(pred).all(), 'source reload probe finite/shape')
                rows.append(dict(task=task, sample_ids=[ds.get_sample_id(i) for i in indices],
                                 standardized_prediction=pred.cpu().tolist()))
        return rows

    def run(self, output):
        output = Path(output); require(not output.exists() and not self.used, 'source output exists/runner used')
        self.used = True; output.mkdir(parents=True, exist_ok=False)
        _write_json(output/'resolved_config.json', self.identity)
        from loss import QuantileRegressionLoss
        loss_fn = QuantileRegressionLoss(); history = []; receipts = []; steps = 0
        for epoch in range(self.config.epochs):
            self.model.train(); exposure = dict.fromkeys(self.tasks, 0); losses = {t: [] for t in self.tasks}; max_norm = 0.
            for task, indices in source_batches(self.tasks, self.counts, self.config.batch_size, self.seed, epoch):
                b = self.batch(task, indices); s = self.scalers[task]
                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_fn.compute_loss(self.model(b, task_name=task)[task], (b.y.reshape(-1, 1)-s['mean'])/s['std'])
                require(torch.isfinite(loss), 'source loss nonfinite'); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True)
                self.optimizer.step(); steps += 1; exposure[task] += len(indices)
                losses[task].append(float(loss.detach())); max_norm = max(max_norm, float(norm))
            require(exposure == self.counts, 'source exposure')
            state = _cpu_state(self.model)
            require(all(torch.isfinite(v).all() for v in state.values()), 'source nonfinite state')
            history.append(dict(epoch=epoch, exposure=exposure, total_steps=steps,
                mean_batch_loss={t: float(np.mean(losses[t])) for t in self.tasks}, max_grad_norm=max_norm))
            payload = dict(identity=self.identity, identity_sha256=self.identity_sha, epoch=epoch,
                model_state=state, model_digest=state_digest(state), optimizer_state=self.optimizer.state_dict(),
                initial_digest=self.initial_digest, initial_encoder=self.initial_encoder, history=list(history),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0) if self.device=='cuda:0' else None)
            path = output/f'epoch_{epoch:03d}.pt'
            with path.open('xb') as f: torch.save(payload, f)
            receipt = dict(epoch=epoch, path=path.name, size_bytes=path.stat().st_size, sha256=digest(path))
            receipts.append(receipt); _write_json(output/f'epoch_{epoch:03d}.receipt.json', receipt)
        summary = dict(task_id=TASK_ID, identity_sha256=self.identity_sha, completed_epochs=self.config.epochs,
            selected_epoch=self.config.epochs-1, selection='fixed_final_epoch', checkpoints=receipts, history=history,
            human_validation_accessed=False, test_predictions_accessed=False, calibration_predictions_accessed=False,
            acceptance_status='PENDING_REVIEW')
        _write_json(output/'train_reload_probe.json', self.probe())
        _write_json(output/'training_summary.json', summary)
        return summary


def verify_source(output, trainer):
    """Bind all epochs to a fresh trainer built from actual frozen train input."""
    output = Path(output); summary = read_json(output/'training_summary.json')
    require(read_json(output/'resolved_config.json') == trainer.identity, 'source config identity')
    require(summary['task_id'] == TASK_ID and summary['identity_sha256'] == trainer.identity_sha, 'source summary identity')
    epochs = trainer.config.epochs
    require(type(summary['completed_epochs']) is int and summary['completed_epochs'] == epochs
        and summary['selected_epoch'] == epochs-1 and summary['selection']=='fixed_final_epoch', 'source epoch/selection')
    require(all(summary[k] is False for k in ('human_validation_accessed','test_predictions_accessed','calibration_predictions_accessed')), 'source holdout scope')
    require(len(summary['history']) == len(summary['checkpoints']) == epochs, 'source epoch matrix')
    nsteps = len(source_batches(trainer.tasks, trainer.counts, trainer.config.batch_size, trainer.seed, 0))
    names = list(dict(trainer.model.named_parameters()))
    reference = trainer.model.state_dict(); previous_digest = trainer.initial_digest
    for epoch, (h, receipt) in enumerate(zip(summary['history'], summary['checkpoints'])):
        require(type(h['epoch']) is int and h['epoch']==receipt['epoch']==epoch, 'source epoch order')
        require(h['exposure']==trainer.counts and h['total_steps']==nsteps*(epoch+1), 'source exposure/steps')
        require(set(h['mean_batch_loss'])==set(trainer.tasks) and all(type(v) in (float,int) and math.isfinite(v) for v in h['mean_batch_loss'].values())
            and math.isfinite(h['max_grad_norm']), 'source numerical history')
        name = f'epoch_{epoch:03d}.pt'; require(receipt['path']==name, 'source path')
        require(read_json(output/f'epoch_{epoch:03d}.receipt.json')==receipt, 'source receipt')
        raw = (output/name).read_bytes()
        import hashlib
        require(len(raw)==receipt['size_bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256'], 'source file SHA')
        p = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        require(p['identity']==trainer.identity and p['identity_sha256']==trainer.identity_sha and p['epoch']==epoch, 'source checkpoint identity')
        require(p['initial_digest']==trainer.initial_digest and p['initial_encoder']==trainer.initial_encoder, 'source initialization')
        require(p['history']==summary['history'][:epoch+1], 'source checkpoint history')
        state = p['model_state']; require(set(state)==set(reference), 'source tensor keys')
        require(all(v.shape==reference[k].shape and v.dtype==reference[k].dtype and torch.isfinite(v).all() for k,v in state.items()), 'source tensor specs')
        require(state_digest(state)==p['model_digest']!=previous_digest, 'source unchanged/corrupt tensors')
        previous_digest = p['model_digest']
        opt = p['optimizer_state']; groups = opt['param_groups']; require(len(groups)==1, 'source optimizer groups')
        g=groups[0]
        require(g['params']==list(range(len(names))) and g['lr']==trainer.config.learning_rate and g['weight_decay']==trainer.config.weight_decay
            and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8, 'source optimizer config')
        for i, param in enumerate(names):
            item = opt['state'].get(i)
            if param.startswith('decoders.'):
                task=param.split('.')[1]; expected=math.ceil(trainer.counts[task]/trainer.config.batch_size)*(epoch+1)
                require(item is not None and float(item['step'])==expected, 'source head optimizer steps')
            elif item is not None: require(float(item['step'])==nsteps*(epoch+1), 'source encoder optimizer steps')
            if item is not None: require(all(torch.isfinite(v).all() for v in item.values() if isinstance(v,torch.Tensor)), 'source optimizer finite')
    trainer.model.load_state_dict(p['model_state'],strict=True)
    require(trainer.probe()==read_json(output/'train_reload_probe.json'), 'source fresh reload forward')
    encoder={k[len('encoder.'):]:v for k,v in p['model_state'].items() if k.startswith('encoder.')}
    return dict(task_id=TASK_ID, content_status='PASS', acceptance_status='PENDING_REVIEW', seed=trainer.seed,
        epochs_checked=epochs, selected_epoch=epochs-1, identity_sha256=trainer.identity_sha,
        source_sha256=summary['checkpoints'][-1]['sha256'], encoder_sha256=state_digest(encoder),
        total_steps=nsteps*epochs, source_tasks=list(trainer.tasks), formal_source=trainer.config==CONFIG)


def load_source(output, trainer):
    """Full content re-verification before target may consume a source file."""
    receipt = verify_source(output, trainer)
    require(receipt['formal_source'] is True, 'nonformal source cannot initialize formal target')
    p=torch.load(Path(output)/f'epoch_{trainer.config.epochs-1:03d}.pt',map_location='cpu',weights_only=True)
    encoder={k[len('encoder.'):]:v for k,v in p['model_state'].items() if k.startswith('encoder.')}
    return encoder, dict(seed=trainer.seed, teacher_sha256=receipt['source_sha256'],
        init_sha256=receipt['encoder_sha256'], route='A', selection='fixed_final_epoch39',
        source_identity_sha256=trainer.identity_sha), receipt


class RouteATrainer(RouteBTrainer):
    ROUTE = 'A'

    def __init__(self, args, source, source_identity, train, validation, **kwargs):
        if kwargs.get('method') != 'B0':
            require(isinstance(source_identity,dict) and source_identity.get('route')=='A'
                and source_identity.get('selection')=='fixed_final_epoch39', 'Route A formal source binding')
            require(source is not None and state_digest(source)==source_identity.get('init_sha256'), 'Route A source encoder digest')
            sha=source_identity.get('source_identity_sha256')
            require(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha), 'Route A source contract digest')
        super().__init__(args,source,source_identity,train,validation,**kwargs)
