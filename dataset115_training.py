"""Route B epoch training engine. No CLI, test/calibration access, or authorization.

The future execution card must freeze configuration and budget. Epoch snapshots
are immutable; a resumed run writes a NEW output directory, never old evidence.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch

from dataset115_adapter import GraphTaskView, TrainOnlyScaler
from dataset115_contract import ContractError, PRIMARY, semantic_digest
from dataset115_model import build_human5_model
from dataset115_smoke import state_digest


def require(ok, message):
    if not ok:
        raise ContractError(message)


@dataclass(frozen=True)
class EpochConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float

    def __post_init__(self):
        for key in ('epochs', 'batch_size'):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, key)
        for key in ('learning_rate', 'weight_decay', 'grad_clip'):
            x = getattr(self, key)
            require(type(x) in (int, float) and math.isfinite(x) and x >= 0, key)
        require(self.learning_rate > 0 and self.grad_clip > 0, 'lr/clip must be positive')


def task_batches(counts, batch_size, seed, epoch):
    """Exactly one pass over every valid observation; proportional task batches.

    Sample shuffles and schedule use isolated generators, identical across
    methods with a common seed. No oversampling, empty batches, or drop_last.
    """
    require(set(counts) == set(PRIMARY) and all(type(n) is int and n > 0 for n in counts.values()), 'task counts')
    require(type(batch_size) is int and batch_size > 0, 'batch size')
    batches = {}
    for j, task in enumerate(PRIMARY):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, j])).permutation(counts[task]).tolist()
        batches[task] = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
    schedule = [(t, k) for t in PRIMARY for k in range(len(batches[t]))]
    np.random.default_rng(seed + epoch).shuffle(schedule)
    return [(t, batches[t][k]) for t, k in schedule]


def validation_metrics(rows):
    """Original published-label scale; unweighted mean of five endpoint RMSEs."""
    out = {}; seen = set()
    require(bool(rows), 'empty validation')
    for row in rows:
        require(row['task'] in PRIMARY and row['split'] == 'validation', 'validation scope')
        key = (row['task'], row['sample_id'])
        require(key not in seen, 'duplicate prediction'); seen.add(key)
        require(all(math.isfinite(row[k]) for k in ('label', 'prediction')), 'nonfinite prediction')
    for t in PRIMARY:
        errors = np.asarray([r['prediction']-r['label'] for r in rows if r['task'] == t], dtype=np.float64)
        require(len(errors) > 0, 'missing validation task')
        out[t] = {'n': len(errors), 'rmse': float(np.sqrt(np.mean(errors**2))), 'mae': float(np.mean(abs(errors)))}
    return {'endpoints': out, 'macro_rmse': float(np.mean([out[t]['rmse'] for t in PRIMARY])),
            'macro_mae': float(np.mean([out[t]['mae'] for t in PRIMARY]))}


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf8') as f:
        f.write(text)


class RouteBTrainer:
    def __init__(self, args, source, source_identity, train, validation, *, method, seed, config, device):
        require(train.route == validation.route == 'B' and train.role == validation.role == 'target', 'Route B target required')
        require(train.split == 'train' and validation.split == 'validation', 'train/validation only')
        require(train.input_identity == validation.input_identity, 'input identity')
        for attr in ('sample_ids', 'canonical', 'groups'):
            require(not set(getattr(train, attr)) & set(getattr(validation, attr)), 'train/validation leakage: '+attr)
        require(type(seed) is int and seed in range(42, 47), 'seed scope')
        require(device in ('cpu', 'cuda:0'), 'device scope')
        require(isinstance(config, EpochConfig), 'config type')
        if method == 'B0':
            require(source is None and source_identity is None, 'B0 source forbidden')
        else:
            require(isinstance(source_identity, dict) and source_identity.get('seed') == seed, 'same-seed source identity')
            for key in ('teacher_sha256', 'init_sha256'):
                v = source_identity.get(key)
                require(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v), 'source digest')
        self.seed, self.config, self.device, self.method = seed, config, device, method
        self.train, self.validation = train, validation
        self.scaler = TrainOnlyScaler.fit(train)
        self.scaler.transform(validation)  # identity validation, no refit
        self.scalers = self.scaler.trainer_scalers()
        self.model = build_human5_model(args, method=method, seed=seed, source_encoder_state=source).to(device)
        self.initial_encoder = state_digest(self.model.encoder.state_dict())
        self.initial_heads = state_digest(self.model.decoders.state_dict())
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate, weight_decay=config.weight_decay)
        self.datasets = {split: {t: GraphTaskView(v, t) for t in PRIMARY}
                         for split, v in [('train', train), ('validation', validation)]}
        require(all(len(d) for ds in self.datasets.values() for d in ds.values()), 'empty task')
        self.cache = {}
        self.identity = dict(schema='dataset115_route_b_epoch_v1', method=method, seed=seed,
            config=asdict(config), architecture=vars(args), source_identity=source_identity,
            input_identity=train.input_identity, train_ids=semantic_digest(list(train.sample_ids)),
            validation_ids=semantic_digest(list(validation.sample_ids)), scaler=self.scaler.to_dict(),
            train_observations=self._view_digest(train), validation_observations=self._view_digest(validation),
            task_names=list(PRIMARY), selection='validation_primary5_macro_rmse_strict_min_first_tie',
            sampling='one_pass_shuffled_task_batches_proportional', tasks_per_update=1,
            loss='QuantileRegressionLoss_EW_single_active_task', torch_version=str(torch.__version__), device=device)
        self.identity_sha = semantic_digest(self.identity)
        self.history = []; self.best_state = None; self.best_rows = None; self.best_epoch = None
        self.total_steps = 0
        torch.manual_seed(seed)

    @staticmethod
    def _view_digest(v):
        return semantic_digest([[v.sample_ids[i], v.canonical[i], v.groups[i],
            [None if np.isnan(x) else float(x) for x in v.labels[i]]] for i in range(len(v.sample_ids))])

    def _batch(self, split, task, indices):
        from dataset import DataCollator
        ds = self.datasets[split][task]; graphs = []
        for i in indices:
            key = (split, task, i)
            if key not in self.cache: self.cache[key] = ds[i]
            graphs.append(self.cache[key])
        batch = DataCollator()(graphs)
        require(not batch.is_empty and batch.y.numel() == len(indices), 'graph dropped from batch')
        require(torch.isfinite(batch.y).all(), 'nonfinite batch labels')
        return batch.to(self.device)

    def evaluate_validation(self):
        self.model.eval(); rows = []
        with torch.no_grad():
            for t in PRIMARY:
                ds = self.datasets['validation'][t]; s = self.scalers[t]
                for start in range(0, len(ds), self.config.batch_size):
                    ids = list(range(start, min(len(ds), start+self.config.batch_size)))
                    batch = self._batch('validation', t, ids)
                    raw = self.model(batch, task_name=t)[t]
                    require(tuple(raw.shape) == (len(ids), 3) and torch.isfinite(raw).all(), 'prediction shape/finite')
                    predicted = (raw[:, 0].double()*s['std']+s['mean']).cpu().tolist()
                    for i,pred in zip(ids, predicted):
                        j = ds.indices[i]
                        rows.append(dict(task=t, split='validation', sample_id=ds.get_sample_id(i),
                            canonical=self.validation.canonical[j], group=self.validation.groups[j],
                            label=float(self.validation.labels[j, self.validation.tasks.index(t)]), prediction=pred))
        return rows, validation_metrics(rows)

    def _restore(self, path, expected_sha):
        raw = Path(path).read_bytes()
        require(isinstance(expected_sha, str) and hashlib.sha256(raw).hexdigest() == expected_sha, 'resume file SHA')
        import io
        p = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        require(p['identity'] == self.identity and p['identity_sha256'] == self.identity_sha, 'resume contract')
        require(p['initial_encoder'] == self.initial_encoder and p['initial_heads'] == self.initial_heads, 'resume initial model identity')
        history = p['history']
        require(history and [h['epoch'] for h in history] == list(range(p['epoch']+1)), 'resume history')
        require(all(math.isfinite(h['validation']['macro_rmse']) for h in history), 'resume metric finite')
        best = min(range(len(history)), key=lambda i: history[i]['validation']['macro_rmse'])
        require(p['best_epoch'] == best, 'resume best selection')
        require(validation_metrics(p['best_rows']) == history[best]['validation'], 'resume best metrics')
        require(state_digest(p['model_state']) == p['model_digest'] and state_digest(p['best_model_state']) == p['best_model_digest'], 'resume tensor digest')
        require(all(torch.isfinite(v).all() for state in [p['model_state'],p['best_model_state']] for v in state.values()), 'resume finite weights')
        self.model.load_state_dict(p['model_state'], strict=True)
        if self.method == 'RPT': require(state_digest(self.model.encoder.state_dict()) == self.initial_encoder, 'resume frozen encoder')
        self.optimizer.load_state_dict(p['optimizer_state'])
        self.history = history; self.best_epoch = best; self.best_state = p['best_model_state']; self.best_rows = p['best_rows']
        self.total_steps = p['total_steps']
        expected_steps = sum(len(task_batches({t:len(d) for t,d in self.datasets['train'].items()}, self.config.batch_size, self.seed,e)) for e in range(len(history)))
        require(type(self.total_steps) is int and self.total_steps == expected_steps, 'resume steps')
        torch.set_rng_state(p['cpu_rng'])
        if self.device == 'cuda:0': torch.cuda.set_rng_state(p['cuda_rng'], device=0)
        return p['epoch'] + 1

    def run(self, output, *, stop_after=None, resume=None, resume_sha=None):
        """stop_after is an absolute epoch limit for ENGINE TESTS; not authorization."""
        output = Path(output)
        require(not output.exists(), 'output exists')
        require(not self.history, 'trainer already used; create a fresh instance')
        limit = self.config.epochs if stop_after is None else stop_after
        require(type(limit) is int and 0 < limit <= self.config.epochs, 'epoch budget')
        require((resume is None) == (resume_sha is None), 'resume path/SHA pair')
        start = 0 if resume is None else self._restore(resume, resume_sha)
        require(start < limit, 'no remaining epochs')
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output/'resolved_config.json', self.identity)
        from loss import QuantileRegressionLoss
        loss_fn = QuantileRegressionLoss(); checkpoints = []
        counts = {t:len(d) for t,d in self.datasets['train'].items()}
        for epoch in range(start, limit):
            self.model.train(); losses = {t:[] for t in PRIMARY}; exposure = {t:0 for t in PRIMARY}
            max_norm = 0.0
            for t,indices in task_batches(counts, self.config.batch_size, self.seed, epoch):
                batch = self._batch('train', t, indices); s = self.scalers[t]
                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_fn.compute_loss(self.model(batch, task_name=t)[t], (batch.y.reshape(-1,1)-s['mean'])/s['std'])
                require(torch.isfinite(loss), 'nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.config.grad_clip, error_if_nonfinite=True)
                if self.method == 'RPT': require(all(p.grad is None for p in self.model.encoder.parameters()), 'frozen gradient leak')
                self.optimizer.step(); self.total_steps += 1
                max_norm = max(max_norm, float(norm)); losses[t].append(float(loss.detach())); exposure[t] += len(indices)
            require(exposure == counts, 'training exposure differs')
            require(all(torch.isfinite(v).all() for v in self.model.state_dict().values()), 'nonfinite weights')
            if self.method == 'RPT': require(state_digest(self.model.encoder.state_dict()) == self.initial_encoder, 'frozen encoder changed')
            rows, metrics = self.evaluate_validation()
            if self.best_epoch is None or metrics['macro_rmse'] < self.history[self.best_epoch]['validation']['macro_rmse']:
                self.best_epoch = epoch; self.best_state = _cpu_state(self.model); self.best_rows = rows
            self.history.append(dict(epoch=epoch, total_steps=self.total_steps, exposure=exposure,
                mean_batch_loss={t:float(np.mean(losses[t])) for t in PRIMARY}, max_grad_norm=max_norm, validation=metrics))
            _write_json(output/f'validation_epoch_{epoch:03d}.json', rows)
            state = _cpu_state(self.model)
            p = dict(identity=self.identity, identity_sha256=self.identity_sha, epoch=epoch,
                model_state=state, model_digest=state_digest(state), optimizer_state=self.optimizer.state_dict(),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0) if self.device=='cuda:0' else None,
                history=self.history, total_steps=self.total_steps, initial_encoder=self.initial_encoder,
                initial_heads=self.initial_heads, best_epoch=self.best_epoch, best_model_state=self.best_state,
                best_model_digest=state_digest(self.best_state), best_rows=self.best_rows)
            path = output/f'epoch_{epoch:03d}.pt'
            with path.open('xb') as f: torch.save(p, f)
            checkpoints.append(dict(epoch=epoch, path=path.name, size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            _write_json(output/f'epoch_{epoch:03d}.receipt.json', checkpoints[-1])
        # Independently recompute the chosen validation after loading chosen weights.
        self.model.load_state_dict(self.best_state, strict=True)
        best_rows, best_metrics = self.evaluate_validation()
        require(best_rows == self.best_rows and best_metrics == self.history[self.best_epoch]['validation'], 'best replay differs')
        _write_json(output/'best_validation.json', best_rows)
        summary = dict(identity_sha256=self.identity_sha, completed_epochs=limit, planned_epochs=self.config.epochs,
            complete=limit==self.config.epochs, best_epoch=self.best_epoch, best_validation=best_metrics,
            checkpoints=checkpoints, inherited_checkpoint_sha256=resume_sha, history=self.history,
            test_predictions_accessed=False, calibration_predictions_accessed=False, acceptance_status='PENDING_REVIEW')
        _write_json(output/'training_summary.json', summary)
        return summary
