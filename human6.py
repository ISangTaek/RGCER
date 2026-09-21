"""Human6-only Route A adaptation. No test/calibration or source training API.

Reuses the existing Graphormer, graph conversion, loss, scaler and P1D optimizer.
Old Human5 globals/contracts are never changed. CLI policy is committed separately.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import math
import os

import numpy as np
import torch
from torch import nn

from architecture.Graphormer import Encoder, Graphormer
from architecture.prediction_heads import TaskPredictionHead
from baselines.scaling import TaskScaler
from dataset115_adapter import GraphTaskView
from dataset115_contract import digest, label, manifest_records, semantic_digest, task_columns
from dataset115_smoke import state_digest
from p1d_optimization import OptimizationControl, OptimizationSpec

TASKS = ('child_oral_LDLo', 'women_oral_LDLo', 'man_oral_LDLo',
         'human_intravenous_TDLo', 'man_intravenous_TDLo', 'man_unreported_LDLo')
METHODS = ('FROZEN', 'B1_low', 'HF_low')
POLICY_PATH = Path(__file__).parent / 'configs/human6_route_a.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def policy():
    p = json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    require(p['status'] == 'APPROVED_2026-09-21' and p['training_authorized'] is True
            and p['test_authorized'] is False and tuple(p['task_order']) == TASKS, 'policy scope')
    return p


def json_write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def save_state(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.pt.tmp')
    with tmp.open('xb') as f:
        torch.save(obj, f)
    os.replace(tmp, path)


@dataclass(frozen=True)
class View:
    split: str
    sample_ids: tuple
    smiles: tuple
    canonical: tuple
    groups: tuple
    labels: np.ndarray
    input_identity: str
    tasks: tuple = TASKS

    def __post_init__(self):
        require(self.split in ('train', 'validation') and self.tasks == TASKS, 'Human6 train/validation only')
        n = len(self.sample_ids)
        require(len(set(self.sample_ids)) == n and all(str(s).startswith('dataset115:row_') for s in self.sample_ids), 'sample identity')
        require(all(len(x) == n for x in (self.smiles, self.canonical, self.groups)), 'metadata shape')
        require(all(isinstance(s, str) and s for seq in (self.smiles, self.canonical, self.groups) for s in seq), 'empty identity')
        y = np.array(self.labels, dtype=np.float64, copy=True)
        require(y.shape == (n, 6) and not np.isinf(y).any(), 'label shape/Inf')
        y.setflags(write=False)
        object.__setattr__(self, 'labels', y)


def load_views(csv_path, split_path, p):
    require(digest(csv_path) == p['data']['csv_sha256'] and digest(split_path) == p['data']['split_sha256'], 'input SHA')
    manifest = json.loads(Path(split_path).read_text(encoding='utf-8'))
    require(manifest['source_csv_sha256'] == p['data']['csv_sha256'], 'manifest source')
    refs = manifest_records(manifest)
    result = {s: ([], []) for s in ('train', 'validation')}
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        task_columns(reader.fieldnames)
        n = 0
        for i, row in enumerate(reader):
            require(i in refs and row['smiles'] == refs[i]['raw_smiles'], 'CSV row mapping')
            require(None not in row and all(v is not None for v in row.values()), 'CSV width')
            n += 1
            split = refs[i]['split']
            if split not in result:
                continue  # DO NOT numerically parse any held-out or non-Human6 label.
            values = [np.nan if (v := label(row[t])) is None else v for t in TASKS]
            if not np.isfinite(values).any():
                continue
            result[split][0].append(refs[i])
            result[split][1].append(values)
    require(n == len(refs), 'CSV row count')
    require(digest(csv_path) == p['data']['csv_sha256'] and digest(split_path) == p['data']['split_sha256'], 'input changed')
    views = {}
    for split, (rows, values) in result.items():
        views[split] = View(split, tuple('dataset115:' + r['sample_id'] for r in rows),
            tuple(r['raw_smiles'] for r in rows), tuple(r['canonical_smiles'] for r in rows),
            tuple(r['split_group'] for r in rows), np.asarray(values), semantic_digest(p['data']))
        got = dict(zip(TASKS, np.isfinite(views[split].labels).sum(axis=0).tolist()))
        require(got == p['counts'][split], 'Human6 observation counts')
    require(not set(views['train'].groups) & set(views['validation'].groups), 'group overlap')
    return views


def schedule(counts, seed, epoch, batch_size=32):
    require(set(counts) == set(TASKS) and all(type(v) is int and v > 0 for v in counts.values()), 'task counts')
    batches = {}
    for j, t in enumerate(TASKS):
        ids = np.random.default_rng(np.random.SeedSequence([seed, epoch, j])).permutation(counts[t]).tolist()
        batches[t] = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
    order = [(t, k) for t in TASKS for k in range(len(batches[t]))]
    np.random.default_rng(seed + epoch).shuffle(order)
    return [(t, batches[t][k]) for t, k in order]


def metrics(rows, view=None):
    seen = set()
    truth = None
    if view is not None:
        truth = {(t, sid): (float(view.labels[i, j]), view.canonical[i], view.groups[i])
                 for i, sid in enumerate(view.sample_ids) for j, t in enumerate(TASKS)
                 if np.isfinite(view.labels[i, j])}
    errors = {t: [] for t in TASKS}
    for r in rows:
        key = (r['task'], r['sample_id'])
        require(r['task'] in TASKS and r['split'] == 'validation' and key not in seen, 'prediction identity')
        require(all(math.isfinite(r[k]) for k in ('label', 'prediction')), 'nonfinite prediction')
        if truth is not None:
            require(key in truth and (r['label'], r['canonical'], r['group']) == truth[key], 'prediction population/truth')
        seen.add(key)
        errors[r['task']].append(r['prediction'] - r['label'])
    require(truth is None or seen == set(truth), 'prediction population incomplete')
    require(all(errors.values()), 'missing endpoint')
    end = {t: dict(n=len(e), rmse=float(np.sqrt(np.mean(np.square(e)))), mae=float(np.mean(np.abs(e)))) for t, e in errors.items()}
    return dict(endpoints=end, macro_rmse=sum(x['rmse'] for x in end.values())/6,
                macro_mae=sum(x['mae'] for x in end.values())/6)


def source_asset(p, seed, source_root=None, tensors=True):
    a = next(x for x in p['source_assets'] if x['seed'] == seed)
    path = (Path(a['server_path']) if source_root is None else
            Path(source_root)/f'source_seed{seed}/epoch_039.pt')
    require(path.is_file() and path.stat().st_size == a['size_bytes'] and digest(path) == a['sha256'], 'source file identity')
    if not tensors:
        return None, a
    payload = torch.load(path, map_location='cpu', weights_only=True)
    ident = payload['identity']
    require(payload['epoch'] == 39 and ident['seed'] == seed and ident['architecture'] == p['architecture'], 'source epoch/architecture')
    require(payload['identity_sha256'] == a['source_identity_sha256'] == semantic_digest(ident), 'source contract')
    require(len(ident['task_names']) == len(set(ident['task_names'])) == 104 and
            all(t.rsplit('_', 2)[0] not in {'child', 'human', 'man', 'women'} for t in ident['task_names']), 'source human supervision')
    state = {k[len('encoder.'):]: v for k, v in payload['model_state'].items() if k.startswith('encoder.')}
    require(state_digest(state) == a['encoder_state_sha256'], 'source encoder digest')
    return state, a


def model_from_source(architecture, state, seed):
    a = SimpleNamespace(**architecture)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        heads = nn.ModuleDict({t: TaskPredictionHead(a.hidden_dim, mode='quantile',
                    head_hidden_dim=a.head_hidden_dim, dropout=a.head_dropout) for t in TASKS})
        model = Graphormer(list(TASKS), Encoder, heads, torch.device('cpu'), a)
    require(getattr(model, 'card', None) is None, 'CARD forbidden')
    expected = model.encoder.state_dict()
    require(set(state) == set(expected) and all(v.shape == expected[k].shape and v.dtype == expected[k].dtype
            and torch.isfinite(v).all() for k, v in state.items()), 'source encoder shape/finite')
    model.encoder.load_state_dict(state, strict=True)
    return model


class Trainer:
    def __init__(self, views, source, architecture, *, method, seed, device, policy_sha, source_sha):
        require(method in METHODS and type(seed) is int and seed in range(42, 47), 'method/seed')
        require(device in ('cpu', 'cuda:0'), 'single-device scope')
        require(set(views) == {'train', 'validation'}, 'train/validation only')
        require(views['train'].input_identity == views['validation'].input_identity, 'input identity differs')
        self.views, self.method, self.seed, self.device = views, method, seed, device
        for split, v in views.items():
            require(v.split == split and v.tasks == TASKS, 'view scope')
        for key in ('sample_ids', 'canonical', 'groups'):
            require(not set(getattr(views['train'], key)) & set(getattr(views['validation'], key)), 'split leakage')
        self.scaler = TaskScaler.fit(views['train'].labels, TASKS, allow_empty=False)
        self.model = model_from_source(architecture, source, seed).to(device)
        self.initial_encoder = state_digest(self.model.encoder.state_dict())
        self.initial_heads = state_digest(self.model.decoders.state_dict())
        self.control = None
        if method == 'FROZEN':
            for v in self.model.encoder.parameters():
                v.requires_grad_(False)
            self.optimizer = torch.optim.AdamW(self.model.decoders.parameters(), lr=.001, weight_decay=1e-5)
        else:
            self.control = OptimizationControl(self.model, OptimizationSpec(method))
            self.optimizer = self.control.optimizer
        self.datasets = {s: {t: GraphTaskView(v, t) for t in TASKS} for s, v in views.items()}
        self.counts = {t: len(d) for t, d in self.datasets['train'].items()}
        require(all(len(d) for ds in self.datasets.values() for d in ds.values()), 'empty endpoint')
        self.identity = dict(method=method, seed=seed, policy_sha=policy_sha, source_sha=source_sha,
            architecture=architecture, tasks=list(TASKS), scaler=self.scaler.to_dict(),
            initial_encoder=self.initial_encoder, initial_heads=self.initial_heads,
            populations={s: semantic_digest([[sid, v.canonical[i], v.groups[i],
                [None if np.isnan(y) else float(y) for y in v.labels[i]]] for i, sid in enumerate(v.sample_ids)]) for s,v in views.items()})
        self.cache = {}
        torch.manual_seed(seed)

    def batch(self, split, task, ids):
        from dataset import DataCollator
        graphs = []
        for i in ids:
            key = (split, task, i)
            if key not in self.cache:
                self.cache[key] = self.datasets[split][task][i]
            graphs.append(self.cache[key])
        b = DataCollator()(graphs)
        require(not b.is_empty and b.y.numel() == len(ids) and torch.isfinite(b.y).all(), 'graph/label dropped')
        return b.to(self.device)

    def evaluate(self):
        self.model.eval()
        rows = []
        v = self.views['validation']
        with torch.no_grad():
            for j, t in enumerate(TASKS):
                ds = self.datasets['validation'][t]
                for start in range(0, len(ds), 32):
                    ids = list(range(start, min(start+32, len(ds))))
                    raw = self.model(self.batch('validation', t, ids), task_name=t)[t]
                    require(raw.shape == (len(ids), 3) and torch.isfinite(raw).all(), 'prediction shape/finite')
                    predictions = (raw[:, 0].double()*self.scaler.stds[j]+self.scaler.means[j]).cpu().tolist()
                    for i, pred in zip(ids, predictions):
                        k = ds.indices[i]
                        rows.append(dict(task=t, split='validation', sample_id=v.sample_ids[k],
                            canonical=v.canonical[k], group=v.groups[k], label=float(v.labels[k,j]), prediction=pred))
        return rows, metrics(rows, v)

    def run(self, output, *, epochs, resume=None, resume_sha=None):
        require(type(epochs) is int and 0 < epochs <= 40, 'epochs')
        require((resume is None) == (resume_sha is None), 'resume SHA pair')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        json_write(output/'resolved_config.json', self.identity)
        history, best_state, best_rows, best_epoch = [], None, None, None
        start = 0
        if resume is not None:
            require(digest(resume) == resume_sha, 'resume SHA')
            p = torch.load(resume, map_location='cpu', weights_only=True)
            require(p['identity'] == self.identity, 'resume identity')
            history = p['history']; start = len(history)
            require(0 < start < epochs and [h['epoch'] for h in history] == list(range(start)), 'resume history')
            require(p['total_steps'] == start*len(schedule(self.counts, self.seed, 0)), 'resume steps')
            best_epoch = min(range(start), key=lambda e: history[e]['validation']['macro_rmse'])
            require(p['best_epoch'] == best_epoch and metrics(p['best_rows'], self.views['validation']) == history[best_epoch]['validation'], 'resume best')
            for key in ('model_state', 'best_state'):
                require(state_digest(p[key]) == p[key+'_sha'] and all(torch.isfinite(v).all() for v in p[key].values()), 'resume state')
                if self.method == 'FROZEN':
                    require(state_digest({k[8:]: v for k,v in p[key].items() if k.startswith('encoder.')}) == self.initial_encoder, 'frozen drift')
            if self.control:
                self.control.validate_optimizer_state(p['optimizer'], start-1, p['model_state'])
            else:
                require(p['optimizer']['param_groups'] == self.optimizer.state_dict()['param_groups'], 'resume optimizer')
            self.model.load_state_dict(p['model_state'], strict=True)
            self.optimizer.load_state_dict(p['optimizer'])
            best_state, best_rows = p['best_state'], p['best_rows']
            torch.set_rng_state(p['cpu_rng'])
            if self.device == 'cuda:0':
                torch.cuda.set_rng_state(p['cuda_rng'], 0)
        from loss import QuantileRegressionLoss
        loss_fn = QuantileRegressionLoss()
        for epoch in range(start, epochs):
            if self.control:
                self.control.begin_epoch(epoch)
            self.model.train()
            exposure = dict.fromkeys(TASKS, 0); losses = {t: [] for t in TASKS}
            for t, ids in schedule(self.counts, self.seed, epoch):
                j = TASKS.index(t); b = self.batch('train', t, ids)
                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_fn.compute_loss(self.model(b, task_name=t)[t],
                        (b.y.reshape(-1,1)-self.scaler.means[j])/self.scaler.stds[j])
                require(torch.isfinite(loss), 'nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_([x for x in self.model.parameters() if x.requires_grad], 1., error_if_nonfinite=True)
                if self.control:
                    self.control.verify_frozen()
                if self.method == 'FROZEN':
                    require(all(x.grad is None for x in self.model.encoder.parameters()), 'frozen gradient')
                self.optimizer.step()
                exposure[t] += len(ids); losses[t].append(float(loss.detach()))
            require(exposure == self.counts, 'exposure changed')
            state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            require(all(torch.isfinite(v).all() for v in state.values()), 'nonfinite state')
            if self.method == 'FROZEN':
                require(state_digest(self.model.encoder.state_dict()) == self.initial_encoder, 'frozen drift')
            rows, met = self.evaluate()
            if best_epoch is None or met['macro_rmse'] < history[best_epoch]['validation']['macro_rmse']:
                best_epoch, best_state, best_rows = epoch, state, rows
            steps = (epoch+1)*len(schedule(self.counts, self.seed, epoch))
            history.append(dict(epoch=epoch, total_steps=steps, exposure=exposure,
                mean_batch_loss={t: float(np.mean(x)) for t,x in losses.items()}, validation=met,
                optimization=self.control.record() if self.control else dict(frozen=True, backbone_lr=0., head_lr=.001)))
            payload = dict(identity=self.identity, history=history, total_steps=steps,
                model_state=state, model_state_sha=state_digest(state), optimizer=self.optimizer.state_dict(),
                best_state=best_state, best_state_sha=state_digest(best_state), best_rows=best_rows, best_epoch=best_epoch,
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0) if self.device=='cuda:0' else None)
            save_state(output/'last.pt', payload)
            json_write(output/'history.json', history)
        save_state(output/'best.pt', dict(identity=self.identity, epoch=best_epoch, model_state=best_state,
                    model_state_sha=state_digest(best_state)))
        json_write(output/'best_validation.json', best_rows)
        # Reopen saved best and replay only its validation, never test/calibration.
        saved = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        require(saved['identity'] == self.identity and state_digest(saved['model_state']) == saved['model_state_sha'], 'saved best')
        self.model.load_state_dict(saved['model_state'], strict=True)
        replay, _ = self.evaluate()
        require(len(replay) == len(best_rows), 'best replay rows')
        require(all({k:v for k,v in a.items() if k != 'prediction'} == {k:v for k,v in b.items() if k != 'prediction'}
                    and math.isclose(a['prediction'], b['prediction'], abs_tol=1e-5, rel_tol=1e-6)
                    for a,b in zip(replay, best_rows)), 'best replay differs')
        result = dict(identity=self.identity, run_directory=str(output.resolve()), epochs=epochs,
            updates=history[-1]['total_steps'], best_epoch=best_epoch,
            metrics=metrics(best_rows, self.views['validation']), validation_replayed=True,
            files={n: dict(sha256=digest(output/n), size_bytes=(output/n).stat().st_size)
                   for n in ('best.pt','last.pt','resolved_config.json','history.json','best_validation.json')})
        json_write(output/'result.json', result)
        return result
