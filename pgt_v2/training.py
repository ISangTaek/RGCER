"""Common point training, epoch-boundary exact resume and validation-only selection."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .artifacts import load_arrays, load_manifest, read_json, write_json
from .cache import GraphCache, file_sha256
from .chemistry import digest
from .config import git_identity
from .data import PGTCollator
from .models import build_model, molecular_features, parameter_counts


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def endpoint_loss(prediction, target, counts, active_rows):
    """Unbiased equal-endpoint objective under uniform active-row sampling."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] < 1:
        raise ValueError('Prediction/target shape mismatch')
    counts = torch.as_tensor(counts, dtype=prediction.dtype, device=prediction.device)
    if counts.shape != (prediction.shape[1],) or bool((counts <= 0).any()) or active_rows < 1:
        raise ValueError('Positive full-view counts and active row count required')
    mask = torch.isfinite(target)
    safe = torch.where(mask, target, torch.zeros_like(target))
    error = (prediction - safe).masked_fill(~mask, 0.0)
    return (error.square() / counts).sum() * (active_rows / (prediction.shape[0] * prediction.shape[1]))


def partition_batches(indices, size, *, batch_norm=False):
    if size < 1 or (batch_norm and size < 2):
        raise ValueError('BatchNorm needs at least two records per microbatch')
    parts = [list(indices[i:i+size]) for i in range(0, len(indices), size)]
    if batch_norm and parts and len(parts[-1]) == 1:
        if len(parts) < 2:
            raise ValueError('A single active row cannot train BatchNorm')
        parts[-2].extend(parts.pop())
    return parts


class ModelInputs:
    """Input-only cache adapter; labels never enter GraphCache or fingerprints."""
    def __init__(self, cache, records, method, device='cpu'):
        self.cache, self.records, self.method, self.device = cache, records, method, device
        self.collate = PGTCollator()
        self.features = None
        required = ('record_id', 'cache_offset', 'raw_row_index', 'chemical_id', 'group_id')
        n = len(records['record_id'])
        if any(len(records[k]) != n for k in required) or len(set(records['record_id'])) != n:
            raise ValueError('Malformed input record registry')

    def __len__(self):
        return len(self.records['record_id'])

    def batch(self, indices):
        graphs, mols = [], []
        for i in indices:
            graph, mol = self.cache.read(int(self.records['cache_offset'][i]))
            expected = (str(self.records['record_id'][i]), int(self.records['raw_row_index'][i]), str(self.records['chemical_id'][i]))
            if (graph.sample_id, graph.raw_row_index, graph.model_input_id) != expected:
                raise ValueError('Input registry does not match cache identity')
            graphs.append(graph)
            mols.append(mol)
        if self.method == 'M1':
            return mols
        if self.method in ('M0', 'M3'):
            values = molecular_features(mols, 'morgan' if self.method == 'M0' else 'avalon')
            return values if self.method == 'M0' else torch.tensor(values, device=self.device)
        batch = self.collate(graphs).to(self.device)
        if self.method in ('N0', 'N4'):
            return batch, torch.tensor(molecular_features(mols, 'avalon'), device=self.device)
        return batch


def metrics(truth, prediction, tasks, groups=None):
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.shape[1] != len(tasks) or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite/misaligned predictions')
    task_rows, rmse = [], []
    errors = []
    for i, name in enumerate(tasks):
        mask = np.isfinite(truth[:, i]); n = int(mask.sum())
        if not n:
            raise ValueError('Fixed task denominator has no labels: ' + name)
        err = prediction[mask, i] - truth[mask, i]
        score = float(np.sqrt(np.mean(err**2))); rmse.append(score); errors.extend(abs(err).tolist())
        denominator = float(np.sum((truth[mask, i] - truth[mask, i].mean())**2))
        task_rows.append(dict(task=name, n=n, rmse=score, mae=float(np.mean(abs(err))),
            r2=1-float(np.sum(err**2))/denominator if n >= 2 and denominator > 0 else None,
            groups=len(set(np.asarray(groups)[mask])) if groups is not None else None))
    measurements = {}
    for kind in ('LD50','LDLo','TDLo'):
        subset = [row for row in task_rows if row['task'].endswith('_'+kind)]
        if subset:
            measurements[kind] = dict(tasks=len(subset), macro_rmse=float(np.mean([r['rmse'] for r in subset])),
                                      macro_mae=float(np.mean([r['mae'] for r in subset])))
    return dict(macro_rmse=float(np.mean(rmse)), micro_mae=float(np.mean(errors)), endpoints=task_rows,measurements=measurements)


def selection_score(report, selection_tasks=None):
    """Predeclared target selection; absent means the legacy all-task criterion."""
    if selection_tasks is None:
        return report['macro_rmse']
    rows = {row['task']: row['rmse'] for row in report['endpoints']}
    if not selection_tasks or len(set(selection_tasks)) != len(selection_tasks) or set(selection_tasks) - set(rows):
        raise ValueError('Selection tasks must be a unique nonempty subset of reported tasks')
    return float(np.mean([rows[name] for name in selection_tasks]))


@torch.no_grad()
def predict_array(model, inputs, stats, method, batch_size=16):
    if method != 'M0':
        model.eval()
    pieces = []
    for indices in partition_batches(list(range(len(inputs))), batch_size):
        x = inputs.batch(indices)
        y = model.predict(x) if method == 'M0' else model(x).cpu().numpy() * stats['scale'] + stats['mean']
        if not np.isfinite(y).all():
            raise FloatingPointError('Nonfinite prediction')
        pieces.append(y)
    if not pieces:
        raise ValueError('Empty prediction partition')
    return np.concatenate(pieces)


def _rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                cuda_device=str(device) if device.type == 'cuda' else None)


def _restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'],device=state['cuda_device'])


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)
    write_json(str(path) + '.sha256.json', {'sha256': file_sha256(path)})


def read_checkpoint(path):
    # Only locally produced, SHA-bound trusted training artifacts may use pickle.
    if file_sha256(path) != read_json(str(path) + '.sha256.json')['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    return torch.load(path, map_location='cpu', weights_only=False)


def train_model(model, train_inputs, train_labels, val_inputs, val_labels, stats, *, config, identity,
                output, method, resume=False, stop_after_epoch=None):
    """Stop/restart at completed epoch boundaries; partial epochs replay from last.pt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    if method == 'M0':
        if resume:
            raise ValueError('RF bundle is atomic; resume only applies to neural training')
        model.fit(train_inputs.batch(list(range(len(train_inputs)))), train_labels)
        pred = predict_array(model, val_inputs, stats, method)
        report = metrics(val_labels, pred, identity['task_names'], val_inputs.records['group_id'])
        save_checkpoint(output/'best.pt', dict(kind='best', identity=identity, forest=model, stats=stats, best_epoch=1, best=report['macro_rmse']))
        write_json(output/'summary.json', dict(identity=identity, method=method, best_epoch=1,
            validation=report, best_sha256=file_sha256(output/'best.pt'), terminal='COMPLETE', optimizer_steps=0))
        return report
    cfg = config
    active = np.where(np.isfinite(train_labels).any(1))[0].tolist()
    if not active or not np.array_equal(np.isfinite(train_labels).sum(0), stats['count']):
        raise ValueError('Training label mask/count mismatch')
    standardized = (train_labels - stats['mean']) / stats['scale']
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    start, best, stale, history, updates, best_epoch = 0, float('inf'), 0, [], 0, None
    if resume:
        checkpoint = read_checkpoint(output/'last.pt')
        if checkpoint['identity'] != identity or checkpoint['kind'] != 'last':
            raise ValueError('Resume run/task/config/input identity mismatch')
        if file_sha256(output/'best.pt') != checkpoint['best_checkpoint_sha256']:
            raise ValueError('Resume best checkpoint changed since the saved epoch')
        if not all(np.array_equal(stats[k], checkpoint['stats'][k]) for k in stats):
            raise ValueError('Resume training statistics mismatch')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, stale, history, updates, best_epoch = [checkpoint[k] for k in ('epoch','best','stale','history','updates','best_epoch')]
        _restore_rng(checkpoint['rng'])
    for epoch in range(start, cfg['max_epochs']):
        if epoch >= cfg['min_epochs'] and stale >= cfg['patience']:
            break
        started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        order = [active[i] for i in torch.randperm(len(active)).tolist()]
        weighted_loss = 0.0
        weighted_regularization = 0.0
        for effective in partition_batches(order, cfg['effective_batch'], batch_norm=method == 'M3'):
            optimizer.zero_grad(set_to_none=True)
            for micro in partition_batches(effective, cfg['micro_batch'], batch_norm=method == 'M3'):
                prediction = model(train_inputs.batch(micro))
                target = torch.as_tensor(standardized[micro], dtype=prediction.dtype, device=device)
                loss = endpoint_loss(prediction, target, stats['count'], len(active))
                if hasattr(model, 'regularization_loss'):
                    regularization = model.regularization_loss()
                    loss = loss + regularization
                    weighted_regularization += float(regularization.detach()) * len(micro)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; no checkpoint promoted')
                (loss * (len(micro) / len(effective))).backward()
                weighted_loss += float(loss.detach()) * len(micro)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); updates += 1
        pred = predict_array(model, val_inputs, stats, method, cfg['micro_batch'])
        report = metrics(val_labels, pred, identity['task_names'], val_inputs.records['group_id'])
        score = selection_score(report, cfg.get('selection_tasks'))
        improved = score < best
        if improved:
            best, stale, best_epoch = score, 0, epoch+1
        else:
            stale += 1
        row = dict(epoch=epoch+1, train_loss=weighted_loss/len(active), validation=report, updates=updates,
            visible_counts=stats['count'].tolist(), active_rows=len(active), elapsed_seconds=time.perf_counter()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            best_epoch=best_epoch, stale=stale, selection_score=score,
            regularization_loss=weighted_regularization/len(active))
        history.append(row)
        checkpoint = dict(format='PGTV2_CHECKPOINT_1', identity=identity, model=copy.deepcopy(model.state_dict()),
            optimizer=optimizer.state_dict(), stats=stats, rng=_rng(device), epoch=epoch+1, best=best, stale=stale,
            best_epoch=best_epoch, history=history, updates=updates)
        if improved:
            save_checkpoint(output/'best.pt', {**checkpoint, 'kind':'best'})
        checkpoint['best_checkpoint_sha256'] = file_sha256(output/'best.pt')
        save_checkpoint(output/'last.pt', {**checkpoint, 'kind':'last'})
        write_json(output/'history.json', history)
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
            break
    completed = history[-1]['epoch'] >= cfg['max_epochs'] or (history[-1]['epoch'] >= cfg['min_epochs'] and stale >= cfg['patience'])
    summary = dict(identity=identity, method=method, terminal='COMPLETE' if completed else 'INTERRUPTED_AT_EPOCH',
        best_epoch=best_epoch, validation=history[best_epoch-1]['validation'], optimizer_steps=updates,
        selection_score=best, selection_tasks=cfg.get('selection_tasks'),
        best_sha256=file_sha256(output/'best.pt'), parameters=parameter_counts(model))
    write_json(output/'summary.json', summary)
    return summary


def run_training(run_file, view_root, cache_root, output, *, device='cpu', chemprop_root=None, resume=False):
    run = read_json(run_file)
    if run.get('state') != 'READY' or run.get('phase') not in ('SYNTHETIC', 'P2', 'P3', 'P4', 'P5'):
        raise ValueError('Training requires a concrete released run; G01 has no fitting units')
    check = {k:v for k,v in run.items() if k != 'run_sha256'}
    if digest(check) != run.get('run_sha256') or digest(run['config']) != run['config_sha256']:
        raise ValueError('Resolved run/config hash mismatch')
    manifest = load_manifest(view_root)
    expected_fold = 'F0' if run['fold'] == 'DEV' and run['phase'] == 'P2' else str(run['fold'])
    if expected_fold != manifest['fold']:
        raise ValueError('Run/fold identity mismatch')
    if run['feature_spec_sha256'] != manifest['feature_spec_sha256']:
        raise ValueError('Run FeatureSpec mismatch')
    if run['train_sha256'] != manifest['files']['train.npz'] or run['validation_sha256'] != manifest['files']['validation.npz']:
        raise ValueError('Run partition SHA mismatch')
    for run_key, manifest_key in [('view_sha256','manifest_sha256'),('view_id','view_id'),('cache_id','cache_id'),
                                 ('source_sha256','source_sha256'),('task_names','task_names')]:
        if run[run_key] != manifest[manifest_key]:
            raise ValueError('Resolved run input mismatch: ' + run_key)
    if run['code_commit'] != git_identity():
        raise ValueError('Run commit does not match this checkout')
    train = load_arrays(view_root, 'train.npz', manifest)
    val = load_arrays(view_root, 'validation.npz', manifest)
    stats = load_arrays(view_root, 'statistics.npz', manifest)
    cache = GraphCache(cache_root, source_sha256=manifest['source_sha256'])
    if cache.manifest['cache_id'] != manifest['cache_id']:
        raise ValueError('Wrong graph cache')
    seed_everything(int(run['init_seed']))
    method = run['method_id']
    model = build_model(method, manifest['task_names'], run['config'], adjacency=stats['adjacency'], counts=stats['count'],
                        chemprop_root=chemprop_root, device=device, seed=int(run['init_seed']))
    identity = {**run, 'view_manifest':manifest, 'runtime_device':str(device), 'torch_version':str(torch.__version__),
                'numpy_version':np.__version__, 'torch_threads':torch.get_num_threads(),
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    return train_model(model, ModelInputs(cache, train, method, device), train['labels'],
        ModelInputs(cache, val, method, device), val['labels'], stats, config=run['config'], identity=identity,
        output=output, method=method, resume=resume)
