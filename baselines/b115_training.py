"""Independent B115 engine. Formal launch/test export require later approval.

Epoch boundaries are the only supported resume points. Each checkpoint carries
optimizer, Torch RNG, deterministic sampler rule, and the resolved data contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from dataset115_contract import PRIMARY, semantic_digest
from .metrics import regression_metrics
from .models.toxacol import ToxACoLNet, toxacol_learning_rate
from .scaling import TaskScaler

LUT = ((20, .001), (40, .0006), (50, .00012), (60, .000024),
       (70, .000012), (80, .000001), (100, .0000001), (120, .00000001))


def configuration(*, seed=42, lr_multiplier=1.0):
    if type(seed) is not int or seed not in range(42, 47) or lr_multiplier not in (.5, 1., 1.5):
        raise ValueError('unapproved seed/LR candidate')
    return dict(epochs=120, batch_size=32, seed=seed, lr_multiplier=float(lr_multiplier),
                momentum=.9, nesterov=True, weight_decay=.0005, dropout=.1,
                lr_lut=[list(row) for row in LUT], sampler='RandomState(seed+epoch):full_permutation')


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ('baselines/b115_training.py', 'baselines/b115_data.py', 'baselines/models/toxacol.py',
             'baselines/scaling.py', 'baselines/metrics.py', 'baselines/features.py',
             'dataset115_adapter.py', 'dataset115_contract.py', 'scripts/smoke_b115.py',
             'scripts/verify_b115_smoke.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def masked_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('loss shape mismatch')
    if mask.dtype != torch.bool or not bool(mask.any()) or not bool(mask.any(dim=1).all()):
        raise ValueError('empty/invalid observation mask')
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target[mask]).all()):
        raise ValueError('nonfinite prediction/observed label')
    # Index before arithmetic: NaN * 0 is not a safe masking operation.
    loss = (prediction[mask] - target[mask]).square().mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite loss')
    return loss


def epoch_candidates(best_epoch, epoch_count=120, count=10):
    if any(type(v) is not int for v in (best_epoch, epoch_count, count)) or not (
            0 <= best_epoch < epoch_count and 1 <= count <= epoch_count):
        raise ValueError('invalid ensemble epoch range')
    return sorted(range(epoch_count), key=lambda epoch: (abs(epoch-best_epoch), epoch))[:count]


def is_improvement(metric, best):
    if metric is None or not math.isfinite(metric):
        raise ValueError('undefined validation selection metric')
    return best is None or metric < best


def validation_metrics(truth, prediction):
    if not np.isfinite(prediction).all() or np.isinf(truth).any():
        raise ValueError('nonfinite validation prediction/label')
    result = regression_metrics(truth, prediction, PRIMARY)
    result['selection_metric'] = 'validation_human5_macro_rmse'
    is_improvement(result['macro_rmse'], None)
    return result


def predict(model, x, scaler, device, batch_size=32):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            raw = model(torch.as_tensor(x[start:start+batch_size], device=device)).cpu().numpy()
            if not np.isfinite(raw).all():
                raise ValueError('nonfinite model output')
            predictions.append(scaler.inverse_transform(raw)[:, -5:])
    return np.concatenate(predictions)


def optimizer_step(model, optimizer, x, y, mask):
    model.train()
    # Same singleton semantics as the legacy ToxACoL runner; no row duplication.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.train(len(x) > 1)
    optimizer.zero_grad(set_to_none=True)
    loss = masked_loss(model(x), y, mask)
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('nonfinite gradient')
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('nonfinite parameter')
    return float(loss.detach())


def save_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def train_engine(*, train_x, train_labels, validation_x, validation_labels,
                 adjacency, endpoint_features, scaler, contract, output, config,
                 device='cpu', resume=None, resume_sha256=None, stop_after_epoch=None):
    """Internal engine, not an authorization API. No test data argument exists."""
    expected = configuration(seed=config['seed'], lr_multiplier=config['lr_multiplier'])
    if config != expected:
        raise ValueError('configuration differs from fixed B115 candidates')
    if (train_labels.shape != (len(train_x), len(scaler.task_names)) or
            train_x.shape[1:] != (1024,) or validation_x.shape[1:] != (1024,) or
            validation_labels.shape != (len(validation_x), 5) or not len(train_x)
            or not len(validation_x) or not np.isfinite(train_x).all()
            or not np.isfinite(validation_x).all() or np.isinf(train_labels).any()
            or not np.isfinite(train_labels).any(axis=1).all()):
        raise ValueError('invalid feature/label arrays')
    if tuple(contract['tasks']) != scaler.task_names or scaler.task_names[-5:] != PRIMARY:
        raise ValueError('task order mismatch')
    if TaskScaler.fit(train_labels, scaler.task_names, allow_empty=False).to_dict() != scaler.to_dict():
        raise ValueError('scaler is not fitted to these training labels')
    # Bind actual arrays as well as the caller's scientific data contract.
    array_hashes = {}
    for name, array in [('train_x', train_x), ('train_y', train_labels),
                        ('validation_x', validation_x), ('validation_y', validation_labels)]:
        a = np.ascontiguousarray(array)
        array_hashes[name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                 sha256=hashlib.sha256(a.tobytes()).hexdigest())
    resolved = dict(data=contract, config=config, device=str(device), arrays=array_hashes,
                    adjacency=np.asarray(adjacency).tolist(), endpoint_features=np.asarray(endpoint_features).tolist(),
                    scaler=scaler.to_dict(), implementation=implementation_identity())
    identity = semantic_digest(resolved)
    end = config['epochs'] if stop_after_epoch is None else stop_after_epoch
    if type(end) is not int or not 1 <= end <= config['epochs']:
        raise ValueError('invalid epoch stop')
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    model = ToxACoLNet(adjacency, endpoint_features, dropout=config['dropout']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LUT[0][1], momentum=.9,
                                nesterov=True, weight_decay=.0005)
    start, best, best_epoch, updates, history = 0, None, None, 0, []
    if resume is not None:
        if resume_sha256 is None or hashlib.sha256(Path(resume).read_bytes()).hexdigest() != resume_sha256:
            raise ValueError('resume SHA mismatch')
        payload = torch.load(resume, map_location='cpu', weights_only=True)
        if (payload.get('format') != 'B115_epoch_v1' or payload.get('identity') != identity
                or payload.get('resolved') != resolved):
            raise ValueError('checkpoint/data/config/implementation mismatch')
        start = payload['epoch'] + 1
        if not 0 < start < end:
            raise ValueError('resume has no next authorized epoch')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['rng_cpu'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(payload['rng_cuda'])
        best, best_epoch, updates, history = (payload[k] for k in ('best', 'best_epoch', 'updates', 'history'))
        if len(history) != start or history[-1]['epoch'] != start-1:
            raise ValueError('checkpoint history mismatch')
        if [r['epoch'] for r in history] != list(range(start)) or any(
                not math.isfinite(r['macro_rmse']) for r in history):
            raise ValueError('checkpoint epoch/metric mismatch')
        selected = min(range(start), key=lambda i: history[i]['macro_rmse'])
        if (best_epoch != selected or best != history[selected]['macro_rmse'] or
                updates != start * math.ceil(len(train_x)/config['batch_size'])):
            raise ValueError('checkpoint selection/cost mismatch')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / 'resolved.json', resolved)
    y, mask = scaler.transform(train_labels)
    mask = mask.astype(bool)
    begun = time.monotonic()
    for epoch in range(start, end):
        lr = toxacol_learning_rate(epoch, [r[0] for r in LUT[:-1]], [r[1] for r in LUT]) * config['lr_multiplier']
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = np.random.RandomState(config['seed'] + epoch).permutation(len(train_x))
        loss_sum, count, epoch_updates = 0., 0, 0
        for offset in range(0, len(order), config['batch_size']):
            idx = order[offset:offset+config['batch_size']]
            x_t, y_t, m_t = (torch.as_tensor(a[idx], device=device) for a in (train_x, y, mask))
            loss = optimizer_step(model, optimizer, x_t, y_t, m_t)
            observations = int(m_t.sum())
            loss_sum += loss * observations
            count += observations
            epoch_updates += 1
        updates += epoch_updates
        predictions = predict(model, validation_x, scaler, device)
        metrics = validation_metrics(validation_labels, predictions)
        if is_improvement(metrics['macro_rmse'], best):
            best, best_epoch = metrics['macro_rmse'], epoch
        history.append(dict(epoch=epoch, lr=lr, train_loss=loss_sum/count,
                            observations=count, epoch_updates=epoch_updates, **metrics))
        payload = dict(format='B115_epoch_v1', identity=identity, resolved=resolved, epoch=epoch,
                       model=model.state_dict(), optimizer=optimizer.state_dict(),
                       rng_cpu=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                       best=best, best_epoch=best_epoch, updates=updates, history=history)
        path = directory / f'epoch_{epoch:03d}.pt'
        temporary = directory / f'epoch_{epoch:03d}.partial'
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
        temporary.rename(path)
        np.save(directory / f'validation_{epoch:03d}.npy', predictions, allow_pickle=False)
        save_json(directory / f'epoch_{epoch:03d}.json', dict(epoch=epoch, best_epoch=best_epoch,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **metrics))
    report = dict(identity=identity, best_epoch=best_epoch, best_validation_macro_rmse=best,
        epochs_completed=end, start_epoch=start, updates=updates, history=history,
        elapsed_seconds=time.monotonic()-begun, test_executed=False,
        resume_parent=None if resume is None else dict(path=str(resume),
            sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest()))
    save_json(directory / 'run.json', report)
    return report
