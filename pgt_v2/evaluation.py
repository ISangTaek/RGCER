"""Validation selection, truth-free sealed prediction, then separate scoring."""
from pathlib import Path

import numpy as np

from .artifacts import load_arrays, load_manifest, read_json, seal_manifest, write_json
from .cache import GraphCache, file_sha256
from .chemistry import digest
from .models import build_model
from .training import ModelInputs, metrics, predict_array, read_checkpoint


def select_config(run_directories, destination, *, required_configs=('C0','C1','C2','C3')):
    if Path(destination).exists():
        raise FileExistsError(destination)
    rows = []
    for directory in run_directories:
        directory = Path(directory)
        row = read_json(directory/'summary.json')
        identity = row['identity']
        if row['terminal'] != 'COMPLETE' or int(identity['init_seed']) != 42:
            raise ValueError('Selection requires complete seed42 candidates')
        if file_sha256(directory/'best.pt') != row['best_sha256']:
            raise ValueError('Selected candidate checkpoint changed')
        checkpoint = read_checkpoint(directory/'best.pt')
        if checkpoint['identity'] != identity or checkpoint['best_epoch'] != row['best_epoch']:
            raise ValueError('Candidate checkpoint/report identity mismatch')
        score = row.get('selection_score', row['validation']['macro_rmse'])
        if not np.isfinite(score) or abs(score - checkpoint['best']) > 1e-12:
            raise ValueError('Invalid candidate validation score')
        rows.append((score, identity['config_id'], directory, row))
    if {x[1] for x in rows} != set(required_configs) or len(rows) != len(required_configs):
        raise ValueError('Exactly the frozen candidate configurations are required')
    signatures = {(x[3]['identity']['method_id'], str(x[3]['identity']['fold']), x[3]['identity']['view_sha256'],
                   x[3]['identity']['code_commit'], tuple(x[3]['identity']['task_names']),
                   tuple(x[3]['identity']['config'].get('selection_tasks') or ())) for x in rows}
    if len(signatures) != 1:
        raise ValueError('Candidate method/fold/view/code/task registry differs')
    minimum = min(x[0] for x in rows)
    score, config, directory, winner = min((x for x in rows if x[0] <= minimum + 1e-12), key=lambda x:x[1])
    result = dict(method_id=winner['identity']['method_id'], fold=winner['identity']['fold'], config_id=config,
        config=winner['identity']['config'], view_sha256=winner['identity']['view_sha256'],
        code_commit=winner['identity']['code_commit'], seed42_checkpoint=str((directory/'best.pt').resolve()),
        seed42_checkpoint_sha256=winner['best_sha256'], seed42_reused=True, refit_train_plus_validation=False,
        validation_macro_rmse=winner['validation']['macro_rmse'], selection_score=score,
        selection_tasks=winner['identity']['config'].get('selection_tasks'), best_epoch=winner['best_epoch'],
        candidates=[dict(config_id=x[1], validation_macro_rmse=x[3]['validation']['macro_rmse'],
                         selection_score=x[0], checkpoint_sha256=x[3]['best_sha256']) for x in rows])
    result['selection_sha256'] = digest(result)
    write_json(destination, result)
    return result


def load_model_checkpoint(checkpoint_path, *, device='cpu', chemprop_root=None):
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint.get('kind') != 'best':
        raise ValueError('Inference requires a validation-selected best checkpoint')
    identity = checkpoint['identity']
    method = identity['method_id']
    if method == 'M0':
        model = checkpoint['forest']
    else:
        model = build_model(method, identity['task_names'], identity['config'],
            adjacency=checkpoint['stats']['adjacency'], counts=checkpoint['stats']['count'], chemprop_root=chemprop_root, device=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    return model, checkpoint


def sealed_predict(checkpoint_path, view_root, cache_root, output, *, role='test', device='cpu', chemprop_root=None):
    if role not in ('train', 'validation', 'test'):
        raise ValueError('Unknown prediction role')
    manifest = load_manifest(view_root)
    model, checkpoint = load_model_checkpoint(checkpoint_path, device=device, chemprop_root=chemprop_root)
    identity = checkpoint['identity']
    if identity['view_sha256'] != manifest['manifest_sha256'] or identity['task_names'] != manifest['task_names']:
        raise ValueError('Checkpoint/view/task order mismatch')
    records = load_arrays(view_root, role + '_inputs.npz', manifest)
    if 'labels' in records:
        raise ValueError('Predict accepts input-only partitions')
    cache = GraphCache(cache_root, source_sha256=manifest['source_sha256'])
    if cache.manifest['cache_id'] != manifest['cache_id']:
        raise ValueError('Prediction cache identity mismatch')
    predictions = predict_array(model, ModelInputs(cache, records, identity['method_id'], device),
                                checkpoint['stats'], identity['method_id'], identity['config'].get('micro_batch',16))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'predictions.npz', **records, prediction=predictions)
    return seal_manifest(output, dict(format='PGTV2_PREDICTIONS_1', role=role, truth_included=False,
        view_sha256=manifest['manifest_sha256'], source_sha256=manifest['source_sha256'],
        feature_spec_sha256=manifest['feature_spec_sha256'], cache_id=manifest['cache_id'],
        checkpoint_sha256=file_sha256(checkpoint_path), identity=identity, task_names=manifest['task_names']), ['predictions.npz'])


def score_prediction(prediction_root, view_root, destination, *, expected_checkpoint_sha256):
    if Path(destination).exists():
        raise FileExistsError(destination)
    sealed = load_manifest(prediction_root)
    view = load_manifest(view_root)
    if (sealed['view_sha256'] != view['manifest_sha256'] or sealed['task_names'] != view['task_names'] or
        sealed['checkpoint_sha256'] != expected_checkpoint_sha256 or sealed['truth_included'] is not False):
        raise ValueError('Prediction seal/view/expected checkpoint mismatch')
    pred = load_arrays(prediction_root, 'predictions.npz', sealed)
    truth = load_arrays(view_root, sealed['role'] + '.npz', view)
    if 'labels' in pred or set(pred) != (set(truth)-{'labels'}) | {'prediction'}:
        raise ValueError('Prediction schema contains labels or unknown fields')
    for key in set(truth)-{'labels'}:
        if not np.array_equal(pred[key], truth[key]):
            raise ValueError('Prediction/source row identity mismatch: ' + key)
    result = metrics(truth['labels'], pred['prediction'], view['task_names'], truth['group_id'])
    result.update(prediction_manifest_sha256=sealed['manifest_sha256'], view_sha256=view['manifest_sha256'],
                  checkpoint_sha256=expected_checkpoint_sha256, role=sealed['role'], identity=sealed['identity'])
    write_json(destination, result)
    return result
