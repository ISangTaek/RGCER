"""Recompute B115 smoke content; PASS is not external scientific acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from baselines.b115_data import AUDIT_SHA, check_bytes
from baselines.b115_training import implementation_identity, validation_metrics, predict
from baselines.features import avalon_matrix
from baselines.scaling import TaskScaler
from baselines.models.toxacol import endpoint_feature_matrix, ToxACoLNet
from dataset115_adapter import Dataset115Table
from dataset115_contract import PRIMARY, semantic_digest


def read_json(path):
    def invalid(value):
        raise ValueError(f'non-standard JSON number {value}')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=invalid, object_pairs_hook=unique)


def verify(output, audit_path, csv_path, split_path, tox_manifest):
    output = Path(output)
    check_bytes(audit_path, AUDIT_SHA)
    audit = read_json(audit_path)
    for path, item in zip((csv_path, split_path, tox_manifest), audit['input']):
        check_bytes(path, item['sha256'])
    report = read_json(output / 'report.json')
    contract = read_json(output / 'data_contract.json')
    route = report['route']
    if route not in ('A', 'B') or report['mode'] != 'one-step':
        raise ValueError('requires actual one-step smoke, not audit-only')
    counts = {'A': (55256,76279,109,36,329,3), 'B': (55670,78324,61,27,320,0)}[route]
    for field, value in zip(('active_rows','observations','tasks','feature_width','edges','source_target_edges'), counts):
        if type(report[field]) is not int or report[field] != value:
            raise ValueError(f'wrong {field}')
    if report['test_executed'] is not False or type(report['updates']) is not int or report['updates'] != 1:
        raise ValueError('wrong execution scope')
    if report['implementation'] != implementation_identity():
        raise ValueError('implementation differs; verify in the executing checkout')
    if contract['tasks'] != audit[route]['tasks'] or contract['input_identity'] != audit['data_identity']:
        raise ValueError('wrong data/task identity')
    features, vocab = endpoint_feature_matrix(tuple(contract['tasks']), extend_vocabulary=True)
    if (features.shape != (counts[2], counts[3]) or contract['vocabulary'] != vocab or contract['audit_sha256'] != AUDIT_SHA or
            contract['rows'] != counts[0] or contract['observations'] != counts[1]):
        raise ValueError('data contract mismatch')
    for i, task in enumerate(contract['tasks']):
        old = audit[route]['scaler'][task]
        if (contract['scaler']['counts'][i] != old['n'] or not np.allclose(
            [contract['scaler']['means'][i],contract['scaler']['stds'][i]],
            [old['mean'],old['std']], rtol=1e-12,atol=1e-12)):
            raise ValueError('scaler mismatch')
    check_bytes(output / 'smoke.pt', report['checkpoint_sha256'])
    payload = torch.load(output / 'smoke.pt',map_location='cpu',weights_only=True)
    if payload['role'] != 'SMOKE_NOT_FORMAL' or payload['contract'] != contract:
        raise ValueError('checkpoint role/contract mismatch')
    adjacency = np.asarray(contract['adjacency'])
    binary = (adjacency > 0).astype(np.float32)
    degree = binary.sum(axis=1)
    normalized = binary / np.sqrt(degree[:,None]*degree[None,:])
    if ((np.count_nonzero(binary)-len(binary))//2 != counts[4]
            or np.count_nonzero(binary[:-5,-5:]) != counts[5]):
        raise ValueError('adjacency edge counts differ')
    np.testing.assert_allclose(adjacency,normalized,rtol=0,atol=1e-7)
    model = ToxACoLNet(adjacency,features)
    model.load_state_dict(payload['state'],strict=True)
    torch.testing.assert_close(model.adjacency,torch.as_tensor(adjacency,dtype=torch.float32),rtol=0,atol=0)
    torch.testing.assert_close(model.endpoint_features,torch.as_tensor(features),rtol=0,atol=0)
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint')
    table = Dataset115Table.load(csv_path,split_path,tox_manifest,expected_tox_sha=audit['input'][2]['sha256'])
    validation = table.view(route,'target','validation')
    keep = np.flatnonzero(np.isfinite(validation.labels).any(axis=1))
    ids = [validation.sample_ids[i] for i in keep]
    truth = validation.labels[keep]
    with np.load(output / 'validation_smoke.npz',allow_pickle=False) as z:
        if z['sample_ids'].tolist() != ids or z['tasks'].tolist() != list(PRIMARY):
            raise ValueError('wrong validation members/tasks')
        np.testing.assert_array_equal(z['truth'],truth)
        prediction = z['prediction'].copy()
    if report['validation_rows'] != len(ids) or contract['validation_ids_sha256'] != semantic_digest(ids):
        raise ValueError('validation identity/count mismatch')
    if contract['validation_labels_sha256'] != semantic_digest(np.where(np.isfinite(truth),truth,None).tolist()):
        raise ValueError('validation label identity mismatch')
    actual = validation_metrics(truth,prediction)
    replay = predict(model, avalon_matrix(tuple(validation.smiles[i] for i in keep)),
                     TaskScaler.from_dict(contract['scaler']), 'cpu')
    np.testing.assert_allclose(replay,prediction,rtol=1e-5,atol=1e-5)
    # Independently calculate each RMSE without the production metrics function.
    independent = [float(np.sqrt(np.mean((prediction[np.isfinite(truth[:,j]),j]
                  -truth[np.isfinite(truth[:,j]),j])**2))) for j in range(5)]
    if not np.isclose(actual['macro_rmse'],np.mean(independent),rtol=1e-12,atol=1e-12):
        raise ValueError('independent metric mismatch')
    if actual != report['metrics'] or report['reload_max_abs'] != 0:
        raise ValueError('reported metric/reload mismatch')
    support_ids = read_json(output/'sample_ids.json')
    if (type(report['support_rows']) is not int or not 2 <= report['support_rows'] <= 128
            or len(support_ids) != report['support_rows'] or len(set(support_ids)) != len(support_ids)
            or not np.isfinite(report['loss'])):
        raise ValueError('support member count mismatch')
    target_train = table.view(route,'target','train')
    permitted_ids = {sid for i,sid in enumerate(target_train.sample_ids)
                     if np.isfinite(target_train.labels[i]).any()}
    if route == 'A':
        source_train = table.view('A','source','train')
        permitted_ids.update(sid for i,sid in enumerate(source_train.sample_ids)
                             if np.isfinite(source_train.labels[i]).any())
    else:
        refs = read_json(tox_manifest)['records']
        permitted_ids.update('toxacute:'+r['sample_id'] for r in refs if r['split']=='train')
    if not set(support_ids).issubset(permitted_ids):
        raise ValueError('smoke support includes a forbidden member')
    # This verifies content, not the truth of external command logs/GPU provenance.
    return dict(content_status='PASS', acceptance_status='PENDING_REVIEW',route=route,
                independently_recomputed_macro_rmse=float(np.mean(independent)),
                checkpoint_replay_max_abs=float(np.max(np.abs(replay-prediction))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','audit-path','csv-path','split-path','tox-manifest'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    print(json.dumps(verify(**vars(args)),ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
