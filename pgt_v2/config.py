"""Four registered candidates and fully resolved run identities."""
import json
import subprocess
from pathlib import Path
from .chemistry import digest
from molecular_features import PGT_FEATURE_SPEC
from .prior_prompts import PRIOR_METHODS

CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'configs' / 'pgt_v2'
METHODS = ('M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'A1', 'A2', 'A3', 'N0', 'N1', 'N1S', 'N2', 'N3', 'N4',
           'N1H', 'N1R', 'N1RP', 'TOX_AUTHOR') + PRIOR_METHODS


def candidate(method, config_id):
    if method == 'TOX_AUTHOR' and config_id == 'AUTHOR':
        return dict(dropout=.1, lr=.001, weight_decay=5e-4, momentum=.9,
            optimizer='sgd_nesterov', label_scale='source', loss='observed_mse',
            schedule='author_lut', grad_clip=None, effective_batch=32, micro_batch=32,
            max_epochs=120, min_epochs=120, patience=121)
    if method in PRIOR_METHODS:
        return candidate('N1', config_id)
    if method in ('N1H', 'N1R', 'N1RP'):
        return {**candidate('N1S', config_id), 'prompt_residual_scale': 0.02,
                'prompt_shrinkage': 0.01, 'prompt_relation_strength': 0.0 if method == 'N1H' else 0.01,
                'prompt_semantic_weight': 0.25, 'prompt_permutation_seed': 20261026}
    if method in ('N0', 'N1', 'N1S', 'N2', 'N3', 'N4'):
        if config_id != 'C2':
            raise ValueError('G02R prompt candidates have one fixed C2 configuration')
        return {**candidate('M6', 'C2'), 'prompt_rank': 8}
    if method not in METHODS or config_id not in ('C0', 'C1', 'C2', 'C3'):
        raise ValueError('A concrete registered method/config ID is required')
    return json.loads((CONFIG_ROOT / (config_id + '.json')).read_text(encoding='utf-8'))[method]


def resolve_run(row, *, view_manifest, selection=None, explicit_config=None):
    """Resolution never trains; it replaces every planned placeholder with identities."""
    expected_fold = 'F0' if row.get('fold') == 'DEV' and row.get('phase') == 'P2' else str(row.get('fold'))
    if expected_fold != str(view_manifest['fold']):
        raise ValueError('Run fold must bind to the actual training view')
    config_id = row['config_id']
    if explicit_config is None and config_id not in ('C0', 'C1', 'C2', 'C3', 'AUTHOR'):
        if config_id not in ('SELECTED', 'INHERIT'):
            raise ValueError('Unknown configuration placeholder')
        source_method = 'M6' if row['method_id'] in ('A1','A2','A3') else row['method_id']
        if not selection or selection['method_id'] != source_method or str(selection['fold']) != str(row['fold']):
            raise ValueError('Selected config requires matching validation selection record')
        if digest({k:v for k,v in selection.items() if k != 'selection_sha256'}) != selection.get('selection_sha256'):
            raise ValueError('Selection checksum mismatch')
        config_id = selection['config_id']
    result = dict(row)
    config = dict(explicit_config) if explicit_config is not None else candidate(row['method_id'], config_id)
    if row['method_id'] in ('N1', 'N1S', 'M2') + PRIOR_METHODS:
        if config['hidden_dim'] % config['heads'] or min(config['layers'], config['ffn_dim']) < 1:
            raise ValueError('Invalid Graphormer dimensions')
    result.update(config_id=config_id, config=config,
        view_id=view_manifest['view_id'], view_sha256=view_manifest['manifest_sha256'],
        source_sha256=view_manifest['source_sha256'], feature_spec_sha256=PGT_FEATURE_SPEC.sha256,
        cache_id=view_manifest['cache_id'], task_names=view_manifest['task_names'],
        train_sha256=view_manifest['files']['train.npz'], validation_sha256=view_manifest['files']['validation.npz'])
    result['config_sha256'] = digest(result['config'])
    if selection is not None:
        result['selection_sha256'] = selection['selection_sha256']
    result['run_sha256'] = digest(result)
    return result


def git_identity():
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
