"""Four registered candidates and fully resolved run identities."""
import json
import subprocess
from pathlib import Path
from .chemistry import digest
from molecular_features import PGT_FEATURE_SPEC

CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'configs' / 'pgt_v2'
METHODS = ('M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'A1', 'A2', 'A3', 'N0', 'N1', 'N2', 'N3', 'N4')


def candidate(method, config_id):
    if method in ('N0', 'N1', 'N2', 'N3', 'N4'):
        if config_id != 'C2':
            raise ValueError('G02R prompt candidates have one fixed C2 configuration')
        return {**candidate('M6', 'C2'), 'prompt_rank': 8}
    if method not in METHODS or config_id not in ('C0', 'C1', 'C2', 'C3'):
        raise ValueError('A concrete registered method/config ID is required')
    return json.loads((CONFIG_ROOT / (config_id + '.json')).read_text(encoding='utf-8'))[method]


def resolve_run(row, *, view_manifest, selection=None):
    """Resolution never trains; it replaces every planned placeholder with identities."""
    expected_fold = 'F0' if row.get('fold') == 'DEV' and row.get('phase') == 'P2' else str(row.get('fold'))
    if expected_fold != str(view_manifest['fold']):
        raise ValueError('Run fold must bind to the actual training view')
    config_id = row['config_id']
    if config_id not in ('C0', 'C1', 'C2', 'C3'):
        if config_id not in ('SELECTED', 'INHERIT'):
            raise ValueError('Unknown configuration placeholder')
        source_method = 'M6' if row['method_id'] in ('A1','A2','A3') else row['method_id']
        if not selection or selection['method_id'] != source_method or str(selection['fold']) != str(row['fold']):
            raise ValueError('Selected config requires matching validation selection record')
        if digest({k:v for k,v in selection.items() if k != 'selection_sha256'}) != selection.get('selection_sha256'):
            raise ValueError('Selection checksum mismatch')
        config_id = selection['config_id']
    result = dict(row)
    result.update(config_id=config_id, config=candidate(row['method_id'], config_id),
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
