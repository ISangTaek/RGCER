"""Small explicit artifacts; train/validation/test labels are separate files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import zipfile

import numpy as np

from molecular_features import PGT_FEATURE_SPEC
from .cache import file_sha256
from .chemistry import digest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seal_manifest(root, meta, files):
    root = Path(root)
    result = {**meta, 'files': {name: file_sha256(root / name) for name in files}}
    result['manifest_sha256'] = digest(result)
    write_json(root / 'manifest.json', result)
    return result


def load_manifest(root):
    value = read_json(Path(root) / 'manifest.json')
    if digest({k: v for k, v in value.items() if k != 'manifest_sha256'}) != value['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    if value['feature_spec_sha256'] != PGT_FEATURE_SPEC.sha256:
        raise ValueError('Manifest FeatureSpec mismatch')
    return value


def load_arrays(root, name, manifest):
    path = Path(root) / name
    if name not in manifest['files'] or file_sha256(path) != manifest['files'][name]:
        raise ValueError('Partition checksum mismatch: ' + name)
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def save_view(root, view, *, cache_manifest, records, roles, labels, fold):
    """Labels already aligned to view.task_names. Never aggregate source observations."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (len(records), len(view.task_names)):
        raise ValueError('Label order/shape mismatch')
    common = {key: np.asarray([r[key] for r in records]) for key in
              ('record_id', 'cache_offset', 'raw_row_index', 'group_id', 'chemical_id')}
    train_lookup = {r: i for i, r in enumerate(view.train_record_ids)}
    files = []
    for role in ('train', 'validation', 'test'):
        index = np.where(np.asarray(roles) == role)[0]
        part = {key: value[index] for key, value in common.items()}
        truth = labels[index].copy()
        if role == 'train':
            truth = view.visible_labels[[train_lookup[r] for r in part['record_id']]].copy()
        np.savez_compressed(root / (role + '.npz'), **part, labels=truth)
        files.append(role + '.npz')
        np.savez_compressed(root / (role + '_inputs.npz'), **part)
        files.append(role + '_inputs.npz')
    np.savez_compressed(root / 'statistics.npz', mean=view.mean, scale=view.scale, count=view.count,
        group_count=view.group_count, scale_fallback=view.scale_fallback, adjacency=view.adjacency)
    files.append('statistics.npz')
    return seal_manifest(root, dict(format='PGTV2_VIEW_1', view_id=view.view_id, fold=str(fold),
        task_names=list(view.task_names), source_sha256=view.source_sha256, feature_spec_sha256=view.feature_spec_sha256,
        cache_id=cache_manifest['cache_id'], active_train_rows=int(view.visible_mask.any(1).sum()),
        visible_labels=int(view.count.sum())), files)


def package_files(files, destination):
    """Explicit named files only. Verify every archived byte against its source."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    items = [(str(name).replace('\\', '/'), Path(path).resolve()) for name, path in files.items()]
    if len({name for name, _ in items}) != len(items):
        raise ValueError('Archive entry collision')
    for name, path in items:
        if name == 'PACKAGE_SHA256.json' or name.startswith('/') or '..' in Path(name).parts or ':' in name or not path.is_file() or path.is_symlink():
            raise ValueError('Unsafe or missing archive member')
        if path.suffix.lower() in ('.pem', '.key', '.env') or path.name in ('id_rsa', 'id_ed25519'):
            raise ValueError('Credential files cannot be packaged')
    manifest = {name: file_sha256(path) for name, path in items}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in items:
            archive.write(path, name)
        archive.writestr('PACKAGE_SHA256.json', json.dumps(manifest, indent=2))
    import hashlib
    with zipfile.ZipFile(destination) as archive:
        if set(archive.namelist()) != set(manifest) | {'PACKAGE_SHA256.json'}:
            raise ValueError('Archive member mismatch')
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Archive content mismatch')
    return {'path': str(destination), 'sha256': file_sha256(destination), 'files': len(items)}
