"""Trusted 087 inputs and explicit target-label/group holdout qualifications."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import json

import numpy as np
import torch

from v12_analytic import require

REPO = Path(__file__).resolve().parent
LOCK = REPO / 'configs/v12_s1_input_lock.json'
SCENES = ('ToxAcute', 'A', 'B')
GRAPH_COMMIT = '9111793be3adf87ca335be8f44aa86065e38427e'
CHEMICAL_COMMIT = '8a58dc56c6111675fcc7ef8421a57c316e274122'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def safe_file(root, name):
    part = PurePosixPath(name)
    require(isinstance(name, str) and name and not part.is_absolute() and '..' not in part.parts
            and ':' not in name and '\\' not in name and name == part.as_posix(), 'unsafe input path')
    root = Path(root).resolve()
    path = root.joinpath(*part.parts)
    require(path.resolve().is_relative_to(root)
            and not any(root.joinpath(*part.parts[:i]).is_symlink() for i in range(1, len(part.parts)+1)),
            'input symlink/escape')
    require(path.is_file(), 'missing frozen input: '+name)
    return path


def check_inputs(root):
    locked = read(LOCK)
    require(locked['schema'] == 'v12_s1_input_lock_v1' and locked['reference_commit'] == CHEMICAL_COMMIT,
            'V12 lock schema/commit')
    for name, expected in locked['files'].items():
        require(sha(safe_file(root, name)) == expected, 'frozen input SHA: '+name)
    return locked


def fold_id(group, n_folds=3):
    require(isinstance(group, str) and group and n_folds == 3, 'fixed group fold contract')
    return int(hashlib.sha256(('V12S1/42/'+group).encode()).hexdigest()[:16], 16) % n_folds


@dataclass
class Scene:
    name: str
    rows: list
    sources: list
    targets: list
    functions: np.ndarray
    canonical: list
    row_chemical: np.ndarray
    fp_packed: np.ndarray
    descriptors: np.ndarray
    missing: np.ndarray
    descriptor_names: list
    identity: dict

    def indices(self, tasks, split='train'):
        return np.asarray([i for i, r in enumerate(self.rows) if r['task'] in tasks and r['split'] == split], dtype=np.int64)

    def qualification(self):
        source = self.indices(self.sources)
        target = self.indices(self.targets)
        overlap = {self.rows[i]['canonical'] for i in source} & {self.rows[i]['canonical'] for i in target}
        folds = {}
        for task in self.targets:
            ids = self.indices([task])
            folds[task] = {str(k): sum(fold_id(self.rows[i]['group']) == k for i in ids) for k in range(3)}
            for k in range(3):
                require(folds[task][str(k)] > 0 and len(ids)-folds[task][str(k)] >= 2,
                        'insufficient fixed-fold target task coverage: '+task)
        return dict(scene=self.name, sources=len(self.sources), targets=len(self.targets),
                    source_observations=len(source), target_observations=len(target),
                    validation_observations=len(self.indices(self.targets, 'validation')),
                    target_train_shared_source_molecules=len(overlap), target_fold_counts=folds,
                    source_teacher='FULL_SOURCE_TRAIN_NOT_STRICT_GROUP_OOF',
                    c3_cv_scope='TARGET_LABEL_HOLDOUT_CONDITIONAL_ON_FIXED_SOURCE_ASSET',
                    c1_relation_scope='OBSERVED_SOURCE_LABEL_RELATION_NOT_DEPLOYABLE_PREDICTION',
                    c4_status='RECORD_LEVEL_SEMANTICS_NOT_CERTIFIED',
                    identity=self.identity)


def validate_scene(scene):
    require(scene.name in SCENES and scene.sources and scene.targets
            and len(set(scene.sources+scene.targets)) == len(scene.sources+scene.targets), 'scene/task identity')
    n = len(scene.rows)
    require(scene.functions.shape == (n, len(scene.sources)) and np.isfinite(scene.functions).all(),
            'source function shape/finite')
    require(scene.row_chemical.shape == (n,) and np.issubdtype(scene.row_chemical.dtype, np.integer),
            'row chemistry index type')
    require(scene.fp_packed.dtype == np.uint8 and scene.fp_packed.shape == (len(scene.canonical), 256),
            'packed Morgan fingerprint')
    require(scene.descriptors.shape == scene.missing.shape == (len(scene.canonical), 24)
            and scene.missing.dtype == bool and np.isfinite(scene.descriptors).all()
            and len(scene.descriptor_names) == len(set(scene.descriptor_names)) == 24,
            'raw descriptors/missing mask')
    require(len(set(scene.canonical)) == len(scene.canonical), 'duplicate chemical table')
    seen, cells, groups = set(), set(), {}
    for i, row in enumerate(scene.rows):
        require(set(row) == {'task', 'sample_id', 'canonical', 'group', 'split', 'label'}, 'row schema')
        require(all(isinstance(row[k], str) and row[k] for k in ('task', 'sample_id', 'canonical', 'group', 'split'))
                and type(row['label']) in (int, float) and np.isfinite(row['label']), 'row values')
        require(row['task'] in scene.sources+scene.targets and row['split'] in ('train', 'validation')
                and (row['split'] == 'train' or row['task'] in scene.targets), 'row task/split scope')
        key, cell = (row['task'], row['sample_id']), (row['task'], row['canonical'])
        require(key not in seen and cell not in cells, 'duplicate observation/canonical task cell')
        seen.add(key); cells.add(cell)
        c = int(scene.row_chemical[i])
        require(0 <= c < len(scene.canonical) and scene.canonical[c] == row['canonical'], 'canonical feature alignment')
        require(row['canonical'] not in groups or groups[row['canonical']] == row['group'], 'canonical group disagreement')
        groups[row['canonical']] = row['group']
    for task in scene.sources+scene.targets:
        require(len(scene.indices([task])) >= 2, 'missing source/target train task')
    train = scene.indices(scene.targets)
    validation = scene.indices(scene.targets, 'validation')
    for field in ('canonical', 'group'):
        require({scene.rows[i][field] for i in train}.isdisjoint({scene.rows[i][field] for i in validation}),
                'target train/validation overlap: '+field)
    scene.qualification()
    return scene


def load_scene(root, scene_name, *, locked=None):
    require(scene_name in SCENES, 'fixed scene')
    # Even a caller supplying a parsed lock cannot bypass comparison to the repository lock.
    expected_lock = read(LOCK)
    if locked is not None:
        require(locked == expected_lock, 'caller lock differs from trusted lock')
    locked = expected_lock
    entry = locked['scenes'][scene_name]
    needed = [entry[k] for k in ('graph_manifest', 'graph_cache', 'chemical_manifest', 'chemical_cache',
                                'source_rows', 'target_rows', 'validation_rows', 'reference_predictions')]
    for name in needed:
        require(name in locked['files'] and sha(safe_file(root, name)) == locked['files'][name], 'scene input SHA: '+name)
    gm, cm = read(Path(root)/entry['graph_manifest']), read(Path(root)/entry['chemical_manifest'])
    require(gm['commit'] == GRAPH_COMMIT and cm['commit'] == CHEMICAL_COMMIT
            and gm['sha256'] == locked['files'][entry['graph_cache']]
            and cm['chemical_sha256'] == locked['files'][entry['chemical_cache']], 'cache manifest commit/hash')
    value = torch.load(Path(root)/entry['graph_cache'], map_location='cpu', weights_only=True)
    chemistry = torch.load(Path(root)/entry['chemical_cache'], map_location='cpu', weights_only=True)
    require(value['commit'] == GRAPH_COMMIT and value['identity'] == gm['identity']
            and value['sources'] == gm['source_tasks'] and value['targets'] == gm['target_tasks']
            and chemistry['spec'] == cm['spec'] == locked['feature_spec']
            and cm['graph_manifest_sha256'] == locked['files'][entry['graph_manifest']], 'cache identity/specification')
    rows = value['rows']
    for key, select in (
        ('source_rows', lambda r: r['task'] in value['sources']),
        ('target_rows', lambda r: r['task'] in value['targets'] and r['split'] == 'train'),
        ('validation_rows', lambda r: r['split'] == 'validation'),
    ):
        expected = read(Path(root)/entry[key])
        order = lambda r: (r['task'], r['sample_id'])
        require(sorted([r for r in rows if select(r)], key=order) == sorted(expected, key=order),
                'external observation identity: '+key)
    result = Scene(scene_name, rows, value['sources'], value['targets'], value['functions'].numpy().astype(np.float64),
                   chemistry['canonical'], chemistry['row_indices'].numpy(), chemistry['fp_packed'].numpy(),
                   chemistry['descriptors'].numpy(), chemistry['missing'].numpy(), chemistry['spec']['descriptors'],
                   dict(source=value['identity']['source'], graph_cache_sha256=locked['files'][entry['graph_cache']],
                        chemical_cache_sha256=locked['files'][entry['chemical_cache']], input_lock_sha256=sha(LOCK)))
    validate_scene(result)
    require({k: result.qualification()[k] for k in entry['expected_counts']} == entry['expected_counts'],
            'frozen scene population')
    return result


def chemistry_features(scene):
    """Fixed random fingerprint projection; descriptor medians fit on source only.

    Returns one feature row per unique canonical. The inherited V11 statistics
    included all target train observations and are intentionally not used here.
    """
    source_rows = scene.indices(scene.sources)
    fit_ids = np.unique(scene.row_chemical[source_rows])
    raw, missing = scene.descriptors, scene.missing
    median = np.array([np.median(raw[fit_ids[~missing[fit_ids, j]], j])
                       if (~missing[fit_ids, j]).any() else 0. for j in range(24)])
    projection = np.random.default_rng(42).choice(np.array([-1., 1.], dtype=np.float32), size=(2048, 32))/np.sqrt(np.float32(32.))
    compressed = []
    for start in range(0, len(raw), 512):
        bits = np.unpackbits(scene.fp_packed[start:start+512], axis=1, bitorder='little')
        compressed.append(bits.astype(np.float32) @ projection)
    q = np.column_stack((np.concatenate(compressed).astype(np.float64), np.where(missing, median, raw), missing.astype(float)))
    return q, dict(seed=42, fingerprint_projection_width=32, source_unique_canonical_count=len(fit_ids),
                   descriptor_medians=median.tolist(), fit_canonical_sha256=digest([scene.canonical[i] for i in fit_ids]),
                   projection_sha256=hashlib.sha256(projection.tobytes()).hexdigest())


def metrics(rows, expected):
    key = lambda r: (r['task'], r['sample_id'])
    require(len(rows) == len(expected) and len({key(r) for r in rows}) == len(rows), 'prediction population size')
    actual = {key(r): r for r in rows}
    endpoints = {}
    for record in expected:
        require(key(record) in actual, 'missing prediction identity')
        r = actual[key(record)]
        require(set(r) == set(record) | {'prediction'} and all(r[k] == v for k, v in record.items())
                and type(r['prediction']) in (int, float) and np.isfinite(r['prediction']), 'prediction identity/finite')
        endpoints.setdefault(r['task'], []).append((r['prediction']-r['label'])**2)
    result = {t: dict(n=len(errors), rmse=float(np.sqrt(np.mean(errors)))) for t, errors in sorted(endpoints.items())}
    return dict(n=len(rows), endpoints=result, macro_rmse=float(np.mean([v['rmse'] for v in result.values()])))
