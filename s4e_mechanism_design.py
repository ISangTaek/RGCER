"""S4E3 offline scientific lock construction; no model loading or inference.

The collection is an explicitly reviewed, incomplete 043 evidence package.
Its original status is retained. These locks do not authorize GPU execution.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

COLLECTION_CHECKSUM_SHA = '6cc6378621d0225b55e41aa51a9943ab4ef9e034336aa4ca174f8e474a7619e3'
EXTERNAL_MANIFEST_SHA = '9caffcebed884840ad1817da5413cb1704a306c26ebdcf0f41235faf3cb0dd32'
EXTERNAL_CSV_SHA = '2119b3641edc8fc7577f7efbaa4c283aad47bef418f06e50bd26ad64e6a18716'
DATA_FINGERPRINT = '7b7bd62a6457501c8010f12b9bae851ec9c8b265539c93ddca583cf4b952be5c'
MANIFEST_HASH = '61a2e494469a4035447f237532272487fd897adb3fadae7879e0dc75d0b01085'
RAW_CSV_SHA = '47b406217dfbe916b0644a11ca071783791dd8a73f2f93685876560b0ab97eae'
HUMAN = ['man_oral_TDLo', 'women_oral_TDLo', 'human_oral_TDLo']
SPLITS = {'train': 0, 'validation': 1, 'calibration': 2, 'test': 3}
SALT = 'RGCER_S4E3_20260915_v1'


class DesignError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DesignError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, f'duplicate JSON key: {k}')
            result[k] = v
        return result
    def invalid(value):
        raise DesignError(f'nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=pairs, parse_constant=invalid)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def datastore_semantic_identity(datastore):
    """Ignore build-time paths/NPZ container timestamps, not scientific content."""
    meta = read_json(datastore/'datastore.json')
    with np.load(datastore/'index.npz', allow_pickle=False) as z:
        index = {k: z[k].tolist() for k in ('sample_ids','row_indices','split_codes','num_nodes')}
    labels = np.load(datastore/'labels.npy', allow_pickle=False)
    # NaNs encode missing labels. Normalize their bit pattern before hashing.
    labels = labels.copy(); labels[np.isnan(labels)] = np.nan
    require(not np.isinf(labels).any(), 'infinite label')
    return {'metadata': {k: meta[k] for k in ('build_id','datastore_fingerprint','raw_csv_sha256',
        'split_manifest_hash','task_names','feature_schema_version','max_path_distance')},
        'index_content_sha256': canonical_hash(index), 'labels_shape': list(labels.shape),
        'labels_dtype': str(labels.dtype), 'labels_values_sha256': hashlib.sha256(labels.tobytes(order='C')).hexdigest(),
        'split_content_sha256': canonical_hash(read_json(datastore/'split_manifest.json'))}


def unique(items, key):
    result = {}
    for x in items:
        require(x[key] not in result, f'duplicate {key}: {x[key]}')
        result[x[key]] = x
    return result


def checked_collection(root):
    root = Path(root).resolve()
    require(sha(root/'checksums.sha256') == COLLECTION_CHECKSUM_SHA, 'unreviewed collection')
    seen = set()
    for line in (root/'checksums.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        p = (root/name).resolve()
        require(p.is_relative_to(root) and p != root and p not in seen, 'unsafe/duplicate member')
        require(sha(p) == digest, f'collection checksum mismatch: {name}')
        seen.add(p)
    require(len(seen) == 169, 'collection file count differs')


def tensor_signature(inventory):
    tensors = unique(inventory['tensors'], 'key')
    require(len(tensors) == inventory['tensor_count'], 'tensor count differs')
    return {k: (x['shape'], x['dtype'], x['numel'], x['tensor_sha256'])
            for k, x in tensors.items()}


def build_assets(collection, low_policy, core_lock):
    from s4e_source_assets import build_run_specs
    from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
    checked_collection(collection)
    specs = build_run_specs(read_json(low_policy), read_json(core_lock))
    links = unique(read_json(collection/'run_source_links.json')['runs'], 'run_id')
    assets = unique(read_json(collection/'source_assets.json')['assets'], 'asset_id')
    scopes = read_json(collection/'training_scope_evidence.json')
    targets = unique(scopes['targets'], 'run_id')
    require(set(links) == {s.run_id for s in specs} == set(targets), 'run set differs')
    require(len(assets) == 12, 'source asset set differs')
    for group in ('sources', 'targets'):
        for row in scopes[group]:
            f = row['scope']['fields']
            for key, expected in [('datastore_fingerprint', DATA_FINGERPRINT),
                                  ('split_manifest_hash', MANIFEST_HASH), ('dataset', 'toxacute'),
                                  ('max_nodes_filter', 512), ('split_seed', 42),
                                  ('splitting', 'scaffold'), ('fit_conformal', False),
                                  ('train_eval_scope', 'validation_only')]:
                require(f[key]['status'] == 'CONSISTENT' and type(f[key]['value']) is type(expected)
                        and f[key]['value'] == expected, f'scope mismatch: {key}')
    runs = []
    for s in specs:
        x = links[s.run_id]
        require(x['best']['sha256'] == s.checkpoint_sha256 and x['best']['epoch'] == s.best_epoch,
                f'best differs: {s.run_id}')
        init, teacher = [assets[x[k+'_asset_id']] for k in ('init', 'teacher')]
        for k, a in [('init', init), ('teacher', teacher)]:
            require(x[k+'_path_verification']['sha256'] == a['sha256'], 'path SHA differs')
        require(teacher['task_names'] == list(ANIMAL_SOURCE_TASKS) and teacher['seed'] == s.seed,
                'teacher task/seed mismatch')
        require(set(unique(teacher['scalers'], 'task')) == set(ANIMAL_SOURCE_TASKS), 'scaler task mismatch')
        for role in ('backbone', 'encoder_non_backbone'):
            require(tensor_signature(init[role]) == tensor_signature(teacher[role]), 'source/init tensor mismatch')
        sidecar = read_json(collection/'evidence/runs'/s.run_id/'init.provenance.json')
        require(sidecar['output_sha256'] == init['sha256']
                and sidecar['teacher_real_checkpoint_sha256'] == teacher['sha256']
                and sidecar['expected_teacher_epoch'] == teacher['epoch'], 'sidecar source mismatch')
        runtime = targets[s.run_id]['scope']['fields']['train_fraction']
        if x['status'] != 'COLLECTED':
            require(s.fraction == 100 and x['reason_code'] == 'TRAINING_SCOPE_INCOMPLETE'
                    and runtime['status'] == 'UNKNOWN' and runtime['value'] is None,
                    'unadjudicated collection error')
        runs.append({'run_id': s.run_id, 'method': s.method, 'seed': s.seed,
                     'declared_fraction_percent': s.fraction, 'runtime_fraction_evidence': runtime,
                     'raw_collection_status': x['status'], 'best': x['best'],
                     'init_asset_id': init['asset_id'], 'teacher_asset_id': teacher['asset_id']})
    return {'schema': 's4e3_asset_lock_v1', 'runs': runs, 'assets': list(assets.values()),
            'training_scope_evidence': scopes, 'authority': 'Codex limited 043 adjudication',
            'gpu_authorized': False}


def aligned_records(manifest, index):
    byid = unique(manifest['records'], 'sample_id')
    ids = [str(x) for x in index['sample_ids']]
    require(len(set(ids)) == len(ids) and set(ids) == set(byid), 'index sample set differs')
    require(np.array_equal(index['row_indices'], np.arange(len(ids))), 'global indices not contiguous')
    rows = []
    for i, sid in enumerate(ids):
        x = byid[sid]
        require(int(index['split_codes'][i]) == SPLITS[x['split']], 'index split differs')
        # Manifest row_index is ORIGINAL CSV row, not DataStore's compact index.
        rows.append(dict(x, global_index=i, num_nodes=int(index['num_nodes'][i])))
    return rows


def choose(canonicals, count):
    require(type(count) is int and count >= 0 and len(set(canonicals)) == len(canonicals), 'invalid selection input')
    require(len(canonicals) >= count, 'insufficient eligible candidates')
    return sorted(canonicals, key=lambda c: (hashlib.sha256((SALT+'\0'+c).encode()).hexdigest(), c))[:count]


def freeze_samples(datastore, external_manifest, external_csv, raw_csv, animal_tasks):
    from rdkit import Chem, rdBase
    require(sha(external_manifest) == EXTERNAL_MANIFEST_SHA, 'external split identity differs')
    require(sha(external_csv) == EXTERNAL_CSV_SHA and sha(raw_csv) == RAW_CSV_SHA, 'raw CSV identity differs')
    meta = read_json(datastore/'datastore.json')
    m, e = read_json(datastore/'split_manifest.json'), read_json(external_manifest)
    require(meta['datastore_fingerprint'] == DATA_FINGERPRINT
            and canonical_hash(m) == MANIFEST_HASH, 'DataStore identity differs')
    require(set(meta['task_names']) == set(animal_tasks) | set(HUMAN), 'task set differs')
    require(len(set(meta['task_names'])) == 59, 'duplicate task')
    with np.load(datastore/'index.npz', allow_pickle=False) as z:
        index = {k: z[k] for k in z.files}
    labels = np.load(datastore/'labels.npy', allow_pickle=False)
    rows = aligned_records(m, index)
    require(labels.shape == (len(rows), 59), 'label shape differs')
    byraw = unique(rows, 'row_index')
    # Verify labels from the raw CSV independently, not only the cached NPY.
    with raw_csv.open(encoding='utf-8-sig', newline='') as f:
        checked = 0
        for n, x in enumerate(csv.DictReader(f)):
            if n not in byraw:
                continue
            rec = byraw[n]
            require(x['smiles'] == rec['raw_smiles'], 'raw smiles alignment differs')
            vals = np.array([float(x[t]) if x[t].strip() else np.nan for t in meta['task_names']], dtype=labels.dtype)
            require(np.array_equal(vals, labels[rec['global_index']], equal_nan=True), 'raw label alignment differs')
            checked += 1
    require(checked == len(rows), 'raw CSV coverage differs')
    train = {x['canonical_smiles'] for x in rows if x['split'] == 'train'}
    train_groups = {x['split_group'] for x in rows if x['split'] == 'train'}
    train_scaffolds = {x['scaffold'] for x in rows if x['split'] == 'train' and x['scaffold'] != '__ACYCLIC__'}
    alltox = {x['canonical_smiles'] for x in rows}
    hi = [meta['task_names'].index(t) for t in HUMAN]
    ai = [meta['task_names'].index(t) for t in animal_tasks]
    H, A, refs = set(), set(), {}
    for x in rows:
        c = x['canonical_smiles']
        if x['split'] != 'test' or c in train or x['num_nodes'] > 512:
            continue
        refs.setdefault(c, []).append(dict(x, source='toxacute'))
        if np.isfinite(labels[x['global_index'], hi]).any(): H.add(c)
        if np.isfinite(labels[x['global_index'], ai]).any(): A.add(c)
    extbyrow = unique(e['records'], 'row_index')
    E, excluded = set(), {'invalid_molecule': 0, 'over_512_nodes': 0}
    with external_csv.open(encoding='utf-8-sig', newline='') as f:
        ext_count = 0
        for n, x in enumerate(csv.DictReader(f)):
            rec = extbyrow[n]; ext_count += 1
            require(x['smiles'] == rec['raw_smiles'], 'external raw alignment differs')
            c = rec['canonical_smiles']
            if rec['split'] != 'test' or c in alltox: continue
            mol = Chem.MolFromSmiles(rec['raw_smiles'])
            if mol is None:
                excluded['invalid_molecule'] += 1; continue
            if mol.GetNumAtoms() > 512:
                excluded['over_512_nodes'] += 1; continue
            E.add(c)
            refs.setdefault(c, []).append(dict(rec, source='dataset115', num_nodes=mol.GetNumAtoms()))
    require(ext_count == len(extbyrow), 'external CSV coverage differs')
    require(len(H) == 28, 'Human3 eligibility changed; scientific re-review required')
    selections = [('human3_test', choose(H, len(H))), ('animal56_test_only', choose(A-H, 100)),
                  ('dataset115_external_test', choose(E, 100))]
    probe = []
    for stratum, selected in selections:
        for c in selected:
            sr = sorted(refs[c], key=lambda x: (x['source'], x['row_index']))
            probe.append({'sample_key': hashlib.sha256(c.encode()).hexdigest(), 'canonical_smiles': c,
                          'stratum': stratum, 'human3_test': c in H, 'animal56_test': c in A,
                          'external_only': c in E, 'input_record': sr[0], 'source_records': sr,
                          'overlap_train_canonical': c in train,
                          'overlap_train_group': any(x['split_group'] in train_groups for x in sr),
                          'overlap_train_cyclic_scaffold': any(x['scaffold'] in train_scaffolds for x in sr)})
    require(len(probe) == len({x['sample_key'] for x in probe}) == 228, 'probe duplicate/count')
    require(not any(x['overlap_train_canonical'] for x in probe), 'training leakage')
    validation = []
    for task, j in zip(animal_tasks, ai):
        observations = [{'sample_id': x['sample_id'], 'global_index': x['global_index'],
                         'raw_row_index': x['row_index'], 'canonical_smiles': x['canonical_smiles'],
                         'label_raw': float(labels[x['global_index'], j]), 'mask': True}
                        for x in rows if x['split'] == 'validation' and x['num_nodes'] <= 512
                        and np.isfinite(labels[x['global_index'], j])]
        require(bool(observations), f'empty validation task: {task}')
        require(not any(x['canonical_smiles'] in train for x in observations), 'validation training overlap')
        validation.append({'task': task, 'n': len(observations), 'observations': observations})
    eligibility = {'rdkit_version': rdBase.rdkitVersion, 'predictions_accessed': False,
                   'raw_label_alignment_rows': checked, 'train_canonical_upper_bound_count': len(train),
                   'human3_test_eligible': len(H), 'animal56_test_eligible': len(A),
                   'animal_excluding_human_eligible': len(A-H), 'external_test_eligible': len(E),
                   'external_exclusions': excluded, 'selection_salt': SALT,
                   'selection': 'ascending SHA256(salt+NUL+canonical), lexical tie break; no error ranking',
                   'probe_count': len(probe), 'validation_observations': sum(x['n'] for x in validation),
                   'training_fraction_imputed': False,
                   'training_bound': 'all frozen ToxAcute train canonical molecules, without label/node filtering',
                   'runtime_train_ids_persisted': False}
    return {'schema': 's4e3_probe_v1', 'ordered_samples': probe}, eligibility, {
        'schema': 's4e3_animal_validation_v1', 'task_order': list(animal_tasks), 'tasks': validation}


def build(collection, low_policy, core_lock, datastore, external_manifest, external_csv, raw_csv, output):
    from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
    require(not output.exists(), 'output exists; use a fresh directory')
    asset_lock = build_assets(collection, low_policy, core_lock)
    probe, eligibility, validation = freeze_samples(datastore, external_manifest, external_csv, raw_csv, ANIMAL_SOURCE_TASKS)
    inputs = {name: sha(p) for name, p in {
        'collection_checksums': collection/'checksums.sha256', 'low_policy': low_policy,
        'core_lock': core_lock, 'datastore_metadata': datastore/'datastore.json',
        'datastore_index': datastore/'index.npz', 'datastore_labels': datastore/'labels.npy',
        'datastore_split': datastore/'split_manifest.json', 'external_manifest': external_manifest,
        'external_csv': external_csv, 'raw_csv': raw_csv}.items()}
    output.mkdir(parents=True)
    documents = {'mechanism_asset_lock.json': asset_lock, 'probe_manifest.json': probe,
                 'probe_eligibility.json': eligibility, 'animal56_validation_manifest.json': validation}
    for name, payload in documents.items(): write_json(output/name, payload)
    write_json(output/'design_manifest.json', {'schema': 's4e3_design_v1', 'inputs': inputs,
        'datastore_semantic_identity': datastore_semantic_identity(datastore),
        'outputs': {name: sha(output/name) for name in documents}, 'gpu_authorized': False,
        'evidence_level': 'OFFLINE_DATA_AND_COLLECTED_METADATA', 'formal_mechanism_metrics_computed': False})
    return eligibility
