"""Pinned P2 membership and budget contracts, never a training authorization."""
from pathlib import Path
import hashlib
import json
import math

PROTOCOL_SHA = '4c5133d8a3e1628021eab06c53ffadc63cd112203343b4c6d5ee6e4411392488'
MEMBERS_SHA = 'ba75d9b8026bc0b73759ce5667918d3567577a7fdf99990de96e06fd4b8fe319'
FINGERPRINT = '7b7bd62a6457501c8010f12b9bae851ec9c8b265539c93ddca583cf4b952be5c'
TASKS = ('man_oral_TDLo', 'women_oral_TDLo', 'human_oral_TDLo')
METHODS = ('FROZEN', 'B1_low', 'HF_low')
PANELS = ('mouse', 'rat')
ROUTES = ('oral', 'intraperitoneal')


class P2Error(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise P2Error(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key: ' + key)
            out[key] = value
        return out
    def bad(value):
        raise P2Error('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def bound_json(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'frozen file SHA mismatch')
    return strict_json(raw)


def run_matrix():
    runs = []
    for panel in PANELS:
        for route in ROUTES:
            for seed in range(42, 47):
                sid = f'P2_{panel}_{route}_s{seed}_source'
                runs.append(dict(run_id=sid, role='source', panel=panel, route=route,
                                 seed=seed, epochs=40, updates=440))
                for method in METHODS:
                    runs.append(dict(run_id=f'P2_{panel}_{route}_s{seed}_{method}',
                                     role='target', panel=panel, route=route, seed=seed,
                                     method=method, source_run_id=sid, epochs=40, updates=240))
    return runs


def validate_members(members):
    require(members['schema'] == 'p2a_members_v1', 'membership schema')
    require(members['data_fingerprint'] == FINGERPRINT, 'data fingerprint')
    require(set(members['panels']) == set(PANELS), 'panel set')
    previous = set()
    for panel in PANELS:
        rows = members['panels'][panel]
        require(len(rows) == 700, 'source membership count')
        groups = [r['group'] for r in rows]
        canonical = [r['canonical'] for r in rows]
        require(len(set(groups)) == len(set(canonical)) == 700, 'one structure per group')
        require(not previous.intersection(groups), 'source panel group overlap')
        previous.update(groups)
        for row in rows:
            require(set(row['observations']) == set(ROUTES), 'route membership')
            for obs in row['observations'].values():
                require(type(obs['value']) in (int, float) and math.isfinite(obs['value']), 'source label')
                require(type(obs['global_index']) is int and obs['global_index'] >= 0, 'global index')
    require(set(members['target']) == {'train', 'validation'}, 'target split permissions')
    for split, counts in [('train', (96, 85, 81)), ('validation', (14, 12, 13))]:
        require(set(members['target'][split]) == set(TASKS), 'target task permissions')
        for task, n in zip(TASKS, counts):
            rows = members['target'][split][task]
            require(len(rows) == len({r['sample_id'] for r in rows}) == n, 'target population')


def load_contract(protocol_path, members_path):
    protocol = bound_json(protocol_path, PROTOCOL_SHA)
    members = bound_json(members_path, MEMBERS_SHA)
    validate_members(members)
    require(protocol['execution_authorized'] is False, 'scientific contract is not execution authorization')
    require(protocol['members']['sha256'] == MEMBERS_SHA, 'membership reference')
    expected = run_matrix()
    require(len(protocol['runs']) == len(expected), 'run matrix size')
    for actual, wanted in zip(protocol['runs'], expected):
        require(all(type(actual[k]) is type(v) and actual[k] == v for k, v in wanted.items()), 'run matrix identity')
    return protocol, members


def validate_datastore(members, datastore_root):
    """Read only pinned source labels and target train/validation; no graph load."""
    import numpy as np
    root = Path(datastore_root)
    metadata = strict_json((root / 'datastore.json').read_bytes())
    require(metadata['datastore_fingerprint'] == FINGERPRINT, 'DataStore fingerprint')
    require(metadata['split_manifest_hash'] == members['split_manifest_hash'], 'split identity')
    manifest = strict_json((root / 'split_manifest.json').read_bytes())
    from split_manifest import manifest_hash
    require(manifest_hash(manifest) == members['split_manifest_hash'], 'actual split identity')
    records = {r['sample_id']: r for r in manifest['records']}
    with np.load(root / 'index.npz', allow_pickle=False) as index:
        ids = index['sample_ids'].tolist()
        codes = index['split_codes'].copy()
        nodes = index['num_nodes'].copy()
    require(len(ids) == len(set(ids)) == len(records), 'DataStore sample IDs')
    require(set(ids) == set(records), 'DataStore population')
    position = {sid: i for i, sid in enumerate(ids)}
    labels = np.load(root / 'labels.npy', mmap_mode='r', allow_pickle=False)
    target_labels = strict_json((Path(__file__).resolve().parent / 'configs/p2_target_label_identity.json').read_bytes())
    require(labels.shape == (len(ids), len(metadata['task_names'])), 'label shape')
    checked = 0
    source_groups = set()
    def check(sid, canonical, group, code):
        require(sid in position, 'unknown sample')
        i = position[sid]; r = records[sid]
        require(r['canonical_smiles'] == canonical and r['split_group'] == group, 'structure identity')
        require(int(codes[i]) == code and int(nodes[i]) <= 512, 'split/node eligibility')
        require(r['split'] == {0:'train', 1:'validation'}[code], 'manifest split')
        return i
    for panel in PANELS:
        for row in members['panels'][panel]:
            source_groups.add(row['group'])
            for route in ROUTES:
                obs = row['observations'][route]
                i = check(obs['sample_id'], row['canonical'], row['group'], 0)
                require(i == obs['global_index'], 'DataStore position')
                column = metadata['task_names'].index(f'{panel}_{route}_LD50')
                require(float(labels[i, column]) == obs['value'], 'actual source label')
                checked += 1
    for r in records.values():
        require(r['split'] == 'train' or r['split_group'] not in source_groups, 'heldout group leakage')
    for split, code in [('train', 0), ('validation', 1)]:
        for task in TASKS:
            column = metadata['task_names'].index(task)
            expected = {ids[i] for i in np.flatnonzero(codes == code) if np.isfinite(labels[i, column])}
            actual = sorted([[sid,float(labels[position[sid],column])] for sid in expected])
            raw = json.dumps(actual,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            require(hashlib.sha256(raw).hexdigest()==target_labels[split][task], 'actual target labels differ')
            rows = members['target'][split][task]
            require(expected == {r['sample_id'] for r in rows}, 'target label population')
            for r in rows:
                check(r['sample_id'], r['canonical'], r['group'], code)
    return dict(scope='MEMBERSHIP_LABELS_ONLY_NO_MODEL',source_observations=checked,
                test_accessed=False,calibration_accessed=False,training_authorized=False)
