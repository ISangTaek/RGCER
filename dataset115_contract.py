"""Read-only dataset115 identity and routing audit. Not a training entry point.

Labels remain in the supplied CSV scale. No inferred dose/unit conversion.
The 1024 trailing numeric columns are not toxicity endpoints.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

CSV_SHA = '2119b3641edc8fc7577f7efbaa4c283aad47bef418f06e50bd26ad64e6a18716'
SPLIT_SHA = '9caffcebed884840ad1817da5413cb1704a306c26ebdcf0f41235faf3cb0dd32'
TOX_SEMANTIC_SHA = '61a2e494469a4035447f237532272487fd897adb3fadae7879e0dc75d0b01085'
PRIMARY = ('child_oral_TDLo', 'human_oral_LDLo', 'human_oral_TDLo', 'man_oral_TDLo', 'women_oral_TDLo')
HUMAN_SPECIES = {'child', 'human', 'man', 'women'}
SPLITS = ('train', 'validation', 'calibration', 'test')
ROW_COUNT = 80172

class ContractError(ValueError):
    pass

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def label(value):
    if not isinstance(value, str):
        raise ContractError('label token must be text')
    if not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ContractError('invalid label token') from exc
    if not math.isfinite(result):
        raise ContractError('nonfinite label')
    return result

def semantic_digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def task_columns(fields):
    if not isinstance(fields, list) or len(fields) != 1141 or len(set(fields)) != len(fields):
        raise ContractError('CSV header count/uniqueness differs')
    if fields[:2] != ['cid', 'smiles'] or fields[117:] != [str(i) for i in range(1024)]:
        raise ContractError('CSV identifier/auxiliary schema differs')
    tasks = fields[2:117]
    if any(len(t.rsplit('_', 2)) != 3 or t.rsplit('_', 1)[1] not in {'LD10', 'LD20', 'LD50', 'LDLo', 'TDLo', 'LC50', 'LCLo', 'TCLo'} for t in tasks):
        raise ContractError('unknown endpoint syntax')
    human = [t for t in tasks if t.rsplit('_', 2)[0] in HUMAN_SPECIES]
    source = [t for t in tasks if t not in human]
    if len(human) != 11 or len(source) != 104 or not set(PRIMARY) <= set(human):
        raise ContractError('Human11/Nonhuman104 identity differs')
    return tasks, human, source

def manifest_records(manifest):
    if manifest.get('split_algorithm') != 'constrained_scaffold_v3' or manifest.get('seed') != 42:
        raise ContractError('split algorithm/seed differs')
    rows = manifest['records']; indices = {}; groups = {}; canonical = set()
    for row in rows:
        i = row['row_index']
        if type(i) is not int or i < 0 or i in indices or row['sample_id'] != f'row_{i}':
            raise ContractError('row identity differs/duplicate')
        split = row['split']; group = row['split_group']; c = row['canonical_smiles']
        if split not in SPLITS or not isinstance(group, str) or not group or not isinstance(c, str) or not c:
            raise ContractError('split/group/canonical invalid')
        if c in canonical:
            raise ContractError('duplicate canonical identity')
        if group in groups and groups[group] != split:
            raise ContractError('group crosses splits')
        indices[i] = row; groups[group] = split; canonical.add(c)
    if set(indices) != set(range(len(rows))):
        raise ContractError('noncontiguous row indices')
    return indices

def namespace_id(source, row_index):
    if source not in {'dataset115', 'toxacute'} or type(row_index) is not int or row_index < 0:
        raise ContractError('invalid namespaced ID')
    return f'{source}:row_{row_index}'

def route_allowed(route, canonical, tox_canonical):
    if route not in {'A', 'B'}:
        raise ContractError('unknown route')
    return route == 'A' or canonical not in tox_canonical

def audit(csv_path, split_path, tox_manifest_path, *, expected_tox_sha):
    if digest(csv_path) != CSV_SHA or digest(split_path) != SPLIT_SHA:
        raise ContractError('frozen dataset115 input SHA differs')
    if len(expected_tox_sha) != 64 or digest(tox_manifest_path) != expected_tox_sha:
        raise ContractError('frozen ToxAcute input SHA differs')
    manifest = json.loads(Path(split_path).read_text(encoding='utf-8'))
    if manifest.get('source_csv_sha256') != CSV_SHA:
        raise ContractError('manifest source CSV differs')
    refs = manifest_records(manifest)
    tox = json.loads(Path(tox_manifest_path).read_text(encoding='utf-8'))
    if semantic_digest(tox) != TOX_SEMANTIC_SHA:
        raise ContractError('frozen ToxAcute semantic manifest identity differs')
    tox_canonical = {r['canonical_smiles'] for r in tox['records']}
    if not tox_canonical or any(not isinstance(c,str) or not c for c in tox_canonical):
        raise ContractError('invalid ToxAcute canonical set')
    counts = {route: {t: Counter(dict.fromkeys(SPLITS, 0)) for t in PRIMARY} for route in ('A','B')}
    molecules = {route: Counter(dict.fromkeys(SPLITS, 0)) for route in ('A','B')}
    source_counts = Counter(); processed = 0; observed = 0
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); tasks, human, source = task_columns(reader.fieldnames)
        source_counts.update(dict.fromkeys(source, 0))
        for i, row in enumerate(reader):
            if None in row or any(v is None for v in row.values()):
                raise ContractError('ragged CSV row')
            if i not in refs or row['smiles'] != refs[i]['raw_smiles'] or not row['cid'].strip():
                raise ContractError('CSV/manifest row alignment differs')
            ref = refs[i]; split = ref['split']; c = ref['canonical_smiles']
            values = {t: label(row[t]) for t in tasks}
            observed += sum(v is not None for v in values.values())
            if split == 'train':
                source_counts.update(t for t in source if values[t] is not None)
            for route in ('A','B'):
                if not route_allowed(route,c,tox_canonical):
                    continue
                molecules[route][split] += 1
                for t in PRIMARY:
                    if values[t] is not None:
                        counts[route][t][split] += 1
            processed += 1
    if processed != ROW_COUNT or processed != len(refs):
        raise ContractError('frozen row count differs')
    return {'scope': 'INPUT_CONTRACT_ONLY_NOT_TRAINING_READY', 'csv_sha256': CSV_SHA,
            'split_sha256': SPLIT_SHA, 'tox_manifest_sha256': expected_tox_sha,
            'tox_manifest_semantic_sha256': TOX_SEMANTIC_SHA,
            'rows':processed, 'task_names':tasks, 'excluded_human11':human,
            'nonhuman104':source, 'primary5':list(PRIMARY), 'auxiliary_columns_ignored':1024,
            'molecule_counts':molecules, 'primary5_counts':counts,
            'source_train_observation_counts':source_counts, 'observed_label_cells':observed,
            'labels': 'CSV values preserved; no additional transform; physical unit provenance not inferred',
            'sample_id_namespace':'dataset115:row_<row_index>',
            'training_executed':False, 'scaler_fitted':False}


def main(argv=None):
    """Write a new audit only; deliberately no training or data-build options."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--split-manifest', required=True, type=Path)
    parser.add_argument('--tox-manifest', required=True, type=Path)
    parser.add_argument('--expected-tox-file-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; choose a new run path')
    try:
        result = audit(args.csv, args.split_manifest, args.tox_manifest,
                       expected_tox_sha=args.expected_tox_file_sha256)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        # Exclusive creation protects existing evidence, including a concurrent writer.
        with args.output.open('x', encoding='utf-8') as f:
            f.write(payload)
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'INPUT_CONTRACT_FAILED: {exc}\n')
    print(f'INPUT_CONTRACT_CHECKED: {result["rows"]} rows; not training ready')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
