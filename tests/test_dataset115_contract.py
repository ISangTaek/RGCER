import pytest
from dataset115_contract import ContractError, label, manifest_records, namespace_id, route_allowed, task_columns
from dataset115_contract import PRIMARY, semantic_digest

def header():
    human = list(PRIMARY) + ['child_oral_LDLo', 'human_intravenous_TDLo',
        'man_intravenous_TDLo', 'man_oral_LDLo', 'man_unreported_LDLo', 'women_oral_LDLo']
    source = ['mouse_intraperitoneal_LD10', 'mouse_intraperitoneal_LD20', 'mouse_oral_LD10']
    source += [f'rat_route{i}_LD50' for i in range(101)]
    return ['cid', 'smiles'] + human + source + list(map(str, range(1024)))

def test_valid_header_low_dose_endpoints_and_source_exclusion():
    tasks, human, source = task_columns(header())
    assert len(tasks) == 115 and len(human) == 11 and len(source) == 104
    assert not set(human) & set(source)
    assert 'mouse_intraperitoneal_LD20' in source

def test_reordered_tasks_are_identified_by_name():
    h = header(); h[2:117] = reversed(h[2:117])
    tasks, human, source = task_columns(h)
    assert set(PRIMARY) <= set(human)
    assert not set(PRIMARY) & set(source)

def test_semantic_identity_is_not_serialized_byte_identity():
    import hashlib, json
    value = {'b': 2, 'a': 1}
    assert semantic_digest(value) == semantic_digest({'a': 1, 'b': 2})
    assert semantic_digest(value) != hashlib.sha256(json.dumps(value, indent=2).encode()).hexdigest()

def manifest():
    return {'split_algorithm':'constrained_scaffold_v3','seed':42,'records':[
        {'row_index':0,'sample_id':'row_0','canonical_smiles':'CC','raw_smiles':'CC','split_group':'g1','split':'train'},
        {'row_index':1,'sample_id':'row_1','canonical_smiles':'CCC','raw_smiles':'CCC','split_group':'g2','split':'test'}]}

def test_missing_is_not_zero():
    assert label(' ') is None
    assert label('0') == 0
    assert label('-1.5') == -1.5

@pytest.mark.parametrize('v',['NaN','inf','-inf','junk'])
def test_bad_label(v):
    with pytest.raises(ContractError):label(v)

def test_identity_namespace():
    assert namespace_id('dataset115',0) != namespace_id('toxacute',0)

def test_manifest_valid():
    assert len(manifest_records(manifest()))==2

def test_group_leak_rejected():
    m=manifest();m['records'][1]['split_group']='g1'
    with pytest.raises(ContractError):manifest_records(m)

def test_duplicate_index_rejected():
    m=manifest();m['records'][1]['row_index']=0
    with pytest.raises(ContractError):manifest_records(m)

def test_duplicate_canonical_rejected():
    m=manifest();m['records'][1]['canonical_smiles']='CC'
    with pytest.raises(ContractError):manifest_records(m)

def test_route_b_removes_overlap():
    for split in ('train','validation','calibration','test'):
        assert not route_allowed('B','CC',{'CC'})
    assert route_allowed('A','CC',{'CC'})
    assert route_allowed('B','CCC',{'CC'})

def test_bad_route():
    with pytest.raises(ContractError):route_allowed('C','CC',set())

def test_bad_header():
    with pytest.raises(ContractError):task_columns(['cid','smiles','0'])

def test_empty_header():
    with pytest.raises(ContractError):task_columns(None)

def test_cli_does_not_overwrite(tmp_path):
    from dataset115_contract import main
    output = tmp_path / 'audit.json'; output.write_text('original')
    with pytest.raises(SystemExit) as exc:
        main(['--csv', 'missing', '--split-manifest', 'missing', '--tox-manifest', 'missing',
              '--expected-tox-file-sha256', '0' * 64, '--output', str(output)])
    assert exc.value.code == 2
    assert output.read_text() == 'original'

def test_bad_input_has_no_success_output(tmp_path):
    from dataset115_contract import main
    output = tmp_path / 'audit.json'
    with pytest.raises(SystemExit) as exc:
        main(['--csv', str(tmp_path / 'absent'), '--split-manifest', 'missing', '--tox-manifest', 'missing',
              '--expected-tox-file-sha256', '0' * 64, '--output', str(output)])
    assert exc.value.code == 2
    assert not output.exists()

@pytest.fixture
def audit_inputs(tmp_path, monkeypatch):
    import csv, json
    import dataset115_contract as dc
    fields = header(); rows = []
    m = {'split_algorithm': 'constrained_scaffold_v3', 'seed': 42, 'records': []}
    for i, split in enumerate(dc.SPLITS):
        smiles = 'C' * (i + 1)
        row = dict.fromkeys(fields, '')
        row.update(cid=str(i + 1), smiles=smiles)
        row[PRIMARY[0]] = '0'; row[PRIMARY[1]] = '1.5'
        row['mouse_intraperitoneal_LD10'] = '2.5'
        rows.append(row)
        m['records'].append(dict(row_index=i, sample_id=f'row_{i}', raw_smiles=smiles,
            canonical_smiles=smiles, split_group=smiles, split=split))
    csv_path = tmp_path / 'source.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    m['source_csv_sha256'] = dc.digest(csv_path)
    split_path = tmp_path / 'split.json'; split_path.write_text(json.dumps(m))
    tox = {'records': [{'canonical_smiles': r['smiles']} for r in rows]}
    tox_path = tmp_path / 'tox.json'; tox_path.write_text(json.dumps(tox))
    monkeypatch.setattr(dc, 'CSV_SHA', dc.digest(csv_path))
    monkeypatch.setattr(dc, 'SPLIT_SHA', dc.digest(split_path))
    monkeypatch.setattr(dc, 'TOX_SEMANTIC_SHA', dc.semantic_digest(tox))
    monkeypatch.setattr(dc, 'ROW_COUNT', 4)
    return csv_path, split_path, tox_path

def test_end_to_end_route_mask_source_and_multiple_observations(audit_inputs):
    from dataset115_contract import audit, digest, SPLITS
    c, s, t = audit_inputs
    result = audit(c, s, t, expected_tox_sha=digest(t))
    assert result['observed_label_cells'] == 12  # Three observations on each molecule.
    assert result['molecule_counts']['A'] == dict.fromkeys(SPLITS, 1)
    assert result['molecule_counts']['B'] == dict.fromkeys(SPLITS, 0)
    assert result['primary5_counts']['A'][PRIMARY[0]] == dict.fromkeys(SPLITS, 1)
    assert result['primary5_counts']['A'][PRIMARY[2]] == dict.fromkeys(SPLITS, 0)
    assert result['source_train_observation_counts']['mouse_intraperitoneal_LD10'] == 1
    assert not set(result['excluded_human11']) & result['source_train_observation_counts'].keys()

def test_tox_semantic_identity_cannot_be_bypassed_with_new_byte_sha(audit_inputs):
    from dataset115_contract import audit, digest
    c, s, t = audit_inputs
    t.write_text('{"records":[{"canonical_smiles":"WRONG"}]}')
    with pytest.raises(ContractError, match='semantic'):
        audit(c, s, t, expected_tox_sha=digest(t))

def test_real_cli_path_with_small_fixture(audit_inputs, tmp_path):
    from dataset115_contract import main, digest
    import json
    c, s, t = audit_inputs; output = tmp_path / 'audit.json'
    assert main(['--csv', str(c), '--split-manifest', str(s), '--tox-manifest', str(t),
                 '--expected-tox-file-sha256', digest(t), '--output', str(output)]) == 0
    assert json.loads(output.read_text())['scope'] == 'INPUT_CONTRACT_ONLY_NOT_TRAINING_READY'
