import json
import numpy as np
import pytest

from s4e_mechanism_design import (DesignError, aligned_records, choose, read_json,
                                  tensor_signature, unique, write_json, checked_collection)


def fixture():
    return {'records': [
        {'sample_id': 'row_9', 'row_index': 9, 'split': 'train'},
        {'sample_id': 'row_11', 'row_index': 11, 'split': 'test'}]}, {
        'sample_ids': np.array(['row_11', 'row_9']), 'row_indices': np.array([0, 1]),
        'split_codes': np.array([3, 0]), 'num_nodes': np.array([8, 9])}


def test_raw_row_is_not_compact_global_index():
    m, i = fixture()
    rows = aligned_records(m, i)
    assert rows[0]['row_index'] == 11 and rows[0]['global_index'] == 0
    assert rows[1]['sample_id'] == 'row_9'


@pytest.mark.parametrize('field,value', [('sample_ids', ['row_9', 'row_9']),
    ('sample_ids', ['row_9', 'row_12']), ('row_indices', [1, 0]), ('split_codes', [0, 3])])
def test_bad_index_rejected(field, value):
    m, i = fixture(); i[field] = np.array(value)
    with pytest.raises(DesignError): aligned_records(m, i)


def test_duplicate_manifest_id_rejected():
    m, i = fixture(); m['records'].append(m['records'][0])
    with pytest.raises(DesignError): aligned_records(m, i)


def test_selection_input_order_invariant_and_nested():
    a = ['C', 'CC', 'CCC', 'O']
    assert choose(a, 3) == choose(list(reversed(a)), 3)
    assert choose(a, 2) == choose(a, 3)[:2]
    with pytest.raises(DesignError): choose(a, 5)


@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_bad_json_rejected(tmp_path, text):
    p = tmp_path/'input.json'; p.write_text(text)
    with pytest.raises(DesignError): read_json(p)


def test_output_is_nonoverwriting(tmp_path):
    p = tmp_path/'result.json'; write_json(p, {'ok': None})
    with pytest.raises(FileExistsError): write_json(p, {})
    assert read_json(p) == {'ok': None}


def test_unreviewed_collection_rejected(tmp_path):
    (tmp_path/'checksums.sha256').write_text('')
    with pytest.raises(DesignError): checked_collection(tmp_path)


def test_tensor_digest_and_duplicate_checks():
    t = {'key': 'a', 'shape': [2], 'dtype': 'torch.float32', 'numel': 2, 'tensor_sha256': 'a'*64}
    inv = {'tensor_count': 1, 'tensors': [t]}
    a = tensor_signature(inv)
    t['tensor_sha256'] = 'b'*64
    assert a != tensor_signature(inv)
    inv['tensors'].append(t)
    with pytest.raises(DesignError): tensor_signature(inv)
