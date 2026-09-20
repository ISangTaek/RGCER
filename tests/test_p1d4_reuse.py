import json
import pytest
from p1d4_reuse import LOCK,bound_history,check_files
from p1d_optimization import OptimizationError


def test_eighteen_inherited_histories_have_exact_complete_epoch_matrix():
    value=json.loads(LOCK.read_bytes())
    assert len(value['runs'])==18
    for alias in value['runs']:
        r=bound_history(alias)
        assert len(r['history'])==40
        assert len({f['path'] for f in r['files']})==len(r['files'])
    for seed in range(42,47):assert bound_history(f'ToxAcute_B1_s{seed}_existing')['best_epoch']==0


def test_missing_or_escaping_reuse_never_falls_back_to_training(tmp_path):
    for path in ('absent.pt','../elsewhere.pt'):
        with pytest.raises(OptimizationError,match='STOP_NOT_RETRAIN'):
            check_files(tmp_path,dict(files=[dict(path=path,size_bytes=1,sha256='0'*64)]))


def test_wrong_reuse_bytes_rejected(tmp_path):
    (tmp_path/'file').write_bytes(b'x')
    with pytest.raises(OptimizationError,match='identity'):
        check_files(tmp_path,dict(files=[dict(path='file',size_bytes=1,sha256='0'*64)]))


def test_unapproved_alias_rejected():
    with pytest.raises(OptimizationError):bound_history('s3_e40_s43')
