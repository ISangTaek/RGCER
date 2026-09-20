import json
import pytest
from p1d4_delivery import snapshot,package
from p1d_optimization import OptimizationError


def test_snapshot_detects_changed_bytes_and_skips_own_receipt(tmp_path):
    (tmp_path/'a').write_bytes(b'one')
    before=snapshot(tmp_path)
    (tmp_path/'final_verification.json').write_text('{}')
    assert snapshot(tmp_path)==before
    (tmp_path/'a').write_bytes(b'two')
    assert snapshot(tmp_path)!=before


def test_package_refuses_post_verification_edit(tmp_path):
    (tmp_path/'a').write_bytes(b'one')
    value=dict(content_status='PASS',checked_runs=36,files=snapshot(tmp_path))
    (tmp_path/'final_verification.json').write_text(json.dumps(value))
    (tmp_path/'a').write_bytes(b'two')
    with pytest.raises(OptimizationError,match='changed after'):package(tmp_path)


@pytest.mark.parametrize('name',['old.zip','private.pem','key.key','id_rsa','.env'])
def test_no_recursive_archives_or_credentials(tmp_path,name):
    (tmp_path/name).write_bytes(b'x')
    with pytest.raises(OptimizationError):snapshot(tmp_path)


def test_package_keeps_best_and_all_metadata_but_preserves_other_weights(tmp_path):
    """Packaging-only synthetic fixtures, not scientific acceptance evidence."""
    import zipfile
    from tests.test_p1d4_batch import fake_results
    from p1d4_batch import execute_matrix
    root=tmp_path/'batch';root.mkdir()
    summary=execute_matrix(lambda jobs:fake_results(jobs,low=False))
    (root/'batch_summary.json').write_text(json.dumps(summary))
    best=[];omitted=[]
    for j in summary['plan']['jobs']:
        if j['action']=='REUSE':continue
        run=root/'runs'/j['run_id'];run.mkdir(parents=True)
        name='summary.json' if j['setting']=='ToxAcute' else 'training_summary.json'
        (run/name).write_text(json.dumps(dict(best_epoch=0)))
        (run/'epoch_000.pt').write_bytes(b'best');(run/'epoch_001.pt').write_bytes(b'other')
        best.append((run/'epoch_000.pt').relative_to(root).as_posix())
        omitted.append((run/'epoch_001.pt').relative_to(root).as_posix())
    verified=dict(content_status='PASS',checked_runs=36,budget=summary['plan']['budget'],files=snapshot(root))
    (root/'final_verification.json').write_text(json.dumps(verified))
    result=package(root)
    with zipfile.ZipFile(result['archive']) as z:
        assert all(n in z.namelist() for n in best)
        assert all(n not in z.namelist() for n in omitted)
        assert 'checksums.sha256' in z.namelist()
    assert all((root/n).is_file() for n in omitted)
    with pytest.raises(OptimizationError,match='already exists'):package(root)
