from pathlib import Path
import pytest
import v11_references as r


def small(tmp_path,monkeypatch):
    root=tmp_path/'source'; root.mkdir(); (root/'keep.json').write_text('{}'); (root/'history.json').write_text('{}')
    hashes={n:r.sha(root/n) for n in ('keep.json','history.json')}
    (root/'checksums.sha256').write_text(''.join(h+'  '+n+'\n' for n,h in hashes.items()))
    locked=dict(manifest_sha256=r.sha(root/'checksums.sha256'),original_content_files=2,subset={'keep.json':hashes['keep.json']})
    monkeypatch.setattr(r,'lock',lambda:locked)
    return root


def test_086_snapshot_uses_complete_trusted_manifest(tmp_path,monkeypatch):
    source=small(tmp_path,monkeypatch); assert r.copy_snapshot(source,tmp_path/'copy')['complete_original']
    assert r.manifest(tmp_path/'copy')['files_checked']==1


@pytest.mark.parametrize('kind',('file','manifest','missing','unlisted'))
def test_frozen_reference_changes_fail_closed(tmp_path,monkeypatch,kind):
    source=small(tmp_path,monkeypatch)
    if kind=='file': (source/'keep.json').write_text('bad')
    elif kind=='manifest': (source/'checksums.sha256').write_text('0'*64+'  keep.json\n')
    elif kind=='missing': (source/'keep.json').unlink()
    else: (source/'extra.json').write_text('{}')
    with pytest.raises(ValueError): r.manifest(source,complete=True)


def test_lock_keeps_latest_scene_winners_and_all_reference_implementations():
    lock=r.lock()
    assert lock['source_commit']=='fd676f1e6585ca3e03f0f217921be1f79b17d1e8'
    assert lock['expected_best']=={'ToxAcute':'086_DCR_RETRIEVAL_FIRST','A':'EQ_BSA_TNP_DANP','B':'086_BSA_MSE'}
    assert lock['original_content_files']==3356 and len(lock['subset'])==314
    assert len([n for n in lock['subset'] if n.startswith('reference_085/')])==240
    assert 'v10_s2_screen.py' in lock['reuse_code_sha256_lf']


@pytest.mark.parametrize('p',('../outside','/root','C:/absolute','a\\b',''))
def test_zip_path_traversal_rejected(p):
    with pytest.raises(ValueError): r.relative_path(p)


def test_cpu_metric_reduction_tolerance_does_not_accept_population_or_real_metric_changes():
    from copy import deepcopy
    a=dict(macro_rmse=1.,endpoints={'x':dict(n=4,rmse=1.)}); b=deepcopy(a)
    b['macro_rmse']+=2e-16; b['endpoints']['x']['rmse']+=2e-16
    assert r.same_score(a,b)
    b['endpoints']['x']['n']=5
    assert not r.same_score(a,b)
    b=deepcopy(a); b['macro_rmse']+=1e-8
    assert not r.same_score(a,b)
