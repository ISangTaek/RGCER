import json

import pytest

from scripts.collect_p1d_reuse import collect, targets, verify, sha


def test_missing_is_a_fact_not_scientific_failure(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();out=tmp_path/'output'
    result=collect(repo,out)
    assert len(result['runs'])==14 and not any(r['directory_exists'] for r in result['runs'])
    assert result['acceptance_status']=='PENDING_CODEX_REVIEW'
    assert verify(out)==result
    with pytest.raises(ValueError):collect(repo,out)


def test_copies_metadata_without_reading_weights(tmp_path):
    repo=tmp_path/'repo';folder=repo/next(iter(targets().values()));folder.mkdir(parents=True)
    (folder/'metrics.json').write_text('{"history": []}')
    (folder/'x.pt').write_bytes(b'not deserializable')
    result=collect(repo,tmp_path/'out');row=result['runs'][0]
    assert row['weights'][0]['sha256'] is None
    record=next(r for r in row['files'] if r['name']=='metrics.json')
    assert record['status']=='COPIED' and record['sha256']==sha(folder/'metrics.json')


@pytest.mark.parametrize('payload',['{"password":"secret"}','{"history":[],"test":{}}'])
def test_sensitive_or_holdout_not_copied(tmp_path,payload):
    repo=tmp_path/'repo';folder=repo/next(iter(targets().values()));folder.mkdir(parents=True)
    (folder/'metrics.json').write_text(payload)
    result=collect(repo,tmp_path/'out')
    assert next(r for r in result['runs'][0]['files'] if r['name']=='metrics.json')['status'].endswith('NOT_COPIED')
    assert not (tmp_path/'out/files').exists()


def test_tamper_or_extra_file_rejected(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();out=tmp_path/'out';collect(repo,out)
    (out/'extra').write_text('bad')
    with pytest.raises(ValueError,match='unlisted'):verify(out)


def test_escape_rejected(tmp_path):
    from scripts.collect_p1d_reuse import safe
    with pytest.raises(ValueError):safe(tmp_path,'../elsewhere')
