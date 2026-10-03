from copy import deepcopy
import json
import stat
import zipfile

import pytest
import torch

import v10_s2_references as r
from tests.test_v10_s1_screen import value
from tests.test_v10_context_models import single_thread


def small_reference(tmp_path,monkeypatch):
    source=tmp_path/'source'; source.mkdir()
    (source/'keep.json').write_text('{}\n')
    (source/'history.json').write_text('{"epoch":1}\n')
    hashes={n:r.sha(source/n) for n in ('keep.json','history.json')}
    (source/'checksums.sha256').write_text(''.join(h+'  '+n+'\n' for n,h in hashes.items()))
    locked=dict(manifest_sha256=r.sha(source/'checksums.sha256'),original_content_files=2,subset={'keep.json':hashes['keep.json']})
    monkeypatch.setattr(r,'lock',lambda:locked)
    return source,locked


def test_complete_archive_to_compact_frozen_snapshot(tmp_path,monkeypatch):
    source,_=small_reference(tmp_path,monkeypatch)
    audit=r.copy_snapshot(source,tmp_path/'copy')
    assert audit['files_checked']==2 and audit['complete_original']
    assert r.manifest(tmp_path/'copy')['files_checked']==1
    assert not (tmp_path/'copy/history.json').exists()
    with pytest.raises(ValueError): r.copy_snapshot(source,tmp_path/'copy')


def reference_zip(source, locked, archive, *, bad=None):
    with zipfile.ZipFile(archive, 'x') as z:
        for p in source.iterdir():
            name = p.name
            if name == 'history.json' and bad == 'path': name = '../history.json'
            if name == 'history.json' and bad == 'duplicate': name = 'keep.json'
            if name == 'history.json' and bad == 'symlink':
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3; entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(entry, 'keep.json')
            else: z.write(p, name)
    locked['source_inner_sha256'] = r.sha(archive)


def test_import_trusted_zip_then_verify_every_file(tmp_path,monkeypatch):
    source,locked=small_reference(tmp_path,monkeypatch)
    archive=tmp_path/'original.zip'; reference_zip(source,locked,archive)
    audit=r.unpack_reference(archive,tmp_path/'unpacked')
    assert audit['complete_original'] and audit['files_checked']==2
    assert audit['archive_sha256']==r.sha(archive)


def test_wrong_archive_fails_before_creating_directory(tmp_path,monkeypatch):
    source,locked=small_reference(tmp_path,monkeypatch)
    archive=tmp_path/'original.zip'; reference_zip(source,locked,archive)
    archive.write_bytes(archive.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='trusted inner archive SHA'):
        r.unpack_reference(archive,tmp_path/'unpacked')
    assert not (tmp_path/'unpacked').exists()


def test_import_never_overwrites_existing_destination(tmp_path,monkeypatch):
    source,locked=small_reference(tmp_path,monkeypatch)
    archive=tmp_path/'original.zip'; reference_zip(source,locked,archive)
    with pytest.raises(ValueError,match='existing 085 extraction'): r.unpack_reference(archive,source)
    assert (source/'keep.json').read_text()=='{}\n'


@pytest.mark.parametrize('bad',('path','duplicate','symlink'))
def test_import_checks_all_members_before_any_extraction(bad,tmp_path,monkeypatch):
    source,locked=small_reference(tmp_path,monkeypatch)
    archive=tmp_path/'original.zip'
    if bad=='duplicate':
        with pytest.warns(UserWarning,match='Duplicate name'): reference_zip(source,locked,archive,bad=bad)
    else: reference_zip(source,locked,archive,bad=bad)
    with pytest.raises(ValueError): r.unpack_reference(archive,tmp_path/'unpacked')
    assert not (tmp_path/'unpacked').exists()


@pytest.mark.parametrize('kind',('file','manifest','unlisted','missing'))
def test_reference_population_and_trusted_hash_fail_closed(kind,tmp_path,monkeypatch):
    source,_=small_reference(tmp_path,monkeypatch)
    if kind=='file': (source/'keep.json').write_text('{"modified":true}')
    elif kind=='manifest':
        (source/'keep.json').write_text('{"modified":true}')
        lines=[r.sha(source/n)+'  '+n+'\n' for n in ('keep.json','history.json')]
        (source/'checksums.sha256').write_text(''.join(lines))
    elif kind=='unlisted': (source/'extra.json').write_text('{}')
    else: (source/'keep.json').unlink()
    with pytest.raises(ValueError): r.manifest(source,complete=True)


@pytest.mark.parametrize('name',('../outside.json','/absolute.json','C:/bad.json','a\\b.json',''))
def test_unsafe_paths_rejected(name,tmp_path):
    with pytest.raises(ValueError): r.safe_file(tmp_path,name)


def test_equal_mixture_pairs_rows_by_identity_and_rejects_wrong_population(value):
    expected=[x for x in value['rows'] if x['split']=='validation']
    a=[dict(x,prediction=x['label']+1) for x in expected]
    b=[dict(x,prediction=x['label']-1) for x in expected][::-1]
    combined=r.equal_mixture([a,b],expected)
    assert all(abs(x['prediction']-x['label'])<1e-12 for x in combined)
    b[0]['canonical']='wrong'
    with pytest.raises(ValueError): r.equal_mixture([a,b],expected)


def test_reuse_lock_pins_accepted_source_and_exact_implementation():
    locked=r.lock()
    assert locked['source_commit']==r.SOURCE_COMMIT
    assert locked['source_inner_sha256']=='a03ffb9726f4e601bc9def98959d9359198c40342f19fbe1170871cc3562e97f'
    assert len(locked['subset'])==239 and locked['original_content_files']==6385
    assert locked['expected_best']=={'ToxAcute':'085_TABR','A':'EQ_BSA_TNP_DANP','B':'085_BSA_TNP'}


def test_code_hash_is_checkout_newline_independent_but_assets_remain_byte_exact(tmp_path):
    a,b=tmp_path/'unix.py',tmp_path/'windows.py'
    a.write_bytes(b'a = 1\nb = 2\n'); b.write_bytes(b'a = 1\r\nb = 2\r\n')
    assert r.code_sha(a)==r.code_sha(b)
    assert r.sha(a)!=r.sha(b)
    b.write_bytes(b'a = 2\r\nb = 2\r\n')
    assert r.code_sha(a)!=r.code_sha(b)


def test_self_consistent_cache_forgery_is_rejected_by_published_hash(value,tmp_path,monkeypatch):
    folder=tmp_path/'cache/ToxAcute'; value=deepcopy(value)
    value['commit']=r.SOURCE_COMMIT
    raw={k:v for k,v in value.items() if k not in ('mean','scale','commit','schema','statistics_population_sha256')}
    r.s1.cache.save_cache(raw,folder,r.SOURCE_COMMIT)
    locked=dict(subset={f'cache/ToxAcute/{n}':r.sha(folder/n) for n in ('cache.pt','manifest.json')})
    monkeypatch.setattr(r,'lock',lambda:locked)
    changed=torch.load(folder/'cache.pt',weights_only=True); changed['h'][0,0]+=1
    torch.save(changed,folder/'cache.pt')
    manifest=r.read(folder/'manifest.json'); manifest['sha256']=r.sha(folder/'cache.pt')
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='trusted 085 cache file'): r.load_features(tmp_path,'ToxAcute')
