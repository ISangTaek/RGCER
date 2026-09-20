import copy
import json
from pathlib import Path
import zipfile

import pytest

from scripts.export_p1d_reuse_assets import export, verify, package, sha


def fixture(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    request={'schema':'p1d2_reuse_assets_v1','runs':[]}
    for seed in (42,44,46):
        rel=f'artifacts/runs/d7/d7_stage_b/s3/d7_s3_e40/seed_{seed}'
        folder=repo/rel;folder.mkdir(parents=True)
        row=dict(seed=seed,run_id=f's3_e40_s{seed}',relative_path=rel,weights=[])
        for name,field in [('args.json','args_sha256'),('metrics.json','metrics_sha256'),('run_metadata.json','metadata_sha256')]:
            p=folder/name;p.write_text('{"test":false,"epochs":40}')
            row[field]=sha(p)
        for name in (f'graphormer_d7_s3_e40_seed{seed}_best.pt',f'graphormer_d7_s3_e40_seed{seed}_last.pt','initial_model.pt'):
            p=folder/name;p.write_bytes(b'NOT_A_TORCH_FILE')
            row['weights'].append(dict(name=name,size=p.stat().st_size))
        request['runs'].append(row)
    return repo,request


def test_exact_copy_no_deserialization(tmp_path):
    repo,req=fixture(tmp_path);out=tmp_path/'output'
    assert len(export(repo,out,req))==18
    assert verify(out)['deserialized'] is False
    with pytest.raises(ValueError,match='new output'):export(repo,out,req)


@pytest.mark.parametrize('case',['metadata','weight_size','missing','scope','duplicate','credential'])
def test_fail_closed_before_export(tmp_path,case):
    repo,req=fixture(tmp_path);r=req['runs'][0];folder=repo/r['relative_path']
    if case=='metadata':(folder/'metrics.json').write_text('{}')
    if case=='weight_size':(folder/r['weights'][0]['name']).write_bytes(b'changed')
    if case=='missing':(folder/r['weights'][0]['name']).unlink()
    if case=='scope':r['relative_path']='../outside'
    if case=='duplicate':r['weights'][0]=copy.deepcopy(r['weights'][1])
    if case=='credential':
        (folder/'args.json').write_text('{"password":"not-exportable"}')
        r['args_sha256']=sha(folder/'args.json')
    with pytest.raises(ValueError):export(repo,tmp_path/'output',req)
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('case',['extra','bytes','manifest'])
def test_verifier_rejects_corruption(tmp_path,case):
    repo,req=fixture(tmp_path);out=tmp_path/'out';export(repo,out,req)
    if case=='extra':(out/'extra.zip').write_bytes(b'bad')
    if case=='bytes':next((out/'files').rglob('*.pt')).write_bytes(b'bad')
    if case=='manifest':
        p=out/'manifest.json';m=json.loads(p.read_text());m['records'].pop();p.write_text(json.dumps(m))
    with pytest.raises(ValueError):verify(out)


def test_pack_outside_and_no_self_zip(tmp_path):
    repo,req=fixture(tmp_path);work=tmp_path/'work';work.mkdir();export(repo,work/'collection',req)
    for name in ('wsl','server'):
        (work/name).mkdir();(work/name/'log.txt').write_text('exit=0')
    (work/'实验记录.md').write_text('read only',encoding='utf8')
    with pytest.raises(ValueError,match='outside'):package(work,work/'bad.zip')
    dest=tmp_path/'review.zip';package(work,dest)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist())==9
        assert not any(n.endswith('.zip') for n in z.namelist())
    assert Path(str(dest)+'.sha256').read_text().startswith(sha(dest))
    (work/'server/old.zip').write_bytes(b'old')
    with pytest.raises(ValueError,match='nested'):package(work,tmp_path/'blocked.zip')
