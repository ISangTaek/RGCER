"""Snapshot and package verified evidence without deleting server weights."""
from pathlib import Path
import hashlib,json,zipfile
from p1d_optimization import require


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def snapshot(root):
    root=Path(root).resolve();files={}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'symlink in delivery')
        if not path.is_file():continue
        name=path.relative_to(root).as_posix()
        if name=='final_verification.json':continue
        require(path.name.lower() not in {'.env','id_rsa','id_ed25519','credentials','cookies.json'}
                and not {'.ssh','.aws','.azure'} & set(path.relative_to(root).parts),'credential path in evidence')
        require(not name.endswith(('.zip','.tgz','.key','.pem')),'unexpected archive/credential in evidence')
        files[name]=dict(sha256=sha(path),size_bytes=path.stat().st_size)
    return files


def package(root):
    root=Path(root).resolve()
    verified=json.loads((root/'final_verification.json').read_bytes())
    require(verified['content_status']=='PASS' and verified['checked_runs']==36,'full content verification required')
    require(snapshot(root)==verified['files'],'evidence changed after verification')
    summary=json.loads((root/'batch_summary.json').read_bytes())
    from p1d_schedule import resolve_plan
    require(summary['plan']==resolve_plan(summary['selections']),'delivery resolved matrix')
    require(verified['budget']==summary['plan']['budget'],'delivery budget')
    keep=set()
    for j in summary['plan']['jobs']:
        if j['action']=='REUSE':continue
        run=root/'runs'/j['run_id']
        name='summary.json' if j['setting']=='ToxAcute' else 'training_summary.json'
        epoch=json.loads((run/name).read_bytes())['best_epoch']
        keep.add(f"runs/{j['run_id']}/epoch_{epoch:03d}.pt")
    selected={n:v for n,v in verified['files'].items() if not n.endswith('.pt') or n in keep}
    require(keep<=set(selected),'selected best checkpoint missing')
    selected['final_verification.json']=dict(sha256=sha(root/'final_verification.json'),size_bytes=(root/'final_verification.json').stat().st_size)
    archive=root.with_name(root.name+'.zip');sidecar=Path(str(archive)+'.sha256')
    require(not archive.exists() and not sidecar.exists(),'delivery already exists; no overwrite')
    checksums=''.join(f"{v['sha256']}  {n}\n" for n,v in sorted(selected.items()))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for name in sorted(selected):z.write(root/name,name)
        z.writestr('checksums.sha256',checksums)
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'ZIP CRC')
        require(set(z.namelist())==set(selected)|{'checksums.sha256'},'ZIP file matrix')
        for name,item in selected.items():
            require(hashlib.sha256(z.read(name)).hexdigest()==item['sha256'],'ZIP member SHA')
    with sidecar.open('x',encoding='utf8') as stream:stream.write(sha(archive)+'  '+archive.name+'\n')
    return dict(archive=str(archive),sha256=sha(archive),members=len(selected)+1,
                epoch_weights_retained_on_server=True,acceptance_status='PENDING_REVIEW')
