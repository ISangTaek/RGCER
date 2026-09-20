"""Copy the nine enumerated historical S3 weights; never deserialize or train."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_p1d_reuse import safe, sha, write_json


def strict_json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    def bad(value):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)


def check_args(value):
    # Config flags named test/calibration are metadata, not holdout observations.
    # Never open paths declared by this JSON. Credential values are not exported.
    forbidden={'password','passwd','api_key','access_token','refresh_token','client_secret','private_key','authorization'}
    if isinstance(value,dict):
        for k,v in value.items():
            if k.lower() in forbidden:raise ValueError('credential field in args; report, do not copy')
            check_args(v)
    elif isinstance(value,list):
        for item in value:check_args(item)
    elif isinstance(value,str) and ('-----BEGIN ' in value and 'PRIVATE KEY' in value):
        raise ValueError('private key material')


def export(repo, output, request):
    repo=Path(repo).resolve();output=Path(output).resolve()
    if output.exists() or output==repo or repo.is_relative_to(output):raise ValueError('new output required')
    if request['schema']!='p1d2_reuse_assets_v1':raise ValueError('request schema')
    if [r['seed'] for r in request['runs']]!=[42,44,46]:raise ValueError('three-seed request required')
    planned=[]
    for run in request['runs']:
        seed=run['seed'];expected=f'artifacts/runs/d7/d7_stage_b/s3/d7_s3_e40/seed_{seed}'
        if run['relative_path']!=expected or run['run_id']!=f's3_e40_s{seed}':raise ValueError('run scope')
        names={f'graphormer_d7_s3_e40_seed{seed}_best.pt',f'graphormer_d7_s3_e40_seed{seed}_last.pt','initial_model.pt'}
        if len(run['weights'])!=3 or {w['name'] for w in run['weights']}!=names:raise ValueError('weight matrix')
        for name,field in [('run_metadata.json','metadata_sha256'),('metrics.json','metrics_sha256'),('args.json','args_sha256')]:
            path=safe(repo,expected+'/'+name)
            if not path.is_file() or path.stat().st_size>16*1024*1024 or sha(path)!=run[field]:raise ValueError('061 metadata changed or missing: '+run['run_id']+'/'+name)
            obj=strict_json(path.read_bytes())
            if name=='args.json':check_args(obj)
            planned.append((run['run_id'],name,path,path.stat().st_size,run[field]))
        for weight in run['weights']:
            path=safe(repo,expected+'/'+weight['name'])
            if not path.is_file() or type(weight['size']) is not int or not 0<weight['size']<32*1024*1024 or path.stat().st_size!=weight['size']:
                raise ValueError('061 weight size/path changed: '+run['run_id']+'/'+weight['name'])
            planned.append((run['run_id'],weight['name'],path,weight['size'],None))
    if len(planned)!=18:raise ValueError('expected nine weights and nine metadata files')
    output.mkdir(parents=True,exist_ok=False);records=[]
    for run_id,name,src,size,expected_sha in planned:
        dest=output/'files'/run_id/name;dest.parent.mkdir(parents=True,exist_ok=True)
        before=sha(src)
        if expected_sha and before!=expected_sha:raise ValueError('metadata changed before copy')
        h=hashlib.sha256()
        with src.open('rb') as f,dest.open('xb') as g:
            for block in iter(lambda:f.read(1024*1024),b''):g.write(block);h.update(block)
        if dest.stat().st_size!=size or h.hexdigest()!=before or sha(src)!=before:raise ValueError('source changed during copy')
        records.append(dict(run_id=run_id,name=name,member=dest.relative_to(output).as_posix(),size=size,sha256=before,
                            expected_sha_source='061_metadata' if expected_sha else 'NEW_READ_ONLY_FILE_BINDING_NOT_REUSE_PASS'))
    write_json(output/'request.json',request)
    write_json(output/'manifest.json',dict(schema='p1d2_asset_export_v1',records=records,
                trained=False,deserialized=False,inference=False,acceptance_status='PENDING_CODEX_REVIEW'))
    with (output/'checksums.sha256').open('x',encoding='utf8') as f:
        f.write(''.join(sha(p)+'  '+p.relative_to(output).as_posix()+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='checksums.sha256'))
    verify(output)
    return records


def verify(output):
    output=Path(output).resolve();manifest=strict_json((output/'manifest.json').read_bytes())
    request=strict_json((output/'request.json').read_bytes())
    expected={(r['run_id'],name) for r in request['runs'] for name in
              [w['name'] for w in r['weights']]+['args.json','run_metadata.json','metrics.json']}
    records=manifest['records']
    if len(records)!=18 or len(expected)!=18 or {(r['run_id'],r['name']) for r in records}!=expected:raise ValueError('export matrix')
    members={'manifest.json','request.json','checksums.sha256'}
    for r in records:
        if r['member']!=f"files/{r['run_id']}/{r['name']}":raise ValueError('member identity')
        p=safe(output,r['member']);members.add(r['member'])
        if p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:raise ValueError('member bytes')
    if {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}!=members:raise ValueError('extra/missing export member')
    listed=[]
    for line in (output/'checksums.sha256').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        if name in listed or sha(safe(output,name))!=digest:raise ValueError('checksum')
        listed.append(name)
    if set(listed)!=members-{'checksums.sha256'}:raise ValueError('checksum coverage')
    return manifest


def package(work, destination):
    """Explicit review allowlist; ZIP is outside source tree, no nested archives."""
    work=Path(work).resolve();destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(work):raise ValueError('new ZIP must be outside WORK')
    verify(work/'collection')
    paths=[work/'实验记录.md']
    for name in ('wsl','server','collection'):
        folder=work/name
        if not folder.is_dir():raise ValueError('missing review directory')
        for p in folder.rglob('*'):
            if not p.resolve().is_relative_to(folder.resolve()):raise ValueError('review path escape')
            if p.is_file():
                if any(part.startswith('pytest_tmp') for part in p.relative_to(folder).parts):continue
                if p.suffix.lower() in ('.zip','.tgz','.gz','.key','.pem'):raise ValueError('nested archive/credential file')
                paths.append(p)
    if not paths[0].is_file():raise ValueError('experiment record required')
    hashes={p.relative_to(work).as_posix():sha(p) for p in paths}
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(work).as_posix())
        z.writestr('review_checksums.sha256',''.join(h+'  '+n+'\n' for n,h in sorted(hashes.items())))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC')
        for name,digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('ZIP member changed')
    with Path(str(destination)+'.sha256').open('x',encoding='utf8') as f:f.write(sha(destination)+'  '+destination.name+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--expected-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--package-work',type=Path)
    a=p.parse_args(argv)
    try:
        if len(a.expected_commit)!=40 or any(c not in '0123456789abcdef' for c in a.expected_commit):raise ValueError('full SHA')
        def git(*args):return subprocess.check_output(['git','-C',str(a.repo),*args],text=True).strip()
        if git('rev-parse','HEAD')!=a.expected_commit or git('status','--porcelain'):raise ValueError('code/worktree')
        if a.package_work:
            if a.verify_only:raise ValueError('incompatible modes')
            package(a.package_work,a.output);print('PACKAGED');return 0
        request=strict_json((a.repo/'configs/p1d2_reuse_asset_request.json').read_bytes())
        if a.verify_only:
            verify(a.output)
            if strict_json((a.output/'request.json').read_bytes())!=request:raise ValueError('request differs from checkout')
        else:export(a.repo,a.output,request)
        print('P1D2: 9 weights + 9 metadata, NO_DESERIALIZATION, PENDING_CODEX_REVIEW');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
