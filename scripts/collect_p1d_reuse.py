"""P1-D bounded read-only inventory. Never loads weights or starts training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METADATA = ('args.json', 'run_metadata.json', 'metrics.json', 'architecture_summary.txt')
LIMIT = 16 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe(repo, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('relative repository path required')
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo):
        raise ValueError('path escapes repository')
    return path


def targets():
    return {f'{arm}_e{epochs}_s{seed}':f'artifacts/runs/d7/d7_stage_{stage}/{arm}/d7_{arm}_e{epochs}/seed_{seed}'
            for arm in ('s2', 's3') for stage, epochs, seeds in [('a',20,(42,45)),('b',40,range(42,47))]
            for seed in seeds}


def write_json(path, obj):
    with path.open('x',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def collect(repo, output):
    repo = Path(repo).resolve(); output = Path(output).resolve()
    if not repo.is_dir() or output.exists() or output == repo or repo.is_relative_to(output):
        raise ValueError('existing repo and new non-ancestor output required')
    # Resolve all inputs first; do not walk home directories or follow escapes.
    resolved = [(key,rel,safe(repo,rel)) for key,rel in targets().items()]
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for key, rel, folder in resolved:
        row=dict(run_id=key,relative_path=rel,directory_exists=folder.is_dir(),files=[],weights=[])
        for name in METADATA:
            path=safe(repo,rel+'/'+name)
            record=dict(name=name,status='MISSING')
            if path.is_file():
                record.update(size=path.stat().st_size,sha256=sha(path))
                if record['size'] > LIMIT:
                    record['status']='SIZE_LIMIT_REFERENCE_ONLY'
                else:
                    raw=path.read_bytes()
                    lower=raw.lower()
                    sensitive=(b'password',b'api_key',b'access_token',b'private key',b'authorization',b'client_secret')
                    if any(term in lower for term in sensitive):
                        record['status']='SENSITIVE_FIELD_NOT_COPIED'
                    elif b'"test"' in lower or b'"calibration"' in lower:
                        record['status']='HOLDOUT_FIELD_NOT_COPIED'
                    else:
                        dest=output/'files'/key/name;dest.parent.mkdir(parents=True,exist_ok=True)
                        with dest.open('xb') as f:f.write(raw)
                        if sha(dest)!=record['sha256']:raise ValueError('input changed during copy')
                        record.update(status='COPIED',member=dest.relative_to(output).as_posix())
            row['files'].append(record)
        if folder.is_dir():
            for item in sorted(folder.glob('*.pt')):
                p=safe(repo,item.relative_to(repo))
                if p.is_file():row['weights'].append(dict(name=item.name,size=p.stat().st_size,sha256=None,reason='NOT_READ_OR_HASHED'))
        rows.append(row)
    result=dict(schema='p1d_reuse_inventory_v1',scope='14_EXPLICIT_D7_DIRECTORIES_METADATA_ONLY',
                runs=rows,training=False,weight_loading=False,
                acceptance_status='PENDING_CODEX_REVIEW',
                absence_scope='only listed paths; not a server-wide absence claim')
    write_json(output/'inventory.json',result)
    members=sorted(p for p in output.rglob('*') if p.is_file())
    with (output/'checksums.sha256').open('x',encoding='utf8') as f:
        f.write(''.join(sha(p)+'  '+p.relative_to(output).as_posix()+'\n' for p in members))
    verify(output)
    return result


def verify(output):
    output=Path(output).resolve()
    lines=(output/'checksums.sha256').read_text(encoding='utf8').splitlines()
    names=[]
    for line in lines:
        digest,name=line.split('  ',1)
        path=safe(output,name)
        if name in names or sha(path)!=digest:raise ValueError('member checksum/duplicate')
        names.append(name)
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}-{'checksums.sha256'}
    if set(names)!=actual:raise ValueError('unlisted/missing member')
    result=json.loads((output/'inventory.json').read_text(encoding='utf8'))
    rows=result['runs']
    if len(rows)!=14 or {r['run_id']:r['relative_path'] for r in rows}!=targets():raise ValueError('inventory matrix')
    for row in rows:
        if [r['name'] for r in row['files']]!=list(METADATA):raise ValueError('metadata matrix')
        for record in row['files']:
            if record['status']=='COPIED':
                p=safe(output,record['member'])
                if p.stat().st_size!=record['size'] or sha(p)!=record['sha256']:raise ValueError('copied evidence identity')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    a=parser.parse_args(argv)
    def git(*args):return subprocess.check_output(['git','-C',str(a.repo),*args],text=True).strip()
    try:
        if len(a.expected_commit)!=40 or any(c not in '0123456789abcdef' for c in a.expected_commit):raise ValueError('full commit required')
        if git('rev-parse','HEAD')!=a.expected_commit or git('status','--porcelain'):raise ValueError('commit/worktree differs')
        result=verify(a.output) if a.verify_only else collect(a.repo,a.output)
        print(json.dumps(dict(directories=14,present=sum(r['directory_exists'] for r in result['runs']),
                              scope=result['scope'],acceptance_status=result['acceptance_status'])))
        return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
