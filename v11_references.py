"""Immutable 086 references and train-only caches for the V11-S1 screen."""
from pathlib import Path, PurePosixPath
import hashlib
import math
import re
import shutil
import stat
import zipfile

import v10_s1_screen as s1
import v10_s2_screen as s2
import v10_s2_references as old

read, write, sha, digest, require = s1.read, s1.write, s1.sha, s1.digest, s1.require
REPO = Path(__file__).resolve().parent
LOCK = REPO / 'configs/v11_s1_reference_lock.json'
SOURCE_COMMIT = 'fd676f1e6585ca3e03f0f217921be1f79b17d1e8'
MIXTURES = (('TABR', 'BSA_TNP'), ('TABR', 'DANP'), ('BSA_TNP', 'DANP'),
            ('TABR', 'BSA_TNP', 'DANP'))


def code_sha(path):
    """Git text checkout normalization only; experiment files stay byte-exact."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def lock():
    value = read(LOCK)
    require(value['schema'] == 'v11_s1_reference_v1' and value['source_commit'] == SOURCE_COMMIT,
            'V11 trusted reference schema/commit')
    for name, expected in value['reuse_code_sha256_lf'].items():
        require(code_sha(REPO/name) == expected, 'reused implementation changed: ' + name)
    return value


def relative_path(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and ':' not in name and p.as_posix() == name,
            'unsafe reference path')
    return p


def safe_file(root, name):
    p = relative_path(name)
    target = root.joinpath(*p.parts)
    require(target.resolve().is_relative_to(root.resolve()) and
            not any((root.joinpath(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts)+1)),
            'reference symlink/escape')
    require(target.is_file(), 'missing reference: ' + name)
    return target


def manifest(root, *, complete=False):
    locked = lock()
    path = safe_file(root, 'checksums.sha256')
    require(sha(path) == locked['manifest_sha256'], '086 trusted manifest SHA')
    entries = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        expected, name = line.split(maxsplit=1)
        require(re.fullmatch('[0-9a-f]{64}', expected) is not None and name not in entries,
                '086 checksum manifest entry')
        entries[name] = expected
    require(len(entries) == locked['original_content_files'] and
            all(entries.get(n) == h for n,h in locked['subset'].items()), '086 manifest scope')
    checked = entries if complete else locked['subset']
    for name, expected in checked.items():
        require(sha(safe_file(root, name)) == expected, '086 reference SHA: ' + name)
    # Reject unlisted members; a compact snapshot intentionally omits unused history.
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(checked) | {'checksums.sha256'}, '086 reference file population')
    return dict(source_commit=SOURCE_COMMIT, manifest_sha256=locked['manifest_sha256'],
                files_checked=len(checked), complete_original=complete)


def unpack_reference(archive, destination):
    """Import the exact accepted inner ZIP into a new directory, never a raw run."""
    locked = lock()
    require(archive.is_file() and sha(archive) == locked['source_inner_sha256'],
            '086 trusted inner archive SHA')
    require(not destination.exists() and not destination.is_symlink(), 'existing 086 extraction')
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [x.filename for x in members]
        require(len(names) == len(set(names)) == locked['original_content_files']+1
                and 'checksums.sha256' in names, '086 archive population')
        for item in members:
            relative_path(item.filename)
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16),
                    '086 archive non-file member')
        require(z.testzip() is None, '086 archive CRC')
        destination.mkdir(parents=True)
        for item in members:
            target = destination/item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open('xb') as out:
                shutil.copyfileobj(source, out)
    return dict(archive_sha256=sha(archive), **manifest(destination, complete=True))


def copy_snapshot(source, destination):
    audit = manifest(source, complete=True)
    require(not destination.exists(), 'existing 086 snapshot')
    destination.mkdir(parents=True)
    for name, expected in dict(lock()['subset'], **{'checksums.sha256':lock()['manifest_sha256']}).items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(source,name), target)
        require(sha(target) == expected, '086 copy changed')
    manifest(destination)
    return audit


def load_features(root, setting):
    return old.load_features(root/'reference_085',setting)


def graph_manifest_sha(root,setting):
    return sha(root/'reference_085/cache'/setting/'manifest.json')


def same_score(a,b):
    """Only tolerate CPU reduction tails; schemas, endpoint populations stay exact."""
    if set(a)!=set(b) or set(a['endpoints'])!=set(b['endpoints']): return False
    if not math.isclose(a['macro_rmse'],b['macro_rmse'],rel_tol=1e-12,abs_tol=1e-12): return False
    for t,x in a['endpoints'].items():
        y=b['endpoints'][t]
        if set(x)!=set(y) or x['n']!=y['n'] or not math.isclose(x['rmse'],y['rmse'],rel_tol=1e-12,abs_tol=1e-12): return False
    return all(a[k]==b[k] for k in a if k not in ('macro_rmse','endpoints'))


def references(root, *, verify_files=True, complete=False):
    if verify_files: manifest(root,complete=complete)
    launch=read(root/'launch.json'); verification=read(root/'verification.json')
    require(launch['task']==verification['task']==s2.TASK and launch['commit']==verification['commit']==SOURCE_COMMIT
            and launch['spec']==s2.SPEC and launch['jobs']==s2.jobs() and verification['content_status']=='PASS'
            and verification['new_jobs']==36,'accepted 086 production identity')
    result=old.references(root/'reference_085')
    for setting,x in result.items():
        x['selected086']={}
        for method in s2.models.NEW+s2.models.CONTROLS:
            options=[]
            for job in s2.jobs():
                if job['setting']!=setting or job['method']!=method: continue
                r=read(root/job['id']/'receipt.json'); rows=read(root/job['id']/'selected_validation.json')
                score=s1.s3.s2.metrics(rows,x['validation'])
                require(r['identity']['commit']==SOURCE_COMMIT and r['identity']['job']==job and r['identity']['spec']==s2.SPEC
                        and same_score(r['selected'],score) and r['test_evaluated'] is False,'086 selected reference identity')
                options.append(dict(name='086_'+method,job=job,best_epoch=r['best_epoch'],score=score,rows=rows,
                                    parameter_count=r['architecture']['parameters'],checkpoint_sha256=r['checkpoint_sha256']))
            x['selected086'][method]=min(options,key=lambda r:r['score']['macro_rmse'])
        x['best']=min([x['best'],*x['selected086'].values()],key=lambda r:r['score']['macro_rmse'])
        require(x['best']['name']==lock()['expected_best'][setting],'V11 strongest frozen reference')
    return result
