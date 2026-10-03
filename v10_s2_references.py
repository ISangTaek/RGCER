"""Immutable 085 references and train-only caches for the V10-S2 screen."""
from pathlib import Path, PurePosixPath
import hashlib
import re
import shutil
import stat
import zipfile

import v10_s1_screen as s1

read, write, sha, digest, require = s1.read, s1.write, s1.sha, s1.digest, s1.require
REPO = Path(__file__).resolve().parent
LOCK = REPO / 'configs/v10_s2_reference_lock.json'
SOURCE_COMMIT = '9111793be3adf87ca335be8f44aa86065e38427e'
MIXTURES = (('TABR', 'BSA_TNP'), ('TABR', 'DANP'), ('BSA_TNP', 'DANP'),
            ('TABR', 'BSA_TNP', 'DANP'))


def code_sha(path):
    """Git text checkout normalization only; experiment files stay byte-exact."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def lock():
    value = read(LOCK)
    require(value['schema'] == 'v10_s2_reference_v1' and value['source_commit'] == SOURCE_COMMIT,
            'S2 trusted reference schema/commit')
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
    require(sha(path) == locked['manifest_sha256'], '085 trusted manifest SHA')
    entries = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        expected, name = line.split(maxsplit=1)
        require(re.fullmatch('[0-9a-f]{64}', expected) is not None and name not in entries,
                '085 checksum manifest entry')
        entries[name] = expected
    require(len(entries) == locked['original_content_files'] and
            all(entries.get(n) == h for n,h in locked['subset'].items()), '085 manifest scope')
    checked = entries if complete else locked['subset']
    for name, expected in checked.items():
        require(sha(safe_file(root, name)) == expected, '085 reference SHA: ' + name)
    # Reject unlisted members; a compact snapshot intentionally omits unused history.
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(checked) | {'checksums.sha256'}, '085 reference file population')
    return dict(source_commit=SOURCE_COMMIT, manifest_sha256=locked['manifest_sha256'],
                files_checked=len(checked), complete_original=complete)


def unpack_reference(archive, destination):
    """Import the exact accepted inner ZIP into a new directory, never a raw run."""
    locked = lock()
    require(archive.is_file() and sha(archive) == locked['source_inner_sha256'],
            '085 trusted inner archive SHA')
    require(not destination.exists() and not destination.is_symlink(), 'existing 085 extraction')
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [x.filename for x in members]
        require(len(names) == len(set(names)) == locked['original_content_files']+1
                and 'checksums.sha256' in names, '085 archive population')
        for item in members:
            relative_path(item.filename)
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16),
                    '085 archive non-file member')
        require(z.testzip() is None, '085 archive CRC')
        destination.mkdir(parents=True)
        for item in members:
            target = destination/item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open('xb') as out:
                shutil.copyfileobj(source, out)
    return dict(archive_sha256=sha(archive), **manifest(destination, complete=True))


def copy_snapshot(source, destination):
    audit = manifest(source, complete=True)
    require(not destination.exists(), 'existing 085 snapshot')
    destination.mkdir(parents=True)
    for name, expected in dict(lock()['subset'], **{'checksums.sha256':lock()['manifest_sha256']}).items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(source,name), target)
        require(sha(target) == expected, '085 copy changed')
    manifest(destination)
    return audit


def load_features(root, setting):
    locked = lock()
    for suffix in ('cache.pt', 'manifest.json'):
        name = f'cache/{setting}/{suffix}'
        require(sha(safe_file(root,name)) == locked['subset'][name], 'trusted 085 cache file')
    value = s1.cache.load_cache(root/'cache'/setting, SOURCE_COMMIT)
    s1.check_binding(value, read(s1.LOCK)['settings'][setting])
    # Independent construction checks all scalar and row identities.
    s1.cache.Data(value)
    packed = s1.cache.pack({k:v for k,v in value.items() if k not in (
        'mean','scale','commit','schema','statistics_population_sha256')}, SOURCE_COMMIT)
    import torch
    require(torch.equal(packed['mean'],value['mean']) and torch.equal(packed['scale'],value['scale'])
            and packed['statistics_population_sha256'] == value['statistics_population_sha256'],
            'frozen train feature statistics')
    return value


def equal_mixture(parts, expected):
    require(len(parts) >= 2, 'mixture members')
    for rows in parts:
        s1.s3.s2.metrics(rows, expected)
    lookup = [{(r['task'],r['sample_id']):r['prediction'] for r in rows} for rows in parts]
    return [dict(r,prediction=sum(l[r['task'],r['sample_id']] for l in lookup)/len(parts)) for r in expected]


def references(root, *, verify_files=True, complete=False):
    if verify_files:
        manifest(root, complete=complete)
    locked = lock()
    launch = read(root/'launch.json'); verification = read(root/'verification.json')
    require(launch['task'] == verification['task'] == s1.TASK and
            launch['commit'] == verification['commit'] == SOURCE_COMMIT and
            launch['spec'] == s1.SPEC and launch['jobs'] == s1.jobs() and
            verification['content_status'] == 'PASS' and verification['jobs'] == 84,
            'accepted 085 production identity')
    old = s1.references(root/'references/084',root/'references/066')
    result = {}
    for setting in s1.SETTINGS:
        expected = old[setting]['validation']; selected = {}
        for method in s1.models.CANDIDATES + s1.models.CONTROLS:
            options = []
            for job in s1.jobs():
                if job['setting'] != setting or job['method'] != method:
                    continue
                r = read(root/job['id']/'receipt.json')
                rows = read(root/job['id']/'selected_validation.json')
                score = s1.s3.s2.metrics(rows,expected)
                require(r['identity']['commit'] == SOURCE_COMMIT and r['identity']['job'] == job and
                        r['identity']['spec'] == s1.SPEC and r['selected'] == score and
                        r['test_evaluated'] is False, '085 selected result identity')
                options.append(dict(name='085_'+method,job=job,best_epoch=r['best_epoch'],score=score,rows=rows,
                                    parameter_count=r['parameter_count'],checkpoint_sha256=r['checkpoint_sha256']))
            selected[method] = min(options,key=lambda r:r['score']['macro_rmse'])
        mixes = {}
        for members in MIXTURES:
            name = 'EQ_' + '_'.join(members)
            rows = equal_mixture([selected[m]['rows'] for m in members],expected)
            mixes[name] = dict(name=name,components=[selected[m]['job'] for m in members],
                               weights=[1/len(members)]*len(members),rows=rows,
                               score=s1.s3.s2.metrics(rows,expected),prediction_sha256=digest(rows),
                               interpretation='post_hoc_085_diagnostic_frozen_for_S2')
        strongest = min([old[setting]['best'],*selected.values(),*mixes.values()],
                        key=lambda r:r['score']['macro_rmse'])
        require(strongest['name'] == locked['expected_best'][setting], 'S2 frozen strongest comparator')
        result[setting] = dict(validation=expected,training=old[setting]['training'],
                               historical=old[setting],selected=selected,mixtures=mixes,best=strongest)
    return result
