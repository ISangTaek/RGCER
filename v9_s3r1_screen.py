"""Complete only the three untrained V9-S3 Route B jobs; inherit immutable 078."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch
import v9_s3_screen as s3

REPO=Path(__file__).resolve().parent
TASK='V9_S3R1_ROUTE_B_COMPLETION_20261002'
REGISTRY='.tmp/v9s3r1_attempt_20261002'
OLD_COMMIT='2a7f28973feabc236048f5a5ca4784cb36454d66'
LOCK=REPO/'configs/v9_s3r1_inherited_lock.json'
TESTS=('tests/test_v9_s3r1_screen.py',*s3.TESTS)
TEST_COUNT=186
read,write,sha,digest,require=s3.read,s3.write,s3.sha,s3.digest,s3.require


def jobs():
    return [j for j in s3.jobs() if j['setting'] == 'B']


def inherited_jobs():
    return [j for j in s3.jobs() if j['setting'] != 'B']


def inherited(root):
    """The committed hashes bind the prior review, not a runtime self-report."""
    root=Path(root);lock=read(LOCK)
    require(lock['schema'] == 'v9_s3r1_inherited_v1' and lock['commit'] == OLD_COMMIT
            and lock['task'] == s3.TASK and lock['spec'] == s3.SPEC
            and lock['completed_jobs'] == inherited_jobs(), '078 lock contract')
    actual=set()
    for p in root.rglob('*'):
        require(not p.is_symlink(), '078 evidence symlink')
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    require(actual-{'checksums.sha256'} == set(lock['files']), '078 exact member set')
    for name,expected in lock['files'].items():
        p=root/name
        require(p.resolve().is_relative_to(root.resolve()) and sha(p) == expected, '078 inherited file '+name)
    launch=read(root/'launch.json')
    require(launch['task'] == s3.TASK and launch['commit'] == OLD_COMMIT and launch['spec'] == s3.SPEC
            and launch['jobs'] == s3.jobs(), '078 launch identity')
    for role in ('wsl','server'):s3.gate_check(root/(role+'_evidence'),OLD_COMMIT,role)
    require(not (root/'verification.json').exists() and
            {p.name for p in (root/'B_B1_1E4_s42').iterdir()} == {'started.json'}
            and all(not (root/j['id']).exists() for j in jobs()[1:]), '078 B must be untrained')
    reference=s3.references(root/'reference_077');results=[]
    for j in inherited_jobs():
        folder=root/j['id'];r=read(folder/'receipt.json');identity=read(folder/'identity.json')
        require(r['job'] == j and r['task'] == s3.TASK and r['commit'] == OLD_COMMIT
                and r['identity'] == identity and identity['spec'] == s3.SPEC, '078 receipt identity')
        validation=read(folder/'validation_observations.json')
        require(digest(validation) == reference[j['id']]['identity']['validation_sha256'], '078 paired validation')
        for h in r['history']:
            epoch=h['epoch']
            require(h == read(folder/f'epoch_{epoch:03d}.json') and h['validation'] == s3.s2.metrics(
                read(folder/f'validation_epoch_{epoch:03d}.json'),validation), '078 epoch metrics')
        best,_=s3.s2.select(r['history'],s3.SPEC)
        selected=read(folder/'selected_validation.json')
        require(r['best_epoch'] == best and selected == read(folder/f'validation_epoch_{best:03d}.json')
                and r['selected'] == s3.s2.metrics(selected,validation), '078 selected result')
        require(r['updates'] == (240 if j['setting'] == 'ToxAcute' else 600), '078 update count')
        results.append(r)
    require(sum(r['updates'] for r in results) == 2520, '078 inherited total')
    return results,reference


def preflight_context(factory, trainer, source, commit, reference, accepted):
    """CPU metadata and initialization only; never forward or update a model."""
    training=s3.s2.observations(factory,trainer,'B','train')
    validation=s3.s2.observations(factory,trainer,'B','validation')
    populations=s3.population_check(factory,source,'B',training,validation)
    require(source.identity == accepted[0]['identity']['source'], 'Animal56 source differs from accepted 078')
    _,counts=s3.s2.c0.tasks_and_counts('B')
    require({t:len(v) for t,v in s3.s2.datasets_for(factory,trainer,'B')['train'].items()} == counts, 'B target counts')
    identities={}
    for j in jobs():
        ident=s3.identity_for(trainer,source,j,training,validation,commit,s3.SPEC,task_id=TASK)
        old=reference[j['id']]['identity']
        require(ident['initial_encoder'] == old['initial_encoder'] == read(s3.s2.c2b.LOCK)['initial_encoders']['B']
                and ident['initial_target_heads'] == old['initial_heads'] and all(ident[k] == old[k]
                for k in ('contract_sha256','train_sha256','validation_sha256')), 'B 077 identity pairing')
        identities[j['id']]=ident
    return dict(task=TASK,commit=commit,jobs=jobs(),inherited_lock_sha256=sha(LOCK),
                population_check=populations,identities=identities,model_forward_calls=0,optimizer_updates=0)


def preflight(commit, split, source_lock, accepted, reference):
    f,t,source=s3.context('B',split,source_lock,'cpu')
    return preflight_context(f,t,source,commit,reference,accepted)


def gate_check(root, commit, role):
    require(read(root/'gate.json') == dict(task=TASK,commit=commit,role=role,tests=list(TESTS),
            junit_sha256=sha(root/'tests.xml')), 'S3R1 gate identity')
    require(read(root/'command.json')['exit_code'] == 0, 'S3R1 code gate failed')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases) == TEST_COUNT and all(not any(c.findall(t) for t in ('error','failure','skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases} == {'tests.'+Path(t).stem for t in TESTS}, 'S3R1 complete passing gate')


def code_gate(root, commit, role):
    require(role in ('wsl','server'), 'gate role');s3.s2.c0.check_code(REPO,commit)
    root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as stream:
        p=subprocess.run(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json',dict(argv=argv,exit_code=p.returncode));require(p.returncode == 0,'code tests failed')
    write(root/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')))
    gate_check(root,commit,role)


def claim_value(commit, root, inherited_root):
    return dict(task=TASK,commit=commit,output=str(root.resolve()),inherited_root=str(inherited_root.resolve()),
                inherited_lock_sha256=sha(LOCK),jobs=jobs(),spec=s3.SPEC)


def check_launch(root, commit, inherited_root):
    expected=claim_value(commit,root,inherited_root)
    require(read(REPO/REGISTRY/'attempt.json') == expected, 'S3R1 one-shot claim')
    launch=read(root/'launch.json')
    require(launch['claim'] == expected and launch['task'] == TASK and launch['commit'] == commit
            and launch['jobs'] == jobs() and launch['spec'] == s3.SPEC, 'S3R1 launch contract')
    return launch


def inherited_summary(accepted):
    return [dict(job=r['job'],commit=r['commit'],task=r['task'],selected=r['selected'],
                 best_epoch=r['best_epoch'],checkpoint_sha256=r['checkpoint_sha256']) for r in accepted]


def verify(root, commit, split, source_lock, inherited_root):
    check_launch(root,commit,inherited_root)
    for role in ('wsl','server'):gate_check(root/(role+'_evidence'),commit,role)
    require(read(root/'inherited_reference.json') == read(LOCK), 'inherited reference copy')
    accepted,reference=inherited(inherited_root)
    require(read(root/'inherited_results.json') == inherited_summary(accepted), 'inherited result summary')
    require(s3.references(root/'reference_077') == reference, '077 copy')
    f,t,source=s3.context('B',split,source_lock,'cpu')
    require(read(root/'preflight.json') == preflight_context(f,t,source,commit,reference,accepted), 'preflight changed')
    results=list(accepted)
    for j in jobs():
        require(read(root/(j['id']+'.command.json'))['exit_code'] == 0, 'B worker failed')
        results.append(s3.verify_job(f,t,source,j,root/j['id'],commit,reference[j['id']],task_id=TASK))
    records=[dict(job=r['job'],origin_package='078' if r['job']['setting'] != 'B' else '079',
                  produced_by_commit=r['commit'],produced_by_task=r['task'],updates=r['updates'],best_epoch=r['best_epoch'],
                  checkpoint_sha256=r['checkpoint_sha256'],selected=r['selected']) for r in results]
    require(sum(r['updates'] for r in results[6:]) == 1080 and sum(len(r['history']) for r in results) == 360, 'completion totals')
    return dict(task=TASK,commit=commit,content_status='PASS',scientific_acceptance='PENDING_REVIEW',
                inherited_content_scope='IMMUTABLE_ACCEPTED_078_PLUS_METRIC_RECOMPUTATION',inherited_updates=2520,
                new_updates=1080,total_updates=3600,total_epochs=360,inherited_smoke_updates=4,new_smoke_updates=0,
                provenance=records,**s3.compare(results,reference))


def run(a):
    from p1d4_batch import free_gpus
    s3.s2.c0.check_code(REPO,a.commit)
    for role in ('wsl','server'):gate_check(getattr(a,role+'_evidence'),a.commit,role)
    root=a.output;previous=a.inherited_root.resolve()
    require(not root.resolve().is_relative_to(previous) and not previous.is_relative_to(root.resolve()), 'output overlaps immutable 078')
    require(not (REPO/REGISTRY/'attempt.json').exists(), 'S3R1 attempt already consumed')
    accepted,reference=inherited(previous)
    require(free_gpus([a.gpu]) == [a.gpu], 'GPU busy')
    root.mkdir(parents=True,exist_ok=False);(REPO/REGISTRY).mkdir(parents=True,exist_ok=True)
    claim=claim_value(a.commit,root,previous);write(REPO/REGISTRY/'attempt.json',claim)
    uuid=s3.s2.c0.gpu_uuid(a.gpu)
    write(root/'launch.json',dict(task=TASK,commit=a.commit,spec=s3.SPEC,jobs=jobs(),claim=claim,gpu_uuid=uuid,gpu=a.gpu))
    try:
        for role in ('wsl','server'):
            shutil.copytree(getattr(a,role+'_evidence'),root/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
        shutil.copyfile(LOCK,root/'inherited_reference.json')
        s3.copy_references(previous/'reference_077',root/'reference_077')
        write(root/'inherited_results.json',inherited_summary(accepted))
        write(root/'preflight.json',preflight(a.commit,a.split_manifest,a.source_lock,accepted,reference))
        for j in jobs():
            require(free_gpus([a.gpu]) == [a.gpu], 'GPU occupied')
            argv=[sys.executable,str(Path(__file__).resolve()),'_worker','--commit',a.commit,'--job',j['id'],
                  '--output',str(root),'--inherited-root',str(previous),'--split-manifest',str(a.split_manifest),
                  '--source-lock',str(a.source_lock)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/(j['id']+'.log')).open('xb') as log:
                p=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(root/(j['id']+'.command.json'),dict(argv=argv,exit_code=p.returncode))
            require(p.returncode == 0, 'B job failed; stop and preserve evidence')
        write(root/'verification.json',verify(root,a.commit,a.split_manifest,a.source_lock,previous))
    except BaseException as exc:
        write(root/'failed.json',dict(task=TASK,error_type=type(exc).__name__,reason=str(exc)));raise


def worker(a):
    require(a.job in [j['id'] for j in jobs()], 'only three B jobs allowed')
    launch=check_launch(a.output,a.commit,a.inherited_root)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == launch['gpu_uuid'] and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, 'single GPU binding')
    job=next(j for j in jobs() if j['id'] == a.job)
    for previous in jobs()[:jobs().index(job)]:
        r=read(a.output/previous['id']/'receipt.json')
        require(r['job'] == previous and r['task'] == TASK and r['commit'] == a.commit
                and r['updates'] == 360 and len(r['history']) == 40, 'preceding B job incomplete')
    accepted,reference=inherited(a.inherited_root)
    f,t,source=s3.context('B',a.split_manifest,a.source_lock,'cuda:0')
    require(read(a.output/'preflight.json') == preflight_context(f,t,source,a.commit,reference,accepted), 'B worker preflight changed')
    out=a.output/a.job;out.mkdir(exist_ok=False)
    write(out/'started.json',dict(task=TASK,commit=a.commit,job=job))
    s3.train_one(f,t,source,job,out,a.commit,'cuda:0',reference[a.job],task_id=TASK)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('code-gate','run','_worker','verify','package'))
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=('wsl','server'));p.add_argument('--gpu',type=int,choices=range(4))
    p.add_argument('--job',choices=[j['id'] for j in jobs()])
    for name in ('split-manifest','source-lock','inherited-root','wsl-evidence','server-evidence'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.action == 'code-gate':code_gate(a.output,a.commit,a.role)
    elif a.action == 'package':print(s3.s2.package(a.output,a.commit))
    else:
        s3.s2.c0.check_code(REPO,a.commit)
        require(all(getattr(a,k) is not None for k in ('split_manifest','source_lock','inherited_root')), 'asset paths required')
        a.inherited_root=a.inherited_root.resolve()
        if a.action == 'run':
            require(all(getattr(a,k) is not None for k in ('gpu','wsl_evidence','server_evidence')), 'runtime arguments');run(a)
        elif a.action == 'verify':print(verify(a.output,a.commit,a.split_manifest,a.source_lock,a.inherited_root))
        else:worker(a)


if __name__ == '__main__':main()
