"""Read-only recovery of the nine completed 080 jobs, one CPU process per job."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch
import v9_s4_screen as s4

REPO=Path(__file__).resolve().parent
TASK='V9_S4R1_READONLY_VERIFICATION_20261002'
PRODUCTION_COMMIT='937170fdfdb049e4f623588ef17a40af275b1817'
REGISTRY='.tmp/v9s4r1_attempt_20261002'
LOCK=REPO/'configs/v9_s4r1_input_lock.json'
ADDITIONS=('configs/v9_s4r1_input_lock.json','tests/test_v9_s4r1_verify.py','v9_s4r1_verify.py')
TESTS=('tests/test_v9_s4r1_verify.py',*s4.TESTS)
TEST_COUNT=250
read,write,sha,digest,require=s4.read,s4.write,s4.sha,s4.digest,s4.require


def check_code(commit):
    s4.s3.s2.c0.check_code(REPO,commit)
    changes=subprocess.check_output(['git','diff','--name-status',PRODUCTION_COMMIT,commit],cwd=REPO,text=True)
    require(set(changes.splitlines())=={'A\t'+p for p in ADDITIONS},'production code must remain unchanged')


def lock_value():
    lock=read(LOCK)
    require(lock['schema']=='v9_s4r1_input_v1' and lock['production_commit']==PRODUCTION_COMMIT and
            lock['task']==s4.TASK and lock['spec']==s4.SPEC and lock['jobs']==s4.jobs() and len(lock['files'])==974,
            '080 lock contract')
    return lock


def locked_files(root,expected):
    root=Path(root);require(root.is_dir() and not root.is_symlink(),'080 input directory')
    actual=set()
    for p in root.rglob('*'):
        require(not p.is_symlink(),'080 input symlink')
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    require(actual-{'checksums.sha256'}==set(expected),'080 exact member set')
    for name,value in expected.items():
        p=root/name
        require(p.resolve().is_relative_to(root.resolve()) and sha(p)==value,'080 immutable file '+name)
    if (root/'checksums.sha256').exists():
        lines=(root/'checksums.sha256').read_text(encoding='utf8').splitlines()
        pairs=[line.split('  ',1) for line in lines]
        require(all(len(p)==2 for p in pairs) and len(pairs)==len(expected) and
                {p[1]:p[0] for p in pairs}==expected,'080 checksum sidecar')
    return digest({p:sha(root/p) for p in sorted(actual)})


def inputs(root):
    lock=lock_value();snapshot=locked_files(root,lock['files'])
    launch=read(root/'launch.json');claim=launch['claim']
    require(launch['task']==s4.TASK and launch['commit']==PRODUCTION_COMMIT and claim['task']==s4.TASK and
            claim['commit']==PRODUCTION_COMMIT and claim['jobs']==s4.jobs() and claim['spec']==s4.SPEC and
            claim['smoke']==list(s4.SMOKE),'080 production launch')
    require(not (root/'verification.json').exists() and
            'already open in this process' in read(root/'failed.json')['reason'],'080 failed verifier retained')
    for role in ('wsl','server'):s4.gate_check(root/(role+'_evidence'),PRODUCTION_COMMIT,role)
    reference=s4.s3.references(root/'reference_077');results=[]
    for job in s4.jobs():
        folder=root/job['id'];r=read(folder/'receipt.json')
        require(r['task']==s4.TASK and r['commit']==PRODUCTION_COMMIT and r['job']==job and
                r['identity']==read(folder/'identity.json') and r['identity']['spec']==s4.SPEC and
                read(root/(job['id']+'.command.json'))['exit_code']==0,'080 completed job identity')
        results.append(r)
    for setting in s4.SMOKE:
        name='smoke_'+setting;r=read(root/name/'receipt.json')
        require(read(root/(name+'.command.json'))['exit_code']==0 and r['task']==s4.TASK and r['commit']==PRODUCTION_COMMIT and
                r['updates']==2 and r['content_status']=='PASS' and sha(root/name/'state.pt')==r['checkpoint_sha256'],
                '080 completed smoke identity')
    return snapshot,results,reference


def separate_output(root,previous):
    require(not root.resolve().is_relative_to(previous.resolve()) and
            not previous.resolve().is_relative_to(root.resolve()),'output overlaps immutable 080')


def claim_value(a):
    return dict(task=TASK,verification_commit=a.commit,production_commit=PRODUCTION_COMMIT,
                output=str(a.output.resolve()),input_root=str(a.input_root.resolve()),input_lock_digest=digest(lock_value()),
                jobs=s4.jobs(),new_training_updates=0,device='cpu')


def check_launch(a):
    expected=claim_value(a);launch=read(a.output/'launch.json')
    require(read(REPO/REGISTRY/'attempt.json')==expected and launch==dict(task=TASK,commit=a.commit,claim=expected),
            'readonly launch and persistent claim')


def run_child(argv,prefix):
    """subprocess.run starts a fresh interpreter and waits for its complete exit."""
    with Path(str(prefix)+'.log').open('xb') as stream:
        result=subprocess.run(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
                              env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    write(Path(str(prefix)+'.command.json'),dict(argv=argv,exit_code=result.returncode,device='cpu'))
    require(result.returncode==0,'readonly child failed; preserve outputs and stop')


def worker(a):
    job=next((j for j in s4.jobs() if j['id']==a.job),None);require(job is not None,'only nine original jobs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only verification required')
    separate_output(a.output,a.input_root);check_launch(a);inputs(a.input_root)
    for previous in s4.jobs()[:s4.jobs().index(job)]:check_worker(a,previous)
    out=a.output/job['id'];out.mkdir(exist_ok=False)
    write(out/'started.json',dict(task=TASK,job=job,verification_commit=a.commit,pid=os.getpid()))
    torch.set_num_threads(1)
    # Exactly one context in this interpreter. Its LMDB owners die with the process.
    factory,trainer,source=s4.s3.context(job['setting'],a.split_manifest,a.source_lock,'cpu')
    require(all(p.device.type=='cpu' for p in trainer.model.parameters()) and not torch.cuda.is_initialized(),'CPU model')
    reference=s4.s3.references(a.input_root/'reference_077')[f'{job["setting"]}_B1_1E4_s42']
    r=s4.verify_job(factory,trainer,source,job,a.input_root/job['id'],PRODUCTION_COMMIT,reference)
    require(r==read(a.input_root/job['id']/'receipt.json'),'original receipt changed')
    write(out/'verification.json',dict(task=TASK,job=job,verification_commit=a.commit,production_commit=PRODUCTION_COMMIT,
          input_lock_digest=digest(lock_value()),original_receipt_sha256=sha(a.input_root/job['id']/'receipt.json'),
          checkpoint_sha256=r['checkpoint_sha256'],content_status='PASS',raw_train_gradient_replayed=True,
          new_training_updates=0,device='cpu',pid=os.getpid(),torch_version=str(torch.__version__)))


def check_worker(a,job):
    name=job['id'];r=read(a.output/name/'verification.json');started=read(a.output/name/'started.json')
    require(type(r.get('pid')) is int and r['pid']>0 and type(r.get('torch_version')) is str and r['torch_version'],'worker runtime identity')
    expected=dict(task=TASK,job=job,verification_commit=a.commit,production_commit=PRODUCTION_COMMIT,
          input_lock_digest=digest(lock_value()),original_receipt_sha256=sha(a.input_root/name/'receipt.json'),
          checkpoint_sha256=read(a.input_root/name/'receipt.json')['checkpoint_sha256'],content_status='PASS',
          raw_train_gradient_replayed=True,new_training_updates=0,device='cpu',pid=r['pid'],torch_version=r['torch_version'])
    require(digest(r)==digest(expected) and started==dict(task=TASK,job=job,verification_commit=a.commit,pid=r['pid']), 'worker verification identity')
    command=read(a.output/(name+'.command.json'))
    require(type(command['exit_code']) is int and command['exit_code']==0 and command['device']=='cpu','worker process exit')
    return r


def run(a):
    check_code(a.commit);separate_output(a.output,a.input_root)
    require(not (REPO/REGISTRY/'attempt.json').exists(),'readonly attempt already consumed')
    for role in ('wsl','server'):gate_check(getattr(a,role+'_evidence'),a.commit,role)
    before,results,reference=inputs(a.input_root)
    a.output.mkdir(parents=True,exist_ok=False);(REPO/REGISTRY).mkdir(parents=True,exist_ok=True)
    claim=claim_value(a);write(REPO/REGISTRY/'attempt.json',claim)
    write(a.output/'launch.json',dict(task=TASK,commit=a.commit,claim=claim))
    try:
        write(a.output/'input_lock.json',lock_value());write(a.output/'input_before.json',dict(snapshot=before))
        for role in ('wsl','server'):
            shutil.copytree(getattr(a,role+'_evidence'),a.output/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
        for job in s4.jobs():
            argv=[sys.executable,str(Path(__file__).resolve()),'_worker','--commit',a.commit,'--job',job['id'],
                  '--output',str(a.output),'--input-root',str(a.input_root),'--split-manifest',str(a.split_manifest),'--source-lock',str(a.source_lock)]
            run_child(argv,a.output/job['id']);check_worker(a,job)
        after,_,_=inputs(a.input_root);require(before==after,'080 input mutated during verification');check_code(a.commit)
        write(a.output/'input_after.json',dict(snapshot=after))
        require(sum(r['updates'] for r in results)==3600 and sum(len(r['history']) for r in results)==360,'080 completed training totals')
        write(a.output/'verification.json',dict(task=TASK,verification_commit=a.commit,production_commit=PRODUCTION_COMMIT,
            content_status='PASS',scientific_acceptance='PENDING_REVIEW',input_snapshot=after,input_lock_digest=digest(lock_value()),
            device='cpu',new_training_updates=0,new_smoke_updates=0,original_epochs=360,original_training_updates=3600,original_smoke_updates=4,
            workers=[check_worker(a,j) for j in s4.jobs()],**s4.compare(results,reference)))
    except BaseException as exc:
        # Failures retain both the original input and partial recovery evidence.
        write(a.output/'failed.json',dict(task=TASK,error_type=type(exc).__name__,reason=str(exc),new_training_updates=0));raise


def gate_check(root,commit,role):
    require(read(root/'gate.json')==dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')),'S4R1 gate identity')
    require(read(root/'command.json')['exit_code']==0,'S4R1 gate exit')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases)==TEST_COUNT and all(not any(c.findall(t) for t in ('failure','error','skipped')) for c in cases) and
            {c.attrib['classname'] for c in cases}=={'tests.'+Path(t).stem for t in TESTS},'S4R1 complete gate')


def code_gate(root,commit,role):
    require(role in ('wsl','server'),'S4R1 gate role');check_code(commit);root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as stream:
        p=subprocess.run(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json',dict(argv=argv,exit_code=p.returncode));require(p.returncode==0,'S4R1 tests failed')
    write(root/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')));gate_check(root,commit,role)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('code-gate','run','_worker','package'))
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=('wsl','server'))
    p.add_argument('--job',choices=[j['id'] for j in s4.jobs()])
    for name in ('input-root','split-manifest','source-lock','wsl-evidence','server-evidence'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.action=='code-gate':code_gate(a.output,a.commit,a.role)
    elif a.action=='package':print(s4.s3.s2.package(a.output,a.commit))
    else:
        require(all(getattr(a,k) is not None for k in ('input_root','split_manifest','source_lock')),'S4R1 asset paths')
        a.input_root=a.input_root.resolve();check_code(a.commit)
        if a.action=='run':
            require(a.wsl_evidence is not None and a.server_evidence is not None,'separate WSL/server gates');run(a)
        else:worker(a)


if __name__=='__main__':main()
