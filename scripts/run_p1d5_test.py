"""P1D5 two-host code gates, fixed 25 inference + 5 reuse, read-only audit."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
import xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from p1d4_batch import check_commit,write,read,utc,free_gpus
from p1d_optimization import require

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gate_check(path,commit,role):
    path=Path(path);r=read(path/'preflight.json')
    require(r['commit']==commit and r['role']==role and r['exit_code']==0,'preflight identity')
    require(digest(path/'tests.xml')==r['junit_sha256'],'JUnit hash')
    cases=ET.parse(path/'tests.xml').findall('.//testcase')
    require(cases and all(not c.findall('failure') and not c.findall('error') and not c.findall('skipped') for c in cases),'JUnit failures/skips')
    require({c.attrib['classname'] for c in cases}=={'tests.'+p.stem for p in (REPO/'tests').glob('test_p1d*.py')},'complete P1D suite')
    cmd=read(path/'tests_command.json');require(cmd['exit_code']==0,'test exit')
    return r

def preflight(a):
    require(not a.output.exists(),'fresh gate output');a.output.mkdir(parents=True)
    files=sorted((REPO/'tests').glob('test_p1d*.py'))
    argv=[sys.executable,'-m','pytest',*[str(p.relative_to(REPO)) for p in files],'-q','--basetemp',str(a.output/'pytest_tmp'),'--junitxml',str(a.output/'tests.xml')]
    with (a.output/'tests.stdout.log').open('xb') as out,(a.output/'tests.stderr.log').open('xb') as err:
        rc=subprocess.call(argv,cwd=REPO,stdout=out,stderr=err)
    write(a.output/'tests_command.json',dict(argv=argv,exit_code=rc,finished_utc=utc()));require(rc==0,'tests failed')
    if a.role=='server':
        require(a.wsl_gate is not None,'WSL gate required');gate_check(a.wsl_gate,a.commit,'wsl')
        shutil.copytree(a.wsl_gate,a.output/'wsl_gate',ignore=shutil.ignore_patterns('pytest_tmp'))
    write(a.output/'preflight.json',dict(role=a.role,commit=a.commit,exit_code=0,junit_sha256=digest(a.output/'tests.xml'),optimizer_updates=0))
    gate_check(a.output,a.commit,a.role)

def archive(a):
    root=a.output;v=read(root/'verification.json')
    require(v['commit']==a.commit and v['content_status']=='PASS' and v['checked_runs']==30,'verified complete batch required')
    actual=snapshot(root);require(actual==v['files'],'changed files after verification')
    selected=dict(actual);selected['verification.json']=digest(root/'verification.json')
    dest=root.with_name(root.name+'.zip');side=Path(str(dest)+'.sha256')
    require(not dest.exists() and not side.exists(),'archive already exists')
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n in sorted(selected):z.write(root/n,n)
        z.writestr('checksums.sha256',''.join(f'{v}  {n}\n' for n,v in sorted(selected.items())))
    with zipfile.ZipFile(dest) as z:
        require(z.testzip() is None,'CRC')
        for n,h in selected.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'archive SHA')
    with side.open('x',encoding='utf8') as f:f.write(digest(dest)+'  '+dest.name+'\n')
    print(str(dest),digest(dest))

def snapshot(root):
    result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'symlink evidence')
        if not p.is_file() or p==root/'verification.json':continue
        require(p.suffix not in ('.pt','.zip','.pem','.key') and p.name not in ('.env','id_rsa','id_ed25519'),'unexpected asset/credential')
        result[p.relative_to(root).as_posix()]=digest(p)
    return result

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('mode',choices=['preflight','run','verify','package'])
    p.add_argument('--commit',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--role',choices=['wsl','server']);p.add_argument('--wsl-gate',type=Path);p.add_argument('--server-gate',type=Path)
    p.add_argument('--lock',type=Path);p.add_argument('--split-manifest',type=Path);p.add_argument('--training-root',type=Path);p.add_argument('--gpu',type=int,choices=range(4))
    a=p.parse_args();a.output=a.output.resolve();check_commit(REPO,a.commit)
    if a.mode=='preflight':
        require(a.role is not None,'role required');preflight(a);return
    if a.mode=='package':archive(a);return
    require(a.lock is not None and a.split_manifest is not None and a.training_root is not None,'frozen inputs required')
    from p1d5_test import load_lock,Data,execute,verify,selected_path,load_state,predict,LOCK_SHA
    lock=load_lock(a.lock)
    if a.mode=='run':
        require(a.server_gate is not None and a.gpu is not None,'server gate and idle GPU required')
        gate_check(a.server_gate,a.commit,'server');gate_check(a.server_gate/'wsl_gate',a.commit,'wsl')
        require(free_gpus([a.gpu])==[a.gpu],'GPU occupied')
        require(not a.output.exists(),'fresh output required')
        registry=REPO/'.tmp/p1d5_test_20260920';registry.mkdir(parents=True,exist_ok=True)
        write(registry/'attempt.json',dict(commit=a.commit,output=str(a.output),started_utc=utc(),lock_sha256=LOCK_SHA))
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU')
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=Data(REPO,a.split_manifest,lock)
        # Every selected artifact is read with locked SHA, including inherited B1.
        loader=lambda row:load_state(selected_path(row,REPO,a.training_root),row)
        started=utc()
        execute(lock,data,a.output,loader,lambda state,row,split:predict(state,row,data,split,'cuda:0'))
        write(a.output/'provenance.json',dict(commit=a.commit,lock_sha256=LOCK_SHA,started_utc=started,finished_utc=utc(),
            argv=sys.argv,torch_version=str(torch.__version__),cuda_version=torch.version.cuda,physical_gpu=a.gpu,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0))
        shutil.copytree(a.server_gate,a.output/'server_gate',ignore=shutil.ignore_patterns('pytest_tmp'))
        shutil.copy2(registry/'attempt.json',a.output/'attempt.json')
        print('EXPORTED_PENDING_READ_ONLY_VERIFICATION');return
    provenance=read(a.output/'provenance.json');require(provenance['commit']==a.commit and provenance['lock_sha256']==LOCK_SHA,'run provenance')
    data=Data(REPO,a.split_manifest,lock)
    for row in lock['rows']:load_state(selected_path(row,REPO,a.training_root),row)
    result=verify(lock,data,a.output);result.update(commit=a.commit,files=snapshot(a.output))
    write(a.output/'verification.json',result);print('VERIFIED_30_PENDING_CODEX_REVIEW')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);raise SystemExit(2)
