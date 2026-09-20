"""Separated WSL code gate and server asset gate; zero optimizer updates."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from p1d4_batch import check_commit,write,read,utc,free_gpus
from p1d_optimization import require


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_wsl(folder,commit):
    folder=Path(folder)
    receipt=read(folder/'preflight.json')
    require(receipt['role']=='wsl' and receipt['commit']==commit and receipt['exit_code']==0,'WSL receipt identity')
    require(sha(folder/'tests.xml')==receipt['junit_sha256'],'WSL JUnit SHA')
    tree=ET.parse(folder/'tests.xml')
    cases=tree.findall('.//testcase')
    require(cases and all(not list(c) or all(x.tag not in ('failure','error','skipped') for x in c) for c in cases),
            'WSL failure/error/skip')
    expected={'tests.'+p.stem for p in (REPO/'tests').glob('test_p1d*.py')}
    require({c.attrib['classname'] for c in cases}==expected,'WSL complete P1D suite')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['wsl','server'],required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--commit',required=True)
    p.add_argument('--split-manifest',type=Path);p.add_argument('--source-lock',type=Path)
    p.add_argument('--wsl-evidence',type=Path);p.add_argument('--gpu',type=int)
    a=p.parse_args();root=a.output.resolve()
    check_commit(REPO,a.commit)
    require(not root.exists(),'new preflight output directory required')
    root.mkdir(parents=True)
    files=sorted((REPO/'tests').glob('test_p1d*.py'))
    argv=[sys.executable,'-m','pytest',*[str(f.relative_to(REPO)) for f in files],'-q',
          '--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.stdout.log').open('xb') as out,(root/'tests.stderr.log').open('xb') as err:
        code=subprocess.call(argv,cwd=REPO,stdout=out,stderr=err)
    write(root/'tests_command.json',dict(argv=argv,exit_code=code,finished_utc=utc()))
    require(code==0,'P1D tests failed')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(cases and all(not c.findall('failure') and not c.findall('error') and not c.findall('skipped') for c in cases),
            'tests failure/error/skip')
    require({c.attrib['classname'] for c in cases}=={'tests.'+f.stem for f in files},'all P1D test files executed')
    receipt=dict(role=a.role,commit=a.commit,exit_code=0,junit_sha256=sha(root/'tests.xml'),
                 tests=len(cases),optimizer_updates=0,finished_utc=utc())
    if a.role=='server':
        require(a.wsl_evidence is not None and a.split_manifest is not None and a.source_lock is not None
                and a.gpu is not None,'server inputs and WSL evidence required')
        receipt['wsl']=check_wsl(a.wsl_evidence,a.commit)
        shutil.copytree(a.wsl_evidence,root/'wsl_evidence',ignore=shutil.ignore_patterns('pytest_tmp'))
        require(free_gpus([a.gpu])==[a.gpu],'preflight GPU occupied')
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        from p1d4_runtime import factory_for
        from p1d4_reuse import LOCK,bound_history,check_files
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one preflight GPU')
        torch.use_deterministic_algorithms(True)
        for setting in ('ToxAcute','A','B'):
            for seed in range(42,47):
                f=factory_for(REPO,setting=setting,seed=seed,split_manifest=a.split_manifest,
                              source_lock=a.source_lock,device='cuda:0')
                t=f.make_trainer('B1_low') if setting=='ToxAcute' else f.controlled_trainer(arm='B1_low',device='cuda:0')
                require(not t.optimizer.state,'preflight optimizer must be empty')
                print('BOUND',setting,seed,flush=True)
                del t,f
                torch.cuda.empty_cache()
        aliases=json.loads(LOCK.read_bytes())['runs']
        for alias in aliases:check_files(REPO,bound_history(alias))
        receipt.update(identities_checked=15,reuse_assets_checked=18,
                       split_manifest_sha256=sha(a.split_manifest),source_lock_sha256=sha(a.source_lock))
    write(root/'preflight.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
