"""Persistent P1D4 supervisor launcher and internal worker entry.

Invoke only under the matching P1D4 READY card; no extra budget is implied.
"""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from p1d4_batch import TASK,REGISTRY,write,read,utc,check_commit,run_batch,worker,free_gpus
from p1d_optimization import require


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['launch','supervisor','worker','status'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--commit')
    p.add_argument('--split-manifest',type=Path)
    p.add_argument('--source-lock',type=Path)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--run-id')
    p.add_argument('--preflight',type=Path)
    a=p.parse_args();root=a.output.resolve()
    if a.mode=='status':
        for name in ('supervisor_exit.json','batch_summary.json','batch_failure.json','launch.json'):
            if (root/name).is_file():
                value=read(root/name)
                if name=='batch_summary.json':value={k:value[k] for k in ('task_id','execution_status','validation_status','acceptance_status')}
                print(name,value)
        return 0
    require(a.split_manifest is not None and a.source_lock is not None,'input attachment paths required')
    split=a.split_manifest.resolve();source=a.source_lock.resolve()
    if a.mode=='launch':
        require(os.name=='posix','formal server Linux required')
        check_commit(REPO,a.commit)
        gpus=[int(x) for x in a.gpus.split(',')]
        available=free_gpus(gpus)
        require(available,'no requested GPU currently free')
        require(split.is_file() and source.is_file(),'input attachments missing')
        from p1d_tox import file_sha
        require(a.preflight is not None,'server preflight receipt required')
        preflight=read(a.preflight/'preflight.json')
        require(preflight['role']=='server' and preflight['commit']==a.commit and preflight['exit_code']==0
                and preflight['optimizer_updates']==0 and preflight['identities_checked']==15
                and preflight['reuse_assets_checked']==18 and preflight['wsl']['commit']==a.commit,
                'server/WSL preflight not complete')
        require(preflight['split_manifest_sha256']==file_sha(split)
                and preflight['source_lock_sha256']==file_sha(source),'preflight input bytes')
        require(not root.exists(),'new output directory required')
        folder=REPO/REGISTRY;folder.mkdir(parents=True,exist_ok=True)
        write(folder/'batch.json',dict(task_id=TASK,root=str(root),commit=a.commit,utc=utc()))
        root.mkdir(parents=True)
        shutil.copytree(a.preflight,root/'server_preflight',ignore=shutil.ignore_patterns('pytest_tmp'))
        write(root/'launch.json',dict(task_id=TASK,commit=a.commit,gpus=gpus,utc=utc(),
                                     split_manifest=str(split),source_lock=str(source),preflight=preflight))
        argv=[sys.executable,'-u',str(Path(__file__).resolve()),'supervisor','--output',str(root),
              '--commit',a.commit,'--split-manifest',str(split),'--source-lock',str(source),'--gpus',a.gpus]
        with (root/'supervisor.stdout.log').open('xb') as out,(root/'supervisor.stderr.log').open('xb') as err:
            process=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        write(root/'supervisor_process.json',dict(pid=process.pid,argv=argv,started_utc=utc()))
        print('LAUNCHED',process.pid,str(root))
        return 0
    launch=read(root/'launch.json')
    require(launch['commit']==a.commit and launch['split_manifest']==str(split)
            and launch['source_lock']==str(source),'launch input identity')
    if a.mode=='worker':return worker(REPO,root,a.commit,a.run_id,split,source)
    code=1
    try:
        gpus=[int(x) for x in a.gpus.split(',')]
        require(gpus==launch['gpus'],'launch GPU identity')
        code=run_batch(REPO,root,a.commit,gpus,split,source)
    except Exception:
        traceback.print_exc()
    finally:
        write(root/'supervisor_exit.json',dict(exit_code=code,pid=os.getpid(),finished_utc=utc()))
    return code


if __name__=='__main__':
    raise SystemExit(main())
