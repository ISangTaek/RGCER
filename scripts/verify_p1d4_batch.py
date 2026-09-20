"""Read all actual P1D4 artifacts, regenerate selection and check the full matrix.

Run on an idle server GPU, not CPU: deterministic best replay uses the original
CUDA arithmetic. No optimizer steps, retry, training, test or calibration.
"""
from pathlib import Path
import argparse,os,sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from p1d4_batch import read,write,check_commit,free_gpus,execute_matrix,REGISTRY
from p1d_optimization import require


def verify(root,commit,gpu):
    root=Path(root).resolve()
    check_commit(REPO,commit)
    require(free_gpus([gpu])==[gpu],'requested verifier GPU is occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from p1d4_runtime import verify_job
    from p1d_tox_training import _same
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one visible CUDA device')
    torch.use_deterministic_algorithms(True)
    launch=read(root/'launch.json')
    reported_summary=read(root/'batch_summary.json')
    require(launch['commit']==commit,'launch commit')
    require(read(root/'supervisor_exit.json')['exit_code']==0,'supervisor incomplete/failed')
    def phase(jobs):
        results={}
        for j in jobs:
            name=j['run_id'];claim=read(REPO/REGISTRY/(name+'.json'))
            require(_same(claim,read(root/'attempt_claims'/(name+'.json'))),'archived claim differs')
            require(_same(claim['job'],j) and claim['commit']==commit and claim['root']==str(root),'claim identity')
            require(read(root/'logs'/name/'exit.json')['exit_code']==0,'worker exit')
            current=verify_job(REPO,j,root/'runs'/name,split_manifest=launch['split_manifest'],
                               source_lock=launch['source_lock'],device='cuda:0')
            saved=read(root/'receipts'/(name+'.json'))
            require(_same(saved['job'],current['job']) and _same(saved['result'],current['result']), 'recomputed receipt differs')
            require(_same(reported_summary['results'][name],saved),'summary result differs from receipt')
            results[name]=current
            print('VERIFIED',name,flush=True)
        return results
    result=execute_matrix(phase)
    summary=read(root/'batch_summary.json')
    for k in ('task_id','selections','plan','execution_status','validation_status','acceptance_status'):
        require(_same(summary[k],result[k]),'recomputed matrix '+k)
    require(set(summary['results'])==set(result['results']),'summary run matrix')
    from p1d4_delivery import snapshot
    write(root/'final_verification.json',dict(task_id=result['task_id'],commit=commit,
          checked_runs=len(result['results']),optimizer_updates_performed=0,
          content_status='PASS',acceptance_status='PENDING_REVIEW',budget=result['plan']['budget'],files=snapshot(root)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--commit',required=True);p.add_argument('--gpu',required=True,type=int)
    args=p.parse_args();verify(args.output,args.commit,args.gpu)
