"""P1D3 seed42 real smoke ONLY: 11 updates per setting; never formal train/test."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('expected-commit','setting','output','split-manifest','source-lock'):
        p.add_argument('--'+k,required=True,choices=('ToxAcute','A','B') if k=='setting' else None)
    p.add_argument('--verify-only',action='store_true')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    import torch
    from p1d_optimization import require
    from p1d_runtime import prepare,verify_live_gradients,historical_check,historical_replay,write_json,claim_attempt
    from p1d_smoke import run_setting_smoke,verify_smoke
    repo=Path(__file__).resolve().parents[1];out=Path(a.output)
    try:
        def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        require(len(a.expected_commit)==40 and all(c in '0123456789abcdef' for c in a.expected_commit),'full commit')
        require(git('rev-parse','HEAD')==a.expected_commit and not git('status','--porcelain'),'commit/worktree')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        require(visible in ('0','1','2','3') and torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU0-3 required')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS config')
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_num_threads(1)
        require(out.is_dir() if a.verify_only else not out.exists(),'output existence')
        lock=json.loads((repo/'configs/p1d3_smoke_lock.json').read_bytes())
        adapter,tasks,identity,tox=prepare(repo,lock,setting=a.setting,split_manifest=a.split_manifest,
                                         source_lock=a.source_lock,device='cuda:0')
        identity['implementation_commit']=a.expected_commit
        identity['runtime']=dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
            device_name=torch.cuda.get_device_name(0),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
        if not a.verify_only:
            # Reject missing/incompatible historical assets before spending any
            # optimizer update. This is validation-only inference, not retraining.
            historical_result=historical_replay(repo,tox)
            claim_attempt(repo,a.setting,a.expected_commit,out)
            run_setting_smoke(adapter,setting=a.setting,task_names=tasks,output_dir=out,expected_identity=identity)
            if historical_result is not None:write_json(out/'historical_validation.json',historical_result)
        content=verify_smoke(out,identity)
        live=verify_live_gradients(out,adapter,setting=a.setting,expected_identity=identity)
        historical=historical_check(repo,out,tox,replay=False)
        verification=dict(task_id=lock['task_id'],setting=a.setting,commit=a.expected_commit,
            observed_optimizer_updates=content['observed_optimizer_updates'],content=content,live=live,
            historical=historical,formal_training_authorized=False,test_authorized=False,acceptance_status='PENDING_CODEX_REVIEW')
        require(verification['observed_optimizer_updates']==11,'smoke budget')
        write_json(out/'content_verification.json',verification)
        print(json.dumps(dict(task_id=lock['task_id'],setting=a.setting,updates=11,content_status='PASS',
                             acceptance_status='PENDING_CODEX_REVIEW'),ensure_ascii=False))
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
