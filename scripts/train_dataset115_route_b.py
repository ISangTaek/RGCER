"""S5B frozen 15-run training/validation entry; no test/calibration option."""
import argparse
from pathlib import Path
import sys
import json
import subprocess
import os

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('csv','split-manifest','tox-manifest','source-lock','expected-commit','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--method',choices=['B0','B1','RPT'],required=True)
    p.add_argument('--seed',type=int,choices=range(42,47),required=True)
    p.add_argument('--verify-only',action='store_true')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    from dataset115_training import RouteBTrainer, require
    from dataset115_route_b_run import CONFIG, SERVER_TOX_SHA, TASK_ID, verify_training_run
    from dataset115_adapter import Dataset115Table
    from dataset115_source import load_binding,model_args,load_route_b_encoder
    import torch
    repo=Path(__file__).resolve().parents[1]; output=Path(a.output)
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    try:
        require(len(a.expected_commit)==40 and all(c in '0123456789abcdef' for c in a.expected_commit),'full commit required')
        require(git('rev-parse','HEAD')==a.expected_commit,'commit differs')
        require(not git('status','--porcelain'),'worktree not clean')
        require(torch.cuda.is_available(),'CUDA unavailable; no CPU training fallback')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        require(visible in ('0','1','2','3'),'exactly one physical GPU0-3 required')
        require(torch.cuda.device_count()==1,'single logical CUDA device required')
        require(a.verify_only or not output.exists(),'output exists')
        # Reproducible kernels or an explicit error; do not silently relax.
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS_WORKSPACE_CONFIG required')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        teacher,_=load_binding(a.source_lock,a.seed)
        source=None;source_identity=None;source_receipt=None;args=model_args(teacher)
        if a.method!='B0':
            source,args,source_receipt=load_route_b_encoder(repo,a.source_lock,a.seed)
            source_identity=dict(seed=a.seed,teacher_sha256=source_receipt['teacher']['sha256'],init_sha256=source_receipt['init']['sha256'])
        table=Dataset115Table.load(a.csv,a.split_manifest,a.tox_manifest,expected_tox_sha=SERVER_TOX_SHA)
        def make_trainer():
            return RouteBTrainer(args,source,source_identity,table.view('B','target','train'),
                table.view('B','target','validation'),method=a.method,seed=a.seed,config=CONFIG,device='cuda:0')
        if not a.verify_only:
            trainer=make_trainer();trainer.run(output);del trainer
            record=dict(task_id=TASK_ID,commit=a.expected_commit,method=a.method,seed=a.seed,
                cuda_visible_devices=visible,gpu_name=torch.cuda.get_device_name(0),
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                source_receipt=source_receipt,scope='TRAIN_AND_VALIDATION_ONLY',
                deterministic_algorithms=True,formal_test_authorized=False,
                peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(0))
            with (output/'run_provenance.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2,allow_nan=False)
        provenance=json.loads((output/'run_provenance.json').read_text(encoding='utf8'))
        require(provenance['task_id']==TASK_ID and provenance['commit']==a.expected_commit
                and provenance['method']==a.method and provenance['seed']==a.seed
                and provenance['scope']=='TRAIN_AND_VALIDATION_ONLY'
                and provenance['formal_test_authorized'] is False,'run provenance differs')
        verification=verify_training_run(output,make_trainer())
        target=output/'content_verification.json'
        if target.exists():
            require(json.loads(target.read_text())==verification,'previous verification differs')
        else:
            with target.open('x',encoding='utf8') as f:json.dump(verification,f,indent=2,allow_nan=False)
        print(json.dumps(verification,ensure_ascii=False,allow_nan=False))
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr)
        return 2
    return 0


if __name__=='__main__': raise SystemExit(main())
