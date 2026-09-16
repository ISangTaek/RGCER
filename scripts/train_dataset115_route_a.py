"""S5F2 frozen source5 + target15 train/validation CLI. No test or HPO option."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('csv','split-manifest','tox-manifest','expected-commit','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=['source','target'],required=True)
    p.add_argument('--seed',type=int,choices=range(42,47),required=True)
    p.add_argument('--method',choices=['B0','B1','RPT'])
    p.add_argument('--source-output')
    p.add_argument('--verify-only',action='store_true')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    import torch
    from dataset115_adapter import Dataset115Table
    from dataset115_route_a_smoke import ARCH,SERVER_TOX_SHA
    from dataset115_route_a_training import CONFIG,TASK_ID,SourceTrainer,RouteATrainer,verify_source,load_source,read_json
    from dataset115_route_b_run import verify_training_run
    from dataset115_training import require,_write_json
    repo=Path(__file__).resolve().parents[1];out=Path(a.output)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    try:
        require(len(a.expected_commit)==40 and all(c in '0123456789abcdef' for c in a.expected_commit),'full commit required')
        require(git('rev-parse','HEAD')==a.expected_commit and not git('status','--porcelain'),'commit/worktree differs')
        require(a.verify_only or not out.exists(),'output exists')
        require((a.role=='source' and a.method is None and a.source_output is None) or
            (a.role=='target' and a.method in ('B0','B1','RPT') and
             ((a.method=='B0' and a.source_output is None) or (a.method!='B0' and a.source_output is not None))), 'role/source/method scope')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        require(visible in ('0','1','2','3') and torch.cuda.is_available() and torch.cuda.device_count()==1,'single CUDA GPU0-3 required')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS config required')
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        table=Dataset115Table.load(a.csv,a.split_manifest,a.tox_manifest,expected_tox_sha=SERVER_TOX_SHA)
        args=SimpleNamespace(**ARCH)
        def make_source():return SourceTrainer(args,table.view('A','source','train'),seed=a.seed,config=CONFIG,device='cuda:0')
        binding=None;source=None;source_receipt=None
        if a.role=='target' and a.method!='B0':source,binding,source_receipt=load_source(a.source_output,make_source())
        def make():
            if a.role=='source':return make_source()
            return RouteATrainer(args,source,binding,table.view('A','target','train'),table.view('A','target','validation'),
                method=a.method,seed=a.seed,config=CONFIG,device='cuda:0')
        if not a.verify_only:
            trainer=make();trainer.run(out);del trainer
            _write_json(out/'run_provenance.json',dict(task_id=TASK_ID,commit=a.expected_commit,role=a.role,method=a.method,
                seed=a.seed,source_receipt=source_receipt,source_output=a.source_output,cuda_visible_devices=visible,
                gpu_name=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),formal_test_authorized=False,
                peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(0)))
        provenance=read_json(out/'run_provenance.json')
        require(provenance['task_id']==TASK_ID and provenance['commit']==a.expected_commit and provenance['role']==a.role
            and provenance['method']==a.method and type(provenance['seed']) is int and provenance['seed']==a.seed
            and provenance['source_receipt']==source_receipt and provenance['formal_test_authorized'] is False,'run provenance')
        verification=verify_source(out,make()) if a.role=='source' else verify_training_run(out,make())
        verification.update(task_id=TASK_ID,role=a.role)
        if (out/'content_verification.json').exists():require(read_json(out/'content_verification.json')==verification,'stale verification')
        else:_write_json(out/'content_verification.json',verification)
        import json
        print(json.dumps(verification,ensure_ascii=False,allow_nan=False))
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
