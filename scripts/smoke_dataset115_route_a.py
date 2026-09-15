"""Seed42 Nonhuman104 -> Human5 GPU smoke; no formal training options."""
import argparse,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('csv','split-manifest','tox-manifest','expected-commit','output'):p.add_argument('--'+n,required=True)
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    from dataset115_route_a_smoke import run_smoke,SERVER_TOX_SHA,require,write_json
    from dataset115_adapter import Dataset115Table
    import torch
    repo=Path(__file__).resolve().parents[1]
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    try:
        require(len(a.expected_commit)==40 and all(c in '0123456789abcdef' for c in a.expected_commit),'full commit required')
        require(git('rev-parse','HEAD')==a.expected_commit and not git('status','--porcelain'),'code identity/dirty worktree')
        require(not Path(a.output).exists(),'output exists')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        require(visible in ('0','1','2','3') and torch.cuda.is_available() and torch.cuda.device_count()==1,'one actual GPU0-3 required')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS setting required')
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        table=Dataset115Table.load(a.csv,a.split_manifest,a.tox_manifest,expected_tox_sha=SERVER_TOX_SHA)
        run_smoke(table,a.output,device='cuda:0')
        write_json(Path(a.output)/'run_provenance.json',dict(commit=a.expected_commit,
            cuda_visible_devices=visible,device='cuda:0',gpu=torch.cuda.get_device_name(0),
            torch_version=str(torch.__version__),peak_memory_bytes=torch.cuda.max_memory_allocated(0),
            formal_epochs=0,source_updates=1,target_updates_per_method=1))
        print('ROUTE_A_SOURCE_TARGET_SMOKE_COMPLETE: PENDING_CODEX_REVIEW');return 0
    except Exception as exc:
        print(f'ROUTE_A_SMOKE_FAILED: {type(exc).__name__}: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
