"""S5C approved selected-best validation regression then fifteen formal tests."""
import argparse,os,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('csv','split-manifest','tox-manifest','selection-lock','expected-commit','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    try:
        import torch
        from dataset115_training import require
        from dataset115_adapter import Dataset115Table
        from dataset115_route_b_run import SERVER_TOX_SHA
        from dataset115_test_export import load_lock,export_all,write_json
        repo=Path(__file__).resolve().parents[1]
        def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        require(len(a.expected_commit)==40 and git('rev-parse','HEAD')==a.expected_commit and not git('status','--porcelain'),'code identity/worktree')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2','3') and torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU required')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS config')
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        lock=load_lock(a.selection_lock)
        table=Dataset115Table.load(a.csv,a.split_manifest,a.tox_manifest,expected_tox_sha=SERVER_TOX_SHA)
        summary=export_all(lock,table,a.output,device='cuda:0')
        write_json(Path(a.output)/'inference_provenance.json',dict(commit=a.expected_commit,device='cuda:0',
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
            selection_lock=str(Path(a.selection_lock).resolve()),training_commit=lock['training_commit']))
        print(json.dumps(dict(runs=len(summary['runs']),status='PENDING_CODEX_REVIEW')))
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
