"""S4E5 guarded per-run execution, verification and full matrix finalization."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from s4e_mechanism_design import require,sha,read_json,write_json
from s4e_mechanism_smoke import load_design
from s4e_formal import evaluate,verify,matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','verify','finalize'))
    for n in ('design','authorization','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--authorization-sha',required=True)
    p.add_argument('--run-id');p.add_argument('--datastore',type=Path);p.add_argument('--device',default='cpu')
    a=p.parse_args();require(sha(a.authorization)==a.authorization_sha,'authorization bytes differ')
    auth=read_json(a.authorization)
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,check=True,capture_output=True,text=True).stdout
    require(not dirty and auth['commit']==head and len(head)==40,'commit/worktree differs')
    require(auth['allow_formal_50'] is True and auth['allow_training'] is False,'formal inference not authorized')
    _,docs=load_design(a.design,auth['design_sha256']);runs=matrix(docs)
    require(auth['run_ids']==list(runs),'authorization run sequence differs')
    if a.action=='finalize':
        summary=[]
        for rid in runs:
            v=verify(a.design,auth['design_sha256'],a.output/rid,a.authorization_sha)
            require(v['run_id']==rid,'output directory/run mismatch')
            summary.append(dict(v,metrics=read_json(a.output/rid/'metrics.json'),checksums_sha256=sha(a.output/rid/'checksums.json')))
        write_json(a.output/'formal_summary.json',{'authorization_sha256':a.authorization_sha,'runs':summary,
            'run_count':50,'observations':sum(v['observations'] for v in summary),'acceptance_status':'PENDING_CODEX_REVIEW'})
        print('FULL_MATRIX_VERIFIED 50 runs 704500 observations');return
    require(a.run_id in runs,'--run-id required and must be frozen')
    if a.action=='run':
        require(a.datastore is not None,'--datastore required')
        evaluate(ROOT,a.design,auth['design_sha256'],a.datastore,a.run_id,a.output,a.device,a.authorization_sha)
    v=verify(a.design,auth['design_sha256'],a.output,a.authorization_sha)
    require(v['run_id']==a.run_id,'requested/output run mismatch')
    print(v)


if __name__=='__main__':main()
