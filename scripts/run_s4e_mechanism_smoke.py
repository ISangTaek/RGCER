"""Run/verify the fixed S4E4 four-run smoke; no formal-run option exists."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from s4e_mechanism_design import require, read_json
from s4e_mechanism_smoke import smoke, verify_smoke


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','verify'))
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--design-sha',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--datastore',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--expected-commit',required=True)
    args=p.parse_args()
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    require(head==args.expected_commit and len(head)==40,'implementation commit differs')
    dirty=subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,check=True,capture_output=True,text=True).stdout
    require(not dirty,'worktree dirty; preserve changes and report')
    if args.action=='run':
        require(args.datastore is not None,'--datastore is required for run')
        smoke(ROOT,args.design,args.design_sha,args.datastore,args.output,args.device)
    require(read_json(args.output/'scope.json')['commit']==head,'output inference commit differs')
    print(json.dumps(verify_smoke(args.design,args.design_sha,args.output),indent=2))


if __name__=='__main__':main()
