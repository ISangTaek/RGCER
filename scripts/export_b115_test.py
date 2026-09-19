"""Accepted B115 ten-best validation replay and test export; CPU, no training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from datetime import datetime,timezone
import platform
import torch
from baselines.b115_formal import check_checkout
from baselines.b115_training import save_json
from baselines.b115_test_export import run,package,POLICY_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','test','verify'))
    for key in ('root','output','commit','csv-path','split-path','tox-manifest'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    check_checkout(args.commit)
    torch.set_num_threads(1)
    start=datetime.now(timezone.utc).isoformat()
    inputs={k:getattr(args,k) for k in ('csv_path','split_path','tox_manifest')}
    result=run(args.action,args.root,args.output,inputs)
    if args.action!='verify':
        save_json(Path(args.output)/'execution.json',dict(action=args.action,argv=sys.argv,
            utc_start=start,utc_end=datetime.now(timezone.utc).isoformat(),exit_code=0,
            inference_commit=args.commit,policy_sha256=POLICY_SHA,python=platform.python_version(),
            torch=str(torch.__version__),device='cpu',threads=1))
    if args.action=='test':
        run('verify',args.root,args.output,inputs)
        print('ARCHIVE',package(args.output))
    print(json.dumps(dict(action=args.action,exit_code=0,test_executed=result.get('test_executed'),
        acceptance_status='PENDING_REVIEW')))


if __name__=='__main__':main()
