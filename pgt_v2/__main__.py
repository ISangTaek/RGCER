"""Minimal PGT V2 experiment interfaces. Run `python -m pgt_v2 --help`."""
import argparse
from pathlib import Path

from .artifacts import read_json, write_json, package_files


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=1,help='Fixed PyTorch CPU thread count')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('inventory','prepare'):
        p=sub.add_parser(name);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('environment');p.add_argument('--output',required=True)
    p=sub.add_parser('reference');p.add_argument('--toxacol-root',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('smoke')
    p.add_argument('--view',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--chemprop-root',required=True);p.add_argument('--toxacol-root',required=True)
    p=sub.add_parser('train')
    p.add_argument('--run',required=True);p.add_argument('--view',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--chemprop-root')
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('resolve')
    p.add_argument('--row',required=True);p.add_argument('--view',required=True);p.add_argument('--selection');p.add_argument('--output',required=True)
    p=sub.add_parser('select');p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('predict')
    p.add_argument('--checkpoint',required=True);p.add_argument('--view',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--role',choices=('train','validation','test'),default='test')
    p.add_argument('--device',default='cpu');p.add_argument('--chemprop-root')
    p=sub.add_parser('score')
    p.add_argument('--predictions',required=True);p.add_argument('--view',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected-checkpoint-sha256',required=True)
    p=sub.add_parser('explain')
    p.add_argument('--checkpoint',required=True);p.add_argument('--view',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--record-id',required=True);p.add_argument('--task',required=True)
    p.add_argument('--other-task');p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    p=sub.add_parser('package');p.add_argument('--files',required=True,help='JSON map of archive names to local files');p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    if args.command in ('inventory','prepare'):
        from .preparation import inventory,prepare
        result=(inventory if args.command=='inventory' else prepare)(args.source,args.output)
    elif args.command=='environment':
        from .verification import environment
        result=environment();write_json(args.output,result)
    elif args.command=='reference':
        from .verification import native_toxacol_parity
        result=native_toxacol_parity(args.toxacol_root);write_json(args.output,result)
    elif args.command=='smoke':
        from .verification import forward_checks
        result=forward_checks(args.view,args.cache,args.output,device=args.device,chemprop_root=args.chemprop_root,toxacol_root=args.toxacol_root)
    elif args.command=='train':
        from .training import run_training
        result=run_training(args.run,args.view,args.cache,args.output,device=args.device,chemprop_root=args.chemprop_root,resume=args.resume)
    elif args.command=='resolve':
        from .artifacts import load_manifest
        from .config import resolve_run
        result=resolve_run(read_json(args.row),view_manifest=load_manifest(args.view),selection=read_json(args.selection) if args.selection else None)
        write_json(args.output,result)
    elif args.command=='select':
        from .evaluation import select_config
        result=select_config(args.runs,args.output)
    elif args.command=='predict':
        from .evaluation import sealed_predict
        result=sealed_predict(args.checkpoint,args.view,args.cache,args.output,role=args.role,device=args.device,chemprop_root=args.chemprop_root)
    elif args.command=='score':
        from .evaluation import score_prediction
        result=score_prediction(args.predictions,args.view,args.output,expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    elif args.command=='explain':
        from .explain import explain_record
        result=explain_record(args.checkpoint,args.view,args.cache,args.reference,args.record_id,args.task,args.output,
                              other_task=args.other_task,device=args.device)
    else:
        result=package_files(read_json(args.files),args.output)
    print({'command':args.command,'status':'completed','output':args.output},flush=True)
    return result


if __name__=='__main__':
    main()
