"""B115 approved campaign init/run/select/finalize. No test or resume command."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import contextlib
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from baselines.b115_formal import (TASK_ID,MULTIPLIERS,sha,code_identity,check_checkout,
    read_campaign,job_id,load_route,verify_job,screen_selection,verify_selection,same_verified_job)
from baselines.b115_training import train_engine,configuration,save_json
from baselines.features import avalon_matrix


def utc(): return datetime.now(timezone.utc).isoformat()


def gpu_snapshot():
    try:
        p=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
        return dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:
        return dict(error=str(e))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','run','select','finalize','package'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--commit',required=True)
    for name in ('audit-path','allowlist-A','allowlist-B','csv-path','split-path','tox-manifest','datastore'):
        parser.add_argument('--'+name)
    parser.add_argument('--phase',choices=['screen','replicate'])
    parser.add_argument('--route',choices=['A','B'])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--trial',type=int)
    args=parser.parse_args()
    input_names=('audit_path','allowlist_A','allowlist_B','csv_path','split_path','tox_manifest','datastore')
    if args.action!='init' and any(getattr(args,k) is not None for k in input_names):
        parser.error('input overrides forbidden after init')
    if args.action!='run' and any(getattr(args,k) is not None for k in ('phase','route','seed','trial')):
        parser.error('job arguments only allowed with run')
    check_checkout(args.commit)
    root=Path(args.root).resolve()
    if args.action=='init':
        names=('audit_path','allowlist_A','allowlist_B','csv_path','split_path','tox_manifest','datastore')
        if any(getattr(args,k) is None for k in names): parser.error('init requires all frozen input paths')
        inputs={k:str(Path(getattr(args,k)).resolve()) for k in names}
        campaign=dict(task_id=TASK_ID,commit=args.commit,code=code_identity(),root=str(root),
            allowed_splits=['train','validation'],max_runs=14,max_model_epochs=1680,inputs=inputs,
            input_sha256={k:sha(v) for k,v in inputs.items() if k!='datastore'},created_utc=utc())
        # Validate both routes before reserving the campaign root.
        for route in ('A','B'): load_route(campaign,route)
        root.mkdir(parents=True,exist_ok=False)
        save_json(root/'campaign.json',campaign)
        print('INIT PASS: 6 screens then 8 selected replications; no test')
        return
    campaign=read_campaign(root,args.commit)
    if args.action=='run':
        job=job_id(args.phase,args.route,args.seed,args.trial)
        if args.phase=='replicate':
            selected=verify_selection(root,campaign)
            if args.trial!=selected['selected'][args.route]: raise ValueError('trial not selected')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('expose exactly one idle CUDA GPU with CUDA_VISIBLE_DEVICES')
        folder=root/job;folder.mkdir(exist_ok=False)
        record=dict(task_id=TASK_ID,job_id=job,phase=args.phase,route=args.route,seed=args.seed,
            trial=args.trial,argv=sys.argv,utc_start=utc(),campaign_sha256=sha(root/'campaign.json'),
            python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
            numpy=np.__version__,gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_before=gpu_snapshot())
        save_json(folder/'started.json',record)
        rc=1
        try:
            train,val,adj,features,scaler,contract=load_route(campaign,args.route)
            np.savez_compressed(folder/'validation_truth.npz',sample_ids=np.asarray(val.sample_ids),
                                tasks=np.asarray(val.tasks),truth=val.labels)
            with (folder/'training_stdout.log').open('x') as stdout, (folder/'training_stderr.log').open('x') as stderr:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    train_engine(train_x=avalon_matrix(train.smiles),train_labels=train.labels,
                        validation_x=avalon_matrix(val.smiles),validation_labels=val.labels,
                        adjacency=adj,endpoint_features=features,scaler=scaler,contract=contract,
                        output=folder/'training',config=configuration(seed=args.seed,lr_multiplier=MULTIPLIERS[args.trial]),device='cuda:0')
            rc=0
        except Exception:
            with (folder/'exception.txt').open('x') as stream: traceback.print_exc(file=stream)
            traceback.print_exc()
            raise
        finally:
            record.update(exit_code=rc,utc_end=utc(),gpu_after=gpu_snapshot())
            save_json(folder/'execution.json',record)
        result=verify_job(root,campaign,args.phase,args.route,args.seed,args.trial)
        save_json(folder/'verification.json',result)
        print(job,'CONTENT PASS; pending external acceptance')
    elif args.action=='select':
        result=screen_selection(root,campaign)
        save_json(root/'selection.json',result)
        print('SELECTION',result['selected'])
    elif args.action=='finalize':
        selected=verify_selection(root,campaign)
        runs=list(selected['screen'])
        for route in ('A','B'):
            loaded=load_route(campaign,route)
            for seed in range(43,47):
                runs.append(verify_job(root,campaign,'replicate',route,seed,selected['selected'][route],loaded))
        actual={p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(('screen_','replicate_'))}
        if actual!={r['job_id'] for r in runs} or len(runs)!=14: raise ValueError('campaign job matrix differs')
        if sum(r['updates'] for r in runs)!=2912280: raise ValueError('campaign cost differs')
        save_json(root/'final_verification.json',dict(task_id=TASK_ID,runs=runs,
            selected=selected['selected'],model_epochs=1680,updates=2912280,
            test_executed=False,content_status='PASS',acceptance_status='PENDING_REVIEW'))
        print('14 RUNS VERIFIED; test remains forbidden')
    else:
        from scripts.verify_b115_smoke import read_json
        final=read_json(root/'final_verification.json')
        selected=verify_selection(root,campaign)
        if final['selected']!=selected['selected'] or len(final['runs'])!=14:
            raise ValueError('final matrix changed')
        expected={r['job_id'] for r in selected['screen']}|{job_id('replicate',r,s,selected['selected'][r]) for r in ('A','B') for s in range(43,47)}
        if {r['job_id'] for r in final['runs']}!=expected:raise ValueError('wrong final jobs')
        files=[root/n for n in ('campaign.json','selection.json','final_verification.json')]
        loaded_by_route={r:load_route(campaign,r) for r in ('A','B')}
        for row in final['runs']:
            phase='screen' if row['seed']==42 else 'replicate'
            if not same_verified_job(row,verify_job(root,campaign,phase,row['route'],row['seed'],row['trial'],loaded_by_route[row['route']])):
                raise ValueError('content changed after final verification')
            folder=root/row['job_id'];train=folder/'training'
            if sha(train/'run.json')!=row['run_sha256']:raise ValueError('run changed after finalize')
            files.extend(p for p in folder.iterdir() if p.is_file())
            files.extend(p for p in train.iterdir() if p.is_file() and p.suffix in ('.json','.npy'))
            best=train/f"epoch_{row['best_epoch']:03d}.pt"
            if sha(best)!=row['best_checkpoint_sha256']:raise ValueError('best changed after finalize')
            files.append(best)
        # All 1680 checkpoints stay on server; 14 bests + full validation history travel.
        hashes={p.relative_to(root).as_posix():sha(p) for p in sorted(set(files))}
        output=root/'B115_formal_results.zip'
        with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:archive.write(root/name,name)
            archive.writestr('checksums.sha256',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP CRC failure')
            import hashlib
            for n,h in hashes.items():
                if hashlib.sha256(archive.read(n)).hexdigest()!=h:raise ValueError('ZIP member mismatch')
        with output.with_suffix('.zip.sha256').open('x') as stream:stream.write(f'{sha(output)}  {output.name}\n')
        print('PACKAGE VERIFIED; preserve all server originals; add WSL/server gate logs to outer review archive')


if __name__=='__main__': main()
