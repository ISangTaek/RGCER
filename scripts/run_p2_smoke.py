"""Shared single-GPU P2 loop; smoke by default, formal via explicit entry point."""
import argparse
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_contract import load_contract, require
from p2_execution import load_release, write_new, audit_validation, selected_runs


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def main(stage='SMOKE'):
    p=argparse.ArgumentParser()
    for name in ('authorization','authorization-sha','approval','protocol','members','datastore','output'):
        p.add_argument('--'+name,required=True)
    if stage=='FORMAL':
        p.add_argument('--panel',choices=['mouse','rat'],required=True)
        p.add_argument('--route',choices=['oral','intraperitoneal'],required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    release=load_release(a.authorization,a.authorization_sha,a.approval,stage)
    runs=selected_runs(release,getattr(a,'panel',None),getattr(a,'route',None))
    budget=sum(r['updates'] for r in runs)
    require(git(repo,'rev-parse','HEAD')==release['commit'],'checkout commit')
    require(not git(repo,'status','--porcelain','--untracked-files=all'),'worktree not clean')
    _,members=load_contract(a.protocol,a.members)
    require(sys.platform=='linux','server Linux required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2','3'),'select one physical GPU 0-3')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','deterministic CUDA workspace')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one visible CUDA GPU required')
    require('A6000' in torch.cuda.get_device_name(0),'RTX A6000 required')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from p2_data import P2Data
    from p2_model import build_model
    from p2_engine import P2Engine, state_digest
    data=P2Data(members,a.datastore)
    output=Path(a.output).resolve()
    require(not output.exists(),'output exists; no retry/overwrite')
    source_initial={};target_initial={};sources={};completed=[];attempted_updates=0
    try:
        output.mkdir(parents=False,exist_ok=False)
        write_new(output/'release.json',release)
        write_new(output/'environment.json',dict(commit=release['commit'],python=sys.version,torch=str(torch.__version__),
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
            deterministic=True,tf32=False,scope=stage))
        for run in runs:
            seed=run['seed']
            folder=output/run['run_id'];folder.mkdir()
            is_source=run['role']=='source';source_sha=None
            if is_source:
                view=data.source(run['panel'],run['route']);model=build_model('source',seed)
                initial=state_digest(model.state_dict())
                source_initial.setdefault(seed,initial)
                require(initial==source_initial[seed],'source initial pairing')
            else:
                view=data.target();source=sources[run['source_run_id']]
                source_sha=source['sha']
                require(hashlib.sha256(source['path'].read_bytes()).hexdigest()==source_sha,'source bytes changed')
                payload=torch.load(source['path'],map_location='cpu',weights_only=True)
                require(payload['identity']['run']['run_id']==run['source_run_id'] and payload['total_updates']==(1 if stage=='SMOKE' else 440),'source identity')
                require(state_digest(payload['model_state'])==payload['model_digest'],'smoke source tensor identity')
                encoder={k.removeprefix('encoder.'):v for k,v in payload['model_state'].items() if k.startswith('encoder.')}
                model=build_model('target',seed,encoder)
                initial=state_digest(model.state_dict())
                target_initial.setdefault(seed,initial)
                require(initial==target_initial[seed],'target full initial pairing')
            identity=dict(**run,release_sha=a.authorization_sha,commit=release['commit'],source_checkpoint_sha=source_sha)
            method='SOURCE' if is_source else run['method']
            def fresh():
                m=build_model('source',seed) if is_source else build_model('target',seed,encoder)
                return P2Engine(m,view,method=method,seed=seed,run_identity=identity,device='cuda:0')
            engine=P2Engine(model,view,method=method,seed=seed,run_identity=identity,device='cuda:0')
            require(engine.per_epoch==(11 if is_source else 6),'actual updates per epoch')
            write_new(folder/'scalers.json',view.scalers)
            write_new(folder/'initial.json',dict(model_sha=initial,view_sha=view.identity,source_checkpoint_sha=source_sha))
            resume_proof=None
            while engine.total_updates<run['updates']:
                attempted_updates+=1
                require(attempted_updates<=budget,'update budget')
                record=engine.step()
                if record is not None:
                    write_new(folder/f"epoch_{record['epoch']:02d}.json",record)
                    if stage=='FORMAL':
                        engine.save(folder/'last.pending.pt')
                        os.replace(folder/'last.pending.pt',folder/'last.pt')
                    print(run['run_id'], 'epoch', record['epoch'], 'updates',engine.total_updates,flush=True)
                # Exercise a real CUDA restore directly before HF unfreezes.
                # No replayed optimizer step: this consumes no extra budget.
                if stage=='SMOKE' and method=='HF_low' and engine.total_updates==30:
                    path=folder/'before_unfreeze.pt';sha=engine.save(path)
                    before=state_digest(engine.model.state_dict())
                    del engine;del model
                    torch.cuda.empty_cache()
                    engine=fresh();engine.restore(path,sha)
                    require(state_digest(engine.model.state_dict())==before,'CUDA restored model mismatch')
                    resume_proof=dict(checkpoint_sha=sha,updates=30,epoch=5,offset=0,model_sha=before,status='RESTORED_NO_REPLAY')
            engine.check_cursor_and_optimizer()
            validation=None if is_source else audit_validation(engine.history,view,engine.best_epoch)
            final=folder/'final.pt';sha=engine.save(final)
            final_model_sha=state_digest(engine.model.state_dict())
            if is_source:sources[run['run_id']]=dict(path=final,sha=sha)
            summary=dict(run=run,updates=engine.total_updates,epochs=engine.epoch,offset=engine.offset,
                final_sha=sha,final_model_sha=final_model_sha,best_epoch=engine.best_epoch,
                validation=validation,resume=resume_proof,initial_sha=initial,source_checkpoint_sha=source_sha,
                gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(0))
            write_new(folder/'summary.json',summary);completed.append(summary)
            del engine
            if stage=='FORMAL' or method!='HF_low':del model
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        require(sum(r['updates'] for r in completed)==attempted_updates==budget,'final budget')
        write_new(output/'budget.json',dict(attempted_updates=attempted_updates,completed_updates=budget,counts={r['run']['run_id']:r['updates'] for r in completed}))
        write_new(output/'summary.json',dict(execution_status='FINISHED',validation_status='PASS',acceptance_status='PENDING_REVIEW',
            delivery_kind=stage,runs=completed,updates=budget,test_access=False,calibration_access=False))
        files={str(f.relative_to(output)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
               for f in output.rglob('*') if f.is_file()}
        write_new(output/'checksums.json',files)
        print(stage+'_COMPLETE_PENDING_CODEX_REVIEW',budget,flush=True)
    except BaseException:
        if output.is_dir() and not (output/'failure.json').exists():
            write_new(output/'failure.json',dict(execution_status='FAILED',acceptance_status='DECISION_REQUIRED',
                completed_runs=len(completed),attempted_updates=attempted_updates,error=traceback.format_exc()))
        raise
    finally:data.close()


if __name__=='__main__':main()
