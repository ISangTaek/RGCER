"""V9-C2B: six bounded SSRA/LoRA code-and-cost jobs; no formal selection."""
from collections import Counter
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import v9_cost_probe as c0
from v9_srgt import TrainSupport, training_records
from v9_ssra import AdaptedGraph, METHODS, SPEC, optimizer_for, save_smoke, load_smoke
import torch
import v9_c2a_source_stats as c2a

REPO=Path(__file__).resolve().parent
LOCK=REPO/'configs/v9_c2b_lock.json'
TASK='V9_C2B_SSRA_LORA_SMOKE_20260930'
REGISTRY='.tmp/v9_c2b_attempts_20260930'
TESTS=('tests/test_v9_ssra.py','tests/test_v9_c2b_probe.py','tests/test_v9_source_portability.py','tests/test_p1d4_identity.py')
TEST_COUNT=67
WALL_SECONDS=1800
read,write,sha,digest,require=c0.read,c0.write,c0.sha,c0.digest,c0.require


def jobs():
    return [dict(id=f'{s}_{m}_s42',setting=s,method=m,seed=42) for s in c0.SETTINGS for m in METHODS]


def checked_job(job):
    require(type(job) is dict and any(digest(job)==digest(j) for j in jobs()),'job outside C2B matrix')


def summarize(rows,tasks,counts,batch_size):
    value=c0.summarize(rows,tasks,counts,batch_size)
    value['includes']=['measured_candidate_forward_backward_update','train_batch_loading_and_transfer']
    value['excludes']=['validation','checkpoint_io','restore_replay','asset_preparation',
                       'support_preparation','stats_verification','adapter_preparation','source_training','independent_confirmation']
    return value


def prior_gate(root):
    expected=read(LOCK)['accepted_074_files']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual in (set(expected),set(expected)-{'checksums.sha256'}),'074 file population')
    require(all(not p.is_symlink() for p in root.rglob('*')),'074 symlink')
    require(all(sha(root/n)==expected[n] for n in actual),'074 immutable bytes')
    require(sha(REPO/'.tmp/v9_c2a_attempts_20260929/attempt.json')==read(LOCK)['accepted_074_claim_sha256'],'074 original claim')
    return dict(source='074_ACCEPTED_SOURCE_STATS_ONLY',canonical_sha256=read(LOCK)['accepted_074_zip_sha256'])


def load_stats(root,setting,encoder):
    from reproducibility import state_dict_sha256
    require(setting in c0.SETTINGS,'statistics scene')
    source='Nonhuman104' if setting=='A' else 'Animal56'
    folder=root/source
    for name in ('moments.pt','selection.json','receipt.json'):
        require(sha(folder/name)==read(LOCK)['accepted_074_files'][source+'/'+name],'074 source bytes')
    payload=torch.load(folder/'moments.pt',map_location='cpu',weights_only=True)
    identity=read(folder/'selection.json')
    require(identity['source']==source and identity['encoder_sha256']==read(LOCK)['initial_encoders'][setting],'074 source binding')
    require(state_dict_sha256(encoder)==identity['encoder_sha256'],'074 actual encoder binding')
    c2a.verify_payload(payload,identity,{k:m.in_features for k,m in c2a.targets(encoder).items()})
    return payload


def metadata_preflight(commit,split_manifest,source_lock,output=None):
    """CPU-only identity check over all allowed TRAIN observations, no forward/update."""
    from p1d4_runtime import factory_for
    from reproducibility import state_dict_sha256
    tick=time.perf_counter();settings={};lock=read(LOCK)
    for setting in c0.SETTINGS:
        print('C2B CPU train-identity scene: '+setting,flush=True)
        factory=factory_for(REPO,setting=setting,seed=42,split_manifest=split_manifest,source_lock=source_lock,device='cpu')
        trainer=make_trainer(factory,setting,'cpu')
        tasks,counts=c0.tasks_and_counts(setting)
        datasets=factory.datasets['train'] if setting=='ToxAcute' else trainer.datasets['train']
        require({t:len(datasets[t]) for t in tasks}==counts,'preflight train counts')
        require(state_dict_sha256(trainer.model.encoder)==lock['initial_encoders'][setting],'preflight source identity')
        bank=TrainSupport(training_records(factory,trainer,setting))
        for task,ds in datasets.items():
            for i in range(len(ds)):
                graph=ds[i];sid=str(ds.get_sample_id(i))
                require(graph.sample_id==sid,'preflight train graph sample identity')
                bank.for_train_batch(task,[sid],[graph.canonical_smiles])
        settings[setting]=dict(counts=counts,support=bank.identity,
            initial_encoder=lock['initial_encoders'][setting],graph_observations_checked=sum(counts.values()),
            source_verification_mode='LOCKED_CPU_CONTENT' if setting=='A' else 'LOCKED_INITIALIZATION')
        if output is not None:
            write(output/(setting+'_metadata.json'),dict(task=TASK,commit=commit,setting=setting,
                  optimizer_updates=0,model_forward_calls=0,identity=settings[setting]))
    return dict(task=TASK,commit=commit,scope='TRAIN_IDENTITY_ONLY',optimizer_updates=0,
                model_forward_calls=0,holdout_predictions_accessed=False,settings=settings,
                elapsed_seconds=time.perf_counter()-tick)


def make_trainer(factory,setting,device):
    """CPU source content audit is explicit; GPU retains historical exact replay."""
    if setting=='ToxAcute':return factory.make_trainer()
    if setting=='A' and device=='cpu':
        return factory.make_trainer(original=True,device=device,source_content_only=True)
    return factory.make_trainer(original=True,device=device)


def gate(root,commit,role):
    require(role in ('wsl','server'),'gate role')
    expected=dict(task=TASK,role=role,commit=commit,tests=list(TESTS),junit_sha256=sha(root/'tests.xml'),real_data_optimizer_updates=0)
    require(read(root/'gate.json')==expected,'C2B gate identity')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases)==TEST_COUNT and all(not any(c.findall(t) for t in ('error','failure','skipped')) for c in cases),'gate count/failure/error/skip')
    require({c.attrib['classname'] for c in cases}=={'tests.'+Path(t).stem for t in TESTS},'gate files')
    return expected


def code_gate(root,commit,role):
    require(role in ('wsl','server'),'gate role')
    c0.check_code(REPO,commit)
    root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as log:
        value=subprocess.run(argv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                             env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json',dict(argv=argv,exit_code=value.returncode))
    require(value.returncode==0,'C2B code tests')
    write(root/'gate.json',dict(task=TASK,role=role,commit=commit,tests=list(TESTS),
                              junit_sha256=sha(root/'tests.xml'),real_data_optimizer_updates=0))
    gate(root,commit,role)


def claim(root,commit):
    folder=REPO/REGISTRY;folder.mkdir(parents=True,exist_ok=True)
    write(folder/'attempt.json',dict(task=TASK,commit=commit,root=str(root.resolve()),max_updates=72,
        max_wall_seconds=WALL_SECONDS,prior_updates=144,total_cost_updates_after_success=216))


def one_job(factory,trainer,job,out,commit,device,stats_root):
    """Uses real factory identities in production; synthetic CPU fixtures in tests."""
    import torch
    from reproducibility import state_dict_sha256
    from loss import QuantileRegressionLoss
    checked_job(job)
    tasks,counts=c0.tasks_and_counts(job['setting'])
    datasets=factory.datasets['train'] if job['setting']=='ToxAcute' else trainer.datasets['train']
    require({t:len(datasets[t]) for t in tasks}==counts,'train counts')
    base=trainer.model
    source_digest=state_dict_sha256(base.encoder)
    require(source_digest==read(LOCK)['initial_encoders'][job['setting']],'070 same source tensor identity')
    t0=time.perf_counter()
    bank=TrainSupport(training_records(factory,trainer,job['setting']))
    support_seconds=time.perf_counter()-t0
    write(out/'support_records.json',bank.records)
    write(out/'support_identity.json',bank.identity)
    t0=time.perf_counter()
    payload=load_stats(stats_root,job['setting'],base.encoder)
    stats_seconds=time.perf_counter()-t0;t0=time.perf_counter()
    model=AdaptedGraph(base,job['method'],counts,payload).to(device)
    model.train();opt=optimizer_for(model)
    if device=='cuda:0':torch.cuda.synchronize()
    adapter_seconds=time.perf_counter()-t0
    initial_target=state_dict_sha256(model.target)
    initial_heads=state_dict_sha256(model.decoders)
    plan=c0.schedule(tasks,counts,c0.BATCH_SIZES[job['setting']])
    identity=dict(schema='v9_c2b_model_identity_v1',method=job['method'],seed=42,spec=SPEC,
        tasks=tasks,counts=counts,contract_sha256=digest(c0.contract(job['setting'])),
        support=bank.identity,initial_encoder=source_digest,initial_heads=initial_heads,source_stats_sha256=sha(stats_root/('Nonhuman104' if job['setting']=='A' else 'Animal56')/'moments.pt'),
        task_updates=dict(Counter(r['task'] for r in plan)))
    write(out/'model_identity.json',identity)
    trainer.model=model
    def sync():
        if device=='cuda:0':torch.cuda.synchronize()
    sync()
    if device=='cuda:0':torch.cuda.reset_peak_memory_stats()
    rows=[]; diagnostics=[]; last_batch=None; last_task=None
    for item in plan:
        task,indices=item['task'],item['indices']
        expected_ids=[str(datasets[task].get_sample_id(i)) for i in indices]
        tick=time.perf_counter()
        if job['setting']=='ToxAcute':
            batch=factory.collator([datasets[task][i] for i in indices]).to(device)
        else:batch=trainer._batch('train',task,indices)
        checked_support=bank.for_train_batch(task,batch.sample_id,batch.canonical_smiles)
        # TrainSupport is reused only as a strict train-member/canonical validator.
        sync();batch_seconds=time.perf_counter()-tick
        require(not batch.is_empty and batch.y.numel()==len(indices) and list(batch.sample_id)==expected_ids,'train batch identity')
        write(out/f'update_{item["step"]:02d}.intent.json',dict(step=item['step'],max_total=item['step']+1))
        tick=time.perf_counter();opt.zero_grad(set_to_none=True)
        if job['setting']=='ToxAcute':
            prediction_loss,bundle=trainer._training_step(batch,task,0)
            aux=bundle['diagnostics']
        else:
            prediction,aux=model(batch,task_name=task,return_aux=True);scaler=trainer.scalers[task]
            prediction_loss=QuantileRegressionLoss().compute_loss(prediction[task],(batch.y.reshape(-1,1)-scaler['mean'])/scaler['std'])
        penalty=model.regularization(task);loss=prediction_loss+penalty
        require(torch.isfinite(loss).item(),'finite objective')
        loss.backward()
        require(not model.encoder.training and all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()),'source gradient/eval scope')
        target_gradient=sum(float(p.grad.detach().abs().sum()) for p in model.target.parameters() if p.grad is not None)
        if item['step']>0:require(target_gradient>0,'target branch did not learn after zero-init release')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step();sync();update_seconds=time.perf_counter()-tick
        row=dict(item,n=len(indices),sample_ids_sha256=digest(expected_ids),max_nodes=int(batch.x.shape[1]),
            real_nodes=int(batch.node_mask.sum()),batch_seconds=batch_seconds,update_seconds=update_seconds,loss=float(loss.detach()))
        diag=dict(step=item['step'],target_gradient_l1=target_gradient,prediction_loss=float(prediction_loss.detach()),
            penalty=float(penalty.detach()),gate_min=float(aux['gates'].detach().min()),gate_max=float(aux['gates'].detach().max()))
        write(out/f'step_{item["step"]:02d}.json',row);write(out/f'diagnostic_{item["step"]:02d}.json',diag)
        rows.append(row);diagnostics.append(diag);last_batch=batch;last_task=task
    require(all(torch.isfinite(v).all().item() for v in model.state_dict().values()),'finite final model')
    require(state_dict_sha256(model.encoder)==source_digest,'source drift')
    require(state_dict_sha256(model.target)!=initial_target and state_dict_sha256(model.decoders)!=initial_heads,'target/head did not update')
    peak_allocated=torch.cuda.max_memory_allocated() if device=='cuda:0' else 0
    peak_reserved=torch.cuda.max_memory_reserved() if device=='cuda:0' else 0
    tick=time.perf_counter()
    checkpoint=out/'smoke_state.pt';save_smoke(checkpoint,model,opt,identity,12)
    save_seconds=time.perf_counter()-tick
    # One deterministic replay on the last TRAIN batch only. No new updates.
    tick=time.perf_counter()
    restored=AdaptedGraph(base,job['method'],counts,payload).to(device);restored_opt=optimizer_for(restored)
    load_smoke(checkpoint,restored,restored_opt,identity,12)
    model.eval();restored.eval()
    with torch.no_grad():
        original=model(last_batch,task_name=last_task)[last_task]
        replay=restored(last_batch,task_name=last_task)[last_task]
    sync()
    require(torch.equal(original,replay),'restored TRAIN output differs')
    require(state_dict_sha256(model)==state_dict_sha256(restored),'restored state differs')
    restore_seconds=time.perf_counter()-tick
    return dict(schema='v9_c2b_job_v1',task=TASK,job=job,commit=commit,identity=identity,rows=rows,diagnostics=diagnostics,
        summary=summarize(rows,tasks,counts,c0.BATCH_SIZES[job['setting']]),support_seconds=support_seconds,
        stats_verification_seconds=stats_seconds,adapter_preparation_seconds=adapter_seconds,
        initial_encoder=source_digest,final_encoder=state_dict_sha256(model.encoder),initial_target=initial_target,
        final_target=state_dict_sha256(model.target),initial_heads=initial_heads,final_heads=state_dict_sha256(model.decoders),
        model_sha256=state_dict_sha256(model),checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_save_seconds=save_seconds,restore_and_train_replay_seconds=restore_seconds,
        restore_updates=0,train_replay_identical=True,total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
        environment=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
                         device_name=torch.cuda.get_device_name(0) if device=='cuda:0' else 'synthetic CPU fixture'),
        scope='CANDIDATE_CODE_AND_COST_ONLY',validation_evaluated=False,holdout_predictions_accessed=False,
        scientific_pass=False,acceptance='PENDING_REVIEW')


def worker(root,job,commit,split,source,stats_root):
    import torch
    from p1d4_runtime import factory_for
    checked_job(job);c0.check_code(REPO,commit)
    launch=read(root/'launch.json')
    require(launch['task']==TASK and launch['commit']==commit and digest(launch['jobs'])==digest(jobs()),'C2B launch')
    require(read(REPO/REGISTRY/'attempt.json')['root']==str(root.resolve()),'C2B attempt')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==launch['gpu_uuid'],'GPU binding')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one CUDA GPU')
    torch.use_deterministic_algorithms(True)
    out=root/job['id'];out.mkdir(exist_ok=False)
    write(out/'started.json',dict(job=job,commit=commit))
    tick=time.perf_counter()
    factory=factory_for(REPO,setting=job['setting'],seed=42,split_manifest=split,source_lock=source,device='cuda:0')
    trainer=make_trainer(factory,job['setting'],'cuda:0')
    asset_seconds=time.perf_counter()-tick
    result=one_job(factory,trainer,job,out,commit,'cuda:0',stats_root)
    result.update(asset_preparation_seconds=asset_seconds,gpu_uuid=launch['gpu_uuid'])
    write(out/'receipt.json',result)


def verify(root,commit,split_manifest,source_lock,stats_root):
    import torch
    from reproducibility import state_dict_sha256
    from p1d4_runtime import factory_for
    launch=read(root/'launch.json')
    require(launch['task']==TASK and launch['commit']==commit and digest(launch['jobs'])==digest(jobs()),'suite identity')
    metadata=read(root/'metadata_preflight.json')
    require(metadata['task']==TASK and metadata['commit']==commit and metadata['scope']=='TRAIN_IDENTITY_ONLY'
            and metadata['optimizer_updates']==0 and metadata['model_forward_calls']==0
            and metadata['holdout_predictions_accessed'] is False
            and set(metadata['settings'])==set(c0.SETTINGS),'metadata preflight scope')
    lock=read(LOCK);reports={};current_setting=None;trusted_trainer=None;trusted_records=None
    for job in jobs():
        out=root/job['id'];r=read(out/'receipt.json');tasks,counts=c0.tasks_and_counts(job['setting'])
        if current_setting!=job['setting']:
            factory=factory_for(REPO,setting=job['setting'],seed=42,split_manifest=split_manifest,source_lock=source_lock,device='cpu')
            trusted_trainer=make_trainer(factory,job['setting'],'cpu')
            trusted_records=training_records(factory,trusted_trainer,job['setting'])
            current_setting=job['setting']
        require(r['schema']=='v9_c2b_job_v1' and r['task']==TASK and digest(r['job'])==digest(job) and r['commit']==commit,'receipt identity')
        require(r['scope']=='CANDIDATE_CODE_AND_COST_ONLY' and r['scientific_pass'] is False and r['acceptance']=='PENDING_REVIEW'
                and r['validation_evaluated'] is False and r['holdout_predictions_accessed'] is False,'receipt scope')
        import math
        require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>=0
                    for k in ('stats_verification_seconds','adapter_preparation_seconds','support_seconds')),'preparation timing')
        require(r['initial_encoder']==r['final_encoder']==lock['initial_encoders'][job['setting']],'source binding')
        require(r['initial_target']!=r['final_target'] and r['initial_heads']!=r['final_heads'],'plasticity')
        bank=TrainSupport(read(out/'support_records.json'))
        trusted_bank=TrainSupport(trusted_records)
        require(digest(metadata['settings'][job['setting']])==digest(dict(counts=counts,support=trusted_bank.identity,
            initial_encoder=lock['initial_encoders'][job['setting']],graph_observations_checked=sum(counts.values()),
            source_verification_mode='LOCKED_CPU_CONTENT' if job['setting']=='A' else 'LOCKED_INITIALIZATION')),'metadata preflight identity')
        require(digest(bank.records)==digest(trusted_bank.records),'support members differ from actual target train')
        require(digest(bank.identity)==digest(read(out/'support_identity.json')),'support recomputation')
        require(Counter(row['task'] for row in bank.records)==Counter(counts),'support task populations')
        rows=[read(out/f'step_{i:02d}.json') for i in range(12)]
        trusted_datasets=factory.datasets['train'] if job['setting']=='ToxAcute' else trusted_trainer.datasets['train']
        for row in rows:
            expected_ids=[str(trusted_datasets[row['task']].get_sample_id(i)) for i in row['indices']]
            require(digest(expected_ids)==row['sample_ids_sha256'],'batch sample IDs differ from actual train')
        diags=[read(out/f'diagnostic_{i:02d}.json') for i in range(12)]
        require(sorted(p.name for p in out.glob('step_*.json'))==[f'step_{i:02d}.json' for i in range(12)],'step matrix')
        require(digest(rows)==digest(r['rows']) and digest(diags)==digest(r['diagnostics']),'step evidence')
        for i,d in enumerate(diags):
            require(read(out/f'update_{i:02d}.intent.json')==dict(step=i,max_total=i+1),'update watermarks')
            require(d['step']==i and 0<=d['gate_min']<=d['gate_max']<=1 and d['penalty']>=0,'gate diagnostics')
            require(d['target_gradient_l1']>=0 and (i==0 or d['target_gradient_l1']>0),'target gradients')
            if job['method']=='LORA':require(d['gate_min']==d['gate_max']==1 and d['penalty']==0,'parent control')
        expected=dict(schema='v9_c2b_model_identity_v1',method=job['method'],seed=42,spec=SPEC,tasks=tasks,counts=counts,
            contract_sha256=digest(c0.contract(job['setting'])),support=bank.identity,
            initial_encoder=lock['initial_encoders'][job['setting']],initial_heads=state_dict_sha256(trusted_trainer.model.decoders),
            source_stats_sha256=sha(stats_root/('Nonhuman104' if job['setting']=='A' else 'Animal56')/'moments.pt'),
            task_updates=dict(Counter(row['task'] for row in rows)))
        require(digest(expected)==digest(r['identity'])==digest(read(out/'model_identity.json')),'model identity')
        summary=summarize(rows,tasks,counts,c0.BATCH_SIZES[job['setting']])
        require(digest(summary)==digest(r['summary']),'cost recomputation')
        checkpoint=out/'smoke_state.pt'
        require(sha(checkpoint)==r['checkpoint_sha256'] and checkpoint.stat().st_size==r['checkpoint_bytes'],'checkpoint bytes')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(payload['updates']==12 and digest(payload['identity'])==digest(expected),'checkpoint identity')
        require(all(isinstance(v,torch.Tensor) and torch.isfinite(v).all().item() for v in payload['model_state'].values()),'checkpoint finite')
        # Recompute the protected encoder hash using the original hash algorithm.
        class State:
            def __init__(self,state):self.state=state
            def state_dict(self):return self.state
        state=payload['model_state']
        encoder={k[len('encoder.'):]:v for k,v in state.items() if k.startswith('encoder.')}
        require(state_dict_sha256(State(encoder))==expected['initial_encoder'],'saved source weights')
        require(state_dict_sha256(State(state))==r['model_sha256'],'saved model weights')
        # Independently validate all parameter shapes, source tensors and Adam
        # step counts against a fresh CPU model and the trusted source factory.
        payload_stats=load_stats(stats_root,job['setting'],trusted_trainer.model.encoder)
        fresh=AdaptedGraph(trusted_trainer.model,job['method'],counts,payload_stats)
        require(state_dict_sha256(fresh.target)==r['initial_target'] and
                state_dict_sha256(fresh.decoders)==r['initial_heads'],'initial trainable tensor identity')
        load_smoke(checkpoint,fresh,optimizer_for(fresh),expected,12)
        require(state_dict_sha256(fresh.target)==r['final_target'] and
                state_dict_sha256(fresh.decoders)==r['final_heads'],'final trainable tensor identity')
        require(r['total_parameters']==sum(p.numel() for p in fresh.parameters()) and
                r['trainable_parameters']==sum(p.numel() for p in fresh.parameters() if p.requires_grad),'parameter counts')
        require(r['train_replay_identical'] is True and type(r['restore_updates']) is int and r['restore_updates']==0,'restoration scope')
        require(type(r['peak_allocated_bytes']) is int and 0<r['peak_allocated_bytes']<=r['peak_reserved_bytes'],'GPU memory')
        reports[job['id']]=dict(receipt_sha256=sha(out/'receipt.json'),checkpoint_sha256=sha(checkpoint),summary=summary,
                               support_identity=bank.identity)
    return dict(task=TASK,commit=commit,content_status='PASS',scientific_acceptance='NOT_ASSESSED',
        optimizer_updates=72,completed_jobs=6,total_cost_updates=216,remaining_cost_updates=144,
        formal_training_authorized=False,results=reports)


def run(root,commit,gpu,split,source,wsl,server,stats_root):
    from p1d4_batch import free_gpus
    os.environ['CUDA_VISIBLE_DEVICES']=''  # CPU supervisor; workers receive the selected UUID.
    c0.check_code(REPO,commit);gate(wsl,commit,'wsl');gate(server,commit,'server')
    accepted=prior_gate(stats_root)
    require(free_gpus([gpu])==[gpu],'GPU occupied')
    root.mkdir(parents=True,exist_ok=False);claim(root,commit)
    uuid=c0.gpu_uuid(gpu)
    write(root/'launch.json',dict(task=TASK,commit=commit,jobs=jobs(),gpu=gpu,gpu_uuid=uuid,
                                max_optimizer_updates=72,max_wall_seconds=WALL_SECONDS,prior=accepted,stats_root=str(stats_root.resolve())))
    for role,path in (('wsl',wsl),('server',server)):
        shutil.copytree(path,root/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
    try:
        print('C2B CPU train-identity preflight: starting (0 model forwards / 0 updates)',flush=True)
        metadata=metadata_preflight(commit,split,source,root)
        write(root/'metadata_preflight.json',metadata)
        print('C2B CPU train-identity preflight: PASS; starting bounded GPU workers',flush=True)
        deadline=time.monotonic()+WALL_SECONDS
        for job in jobs():
            require(free_gpus([gpu])==[gpu],'GPU occupied before next job')
            remaining=deadline-time.monotonic();require(remaining>0,'C2B wall limit')
            argv=[sys.executable,str(Path(__file__).resolve()),'_worker','--output',str(root),'--commit',commit,
                  '--job',job['id'],'--split-manifest',str(split),'--source-lock',str(source),'--stats-root',str(stats_root)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/(job['id']+'.log')).open('xb') as log:
                result=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
            write(root/(job['id']+'.command.json'),dict(argv=argv,exit_code=result.returncode))
            require(result.returncode==0,'C2B job failed; stop without retry')
        training_wall_seconds=WALL_SECONDS-(deadline-time.monotonic())
        value=verify(root,commit,split,source,stats_root);value['training_wall_seconds']=training_wall_seconds
        write(root/'verification.json',value)
    except BaseException as exc:
        write(root/'failed.json',dict(task=TASK,error_type=type(exc).__name__,reason=str(exc),formal_training_authorized=False))
        raise


def package(root):
    import hashlib,zipfile
    require((root/'launch.json').is_file(),'C2B launch evidence required')
    archive=root.with_name(root.name+'.zip');sidecar=Path(str(archive)+'.sha256')
    require(not archive.exists() and not sidecar.exists(),'package exists')
    files={}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'symlink in evidence')
        if not path.is_file():continue
        require(path.suffix in ('.json','.log','.xml','.pt'),'unexpected file type')
        if path.suffix=='.pt':require(path.name=='smoke_state.pt','unexpected weights')
        files[path.relative_to(root).as_posix()]=sha(path)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(root/name,name)
        z.writestr('checksums.sha256',''.join(f'{v}  {k}\n' for k,v in files.items()))
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None and set(z.namelist())==set(files)|{'checksums.sha256'},'package members/CRC')
        require(all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in files.items()),'package member SHA')
    with sidecar.open('x',encoding='utf8') as f:f.write(f'{sha(archive)}  {archive.name}\n')
    return dict(archive=str(archive),sha256=sha(archive),scientific_acceptance='NOT_ASSESSED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['code-gate','run','verify','package','_worker'])
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=['wsl','server']);p.add_argument('--gpu',type=int,choices=range(4))
    p.add_argument('--wsl-evidence',type=Path);p.add_argument('--server-evidence',type=Path)
    p.add_argument('--stats-root',type=Path)
    p.add_argument('--split-manifest',type=Path);p.add_argument('--source-lock',type=Path)
    p.add_argument('--job',choices=[j['id'] for j in jobs()]);a=p.parse_args();root=a.output.resolve()
    if a.action=='code-gate':code_gate(root,a.commit,a.role)
    elif a.action=='verify':
        os.environ['CUDA_VISIBLE_DEVICES']=''
        require(a.split_manifest is not None and a.source_lock is not None and a.stats_root is not None,'read-only verification asset paths required')
        print(json.dumps(verify(root,a.commit,a.split_manifest.resolve(),a.source_lock.resolve(),a.stats_root.resolve()),ensure_ascii=False,indent=2))
    elif a.action=='package':
        require(read(root/'launch.json')['commit']==a.commit,'package commit')
        print(json.dumps(package(root),ensure_ascii=False,indent=2))
    else:
        require(a.split_manifest is not None and a.source_lock is not None and a.stats_root is not None,'asset paths required')
        if a.action=='run':
            require(a.gpu is not None and a.wsl_evidence is not None and a.server_evidence is not None and a.stats_root is not None,'C2B runtime inputs')
            run(root,a.commit,a.gpu,a.split_manifest.resolve(),a.source_lock.resolve(),a.wsl_evidence.resolve(),a.server_evidence.resolve(),a.stats_root.resolve())
        else:
            require(a.job is not None,'worker job')
            worker(root,next(j for j in jobs() if j['id']==a.job),a.commit,a.split_manifest.resolve(),a.source_lock.resolve(),a.stats_root.resolve())


if __name__=='__main__':main()
