"""V9-S1: 18 validation-selected trajectories, no test evaluation or cost gate."""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

import torch
import v9_cost_probe as c0
import v9_c2b_probe as c2b
import v9_srgt as srgt
import v9_ssra as ssra
from reproducibility import seed_everything, state_dict_sha256

REPO = Path(__file__).resolve().parent
TASK = 'V9_S1_PERFORMANCE_SCREEN_20260930'
METHODS = ('FROZEN', 'B1_LOW', 'REFINE_GRAPH', 'SRGT', 'LORA', 'SSRA')
CONTROLS = ('FROZEN', 'B1_LOW', 'REFINE_GRAPH', 'LORA')
SPEC = dict(seed=42, max_epochs=40, patience=8, min_delta=0., head_lr=.001,
            full_encoder_lr=.00003, adapter_lr=.0001, weight_decay=.00001,
            grad_clip=1., selection='validation_macro_rmse_strict_min_first_tie',
            sampling='one_pass_shuffled_task_batches', frozen_encoder_mode='eval',
            test_access=False, source_training=False)
TESTS = ('tests/test_v9_screen.py', 'tests/test_v9_srgt.py', 'tests/test_v9_ssra.py',
         'tests/test_v9_source_portability.py', 'tests/test_p1d4_identity.py')
TEST_COUNT = 97
read, write, sha, digest, require = c0.read, c0.write, c0.sha, c0.digest, c0.require


def jobs():
    return [dict(id=f'{s}_{m}_s42', setting=s, method=m, seed=42)
            for s in c0.SETTINGS for m in METHODS]


def epoch_plan(tasks, counts, batch_size, epoch):
    rng = random.Random(42 + epoch)
    batches = []
    for task in tasks:
        indices = list(range(counts[task])); rng.shuffle(indices)
        batches.extend(dict(task=task, indices=indices[i:i+batch_size])
                       for i in range(0, len(indices), batch_size))
    rng.shuffle(batches)
    return batches


def build(base, method, counts, stats_root, setting, device):
    if method in srgt.METHODS:
        model = srgt.DualGraph(base, method, counts).to(device)
        return model, srgt.optimizer_for(model)
    if method in ssra.METHODS:
        model = ssra.AdaptedGraph(base, method, counts,
                    c2b.load_stats(stats_root, setting, base.encoder)).to(device)
        return model, ssra.optimizer_for(model)
    require(method in ('FROZEN', 'B1_LOW'), 'unknown method')
    model = deepcopy(base).to(device)
    model.requires_grad_(True)
    groups = [dict(params=list(model.decoders.parameters()), lr=SPEC['head_lr'])]
    if method == 'FROZEN':
        model.encoder.requires_grad_(False)
    else:
        groups.append(dict(params=list(model.encoder.parameters()), lr=SPEC['full_encoder_lr']))
    return model, torch.optim.AdamW(groups, weight_decay=SPEC['weight_decay'])


def datasets_for(factory, trainer, setting):
    return factory.datasets if setting == 'ToxAcute' else trainer.datasets


def observations(factory, trainer, setting, split):
    """Trusted population/labels independent of prediction output."""
    datasets = datasets_for(factory, trainer, setting)[split]
    manifest = None
    if setting == 'ToxAcute':
        manifest = {r['sample_id']: r for r in read(factory.store.root/'split_manifest.json')['records']}
    rows = []
    for task, ds in datasets.items():
        for i in range(len(ds)):
            sid = str(ds.get_sample_id(i))
            if manifest is not None:
                meta = manifest[sid]
                require(meta['split'] == split, 'Tox observation split')
                canonical, group = meta['canonical_smiles'], meta['split_group']
                graph = ds[i]
                require(graph.sample_id == sid and graph.canonical_smiles == canonical, 'Tox graph identity')
                label = float(graph.y.reshape(-1)[0])
            else:
                view = ds.view; j = ds.indices[i]
                require(view.split == split and view.role == 'target', 'PubChem target split')
                canonical, group = view.canonical[j], view.groups[j]
                label = float(view.labels[j, view.tasks.index(task)])
            require(math.isfinite(label), 'finite observation')
            rows.append(dict(task=task, sample_id=sid, split=split,
                             canonical=canonical, group=group, label=label))
    require(len({(r['task'],r['sample_id']) for r in rows}) == len(rows), 'duplicate observation')
    return rows


def metrics(rows, expected):
    require(type(rows) is list and len(rows) == len(expected), 'validation population')
    lookup = {(r['task'],r['sample_id']): r for r in expected}
    seen = set(); errors = {}
    for row in rows:
        key = (row['task'],row['sample_id'])
        require(key in lookup and key not in seen, 'validation member/duplicate')
        require(set(row) == set(lookup[key]) | {'prediction'} and
                all(row[k] == v for k,v in lookup[key].items()), 'validation observation identity')
        require(type(row['prediction']) in (float,int) and math.isfinite(row['prediction']), 'finite prediction')
        seen.add(key); errors.setdefault(row['task'], []).append((row['prediction']-row['label'])**2)
    endpoints = {t:dict(n=len(e), rmse=math.sqrt(math.fsum(e)/len(e))) for t,e in errors.items()}
    return dict(endpoints=endpoints, macro_rmse=math.fsum(v['rmse'] for v in endpoints.values())/len(endpoints))


def batch_for(factory, trainer, setting, split, task, indices, device):
    ds = datasets_for(factory, trainer, setting)[split][task]
    batch = (factory.collator([ds[i] for i in indices]).to(device) if setting == 'ToxAcute'
             else trainer._batch(split, task, indices))
    require(not batch.is_empty and list(batch.sample_id) == [str(ds.get_sample_id(i)) for i in indices]
            and batch.y.numel() == len(indices), 'batch population')
    return batch


def evaluate(factory, trainer, setting, bank, expected, device, epoch):
    trainer.model.eval(); rows = []
    lookup = {(r['task'],r['sample_id']):r for r in expected}
    with torch.no_grad():
        for task, ds in datasets_for(factory, trainer, setting)['validation'].items():
            size = c0.BATCH_SIZES[setting]
            for start in range(0, len(ds), size):
                indices = list(range(start,min(len(ds),start+size)))
                batch = batch_for(factory,trainer,setting,'validation',task,indices,device)
                meta = [lookup[(task,str(sid))] for sid in batch.sample_id]
                require(list(batch.canonical_smiles) == [r['canonical'] for r in meta], 'validation graph chemistry')
                if getattr(trainer.model,'method',None) == 'SRGT':
                    batch.v9_support = bank.for_inference([r['canonical'] for r in meta], [r['group'] for r in meta]).to(device)
                if setting == 'ToxAcute':
                    raw,_ = trainer._forward_task(batch,task,epoch,return_aux=True)
                    prediction = trainer.decode_task_output(task,raw[task],apply_conformal=False)['median'].reshape(-1)
                else:
                    raw = trainer.model(batch,task_name=task)[task]
                    require(tuple(raw.shape) == (len(indices),3), 'quantile output shape')
                    scaler = trainer.scalers[task]
                    prediction = raw[:,0].double()*scaler['std']+scaler['mean']
                require(prediction.numel() == len(meta), 'prediction count')
                rows.extend(dict(r,prediction=float(p)) for r,p in zip(meta,prediction.cpu()))
    return rows, metrics(rows,expected)


def select(history, spec=SPEC):
    require(history and len(history) <= spec['max_epochs'], 'epoch bounds')
    best = 0
    for i, row in enumerate(history):
        require(row['epoch'] == i+1 and math.isfinite(row['validation']['macro_rmse']), 'epoch sequence/metric')
        if row['validation']['macro_rmse'] < history[best]['validation']['macro_rmse']: best = i
        require(i == len(history)-1 or i-best < spec['patience'], 'training continued after early stop')
    return best+1, len(history)-1-best >= spec['patience']


def train_one(factory, trainer, job, out, commit, device, stats_root, spec=SPEC):
    from loss import QuantileRegressionLoss
    require(job in jobs(), 'job outside matrix')
    setting, method = job['setting'], job['method']
    tasks, counts = c0.tasks_and_counts(setting)
    ds = datasets_for(factory,trainer,setting)['train']
    require({t:len(ds[t]) for t in tasks} == counts, 'frozen train counts')
    bank = srgt.TrainSupport(srgt.training_records(factory,trainer,setting))
    expected = observations(factory,trainer,setting,'validation')
    training = observations(factory,trainer,setting,'train')
    require(not {r['sample_id'] for r in training} & {r['sample_id'] for r in expected}, 'train/validation sample overlap')
    require(not {r['group'] for r in training} & {r['group'] for r in expected}, 'train/validation group overlap')
    source = state_dict_sha256(trainer.model.encoder)
    require(source == read(c2b.LOCK)['initial_encoders'][setting], 'accepted source encoder')
    seed_everything(42, deterministic_algorithms=True)
    model,opt = build(trainer.model,method,counts,stats_root,setting,device)
    trainer.model = model
    identity = dict(task=TASK,job=job,commit=commit,spec=spec,contract_sha256=digest(c0.contract(setting)),
                    initial_encoder=source, initial_heads=state_dict_sha256(model.decoders),
                    train_sha256=digest(training),validation_sha256=digest(expected),support=bank.identity)
    write(out/'identity.json',identity);write(out/'train_observations.json',training)
    write(out/'validation_observations.json',expected)
    history=[]; best=float('inf'); steps=0; best_epoch=0; started=time.monotonic()
    for epoch in range(spec['max_epochs']):
        model.train()
        if method == 'FROZEN':model.encoder.eval()
        plan=epoch_plan(tasks,counts,c0.BATCH_SIZES[setting],epoch)
        losses=[]
        for item in plan:
            task=item['task'];indices=item['indices']
            batch=batch_for(factory,trainer,setting,'train',task,indices,device)
            support=bank.for_train_batch(task,batch.sample_id,batch.canonical_smiles)
            if method == 'SRGT':batch.v9_support=support.to(device)
            opt.zero_grad(set_to_none=True)
            if setting == 'ToxAcute':loss,_=trainer._training_step(batch,task,epoch)
            else:
                raw=model(batch,task_name=task)[task];s=trainer.scalers[task]
                loss=QuantileRegressionLoss().compute_loss(raw,(batch.y.reshape(-1,1)-s['mean'])/s['std'])
            if method in srgt.METHODS:loss=loss+model.regularization()
            if method in ssra.METHODS:loss=loss+model.regularization(task)
            require(torch.isfinite(loss).item(), 'finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],spec['grad_clip'],error_if_nonfinite=True)
            opt.step();steps+=1;losses.append(float(loss.detach()))
        rows, scores=evaluate(factory,trainer,setting,bank,expected,device,epoch)
        write(out/f'validation_epoch_{epoch+1:03d}.json',rows)
        record=dict(epoch=epoch+1,validation=scores,updates=len(plan),total_updates=steps,
                    schedule_sha256=digest(plan),mean_batch_objective=math.fsum(losses)/len(losses))
        write(out/f'epoch_{epoch+1:03d}.json',record);history.append(record)
        if scores['macro_rmse'] < best:
            best=scores['macro_rmse'];best_epoch=epoch+1
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            require(all(torch.isfinite(v).all().item() for v in state.values()), 'finite selected state')
            with (out/'best.pending.pt').open('xb') as f:
                torch.save(dict(identity=identity,epoch=best_epoch,model_state=state),f)
            os.replace(out/'best.pending.pt',out/'best.pt')
        print(f'{job["id"]} epoch={epoch+1} val_macro_rmse={scores["macro_rmse"]:.8f} best_epoch={best_epoch}',flush=True)
        if epoch+1-best_epoch >= spec['patience']:break
    require(select(history,spec)[0] == best_epoch, 'best selection')
    final_source=state_dict_sha256(model.encoder)
    if method != 'B1_LOW':require(final_source == source, 'frozen source drift')
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(payload['model_state'],strict=True)
    replay,replay_metrics=evaluate(factory,trainer,setting,bank,expected,device,best_epoch-1)
    require(digest(replay) == digest(read(out/f'validation_epoch_{best_epoch:03d}.json')), 'best checkpoint replay')
    write(out/'selected_validation.json',replay)
    result=dict(task=TASK,job=job,commit=commit,identity=identity,history=history,best_epoch=best_epoch,
                selected=replay_metrics,updates=steps,checkpoint_sha256=sha(out/'best.pt'),
                selected_state_sha256=state_dict_sha256(model),final_encoder=final_source,
                selected_encoder=state_dict_sha256(model.encoder),elapsed_seconds=time.monotonic()-started,
                environment=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
                                 device=str(device),gpu_name=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None),
                checkpoint_validation_replay=True,test_evaluated=False,scientific_acceptance='PENDING_REVIEW')
    write(out/'receipt.json',result)
    return result


def rank(results):
    require(len(results) == len(jobs()) and {r['job']['id'] for r in results} == {j['id'] for j in jobs()}, 'incomplete screening matrix')
    values={(r['job']['setting'],r['job']['method']):r['selected']['macro_rmse'] for r in results}
    require(all(math.isfinite(v) and v >= 0 for v in values.values()), 'ranking metric')
    baselines={s:min(values[s,m] for m in CONTROLS) for s in c0.SETTINGS}
    scores=[]
    for method in METHODS:
        relative={s:(values[s,method]-baselines[s])/max(baselines[s],1e-12) for s in c0.SETTINGS}
        scores.append(dict(method=method,macro_rmse={s:values[s,method] for s in c0.SETTINGS},relative_to_best_control=relative,
                           worst_relative_change=max(relative.values()),mean_relative_change=math.fsum(relative.values())/3,
                           better_than_all_controls_in_all_scenes=all(v < 0 for v in relative.values())))
    scores.sort(key=lambda r:(r['worst_relative_change'],r['mean_relative_change'],METHODS.index(r['method'])))
    return dict(ranking=scores,recommended_for_confirmation=scores[0]['method'],
                per_scene_winners={s:min(METHODS,key=lambda m:values[s,m]) for s in c0.SETTINGS},
                scope='SINGLE_SEED_VALIDATION_SCREEN_ONLY',historical_strong_baselines_compared=False,
                unified_superiority_confirmed=False)


def verify_job(factory, trainer, job, out, commit, stats_root, spec=SPEC):
    r=read(out/'receipt.json');identity=read(out/'identity.json')
    require(r['task'] == TASK and r['job'] == job and r['commit'] == commit and r['identity'] == identity, 'receipt identity')
    require(identity['spec'] == spec and identity['job'] == job and identity['commit'] == commit
            and identity['contract_sha256'] == digest(c0.contract(job['setting'])), 'training contract')
    expected=observations(factory,trainer,job['setting'],'validation')
    training=observations(factory,trainer,job['setting'],'train')
    require(identity['train_sha256'] == digest(training) and identity['validation_sha256'] == digest(expected), 'trusted observations')
    require(read(out/'train_observations.json') == training and read(out/'validation_observations.json') == expected, 'saved observations')
    bank=srgt.TrainSupport(srgt.training_records(factory,trainer,job['setting']))
    require(identity['support'] == bank.identity, 'train reference identity')
    tasks,counts=c0.tasks_and_counts(job['setting']);history=r['history'];total=0
    for i,h in enumerate(history):
        plan=epoch_plan(tasks,counts,c0.BATCH_SIZES[job['setting']],i);total+=len(plan)
        require(h == read(out/f'epoch_{i+1:03d}.json') and h['schedule_sha256'] == digest(plan)
                and h['updates'] == len(plan) and h['total_updates'] == total, 'epoch update accounting')
        require(h['validation'] == metrics(read(out/f'validation_epoch_{i+1:03d}.json'),expected), 'independent metrics')
    best,stopped=select(history,spec)
    require(stopped or len(history) == spec['max_epochs'], 'premature truncation')
    require(r['best_epoch'] == best and r['updates'] == total, 'selection/update count')
    rows=read(out/'selected_validation.json')
    require(rows == read(out/f'validation_epoch_{best:03d}.json') and r['selected'] == metrics(rows,expected), 'selected rows')
    require(sha(out/'best.pt') == r['checkpoint_sha256'], 'checkpoint bytes')
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    require(set(payload) == {'identity','epoch','model_state'} and payload['identity'] == identity and payload['epoch'] == best, 'checkpoint binding')
    source=state_dict_sha256(trainer.model.encoder)
    require(identity['initial_encoder'] == source == read(c2b.LOCK)['initial_encoders'][job['setting']], 'source binding')
    model,_=build(trainer.model,job['method'],counts,stats_root,job['setting'],'cpu')
    require(identity['initial_heads'] == state_dict_sha256(model.decoders), 'initial heads')
    expected_state=model.state_dict();actual=payload['model_state']
    require(set(actual) == set(expected_state), 'model state keys')
    for k,v in expected_state.items():
        x=actual[k]
        require(isinstance(x,torch.Tensor) and x.dtype == v.dtype and x.shape == v.shape and torch.isfinite(x).all().item(), 'state tensor '+k)
        if (k.startswith('encoder.') and job['method'] != 'B1_LOW') or k in dict(model.named_buffers()):
            require(torch.equal(x.cpu(),v.cpu()), 'protected tensor '+k)
    model.load_state_dict(actual,strict=True)
    require(state_dict_sha256(model) == r['selected_state_sha256'] and state_dict_sha256(model.encoder) == r['selected_encoder'], 'selected tensor identity')
    if job['method'] != 'B1_LOW':require(r['final_encoder'] == r['selected_encoder'] == source, 'source drift')
    require(r['test_evaluated'] is False and r['checkpoint_validation_replay'] is True
            and r['scientific_acceptance'] == 'PENDING_REVIEW', 'scope')
    return r


def gate_check(root, commit, role):
    receipt=read(root/'gate.json');command=read(root/'command.json')
    require(receipt == dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')), 'gate identity')
    require(command['exit_code'] == 0, 'gate command failed')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases) == TEST_COUNT and all(not any(c.findall(t) for t in ('error','failure','skipped')) for c in cases), 'incomplete/failed/skipped gate')
    require({c.attrib['classname'] for c in cases} == {'tests.'+Path(t).stem for t in TESTS}, 'gate suite')


def code_gate(root, commit, role):
    require(role in ('wsl','server'), 'gate role');c0.check_code(REPO,commit)
    root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as f:
        p=subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json',dict(argv=argv,exit_code=p.returncode));require(p.returncode == 0,'code tests failed')
    write(root/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')))
    gate_check(root,commit,role)


def factory_for(setting, split, source, device):
    import p1d4_runtime
    f=p1d4_runtime.factory_for(REPO,setting=setting,seed=42,split_manifest=split,source_lock=source,device=device)
    return f,c2b.make_trainer(f,setting,device)


def verify(root, commit, split, source, stats):
    launch=read(root/'launch.json')
    require(launch['task'] == TASK and launch['commit'] == commit and launch['spec'] == SPEC and launch['jobs'] == jobs(), 'launch matrix')
    gate_check(root/'wsl_evidence',commit,'wsl');gate_check(root/'server_evidence',commit,'server')
    c2b.prior_gate(stats)
    results=[]
    for setting in c0.SETTINGS:
        f,t=factory_for(setting,split,source,'cpu')
        for job in [j for j in jobs() if j['setting'] == setting]:
            results.append(verify_job(f,t,job,root/job['id'],commit,stats))
    return dict(task=TASK,commit=commit,content_status='PASS',results=[dict(job=r['job'],selected=r['selected'],best_epoch=r['best_epoch'],updates=r['updates']) for r in results],**rank(results))


def run(a):
    from p1d4_batch import free_gpus
    c0.check_code(REPO,a.commit);gate_check(a.wsl_evidence,a.commit,'wsl');gate_check(a.server_evidence,a.commit,'server')
    c2b.prior_gate(a.stats_root);require(free_gpus([a.gpu]) == [a.gpu], 'GPU busy')
    root=a.output;root.mkdir(parents=True,exist_ok=False)
    registry=REPO/'.tmp/v9s1_attempt_20260930';registry.mkdir(parents=True,exist_ok=True)
    write(registry/'attempt.json',dict(task=TASK,commit=a.commit,output=str(root.resolve()),jobs=jobs()))
    uuid=c0.gpu_uuid(a.gpu)
    write(root/'launch.json',dict(task=TASK,commit=a.commit,spec=SPEC,jobs=jobs(),gpu_uuid=uuid,gpu=a.gpu))
    for role in ('wsl','server'):
        shutil.copytree(getattr(a,role+'_evidence'),root/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
    try:
        for job in jobs():
            require(free_gpus([a.gpu]) == [a.gpu], 'GPU occupied before next job')
            argv=[sys.executable,str(Path(__file__).resolve()),'_worker','--commit',a.commit,'--output',str(root),
                  '--job',job['id'],'--split-manifest',str(a.split_manifest),'--source-lock',str(a.source_lock),'--stats-root',str(a.stats_root)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/(job['id']+'.log')).open('xb') as log:
                p=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(root/(job['id']+'.command.json'),dict(argv=argv,exit_code=p.returncode))
            require(p.returncode == 0, 'job failed; preserve partial results')
        write(root/'verification.json',verify(root,a.commit,a.split_manifest,a.source_lock,a.stats_root))
    except BaseException as exc:
        write(root/'failed.json',dict(task=TASK,error_type=type(exc).__name__,reason=str(exc)))
        raise


def package(root, commit):
    require(read(root/'launch.json')['commit'] == commit, 'package commit')
    files={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'package symlink')
        if p.is_file():files[p.relative_to(root).as_posix()]=sha(p)
    path=root.with_suffix('.zip')
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(root/name,name)
        z.writestr('checksums.sha256',''.join(f'{v}  {k}\n' for k,v in files.items()))
    with zipfile.ZipFile(path) as z:require(z.testzip() is None,'ZIP CRC')
    value=dict(path=str(path),sha256=sha(path));write(Path(str(path)+'.sha256.json'),value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('code-gate','run','_worker','verify','package'))
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=('wsl','server'));p.add_argument('--gpu',type=int,choices=range(4))
    p.add_argument('--job',choices=[j['id'] for j in jobs()])
    for name in ('split-manifest','source-lock','stats-root','wsl-evidence','server-evidence'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.action == 'code-gate':code_gate(a.output,a.commit,a.role)
    elif a.action == 'package':print(json.dumps(package(a.output,a.commit),indent=2))
    else:
        require(all(getattr(a,k) is not None for k in ('split_manifest','source_lock','stats_root')), 'asset paths required')
        if a.action == 'verify':
            c0.check_code(REPO,a.commit)
            print(json.dumps(verify(a.output,a.commit,a.split_manifest,a.source_lock,a.stats_root),indent=2))
        elif a.action == 'run':
            require(a.gpu is not None and a.wsl_evidence is not None and a.server_evidence is not None,'runtime inputs')
            run(a)
        else:
            c0.check_code(REPO,a.commit)
            launch=read(a.output/'launch.json');claim=read(REPO/'.tmp/v9s1_attempt_20260930/attempt.json')
            require(claim == dict(task=TASK,commit=a.commit,output=str(a.output),jobs=jobs()),'worker claim')
            require(launch['commit'] == a.commit and launch['jobs'] == jobs() and launch['spec'] == SPEC,'worker launch')
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == launch['gpu_uuid'] and torch.cuda.is_available()
                    and torch.cuda.device_count() == 1,'single GPU binding')
            job=next(j for j in jobs() if j['id'] == a.job)
            out=a.output/job['id'];out.mkdir(exist_ok=False)
            write(out/'started.json',dict(job=job,commit=a.commit))
            f,t=factory_for(job['setting'],a.split_manifest,a.source_lock,'cuda:0')
            train_one(f,t,job,out,a.commit,'cuda:0',a.stats_root)


if __name__ == '__main__':main()
