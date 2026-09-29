"""Bounded SOURCE-TRAIN activation preparation for SSRA; zero optimization.

Two distinct sources, seed42, <=256 unique molecules each, batch8. Collect
atom-only, molecule-equal moments for Q/V/FFN inputs. Not an SSRA performance
run, a CorDA reproduction, or authorization of the remaining training matrix.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch
import v9_cost_probe as c0
import v9_c1_probe as c1
from reproducibility import state_dict_sha256

REPO=Path(__file__).resolve().parent
LOCK=REPO/'configs/v9_c2a_lock.json'
TASK='V9_C2A_SOURCE_STATS_20260929'
REGISTRY='.tmp/v9_c2a_attempts_20260929'
TESTS=('tests/test_v9_c2a_source_stats.py','tests/test_v9_source_portability.py','tests/test_p1d4_identity.py')
SOURCES=('Animal56','Nonhuman104')
SPEC=dict(schema='v9_c2a_source_stats_v1',seed=42,max_molecules=256,batch_size=8,
          selection='sha256_canonical_seed42',max_nodes=512,atom_only=True,molecule_equal=True,
          shrinkage=.1,projector_rank=16,half_policy='first_second_half_of_hash_order')
read,write,sha,digest,require=c0.read,c0.write,c0.sha,c0.digest,c0.require


def select_records(records):
    unique={};ids=set()
    for row in records:
        require(set(row)=={'sample_id','canonical','group','split','index'},'source record fields')
        require(row['split']=='train' and all(type(row[k]) is str and row[k] for k in ('sample_id','canonical','group')),'source train identity')
        require(type(row['index']) is int and row['index']>=0 and row['sample_id'] not in ids,'source row identity')
        ids.add(row['sample_id']);key=row['canonical']
        if key in unique:
            require(unique[key]['group']==row['group'],'source canonical crosses groups')
            if row['sample_id']<unique[key]['sample_id']:unique[key]=row
        else:unique[key]=row
    selected=sorted(unique.values(),key=lambda r:(digest(dict(seed=42,canonical=r['canonical'])),r['canonical']))[:256]
    require(len(selected)>=32,'insufficient source molecules for bounded statistics')
    return selected,dict(eligible_rows=len(records),eligible_unique=len(unique),selected=len(selected),
                         eligible_records_sha256=digest(sorted(records,key=lambda r:r['sample_id'])))


def source_context(source,split,lock):
    """Frozen real factory; only source-train molecules can enter this pool."""
    import numpy as np
    from p1d4_runtime import factory_for
    from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
    from dataset import DataCollator
    require(source in SOURCES,'source scope')
    setting='ToxAcute' if source=='Animal56' else 'A'
    f=factory_for(REPO,setting=setting,seed=42,split_manifest=split,source_lock=lock,device='cpu')
    trainer=c1.make_trainer(f,setting,'cpu')
    records=[]
    if source=='Animal56':
        store=f.store;manifest=read(store.root/'split_manifest.json')
        by_id={r['sample_id']:r for r in manifest['records']}
        require(len(by_id)==len(manifest['records']) and set(by_id)==set(str(s) for s in store.sample_ids),'source manifest bijection')
        indices=sorted({int(i) for t in ANIMAL_SOURCE_TASKS for i in store.get_task_indices(t,split='train',max_nodes=512)})
        for i in indices:
            sid=str(store.sample_ids[i]);r=by_id[sid]
            records.append(dict(sample_id=sid,canonical=r['canonical_smiles'],group=r['split_group'],split=r['split'],index=i))
        graph=lambda row:store.get_graph_data(row['index'])
        source_tasks=list(ANIMAL_SOURCE_TASKS)
    else:
        from dataset115_route_a_smoke import check_source_tasks
        from preprocess_data import get_graph_data_from_smiles
        from rdkit import Chem
        view=f._table.view('A','source','train');check_source_tasks(view.tasks)
        for i,sid in enumerate(view.sample_ids):
            if not np.isfinite(view.labels[i]).any():continue
            mol=Chem.MolFromSmiles(view.canonical[i]);require(mol is not None,'source molecule')
            if mol.GetNumAtoms()>512:continue
            records.append(dict(sample_id=sid,canonical=view.canonical[i],group=view.groups[i],split=view.split,index=i))
        def graph(row):
            i=row['index']
            return get_graph_data_from_smiles(view.smiles[i],0.,sample_id=row['sample_id'],max_path_distance=8)
        source_tasks=list(view.tasks)
    selected,population=select_records(records)
    encoder=trainer.model.encoder.eval().requires_grad_(False)
    expected=read(c1.LOCK)['initial_encoders'][setting]
    require(state_dict_sha256(encoder)==expected,'frozen source encoder')
    identity=dict(source=source,seed=42,encoder_sha256=expected,contract_sha256=digest(c0.contract(setting)),
                  source_tasks=source_tasks,spec=SPEC,population=population,selected=selected)
    return encoder,graph,DataCollator(),identity


def targets(encoder):
    result={}
    for i,layer in enumerate(encoder.backbone.layers):
        for suffix,module in (('attention.q_proj',layer.attention.q_proj),('attention.v_proj',layer.attention.v_proj),
                              ('ffn.0',layer.ffn[0]),('ffn.3',layer.ffn[3])):
            require(isinstance(module,torch.nn.Linear),'source linear module')
            result[f'layers.{i}.{suffix}']=module
    return result


def molecule_moments(inputs,mask):
    require(inputs.ndim==3 and mask.dtype==torch.bool and tuple(inputs.shape[:2])==(mask.shape[0],mask.shape[1]+1),'activation/mask shape')
    values=[]
    for x,m in zip(inputs[:,1:,:],mask):
        x=x[m].detach().to(device='cpu',dtype=torch.float64)
        require(len(x)>0 and torch.isfinite(x).all(),'valid finite source atoms')
        values.append((x.mean(0),x.T@x/len(x)))
    return values


def derive(sum_mean,sum_second,n):
    require(type(n) is int and n>=16,'source moment count')
    require(sum_mean.dtype==sum_second.dtype==torch.float64 and sum_second.shape==(sum_mean.numel(),sum_mean.numel()),'moment specs')
    require(torch.isfinite(sum_mean).all() and torch.isfinite(sum_second).all(),'moment finite')
    mu=sum_mean/n;raw=sum_second/n-torch.outer(mu,mu);raw=(raw+raw.T)/2
    d=len(mu);require(d>=16,'source feature dimension')
    require(torch.linalg.eigvalsh(raw).min()>=-1e-8,'raw covariance PSD')
    covariance=.9*raw+.1*torch.trace(raw)/d*torch.eye(d,dtype=torch.float64)
    eig,vec=torch.linalg.eigh(covariance);p=vec[:,-16:]@vec[:,-16:].T
    require(torch.isfinite(p).all() and torch.trace(covariance)>0,'nondegenerate source covariance')
    return dict(mean=mu,covariance=covariance,eigenvalues=eig,projector=p)


def collect(encoder,graph,collator,identity,device):
    selected=identity['selected'];encoder.to(device).eval().requires_grad_(False)
    before=state_dict_sha256(encoder);modules=targets(encoder);stats={};cache={};handles=[]
    for name,module in modules.items():
        d=module.in_features
        stats[name]=dict(count=[0,0],sum_mean=torch.zeros(2,d,dtype=torch.float64),sum_second=torch.zeros(2,d,d,dtype=torch.float64))
        def hook(mod,args,key=name):
            require(key not in cache,'duplicate source module invocation');cache[key]=molecule_moments(args[0],mask)
        handles.append(module.register_forward_pre_hook(hook))
    start=time.perf_counter();batches=0
    if device=='cuda:0':torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            for offset in range(0,len(selected),8):
                rows=selected[offset:offset+8];graphs=[graph(r) for r in rows]
                for g,r in zip(graphs,rows):
                    require(g.sample_id==r['sample_id'] and g.canonical_smiles==r['canonical'] and 0<len(g.x)<=512,'source graph identity/size')
                batch=collator(graphs).to(device)
                require(not batch.is_empty and list(batch.sample_id)==[r['sample_id'] for r in rows],'source batch identity')
                mask=batch.node_mask;cache.clear();encoder(batch);batches+=1
                require(set(cache)==set(modules),'complete source hook matrix')
                for key,values in cache.items():
                    for j,(mean,second) in enumerate(values):
                        half=int(offset+j>=len(selected)//2);s=stats[key]
                        s['count'][half]+=1;s['sum_mean'][half]+=mean;s['sum_second'][half]+=second
        if device=='cuda:0':torch.cuda.synchronize()
    finally:
        for handle in handles:handle.remove()
    require(state_dict_sha256(encoder)==before==identity['encoder_sha256'],'source drift')
    require(all(p.grad is None and not p.requires_grad for p in encoder.parameters()),'source gradient')
    seconds=time.perf_counter()-start
    decomposition_start=time.perf_counter()
    for s in stats.values():
        s['derived']=[derive(s['sum_mean'][j],s['sum_second'][j],s['count'][j]) for j in range(2)]
        s['pooled']=derive(s['sum_mean'].sum(0),s['sum_second'].sum(0),sum(s['count']))
    return dict(identity=identity,stats=stats),dict(forward_batches=batches,molecules=len(selected),
        collection_seconds=seconds,decomposition_seconds=time.perf_counter()-decomposition_start,optimizer_updates=0,head_predictions=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda:0' else 0)


def verify_payload(payload,identity,module_dims):
    require(set(payload)=={'identity','stats'} and digest(payload['identity'])==digest(identity),'statistics identity')
    require(set(payload['stats'])==set(module_dims),'statistics module matrix')
    diagnostics={};n=len(identity['selected'])
    for key,s in payload['stats'].items():
        require(set(s)=={'count','sum_mean','sum_second','derived','pooled'},'statistics fields')
        require(s['count']==[n//2,n-n//2] and all(type(v) is int for v in s['count']),'molecule exposure')
        require(tuple(s['sum_mean'].shape)==(2,module_dims[key]) and tuple(s['sum_second'].shape)==(2,module_dims[key],module_dims[key]),'statistics shapes')
        expected=[derive(s['sum_mean'][j],s['sum_second'][j],s['count'][j]) for j in range(2)]
        pooled=derive(s['sum_mean'].sum(0),s['sum_second'].sum(0),n)
        require(type(s['derived']) is list and len(s['derived'])==2,'half statistics')
        for a,b in zip(s['derived']+[s['pooled']],expected+[pooled]):
            require(set(a)==set(b),'derived fields')
            for field in b:require(a[field].dtype==b[field].dtype and a[field].shape==b[field].shape and torch.allclose(a[field],b[field],atol=1e-10,rtol=1e-8),'derived statistics differ')
        p,q=expected[0]['projector'],expected[1]['projector']
        diagnostics[key]=dict(half_projector_distance=float(torch.linalg.matrix_norm(p-q)/32**.5),
                              effective_trace=float(torch.trace(pooled['covariance'])),
                              relative_rank16_gap=float((pooled['eigenvalues'][-16]-pooled['eigenvalues'][-17])/pooled['eigenvalues'][-1]) if module_dims[key]>16 else None)
    return diagnostics


def prior_gate(root):
    expected=read(LOCK)['accepted_073_files'];actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual in (set(expected),set(expected)-{'checksums.sha256'}),'073 accepted file population')
    require(all(not p.is_symlink() for p in root.rglob('*')),'073 evidence symlink')
    require(all(sha(root/name)==expected[name] for name in actual),'073 accepted bytes')


def gate(root,commit,role):
    value=read(root/'gate.json')
    require(value==dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')),'C2A gate identity')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(cases and {c.attrib['classname'] for c in cases}=={'tests.'+Path(t).stem for t in TESTS},'C2A gate files')
    require(all(not any(c.findall(t) for t in ('failure','error','skipped')) for c in cases),'C2A test failure')


def code_gate(root,commit,role):
    c0.check_code(REPO,commit);root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as f:r=subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=600)
    write(root/'command.json',dict(argv=argv,exit_code=r.returncode));require(r.returncode==0,'C2A code gate')
    write(root/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')));gate(root,commit,role)


def verify(root,commit,split,source_lock):
    launch=read(root/'launch.json');require(launch['task']==TASK and launch['commit']==commit and launch['sources']==list(SOURCES),'C2A launch')
    reports={}
    for source in SOURCES:
        encoder,_,_,identity=source_context(source,split,source_lock)
        folder=root/source;receipt=read(folder/'receipt.json');path=folder/'moments.pt'
        require(receipt['sha256']==sha(path) and receipt['commit']==commit and receipt['task']==TASK,'C2A receipt')
        require(receipt['optimizer_updates']==receipt['head_predictions']==0 and receipt['molecules']==len(identity['selected'])
                and receipt['forward_batches']==(len(identity['selected'])+7)//8,'C2A forward scope')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        diagnostics=verify_payload(payload,identity,{k:m.in_features for k,m in targets(encoder).items()})
        reports[source]=dict(receipt_sha256=sha(folder/'receipt.json'),moments_sha256=sha(path),diagnostics=diagnostics)
    return dict(task=TASK,commit=commit,content_status='PASS',scientific_acceptance='NOT_ASSESSED',
                scope='SOURCE_TRAIN_STATS_ONLY',optimizer_updates=0,total_cost_updates=144,remaining_cost_updates=216,
                formal_training_authorized=False,results=reports)


def run(root,commit,gpu,split,source_lock,wsl,server,prior):
    from p1d4_batch import free_gpus
    c0.check_code(REPO,commit);gate(wsl,commit,'wsl');gate(server,commit,'server');prior_gate(prior)
    require(free_gpus([gpu])==[gpu],'GPU occupied')
    root.mkdir(parents=True,exist_ok=False);registry=REPO/REGISTRY;registry.mkdir(parents=True,exist_ok=True)
    write(registry/'attempt.json',dict(task=TASK,commit=commit,root=str(root),optimizer_updates=0,max_molecules=512,max_wall_seconds=1800))
    uuid=c0.gpu_uuid(gpu);write(root/'launch.json',dict(task=TASK,commit=commit,sources=list(SOURCES),gpu_uuid=uuid,max_wall_seconds=1800,optimizer_updates=0))
    for role,p in (('wsl',wsl),('server',server)):shutil.copytree(p,root/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
    try:
        deadline=time.monotonic()+1800
        for source in SOURCES:
            require(free_gpus([gpu])==[gpu],'GPU occupied before source job')
            remaining=deadline-time.monotonic();require(remaining>0,'C2A wall limit')
            argv=[sys.executable,__file__,'_worker','--commit',commit,'--output',str(root),'--source',source,'--split-manifest',str(split),'--source-lock',str(source_lock)]
            with (root/(source+'.log')).open('xb') as f:r=subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,timeout=remaining,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            write(root/(source+'.command.json'),dict(argv=argv,exit_code=r.returncode));require(r.returncode==0,'source preparation failed; no retry')
        write(root/'verification.json',verify(root,commit,split,source_lock))
    except BaseException as exc:
        write(root/'failed.json',dict(task=TASK,error=str(exc),optimizer_updates=0));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['code-gate','run','verify','package','_worker'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--role',choices=['wsl','server']);p.add_argument('--gpu',type=int,choices=range(4));p.add_argument('--source',choices=SOURCES)
    for name in ('split-manifest','source-lock','wsl-evidence','server-evidence','c1-evidence'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();root=a.output.resolve()
    if a.action!='_worker':os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.action=='code-gate':code_gate(root,a.commit,a.role)
    elif a.action=='package':
        require(read(root/'launch.json')['commit']==a.commit,'C2A package commit')
        # Dedicated package avoids C1's smoke_state filename whitelist.
        require(not root.with_suffix('.zip').exists(),'C2A package exists')
        import zipfile
        files={x.relative_to(root).as_posix():sha(x) for x in root.rglob('*') if x.is_file()}
        require(all(not x.is_symlink() for x in root.rglob('*')),'C2A package symlink')
        archive=root.with_suffix('.zip')
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name in files:z.write(root/name,name)
            z.writestr('checksums.sha256',''.join(f'{s}  {n}\n' for n,s in sorted(files.items())))
        Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf8')
        print(json.dumps(dict(archive=str(archive),sha256=sha(archive))))
    else:
        require(a.split_manifest is not None and a.source_lock is not None,'C2A input paths');c0.check_code(REPO,a.commit)
        if a.action=='verify':print(json.dumps(verify(root,a.commit,a.split_manifest,a.source_lock),indent=2))
        elif a.action=='run':
            require(all(v is not None for v in (a.gpu,a.wsl_evidence,a.server_evidence,a.c1_evidence)),'C2A run inputs')
            run(root,a.commit,a.gpu,a.split_manifest,a.source_lock,a.wsl_evidence,a.server_evidence,a.c1_evidence)
        else:
            require(a.source in SOURCES,'source job');launch=read(root/'launch.json')
            require(launch['task']==TASK and launch['commit']==a.commit and os.environ.get('CUDA_VISIBLE_DEVICES')==launch['gpu_uuid'],'C2A worker identity')
            require(read(REPO/REGISTRY/'attempt.json')['root']==str(root),'C2A claim')
            require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU');torch.use_deterministic_algorithms(True)
            folder=root/a.source;folder.mkdir(exist_ok=False)
            tick=time.perf_counter();encoder,graph,collator,identity=source_context(a.source,a.split_manifest,a.source_lock)
            asset_seconds=time.perf_counter()-tick
            payload,r=collect(encoder,graph,collator,identity,'cuda:0')
            tick=time.perf_counter()
            with (folder/'moments.pt').open('xb') as f:torch.save(payload,f)
            save_seconds=time.perf_counter()-tick
            write(folder/'selection.json',identity);write(folder/'receipt.json',dict(r,task=TASK,commit=a.commit,sha256=sha(folder/'moments.pt'),
                asset_seconds=asset_seconds,save_seconds=save_seconds))


if __name__=='__main__':main()
