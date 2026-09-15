"""Frozen S4E5 per-run full mechanism evaluation. No training or HPO."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch

from s4e_mechanism_design import require, read_json, write_json, sha, unique, datastore_semantic_identity
from s4e_mechanism_metrics import parameter_drift, linear_cka, hybrid_state, functional_forgetting
from s4e_mechanism_smoke import load_design, state_from_asset, SourceModel, embeddings, PREFIX
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS


def matrix(docs):
    runs = docs['mechanism_asset_lock.json']['runs']
    keyed = unique(runs, 'run_id')
    expected = {(m,s,f) for m in ('B1','RPT') for s in range(42,47) for f in (10,25,50,75,100)}
    keys = [(x['method'],x['seed'],x['declared_fraction_percent']) for x in runs]
    require(len(keys)==50 and set(keys)==expected, 'formal matrix differs')
    return keyed


def grouped_drift(reference,candidate,layers=8):
    global_value=parameter_drift(reference,candidate)
    groups={f'block_{i}':[] for i in range(layers)};groups['nonblock']=[]
    for k in sorted(reference):
        if k.startswith(PREFIX+'layers.'):
            i=k[len(PREFIX+'layers.'):].split('.')[0]
            require(i.isdigit() and f'block_{int(i)}' in groups, 'unexpected backbone layer')
            groups[f'block_{int(i)}'].append(k)
        else:groups['nonblock'].append(k)
    result={'global':global_value}
    for name,keys in groups.items():
        require(bool(keys),'empty backbone group')
        result[name]=parameter_drift({k:reference[k] for k in keys},{k:candidate[k] for k in keys})
    require(set(k for v in groups.values() for k in v)==set(reference), 'group coverage differs')
    return result


def output_checksums(output):
    write_json(output/'checksums.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


def evaluate(repo,design,design_sha,datastore,run_id,output,device_name,authorization_sha):
    from dataset import DataCollator
    from preprocess_data import get_graph_data_from_smiles
    from toxacute_datastore import ToxAcuteDataStore
    manifest,docs=load_design(design,design_sha);runs=matrix(docs)
    require(run_id in runs,'run outside matrix');require(not output.exists(),'existing output; verify or choose new directory')
    require(datastore_semantic_identity(datastore)==manifest['datastore_semantic_identity'],'DataStore contents differ')
    run=runs[run_id];assets=unique(docs['mechanism_asset_lock.json']['assets'],'asset_id')
    teacher=assets[run['teacher_asset_id']];init=assets[run['init_asset_id']]
    payload,source,source_receipt=state_from_asset(repo,teacher)
    _,initial,init_receipt=state_from_asset(repo,init)
    _,best,best_receipt=state_from_asset(repo,run['best'])
    reference={k:v for k,v in initial.items() if k.startswith(PREFIX)}
    source_bb={k:v for k,v in source.items() if k.startswith(PREFIX)}
    target={k:v for k,v in best.items() if k.startswith(PREFIX)}
    require(set(reference)==set(source_bb),'source/init backbone keys differ')
    require(all(torch.equal(reference[k],source_bb[k]) and reference[k].dtype==source_bb[k].dtype for k in reference),'source/init tensors differ')
    hybrid=hybrid_state(source,target)  # verifies candidate key/shape/dtype before metrics
    drift=grouped_drift({k:v.numpy() for k,v in reference.items()},{k:v.numpy() for k,v in target.items()})
    scaler=unique(teacher['scalers'],'task')
    for t in ANIMAL_SOURCE_TASKS:
        require(payload['task_scalers'][t]['mean']==scaler[t]['mean'] and
                payload['task_scalers'][t]['std']==scaler[t]['std'] and scaler[t]['std']>0,'scaler mismatch')
    device=torch.device(device_name);torch.set_num_threads(1);torch.manual_seed(42)
    if device.type=='cuda':
        require(torch.cuda.is_available(),'CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=SourceModel(teacher['architecture_config']).to(device).eval()
    store=ToxAcuteDataStore(datastore);collator=DataCollator(spatial_pos_max_clip=20,max_node_filter=None)
    probe=docs['probe_manifest.json']['ordered_samples'];ids=[x['sample_key'] for x in probe]
    require(len(ids)==len(set(ids))==228,'probe count differs')
    tasks=docs['animal56_validation_manifest.json']['tasks']
    require([x['task'] for x in tasks]==list(ANIMAL_SOURCE_TASKS),'validation task order differs')
    require(sum(len(x['observations']) for x in tasks)==14090,'validation count differs')
    graph_indices=sorted({o['global_index'] for t in tasks for o in t['observations']})
    pos={i:n for n,i in enumerate(graph_indices)}
    output.mkdir(parents=True)
    write_json(output/'scope.json',{'run_id':run_id,'method':run['method'],'seed':run['seed'],
        'fraction_percent':run['declared_fraction_percent'],'design_sha256':design_sha,
        'authorization_sha256':authorization_sha,'scope':'FORMAL_S4E5','device':device_name,
        'probe_ids':ids,'source':source_receipt,'init':init_receipt,'best':best_receipt,
        'source_cache':'NONE_each_run_revalidates_and_recomputes_source',
        'batch_size':4,'torch_version':torch.__version__})
    try:
        graphs=[]
        for sample in probe:
            rec=sample['input_record']
            if rec['source']=='toxacute':
                g=store.get_graph_data(rec['global_index'])
                require(g.sample_id==rec['sample_id'],'probe graph ID differs')
            else:
                g=get_graph_data_from_smiles(rec['raw_smiles'],0.,sample_id=sample['sample_key'],
                    max_path_distance=store.metadata['max_path_distance'])
            require(g.canonical_smiles==sample['canonical_smiles'] and g.x.shape[0]==rec['num_nodes'],'probe graph identity differs')
            g.sample_id=sample['sample_key'];graphs.append(g)
        for item in tasks:
            t=item['task']
            for o in item['observations']:
                require(store.sample_ids[o['global_index']]==o['sample_id'] and
                        store.get_label(o['global_index'],t)==o['label_raw'],'functional graph/label identity differs')
        representation={};predictions={}
        for role,state in [('source',source),('hybrid',hybrid)]:
            model.load_state_dict(state,strict=True);model.eval()
            parts={}
            for start in range(0,len(graphs),4):
                batch=collator([g.clone() for g in graphs[start:start+4]]).to(device)
                require(list(batch.sample_id)==ids[start:start+4],'collated probe order differs')
                values=embeddings(model,batch)
                for k,v in values.items():parts.setdefault(k,[]).append(v)
                del batch
            representation[role]={k:np.concatenate(v,axis=0) for k,v in parts.items()}
            parts={t:[] for t in ANIMAL_SOURCE_TASKS}
            for start in range(0,len(graph_indices),4):
                selected=graph_indices[start:start+4]
                batch=collator([store.get_graph_data(i) for i in selected]).to(device)
                require(list(batch.sample_id)==[str(store.sample_ids[i]) for i in selected],'functional collator order differs')
                with torch.no_grad():values=model(batch)
                for t,v in values.items():parts[t].append(v.detach().cpu().numpy())
                del batch
            predictions[role]={t:np.concatenate(v).astype('f8')*scaler[t]['std']+scaler[t]['mean'] for t,v in parts.items()}
        cka={k:linear_cka(representation['source'][k],representation['hybrid'][k],ids,ids) for k in representation['source']}
        labels={};sp={};hp={};eid={};rows=[]
        for item in tasks:
            t=item['task'];obs=item['observations'];eid[t]=[o['sample_id'] for o in obs]
            labels[t]=[o['label_raw'] for o in obs]
            a=[float(predictions['source'][t][pos[o['global_index']]]) for o in obs]
            b=[float(predictions['hybrid'][t][pos[o['global_index']]]) for o in obs]
            sp[t]=(eid[t],a);hp[t]=(eid[t],b)
            rows.extend(dict(o,task=t,source_prediction=x,hybrid_prediction=y) for o,x,y in zip(obs,a,b))
        metrics=functional_forgetting(labels,sp,hp,list(ANIMAL_SOURCE_TASKS),eid)
        if run['method']=='RPT':
            require(drift['global']['difference_l2']==0 and all(x['delta_rmse']==0 for x in metrics['tasks']),'RPT invariance failed')
        np.savez_compressed(output/'embeddings.npz',**{role+'_'+k:v for role,parts in representation.items() for k,v in parts.items()})
        write_json(output/'predictions.json',rows)
        write_json(output/'metrics.json',{'scope':'FORMAL_S4E5','drift':drift,'cka':cka,'functional':metrics})
        # Compact per-tensor evidence supports independent aggregation checks.
        write_json(output/'parameter_norms.json',{k:parameter_drift({k:reference[k].numpy()},{k:target[k].numpy()}) for k in sorted(reference)})
    finally:store.close()
    output_checksums(output)


def verify(design,design_sha,output,authorization_sha):
    _,docs=load_design(design,design_sha);runs=matrix(docs)
    checks=read_json(output/'checksums.json')
    require(set(checks)=={'scope.json','metrics.json','predictions.json','embeddings.npz','parameter_norms.json'},'output file set differs')
    for name,digest in checks.items():require(sha(output/name)==digest,'output checksum differs: '+name)
    scope=read_json(output/'scope.json');require(scope['scope']=='FORMAL_S4E5' and scope['design_sha256']==design_sha and scope['authorization_sha256']==authorization_sha,'output scope differs')
    run=runs[scope['run_id']];assets=unique(docs['mechanism_asset_lock.json']['assets'],'asset_id')
    for k,a in [('source',assets[run['teacher_asset_id']]),('init',assets[run['init_asset_id']]),('best',run['best'])]:
        require(scope[k]['sha256']==a['sha256'],'output asset differs')
    require((scope['method'],scope['seed'],scope['fraction_percent'])==(run['method'],run['seed'],run['declared_fraction_percent']),'output run identity differs')
    ids=[x['sample_key'] for x in docs['probe_manifest.json']['ordered_samples']]
    require(scope['probe_ids']==ids,'output probe IDs differ')
    metrics=read_json(output/'metrics.json');require(metrics['scope']=='FORMAL_S4E5','metric scope differs')
    layers=[f'block_{i}' for i in range(8)]+['final_graph_token']
    with np.load(output/'embeddings.npz',allow_pickle=False) as z:
        require(set(z.files)=={r+'_'+k for r in ('source','hybrid') for k in layers},'embedding layers differ')
        hidden=assets[run['teacher_asset_id']]['architecture_config']['hidden_dim']
        require(all(z[n].shape==(len(ids),hidden) for n in z.files),'embedding dimensions differ')
        computed={k:linear_cka(z['source_'+k],z['hybrid_'+k],ids,ids) for k in layers}
    require(computed==metrics['cka'],'CKA recomputation differs')
    rows=read_json(output/'predictions.json');table={}
    for row in rows:
        key=(row['task'],row['sample_id']);require(key not in table,'duplicate prediction');table[key]=row
    tasks=docs['animal56_validation_manifest.json']['tasks']
    expected={(t['task'],o['sample_id']) for t in tasks for o in t['observations']}
    require(set(table)==expected and len(rows)==14090,'prediction sample set differs')
    labels={};sp={};hp={};eid={}
    for item in tasks:
        t=item['task'];obs=item['observations'];eid[t]=[o['sample_id'] for o in obs]
        for o in obs:require(all(type(table[t,o['sample_id']][k]) is type(v) and table[t,o['sample_id']][k]==v for k,v in o.items()),'label or sample identity differs')
        labels[t]=[o['label_raw'] for o in obs]
        sp[t]=(eid[t],[table[t,i]['source_prediction'] for i in eid[t]])
        hp[t]=(eid[t],[table[t,i]['hybrid_prediction'] for i in eid[t]])
    require(functional_forgetting(labels,sp,hp,list(ANIMAL_SOURCE_TASKS),eid)==metrics['functional'],'functional recomputation differs')
    norms=read_json(output/'parameter_norms.json')
    inv=assets[run['init_asset_id']]['backbone']['tensors'];keys={x['key'] for x in inv}
    require(set(norms)==keys,'parameter keys differ')
    groups={'global':sorted(keys),'nonblock':sorted(k for k in keys if not k.startswith(PREFIX+'layers.'))}
    groups.update({f'block_{i}':sorted(k for k in keys if k.startswith(PREFIX+f'layers.{i}.')) for i in range(8)})
    require(set(metrics['drift'])==set(groups),'drift groups differ')
    for k,x in norms.items():
        require(x['keys']==[k] and all(type(x[n]) in (int,float) and math.isfinite(x[n]) and x[n]>=0 for n in ('difference_l2','reference_l2')),'invalid tensor norm')
    for group,ks in groups.items():
        require(metrics['drift'][group]['keys']==ks,'drift group key partition differs')
        d=math.hypot(*(norms[k]['difference_l2'] for k in ks));r=math.hypot(*(norms[k]['reference_l2'] for k in ks))
        actual=metrics['drift'][group]
        require(math.isclose(d,actual['difference_l2'],rel_tol=1e-12,abs_tol=1e-12) and math.isclose(r,actual['reference_l2'],rel_tol=1e-12,abs_tol=1e-12),'drift norm aggregation differs')
        require((actual['relative_l2'] is None and actual['reason']=='zero_reference_norm') if r==0 else (actual['reason'] is None and math.isclose(actual['relative_l2'],d/r,rel_tol=1e-12,abs_tol=1e-12)),'relative drift differs')
    if run['method']=='RPT':require(metrics['drift']['global']['difference_l2']==0 and all(x['delta_rmse']==0 for x in metrics['functional']['tasks']),'RPT invariant differs')
    return {'run_id':run['run_id'],'validation_status':'PASS','acceptance_status':'PENDING_CODEX_REVIEW','observations':len(rows),'cka_layers':9}
