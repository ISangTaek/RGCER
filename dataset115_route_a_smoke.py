"""Route A source-to-target smoke. No formal epochs, selection or test access."""
from pathlib import Path
from types import SimpleNamespace
import json,math
import numpy as np
import torch
from torch import nn
from architecture.Graphormer import Encoder,Graphormer
from architecture.prediction_heads import TaskPredictionHead
from dataset115_contract import ContractError,PRIMARY,semantic_digest,digest
from dataset115_adapter import GraphTaskView,TrainOnlyScaler
from dataset115_smoke import state_digest,one_step,verify_step_records

ARCH=dict(hidden_dim=96,a_layers=8,a_heads=4,mid_dim=128,
          head_hidden_dim=96,head_dropout=.1,edge_bias_mode='path',spatial_pos_clip=20)
TASK_ID='S5F1_ROUTEA_SOURCE_TARGET_SMOKE_20260916'
SERVER_TOX_SHA='eeba19c20362e88d309aab9cd59200f3eff3d4b2a48e5f4c8dc5e57b712f6ae8'

def require(ok,msg):
    if not ok:raise ContractError(msg)

def check_source_tasks(tasks):
    require(isinstance(tasks,(tuple,list)) and len(tasks)==104 and len(set(tasks))==104,'Nonhuman104 task count/duplicates')
    require(all(isinstance(t,str) and len(t.rsplit('_',2))==3 and
                t.rsplit('_',2)[0] not in {'human','man','women','child'} for t in tasks),'human/invalid source task')

def build_source(args,tasks,seed):
    check_source_tasks(tasks)
    require(type(seed) is int and seed in range(42,47),'source seed')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        heads=nn.ModuleDict({t:TaskPredictionHead(args.hidden_dim,mode='quantile',
            head_hidden_dim=args.head_hidden_dim,dropout=args.head_dropout) for t in tasks})
        model=Graphormer(list(tasks),Encoder,heads,torch.device('cpu'),args)
    require(getattr(model,'card',None) is None,'CARD forbidden')
    return model

def inventory(table):
    """Actual role/split isolation; no graph or prediction for holdout partitions."""
    source=table.view('A','source','train');target=table.view('A','target','train')
    check_source_tasks(source.tasks)
    require(source.tasks==table.source_tasks and target.tasks==PRIMARY,'table task mapping')
    require(source.route==target.route=='A' and source.role=='source' and target.role=='target'
            and source.split==target.split=='train','source/target route scope')
    require(source.input_identity==target.input_identity==table.identity,'data identity')
    require(source.sample_ids==target.sample_ids and source.groups==target.groups,'Route A shared train population')
    scalers={};counts={};steps={}
    for role,view in [('source',source),('target',target)]:
        sc=TrainOnlyScaler.fit(view);scalers[role]=sc
        counts[role]={t:int(np.isfinite(view.labels[:,j]).sum()) for j,t in enumerate(view.tasks)}
        require(all(n>0 for n in counts[role].values()),'empty training task')
        require(role!='target' or all(n>=2 for n in counts[role].values()),'target smoke needs two observations')
        steps[role]=sum(math.ceil(n/32) for n in counts[role].values())
    return source,target,scalers,dict(input_identity=table.identity,source_tasks=list(source.tasks),
        target_tasks=list(target.tasks),train_counts=counts,train_molecules=len(source.sample_ids),
        batch32_updates_per_epoch=steps,source_train_ids_sha256=semantic_digest(list(source.sample_ids)),
        source_scaler=scalers['source'].to_dict(),target_scaler=scalers['target'].to_dict())

def select_batches(view):
    from dataset import DataCollator
    require(view.route=='A' and view.split=='train','smoke graph scope')
    batches={};ids={}
    for t in view.tasks:
        ds=GraphTaskView(view,t);n=min(2,len(ds));require(n>0,'empty smoke task')
        batches[t]=DataCollator()([ds[i] for i in range(n)])
        require(not batches[t].is_empty and batches[t].y.numel()==n,'graph dropped')
        ids[t]=[ds.get_sample_id(i) for i in range(n)]
    return batches,ids

def write_json(path,obj):
    with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)

def verify_saved(root,info,target_scalers):
    """Re-read real checkpoint tensors and role metadata before a smoke receipt."""
    root=Path(root);r=json.loads((root/'smoke_result.json').read_text(encoding='utf8'))
    require(r['task_id']==TASK_ID and r['formal_source_reusable'] is False and
            r['formal_training_ready'] is False and r['input_identity']==info['input_identity'],'saved run scope')
    require(all(r[k] is False for k in ('test_predictions_accessed','calibration_predictions_accessed','selection_performed')),'saved holdout scope')
    require(json.loads((root/'data_inventory.json').read_text(encoding='utf8'))==info,'saved inventory')
    sr=r['source'];p=root/'source_smoke.pt'
    require(sr['checkpoint']==p.name and digest(p)==sr['checkpoint_sha256'],'source file SHA')
    q=torch.load(p,map_location='cpu',weights_only=True)
    require(q['schema']=='route_a_source_smoke_v1' and q['task_id']==TASK_ID and type(q['seed']) is int and q['seed']==42
            and type(q['steps']) is int and q['steps']==1 and q['formal_source'] is False,'source saved scope')
    require(q['task_names']==info['source_tasks']==sr['task_names'] and q['identity']==info,'source saved task/data identity')
    check_source_tasks(q['task_names'])
    state={k[len('encoder.'):]:v for k,v in q['model_state'].items() if k.startswith('encoder.')}
    require(bool(state) and all(torch.isfinite(v).all() for v in q['model_state'].values()),'saved source finite tensors')
    require(state_digest(state)==sr['encoder_after']!=sr['encoder_before'],'saved source tensor identity')
    require(set(sr['head_changed'])==set(info['source_tasks']) and all(v is True for v in sr['head_changed'].values()),'source updated heads')
    require(set(sr['loss_per_task'])==set(info['source_tasks'])==set(sr['reload_max_error']),'source numerical task matrix')
    require(all(type(v) in (int,float) and math.isfinite(v) for v in sr['loss_per_task'].values()),'saved source loss')
    require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1e-6 for v in sr['reload_max_error'].values()),'saved source reload')
    verify_step_records(root/'target',r['target_runs'])
    require(r['target_runs'][1]['encoder_before']==r['target_runs'][2]['encoder_before']==sr['encoder_after'],'saved source-target chain')
    for row in r['target_runs']:
        payload=torch.load(root/'target'/row['checkpoint'],map_location='cpu',weights_only=True)
        require(payload['scalers']==target_scalers,'saved target scaler')
    return {'status':'CONTENT_PASS','source_checkpoints':1,'target_checkpoints':3,'formal_training_ready':False}

def source_step(args,tasks,batches,scalers,identity,output,*,device):
    """One accumulated update across all source tasks, not a source epoch."""
    from loss import QuantileRegressionLoss
    check_source_tasks(tasks);path=Path(output)
    require(not path.exists(),'source checkpoint exists')
    require(set(batches)==set(scalers)==set(tasks),'source batch/scaler matrix')
    for t in tasks:
        s=scalers[t];require(type(s['std']) in (int,float) and math.isfinite(s['std']) and s['std']>0 and math.isfinite(s['mean']),'source scaler numeric')
        require(not getattr(batches[t],'is_empty',False) and batches[t].y.numel() in (1,2),'source smoke batch size')
    torch.manual_seed(42);model=build_source(args,tasks,42).to(device);model.train()
    before=state_digest(model.encoder.state_dict());heads={t:state_digest(model.decoders[t].state_dict()) for t in tasks}
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5);opt.zero_grad(set_to_none=True)
    loss_fn=QuantileRegressionLoss();losses={}
    for t in tasks:
        b=batches[t].to(device);y=(b.y.reshape(-1,1)-scalers[t]['mean'])/scalers[t]['std']
        loss=loss_fn.compute_loss(model(b,task_name=t)[t],y)
        require(bool(torch.isfinite(loss)),'nonfinite source loss')
        (loss/len(tasks)).backward();losses[t]=float(loss.detach())
    require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoders.parameters()),'source head gradients missing/nonfinite')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'nonfinite source tensors')
    after=state_digest(model.encoder.state_dict());changed={t:heads[t]!=state_digest(model.decoders[t].state_dict()) for t in tasks}
    require(before!=after and all(changed.values()),'source encoder/head did not update')
    payload=dict(schema='route_a_source_smoke_v1',task_id=TASK_ID,seed=42,steps=1,formal_source=False,
        task_names=list(tasks),architecture=vars(args),identity=identity,scalers=scalers,
        model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer_state=opt.state_dict())
    with path.open('xb') as f:torch.save(payload,f)
    q=torch.load(path,map_location='cpu',weights_only=True)
    require(q['task_names']==list(tasks) and q['identity']==identity and q['steps']==1 and q['formal_source'] is False,'saved source identity')
    require(state_digest(q['model_state'])==state_digest(model.state_dict()),'saved source tensors')
    restored=build_source(args,tasks,42).to(device);restored.load_state_dict(q['model_state'],strict=True)
    restored_opt=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=1e-5);restored_opt.load_state_dict(q['optimizer_state'])
    require(len(restored_opt.state)==len(opt.state),'source optimizer reload')
    model.eval();restored.eval();errors={}
    with torch.no_grad():
        for t in tasks:
            a=model(batches[t],task_name=t)[t];b=restored(batches[t],task_name=t)[t]
            require(torch.allclose(a,b,atol=1e-6,rtol=1e-6),'source reload predictions');errors[t]=float((a-b).abs().max())
    encoder={k[len('encoder.'):]:v.clone() for k,v in q['model_state'].items() if k.startswith('encoder.')}
    require(state_digest(encoder)==after,'source encoder extraction')
    return encoder,dict(checkpoint=path.name,checkpoint_sha256=digest(path),encoder_before=before,
        encoder_after=after,head_changed=changed,loss_per_task=losses,grad_norm=float(norm),reload_max_error=errors,
        task_names=list(tasks),steps=1,formal_source=False)

def run_smoke(table,output,*,device,args=None):
    root=Path(output);require(not root.exists(),'output exists');args=SimpleNamespace(**ARCH) if args is None else args
    source,target,scalers,info=inventory(table)
    sb,si=select_batches(source);tb,ti=select_batches(target)
    root.mkdir(parents=True,exist_ok=False)
    info.update(source_smoke_ids=si,target_smoke_ids=ti);write_json(root/'data_inventory.json',info)
    state,sr=source_step(args,source.tasks,sb,scalers['source'].trainer_scalers(),info,root/'source_smoke.pt',device=device)
    rows=one_step(args,state,tb,scalers['target'].trainer_scalers(),root/'target',device=device)
    verify=verify_step_records(root/'target',rows)
    require(rows[1]['encoder_before']==rows[2]['encoder_before']==sr['encoder_after'],'source-to-target tensor chain')
    result=dict(task_id=TASK_ID,scope='ROUTE_A_SOURCE_AND_TARGET_ONE_STEP_SMOKE_ONLY',source=sr,
        target_runs=rows,saved_target_verification=verify,input_identity=table.identity,
        test_predictions_accessed=False,calibration_predictions_accessed=False,selection_performed=False,
        formal_source_reusable=False,formal_training_ready=False,acceptance_status='PENDING_CODEX_REVIEW')
    write_json(root/'smoke_result.json',result)
    write_json(root/'content_verification.json',verify_saved(root,info,scalers['target'].trainer_scalers()))
    return result
