"""P2 bounded epoch engine. No CLI or permission to launch an experiment.

The release layer must bind run identity, output ownership and global budget.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,random
import numpy as np
import torch
from p2_contract import require
from p2_data import digest,schedule
from p2_model import P2Optimization


def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(json.dumps([k,list(v.shape),str(v.dtype)],separators=(',',':')).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def rng_state(device):
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
                cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0) if device=='cuda:0' else None)


def restore_rng(state,device):
    require(set(state)=={'python','numpy','cpu','cuda'},'RNG schema')
    require((state['cuda'] is None)==(device=='cpu'),'RNG device')
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['cpu'].cpu())
    if device=='cuda:0':torch.cuda.set_rng_state(state['cuda'].cpu(),0)


def metrics(rows,tasks):
    require(rows and set(r['task'] for r in rows)==set(tasks),'metric task population')
    require(len({(r['task'],r['sample_id']) for r in rows})==len(rows),'duplicate metric observations')
    result={}
    for task in tasks:
        rr=[r for r in rows if r['task']==task]
        require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) for r in rr for k in ('label','prediction')),'nonfinite metrics')
        y=np.array([r['label'] for r in rr],dtype=np.float64);p=np.array([r['prediction'] for r in rr],dtype=np.float64)
        se=float(((p-y)**2).sum());sst=float(((y-y.mean())**2).sum())
        result[task]=dict(n=len(rr),rmse=math.sqrt(se/len(rr)),mae=float(np.abs(p-y).mean()),
                          r2=None if sst==0 else 1-se/sst,r2_reason='constant_labels' if sst==0 else None)
    return dict(endpoints=result,macro_rmse=math.fsum(v['rmse'] for v in result.values())/len(tasks))


class P2Engine:
    def __init__(self,model,view,*,method,seed,run_identity,device='cpu'):
        require(device in ('cpu','cuda:0'),'single device only')
        require(type(seed) is int and seed in range(42,47),'engine seed')
        self.model=model.to(device);self.view=view;self.device=device;self.seed=seed;self.method=method
        self.control=P2Optimization(model,method)
        self.initial=cpu_state(model)
        self.identity=dict(schema='p2_epoch_engine_v1',run=deepcopy(run_identity),method=method,seed=seed,
            view=view.identity,initial_model=state_digest(self.initial),device=device,torch_version=str(torch.__version__))
        self.identity_sha=digest(self.identity)
        self.tasks=view.tasks;self.counts={t:len(view.rows['train'][t]) for t in self.tasks}
        self.per_epoch=len(schedule(self.tasks,self.counts,seed,0))
        require((method=='SOURCE') == (not view.rows['validation']),'source/target validation permissions')
        self.epoch=0;self.offset=0;self.total_updates=0;self.history=[]
        self.running=[];self.best_epoch=None;self.best_state=None;self.best_rows=None
        random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
        if device=='cuda:0':torch.cuda.manual_seed(seed)

    def evaluate(self):
        require(self.method!='SOURCE','source cannot inspect target validation')
        was_training=self.model.training;self.model.eval();rows=[]
        try:
            with torch.no_grad():
                for task in self.tasks:
                    rr=self.view.rows['validation'][task]
                    for start in range(0,len(rr),64):
                        indices=list(range(start,min(start+64,len(rr))))
                        batch,selected=self.view.batch('validation',task,indices,self.device)
                        raw=self.model(batch,task_name=task)[task]
                        require(raw.shape==(len(indices),3) and torch.isfinite(raw).all().item(),'validation prediction shape/finite')
                        s=self.view.scalers[task];values=raw[:,0].detach().cpu().double().numpy()*s['std']+s['mean']
                        for record,pred in zip(selected,values):
                            rows.append(dict(task=task,sample_id=record['sample_id'],canonical=record['canonical'],group=record['group'],
                                             label=record['label'],prediction=float(pred),split='validation'))
        finally:self.model.train(was_training)
        return rows,metrics(rows,self.tasks)

    def step(self):
        require(self.epoch<40,'40-epoch budget exhausted')
        epoch=self.epoch;plan=schedule(self.tasks,self.counts,self.seed,epoch)
        if self.offset==0:self.control.begin_epoch(epoch)
        self.model.train();task,indices=plan[self.offset]
        batch,selected=self.view.batch('train',task,indices,self.device)
        require(len(selected)==len(indices),'training population')
        s=self.view.scalers[task];target=(batch.y-s['mean'])/s['std']
        require(target.shape==(len(indices),1) and torch.isfinite(target).all().item(),'training label shape/finite')
        self.control.optimizer.zero_grad(set_to_none=True)
        raw=self.model(batch,task_name=task)[task]
        require(raw.shape==(len(indices),3) and torch.isfinite(raw).all().item(),'training prediction shape/finite')
        from loss import QuantileRegressionLoss
        loss=QuantileRegressionLoss().compute_loss(raw,target)
        require(torch.isfinite(loss).item(),'nonfinite loss');loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        self.control.check();self.control.optimizer.step();self.control.check()
        self.running.append(dict(task=task,sample_ids=[r['sample_id'] for r in selected],loss=float(loss.detach()),grad_norm=float(norm)))
        self.offset+=1;self.total_updates+=1
        record=None
        if self.offset==len(plan):
            record=dict(epoch=epoch,total_updates=self.total_updates,train=deepcopy(self.running),
                        frozen=epoch<self.control.warmup,backbone_lr=self.control.backbone_lr,head_lr=.001)
            if self.method!='SOURCE':
                rows,score=self.evaluate();record['validation']=score
                record['validation_rows']=rows
                if self.best_epoch is None or score['macro_rmse']<self.history[self.best_epoch]['validation']['macro_rmse']:
                    self.best_epoch=epoch;self.best_state=cpu_state(self.model);self.best_rows=rows
            self.history.append(record);self.epoch+=1;self.offset=0;self.running=[]
        self.check_cursor_and_optimizer()
        return record

    def check_cursor_and_optimizer(self):
        require(type(self.epoch) is int and 0<=self.epoch<=40 and type(self.offset) is int and 0<=self.offset<self.per_epoch,'cursor types/range')
        require(self.epoch<40 or self.offset==0,'completed cursor')
        require(type(self.total_updates) is int and self.total_updates==self.epoch*self.per_epoch+self.offset,'update counter')
        require(len(self.history)==self.epoch and [h['epoch'] for h in self.history]==list(range(self.epoch)),'history cursor')
        require(len(self.running)==self.offset,'partial history cursor')
        if self.offset:
            for entry,(task,ix) in zip(self.running,schedule(self.tasks,self.counts,self.seed,self.epoch)):
                require(entry['task']==task and entry['sample_ids']==[self.view.rows['train'][task][i]['sample_id'] for i in ix],'partial history sample order')
                require(all(type(entry[k]) in (int,float) and math.isfinite(entry[k]) for k in ('loss','grad_norm')),'partial history finite')
        used=[]
        for e in range(self.epoch):used.extend((e,t,ix) for t,ix in schedule(self.tasks,self.counts,self.seed,e))
        if self.offset:used.extend((self.epoch,t,ix) for t,ix in schedule(self.tasks,self.counts,self.seed,self.epoch)[:self.offset])
        require(len(used)==self.total_updates,'schedule cursor')
        states=self.control.optimizer.state
        for name,p in self.control.backbone+self.control.heads:
            if name.startswith('decoders.'):
                task=name.split('.')[1];steps=sum(t==task for _,t,_ in used)
            else:
                steps=sum(e>=self.control.warmup for e,_,_ in used)
                if name.startswith('encoder.backbone.direct_bond_embeddings.'):
                    steps=0
                    require(torch.equal(p.detach().cpu(),self.initial[name]),'inactive edge embedding changed')
            state=states.get(p)
            if steps==0:require(not state,'unexpected Adam state before update')
            else:require(state is not None and float(state['step'])==steps,'Adam step counter mismatch: '+name)
        if self.total_updates:
            expected=self.epoch if self.offset else self.epoch-1
            require(self.control.epoch==expected,'optimizer epoch cursor');self.control.check()

    def snapshot(self):
        self.check_cursor_and_optimizer()
        state=cpu_state(self.model)
        return dict(schema='p2_checkpoint_v1',identity=deepcopy(self.identity),identity_sha=self.identity_sha,
            model_state=state,model_digest=state_digest(state),optimization=self.control.snapshot() if self.total_updates else None,
            epoch=self.epoch,offset=self.offset,total_updates=self.total_updates,history=deepcopy(self.history),running=deepcopy(self.running),
            best_epoch=self.best_epoch,best_state=deepcopy(self.best_state),best_rows=deepcopy(self.best_rows),
            best_digest=None if self.best_state is None else state_digest(self.best_state),rng=rng_state(self.device))

    def save(self,path):
        payload=self.snapshot();path=Path(path)
        with path.open('xb') as f:torch.save(payload,f)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def restore(self,path,expected_sha):
        require(self.total_updates==0 and self.control.epoch is None,'restore fresh engine only')
        raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==expected_sha,'checkpoint bytes')
        import io
        p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        require(p['schema']=='p2_checkpoint_v1' and p['identity']==self.identity and p['identity_sha']==self.identity_sha,'checkpoint identity')
        require(state_digest(p['model_state'])==p['model_digest'],'checkpoint model digest')
        self.model.load_state_dict(p['model_state'],strict=True)
        self.epoch=p['epoch'];self.offset=p['offset'];self.total_updates=p['total_updates'];self.history=p['history'];self.running=p['running']
        if self.total_updates:self.control.restore(p['optimization'],allow_partial_epoch=self.offset>0)
        else:
            require(p['optimization'] is None,'initial optimizer state')
            require(p['model_digest']==self.identity['initial_model'],'initial model changed')
        self.check_cursor_and_optimizer()
        for h in self.history:
            plan=schedule(self.tasks,self.counts,self.seed,h['epoch'])
            require(h['total_updates']==(h['epoch']+1)*self.per_epoch and len(h['train'])==len(plan),'history updates')
            require(h['frozen']==(h['epoch']<self.control.warmup),'history freeze')
            require(h['backbone_lr']==self.control.backbone_lr and h['head_lr']==.001,'history learning rates')
            for entry,(task,ix) in zip(h['train'],plan):
                require(entry['task']==task and entry['sample_ids']==[self.view.rows['train'][task][i]['sample_id'] for i in ix],'history sample order')
                require(all(type(entry[k]) in (int,float) and math.isfinite(entry[k]) for k in ('loss','grad_norm')),'history finite')
            if self.method!='SOURCE':
                score=h['validation'];endpoints=score['endpoints']
                require(set(endpoints)==set(self.tasks),'history validation tasks')
                require(all(type(endpoints[t]['n']) is int and endpoints[t]['n']==len(self.view.rows['validation'][t]) for t in self.tasks),'history validation population')
                require(all(type(endpoints[t]['rmse']) in (int,float) and math.isfinite(endpoints[t]['rmse']) and endpoints[t]['rmse']>=0 for t in self.tasks),'history validation finite')
                require(score['macro_rmse']==math.fsum(endpoints[t]['rmse'] for t in self.tasks)/len(self.tasks),'history validation macro')
        self.best_epoch=p['best_epoch'];self.best_state=p['best_state'];self.best_rows=p['best_rows']
        if self.method=='SOURCE' or not self.history:
            require(self.best_epoch is None and self.best_state is None and self.best_rows is None,'unexpected best state')
        else:
            best=min(range(len(self.history)),key=lambda e:self.history[e]['validation']['macro_rmse'])
            require(type(self.best_epoch) is int and self.best_epoch==best,'best selection')
            require(state_digest(self.best_state)==p['best_digest'],'best model digest')
            current=cpu_state(self.model);self.model.load_state_dict(self.best_state,strict=True)
            rows,score=self.evaluate();self.model.load_state_dict(current,strict=True)
            require(rows==self.best_rows and score==self.history[best]['validation'],'best checkpoint replay')
        restore_rng(p['rng'],self.device)
