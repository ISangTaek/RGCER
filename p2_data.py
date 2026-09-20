"""Exact P2 task views over the existing read-only DataStore; no split mutation."""
import hashlib
import json
import math
import numpy as np
from p2_contract import PANELS, ROUTES, TASKS, require, validate_datastore


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fit_scaler(rows):
    require(rows and all(type(r['label']) in (int,float) and math.isfinite(r['label']) for r in rows),'finite training labels')
    values=np.asarray([r['label'] for r in rows],dtype=np.float64)
    mean=float(values.mean());raw_std=float(values.std(ddof=0))
    require(math.isfinite(mean) and math.isfinite(raw_std),'scaler overflow')
    return dict(mean=mean,std=1. if raw_std<1e-8 else raw_std,count=len(rows),
                std_reason='constant_or_near_constant' if raw_std<1e-8 else None)


def schedule(tasks,counts,seed,epoch):
    require(type(seed) is int and seed in range(42,47),'schedule seed')
    require(type(epoch) is int and 0<=epoch<40,'schedule epoch')
    require(set(tasks)==set(counts) and len(tasks)==len(set(tasks)),'schedule tasks')
    batches=[]
    for j,t in enumerate(tasks):
        require(type(counts[t]) is int and counts[t]>0,'schedule count')
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch,j])).permutation(counts[t]).tolist()
        batches.extend((t,order[i:i+64]) for i in range(0,len(order),64))
    permutation=np.random.default_rng(np.random.SeedSequence([seed,epoch,1000])).permutation(len(batches))
    return [batches[int(i)] for i in permutation]


class P2View:
    def __init__(self,store,train,validation):
        self.store=store
        self.rows={'train':train,'validation':validation}
        self.tasks=tuple(train)
        require(self.tasks and set(validation) in (set(),set(self.tasks)),'view task sets')
        self.scalers={t:fit_scaler(train[t]) for t in self.tasks}
        self.identity=digest(dict(train=train,validation=validation,scalers=self.scalers))

    def batch(self,split,task,indices,device):
        require(split in ('train','validation') and task in self.rows[split],'view split/task permission')
        rows=self.rows[split][task]
        require(indices and len(set(indices))==len(indices) and all(type(i) is int and 0<=i<len(rows) for i in indices),'view indices')
        from dataset import DataCollator
        selected=[rows[i] for i in indices]
        graphs=[self.store.get_graph_data(r['global_index'],task_name=task,label=r['label']) for r in selected]
        batch=DataCollator(spatial_pos_max_clip=20,max_node_filter=512)(graphs)
        require(not batch.is_empty and batch.sample_id==[r['sample_id'] for r in selected],'collator population changed')
        require(batch.y.numel()==len(indices),'batch label shape')
        return batch.to(device),selected


class P2Data:
    def __init__(self,members,datastore_build):
        self.receipt=validate_datastore(members,datastore_build)
        from toxacute_datastore import ToxAcuteDataStore
        self.store=ToxAcuteDataStore(datastore_build)
        self.members=members
        self.positions={str(s):i for i,s in enumerate(self.store.sample_ids)}

    def source(self,panel,route):
        require(panel in PANELS and route in ROUTES,'source panel/route')
        rows=[]
        for r in self.members['panels'][panel]:
            o=r['observations'][route]
            rows.append(dict(sample_id=o['sample_id'],global_index=o['global_index'],canonical=r['canonical'],
                             group=r['group'],label=o['value']))
        return P2View(self.store,{'source':rows},{})

    def target(self):
        splits={}
        for split in ('train','validation'):
            splits[split]={}
            for task in TASKS:
                rows=[]
                for r in self.members['target'][split][task]:
                    i=self.positions[r['sample_id']]
                    rows.append(dict(**r,global_index=i,label=float(self.store.get_label(i,task))))
                splits[split][task]=rows
        return P2View(self.store,splits['train'],splits['validation'])

    def close(self):
        self.store.close()
