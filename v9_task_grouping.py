"""Train-only task grouping for the bounded V9-S4 graph screen.

TGLoRA Eq. (1)/(2) with a bounded gradient probe and agglomerative search.
This is a graph adaptation, not a reproduction of the vision benchmark.
"""
import math
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from v9_cost_probe import require, digest
from v9_c2a_source_stats import targets

METHODS = ('SHARED_LORA', 'TG_LORA_GRAPH', 'SGTA')
SPEC = dict(rank=8, scale=1., adapter_dropout=0., shallow_fraction=.5,
            max_groups=2, probe_groups_per_task=8, min_reliable_groups=4,
            bootstrap_replicates=256, bootstrap_seed=420602,
            reliability_quantile=.95, gradient_scope='last_ffn_output_weight',
            probe_sampling='one_hash_selected_molecule_per_hash_selected_train_scaffold',
            similarity='TGLoRA_coordinate_normalized_example_pair_cosine',
            group_search='TGLoRA_partition_score_greedy_merging_from_singletons',
            reliability='upper_bootstrap_similarity_or_one_if_insufficient_groups')


def probe_selection(rows, tasks, spec=SPEC):
    require(rows and len(tasks) == len(set(tasks)), 'probe population')
    by = {t:{} for t in tasks}; seen = set()
    for row in rows:
        require(row['split'] == 'train' and row['task'] in by and
                all(type(row[k]) is str and row[k] for k in ('sample_id','canonical','group'))
                and math.isfinite(row['label']), 'train-only probe member')
        key = row['task'],row['sample_id']; require(key not in seen, 'probe duplicate'); seen.add(key)
        by[row['task']].setdefault(row['group'], []).append(row)
    selected = {}
    for task in tasks:
        require(by[task], 'probe empty task')
        groups = sorted(by[task], key=lambda g:(digest(dict(seed=42,group=g)),g))[:spec['probe_groups_per_task']]
        selected[task] = [min(by[task][g], key=lambda r:(digest(dict(seed=42,canonical=r['canonical'])),r['sample_id'])) for g in groups]
    return selected


def pair_cosines(x, y):
    """Paper Eq. (1), with zero/zero coordinates defined as zero."""
    require(x.ndim == y.ndim == 2 and x.shape[1] == y.shape[1] and
            len(x) and len(y) and np.isfinite(x).all() and np.isfinite(y).all(), 'gradient pair')
    den = np.abs(x[:,None,:]) + np.abs(y[None,:,:])
    a = np.divide(x[:,None,:],den,out=np.zeros_like(den),where=den>0)
    b = np.divide(y[None,:,:],den,out=np.zeros_like(den),where=den>0)
    ab = (a*b).sum(-1,dtype=np.float64)
    norm = np.sqrt((a*a).sum(-1,dtype=np.float64)*(b*b).sum(-1,dtype=np.float64))
    return np.divide(ab,norm,out=np.zeros_like(ab),where=norm>0).clip(-1,1)


def similarities(gradients, selection, tasks, spec=SPEC):
    require(set(gradients) == set(selection) == set(tasks), 'gradient task identity')
    vectors = {}
    for t in tasks:
        x = gradients[t]
        require(isinstance(x,torch.Tensor) and x.device.type == 'cpu' and x.dtype == torch.float32 and
                x.ndim == 2 and x.shape[0] == len(selection[t]) and torch.isfinite(x).all(), 'gradient probe tensor')
        require(len({r['group'] for r in selection[t]}) == len(selection[t]), 'one probe per scaffold')
        vectors[t] = x.numpy()
    require(len({x.shape[1] for x in vectors.values()}) == 1, 'gradient dimension')
    n = len(tasks); mean = np.eye(n); upper = np.eye(n); raw = []
    for i,t in enumerate(tasks):
        for j in range(i+1,n):
            u = tasks[j]; scores = pair_cosines(vectors[t],vectors[u]); mu = float(scores.mean())
            # A shared scaffold gets one common bootstrap count on both tasks.
            union = sorted({r['group'] for r in selection[t]+selection[u]})
            rng = np.random.default_rng(spec['bootstrap_seed']+i*n+j)
            weights = rng.multinomial(len(union),np.full(len(union),1/len(union)),size=spec['bootstrap_replicates'])
            left = weights[:,[union.index(r['group']) for r in selection[t]]]
            right = weights[:,[union.index(r['group']) for r in selection[u]]]
            den = left.sum(1)*right.sum(1); good = den>0
            values = np.einsum('bi,ij,bj->b',left,scores,right)[good]/den[good]
            require(len(values)>0, 'empty relation bootstrap')
            hi = float(np.quantile(values,spec['reliability_quantile']))
            reliable = min(len(selection[t]),len(selection[u])) >= spec['min_reliable_groups']
            conservative = max(mu,hi) if reliable else 1.
            mean[i,j] = mean[j,i] = mu; upper[i,j] = upper[j,i] = conservative
            raw.append(dict(left=t,right=u,mean=mu,upper=hi,used=len(values),sufficient_groups=reliable))
    return dict(tasks=tasks,mean=mean.tolist(),conservative=upper.tolist(),pairs=raw)


def partition_score(matrix, groups):
    return math.fsum(float(matrix[np.ix_(g,g)].sum()-np.trace(matrix[np.ix_(g,g)]))/(len(g)-1)
                     for g in groups if len(g)>1)


def greedy_groups(matrix, count=2):
    """Eq. (2) merging; negative scores are valid, ties are deterministic."""
    matrix=np.asarray(matrix,dtype=np.float64);n=len(matrix)
    require(matrix.shape==(n,n) and n>=count>=1 and np.isfinite(matrix).all() and
            np.allclose(matrix,matrix.T), 'group similarity matrix')
    groups=[(i,) for i in range(n)]
    while len(groups)>count:
        current=[partition_score(matrix,[g]) for g in groups]
        best=None
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                merged=tuple(sorted(groups[i]+groups[j]))
                gain=partition_score(matrix,[merged])-current[i]-current[j]
                candidate=(gain,-i,-j)
                if best is None or candidate>best[0]:best=(candidate,i,j,merged)
        _,i,j,merged=best
        groups=sorted([g for k,g in enumerate(groups) if k not in (i,j)]+[merged])
    return [list(g) for g in groups]


def choose_groups(method, relations):
    require(method in METHODS, 'group method'); tasks=relations['tasks'];n=len(tasks)
    shared=[list(range(n))]
    if method=='SHARED_LORA':return [tasks.copy()],dict(reason='shared_control',split=False)
    field='mean' if method=='TG_LORA_GRAPH' else 'conservative'
    matrix=np.asarray(relations[field]);groups=greedy_groups(matrix,min(2,n))
    gain=partition_score(matrix,groups)-partition_score(matrix,shared)
    between=float(matrix[np.ix_(groups[0],groups[1])].mean()) if len(groups)==2 else 0.
    split=method=='TG_LORA_GRAPH' or (gain>0 and between<0)
    selected=groups if split else shared
    return [[tasks[i] for i in g] for g in selected],dict(reason=field,split=split,score_gain=gain,between=between)


def ranks_for(groups, rank=8):
    require(groups and len(groups)<=rank and all(groups), 'rank groups')
    # Largest-remainder proportional allocation with one rank guaranteed per group.
    desired=np.asarray([len(g) for g in groups],dtype=float)*rank/sum(map(len,groups))
    ranks=np.maximum(1,np.floor(desired).astype(int))
    while ranks.sum()<rank:
        i=max(range(len(groups)),key=lambda i:(desired[i]-ranks[i],-i));ranks[i]+=1
    while ranks.sum()>rank:
        i=max((i for i in range(len(groups)) if ranks[i]>1),key=lambda i:(ranks[i]-desired[i],-i));ranks[i]-=1
    return ranks.tolist()


class Residual(nn.Module):
    def __init__(self, linear, ranks):
        super().__init__()
        template=torch.empty(sum(ranks),linear.in_features);nn.init.kaiming_uniform_(template,a=math.sqrt(5))
        self.A=nn.ParameterList();self.B=nn.ParameterList();start=0
        for rank in ranks:
            self.A.append(nn.Parameter(template[start:start+rank].clone()))
            self.B.append(nn.Parameter(torch.zeros(linear.out_features,rank)));start+=rank

    def forward(self, x, group):
        return F.linear(F.linear(x,self.A[group]),self.B[group])


class GroupedGraph(nn.Module):
    def __init__(self, base, source, method):
        super().__init__();require(method in METHODS,'grouped model method');self.method=method
        self.encoder=deepcopy(base.encoder).requires_grad_(False).eval()
        self.decoders=deepcopy(base.decoders);self.decoders.update(deepcopy(source.heads))
        self.task_name=list(base.task_name)+list(source.tasks);self.task_num=len(self.task_name)
        require(len(self.task_name)==len(set(self.task_name)), 'duplicate model task')
        self._active=None;self._handles=[];self.target=nn.ModuleDict();self.groups=None
        self.configure([self.task_name.copy()])
        for name,module in targets(self.encoder).items():
            key=name.replace('.','__');layer=int(name.split('.')[1])
            def hook(mod,args,result,key=key,layer=layer):
                require(self._active in self.task_name,'explicit graph task required')
                group=0 if layer<len(self.encoder.backbone.layers)//2 else self._assignment[self._active]
                return result+self.target[key](args[0],group)
            self._handles.append(module.register_forward_hook(hook))

    def configure(self, groups):
        flattened=sum(groups,[])
        require(len(groups) in (1,2) and len(flattened)==len(set(flattened)) and
                set(flattened)==set(self.task_name) and all(groups), 'complete task partition')
        require(self._active is None,'cannot repartition during forward')
        self.groups=deepcopy(groups);self._assignment={t:i for i,g in enumerate(groups) for t in g}
        new=nn.ModuleDict();layers=len(self.encoder.backbone.layers)
        with torch.random.fork_rng(devices=[]):
            # Only CPU matrices are initialized here; do not reseed CUDA dropout.
            torch.random.default_generator.manual_seed(42)
            for name,module in targets(self.encoder).items():
                ranks=[8] if int(name.split('.')[1])<layers//2 else ranks_for(groups)
                new[name.replace('.','__')]=Residual(module,ranks)
        self.target=new.to(next(self.encoder.parameters()).device)

    def train(self, mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self, inputs, task_name=None, return_aux=False):
        require(task_name in self.task_name and self._active is None,'explicit task/no nested call')
        require(not self.encoder.training and all(not p.requires_grad for p in self.encoder.parameters()),'frozen encoder')
        self._active=task_name
        try:representation=self.encoder(inputs)
        finally:self._active=None
        raw=self.decoders[task_name](representation);out={task_name:raw}
        if not return_aux:return out
        ones=raw.new_ones((len(raw),1))
        return out,dict(final_raw=raw,base_raw=raw,route_raw=raw,final_representation=representation,
                        base_representation=representation,route_representation=representation,null_weight=ones)


def optimizer_for(model):
    return torch.optim.AdamW([dict(params=list(model.target.parameters()),lr=.0001,scope='adaptation'),
                             dict(params=list(model.decoders.parameters()),lr=.001,scope='heads')],weight_decay=.00001)


def repartition(model, optimizer, groups):
    require(all(p not in optimizer.state for p in model.target.parameters()),'adapter optimizer already active')
    require(all(torch.count_nonzero(p)==0 for n,p in model.target.named_parameters() if '.B.' in n),'adapter already changed')
    model.configure(groups)
    optimizer.param_groups[0]['params']=list(model.target.parameters())
    # The head parameters and all of their Adam moments/step counters are retained.


def begin_epoch(model, epoch, warmup_epochs):
    model.train();model.encoder.requires_grad_(False)
    model.decoders.requires_grad_(True);model.target.requires_grad_(epoch>=warmup_epochs)
    return 'HEAD_WARMUP' if epoch<warmup_epochs else 'GROUPED_ADAPTATION'
