"""Fixed-rank graph SSRA candidate and matched LoRA; not CorDA/CorDA++.

Adapters live outside the frozen encoder and modify linear outputs via hooks.
Thus encoder state hashes certify original weights, not an unchanged function.
"""
from copy import deepcopy
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import v9_c2a_source_stats as stats
from v9_cost_probe import require, digest
from v9_srgt import save_smoke, load_smoke as _load_smoke

METHODS=('LORA','SSRA')
SPEC=dict(schema='v9_ssra_v1',rank=8,free_rank=4,source_rank=4,projector_rank=16,
          beta='per_layer_per_task_sigmoid',beta_init=.1,beta_penalty=.01,
          source_penalty=.01,source_moment='empirical_uncentered_molecule_equal',
          penalty_normalization='mean_modules_traceM_times_output_dim',
          adapter_lr=.0001,head_lr=.001,weight_decay=.00001,grad_clip=1.,
          lora_scale=1.,adapter_dropout=0.,encoder_dropout='eval',head_dropout='inherited',
          adaptation_tokens='all_tokens_including_graph_token',statistics_tokens='atoms_only')


class Residual(nn.Module):
    def __init__(self,linear,method,stat):
        super().__init__();self.method=method
        d,o=linear.in_features,linear.out_features
        # Identical A draw and total A/B parameter counts for both arms.
        self.A=nn.Parameter(torch.empty(8,d));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self.B=nn.Parameter(torch.zeros(o,8))
        if method=='SSRA':
            p=stat['pooled']['projector'];n=sum(stat['count']);moment=stat['sum_second'].sum(0)/n
            require(p.shape==moment.shape==(d,d) and torch.isfinite(moment).all(),'source moment dimensions')
            require(torch.allclose(p,p.T,atol=1e-9) and torch.allclose(p@p,p,atol=1e-8),'source projector')
            require(torch.linalg.eigvalsh(moment).min()>=-1e-8 and torch.trace(moment)>0,'source second moment')
            self.register_buffer('P',p.float().clone());self.register_buffer('M',moment.float().clone())

    def delta(self,beta):
        if self.method=='LORA':return self.B@self.A
        ap=self.A@self.P
        return self.B[:,:4]@(self.A[:4]-ap[:4])+beta*(self.B[:,4:]@ap[4:])

    def penalty(self,beta):
        w=self.delta(beta)
        return ((w@self.M)*w).sum().clamp_min(0)/(torch.trace(self.M)*w.shape[0])


class AdaptedGraph(nn.Module):
    def __init__(self,base,method,counts,payload):
        super().__init__();require(method in METHODS,'adapter method');self.method=method
        self.task_name=list(base.task_name);require(set(counts)==set(self.task_name),'adapter tasks')
        self.encoder=deepcopy(base.encoder).requires_grad_(False).eval()
        self.decoders=deepcopy(base.decoders).requires_grad_(True)
        modules=stats.targets(self.encoder);require(set(modules)==set(payload['stats']),'adapter statistics modules')
        self.target=nn.ModuleDict();self.target.edge_bias_mode='direct_plus_path'
        self._active=None;self._handles=[];self._keys={}
        layers=len(self.encoder.backbone.layers)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            for name,module in modules.items():
                key=name.replace('.','__');self._keys[name]=key
                self.target[key]=Residual(module,method,payload['stats'][name])
            if method=='SSRA':
                # One dense parameter makes Adam's per-tensor update count explicit.
                self.target.register_parameter('beta_logits',nn.Parameter(torch.full((layers,len(self.task_name)),math.log(.1/.9))))
        for name,module in modules.items():
            layer=int(name.split('.')[1]);key=self._keys[name]
            def hook(mod,args,result,layer=layer,key=key):
                require(self._active is not None,'encoder may only run inside explicit task forward')
                beta=self.beta(layer,self._active)
                return result+F.linear(args[0],self.target[key].delta(beta))
            self._handles.append(module.register_forward_hook(hook))
        self.train()

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def beta(self,layer,task):
        if self.method=='LORA':return next(self.target.parameters()).new_tensor(1.)
        return torch.sigmoid(self.target.beta_logits[layer,self.task_name.index(task)])

    def regularization(self,task=None):
        require(task in self.task_name,'regularizer requires explicit task')
        if self.method=='LORA':return next(self.target.parameters()).new_zeros(())
        penalties=[self.target[key].penalty(self.beta(int(name.split('.')[1]),task)) for name,key in self._keys.items()]
        betas=torch.stack([self.beta(i,task) for i in range(len(self.encoder.backbone.layers))])
        return .01*torch.stack(penalties).mean()+.01*(betas-.1).square().mean()

    def forward(self,inputs,task_name=None,return_aux=False):
        require(task_name in self.task_name and self._active is None,'explicit task/no nested forward')
        require(not self.encoder.training and all(not p.requires_grad for p in self.encoder.parameters()),'frozen source scope')
        self._active=task_name
        try:representation=self.encoder(inputs)
        finally:self._active=None
        raw=self.decoders[task_name](representation);out={task_name:raw}
        if not return_aux:return out
        gates=torch.stack([self.beta(i,task_name) for i in range(len(self.encoder.backbone.layers))])
        return out,dict(final_raw=raw,base_raw=raw,route_raw=raw,final_representation=representation,gates=gates)


def optimizer_for(model):
    return torch.optim.AdamW([dict(params=list(model.target.parameters()),lr=.0001,scope='adaptation'),
        dict(params=list(model.decoders.parameters()),lr=.001,scope='heads')],weight_decay=.00001)


def load_smoke(path,model,optimizer,identity,updates):
    payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    # Source statistics are immutable, even if a forged checkpoint is self-consistent.
    for key,expected in model.named_buffers():
        actual=payload['model_state'].get(key)
        require(isinstance(actual,torch.Tensor) and torch.equal(actual.cpu(),expected.detach().cpu()),'protected source buffer '+key)
    _load_smoke(path,model,optimizer,identity,updates)
