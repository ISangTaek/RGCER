"""P2 paired model initialization and optimizer primitives; no experiment CLI."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from torch import nn
from p2_contract import METHODS, TASKS, require


def build_model(role, seed, source_encoder=None):
    require(role in ('source', 'target'), 'P2 model role')
    require(type(seed) is int and seed in range(42, 47), 'P2 seed')
    require((role == 'source') == (source_encoder is None), 'source encoder role')
    from architecture.Graphormer import Encoder, Graphormer
    from architecture.prediction_heads import TaskPredictionHead
    tasks = ('source',) if role == 'source' else TASKS
    args = SimpleNamespace(a_layers=8,a_heads=4,hidden_dim=96,mid_dim=128,
                           edge_bias_mode='path',spatial_pos_clip=20,card_lambda_delta=0.)
    # Head construction is independent of pretrained weights or route names.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        decoders = nn.ModuleDict({t:TaskPredictionHead(96, mode='quantile',head_hidden_dim=96,dropout=.1) for t in tasks})
        model = Graphormer(list(tasks),Encoder,decoders,torch.device('cpu'),args)
    require(model.card is None, 'P2 forbids CARD')
    require(isinstance(model.encoder.readout,nn.Identity), 'P2 readout must be parameterless')
    if source_encoder is not None:
        expected = model.encoder.state_dict()
        require(set(source_encoder) == set(expected), 'source encoder keys')
        for key, value in source_encoder.items():
            require(isinstance(value,torch.Tensor) and value.shape == expected[key].shape
                    and value.dtype == expected[key].dtype and torch.isfinite(value).all().item(), 'source tensor: '+key)
        model.encoder.load_state_dict(source_encoder,strict=True)
    return model


class P2Optimization:
    """Continuous head Adam, fixed LR, frozen backbone has no state/decay.

    The future runner owns data ordering, model modes, RNG and update budgets.
    This component alone is deliberately not a ready-to-run training loop.
    """
    def __init__(self,model,method):
        require(method in ('SOURCE',)+METHODS,'P2 method')
        self.model,self.method=model,method
        self.warmup={'SOURCE':0,'FROZEN':40,'B1_low':0,'HF_low':5}[method]
        self.backbone_lr=.001 if method=='SOURCE' else .0001
        named=list(model.named_parameters())
        require(named and all(p.requires_grad for _,p in named),'start from unfrozen initial model')
        require(all(n.startswith(('encoder.backbone.','decoders.')) for n,_ in named),'unexpected parameter scope')
        self.backbone=[(n,p) for n,p in named if n.startswith('encoder.backbone.')]
        self.heads=[(n,p) for n,p in named if n.startswith('decoders.')]
        require(self.backbone and self.heads,'empty parameter scope')
        self.optimizer=torch.optim.AdamW([
            dict(params=[p for _,p in self.backbone],lr=self.backbone_lr),
            dict(params=[p for _,p in self.heads],lr=.001)],weight_decay=1e-5)
        self.groups=deepcopy(self.optimizer.state_dict()['param_groups'])
        self.initial={n:p.detach().cpu().clone() for n,p in self.backbone}
        self.epoch=None

    def begin_epoch(self,epoch):
        require(type(epoch) is int and 0<=epoch<40,'epoch range/type')
        require(self.epoch is None and epoch==0 or self.epoch is not None and epoch==self.epoch+1,'epoch sequence')
        self.epoch=epoch
        for _,p in self.backbone:
            p.requires_grad_(epoch>=self.warmup)
            if epoch<self.warmup:p.grad=None
        self.check()

    def check(self):
        require(self.epoch is not None,'epoch not started')
        require(self.optimizer.state_dict()['param_groups']==self.groups,'optimizer contract changed')
        require(all(p.requires_grad for _,p in self.heads),'head frozen')
        frozen=self.epoch<self.warmup
        for n,p in self.backbone:
            require(p.requires_grad != frozen,'backbone gradient mode')
            if frozen:
                require(p.grad is None,'frozen gradient leak')
                require(p not in self.optimizer.state or not self.optimizer.state[p],'frozen Adam state')
                require(torch.equal(p.detach().cpu(),self.initial[n]),'frozen backbone changed')
        for _,p in self.backbone+self.heads:
            require(torch.isfinite(p).all().item(),'nonfinite parameters')

    def snapshot(self):
        self.check()
        return dict(method=self.method,epoch=self.epoch,optimizer=deepcopy(self.optimizer.state_dict()))

    def restore(self,snapshot):
        """Optimizer-only restore; caller must restore and verify model/RNG too."""
        require(self.epoch is None and not self.optimizer.state,'restore only into fresh controller')
        require(set(snapshot)=={'method','epoch','optimizer'} and snapshot['method']==self.method,'restore identity')
        epoch=snapshot['epoch'];require(type(epoch) is int and 0<=epoch<40,'restore epoch')
        state=snapshot['optimizer'];require(set(state)=={'state','param_groups'},'optimizer schema')
        require(state['param_groups']==self.groups,'restored parameter groups')
        ids=[i for g in self.groups for i in g['params']]
        parameters=[p for _,p in self.backbone+self.heads]
        require(set(state['state']).issubset(ids),'unknown optimizer parameter')
        require(set(self.groups[1]['params']).issubset(state['state']),'missing completed-epoch head Adam state')
        for pid,item in state['state'].items():
            require(type(pid) is int and set(item)=={'step','exp_avg','exp_avg_sq'},'Adam state schema')
            p=parameters[ids.index(pid)]
            for key in ('exp_avg','exp_avg_sq'):
                value=item[key]
                require(isinstance(value,torch.Tensor) and value.shape==p.shape and value.dtype==p.dtype
                        and torch.isfinite(value).all().item(),'Adam moments')
            step=item['step']
            require(isinstance(step,torch.Tensor) and step.numel()==1 and torch.isfinite(step).all().item()
                    and float(step)>0 and float(step).is_integer(),'Adam step')
        if epoch<self.warmup:
            require(not set(self.groups[0]['params']).intersection(state['state']),'restored frozen Adam state')
        self.optimizer.load_state_dict(deepcopy(state));self.epoch=epoch
        for _,p in self.backbone:p.requires_grad_(epoch>=self.warmup)
        self.check()
