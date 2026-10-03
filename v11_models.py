"""V11 chemical multi-view adaptations; exact deviations are in docs/V11_S1_METHODS.md.

Independent PyTorch implementations of the cited mathematical cores. These are
not claims to reproduce the authors' full benchmark or hyperparameter search.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

import v11_features as features
from v10_context_models import check_training_labels
from v10_functional_transfer import require

NEW=('REALMLP_MV','TABM_MV','FT_TRANSFORMER_MV','EXCELFORMER_MV','AMFORMER_MV',
     'T2G_FORMER_MV','MAMBULAR_MV','GRANDE_MV','MODERNNCA_MV','TCF')
CONTROLS=('MV_CONCAT_MLP','TCF_GRAPH_ONLY')
CONFIGS={
    'REALMLP_MV':dict(width=256,depth=3,lr=.07,decay=0.,betas=[.9,.95],schedule='coslog4'),
    'TABM_MV':dict(width=512,depth=3,k=32,dropout=.1,lr=.002,decay=3e-4),
    'FT_TRANSFORMER_MV':dict(width=192,depth=3,heads=8,lr=1e-4,decay=1e-5),
    'EXCELFORMER_MV':dict(width=192,depth=3,heads=8,lr=1e-4,decay=1e-5),
    'AMFORMER_MV':dict(width=192,depth=3,heads=8,groups=32,lr=1e-4,decay=1e-5),
    'T2G_FORMER_MV':dict(width=192,depth=3,heads=8,lr=1e-4,decay=1e-5),
    'MAMBULAR_MV':dict(width=64,depth=4,state=128,expand=2,conv=4,lr=1e-4,decay=1e-6),
    'GRANDE_MV':dict(depth=5,trees=1024,selected=50,dropout=.2,lr=.001,decay=0.),
    'MODERNNCA_MV':dict(width=128,depth=1,block=512,dropout=.1,sample_rate=.5,lr=.01,decay=2e-4),
    'TCF':dict(width=64,rank=4,lr=.001,decay=1e-4),
    'MV_CONCAT_MLP':dict(width=64,rank=4,lr=.001,decay=1e-4),
    'TCF_GRAPH_ONLY':dict(width=64,rank=4,lr=.001,decay=1e-4)}


def count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Inputs(nn.Module):
    def __init__(self,h,sources,metadata):
        super().__init__(); self.h=h; self.sources=tuple(sources); self.metadata=metadata
        self.fields=tuple(len(x) for x in metadata['vocabulary'].values())
        self.dims=(h+2*len(sources),2048,48,sum(self.fields))
        self.input_dim=sum(self.dims)

    def views(self,e,context=False):
        require(isinstance(e,features.Episode),'multi-view Episode required')
        b=e.base; b.validate(self.h,len(self.sources),self.dims[3])
        x,p=(b.context_features,b.context_functions) if context else (b.query_features,b.query_functions)
        fp,d=(e.context_fp,e.context_descriptors) if context else (e.query_fp,e.query_descriptors)
        require(fp.shape==(len(x),2048) and d.shape==(len(x),48) and fp.dtype==d.dtype==x.dtype
                and fp.device==d.device==x.device and bool(torch.isfinite(fp).all())
                and bool(torch.isfinite(d).all()) and bool(((fp==0)|(fp==1)).all()),'multi-view tensors')
        mask=torch.ones_like(p)
        if b.task in self.sources: mask[:,self.sources.index(b.task)]=0
        return torch.cat((x,p*mask,mask),-1),fp,d,b.metadata.expand(len(x),-1)

    def flat(self,e,context=False): return torch.cat(self.views(e,context),-1)

    def loss(self,e,labels):
        check_training_labels(e.base,labels)
        return F.mse_loss(self(e),labels)


class NTKLinear(nn.Module):
    def __init__(self,a,b,zero=False):
        super().__init__(); self.weight=nn.Parameter(torch.randn(b,a)); self.bias=nn.Parameter(torch.randn(b))
        if zero: nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)

    def forward(self,x): return F.linear(x,self.weight/math.sqrt(self.weight.shape[1]),self.bias)


class RealMLP(Inputs):
    def __init__(self,*args,config,center,factors):
        super().__init__(*args); w=config['width']
        self.register_buffer('center',center.clone()); self.register_buffer('factors',factors.clone())
        self.front_scale=nn.Parameter(torch.ones(self.input_dim))
        self.layers=nn.ModuleList([NTKLinear(a,w) for a in [self.input_dim]+[w]*(config['depth']-1)])
        self.output=NTKLinear(w,1,zero=True)

    def forward(self,e):
        x=(self.flat(e)-self.center)*self.factors
        x=(x/torch.sqrt(1+x.square()/9))*self.front_scale
        for layer in self.layers: x=F.mish(layer(x))
        return self.output(x)


class BatchEnsembleLinear(nn.Module):
    def __init__(self,a,b,k,first=False):
        super().__init__(); self.weight=nn.Parameter(torch.empty(b,a)); nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        self.r=nn.Parameter(torch.ones(k,a)); self.s=nn.Parameter(torch.ones(k,b))
        self.bias=nn.Parameter(torch.empty(k,b)); nn.init.uniform_(self.bias,-1/math.sqrt(a),1/math.sqrt(a))
        if first:
            with torch.no_grad(): self.r.bernoulli_(.5).mul_(2).sub_(1)

    def forward(self,x): return F.linear(x*self.r,self.weight)*self.s+self.bias


class TabM(Inputs):
    def __init__(self,*args,config):
        super().__init__(*args); self.k=config['k']; w=config['width']; self.drop=config['dropout']
        self.layers=nn.ModuleList([BatchEnsembleLinear(a,w,self.k,i==0)
            for i,a in enumerate([self.input_dim]+[w]*(config['depth']-1))])
        self.output_weight=nn.Parameter(torch.empty(self.k,w,1)); nn.init.uniform_(self.output_weight,-1/math.sqrt(w),1/math.sqrt(w))
        self.output_bias=nn.Parameter(torch.zeros(self.k,1))

    def members(self,e):
        x=self.flat(e)[:,None,:].expand(-1,self.k,-1)
        for layer in self.layers: x=F.dropout(F.relu(layer(x)),self.drop,self.training)
        return torch.einsum('bki,kio->bko',x,self.output_weight)+self.output_bias

    def forward(self,e): return self.members(e).mean(1)

    def loss(self,e,labels):
        check_training_labels(e.base,labels); p=self.members(e)
        return (p-labels[:,None,:]).square().mean()


class BlockTokenizer(nn.Module):
    """No bit dropping: h/F/mask blocks16, Morgan blocks64, paired D+mask, task fields."""
    def __init__(self,h,ns,fields,width,gated=False):
        super().__init__(); sizes=[]; self.parts=[]
        start=0
        for length,block in ((h,16),(ns,16),(ns,16),(2048,64)):
            for a in range(0,length,block):
                ids=list(range(start+a,start+min(length,a+block))); self.parts.append(ids); sizes.append(len(ids))
            start+=length
        for i in range(24): self.parts.append([start+i,start+24+i]); sizes.append(2)
        start+=48
        for n in fields: self.parts.append(list(range(start,start+n))); sizes.append(n); start+=n
        self.projections=nn.ModuleList(nn.Linear(n,width) for n in sizes)
        self.gates=nn.ModuleList(nn.Linear(n,width) for n in sizes) if gated else None
        self.n_tokens=len(sizes)

    def forward(self,x):
        out=[p(x[:,ids]) for p,ids in zip(self.projections,self.parts)]
        if self.gates is not None:
            out=[a*torch.tanh(p(x[:,ids])) for a,p,ids in zip(out,self.gates,self.parts)]
        return torch.stack(out,1)


class Attention(nn.Module):
    def __init__(self,d,heads,tokens,kind):
        super().__init__(); self.heads=heads; self.kind=kind; self.dh=d//heads
        self.q=nn.Linear(d,d); self.k=None if kind=='t2g' else nn.Linear(d,d)
        self.v=nn.Linear(d,d); self.out=nn.Linear(d,d)
        for layer in (self.q,self.k,self.v,self.out):
            if layer is not None: nn.init.zeros_(layer.bias)
        if kind=='excel':
            for layer in (self.q,self.k,self.v):
                with torch.no_grad(): layer.weight.mul_(.1); layer.bias.zero_()
        if kind=='t2g':
            self.relation=nn.Parameter(torch.ones(heads,self.dh))
            self.columns=nn.Parameter(torch.randn(heads,tokens,math.ceil(2*math.log2(tokens))))
            self.column_tail=nn.Parameter(torch.empty_like(self.columns))
            for p in (self.columns,self.column_tail): nn.init.kaiming_uniform_(p,a=math.sqrt(5))
            self.edge_bias=nn.Parameter(torch.zeros(()))
            for layer in (self.q,self.v,self.out): nn.init.zeros_(layer.bias)

    def adjacency(self):
        c=F.normalize(self.columns,dim=-1); tail=F.normalize(self.column_tail,dim=-1)
        p=(c@tail.transpose(-1,-2)+self.edge_bias).sigmoid()
        a=(p>.5).to(p.dtype)-p.detach()+p
        mask=~torch.eye(p.shape[-1],dtype=torch.bool,device=p.device); mask[:,0]=False
        return a*mask

    def forward(self,x):
        def heads(y): return y.reshape(len(x),x.shape[1],self.heads,self.dh).transpose(1,2)
        q=heads(self.q(x)); k=heads((self.k or self.q)(x)); v=heads(self.v(x))
        if self.kind=='t2g': q=q*self.relation[None,:,None,:]
        score=q@k.transpose(-1,-2)/math.sqrt(self.dh)
        if self.kind=='t2g': score=score+(1-self.adjacency()[None])*(-1e4)
        if self.kind=='excel':
            mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
            score=score.masked_fill(mask,-1e4)
        a=F.dropout(score.softmax(-1),.2,self.training)
        return self.out((a@v).transpose(1,2).reshape_as(x))


class TransformerBlock(nn.Module):
    def __init__(self,d,heads,tokens,kind,first):
        super().__init__(); self.kind=kind; self.first=first
        self.norm1=nn.Identity() if first else nn.LayerNorm(d); self.norm2=nn.LayerNorm(d)
        self.attn=Attention(d,heads,tokens,kind); middle=d if kind=='excel' else int(d*4/3)
        self.ff1=nn.Linear(d,2*middle); self.ff2=nn.Linear(middle,d)

    def forward(self,x):
        x=x+self.attn(self.norm1(x)); a,b=self.ff1(self.norm2(x)).chunk(2,-1)
        y=a*(torch.tanh(b) if self.kind=='excel' else F.relu(b))
        return x+self.ff2(F.dropout(y,.1,self.training))


class TokenTransformer(Inputs):
    def __init__(self,*args,config,kind):
        super().__init__(*args); d=config['width']; self.kind=kind
        self.tokens=BlockTokenizer(self.h,len(self.sources),self.fields,d,kind=='excel')
        if kind=='excel': self.pool=nn.Linear(self.tokens.n_tokens,1)
        else: self.cls=nn.Parameter(torch.empty(1,1,d)); nn.init.uniform_(self.cls,-1/math.sqrt(d),1/math.sqrt(d))
        n=self.tokens.n_tokens+(kind!='excel')
        self.blocks=nn.ModuleList(TransformerBlock(d,config['heads'],n,kind,i==0) for i in range(config['depth']))
        self.head=nn.Sequential(nn.LayerNorm(d),nn.PReLU() if kind=='excel' else nn.ReLU(),nn.Linear(d,1))

    def forward(self,e):
        x=self.tokens(self.flat(e))
        if self.kind!='excel': x=torch.cat((self.cls.expand(len(x),-1,-1),x),1)
        for block in self.blocks: x=block(x)
        x=self.pool(x.transpose(1,2)).squeeze(-1) if self.kind=='excel' else x[:,0]
        return self.head(x)


class PromptAttention(nn.Module):
    """AMFormer dense prompt option; per-row product rescale avoids batch coupling."""
    def __init__(self,d,heads,groups,product):
        super().__init__(); self.product=product; self.heads=heads; self.dh=d//heads
        self.prompt=nn.Parameter(torch.randn(groups,d)); self.q=nn.Linear(d,d,bias=False)
        self.k=nn.Linear(d,d,bias=False); self.v=nn.Linear(d,d,bias=False); self.out=nn.Linear(d,d)

    def forward(self,x):
        if self.product: x=torch.log1p(F.relu(x))
        q=self.q(self.prompt).view(-1,self.heads,self.dh).transpose(0,1)
        def heads(z): return z.view(len(x),x.shape[1],self.heads,self.dh).transpose(1,2)
        k,v=heads(self.k(x)),heads(self.v(x)); a=(q[None]@k.transpose(-1,-2)/math.sqrt(self.dh)).softmax(-1)
        y=(F.dropout(a,.1,self.training)@v).transpose(1,2).reshape(len(x),-1,self.heads*self.dh)
        if self.product:
            lo=y.amin((1,2),keepdim=True); hi=y.amax((1,2),keepdim=True)
            y=torch.exp((y-lo)/(hi-lo).clamp_min(1e-6))
        return self.out(y)


class AMBlock(nn.Module):
    def __init__(self,d,heads,n,groups):
        super().__init__(); self.norm=nn.LayerNorm(d)
        self.sum=PromptAttention(d,heads,groups,False); self.product=PromptAttention(d,heads,groups,True)
        self.pool=nn.Linear(2*groups,groups); self.skip=nn.Linear(n,groups)
        self.ffnorm=nn.LayerNorm(d); self.ff1=nn.Linear(d,8*d); self.ff2=nn.Linear(4*d,d)

    def forward(self,x):
        y=self.norm(x)
        y=self.pool(torch.cat((self.sum(y),self.product(y)),1).transpose(1,2)).transpose(1,2)
        x=self.skip(x.transpose(1,2)).transpose(1,2)+y
        a,b=self.ff1(self.ffnorm(x)).chunk(2,-1)
        return x+self.ff2(F.dropout(a*F.gelu(b),.1,self.training))


class AMFormer(Inputs):
    def __init__(self,*args,config):
        super().__init__(*args); d=config['width']; g=config['groups']
        self.tokens=BlockTokenizer(self.h,len(self.sources),self.fields,d)
        self.blocks=nn.ModuleList(AMBlock(d,config['heads'],self.tokens.n_tokens if i==0 else g,g) for i in range(config['depth']))
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Flatten(),nn.Linear(g*d,d),nn.ReLU(),nn.Linear(d,1))

    def forward(self,e):
        x=self.tokens(self.flat(e))
        for block in self.blocks: x=block(x)
        return self.head(x)


class SelectiveSSM(nn.Module):
    def __init__(self,d,state=128,expand=2,conv=4):
        super().__init__(); self.inner=d*expand; self.rank=math.ceil(d/16); self.state=state
        self.input=nn.Linear(d,2*self.inner,bias=False)
        self.conv=nn.Conv1d(self.inner,self.inner,conv,padding=conv-1,groups=self.inner,bias=False)
        self.select=nn.Linear(self.inner,self.rank+2*state,bias=False); self.dt=nn.Linear(self.rank,self.inner)
        nn.init.uniform_(self.dt.weight,-self.rank**-.5,self.rank**-.5)
        with torch.no_grad():
            dt=torch.exp(torch.rand(self.inner)*(math.log(.1)-math.log(1e-4))+math.log(1e-4)).clamp_min(1e-4)
            self.dt.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.A_log=nn.Parameter(torch.arange(1,state+1,dtype=torch.float32).log().repeat(self.inner,1))
        self.D=nn.Parameter(torch.ones(self.inner)); self.output=nn.Linear(self.inner,d,bias=False)

    def forward(self,x):
        v,z=self.input(x).chunk(2,-1); v=F.silu(self.conv(v.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        dt,b,c=torch.split(self.select(v),(self.rank,self.state,self.state),-1)
        dt=F.softplus(self.dt(dt)); a=-self.A_log.exp(); h=v.new_zeros(len(x),self.inner,self.state); ys=[]
        for i in range(x.shape[1]):
            h=torch.exp(dt[:,i,:,None]*a)*h+dt[:,i,:,None]*b[:,i,None,:]*v[:,i,:,None]
            ys.append((h*c[:,i,None,:]).sum(-1)+self.D*v[:,i])
        return self.output(torch.stack(ys,1)*F.silu(z))


class Mambular(Inputs):
    def __init__(self,*args,config):
        super().__init__(*args); d=config['width']
        self.tokens=BlockTokenizer(self.h,len(self.sources),self.fields,d)
        self.norms=nn.ModuleList(nn.RMSNorm(d,eps=1e-5) for _ in range(config['depth']))
        self.blocks=nn.ModuleList(SelectiveSSM(d,config['state'],config['expand'],config['conv']) for _ in self.norms)
        self.head=nn.Linear(d,1)

    def forward(self,e):
        x=self.tokens(self.flat(e))
        for norm,block in zip(self.norms,self.blocks): x=x+block(norm(x))
        return self.head(x.mean(1))


class Grande(Inputs):
    def __init__(self,*args,config):
        super().__init__(*args); t=config['trees']; depth=config['depth']; n=2**depth-1; leaves=2**depth
        f=min(self.input_dim,config['selected']); self.drop=config['dropout']
        gen=torch.Generator().manual_seed(42)
        self.register_buffer('columns',torch.stack([torch.randperm(self.input_dim,generator=gen)[:f] for _ in range(t)]))
        self.thresholds=nn.Parameter(torch.randn(t,n,f)*.05); self.selectors=nn.Parameter(torch.randn(t,n,f)*.05)
        self.leaves=nn.Parameter(torch.randn(t,leaves)*.05); self.weights=nn.Parameter(torch.randn(t,leaves)*.05)
        paths=[]; directions=[]
        for leaf in range(leaves):
            node=0; path=[]; bits=[]
            for level in range(depth):
                bit=(leaf>>(depth-1-level))&1; path.append(node); bits.append(bit); node=2*node+1+bit
            paths.append(path); directions.append(bits)
        self.register_buffer('paths',torch.tensor(paths)); self.register_buffer('directions',torch.tensor(directions,dtype=torch.float32))

    def forward(self,e):
        x=self.flat(e)[:,self.columns]
        prob=self.selectors.softmax(-1); one=F.one_hot(prob.argmax(-1),prob.shape[-1]).to(prob)
        selector=one-prob.detach()+prob
        threshold=(self.thresholds*selector).sum(-1); value=torch.einsum('btf,tnf->btn',x,selector)
        soft=(F.softsign(threshold-value)+1)/2
        left=soft.round()-soft.detach()+soft
        p=left[:,:,self.paths]; d=self.directions
        leaf_prob=(p*(1-d)+(1-p)*d).prod(-1)
        weight=(leaf_prob*self.weights).sum(-1).softmax(-1)
        if self.training:
            keep=(torch.rand_like(weight)>=self.drop).to(weight)
            # Keep the top expert only if every expert was dropped (tiny-test case).
            keep=keep+F.one_hot(weight.argmax(-1),weight.shape[-1])*(keep.sum(-1,keepdim=True)==0)
            weight=weight*keep; weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)
        return (weight*(leaf_prob*self.leaves).sum(-1)).sum(-1,keepdim=True)


class ModernNCA(Inputs):
    def __init__(self,*args,config):
        super().__init__(*args); d=config['width']; self.rate=config['sample_rate']
        self.encoder=nn.Linear(self.input_dim,d); layers=[]
        for _ in range(config['depth']):
            layers.extend((nn.BatchNorm1d(d),nn.Linear(d,config['block']),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(config['block'],d)))
        layers.append(nn.BatchNorm1d(d)); self.post=nn.Sequential(*layers)

    def forward(self,e):
        c,q=self.flat(e,True),self.flat(e); labels=e.base.context_labels
        if self.training:
            keep=torch.randperm(len(c),device=c.device)[:max(2,math.ceil(len(c)*self.rate))]
            c,labels=c[keep],labels[keep]
        joined=self.post(self.encoder(torch.cat((c,q)))); c,q=joined[:len(c)],joined[len(c):]
        return (-torch.cdist(q,c,p=2)).softmax(-1)@labels


def encoder(a,w): return nn.Sequential(nn.Linear(a,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU())


class TCF(Inputs):
    def __init__(self,*args,config,graph_only=False):
        super().__init__(*args); w=config['width']; r=config['rank']; self.graph_only=graph_only; self.rank=r
        self.graph=encoder(self.dims[0],w); self.task=encoder(self.dims[3],w)
        if not graph_only:
            self.fp=encoder(2048,w); self.desc=encoder(48,w)
            self.factors=nn.ModuleList(nn.ModuleList(nn.Linear(w,r*w,bias=False) for _ in range(3)) for _ in range(3))
            read_width=w; inp=7*w
        else:
            goal=tcf_count(self.dims,w,r); base=count(self); inp=2*w
            read_width=max(1,round((goal-base-1)/(inp+2)))
        self.readout=nn.Sequential(nn.Linear(inp,read_width),nn.ReLU(),nn.Linear(read_width,1))

    def forward(self,e):
        g,p,d,t=self.views(e); z=[self.graph(g)]; task=self.task(t)
        if self.graph_only: return self.readout(torch.cat((z[0],task),-1))
        z.extend((self.fp(p),self.desc(d))); pairs=[]
        for (i,j),factors in zip(((0,1),(0,2),(1,2)),self.factors):
            a,b,c=[layer(v).view(len(g),self.rank,-1) for layer,v in zip(factors,(z[i],z[j],task))]
            pairs.append((a*b*c).sum(1))
        return self.readout(torch.cat((*z,task,*pairs),-1))


def tcf_count(dims,w,r):
    enc=sum((a+1)*w+(w+1)*w for a in dims)
    return enc+9*r*w*w+(7*w+1)*w+w+1


class ConcatMLP(Inputs):
    def __init__(self,*args,config):
        super().__init__(*args); w=config['width']; goal=tcf_count(self.dims,w,config['rank'])
        hidden=max(1,round((goal-1)/(self.input_dim+2)))
        self.net=nn.Sequential(nn.Linear(self.input_dim,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,e): return self.net(self.flat(e))


def build(name,data,*,tiny=False):
    require(name in NEW+CONTROLS,'unknown V11 method'); cfg=dict(CONFIGS[name])
    if tiny:
        cfg.update(width=8,depth=1)
        for k,v in dict(k=3,heads=2,groups=4,state=3,trees=8,selected=5,block=16).items():
            if k in cfg: cfg[k]=v
    args=(data.h.shape[1],data.sources,data.value['metadata'])
    if name=='REALMLP_MV': model=RealMLP(*args,config=cfg,center=data.chemical['robust_center'],factors=data.chemical['robust_factors'])
    elif name in ('FT_TRANSFORMER_MV','EXCELFORMER_MV','T2G_FORMER_MV'):
        model=TokenTransformer(*args,config=cfg,kind={'FT_TRANSFORMER_MV':'ft','EXCELFORMER_MV':'excel','T2G_FORMER_MV':'t2g'}[name])
    elif name in ('TCF','TCF_GRAPH_ONLY'): model=TCF(*args,config=cfg,graph_only=name=='TCF_GRAPH_ONLY')
    else:
        cls={'TABM_MV':TabM,'AMFORMER_MV':AMFormer,'MAMBULAR_MV':Mambular,'GRANDE_MV':Grande,
             'MODERNNCA_MV':ModernNCA,'MV_CONCAT_MLP':ConcatMLP}[name]
        model=cls(*args,config=cfg)
    model.method=name; model.config=cfg
    if name in CONTROLS:
        require(abs(count(model)/tcf_count(model.dims,cfg['width'],cfg['rank'])-1)<=.05,'control parameter matching')
    return model.to(data.device)


def optimizer(model,config):
    require(config in (0,1),'fixed optimization configuration')
    c=model.config; lr=c['lr']*(1. if config==0 else .3)
    if model.method=='REALMLP_MV':
        groups=[dict(params=[p for n,p in model.named_parameters() if n.endswith('weight')],lr=lr),
                dict(params=[p for n,p in model.named_parameters() if n.endswith('bias')],lr=lr*.1),
                dict(params=[model.front_scale],lr=lr*6)]
    elif model.method=='GRANDE_MV':
        groups=[dict(params=[getattr(model,n)],lr=lr*f) for n,f in (('weights',1),('selectors',10),('thresholds',50),('leaves',50))]
    else: groups=[dict(params=list(model.parameters()),lr=lr)]
    for g in groups: g['base_lr']=g['lr']
    return torch.optim.AdamW(groups,lr=lr,weight_decay=c['decay'],betas=tuple(c.get('betas',[.9,.999])))


def schedule(optim,model,step,total):
    factor=.5*(1-math.cos(2*math.pi*math.log2(1+15*step/total))) if model.config.get('schedule')=='coslog4' else 1.
    for group in optim.param_groups: group['lr']=group['base_lr']*factor


def describe(model): return dict(method=model.method,configuration=model.config,parameters=count(model),core=type(model).__name__)


def diagnostics(model): return {}
