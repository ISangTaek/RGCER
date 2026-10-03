"""Synthetic-only V10 architectural probe. Never reads experimental datasets."""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json

import torch

from v10_functional_transfer import Episode, RowIdentity, FunctionalContextRidge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(1); torch.manual_seed(42010)
    source_tasks=tuple(f'source_{i}' for i in range(6))
    tasks=source_tasks+('target_0','target_1')
    g=torch.Generator().manual_seed(42010)
    # Analytic synthetic functions; these are not pretrained toxicity models.
    weights=torch.randn(len(tasks),3,generator=g)
    def episode(task_index, step):
        x=torch.randn(24,8,generator=g)
        latent=torch.stack((torch.sin(x[:,0]),x[:,1]*x[:,2],x[:,3]),dim=1)
        functions=latent@weights[:len(source_tasks)].T
        labels=latent@weights[task_index:task_index+1].T+.05*torch.randn(24,1,generator=g)
        task=tasks[task_index]
        rows=tuple(RowIdentity(f'{task}_{step}_{i}',f'c{task}_{step}_{i}',f'g{task}_{step}_{i}','train',task)
                   for i in range(24))
        e=Episode(task,rows[:16],rows[16:],x[:16],functions[:16],labels[:16],x[16:],functions[16:],
                  torch.tensor([[float(task_index>=len(source_tasks)),task_index/len(tasks)]]))
        return e,labels[16:]
    probes=[episode(i,-1) for i in range(len(tasks))]
    model=FunctionalContextRidge(8,source_tasks,2,hidden=16)
    def scores():
        with torch.no_grad():
            return {tasks[i]:float((model(e)-y).square().mean()) for i,(e,y) in enumerate(probes)}
    before=scores(); opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-5)
    coverage=dict.fromkeys(tasks,0)
    for epoch in range(40):
        for i,task in enumerate(tasks):
            e,y=episode(i,epoch)
            opt.zero_grad(set_to_none=True); loss=(model(e)-y).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('nonfinite synthetic loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); coverage[task]+=1
    after=scores()
    receipt=dict(schema='v10_functional_toy_v1',scope='synthetic engineering feasibility only',
        real_data_access=False,server_connection=False,scientific_superiority_established=False,
        seed=42010,updates=sum(coverage.values()),task_updates=coverage,source_tasks=len(source_tasks),
        target_tasks=2,query_labels_in_forward=False,source_own_head_masked=True,
        parameters=sum(p.numel() for p in model.parameters()),before_mse=before,after_mse=after,
        before_macro_mse=sum(before.values())/len(before),after_macro_mse=sum(after.values())/len(after),
        note='Probe queries are synthetic; no model selection or claim about ToxAcute/PubChem.',
        code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('v10_functional_transfer.py','v10_functional_toy.py')})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2); f.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
