from copy import deepcopy
from types import SimpleNamespace
import hashlib
import random
import numpy as np
import pytest
import torch
from torch import nn
from p2_contract import P2Error
from p2_data import fit_scaler, schedule, digest
from p2_engine import P2Engine, metrics, state_digest


class View:
    def __init__(self, source=False):
        self.tasks=('source',) if source else ('a','b','c')
        def rows(n):
            return [dict(sample_id=str(i),canonical=str(i),group=str(i),label=float(i%7)) for i in range(n)]
        self.rows=dict(train={t:rows(65) for t in self.tasks},validation={} if source else {t:rows(3) for t in self.tasks})
        self.scalers={t:fit_scaler(self.rows['train'][t]) for t in self.tasks}
        self.identity=digest(self.rows)

    def batch(self,split,task,indices,device):
        rows=[self.rows[split][task][i] for i in indices]
        # Exercise Python and NumPy RNG restoration as well as torch dropout.
        x=torch.tensor([[float(i%5),random.random(),float(np.random.random())] for i in indices])
        if split=='validation':x=torch.tensor([[float(i%5),.2,.3] for i in indices])
        return SimpleNamespace(x=x.to(device),y=torch.tensor([[r['label']] for r in rows],device=device)),rows


class Model(nn.Module):
    def __init__(self,tasks):
        super().__init__()
        self.encoder=nn.Module();self.encoder.backbone=nn.Linear(3,3)
        self.decoders=nn.ModuleDict({t:nn.Linear(3,3) for t in tasks})
        self.dropout=nn.Dropout(.1)

    def forward(self,batch,task_name):
        x=self.decoders[task_name](self.dropout(self.encoder.backbone(batch.x)))
        return {task_name:x}


def engine(method):
    view=View(method=='SOURCE')
    with torch.random.fork_rng():
        torch.manual_seed(42);model=Model(view.tasks)
    return P2Engine(model,view,method=method,seed=42,run_identity={'run':'synthetic'})


@pytest.mark.parametrize('method',['SOURCE','FROZEN','B1_low','HF_low'])
@pytest.mark.parametrize('cut',[1,6,31])
def test_exact_resume(tmp_path,method,cut):
    a=engine(method)
    for _ in range(cut):a.step()
    path=tmp_path/'resume.pt';sha=a.save(path)
    for _ in range(7):a.step()
    expected=state_digest(a.model.state_dict());history=deepcopy(a.history)
    random_values=(random.random(),np.random.random(),torch.rand(3))
    b=engine(method);b.restore(path,sha)
    for _ in range(7):b.step()
    assert state_digest(b.model.state_dict())==expected
    assert b.history==history
    assert random.random()==random_values[0]
    assert np.random.random()==random_values[1]
    assert torch.equal(torch.rand(3),random_values[2])


@pytest.mark.parametrize('field',['total_updates','offset','running','identity','model_digest'])
def test_reject_corrupt_checkpoint(tmp_path,field):
    a=engine('HF_low');a.step();p=a.snapshot()
    if field=='running':p[field][0]['sample_ids']=['forged']
    elif field=='identity':p[field]['seed']=43
    elif field=='model_digest':p[field]='wrong'
    else:p[field]=999
    path=tmp_path/'bad.pt';torch.save(p,path)
    with pytest.raises(P2Error):engine('HF_low').restore(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_source_budget_and_no_validation():
    a=engine('SOURCE')
    for _ in range(80):a.step()
    assert a.epoch==40 and a.best_state is None
    with pytest.raises(P2Error):a.step()
    with pytest.raises(P2Error):a.evaluate()


def test_scaler_and_schedule():
    assert fit_scaler([{'label':2.},{'label':4.}])['std']==1.
    assert fit_scaler([{'label':2.}])['std_reason']=='constant_or_near_constant'
    with pytest.raises(P2Error):fit_scaler([{'label':float('nan')}])
    counts={'a':96,'b':85,'c':81}
    plan=schedule(tuple(counts),counts,42,0)
    assert len(plan)==6 and plan==schedule(tuple(counts),counts,42,0)
    for task,n in counts.items():
        assert sorted(i for t,ix in plan if t==task for i in ix)==list(range(n))
    assert plan!=schedule(tuple(counts),counts,42,1)


def test_metrics_reject_bad_predictions():
    with pytest.raises(P2Error):metrics([dict(task='a',sample_id='1',label=1.,prediction=float('nan'))],['a'])
    with pytest.raises(P2Error):metrics([dict(task='a',sample_id='1',label=1.,prediction=1.)],['a','b'])


def test_full_target_and_no_overwrite(tmp_path):
    a=engine('HF_low')
    for _ in range(240):a.step()
    assert a.epoch==40 and len(a.history)==40
    assert a.best_epoch==min(range(40),key=lambda i:a.history[i]['validation']['macro_rmse'])
    path=tmp_path/'final.pt';sha=a.save(path)
    with pytest.raises(FileExistsError):a.save(path)
    b=engine('HF_low');b.restore(path,sha)
    assert state_digest(a.model.state_dict())==state_digest(b.model.state_dict())
    with pytest.raises(P2Error):b.step()


@pytest.mark.parametrize('field',['macro','population','lr','initial'])
def test_history_contract(tmp_path,field):
    a=engine('B1_low')
    if field!='initial':
        for _ in range(6):a.step()
    p=a.snapshot()
    if field=='macro':p['history'][0]['validation']['macro_rmse']+=1
    elif field=='population':p['history'][0]['validation']['endpoints']['a']['n']=2
    elif field=='lr':p['history'][0]['head_lr']=.01
    else:
        next(iter(p['model_state'].values())).add_(1)
        p['model_digest']=state_digest(p['model_state'])
    path=tmp_path/'bad.pt';torch.save(p,path)
    with pytest.raises(P2Error):engine('B1_low').restore(path,hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('split,task,indices',[('test','a',[0]),('calibration','a',[0]),('train','b',[0]),('train','a',[0,0]),('train','a',[True]),('train','a',[2])])
def test_view_permission_before_graph_access(split,task,indices):
    from p2_data import P2View
    class ForbiddenStore:
        def get_graph_data(self,*args,**kwargs):raise AssertionError('graph read before permission')
    view=P2View(ForbiddenStore(),{'a':[dict(label=1.)]}, {})
    with pytest.raises(P2Error):view.batch(split,task,indices,'cpu')
