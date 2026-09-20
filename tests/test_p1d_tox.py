from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
from torch.utils.data import DataLoader

from dataset import DataCollator
from dataset115_adapter import GraphTaskView
from tests.test_dataset115_adapter import view
from p1d_tox import ToxFactory, TASKS, tox_step, file_sha, load_legacy, validate_replay_rows
from p1d_optimization import OptimizationError
from reproducibility import state_dict_sha256,seed_everything,loader_generator


def fixture_factory():
    """Real original Human3 model/loss, synthetic two-record graph batches."""
    from main import _build_model_components,build_task_dict
    from trainer import Trainer
    from config import prepare_args
    import weighting
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/p1d3_smoke_lock.json').read_bytes())['tox']
    c=deepcopy(c);c['args'].update(hidden_dim=16,head_hidden_dim=16,a_layers=1,a_heads=4,mid_dim=24)
    a=SimpleNamespace(**c['args']);a.gpu_id='cpu';a.datastore_metadata={};a.datastore_context=None
    seed_everything(42);kw,optim=prepare_args(a)
    enc,arch,heads=_build_model_components(a,list(TASKS),torch.device('cpu'))
    original=Trainer(build_task_dict(a,list(TASKS)),weighting.EW,arch,enc,heads,optim,a)
    f=ToxFactory.__new__(ToxFactory);f.device='cpu';f.repo=Path('.')
    f.contract=c;f.initial={k:v.detach().clone() for k,v in original.model.state_dict().items()}
    f.contract['initial_full_model_digest']=state_dict_sha256(original.model)
    f.store=SimpleNamespace(metadata={},context=None,root=Path('.'))
    f.collator=DataCollator()
    ds=GraphTaskView(view(),view().tasks[0])
    f.datasets={s:{t:ds for t in TASKS} for s in ('train','validation')}
    f.loaders={s:{t:DataLoader(ds,batch_size=64,shuffle=s=='train',
                 generator=loader_generator(42,t,0) if s=='train' else None,collate_fn=f.collator)
                 for t in TASKS} for s in ('train','validation')}
    f.contract['counts']={s:{t:len(ds) for t in TASKS} for s in ('train','validation')}
    return f


def test_real_tox_engine_smoke_and_read_only_verifier(tmp_path):
    from p1d_smoke import SmokeAdapter,run_setting_smoke,verify_smoke
    f=fixture_factory()
    result=run_setting_smoke(SmokeAdapter(f.make_trainer,tox_step,f.batch),setting='ToxAcute',
        task_names=TASKS,output_dir=tmp_path/'smoke',expected_identity={'fixture':'synthetic'})
    assert result['observed_optimizer_updates']==11
    assert verify_smoke(tmp_path/'smoke',expected_identity={'fixture':'synthetic'})['observed_optimizer_updates']==11
    from p1d_runtime import verify_live_gradients
    live=verify_live_gradients(tmp_path/'smoke',SmokeAdapter(f.make_trainer,tox_step,f.batch),
        setting='ToxAcute',expected_identity={'fixture':'synthetic'})
    assert live['optimizer_updates']==0 and len(live['checks'])==11


def test_tox_refuses_wrong_post_overlay_init():
    f=fixture_factory();f.contract['initial_full_model_digest']='0'*64
    with pytest.raises(OptimizationError,match='training init'):f.make_trainer()


def test_tox_cpu_explicit_even_if_cuda_available(monkeypatch):
    f=fixture_factory()
    # Only the device resolver is checked; seed_everything may use CUDA hooks.
    from trainer import Trainer
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    assert str(Trainer._resolve_device(SimpleNamespace(gpu_id='cpu')))=='cpu'


def test_hash_locked_legacy_load_and_no_unrestricted_fallback(tmp_path):
    path=tmp_path/'init.pt';torch.save({'model_state':{'x':torch.ones(2)}},path)
    assert torch.equal(load_legacy(path,file_sha(path))['model_state']['x'],torch.ones(2))
    with pytest.raises(OptimizationError,match='SHA'):load_legacy(path,'0'*64)
    class_name=SimpleNamespace(x=1)
    bad=tmp_path/'bad.pt';torch.save(class_name,bad)
    with pytest.raises(Exception,match='Weights only load failed'):load_legacy(bad,file_sha(bad))


def test_legacy_numpy_rng_containers_are_safely_loaded(tmp_path):
    import numpy as np
    path=tmp_path/'rng.pt'
    torch.save(dict(rng=np.asarray([1,2,3],dtype=np.uint32),scalar=np.asarray([1.],dtype=np.float64)),path)
    loaded=load_legacy(path,file_sha(path))
    assert loaded['rng'].tolist()==[1,2,3] and loaded['scalar'].tolist()==[1.]


def test_validation_raw_population_guard():
    f=fixture_factory();rows=[]
    for t,ds in f.datasets['validation'].items():
        for i in range(len(ds)):
            y=float(ds[i].y.reshape(-1)[0])
            rows.append(dict(task=t,sample_id=ds.get_sample_id(i),split='validation',label=y,prediction=y))
    assert validate_replay_rows(f,rows)['macro_rmse']==0
    with pytest.raises(OptimizationError,match='population'):validate_replay_rows(f,rows[:-1])
    bad=deepcopy(rows);bad[0]['prediction']=float('nan')
    with pytest.raises(OptimizationError,match='raw identity'):validate_replay_rows(f,bad)
    bad=deepcopy(rows);bad[0]['label']+=1
    with pytest.raises(OptimizationError,match='raw identity'):validate_replay_rows(f,bad)


def test_constructor_builds_only_train_validation(monkeypatch,tmp_path):
    import dataset
    import toxacute_datastore as store_module
    f=fixture_factory();c=deepcopy(f.contract)
    path=tmp_path/'init.pt';torch.save(f.initial,path)
    c.update(init_path='init.pt',init_file_sha256=file_sha(path))
    observed=[]
    store=SimpleNamespace(metadata=deepcopy(c['data_identity']),context=None,root=tmp_path,
        validate=lambda **kw:observed.append(('validate',kw['expected_task_names'])))
    monkeypatch.setattr(store_module.ToxAcuteDataStore,'resolve',lambda p:store)
    def make_dataset(actual,task,*,split,max_nodes):
        assert actual is store and max_nodes==512
        observed.append((split,task))
        return f.datasets[split][task]
    monkeypatch.setattr(store_module,'ToxAcuteTaskDataset',make_dataset)
    class Wrapper:
        def __init__(self,**kwargs):
            assert kwargs['loader_seed']==42 and kwargs['batch_size']==64
            self.collate_fn_for_loader=kwargs['collate_fn_for_loader']
        def _v2_loader(self,ds,split,task):return f.loaders[split][task]
        def get_data_loaders(self):pytest.fail('holdout loader lifecycle is forbidden')
    monkeypatch.setattr(dataset,'DataloaderWrapper',Wrapper)
    actual=ToxFactory(repo=tmp_path,datastore=tmp_path,contract=c,device='cpu')
    assert set(actual.datasets)=={'train','validation'}
    assert observed==[('validate',list(TASKS))]+[(s,t) for s in ('train','validation') for t in TASKS]
