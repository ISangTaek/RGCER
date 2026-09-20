"""P1-D Human3 factory and read-only historical validation replay.

Use the original Trainer/model/loss and original DataLoader constructor. Only
train and validation datasets are constructed. No main() lifecycle side effects,
calibration/test loaders, new source training or native compilation.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import importlib

import numpy as np
import torch

from p1d_optimization import OptimizationControl, OptimizationSpec, require
from reproducibility import seed_everything, state_dict_sha256

TASKS = ('man_oral_TDLo', 'women_oral_TDLo', 'human_oral_TDLo')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_legacy(path, expected_sha):
    """Restrict legacy checkpoint deserialization to reviewed data containers."""
    from toxacute_datastore import DataStoreContext
    require(file_sha(path)==expected_sha, 'checkpoint SHA')
    # Historical RNG buffers use NumPy array reconstruction. Never fall back to
    # unrestricted pickle when an unrecognized global appears.
    multiarray=importlib.import_module('numpy.core.multiarray')
    allowed=[DataStoreContext,np.dtype,np.ndarray,multiarray._reconstruct,
             (multiarray._reconstruct,'numpy._core.multiarray._reconstruct'),
             (multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
             type(np.dtype('uint32')),type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        value=torch.load(path,map_location='cpu',weights_only=True)
    require(file_sha(path)==expected_sha, 'checkpoint changed during load')
    return value


class ToxFactory:
    def __init__(self, *, repo, datastore, contract, device):
        from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset
        from dataset import DataloaderWrapper, DataCollator
        self.repo=Path(repo).resolve();self.contract=deepcopy(contract);self.device=device
        require(device in ('cpu','cuda:0'), 'device')
        c=self.contract; a=c['args']
        require(a['seed']==42 and type(a['seed']) is int and a['arch']=='Graphormer', 'Tox seed/architecture')
        for key,value in dict(epochs=40,bs=64,lr=.001,weight_decay=1e-5,grad_clip=1,
                              optim='adamw',weighting='EW',task_sampling='proportional',
                              train_eval_scope='validation_only',fit_conformal=False,
                              toxacute_task_scope='human3',selection_scope='human3').items():
            require(type(a[key]) is type(value) and a[key]==value,'Tox configuration '+key)
        self.store=ToxAcuteDataStore.resolve(datastore)
        expected=c['data_identity'];actual=self.store.metadata
        for k,v in expected.items():require(actual[k]==v,'Tox data identity '+k)
        self.store.validate(strict=False,expected_task_names=list(TASKS),expected_max_path_distance=8)
        path=(self.repo/c['init_path']).resolve()
        require(path.is_relative_to(self.repo),'init outside repository')
        payload=load_legacy(path,c['init_file_sha256'])
        self.initial=payload.get('model_state',payload)
        require(type(self.initial) is dict or isinstance(self.initial,dict),'init mapping')
        require(all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in self.initial.values()),'init tensors')
        wrapper=DataloaderWrapper(task_list=list(TASKS),data_store=self.store,batch_size=64,
            splitting='scaffold',valid_size=.1,calibration_size=.1,test_size=.1,
            num_workers=0,split_seed=42,max_nodes_filter=512,loader_seed=42,
            collate_fn_for_loader=DataCollator(spatial_pos_max_clip=20,max_node_filter=None))
        # Do not call get_data_loaders(), which constructs all four split views.
        self.datasets={split:{t:ToxAcuteTaskDataset(self.store,t,split=split,max_nodes=512)
                              for t in TASKS} for split in ('train','validation')}
        self.loaders={split:{t:wrapper._v2_loader(ds,split,t) for t,ds in dsmap.items()}
                      for split,dsmap in self.datasets.items()}
        for split in ('train','validation'):
            require({t:len(ds) for t,ds in self.datasets[split].items()}==c['counts'][split],
                    'Tox population count '+split)
        self.collator=wrapper.collate_fn_for_loader

    def make_trainer(self, arm=None):
        from main import _build_model_components,build_task_dict
        from config import prepare_args
        from trainer import Trainer
        import weighting
        a=deepcopy(self.contract['args'])
        a.update(gpu_id='0' if self.device=='cuda:0' else 'cpu',save_path=None,load_path=None,
                 freeze_backbone_epochs=0,backbone_lr_multiplier=1.0,
                 trainable_last_blocks=0,d6_candidate='none',d7_candidate='none',
                 datastore_metadata=self.store.metadata,datastore_context=self.store.context,
                 data_store_dir=str(self.store.root))
        params=SimpleNamespace(**a)
        seed_everything(42)
        kw,optim=prepare_args(params)
        enc,arch,heads=_build_model_components(params,list(TASKS),torch.device(self.device))
        trainer=Trainer(build_task_dict(params,list(TASKS)),weighting.EW,arch,enc,heads,
                        optim,params,save_path=None,load_path=None,**kw)
        require(str(trainer.device)==self.device, 'Tox actual device')
        trainer.model.load_state_dict(self.initial,strict=True)
        require(not list(trainer.loss_balancer.parameters()),'EW must be parameterless')
        trainer.initial_model_sha256=state_dict_sha256(trainer.model)
        require(trainer.initial_model_sha256==self.contract['initial_full_model_digest'],'Tox actual training init')
        # Reset each independent train-loader generator before the inherited fit.
        from reproducibility import reseed_train_loaders
        reseed_train_loaders(self.loaders['train'],42,0)
        trainer._fit_task_scalers(self.loaders['train'])
        require({t:s['count'] for t,s in trainer.task_scalers.items()}==self.contract['counts']['train'], 'scaler population')
        trainer.optimization=None
        if arm is not None:
            trainer.optimization=OptimizationControl(trainer.model,OptimizationSpec(arm))
            trainer.optimizer=trainer.optimization.optimizer
        return trainer

    def batch(self, task, index):
        require(task in TASKS and type(index) is int and 0<=index<8,'smoke batch scope')
        ds=self.datasets['train'][task]
        batch=self.collator([ds[i] for i in range(min(2,len(ds)))])
        require(not batch.is_empty and len(batch.sample_id)==min(2,len(ds)), 'smoke train batch')
        return batch.to(self.device)

    def identity(self):
        return dict(setting='ToxAcute',seed=42,contract=self.contract,
                    batch_policy='first_two_train_records_per_task_in_fixed_view',
                    tasks=list(TASKS),test_accessed=False,calibration_accessed=False)


def tox_step(trainer, task, batch):
    """Original EW loss/backward/clip/step path, with finite-gradient guard."""
    trainer.model.train()
    trainer.loss_balancer.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    epoch=trainer.optimization.epoch if trainer.optimization is not None else 0
    loss,_=trainer._training_step(batch,task,epoch)
    require(torch.isfinite(loss),'nonfinite loss')
    index=trainer.task_name.index(task)
    losses=loss.new_zeros(trainer.task_num);losses[index]=loss
    mask=loss.new_zeros(trainer.task_num,dtype=torch.bool);mask[index]=True
    trainer.loss_balancer.backward(losses,active_mask=mask)
    torch.nn.utils.clip_grad_norm_(list(trainer.model.parameters())+list(trainer.loss_balancer.parameters()),1.,error_if_nonfinite=True)
    trainer.optimizer.step()
    return float(loss.detach())


def replay_historical(factory, *, path, expected_sha, expected_epoch, expected_rmse):
    """Read a hash-locked legacy best; raw validation records are independently keyed."""
    q=load_legacy(path,expected_sha)
    require(q['epoch']==expected_epoch and type(q['epoch']) is int,'historical epoch')
    require(q['task_names']==list(TASKS),'historical tasks')
    require(q['reproducibility']['initial_model_sha256']==factory.contract['initial_full_model_digest'], 'historical init')
    for k in ('datastore_fingerprint','feature_schema_version'):
        require(q['data_config'][k]==factory.contract['data_identity'][k], 'historical data '+k)
    require(q['split_manifest_hash']==factory.contract['data_identity']['split_manifest_hash'],'historical split')
    trainer=factory.make_trainer(None)
    trainer.model.load_state_dict(q['model_state'],strict=True)
    require(all(torch.isfinite(v).all() for v in trainer.model.state_dict().values()),'historical finite weights')
    trainer.task_scalers=deepcopy(q['task_scalers'])
    require(set(trainer.task_scalers)==set(TASKS),'historical scaler tasks')
    for t,s in trainer.task_scalers.items():
        require(set(s)=={'mean','std','count'} and s['count']==factory.contract['counts']['train'][t]
                and math.isfinite(s['mean']) and math.isfinite(s['std']) and s['std']>0,'historical scaler')
    trainer.model.eval();rows=[]
    with torch.no_grad():
        for t in TASKS:
            for batch in factory.loaders['validation'][t]:
                require(not batch.is_empty,'empty validation batch')
                batch=batch.to(factory.device)
                output,_=trainer._forward_task(batch,t,expected_epoch,return_aux=True)
                prediction=trainer.decode_task_output(t,output[t],apply_conformal=False)['median'].reshape(-1)
                require(prediction.numel()==batch.y.numel() and torch.isfinite(prediction).all(),'validation prediction')
                rows.extend(dict(task=t,sample_id=str(sid),split='validation',label=float(y),prediction=float(p))
                            for sid,y,p in zip(batch.sample_id,batch.y.reshape(-1).cpu(),prediction.cpu()))
    metrics=validate_replay_rows(factory,rows)
    require(abs(metrics['macro_rmse']-expected_rmse)<=1e-5,'historical validation replay differs')
    return dict(checkpoint_sha256=expected_sha,epoch=expected_epoch,metrics=metrics,rows=rows,
                rmse_tolerance=1e-5,absolute_delta=abs(metrics['macro_rmse']-expected_rmse))


def validate_replay_rows(factory,rows):
    expected={}
    for t,ds in factory.datasets['validation'].items():
        for i in range(len(ds)):
            item=ds[i];expected[(t,ds.get_sample_id(i))]=float(item.y.reshape(-1)[0])
    keys=[(r['task'],r['sample_id']) for r in rows]
    require(len(keys)==len(set(keys)) and set(keys)==set(expected),'replay population')
    for r in rows:
        require(r['split']=='validation' and r['label']==expected[(r['task'],r['sample_id'])]
                and type(r['prediction']) in (int,float) and math.isfinite(r['prediction']),'replay raw identity')
    metrics={t:dict(n=sum(r['task']==t for r in rows),
                   rmse=math.sqrt(math.fsum((r['prediction']-r['label'])**2 for r in rows if r['task']==t)/factory.contract['counts']['validation'][t])) for t in TASKS}
    return dict(endpoints=metrics,macro_rmse=math.fsum(m['rmse'] for m in metrics.values())/3)
