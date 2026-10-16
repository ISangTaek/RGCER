"""Focused numerical controls; no toxicity optimization or remote connection."""
from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import rdBase
import torch

from architecture.toxacute_tasks import TOXACUTE_TASKS
from baselines.constants import TOXACOL_COMMIT
from baselines.models.toxacol import endpoint_feature_matrix
from molecular_features import PGT_FEATURE_SPEC
from .artifacts import load_arrays, load_manifest, read_json, write_json
from .cache import GraphCache, file_sha256
from .config import CONFIG_ROOT, METHODS, candidate, git_identity
from .models import build_model, parameter_counts
from .training import ModelInputs, endpoint_loss, seed_everything


def environment():
    import algos
    library=Path(algos.__file__)
    if library.suffix not in ('.pyd','.so'):
        raise RuntimeError('Expected a compiled Cython runtime, not an unverified replacement')
    PGT_FEATURE_SPEC.validate_runtime()
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        torch=str(torch.__version__),numpy=np.__version__,rdkit=rdBase.rdkitVersion,
        cuda_available=torch.cuda.is_available(),code_commit=git_identity(),torch_threads=torch.get_num_threads(),
        cython_library_name=library.name,cython_library_sha256=file_sha256(library),
        cython_source_sha256=file_sha256(CONFIG_ROOT.parents[1]/'algos/algos.pyx'),feature_spec_sha256=PGT_FEATURE_SPEC.sha256)


def native_toxacol_parity(source_root):
    source_root=Path(source_root).resolve()
    head=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
    if head!=TOXACOL_COMMIT:
        raise ValueError('ToxACoL author commit mismatch')
    changed=subprocess.check_output(['git','-C',str(source_root),'status','--porcelain','--untracked-files=no'],text=True)
    if changed.strip():
        raise ValueError('Tracked author source changed')
    path=source_root/'models/CorrelationNet.py'
    expected=read_json(CONFIG_ROOT/'dependency_lock.json')['toxacol_model_sha256']
    if file_sha256(path)!=expected:
        raise ValueError('Author CorrelationNet file checksum mismatch')
    spec=importlib.util.spec_from_file_location('pgt_toxacol_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    features,_=endpoint_feature_matrix(TOXACUTE_TASKS)
    # The author's loaders only provide constants; replace file I/O, not math.
    class Reference(module.CorrelationNet):
        def init_adjacency_matrix(self,*args): return torch.eye(59)
        def init_gcn_input(self,*args): return torch.tensor(features)
    seed_everything(42)
    native=Reference(dict(in_features_dnn=1024,in_features_gcn=26,out_features=[768,512,384,64],
        num_layers=4,Dropout_p=0.1,num_task=59)).double().eval()
    port=build_model('M3',TOXACUTE_TASKS,candidate('M3','C0'),adjacency=np.eye(59)).double().eval()
    with torch.no_grad():
        port.tail_weight.copy_(native.tail_w)
        for i in range(4):
            port.dnn[i].linear.load_state_dict(native.dnn[i].layers.Linear.state_dict())
            port.dnn[i].batch_norm.load_state_dict(native.dnn[i].layers.BatchNorm.state_dict())
            port.gcn[i].weight.copy_(native.gcn[i].W)
            port.correlation[i].weight.copy_(native.correlation_net[i].W)
    x=torch.randn(3,1024,dtype=torch.float64,requires_grad=True)
    a,b=native(x),port(x)
    grad_a,=torch.autograd.grad(a.sum(),x,retain_graph=True)
    grad_b,=torch.autograd.grad(b.sum(),x)
    forward=float((a-b).abs().max().detach());backward=float((grad_a-grad_b).abs().max())
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(grad_a,grad_b,atol=1e-10,rtol=1e-10)
    return dict(commit=head,model_sha256=file_sha256(path),forward_max_error=forward,
                input_gradient_max_error=backward,task_channels=59,endpoint_features=26)


def forward_checks(view_root,cache_root,output,*,device='cpu',chemprop_root=None,toxacol_root=None):
    """Limited training-structure forward/loss checks, zero optimizer steps."""
    manifest=load_manifest(view_root)
    records=load_arrays(view_root,'train.npz',manifest)
    stats=load_arrays(view_root,'statistics.npz',manifest)
    cache=GraphCache(cache_root,source_sha256=manifest['source_sha256'])
    if cache.manifest['cache_id']!=manifest['cache_id']:
        raise ValueError('Forward-check cache/view mismatch')
    active=np.where(np.isfinite(records['labels']).any(1))[0]
    if len(active)<2:
        raise ValueError('At least two training structures needed for common forward check')
    # Include largest registered molecule, so no >128-node silent filtering.
    sample=active[:2].tolist()
    registry=Path(cache_root).parent/'record_identity.tsv'
    if registry.is_file():
        import csv
        with registry.open(encoding='utf-8',newline='') as stream:
            sizes={r['record_id']:int(r['num_atoms']) for r in csv.DictReader(stream,delimiter='\t')}
        largest=max(active,key=lambda i:sizes[str(records['record_id'][i])])
        sample=[int(largest),int(next(i for i in active if i != largest))]
    report=dict(environment=environment(),device=str(device),view_sha256=manifest['manifest_sha256'],checks=[],optimizer_steps=0,
                scientific_acceptance='CODEX_REVIEW_PENDING',record_ids=records['record_id'][sample].tolist())
    for method in METHODS:
        configurations=('C0','C1') if method=='M1' else ('C0',)
        for config_id in configurations:
            seed_everything(42)
            model=build_model(method,manifest['task_names'],candidate(method,config_id),adjacency=stats['adjacency'],
                              chemprop_root=chemprop_root,device=device)
            inputs=ModelInputs(cache,records,method,device)
            x=inputs.batch(sample)
            row=dict(method=method,config_id=config_id,parameters=parameter_counts(model))
            if method=='M0':
                row.update(feature_shape=list(x.shape),finite=bool(np.isfinite(x).all()),rf_fits=0)
            else:
                model.eval()
                with torch.no_grad():
                    pred=model(x)
                    target=torch.tensor((records['labels'][sample]-stats['mean'])/stats['scale'],device=device,dtype=pred.dtype)
                    loss=endpoint_loss(pred,target,stats['count'],manifest['active_train_rows'])
                if pred.shape!=(2,len(manifest['task_names'])) or not torch.isfinite(pred).all() or not torch.isfinite(loss):
                    raise ValueError('Nonfinite or misaligned model forward: '+method)
                row.update(output_shape=list(pred.shape),loss=float(loss),finite=True)
            report['checks'].append(row)
            print(f'forward {method}/{config_id}: finite; optimizer_steps=0',flush=True)
    if toxacol_root is not None:
        report['toxacol_author_parity']=native_toxacol_parity(toxacol_root)
    write_json(output,report)
    return report
