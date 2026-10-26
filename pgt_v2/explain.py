"""Additive accounting and full-model embedding integrated gradients."""
from __future__ import annotations

import numpy as np
import torch

from .chemistry import digest


@torch.no_grad()
def additive_reference(model, batches, *, record_ids, view_id, checkpoint_sha256):
    if model.method not in ('A2', 'A3') or not record_ids or len(record_ids) > 512 or len(set(record_ids)) != len(record_ids):
        raise ValueError('Additive centering needs a fixed unique training reference, <=512 rows')
    model.eval(); total = None; n = 0
    for batch in batches:
        _, aux = model(batch, return_aux=True)
        beta = (aux['attention'] * aux['values']).sum(-1).sum(0)
        total = beta if total is None else total + beta
        n += batch.x.shape[0]
    if n != len(record_ids):
        raise ValueError('Reference records/batches count mismatch')
    result = dict(beta=(total/n).cpu().tolist(), record_ids=list(record_ids), view_id=view_id,
                  checkpoint_sha256=checkpoint_sha256, definition='mean reference attention-weighted value per task and head')
    result['reference_sha256'] = digest(result)
    return result


def additive_accounting(prediction, aux, *, mean, scale, reference=None):
    dtype, device = prediction.dtype, prediction.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    scale = torch.as_tensor(scale, dtype=dtype, device=device)
    values, attention = aux['values'], aux['attention']
    beta = torch.zeros(values.shape[1:3], dtype=dtype, device=device)
    if reference is not None:
        if digest({k:v for k,v in reference.items() if k != 'reference_sha256'}) != reference['reference_sha256']:
            raise ValueError('Reference checksum mismatch')
        beta = torch.as_tensor(reference['beta'], dtype=dtype, device=device)
        if beta.shape != values.shape[1:3]:
            raise ValueError('Reference task/head shape mismatch')
    contribution = (attention * (values - beta[None, :, :, None])).sum(2)
    bias = aux['bias'] + beta.sum(1)
    original_contribution = contribution * scale[None, :, None]
    original_bias = mean + scale * bias
    original_prediction = mean + scale * prediction
    error = (original_bias + original_contribution.sum(-1) - original_prediction).abs().max()
    tolerance = 1e-5 + 1e-5 * original_prediction.abs().max()
    if error > tolerance:
        raise ValueError('Additive original-scale reconstruction failed')
    return dict(prediction=original_prediction, bias=original_bias, contributions=original_contribution,
                max_reconstruction_error=float(error.detach()))


def endpoint_difference(accounting, left, right):
    contribution = accounting['contributions'][:, left] - accounting['contributions'][:, right]
    bias = accounting['bias'][left] - accounting['bias'][right]
    prediction = accounting['prediction'][:, left] - accounting['prediction'][:, right]
    return dict(contributions=contribution, bias=bias, prediction=prediction,
                max_reconstruction_error=float((bias + contribution.sum(-1) - prediction).abs().max().detach()))


def integrated_gradients(model, batch, *, task, mean, scale, baseline=None, other_task=None, points=64):
    """64 Gauss-Legendre gradient points; topology/graph token stay fixed.

    N1/N1S use alpha=u^2 to resolve the high curvature near the baseline.
    Completeness failures are reported, never hidden by extra integration runs.
    """
    if points != 64:
        raise ValueError('Registered IG uses exactly 64 gradient evaluation points')
    model.eval()
    inputs = model.backbone._encode_atoms(batch.x.long(), batch.in_degree.long(), batch.out_degree.long()).detach()
    mask = ~batch.padding_mask
    if baseline is None:
        base = torch.zeros_like(inputs)
    else:
        base = torch.as_tensor(baseline, dtype=inputs.dtype, device=inputs.device).expand_as(inputs).clone()
    inputs = inputs.masked_fill(~mask[..., None], 0)
    base = base.masked_fill(~mask[..., None], 0)
    scale = torch.as_tensor(scale, dtype=inputs.dtype, device=inputs.device)
    mean = torch.as_tensor(mean, dtype=inputs.dtype, device=inputs.device)
    def scalar(embeddings):
        raw = model(batch, atom_embeddings=embeddings)
        result = mean[task] + scale[task]*raw[:, task]
        if other_task is not None:
            result = result - mean[other_task] - scale[other_task]*raw[:, other_task]
        return result
    nodes, weights = np.polynomial.legendre.leggauss(points)
    alphas, weights = (nodes+1)/2, weights/2
    # Deep prompts + LayerNorm can vary rapidly near an all-zero baseline.
    # alpha=u^2 is an exact change of variable, still using only 64 points.
    path_power = 2 if getattr(model, 'method', None) in ('N1', 'N1S') else 1
    if path_power == 2:
        weights = weights * 2 * alphas
        alphas = alphas ** 2
    integral = torch.zeros_like(inputs)
    for alpha, weight in zip(alphas, weights):
        interpolated = (base + float(alpha)*(inputs-base)).requires_grad_(True)
        gradient, = torch.autograd.grad(scalar(interpolated).sum(), interpolated)
        integral += float(weight)*gradient.detach()
    attribution = ((inputs-base)*integral).sum(-1).masked_fill(~mask, 0)
    with torch.no_grad():
        prediction, baseline_prediction = scalar(inputs), scalar(base)
        residual = attribution.sum(-1) - (prediction-baseline_prediction)
        normalized = residual.abs() / (prediction-baseline_prediction).abs().clamp_min(1e-8)
    return dict(attributions=attribution, prediction=prediction, baseline_prediction=baseline_prediction,
        completeness_residual=residual, normalized_residual=normalized, unreliable=normalized > 1e-3,
        path_power=path_power,
        definition='fixed-topology full-model embedding attribution; not a causal deletion effect')


@torch.no_grad()
def training_embedding_mean(model, batches):
    total, n = None, 0
    for batch in batches:
        embeddings = model.backbone._encode_atoms(batch.x.long(), batch.in_degree.long(), batch.out_degree.long())
        mask = ~batch.padding_mask
        value = embeddings[mask].sum(0)
        total = value if total is None else total + value
        n += int(mask.sum())
    if not n:
        raise ValueError('Empty training reference')
    return total/n


@torch.no_grad()
def representation_intervention(model, batch, atom_mask, *, replacement=0.0):
    """Rerun the complete model with selected input embeddings replaced."""
    if atom_mask.shape != batch.padding_mask.shape or atom_mask.dtype != torch.bool or bool((atom_mask & batch.padding_mask).any()):
        raise ValueError('Intervention must select valid atoms only')
    model.eval()
    embedding = model.backbone._encode_atoms(batch.x.long(), batch.in_degree.long(), batch.out_degree.long())
    replacement = torch.as_tensor(replacement, dtype=embedding.dtype, device=embedding.device)
    return model(batch, atom_embeddings=torch.where(atom_mask[..., None], replacement, embedding))


def explain_record(checkpoint_path, view_root, cache_root, reference_path, record_id, task, output,
                   *, other_task=None, device='cpu'):
    from pathlib import Path
    from .artifacts import load_arrays, load_manifest, read_json, seal_manifest, write_json
    from .cache import GraphCache, file_sha256
    from .chemistry import framed_digest
    from .evaluation import load_model_checkpoint
    from .training import ModelInputs, partition_batches
    manifest=load_manifest(view_root)
    model,checkpoint=load_model_checkpoint(checkpoint_path,device=device)
    identity=checkpoint['identity'];method=identity['method_id']
    if method not in ('M2','M5','M6','A1','A2','A3','N1','N1S') or identity['view_sha256']!=manifest['manifest_sha256']:
        raise ValueError('Explanation method/checkpoint/view mismatch')
    tasks=manifest['task_names']
    left=tasks.index(task);right=tasks.index(other_task) if other_task is not None else None
    if other_task is not None and task.rsplit('_',1)[-1] != other_task.rsplit('_',1)[-1]:
        raise ValueError('Endpoint difference requires the same measurement semantics')
    # Reference contains input identities, never a test-derived statistic.
    reference=read_json(reference_path)
    train=load_arrays(view_root,'train_inputs.npz',manifest)
    unique={}
    for chemical,record in zip(train['chemical_id'],train['record_id']):
        unique.setdefault(str(chemical),str(record))
    ordered=sorted(unique,key=lambda c:(framed_digest([manifest['fold'],c]),c))[:512]
    expected=[unique[c] for c in ordered]
    if reference['record_ids']!=expected or reference['source_sha256']!=manifest['source_sha256'] or reference['fold']!=manifest['fold']:
        raise ValueError('Reference is not the registered full-training input set')
    lookup={r:i for i,r in enumerate(train['record_id'])}
    cache=GraphCache(cache_root,source_sha256=manifest['source_sha256'])
    if cache.manifest['cache_id']!=manifest['cache_id']:
        raise ValueError('Explanation cache mismatch')
    inputs=ModelInputs(cache,train,method,device)
    ref_indices=[lookup[r] for r in expected]
    def batches():
        return (inputs.batch(part) for part in partition_batches(ref_indices,identity['config']['micro_batch']))
    selected=None
    for role in ('train','validation','test'):
        records=load_arrays(view_root,role+'_inputs.npz',manifest)
        where=np.where(records['record_id']==record_id)[0]
        if len(where):
            if selected is not None:
                raise ValueError('Record appears in multiple partitions')
            selected=(ModelInputs(cache,records,method,device),int(where[0]),role)
    if selected is None:
        raise ValueError('Unknown explanation record')
    reader,index,role=selected
    batch=reader.batch([index]);stats=checkpoint['stats'];arrays={}
    provenance=dict(record_id=record_id,role=role,method=method,task=task,other_task=other_task,
        reference_records=expected,reference_input_sha256=digest(reference),checkpoint_sha256=file_sha256(checkpoint_path),
        view_sha256=manifest['manifest_sha256'],source_sha256=manifest['source_sha256'],
        feature_spec_sha256=manifest['feature_spec_sha256'],raw_to_model=batch.raw_to_model,
        model_to_raw=batch.model_to_raw,symmetry_classes=batch.symmetry_classes)
    if method in ('A2','A3'):
        centered=additive_reference(model,batches(),record_ids=expected,view_id=manifest['view_id'],checkpoint_sha256=file_sha256(checkpoint_path))
        with torch.no_grad():
            pred,aux=model(batch,return_aux=True)
            accounting=additive_accounting(pred,aux,mean=stats['mean'],scale=stats['scale'],reference=centered)
            selected_accounting=endpoint_difference(accounting,left,right) if right is not None else accounting
            arrays={k:v.detach().cpu().numpy() for k,v in selected_accounting.items() if isinstance(v,torch.Tensor)}
        provenance.update(centering=centered,max_reconstruction_error=selected_accounting['max_reconstruction_error'],
                          interpretation='exact model accounting on the provided label scale; no causal/mechanistic claim')
    else:
        baselines={'zero':None,'training_mean':training_embedding_mean(model,batches())}
        diagnostics={}
        for name,baseline in baselines.items():
            if baseline is not None:
                arrays[name+'_embedding_baseline']=baseline.detach().cpu().numpy()
            result=integrated_gradients(model,batch,task=left,other_task=right,mean=stats['mean'],scale=stats['scale'],baseline=baseline)
            arrays.update({name+'_'+k:v.detach().cpu().numpy() for k,v in result.items() if isinstance(v,torch.Tensor)})
            diagnostics[name]=dict(path_power=result['path_power'], normalized_residual=result['normalized_residual'].cpu().tolist(),
                                   unreliable=result['unreliable'].cpu().tolist())
        provenance.update(ig_gradient_points=64,diagnostics=diagnostics,
                          interpretation='full-model fixed-topology embedding attribution; flagged failures remain in denominator')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'attributions.npz',**arrays)
    write_json(output/'provenance.json',provenance)
    return seal_manifest(output,provenance,['attributions.npz','provenance.json'])
