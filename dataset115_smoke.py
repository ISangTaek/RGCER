"""One-step Route B smoke, not an epoch-based training or test exporter."""
from pathlib import Path
import hashlib
import json

import torch

from dataset115_contract import ContractError, PRIMARY
from dataset115_model import build_human5_model


def state_digest(state):
    h = hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(json.dumps([k,list(v.shape),str(v.dtype)],separators=(',',':')).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_step_records(output, records):
    """Reopen saved tensors and recompute identities, rather than trusting PASS."""
    if [r['method'] for r in records] != ['B0','B1','RPT']:
        raise ContractError('smoke result method matrix differs')
    if len({r['head_before'] for r in records}) != 1:
        raise ContractError('initial heads not paired')
    if records[1]['encoder_before'] != records[2]['encoder_before']:
        raise ContractError('B1/RPT initial encoders not paired')
    for row in records:
        method=row['method'];path=Path(output)/f'{method}_smoke.pt'
        if row['checkpoint'] != path.name or hashlib.sha256(path.read_bytes()).hexdigest() != row['checkpoint_sha256']:
            raise ContractError('saved checkpoint identity differs')
        p=torch.load(path,map_location='cpu',weights_only=True)
        if p['task_names'] != list(PRIMARY) or p['method'] != method or p['seed'] != 42 or p['steps'] != 1:
            raise ContractError('saved smoke scope differs')
        for prefix,field in [('encoder.','encoder_after'),('decoders.','head_after')]:
            state={k[len(prefix):]:v for k,v in p['model_state'].items() if k.startswith(prefix)}
            if not state or state_digest(state) != row[field]:
                raise ContractError('saved tensor digest differs')
        if ((method=='RPT') != (row['encoder_before']==row['encoder_after'])
                or row['head_before']==row['head_after']):
            raise ContractError('saved update/freeze invariant differs')
        import math
        if any(not math.isfinite(row[k]) for k in ['loss','grad_norm','reload_max_error']) or row['reload_max_error']>1e-6:
            raise ContractError('invalid smoke numerical result')
    return {'saved_checkpoint_tensor_checks':3,'scope':'SMOKE_ONLY_NOT_FORMAL_TRAINING'}


def one_step(args, source, batches, scalers, output, *, device, seed=42):
    from loss import QuantileRegressionLoss
    output = Path(output)
    if output.exists():
        raise ContractError('smoke output exists')
    if seed != 42 or set(batches) != set(PRIMARY) or set(scalers) != set(PRIMARY):
        raise ContractError('smoke seed/task scope differs')
    if any(getattr(b,'is_empty',False) or b.y.numel() != 2 for b in batches.values()):
        raise ContractError('smoke requires two training observations per task')
    output.mkdir(parents=True, exist_ok=False)
    records=[]; head_identity=None
    for method in ('B0', 'B1', 'RPT'):
        model = build_human5_model(args,method=method,seed=seed,
                                  source_encoder_state=None if method=='B0' else source).to(device)
        before=state_digest(model.encoder.state_dict())
        heads=state_digest(model.decoders.state_dict())
        if head_identity is None:head_identity=heads
        if heads != head_identity:raise ContractError('paired heads differ')
        torch.manual_seed(seed)
        model.train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.001,weight_decay=1e-5)
        loss_fn=QuantileRegressionLoss()
        loss=0
        for t in PRIMARY:
            batch=batches[t].to(device); s=scalers[t]
            y=(batch.y.reshape(-1,1)-s['mean'])/s['std']
            loss=loss+loss_fn.compute_loss(model(batch,task_name=t)[t],y)/len(PRIMARY)
        if not torch.isfinite(loss):raise ContractError('nonfinite smoke loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        if method=='RPT' and any(p.grad is not None for p in model.encoder.parameters()):
            raise ContractError('RPT source gradient leak')
        opt.step()
        after=state_digest(model.encoder.state_dict()); after_heads=state_digest(model.decoders.state_dict())
        if (method=='RPT') != (before==after) or after_heads==heads:
            raise ContractError('smoke update/freeze invariant failed')
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ContractError('nonfinite model state')
        path=output/f'{method}_smoke.pt'
        with path.open('xb') as f:
            torch.save(dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),
                            task_names=list(PRIMARY),method=method,seed=seed,steps=1,scalers=scalers),f)
        restored=build_human5_model(args,method=method,seed=seed,
                                   source_encoder_state=None if method=='B0' else source).to(device)
        payload=torch.load(path,map_location=device,weights_only=True)
        restored.load_state_dict(payload['model_state'],strict=True)
        restored_opt=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=0.001,weight_decay=1e-5)
        restored_opt.load_state_dict(payload['optimizer_state'])
        if len(restored_opt.state) != len(opt.state):raise ContractError('optimizer reload differs')
        model.eval();restored.eval(); max_error=0.0
        with torch.no_grad():
            for t in PRIMARY:
                a=model(batches[t],task_name=t)[t]; b=restored(batches[t],task_name=t)[t]
                if not torch.allclose(a,b,atol=1e-6,rtol=1e-6):raise ContractError('reload prediction differs')
                max_error=max(max_error,float((a-b).abs().max()))
        records.append(dict(method=method,seed=seed,steps=1,loss=float(loss.detach()),grad_norm=float(norm),
            encoder_before=before,encoder_after=after,head_before=heads,head_after=after_heads,
            reload_max_error=max_error,checkpoint=path.name,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del model,restored,opt,restored_opt
    return records
