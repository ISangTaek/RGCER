"""Read-only B115 audit or exactly one GPU update. No formal/test entry point."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from baselines.b115_data import load_b115
from baselines.b115_training import (implementation_identity, optimizer_step, predict,
                                     save_json, validation_metrics)
from baselines.features import avalon_matrix
from baselines.models.toxacol import ToxACoLNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=['A', 'B'], required=True)
    for key in ('audit-path', 'allowlist', 'csv-path', 'split-path', 'tox-manifest', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--datastore')
    parser.add_argument('--mode', choices=['audit', 'one-step'], default='audit')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.mode == 'one-step' and not args.device.startswith('cuda'):
        parser.error('real one-step execution is server CUDA only; use audit locally')
    if args.mode == 'audit' and args.device != 'cpu':
        parser.error('read-only audit uses CPU')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    train, validation, adjacency, features, scaler, contract = load_b115(
        **{k: getattr(args, k) for k in ('route', 'audit_path', 'allowlist', 'csv_path',
                                        'split_path', 'tox_manifest', 'datastore')})
    save_json(output / 'data_contract.json', contract)
    report = dict(route=args.route, mode=args.mode, implementation=implementation_identity(),
        active_rows=len(train.sample_ids), observations=int(np.isfinite(train.labels).sum()),
        tasks=len(train.tasks), feature_width=features.shape[1],
        edges=contract['graph']['undirected_edges'],
        source_target_edges=int(np.count_nonzero(adjacency[:-5, -5:])),
        validation_rows=len(validation.sample_ids), test_executed=False, updates=0)
    if args.mode == 'one-step':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required, no CPU fallback')
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Deterministic support batch covers every permitted task at least once.
        selected = sorted({int(np.flatnonzero(np.isfinite(train.labels[:, j]))[0])
                           for j in range(len(train.tasks))} | {0, 1})
        if len(selected) > 128:
            raise RuntimeError('smoke support batch exceeds 128 rows')
        x = avalon_matrix(tuple(train.smiles[i] for i in selected))
        y, mask = scaler.transform(train.labels[selected])
        model = ToxACoLNet(adjacency, features).to(args.device)
        before = model.tail_weight.detach().cpu().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9,
                                    nesterov=True, weight_decay=.0005)
        loss = optimizer_step(model, optimizer, torch.as_tensor(x, device=args.device),
            torch.as_tensor(y, device=args.device), torch.as_tensor(mask.astype(bool), device=args.device))
        if torch.equal(before, model.tail_weight.detach().cpu()):
            raise RuntimeError('parameters did not update')
        checkpoint = output / 'smoke.pt'
        torch.save(dict(role='SMOKE_NOT_FORMAL', state=model.state_dict(), contract=contract), checkpoint)
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        clone = ToxACoLNet(adjacency, features).to(args.device)
        clone.load_state_dict(loaded['state'], strict=True)
        vx = avalon_matrix(validation.smiles)
        p1 = predict(model, vx, scaler, args.device)
        p2 = predict(clone, vx, scaler, args.device)
        if not np.array_equal(p1, p2):
            raise RuntimeError('checkpoint reload changes validation predictions')
        metrics = validation_metrics(validation.labels, p2)
        np.savez_compressed(output / 'validation_smoke.npz', sample_ids=np.asarray(validation.sample_ids),
            tasks=np.asarray(validation.tasks), truth=validation.labels, prediction=p2)
        save_json(output / 'sample_ids.json', [train.sample_ids[i] for i in selected])
        report.update(updates=1, support_rows=len(selected), loss=loss, metrics=metrics,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            reload_max_abs=float(np.max(np.abs(p1-p2))),
            device=torch.cuda.get_device_name(torch.device(args.device)),
            peak_memory_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)))
    report['elapsed_seconds'] = time.monotonic()-start
    report['validation_status'] = 'PASS'
    report['acceptance_status'] = 'PENDING_REVIEW'
    save_json(output / 'report.json', report)
    print(f'B115 {args.route} {args.mode} PASS; formal training/test NOT executed')


if __name__ == '__main__':
    main()
