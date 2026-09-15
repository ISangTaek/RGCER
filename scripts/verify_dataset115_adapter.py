"""Read-only dataset115 adapter check: train scalers and ten training graphs.

No optimizer, checkpoint load, GPU, predictions, or datastore writes.
"""
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(table):
    import numpy as np
    from dataset115_adapter import TrainOnlyScaler, GraphTaskView
    from dataset115_contract import PRIMARY
    scalers = {}; graphs = []
    for route, role in [('A', 'source'), ('A', 'target'), ('B', 'target')]:
        view = table.view(route, role, 'train')
        scaler = TrainOnlyScaler.fit(view)
        for i, task in enumerate(view.tasks):
            values = view.labels[np.isfinite(view.labels[:, i]), i].tolist()
            mean = math.fsum(values) / len(values)
            std = max(math.sqrt(math.fsum((v-mean)**2 for v in values) / len(values)), 1e-6)
            if (abs(mean - scaler.scaler.means[i]) > 1e-10
                    or abs(std - scaler.scaler.stds[i]) > 1e-10
                    or len(values) != scaler.scaler.counts[i]):
                raise ValueError(f'independent scaler check failed: {route}/{role}/{task}')
        scalers[f'{route}_{role}'] = scaler.to_dict()
        if role == 'target':
            for task in PRIMARY:
                ds = GraphTaskView(view, task)
                graph = ds[0]  # First available training observation, no performance selection.
                raw = float(view.labels[ds.indices[0], view.tasks.index(task)])
                if abs(float(graph.y[0]) - raw) > 1e-6:
                    raise ValueError('raw graph label changed')
                graphs.append(dict(route=route, task=task, sample_id=graph.sample_id,
                    raw_y=raw, graph_y=float(graph.y[0]), nodes=int(graph.x.shape[0]),
                    feature_schema=graph.feature_schema_version))
    return dict(scope='READ_ONLY_DATA_AND_GRAPH_NOT_TRAINING_READY', input_identity=table.identity,
        scalers=scalers, graph_fixtures=graphs, scaler_tasks_checked=114,
        input_audit=table.report, training_executed=False, gpu_executed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('csv', 'split-manifest', 'tox-manifest', 'expected-tox-file-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('output exists; use a new run path')
    try:
        from dataset115_adapter import Dataset115Table
        table = Dataset115Table.load(args.csv, args.split_manifest, args.tox_manifest,
                                    expected_tox_sha=args.expected_tox_file_sha256)
        result = verify(table)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        with output.open('x', encoding='utf8') as f:
            f.write(payload)
    except Exception as exc:
        parser.exit(2, f'DATASET115_ADAPTER_CHECK_FAILED: {type(exc).__name__}: {exc}\n')
    print('READ_ONLY_ADAPTER_CHECKED: 114 train scalers; 10 training graph fixtures; NOT_TRAINING_READY')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
