"""V4 preparation, T01 execution, and separately authorized H01 search."""
import argparse
from pathlib import Path
import torch

from .artifacts import load_manifest, read_json, write_json
from .config import candidate, git_identity, resolve_run


def prepare_author(views, groups, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sparse = read_json(groups)['S21']
    runs = []
    for view_name in ('FULL', 'S21_P10'):
        manifest = load_manifest(Path(views)/view_name)
        if manifest['fold'] != 'S0' or len(manifest['task_names']) != 59:
            raise ValueError('T01 requires the formal S0/59-task assets')
        for seed in (42, 43, 44):
            cfg = candidate('TOX_AUTHOR', 'AUTHOR')
            # T01 compares directly with F01's ALL59-selected M3. Save the
            # S21-selected weights for later H01 comparisons, without refitting.
            cfg.update(selection_tasks=None, auxiliary_selection_tasks=sparse,
                auxiliary_checkpoint='best_s21.pt', auxiliary_role='S21')
            name = f'T01_TOX_AUTHOR_S0_{view_name}_S{seed}'
            row = dict(run_id=name, experiment='T01', method_id='TOX_AUTHOR', config_id='AUTHOR',
                phase='P3', state='READY', fold='S0', init_seed=seed, code_commit=git_identity(), view_name=view_name)
            run = resolve_run(row, view_manifest=manifest, explicit_config=cfg)
            write_json(output/'runs'/(name+'.json'), run)
            runs.append(name)
    write_json(output/'release.json', dict(experiment='T01', code_commit=git_identity(), runs=runs,
        fit_budget=6, auto_next_stage=False, author_source_commit='704edcb951ab397e31561bc130154d01e0103b3c'))
    return runs


def run_author(release, views, cache, output, seed, device):
    from .training import run_training
    from .evaluation import sealed_predict, score_prediction
    from .cache import file_sha256
    release, output = Path(release), Path(output)
    info = read_json(release/'release.json')
    if info['code_commit'] != git_identity() or seed not in (42, 43, 44) or info['fit_budget'] != 6:
        raise ValueError('Wrong T01 release/commit/seed')
    for name in info['runs']:
        run_path = release/'runs'/(name+'.json')
        run = read_json(run_path)
        if run['init_seed'] != seed:
            continue
        dest = output/name
        if (dest/'complete.json').exists():
            if read_json(dest/'complete.json')['run_sha256'] != run['run_sha256']:
                raise ValueError('Completed run differs')
            continue
        dest.mkdir(parents=True, exist_ok=True)
        train = dest/'training'
        view = Path(views)/run['view_name']
        summary = read_json(train/'summary.json') if (train/'summary.json').exists() else None
        if not summary or summary['terminal'] != 'COMPLETE':
            summary = run_training(run_path, view, cache, train, device=device,
                resume=(train/'last.pt').exists(), verify_database=False)
        if summary['identity']['run_sha256'] != run['run_sha256'] or summary['terminal'] != 'COMPLETE':
            raise ValueError('Training incomplete or mismatched')
        for role in ('validation', 'test'):
            pred = dest/role
            if not (pred/'manifest.json').exists():
                sealed_predict(train/'best.pt', view, cache, pred, role=role, device=device, verify_database=False)
            metrics = dest/(role+'_metrics.json')
            if not metrics.exists():
                score_prediction(pred, view, metrics, expected_checkpoint_sha256=file_sha256(train/'best.pt'))
        write_json(dest/'complete.json', dict(run_sha256=run['run_sha256'], status='COMPLETE', seed=seed))
    return dict(status='T01_SEED_COMPLETE', seed=seed, next_stage_released=False)


def preflight_author(release, views, cache, output, device):
    import numpy as np
    from .artifacts import load_arrays
    from .cache import GraphCache
    from .models import build_model
    from .training import ModelInputs, training_statistics
    info = read_json(Path(release)/'release.json')
    if info['code_commit'] != git_identity() or not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Server check requires the released commit and CUDA')
    run = read_json(Path(release)/'runs'/(info['runs'][0]+'.json'))
    view = Path(views)/run['view_name']
    manifest = load_manifest(view)
    if run['view_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Wrong formal view')
    train = load_arrays(view, 'train.npz', manifest)
    stats = training_statistics(load_arrays(view, 'statistics.npz', manifest), 'TOX_AUTHOR')
    graph_cache = GraphCache(cache, source_sha256=manifest['source_sha256'], verify_database=False)
    if graph_cache.manifest['cache_id'] != manifest['cache_id']:
        raise ValueError('Wrong accepted cache')
    indices = np.flatnonzero(np.isfinite(train['labels']).any(1))[:16].tolist()
    model = build_model('TOX_AUTHOR', manifest['task_names'], run['config'], adjacency=stats['adjacency'], device=device).eval()
    with torch.no_grad():
        y = model(ModelInputs(graph_cache, train, 'TOX_AUTHOR', device).batch(indices))
    if tuple(y.shape) != (len(indices), 59) or not torch.isfinite(y).all():
        raise ValueError('Invalid author-model forward')
    result = dict(commit=git_identity(), model='TOX_AUTHOR', device=device, output_shape=list(y.shape),
                  finite=True, optimizer_steps=0, raw_label_scale=True)
    write_json(output, result)
    return result


def collect_author(work):
    import zipfile
    from .cache import file_sha256
    work = Path(work)
    info = read_json(work/'release/release.json')
    complete = [name for name in info['runs'] if (work/'results'/name/'complete.json').exists()]
    receipt = dict(experiment='T01', code_commit=info['code_commit'], expected=6, completed=len(complete),
        completed_runs=complete, status='COMPLETE' if len(complete)==6 else 'PARTIAL', scientific_review='PENDING')
    write_json(work/'return_receipt.json', receipt)
    archive = work/'PGTV2_T01_RESULTS_20261028.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        package.write(work/'return_receipt.json', 'return_receipt.json')
        for folder in ('release', 'results', 'operations'):
            for path in sorted((work/folder).rglob('*')):
                if path.is_file() and path.name not in ('last.pt', 'last.pt.sha256.json'):
                    package.write(path, path.relative_to(work).as_posix())
    identity = dict(file=archive.name, bytes=archive.stat().st_size, sha256=file_sha256(archive), **receipt)
    write_json(work/'return_package_identity.json', identity)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare-author')
    p.add_argument('--views', required=True); p.add_argument('--groups', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('run-author')
    for arg in ('release', 'views', 'cache', 'output'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--seed', type=int, choices=(42,43,44), required=True)
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('preflight-author')
    for arg in ('release', 'views', 'cache', 'output'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('collect-author'); p.add_argument('--work', required=True)
    p = sub.add_parser('hpo')
    for arg in ('method', 'view', 'groups', 'cache', 'output'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--resume', action='store_true')
    p.add_argument('--trials', type=int, choices=(12,24), default=12)
    args = vars(parser.parse_args())
    command = args.pop('command')
    torch.set_num_threads(1)
    if command == 'prepare-author':
        result = prepare_author(**args)
    elif command == 'run-author':
        result = run_author(**args)
    elif command == 'preflight-author':
        result = preflight_author(**args)
    elif command == 'collect-author':
        result = collect_author(**args)
    else:
        from .hpo import run_search
        result = run_search(**args)
    print(result, flush=True)


if __name__ == '__main__':
    main()
