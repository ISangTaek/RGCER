"""Bounded V4 Optuna search. Only training and validation assets are read."""
import pickle
from pathlib import Path

from .artifacts import load_manifest, read_json, write_json
from .config import candidate, git_identity, resolve_run
from .training import run_training

OPTUNA_VERSION = '4.5.0'
REFERENCE = dict(layers=4, hidden_dim=96, heads=4, ffn_ratio=2, dropout=.1,
                 lr=3e-4, weight_decay=1e-4)


def suggest_config(trial, method, targets):
    cfg = candidate(method, 'C2')
    cfg.pop('prompt_rank', None)
    cfg.update(layers=trial.suggest_categorical('layers', [2, 4, 6, 8]),
        hidden_dim=trial.suggest_categorical('hidden_dim', [64, 96, 128, 192]),
        heads=trial.suggest_categorical('heads', [4, 8]),
        dropout=trial.suggest_categorical('dropout', [0., .1, .2, .3]),
        lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True))
    cfg['ffn_dim'] = cfg['hidden_dim'] * trial.suggest_categorical('ffn_ratio', [2, 4])
    cfg['selection_tasks'] = list(targets)
    return cfg


def export_trials(study, output):
    rows = [dict(number=t.number, state=t.state.name, value=t.value,
                 params=t.params, attributes=t.user_attrs) for t in study.trials]
    write_json(Path(output)/'trials.json', rows)
    complete = sorted((r for r in rows if r['state'] == 'COMPLETE'), key=lambda r: (r['value'], r['number']))
    write_json(Path(output)/'top2_validation.json', complete[:2])


def run_search(method, view, groups, cache, output, *, device='cuda:0', resume=False, trials=24):
    import optuna
    from optuna.trial import TrialState
    if optuna.__version__ != OPTUNA_VERSION:
        raise ValueError('Use optuna==' + OPTUNA_VERSION)
    if method not in ('N1', 'N1S'):
        raise ValueError('H01 searches only N1 and N1S')
    if trials not in (12, 24):
        raise ValueError('Use the released 12- or 24-trial budget')
    manifest = load_manifest(view)
    targets = read_json(groups)['S21']
    if manifest['fold'] != 'S0' or not set(targets) <= set(manifest['task_names']):
        raise ValueError('H01 requires the registered S0 view and S21 task list')
    # Check the label budget, without opening any test labels.
    from .artifacts import load_arrays
    counts = load_arrays(view, 'statistics.npz', manifest)['count']
    target_counts = [int(counts[manifest['task_names'].index(t)]) for t in targets]
    if len(targets) != 21 or sum(target_counts) != 228 or min(target_counts) != 8 or max(target_counts) != 14:
        raise ValueError('H01 requires the frozen S0/S21_P10 label budget')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    binding = dict(method=method, view=manifest['manifest_sha256'], tasks=targets,
                   commit=git_identity(), optuna=OPTUNA_VERSION, budget=trials, seed=42, sampler_seed=20261028)
    binding_file = output/'study_config.json'
    if binding_file.exists() and read_json(binding_file) != binding:
        raise ValueError('Study method/data/code/selection configuration changed')
    write_json(binding_file, binding)
    sampler_file = output/'sampler.pkl'
    # This is our own local sampler state, never an untrusted downloaded pickle.
    sampler = (pickle.loads(sampler_file.read_bytes()) if sampler_file.exists() else
        optuna.samplers.TPESampler(seed=20261028, n_startup_trials=8, multivariate=False, constant_liar=False))
    study = optuna.create_study(study_name=method+'_S0_P10', direction='minimize',
        storage='sqlite:///'+(output/'study.sqlite3').as_posix(), load_if_exists=resume,
        sampler=sampler, pruner=optuna.pruners.NopPruner())
    if not study.trials:
        study.enqueue_trial(REFERENCE)
    while True:
        running = [t for t in study.trials if t.state == TrialState.RUNNING]
        if len(running) > 1:
            raise RuntimeError('Only one worker per study is allowed')
        if running:
            trial = optuna.trial.Trial(study, running[0]._trial_id)
        else:
            attempted = [t for t in study.trials if t.state != TrialState.WAITING]
            if len(attempted) >= trials:
                break
            trial = study.ask()
        trial_dir = output/f'trial_{trial.number:03d}'
        trial_dir.mkdir(exist_ok=True)
        config = suggest_config(trial, method, targets)
        row = dict(method_id=method, config_id=f'TRIAL_{trial.number:03d}', fold='S0', phase='P2',
            state='READY', init_seed=42, code_commit=binding['commit'], experiment='H01')
        run = resolve_run(row, view_manifest=manifest, explicit_config=config)
        run_path = trial_dir/'run.json'
        if run_path.exists() and read_json(run_path) != run:
            raise ValueError('Interrupted trial configuration changed')
        write_json(run_path, run)
        temp = sampler_file.with_suffix('.tmp')
        temp.write_bytes(pickle.dumps(sampler)); temp.replace(sampler_file)
        training = trial_dir/'training'
        try:
            summary = read_json(training/'summary.json') if (training/'summary.json').exists() else None
            if not summary or summary['terminal'] != 'COMPLETE':
                summary = run_training(run_path, view, cache, training, device=device,
                    resume=(training/'last.pt').exists(), verify_database=False)
            if summary['terminal'] != 'COMPLETE':
                raise RuntimeError('Trial has not finished its frozen stopping rule')
            trial.set_user_attr('validation', summary['validation'])
            trial.set_user_attr('best_epoch', summary['best_epoch'])
            trial.set_user_attr('parameters', summary['parameters'])
            study.tell(trial, summary['selection_score'])
        except Exception as exc:
            trial.set_user_attr('failure', str(exc))
            study.tell(trial, state=TrialState.FAIL)
            export_trials(study, output)
            raise  # failed trial counts; never silently replace it or launch extra jobs
        export_trials(study, output)
    export_trials(study, output)
    return dict(status='H01_COMPLETE_REVIEW_REQUIRED', attempted=len(study.trials), test_evaluations=0)
