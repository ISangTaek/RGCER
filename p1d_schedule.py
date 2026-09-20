"""Deterministic P1-D budget/alias planner, NOT a GPU training authorization.

The execution layer must bind input histories to actual validation artifacts
before calling this planner. It must not silently fall back to new training
when a reused asset cannot be located or fails verification.
"""
import math

SETTINGS = ('ToxAcute', 'A', 'B')
ARMS = ('B1_high', 'B1_low', 'HF_high', 'HF_low')
UPDATES = {'ToxAcute': 240, 'A': 600, 'B': 360}
S3_SEEDS = (42, 44, 46)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def job(setting, arm, seed, phase):
    require(setting in SETTINGS and arm in ARMS, 'setting/arm outside protocol')
    require(type(seed) is int and seed in range(42, 47), 'seed outside protocol')
    require((phase == 'screen' and seed == 42) or
            (phase == 'replication' and seed != 42), 'phase/seed mismatch')
    alias = None
    if arm == 'B1_high':
        alias = f'{setting}_B1_s{seed}_existing'
    elif setting == 'ToxAcute' and arm == 'HF_low' and seed in S3_SEEDS:
        alias = f's3_e40_s{seed}'
    return dict(run_id=f'{setting}_{arm}_s{seed}', setting=setting, arm=arm,
                seed=seed, phase=phase, action='REUSE' if alias else 'TRAIN',
                alias=alias, new_epochs=0 if alias else 40,
                new_updates=0 if alias else UPDATES[setting],
                missing_reuse_action='STOP_NOT_RETRAIN' if alias else None)


def screening_jobs():
    """Eight new trajectories and four reused controls, not twelve new runs."""
    return [job(s, a, 42, 'screen') for s in SETTINGS for a in ARMS]


def validate_selection(selection, setting):
    require(type(selection) is dict, 'selection must be mapping')
    require(set(selection) == {'setting', 'seed', 'scores', 'selected', 'scope',
                              'acceptance_status'}, 'selection fields')
    require(selection['setting'] == setting and type(selection['seed']) is int
            and selection['seed'] == 42, 'selection setting/seed')
    require(selection['scope'] == 'DEVELOPMENT_VALIDATION_ONLY', 'selection scope')
    require(selection['acceptance_status'] == 'PENDING_REVIEW',
            'mechanical selection is not external acceptance')
    scores = selection['scores']
    require(type(scores) is dict and set(scores) == set(ARMS), 'four scores required')
    for score in scores.values():
        require(type(score) is dict and set(score) == {'best_epoch', 'best_validation_macro_rmse'},
                'score fields')
        epoch = score['best_epoch']; value = score['best_validation_macro_rmse']
        require(type(epoch) is int and 0 <= epoch < 40, 'best epoch')
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                'nonfinite/invalid selection score')
    expected = {f: min((f + '_high', f + '_low'),
                      key=lambda a: scores[a]['best_validation_macro_rmse'])
                for f in ('B1', 'HF')}
    require(type(selection['selected']) is dict and selection['selected'] == expected,
            'selected arm conflicts with scores/high-LR tie rule')
    return expected


def resolve_plan(selections):
    """Resolve approved conditional branches; actual runner verifies evidence."""
    require(type(selections) is dict and set(selections) == set(SETTINGS),
            'all three setting selections required')
    jobs = screening_jobs()
    for setting in SETTINGS:
        winners = validate_selection(selections[setting], setting)
        for family in ('B1', 'HF'):
            jobs.extend(job(setting, winners[family], seed, 'replication')
                        for seed in range(43, 47))
    require(len({r['run_id'] for r in jobs}) == len(jobs) == 36, 'duplicate run')
    new = [r for r in jobs if r['action'] == 'TRAIN']
    epochs = sum(r['new_epochs'] for r in jobs)
    updates = sum(r['new_updates'] for r in jobs)
    require(len(new) <= 32 and epochs <= 1280 and updates <= 12960,
            'post-reuse budget overflow')
    return dict(schema='p1d_resolved_schedule_v1', jobs=jobs,
                budget=dict(new_trajectories=len(new), model_epochs=epochs,
                            optimizer_updates=updates, smoke_updates_cap=36,
                            saved_budget_reallocation=False),
                test_authorized=False, source_pretraining_authorized=False,
                execution_authorized=False,
                reason='Requires separately released runtime, asset checks and READY card')
