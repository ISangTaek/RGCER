"""Pooled OOF estimates and paired, within-fold structure-group bootstrap."""
import numpy as np

from .training import metrics


def pooled_oof(tables, tasks, *, seeds=(42,43,44), folds=5):
    """One table per seed/fold; preserve source observations, pool before RMSE."""
    if {(int(t['seed']),int(t['fold'])) for t in tables} != {(s,f) for s in seeds for f in range(folds)} or len(tables)!=len(seeds)*folds:
        raise ValueError('Incomplete or duplicate OOF seed/fold cells')
    rows_by_seed=[];reference=None
    for seed in seeds:
        selected=sorted((t for t in tables if int(t['seed'])==seed),key=lambda t:int(t['fold']))
        ids=[];groups=[];fold_id=[];truth=[];prediction=[];group_fold={}
        for table in selected:
            if list(table['task_names'])!=list(tasks):
                raise ValueError('OOF task order mismatch')
            if len(table['record_id'])!=len(table['labels']) or len(table['group_id'])!=len(table['labels']):
                raise ValueError('OOF identity row count mismatch')
            for group in table['group_id']:
                if group in group_fold and group_fold[group]!=int(table['fold']):
                    raise ValueError('A structure group crossed outer test folds')
                group_fold[group]=int(table['fold'])
            ids.extend(table['record_id']);groups.extend(table['group_id'])
            fold_id.extend([int(table['fold'])]*len(table['record_id']))
            truth.extend(table['labels']);prediction.extend(table['prediction'])
        if len(set(ids))!=len(ids):
            raise ValueError('An OOF source record was predicted in multiple folds')
        order=np.argsort(np.array(ids))
        aligned=dict(record_id=np.asarray(ids)[order],group_id=np.asarray(groups)[order],fold=np.asarray(fold_id)[order],
                     labels=np.asarray(truth,dtype=float)[order],prediction=np.asarray(prediction,dtype=float)[order])
        if reference is None:
            reference=aligned
        else:
            for key in ('record_id','group_id','fold'):
                if not np.array_equal(reference[key],aligned[key]):
                    raise ValueError('OOF identity changed across seeds')
            if not np.array_equal(reference['labels'],aligned['labels'],equal_nan=True):
                raise ValueError('OOF truth changed across seeds')
        rows_by_seed.append(metrics(aligned['labels'],aligned['prediction'],tasks,aligned['group_id']))
    return dict(aggregation='pool all outer folds per seed -> endpoint RMSE -> equal macro -> mean of seeds',
        seeds=list(seeds),per_seed=rows_by_seed,macro_rmse=float(np.mean([r['macro_rmse'] for r in rows_by_seed])),
        source_records=len(reference['record_id']),independent_groups=len(set(reference['group_id'])))


def paired_group_bootstrap(truth,left,right,groups,fold_ids,*,attempts=10000,seed=20261016):
    """Prediction arrays are [seed,row,task]; negative difference favors left."""
    truth=np.asarray(truth,dtype=np.float64);left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    groups=np.asarray(groups);fold_ids=np.asarray(fold_ids)
    if (left.shape!=right.shape or left.ndim!=3 or left.shape[1:]!=truth.shape or
        truth.shape[0]!=len(groups) or len(groups)!=len(fold_ids) or attempts<1 or
        not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('Invalid paired OOF arrays')
    observed=np.isfinite(truth);safe=np.where(observed,truth,0.)
    count=observed.sum(0)
    if (count==0).any():
        raise ValueError('Fixed endpoint denominator has absent truth')
    unique=sorted(set(groups.tolist()));lookup={g:i for i,g in enumerate(unique)}
    positions=np.asarray([lookup[g] for g in groups])
    by_fold={}
    for group in unique:
        f=set(fold_ids[groups==group].tolist())
        if len(f)!=1:
            raise ValueError('Paired structure group crossed folds')
        by_fold.setdefault(next(iter(f)),[]).append(lookup[group])
    counts=np.zeros((len(unique),truth.shape[1]))
    np.add.at(counts,positions,observed.astype(float))
    sums=[]
    for predictions in (left,right):
        error=np.where(observed[None],(predictions-safe[None])**2,0.)
        grouped=np.zeros((left.shape[0],len(unique),truth.shape[1]))
        for s in range(left.shape[0]):
            np.add.at(grouped[s],positions,error[s])
        sums.append(grouped)
    def differences(weights):
        n=weights@counts
        if (n==0).any():return None
        per_endpoint=[np.sqrt(np.einsum('g,sgt->st',weights,s)/n[None]).mean(0) for s in sums]
        return per_endpoint[0]-per_endpoint[1]
    point=differences(np.ones(len(unique)))
    rng=np.random.default_rng(seed);draws=[];invalid=0
    for _ in range(attempts):
        weights=np.zeros(len(unique))
        for indices in by_fold.values():
            draws_indices=rng.choice(indices,size=len(indices),replace=True)
            weights+=np.bincount(draws_indices,minlength=len(unique))
        difference=differences(weights)
        if difference is None:
            invalid+=1
        else:
            draws.append(float(difference.mean()))
    ci=lambda q:np.quantile(draws,q).tolist() if draws else None
    return dict(delta_macro_rmse=float(point.mean()),per_endpoint_delta=point.tolist(),
        confidence_95=ci([.025,.975]),confidence_bonferroni_98_3333=ci([.05/6,1-.05/6]),
        attempts=attempts,valid=len(draws),invalid=invalid,invalid_rate=invalid/attempts,
        unstable=invalid/attempts>.05,bootstrap_seed=seed,
        sampling='paired group resampling within each outer fold, shared across every endpoint and seed',
        denominator='all fixed tasks; no redraw or omission when a task is absent')
