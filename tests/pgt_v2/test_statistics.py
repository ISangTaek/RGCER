import numpy as np
import pytest
from pgt_v2.statistics import pooled_oof, paired_group_bootstrap


def test_oof_pools_rows_before_rmse_and_then_averages_seed_metrics():
    tables=[]
    for seed in (42,43,44):
        for fold,count in ((0,1),(1,9)):
            tables.append(dict(seed=seed,fold=fold,task_names=['task'],record_id=[f'{fold}_{i}' for i in range(count)],
                group_id=[f'g{fold}_{i}' for i in range(count)],labels=np.zeros((count,1)),
                prediction=np.full((count,1),10. if fold==0 else 0.)))
    result=pooled_oof(tables,['task'],folds=2)
    assert result['macro_rmse']==pytest.approx(np.sqrt(10.))
    assert result['macro_rmse']!=5.
    tables[-1]['group_id'][0]='g0_0'
    with pytest.raises(ValueError,match='crossed'):pooled_oof(tables,['task'],folds=2)


def test_paired_bootstrap_fixed_tasks_and_adverse_endpoint():
    truth=np.zeros((8,2))
    left=np.tile([[1.,3.]],(3,8,1));right=np.tile([[2.,2.]],(3,8,1))
    groups=[f'g{i}' for i in range(8)];folds=[0]*4+[1]*4
    result=paired_group_bootstrap(truth,left,right,groups,folds,attempts=128)
    assert result['per_endpoint_delta']==[-1.,1.]
    assert result['delta_macro_rmse']==0. and result['confidence_95']==[0.,0.]
    assert result['invalid']==0
    sparse=truth.copy();sparse[1:,1]=np.nan
    bad=paired_group_bootstrap(sparse,left,right,groups,folds,attempts=128)
    assert bad['invalid']>0 and bad['valid']+bad['invalid']==128 and bad['unstable']
    # No resampling until a favorable or complete sample appears.
    assert bad==paired_group_bootstrap(sparse,left,right,groups,folds,attempts=128)
