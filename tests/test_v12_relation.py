from collections import Counter
import numpy as np
import pytest

import v12_relation as r
from v12_s1_data import fold_id


def test_pair_degree_determinism_and_no_self_pairs():
    molecules = ['m'+str(i) for i in range(60)]
    pairs = r.make_pairs(molecules, 30)
    assert pairs == r.make_pairs(molecules[::-1], 30)
    count = Counter(m for pair in pairs for m in pair)
    assert len(count) == 30 and max(count.values()) <= 4
    assert all(a != b for a,b in pairs)


def test_pair_mass_equals_unique_groups_not_number_of_pairs():
    pairs = [('a','b'),('a','c'),('b','c'),('c','d')]
    groups = dict(a='g0',b='g0',c='g1',d='g2')
    assert r.group_weights(pairs,groups).sum() == pytest.approx(3.)


def test_conditioned_relation_recovers_known_conditional_slope_out_of_sample():
    rng = np.random.default_rng(18)
    ds, c = rng.normal(size=900), rng.normal(size=(900,4))
    target = (1.+2*c[:,0])*ds
    fit = r.fit_relation(ds[:600], target[:600], c[:600], np.ones(600))
    pred = r.predict_relation(fit, ds[600:], c[600:])
    errors = {m:np.mean((p-target[600:])**2) for m,p in pred.items()}
    assert errors['CONDITIONAL'] < .001*errors['GLOBAL']
    assert errors['CONDITIONAL'] < .001*errors['SHUFFLED_CONDITION']


def test_zero_difference_and_pair_exchange_antisymmetry():
    rng = np.random.default_rng(19)
    ds, c, target = rng.normal(size=20), rng.normal(size=(20,4)), rng.normal(size=20)
    fit = r.fit_relation(ds,target,c,np.ones(20))
    p, negative, zero = (r.predict_relation(fit, x, c) for x in (ds,-ds,np.zeros(20)))
    for method in r.MODELS:
        assert np.allclose(p[method], -negative[method])
        assert np.array_equal(zero[method], np.zeros(20))


def test_group_folds_are_stable_and_never_use_labels():
    ids = [fold_id('g'+str(i)) for i in range(100)]
    assert ids == [fold_id('g'+str(i)) for i in range(100)]
    assert set(ids) == {0,1,2}
    with pytest.raises(ValueError): fold_id('',3)
    with pytest.raises(ValueError): fold_id('g',4)


def test_intercept_correction_degeneracy_identity():
    f_query, f_anchors = np.array([1.,3.,-2.]), np.array([.5,2.,7.])
    y_anchors = np.array([1.,1.,9.])
    anchor_prediction = (y_anchors[None,:]+f_query[:,None]-f_anchors[None,:]).mean(1)
    assert np.allclose(anchor_prediction, f_query+(y_anchors-f_anchors).mean())
