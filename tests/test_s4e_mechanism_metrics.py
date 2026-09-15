import numpy as np
import pytest
from s4e_mechanism_metrics import (linear_cka, parameter_drift, functional_forgetting,
                                   hybrid_state, DesignError)


def test_cka_independent_gram_reference():
    rng = np.random.RandomState(17)
    x, y = rng.randn(15, 4), rng.randn(15, 7)
    ids = [str(i) for i in range(15)]
    center = np.eye(15)-np.ones((15, 15))/15
    k, l = center@x@x.T@center, center@y@y.T@center
    expected = np.sum(k*l)/(np.linalg.norm(k)*np.linalg.norm(l))
    assert linear_cka(x, y, ids, ids)['value'] == pytest.approx(expected, abs=1e-13)


def test_cka_invariances_and_degenerate():
    x = np.arange(12).reshape(4, 3); ids = ['a','b','c','d']
    assert linear_cka(x, -3*x+5, ids, ids)['value'] == pytest.approx(1)
    assert linear_cka(x, np.ones_like(x), ids, ids)['reason'] == 'zero_centered_norm'
    assert linear_cka(x*1e200, x, ids, ids)['value'] == pytest.approx(1)


@pytest.mark.parametrize('ids', [['b','a','c'], ['a','a','c'], ['a','b']])
def test_cka_pairing_rejected(ids):
    with pytest.raises(DesignError): linear_cka(np.eye(3), np.eye(3), ['a','b','c'], ids)


@pytest.mark.parametrize('x', [np.array([[np.nan],[1],[2]]), np.array([[np.inf],[1],[2]]), np.ones(3)])
def test_cka_invalid_values(x):
    with pytest.raises(DesignError): linear_cka(x, np.eye(3), ['a','b','c'], ['a','b','c'])


def test_drift_values_and_zero_reference():
    a = {'x': np.array([3.,4.]), 'y': np.array([12.])}
    assert parameter_drift(a, a)['relative_l2'] == 0
    b = {k: 2*v for k,v in a.items()}
    out = parameter_drift(a, b)
    assert out['relative_l2'] == 1 and out['reference_l2'] == 13
    assert parameter_drift({'x':np.zeros(2)}, {'x':np.ones(2)})['relative_l2'] is None


def test_drift_key_shape_errors():
    with pytest.raises(DesignError): parameter_drift({'a':[1]}, {'b':[1]})
    with pytest.raises(DesignError): parameter_drift({'a':[1]}, {'a':[1,2]})


def test_hybrid_preserves_nonbackbone_and_rejects_partial():
    src = {'encoder.backbone.x':np.ones(2,dtype='f4'), 'decoders.a':np.zeros(2,dtype='f4')}
    bb = {'encoder.backbone.x':np.zeros(2,dtype='f4')}
    out = hybrid_state(src, bb)
    assert out['decoders.a'] is src['decoders.a']
    assert np.array_equal(src['encoder.backbone.x'], np.ones(2))
    with pytest.raises(DesignError): hybrid_state(src, {})
    with pytest.raises(DesignError): hybrid_state(src, {'encoder.backbone.x':np.zeros(2,dtype='f8')})


def test_functional_direction_and_empty_macro():
    labels={'a':[0.,0.], 'b':[]}; ids={'a':['x','y'], 'b':[]}
    s={'a':(['x','y'],[1.,1.]), 'b':([],[])}
    h={'a':(['x','y'],[2.,2.]), 'b':([],[])}
    out = functional_forgetting(labels,s,h,['a','b'],ids)
    assert out['tasks'][0]['delta_rmse'] == 1
    assert out['macro_delta_rmse'] is None and len(out['tasks']) == 2
    out = functional_forgetting({'a':[0.,0.]},{'a':s['a']},{'a':s['a']},['a'],{'a':ids['a']})
    assert out['macro_delta_rmse'] == 0


def test_functional_wrong_identity_rejected():
    with pytest.raises(DesignError):
        functional_forgetting({'a':[0.]},{'a':(['bad'],[1.])},{'a':(['x'],[1.])},['a'],{'a':['x']})
