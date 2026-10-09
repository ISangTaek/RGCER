import numpy as np
import pytest

import v12_analytic as a


@pytest.fixture(autouse=True)
def numerical_threads():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield


def test_posterior_matches_independent_augmented_weighted_least_squares():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(17, 4)), rng.normal(size=17)
    p = np.diag([1., 2., 3., 4.]); mean = rng.normal(size=4)
    weight = np.linspace(.2, 2, 17)
    fit = a.fit_linear(x, y, p, mean, weight)
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.column_stack((np.zeros(4), np.linalg.cholesky(p).T))
    lhs = np.vstack((design*np.sqrt(weight)[:, None], penalty))
    rhs = np.concatenate((y*np.sqrt(weight), np.linalg.cholesky(p).T@mean))
    expected = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    assert np.allclose(np.r_[fit.intercept, fit.coefficient], expected, atol=1e-12)


def test_duplicate_rows_with_split_weights_do_not_strengthen_likelihood():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(9, 3)), rng.normal(size=9)
    first = a.fit_linear(x, y, np.eye(3))
    repeated = a.fit_linear(np.repeat(x, 2, 0), np.repeat(y, 2), np.eye(3), weights=np.full(18, .5))
    assert np.allclose(first.predict(x), repeated.predict(x), atol=1e-12)


@pytest.mark.parametrize('kind', ['negative_precision', 'asymmetric', 'nan', 'negative_weight', 'one_point', 'wrong_prior'])
def test_invalid_posterior_inputs_fail_closed(kind):
    x, y, p, mean, weights = np.ones((3, 2)), np.ones(3), np.eye(2), np.zeros(2), np.ones(3)
    if kind == 'negative_precision': p[0, 0] = -1
    elif kind == 'asymmetric': p[0, 1] = .2
    elif kind == 'nan': x[0, 0] = np.nan
    elif kind == 'negative_weight': weights[0] = -1
    elif kind == 'one_point': x, y, weights = x[:1], y[:1], weights[:1]
    else: mean = np.ones(3)
    with pytest.raises(ValueError): a.fit_linear(x, y, p, mean, weights)


def test_source_preprocessing_roundtrip_and_constant_features():
    rng = np.random.default_rng(3)
    f, q = rng.normal(size=(20, 12)), rng.normal(size=(20, 8))
    f[:, 0] = 1; q[:, 0] = 3
    transform = a.SourceTransform.fit(f, q)
    state = transform.state()
    loaded = a.SourceTransform.from_state(state)
    z, chem = transform.transform(f, q)
    assert z.shape == (20, 8) and np.isfinite(chem).all()
    assert np.allclose(z, loaded.transform(f, q)[0])
    assert np.allclose(chem[:, 0], 0)


def test_source_prior_uses_every_task_and_equal_task_weight():
    z = np.linspace(-2, 2, 30)[:, None]
    tasks = ['a']*10+['b']*20
    y = np.r_[z[:10, 0], -z[10:, 0]]
    result = a.source_prior(z, y, tasks, ['a', 'b'])
    assert result['counts'] == {'a':10, 'b':20}
    assert np.allclose(result['mean'], np.mean(result['task_coefficients'], axis=0))
    assert np.linalg.eigvalsh(result['precision']).min() > 0
    with pytest.raises(ValueError): a.source_prior(z, y, tasks, ['a', 'c'])


@pytest.mark.parametrize('method', a.METHODS)
def test_predictor_serialization_permutation_and_label_affine_equivariance(method):
    import json
    rng = np.random.default_rng(6)
    z, q, y = rng.normal(size=(23, 3)), rng.normal(size=(23, 5)), rng.normal(size=23)
    prior = dict(mean=[.1, -.2, .3], precision=np.eye(3).tolist())
    state = a.fit_model(z, q, y, prior, method, 1.)
    pred = a.predict_model(json.loads(json.dumps(state)), z, q)
    assert np.allclose(pred[::-1], a.predict_model(state, z[::-1], q[::-1]))
    shifted = a.fit_model(z, q, 3*y+7, prior, method, 1.)
    assert np.allclose(a.predict_model(shifted, z, q), 3*pred+7, atol=1e-10)


def test_kernel_agrees_with_independent_intercept_block_system():
    rng = np.random.default_rng(7)
    z, q, y = rng.normal(size=(13, 2)), rng.normal(size=(13, 3)), rng.normal(size=13)
    state = a.fit_model(z, q, y, {}, 'KERNEL', 1.)
    x = np.column_stack((z, q)); k = a.kernel(x, x, state['bandwidth'])
    lhs = np.block([[k+np.eye(len(x)), np.ones((len(x),1))], [np.ones((1,len(x))), np.zeros((1,1))]])
    solution = np.linalg.solve(lhs, np.r_[y, 0])
    expected = k@solution[:-1]+solution[-1]
    assert np.allclose(a.predict_model(state, z, q), expected, atol=1e-11)


def test_structure_residual_and_prior_ablation_change_predictions():
    rng = np.random.default_rng(8)
    z, q = rng.normal(size=(12, 2)), rng.normal(size=(12, 4))
    y = 2*q[:, 0]+.2*z[:, 0]
    prior = dict(mean=[5., -5.], precision=(np.eye(2)*10).tolist())
    predictions = [a.predict_model(a.fit_model(z,q,y,prior,m,1.),z,q) for m in ('C3','C3_ZERO_MEAN','C3_NO_CHEM')]
    assert all(not np.allclose(predictions[0], p) for p in predictions[1:])
