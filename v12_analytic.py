"""Small conditional empirical-Bayes regressors for V12-S1.

No graph training, validation-driven preprocessing, or claim of calibrated
posterior uncertainty. Priors are estimated from source tasks, equally weighted.
"""
from dataclasses import dataclass

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array(value, ndim):
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == ndim and np.isfinite(result).all(), 'finite array shape')
    return result


@dataclass
class LinearFit:
    coefficient: np.ndarray
    intercept: float

    def predict(self, x):
        x = array(x, 2)
        require(x.shape[1] == len(self.coefficient), 'linear feature width')
        return x @ self.coefficient + self.intercept

    def state(self):
        return dict(coefficient=self.coefficient.tolist(), intercept=self.intercept)

    @classmethod
    def from_state(cls, state):
        require(set(state) == {'coefficient', 'intercept'}, 'linear state schema')
        coef = array(state['coefficient'], 1)
        intercept = float(state['intercept'])
        require(np.isfinite(intercept), 'finite intercept')
        return cls(coef, intercept)


def fit_linear(x, y, precision, mean=None, weights=None):
    """Weighted sum squared loss plus a quadratic prior; free intercept.

    The supplied weights are not renormalized. Callers specify their effective
    likelihood mass, so pair replication need not increase regularization power.
    """
    x, y, p = array(x, 2), array(y, 1), array(precision, 2)
    n, width = x.shape
    require(n >= 2 and len(y) == n and p.shape == (width, width), 'linear dimensions')
    require(np.allclose(p, p.T, rtol=0, atol=1e-12)
            and np.linalg.eigvalsh(p).min() > 0, 'positive definite precision')
    mean = np.zeros(width) if mean is None else array(mean, 1)
    w = np.ones(n) if weights is None else array(weights, 1)
    require(mean.shape == (width,) and w.shape == (n,) and (w >= 0).all()
            and (w > 0).sum() >= 2, 'prior mean/weights')
    total = w.sum()
    xm, ym = (x * w[:, None]).sum(0) / total, float(y @ w / total)
    xc, yc = x - xm, y - ym
    lhs = xc.T @ (w[:, None] * xc) + p
    rhs = xc.T @ (w * yc) + p @ mean
    coefficient = np.linalg.solve(lhs, rhs)
    require(np.isfinite(coefficient).all(), 'finite analytic solution')
    return LinearFit(coefficient, float(ym - xm @ coefficient))


@dataclass
class SourceTransform:
    function_mean: np.ndarray
    function_scale: np.ndarray
    components: np.ndarray
    component_scale: np.ndarray
    chemical_mean: np.ndarray
    chemical_scale: np.ndarray

    @classmethod
    def fit(cls, functions, chemistry, rank=8):
        f, q = array(functions, 2), array(chemistry, 2)
        require(len(f) == len(q) and len(f) >= 2 and type(rank) is int and rank > 0,
                'source preprocessing population')
        fm, fs = f.mean(0), f.std(0)
        fs = np.where(fs > 1e-6, fs, 1.)
        centered = (f - fm) / fs
        covariance = centered.T @ centered / len(f)
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        keep = np.argsort(eigenvalue, kind='stable')[::-1][:min(rank, f.shape[1], len(f)-1)]
        components = eigenvector[:, keep]
        # Fix the otherwise arbitrary eigenvector signs for portable artifacts.
        for j in range(components.shape[1]):
            i = np.abs(components[:, j]).argmax()
            if components[i, j] < 0:
                components[:, j] *= -1
        zs = np.sqrt(np.maximum(eigenvalue[keep], 1e-12))
        qm, qs = q.mean(0), q.std(0)
        qs = np.where(qs > 1e-6, qs, 1.)
        return cls(fm, fs, components, zs, qm, qs)

    def transform(self, functions, chemistry):
        f, q = array(functions, 2), array(chemistry, 2)
        require(len(f) == len(q) and f.shape[1] == len(self.function_mean)
                and q.shape[1] == len(self.chemical_mean), 'transform dimensions')
        z = ((f - self.function_mean) / self.function_scale) @ self.components / self.component_scale
        q = (q - self.chemical_mean) / self.chemical_scale
        require(np.isfinite(z).all() and np.isfinite(q).all(), 'finite transformed features')
        return z, q

    def state(self):
        return {name: value.tolist() for name, value in vars(self).items()}

    @classmethod
    def from_state(cls, state):
        require(set(state) == set(cls.__dataclass_fields__), 'transform state schema')
        values = {k: array(v, 2 if k == 'components' else 1) for k, v in state.items()}
        require(all((values[k] > 0).all() for k in ('function_scale', 'component_scale', 'chemical_scale')),
                'transform positive scales')
        return cls(**values)


def source_prior(z, labels, tasks, source_tasks):
    """All source train rows fit their task; equal-task mean and shrunk covariance."""
    z, labels = array(z, 2), array(labels, 1)
    tasks = np.asarray(tasks)
    require(len(z) == len(labels) == len(tasks) and len(source_tasks) >= 2
            and set(tasks) == set(source_tasks), 'complete source prior tasks')
    coefficients, counts = [], {}
    for task in source_tasks:
        ids = np.flatnonzero(tasks == task)
        require(len(ids) >= 2, 'source prior task needs two observations')
        y = labels[ids]
        scaled = (y - y.mean()) / max(float(y.std()), 1e-6)
        fit = fit_linear(z[ids], scaled, np.eye(z.shape[1]) * 10.)
        coefficients.append(fit.coefficient)
        counts[task] = len(ids)
    coefficients = np.asarray(coefficients)
    mean = coefficients.mean(0)
    centered = coefficients - mean
    covariance = centered.T @ centered / (len(coefficients) - 1)
    covariance = .5 * covariance + .5 * np.diag(np.diag(covariance)) + .01 * np.eye(len(mean))
    precision = np.linalg.solve(covariance, np.eye(len(mean)))
    return dict(mean=mean.tolist(), precision=precision.tolist(), covariance=covariance.tolist(),
                task_coefficients=coefficients.tolist(), source_tasks=list(source_tasks), counts=counts)


METHODS = ('C3', 'RIDGE', 'KERNEL', 'C3_ZERO_MEAN', 'C3_NO_CHEM')
LAMBDAS = (1., 10.)


def kernel(x, other, bandwidth):
    x, other = array(x, 2), array(other, 2)
    require(x.shape[1] == other.shape[1] and np.isfinite(bandwidth) and bandwidth > 0,
            'kernel dimensions/bandwidth')
    distance = np.maximum((x*x).sum(1)[:, None] + (other*other).sum(1)[None, :] - 2*x@other.T, 0.)
    return np.exp(-distance / (2 * bandwidth))


def fit_model(z, q, y, prior, method, penalty):
    z, q, y = array(z, 2), array(q, 2), array(y, 1)
    require(method in METHODS and penalty in LAMBDAS and len(z) == len(q) == len(y),
            'fixed analytic method/configuration')
    ym, ys = float(y.mean()), max(float(y.std()), 1e-6)
    scaled = (y - ym) / ys
    x = z if method == 'C3_NO_CHEM' else np.column_stack((z, q))
    state = dict(method=method, penalty=penalty, label_mean=ym, label_scale=ys,
                 source_width=z.shape[1], chemical_width=q.shape[1])
    if method == 'KERNEL':
        distance = np.maximum((x*x).sum(1)[:, None] + (x*x).sum(1)[None, :] - 2*x@x.T, 0.)
        positive = distance[np.triu_indices(len(x), 1)]
        positive = positive[positive > 1e-12]
        bandwidth = float(np.median(positive)) if len(positive) else 1.
        gram = kernel(x, x, bandwidth)
        # Unpenalized intercept in kernel ridge, including nonzero empirical row means.
        rowmean, grandmean = gram.mean(1), float(gram.mean())
        centered = gram - rowmean[:, None] - rowmean[None, :] + grandmean
        coefficient = np.linalg.solve(centered + penalty*np.eye(len(x)), scaled)
        state.update(train_features=x.tolist(), bandwidth=bandwidth, coefficient=coefficient.tolist(),
                     kernel_train_mean=rowmean.tolist(), kernel_grand_mean=grandmean)
    else:
        width = x.shape[1]
        precision, mean = np.eye(width)*penalty, np.zeros(width)
        if method != 'RIDGE':
            size = z.shape[1]
            p, m = array(prior['precision'], 2), array(prior['mean'], 1)
            require(p.shape == (size, size) and m.shape == (size,), 'source prior dimensions')
            precision[:size, :size] = penalty*p
            if width > size:
                precision[size:, size:] *= 10.
            if method != 'C3_ZERO_MEAN':
                mean[:size] = m
        state['linear'] = fit_linear(x, scaled, precision, mean).state()
    return state


def predict_model(state, z, q):
    z, q = array(z, 2), array(q, 2)
    require(z.shape[1] == state['source_width'] and q.shape[1] == state['chemical_width']
            and len(z) == len(q) and state['method'] in METHODS, 'prediction shape/method')
    x = z if state['method'] == 'C3_NO_CHEM' else np.column_stack((z, q))
    if state['method'] == 'KERNEL':
        k = kernel(x, state['train_features'], state['bandwidth'])
        k = k - k.mean(1)[:, None] - np.array(state['kernel_train_mean'])[None, :] + state['kernel_grand_mean']
        result = k @ np.array(state['coefficient'])
    else:
        result = LinearFit.from_state(state['linear']).predict(x)
    result = result * state['label_scale'] + state['label_mean']
    require(np.isfinite(result).all(), 'finite analytic prediction')
    return result
