"""Pure numerical S4E3 diagnostics. No checkpoint selection or training.

Linear CKA: Kornblith et al., ICML 2019, PMLR 97:3519-3529.
https://proceedings.mlr.press/v97/kornblith19a.html
"""
from __future__ import annotations

import math
import numpy as np

from s4e_mechanism_design import DesignError, require


def finite_array(value):
    x = np.asarray(value)
    require(x.dtype.kind in 'fiu', 'expected real numeric array')
    x = x.astype(np.float64)
    require(np.isfinite(x).all(), 'nonfinite numeric input')
    return x


def paired_ids(left, right):
    require(isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)), 'sample IDs must be sequences')
    require(all(type(x) is str and x for x in list(left) + list(right)), 'sample IDs must be nonempty strings')
    require(len(left) == len(set(left)) and list(left) == list(right), 'sample identities/order differ')


def linear_cka(x, y, ids_x, ids_y):
    paired_ids(ids_x, ids_y)
    x, y = finite_array(x), finite_array(y)
    require(x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0] == len(ids_x)
            and x.shape[0] >= 2 and min(x.shape[1], y.shape[1]) > 0, 'invalid representation shape')
    # Rescale each input before centering to avoid avoidable overflow; CKA is
    # invariant to a nonzero scalar applied to either representation.
    sx, sy = float(np.abs(x).max()), float(np.abs(y).max())
    if sx: x = x / sx
    if sy: y = y / sy
    x, y = x-x.mean(axis=0), y-y.mean(axis=0)
    numerator = float(np.sum((x.T@y)**2))
    denominator = float(np.linalg.norm(x.T@x) * np.linalg.norm(y.T@y))
    if denominator == 0:
        return {'value': None, 'reason': 'zero_centered_norm', 'n': len(ids_x)}
    value = numerator / denominator
    require(math.isfinite(value) and -1e-12 <= value <= 1+1e-12, 'invalid computed CKA')
    return {'value': value, 'reason': None, 'n': len(ids_x)}


def parameter_drift(reference, candidate):
    require(bool(reference) and set(reference) == set(candidate), 'backbone key sets differ')
    numerator, denominator = 0.0, 0.0
    for k in sorted(reference):
        a, b = finite_array(reference[k]), finite_array(candidate[k])
        require(a.shape == b.shape, f'backbone tensor shape differs: {k}')
        numerator = math.hypot(numerator, float(np.linalg.norm(b-a)))
        denominator = math.hypot(denominator, float(np.linalg.norm(a)))
    require(math.isfinite(numerator) and math.isfinite(denominator), 'drift overflow')
    value = numerator / denominator if denominator else None
    require(value is None or math.isfinite(value), 'relative drift overflow')
    return {'difference_l2': numerator, 'reference_l2': denominator,
            'relative_l2': value, 'reason': None if denominator else 'zero_reference_norm',
            'keys': sorted(reference)}


def functional_forgetting(labels, source, hybrid, expected_tasks, expected_ids):
    require(len(set(expected_tasks)) == len(expected_tasks) > 0, 'invalid task list')
    expected = set(expected_tasks)
    require(set(labels) == set(source) == set(hybrid) == set(expected_ids) == expected,
            'functional task sets differ')
    result = []
    for t in expected_tasks:
        sid, sp = source[t]; hid, hp = hybrid[t]
        paired_ids(expected_ids[t], sid); paired_ids(expected_ids[t], hid)
        y, s, h = [finite_array(x) for x in (labels[t], sp, hp)]
        require(y.ndim == s.ndim == h.ndim == 1 and y.shape == s.shape == h.shape
                and len(y) == len(expected_ids[t]), 'functional shape differs')
        if not len(y):
            result.append({'task': t, 'n': 0, 'source_rmse': None, 'hybrid_rmse': None,
                           'delta_rmse': None, 'reason': 'no_observed_labels'})
            continue
        a, b = float(np.linalg.norm(s-y)/math.sqrt(len(y))), float(np.linalg.norm(h-y)/math.sqrt(len(y)))
        require(math.isfinite(a) and math.isfinite(b), 'functional metric overflow')
        result.append({'task': t, 'n': len(y), 'source_rmse': a, 'hybrid_rmse': b,
                       'delta_rmse': b-a, 'reason': None})
    complete = all(x['reason'] is None for x in result)
    return {'tasks': result, 'macro_delta_rmse': sum(x['delta_rmse'] for x in result)/len(result) if complete else None,
            'macro_reason': None if complete else 'incomplete_endpoint_set', 'endpoint_count': len(result)}


def hybrid_state(source, backbone):
    """Exact replacement; never cast, silently skip keys, or touch source heads."""
    prefix = 'encoder.backbone.'
    expected = {k for k in source if k.startswith(prefix)}
    require(bool(expected) and set(backbone) == expected, 'hybrid backbone key sets differ')
    result = dict(source)
    for k in sorted(expected):
        require(source[k].shape == backbone[k].shape and source[k].dtype == backbone[k].dtype,
                f'hybrid tensor shape/dtype differs: {k}')
        result[k] = backbone[k]
    return result
