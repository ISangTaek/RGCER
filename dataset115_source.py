"""Bind Route B to already-reviewed full-fraction Animal56 source assets."""
from pathlib import Path
from types import SimpleNamespace
import json

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
from dataset115_contract import ContractError, digest

SOURCE_LOCK_SHA = 'c44ccba906ca74b3638619ad3db5fdc500be432c8477794b4c30acc681cebda7'


def binding(lock, seed):
    if type(seed) is not int or seed not in range(42, 47):
        raise ContractError('unknown Route B seed')
    runs = [r for r in lock['runs'] if r['seed'] == seed and r['declared_fraction_percent'] == 100]
    if len(runs) != 2 or {r['method'] for r in runs} != {'B1', 'RPT'}:
        raise ContractError('full-fraction paired source missing')
    if any(r[k] != runs[0][k] for r in runs for k in ('teacher_asset_id', 'init_asset_id')):
        raise ContractError('B1/RPT source pairing differs')
    assets = {a['asset_id']: a for a in lock['assets']}
    if len(assets) != len(lock['assets']):
        raise ContractError('duplicate source asset ID')
    teacher = assets[runs[0]['teacher_asset_id']]; init = assets[runs[0]['init_asset_id']]
    if (teacher['seed'] != seed or type(teacher['seed']) is not int
            or teacher['epoch'] != 39 or type(teacher['epoch']) is not int
            or teacher['kind'] != 'teacher' or init['kind'] != 'init'
            or teacher['task_names'] != list(ANIMAL_SOURCE_TASKS)):
        raise ContractError('Animal56 source seed/epoch/task identity differs')
    arch = teacher['architecture_config']
    if (arch['architecture'] != 'Graphormer' or arch['prediction_mode'] != 'quantile'
            or arch['task_names'] != list(ANIMAL_SOURCE_TASKS) or arch['card_enabled'] is not False):
        raise ContractError('source architecture differs')
    return teacher, init


def load_binding(path, seed):
    if digest(path) != SOURCE_LOCK_SHA:
        raise ContractError('unapproved source lock')
    return binding(json.loads(Path(path).read_text(encoding='utf8')), seed)


def model_args(teacher):
    a = teacher['architecture_config']
    return SimpleNamespace(**{k:a[k] for k in ('hidden_dim','a_layers','a_heads','mid_dim','head_hidden_dim','head_dropout','edge_bias_mode')},
                           spatial_pos_clip=a['spatial_pos_max_clip'])


def encoder_from_states(teacher, payload, teacher_state, init_state):
    import torch
    if (type(payload.get('epoch')) is not int or payload['epoch'] != teacher['epoch']
            or type(payload['configuration']['seed']) is not int
            or payload['configuration']['seed'] != teacher['seed']
            or payload['architecture_config']['task_names'] != list(ANIMAL_SOURCE_TASKS)):
        raise ContractError('live source metadata differs')
    for k in ('datastore_fingerprint', 'split_manifest_hash', 'feature_schema_version'):
        if payload['data_config'][k] != teacher['data_config'][k]:
            raise ContractError('live source data identity differs')
    source = {k:v for k,v in teacher_state.items() if k.startswith('encoder.')}
    initial = {k:v for k,v in init_state.items() if k.startswith('encoder.')}
    if not source or set(source) != set(initial):
        raise ContractError('source/init encoder keys differ')
    for k,v in source.items():
        if (not isinstance(v, torch.Tensor) or not isinstance(initial[k], torch.Tensor)
                or v.dtype != initial[k].dtype or v.shape != initial[k].shape
                or not torch.isfinite(v).all() or not torch.equal(v, initial[k])):
            raise ContractError('source/init encoder tensors differ')
    return {k[len('encoder.'):]:v.detach().clone() for k,v in source.items()}


def load_route_b_encoder(repo, lock_path, seed):
    from s4e_mechanism_smoke import state_from_asset
    teacher, init = load_binding(lock_path, seed)
    payload, teacher_state, tr = state_from_asset(Path(repo), teacher)
    _, init_state, ir = state_from_asset(Path(repo), init)
    state = encoder_from_states(teacher, payload, teacher_state, init_state)
    return state, model_args(teacher), dict(seed=seed, teacher=tr, init=ir,
        source_lock_sha256=SOURCE_LOCK_SHA, source_epoch=39, source_tasks=list(ANIMAL_SOURCE_TASKS),
        encoder_equality=True, old_human_heads_reused=False)
