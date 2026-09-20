"""Committed five-seed identities; no filesystem search or runtime inference."""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import json

from p1d_optimization import require

LOCK=Path(__file__).resolve().parent/'configs/p1d4_identity_lock.json'


def contract_for(setting,seed):
    require(type(setting) is str and setting in ('ToxAcute','A','B'), 'setting scope')
    require(type(seed) is int and seed in range(42,47), 'seed scope')
    lock=json.loads(LOCK.read_bytes())
    require(lock['schema']=='p1d4_identity_lock_v1' and lock['execution_authorized'] is False,
            'identity lock is not execution authorization')
    require(set(lock['settings'])=={'ToxAcute','A','B'} and all(set(v)=={str(i) for i in range(42,47)}
            for v in lock['settings'].values()), 'complete fifteen identities')
    value=deepcopy(lock['settings'][setting][str(seed)])
    if setting=='ToxAcute':
        require(type(value['args']['seed']) is int and value['args']['seed']==seed, 'Tox locked seed')
        path=PurePosixPath(value['init_path'])
        require(not path.is_absolute() and '..' not in path.parts
                and path.name==f'b1_init_seed{seed}.pt' and value['args']['init_state_path']==str(path),
                'Tox locked init path/seed')
        for field in ('init_file_sha256','initial_full_model_digest'):
            require(type(value[field]) is str and len(value[field])==64
                    and set(value[field])<=set('0123456789abcdef'), 'Tox init digest')
    else:
        require(type(value['source_identity']['seed']) is int and value['source_identity']['seed']==seed,
                'same-seed source')
    return value
