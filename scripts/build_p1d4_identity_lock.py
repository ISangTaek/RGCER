"""Codex-side deterministic derivation from accepted inventories; stdout only.

Not a GLM task or a means of authorizing training. The generated lock must be
reviewed and committed; the production runner consumes the committed lock.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(repo):
    folder=repo/'Plans&Results/Results/2026-09-20_P1D优化诊断准备'
    assets_path=folder/'optimization_assets.json'
    pairing_path=folder/'initialization_pairing.json'
    smoke_path=repo/'configs/p1d3_smoke_lock.json'
    assets=json.loads(assets_path.read_bytes())
    pairing=json.loads(pairing_path.read_bytes())
    smoke=json.loads(smoke_path.read_bytes())
    pairs={r['seed']:r for r in pairing['pairs']}
    assert set(pairs)==set(range(42,47))
    selected=[r for r in assets if r['method']=='B1']
    assert len(selected)==15
    assert {(r['setting'],r['seed']) for r in selected}=={(s,i) for s in ('ToxAcute','A','B') for i in range(42,47)}
    result=dict(schema='p1d4_identity_lock_v1',execution_authorized=False,
                provenance=[dict(path=str(p.relative_to(repo)).replace('\\','/'),sha256=digest(p))
                            for p in (assets_path,pairing_path,smoke_path)],
                inputs=smoke['inputs'], route_a_sources={}, route_a_source_provenance=[], settings={s:{} for s in ('ToxAcute','A','B')})
    fields=('input_identity','source_identity','train_ids','validation_ids','train_observations',
            'validation_observations','task_names','scaler','architecture')
    for r in selected:
        setting,seed=r['setting'],r['seed']
        conf=r['configuration']
        assert conf['seed']==seed
        if setting=='ToxAcute':
            base=deepcopy(smoke['tox'])
            args={k:deepcopy(conf[k]) for k in base['args']}
            # Historical JSON spells a few fixed integral constants as floats.
            # Preserve smoke-vetted types only after value equality is checked.
            for k,v in base['args'].items():
                if type(v) is int and type(args[k]) is float:
                    assert args[k]==v
                    args[k]=v
            for k in ('epochs','bs','lr','weight_decay','grad_clip','task_sampling','split_seed',
                      'weighting','arch','a_layers','a_heads','hidden_dim','mid_dim','prediction_mode'):
                assert args[k]==base['args'][k]
            pair=pairs[seed]
            expected_path=pair['init']['path'].split('/RGCER/',1)[1]
            assert Path(expected_path).name==f'b1_init_seed{seed}.pt'
            assert expected_path.startswith('artifacts/inits/') and '..' not in Path(expected_path).parts
            assert args['init_state_path']==r['source_init_path']==expected_path
            assert pair['init']['path'].endswith('/'+expected_path)
            assert pair['initial_full_model_digest']==r['initial_full_model_digest']
            assert conf['init_overlay_provenance']['init_state_sha256']==pair['init']['sha256']
            for k,v in base['data_identity'].items():
                assert conf['datastore_metadata'][k]==v
            assert r['training_observations']==base['counts']['train']
            base.update(args=args,init_path=expected_path,init_file_sha256=pair['init']['sha256'],
                        initial_full_model_digest=pair['initial_full_model_digest'])
            if seed==42:assert base==smoke['tox']
            result['settings'][setting][str(seed)]=base
        else:
            assert conf['source_identity']['seed']==seed
            expected={k:deepcopy(conf[k]) for k in fields}
            expected.update(initial_encoder=r['initial_encoder'],initial_heads=r['initial_heads'])
            if setting=='B':assert conf['source_identity']['init_sha256']==pairs[seed]['init']['sha256']
            result['settings'][setting][str(seed)]=expected
            if setting=='A':
                provenance_path=repo/Path(r['evidence_root'])/'run_provenance.json'
                provenance=json.loads(provenance_path.read_bytes())
                path=provenance['source_output'].split('/RGCER/',1)[1]
                assert Path(path).name==f'source_seed{seed}' and '..' not in Path(path).parts
                assert provenance['source_receipt']['seed']==seed
                assert provenance['source_receipt']['source_sha256']==conf['source_identity']['teacher_sha256']
                result['route_a_sources'][str(seed)]=path
                result['route_a_source_provenance'].append(dict(path=str(provenance_path.relative_to(repo)).replace('\\','/'),sha256=digest(provenance_path)))
    return result


if __name__=='__main__':
    print(json.dumps(build(Path(__file__).resolve().parents[1]),ensure_ascii=False,indent=2,allow_nan=False))
