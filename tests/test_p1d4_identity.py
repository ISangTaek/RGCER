import json
from pathlib import Path
import pytest

from p1d4_identity import contract_for,LOCK
from p1d_optimization import OptimizationError


def test_all_fifteen_bound_identities():
    for seed in range(42,47):
        tox=contract_for('ToxAcute',seed)
        assert tox['args']['seed']==seed
        assert contract_for('B',seed)['source_identity']['init_sha256']==tox['init_file_sha256']
        assert contract_for('A',seed)['source_identity']['seed']==seed
    assert len({contract_for('ToxAcute',s)['initial_full_model_digest'] for s in range(42,47)})==5
    # These are real historical paths, not a seed42 path substitution.
    for seed in (43,45):assert '/d8_stage0/' in contract_for('ToxAcute',seed)['init_path']
    for seed in (42,44,46):assert '/d7_stage_b/' in contract_for('ToxAcute',seed)['init_path']


def test_seed42_unchanged_from_accepted_smoke():
    old=json.loads((LOCK.parent/'p1d3_smoke_lock.json').read_bytes())
    assert contract_for('ToxAcute',42)==old['tox']


def test_each_return_is_isolated():
    c=contract_for('ToxAcute',42);c['args']['seed']=43
    assert contract_for('ToxAcute',42)['args']['seed']==42


@pytest.mark.parametrize('setting,seed',[('A',True),('A','42'),('A',47),('test',42),('ToxAcute',41)])
def test_invalid_identity_keys(setting,seed):
    with pytest.raises(OptimizationError):contract_for(setting,seed)


def test_wrong_seed_in_relabelled_lock_rejected(tmp_path,monkeypatch):
    import p1d4_identity
    value=json.loads(LOCK.read_bytes())
    value['settings']['ToxAcute']['43']=value['settings']['ToxAcute']['42']
    path=tmp_path/'bad.json';path.write_text(json.dumps(value),encoding='utf8')
    monkeypatch.setattr(p1d4_identity,'LOCK',path)
    with pytest.raises(OptimizationError,match='locked seed'):contract_for('ToxAcute',43)
