from copy import deepcopy

import pytest
import torch

from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory
import v10_feature_cache as c
import v10_s1_screen as s


def named_source(source):
    mapping=dict(zip(source.tasks,('rat_oral_LD50','mouse_oral_LD50')))
    return c.s3.aux.Source({mapping[t]:d for t,d in source.datasets.items()},
        [dict(r,task=mapping[r['task']]) for r in source.rows],
        torch.nn.ModuleDict({mapping[t]:deepcopy(h) for t,h in source.heads.items()}),
        {mapping[t]:v for t,v in source.scalers.items()},dict(synthetic=True))


@pytest.mark.parametrize('setting',s.SETTINGS)
def test_graphormer_cache_identity_and_independent_raw_replay(setting,joint,tmp_path):
    f,t,source,_=joint(setting); source=named_source(source)
    before=c.s3.state_dict_sha256(t.model.encoder)
    raw=c.extract(f,t,source,setting,'cpu')
    assert all(r['split'] in ('train','validation') for r in raw['rows'])
    value=c.save_cache(raw,tmp_path/'cache','a'*40)
    assert c.load_cache(tmp_path/'cache','a'*40)['identity']==raw['identity']
    fresh,audit=s.audit_cache(f,t,source,setting,value)
    assert audit['target_rows']==sum(r['task'] in value['targets'] for r in value['rows'])
    assert c.s3.state_dict_sha256(t.model.encoder)==before
    for key in ('h','functions'): torch.testing.assert_close(fresh[key],value[key],atol=1e-6,rtol=1e-6)
    # Trusted provenance is required even for a self-consistent cache manifest.
    with pytest.raises(ValueError,match='trusted cache identity'):
        c.load_cache(tmp_path/'cache','a'*40,identity={'wrong':True})
    changed=deepcopy(value); changed['functions'][0,0]+=1
    with pytest.raises(ValueError,match='raw graph feature'):
        s.audit_cache(f,t,source,setting,changed)


def test_train_only_feature_statistics_ignore_validation_features(joint):
    f,t,source,_=joint('A'); raw=c.extract(f,t,named_source(source),'A','cpu')
    changed=deepcopy(raw)
    for i,r in enumerate(raw['rows']):
        if r['split']=='validation': changed['h'][i]+=1e6
    a,b=c.pack(raw,'a'*40),c.pack(changed,'a'*40)
    assert torch.equal(a['mean'],b['mean']) and torch.equal(a['scale'],b['scale'])
