from copy import deepcopy
import pytest
import torch
import v11_features as f
from tests.test_v10_s1_screen import value
from tests.test_v10_context_models import single_thread


@pytest.fixture
def chemical_value(value):
    v=deepcopy(value)
    for row in v['rows']:
        i=int(row['sample_id'].replace('train','').replace('validation',''))
        row['canonical']='C'*(i+1+(12 if row['split']=='validation' else 0))
    v=f.graph.pack({k:x for k,x in v.items() if k not in ('mean','scale','schema','commit','statistics_population_sha256')},v['commit'])
    return v


@pytest.fixture
def data(chemical_value): return f.Data(chemical_value,f.build(chemical_value))


def test_rdkit_recipe_train_statistics_and_raw_replay(chemical_value,tmp_path):
    chem=f.save(chemical_value,tmp_path/'chem','b'*64,'c'*40)
    got=f.load(chemical_value,tmp_path/'chem','b'*64,'c'*40)
    f.check(chemical_value,got,raw_replay=True,robust=True)
    assert chem['fp_packed'].shape==(15,256)
    assert chem['descriptors'].shape==(15,24)
    assert not chem['missing'].any()
    d=f.Data(chemical_value,got)
    e=d.episode(d.targets[0],d.by[d.targets[0],'validation'])
    assert e.context_fp.shape[1]==2048 and e.query_descriptors.shape==(3,48)


def test_validation_values_do_not_fit_statistics(chemical_value):
    a=f.build(chemical_value); other=deepcopy(chemical_value)
    for row in other['rows']:
        if row['split']=='validation': row['label']+=1e6; row['canonical']='CC(O)C(=O)O'
    b=f.build(other)
    for key in a['statistics']: assert torch.equal(a['statistics'][key],b['statistics'][key])
    for key in ('robust_center','robust_factors'): assert torch.equal(a[key],b[key])


@pytest.mark.parametrize('kind',('fingerprint','descriptors','mapping','row','robust','stats'))
def test_forgery_rejected_even_with_new_file_hashes(chemical_value,kind):
    c=f.build(chemical_value)
    if kind=='fingerprint': c['fp_packed'][0,0]^=1
    elif kind=='descriptors': c['descriptors'][0,0]+=1
    elif kind=='mapping': c['row_indices'][0]=1
    elif kind=='row': c['rows_sha256']='0'*64
    elif kind=='robust': c['robust_factors'][0]+=1
    else: c['statistics']['mean'][0]+=1
    with pytest.raises(ValueError): f.check(chemical_value,c,raw_replay=True,robust=True)


def test_missing_descriptor_imputation_and_constants_are_finite():
    x=torch.zeros(4,24,dtype=torch.float64); m=torch.zeros_like(x,dtype=torch.bool)
    m[:,0]=True; x[0,1]=10; m[0,1]=True
    s=f.descriptor_statistics(x,m,torch.tensor([0,1,2]))
    assert s['all_missing'][0] and s['mean'][1]==0 and (s['scale']>0).all()


def test_invalid_smiles_and_rdkit_version_fail_closed(monkeypatch):
    with pytest.raises(ValueError): f.chemistry(['invalid smiles'])
    monkeypatch.setattr(f.rdBase,'rdkitVersion','0')
    with pytest.raises(ValueError,match='version'): f.chemistry(['CC'])


def test_source_own_head_mask_agrees_between_robust_fit_and_model(data):
    import v11_models as m
    model=m.build('TCF',data,tiny=True); t=data.sources[0]; ids=data.by[t,'train'][:2]
    e=data.episode(t,ids)
    assert torch.equal(model.flat(e),data.flat_rows(ids))
    assert (model.views(e)[0][:,data.h.shape[1]]==0).all()
