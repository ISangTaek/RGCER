import json
import pytest
from p2_contract import P2Error, strict_json, bound_json, run_matrix
from p2_contract import validate_members,FINGERPRINT,TASKS

def synthetic_members():
    panels={}
    for p in ('mouse','rat'):
        panels[p]=[dict(canonical=f'{p}_{i}',group=f'g_{p}_{i}',observations={
            route:dict(value=1.,global_index=i,sample_id=f'{p}_{i}')
            for route in ('oral','intraperitoneal')}) for i in range(700)]
    target={split:{t:[dict(sample_id=f'{split}_{i}') for i in range(n)]
                   for t,n in zip(TASKS,counts)}
            for split,counts in [('train',(96,85,81)),('validation',(14,12,13))]}
    return dict(schema='p2a_members_v1',data_fingerprint=FINGERPRINT,panels=panels,target=target)

def test_members_valid():validate_members(synthetic_members())

@pytest.mark.parametrize('mutation',['fingerprint','count','same_group','cross_group','missing_route','bool_label','nan_label','bool_index','test_split','duplicate_target'])
def test_members_fail_closed(mutation):
    m=synthetic_members();r=m['panels']['mouse'][0];o=r['observations']['oral']
    if mutation=='fingerprint':m['data_fingerprint']='0'*64
    if mutation=='count':m['panels']['mouse'].pop()
    if mutation=='same_group':r['group']=m['panels']['mouse'][1]['group']
    if mutation=='cross_group':r['group']=m['panels']['rat'][0]['group']
    if mutation=='missing_route':del r['observations']['oral']
    if mutation=='bool_label':o['value']=True
    if mutation=='nan_label':o['value']=float('nan')
    if mutation=='bool_index':o['global_index']=False
    if mutation=='test_split':m['target']['test']={}
    if mutation=='duplicate_target':m['target']['train'][TASKS[0]][0]=m['target']['train'][TASKS[0]][1]
    with pytest.raises(P2Error):validate_members(m)

@pytest.mark.parametrize('raw',['{"a":1,"a":2}','{"v":NaN}','{"v":Infinity}','{"v":-Infinity}'])
def test_strict_json_rejects(raw):
    with pytest.raises(P2Error):strict_json(raw)

def test_matrix():
    runs=run_matrix()
    assert len(runs)==len({r['run_id'] for r in runs})==80
    assert sum(r['epochs'] for r in runs)==3200
    assert sum(r['updates'] for r in runs)==23200
    sources=[r for r in runs if r['role']=='source'];assert len(sources)==20
    for src in sources:
        children=[r for r in runs if r.get('source_run_id')==src['run_id']]
        assert len(children)==3
        assert all((r['seed'],r['panel'],r['route'])==(src['seed'],src['panel'],src['route']) for r in children)

def test_file_binding(tmp_path):
    import hashlib
    p=tmp_path/'x.json';p.write_text('{"value":1}')
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert bound_json(p,h)=={'value':1}
    p.write_text('{"value":2}')
    with pytest.raises(P2Error):bound_json(p,h)
