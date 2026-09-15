from copy import deepcopy
import json
import pytest
import torch

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
from dataset115_contract import ContractError, PRIMARY
from dataset115_source import binding, load_binding, encoder_from_states
from dataset115_smoke import one_step,verify_step_records
from dataset115_model import build_human5_model
from tests.test_dataset115_adapter import args, view


def lock():
    return dict(runs=[dict(seed=42,declared_fraction_percent=100,method=m,
            teacher_asset_id='t',init_asset_id='i') for m in ('B1','RPT')],
        assets=[dict(asset_id='t',kind='teacher',seed=42,epoch=39,task_names=list(ANIMAL_SOURCE_TASKS),
            architecture_config=dict(architecture='Graphormer',prediction_mode='quantile',
                task_names=list(ANIMAL_SOURCE_TASKS),card_enabled=False)),dict(asset_id='i',kind='init')])


def test_full_fraction_pair():
    teacher,init=binding(lock(),42)
    assert teacher['epoch']==39 and init['kind']=='init'


@pytest.mark.parametrize('case',['missing','fraction','pair','seed','epoch','human','duplicate'])
def test_wrong_binding_rejected(case):
    d=lock()
    if case=='missing':d['runs'].pop()
    if case=='fraction':d['runs'][0]['declared_fraction_percent']=50
    if case=='pair':d['runs'][0]['init_asset_id']='other'
    if case=='seed':d['assets'][0]['seed']=43
    if case=='epoch':d['assets'][0]['epoch']=12
    if case=='human':d['assets'][0]['task_names'][0]='human_oral_TDLo'
    if case=='duplicate':d['assets'].append(deepcopy(d['assets'][0]))
    with pytest.raises(ContractError):binding(d,42)


def test_unapproved_lock_not_loaded(tmp_path):
    p=tmp_path/'lock.json';p.write_text(json.dumps(lock()))
    with pytest.raises(ContractError,match='unapproved'):load_binding(p,42)


def states():
    t=lock()['assets'][0];t['data_config']={k:'frozen' for k in ['datastore_fingerprint','split_manifest_hash','feature_schema_version']}
    p=dict(epoch=39,configuration=dict(seed=42),architecture_config=dict(task_names=list(ANIMAL_SOURCE_TASKS)),data_config=deepcopy(t['data_config']))
    s={'encoder.backbone.weight':torch.ones(2), 'decoders.animal.weight':torch.zeros(1)}
    i={'encoder.backbone.weight':torch.ones(2), 'decoders.human.weight':torch.ones(1)}
    return t,p,s,i


def test_encoder_only_no_old_heads():
    t,p,s,i=states();out=encoder_from_states(t,p,s,i)
    assert set(out)=={'backbone.weight'}
    out['backbone.weight'][0]=0
    assert s['encoder.backbone.weight'][0]==1


@pytest.mark.parametrize('case',['tensor','keys','nonfinite','dtype','live_seed','data'])
def test_live_source_mismatch(case):
    t,p,s,i=states()
    if case=='tensor':i['encoder.backbone.weight'][0]=2
    if case=='keys':i['encoder.extra']=torch.ones(2)
    if case=='nonfinite':s['encoder.backbone.weight'][0]=float('nan')
    if case=='dtype':i['encoder.backbone.weight']=i['encoder.backbone.weight'].double()
    if case=='live_seed':p['configuration']['seed']=43
    if case=='data':p['data_config']['datastore_fingerprint']='wrong'
    with pytest.raises(ContractError):encoder_from_states(t,p,s,i)


def batches_and_scalers():
    from dataset import DataCollator
    from dataset115_adapter import GraphTaskView,TrainOnlyScaler
    v=view(); batches={}
    for t in PRIMARY:
        ds=GraphTaskView(v,t);batches[t]=DataCollator()([ds[0],ds[1]])
    return batches,TrainOnlyScaler.fit(v).trainer_scalers()


def test_one_step_updates_and_reload(tmp_path):
    a=args();source=build_human5_model(a,method='B0',seed=123).encoder.state_dict()
    before={k:v.clone() for k,v in source.items()}
    batches,scalers=batches_and_scalers()
    rows=one_step(a,source,batches,scalers,tmp_path/'run',device='cpu')
    assert [r['method'] for r in rows]==['B0','B1','RPT']
    assert len({r['head_before'] for r in rows})==1
    assert rows[2]['encoder_before']==rows[2]['encoder_after']
    assert all(r['reload_max_error']==0 for r in rows)
    assert all(torch.equal(before[k],source[k]) for k in source)
    assert all((tmp_path/'run'/r['checkpoint']).is_file() for r in rows)
    assert verify_step_records(tmp_path/'run',rows)['saved_checkpoint_tensor_checks']==3
    changed=deepcopy(rows);changed[1]['encoder_after']='0'*64
    with pytest.raises(ContractError,match='digest'):
        verify_step_records(tmp_path/'run',changed)
    (tmp_path/'run'/rows[0]['checkpoint']).write_bytes(b'corrupted')
    with pytest.raises(ContractError,match='checkpoint identity'):
        verify_step_records(tmp_path/'run',rows)


def test_no_overwrite_or_extra_seed(tmp_path):
    batches,scalers=batches_and_scalers()
    with pytest.raises(ContractError,match='exists'):
        one_step(args(),{},batches,scalers,tmp_path,device='cpu')
    with pytest.raises(ContractError,match='scope'):
        one_step(args(),{},batches,scalers,tmp_path/'new',device='cpu',seed=43)


def test_other_batch_size_rejected(tmp_path):
    batches,scalers=batches_and_scalers();batches[PRIMARY[0]].y=torch.ones(3,1)
    with pytest.raises(ContractError,match='two training'):
        one_step(args(),{},batches,scalers,tmp_path/'new',device='cpu')
