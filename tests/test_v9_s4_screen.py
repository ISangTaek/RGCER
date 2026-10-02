from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

import v9_s4_screen as m
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime,one_cpu_thread,setup_factory


def spec():return dict(m.SPEC,max_epochs=3,warmup_epochs=1,grouping=dict(m.group.SPEC,probe_groups_per_task=4,bootstrap_replicates=16))


@pytest.mark.parametrize('job',m.jobs(),ids=lambda j:j['id'])
def test_nine_real_graph_paths_selection_and_content_verifier(job,tmp_path,joint):
    f,t,a,ref=joint(job['setting']);r=m.train_one(f,t,a,job,tmp_path,'a'*40,'cpu',ref,spec())
    f,t,a,ref=joint(job['setting']);assert r==m.verify_job(f,t,a,job,tmp_path,'a'*40,ref,spec())
    assert len(r['history'])==3 and all(v['updates']>0 for v in r['coverage'].values())


@pytest.mark.parametrize('setting',m.SMOKE)
def test_real_smoke_exercises_both_routes_and_gradient_probe(setting,tmp_path,joint):
    f,t,a,_=joint(setting);m.smoke(f,t,a,setting,tmp_path,'a'*40,'cpu')
    r=m.read(tmp_path/'receipt.json');assert r['updates']==r['gradient_probe_calls']==2


def test_probe_uses_train_only_and_leaves_weights_grad_flags_rng_unchanged(joint):
    f,t,a,_=joint('A');training=m.s3.s2.observations(f,t,'A','train')
    t.model=m.group.GroupedGraph(t.model,a,'SGTA');m.group.begin_epoch(t.model,0,1)
    before=m.s3.state_dict_sha256(t.model);flags={n:p.requires_grad for n,p in t.model.named_parameters()};rng=torch.get_rng_state().clone()
    selected,grad=m.probe(f,t,a,'A',training,spec())
    assert before==m.s3.state_dict_sha256(t.model) and flags=={n:p.requires_grad for n,p in t.model.named_parameters()}
    assert torch.equal(rng,torch.get_rng_state()) and set(grad)==set(t.model.task_name)
    assert all(r['split']=='train' for v in selected.values() for r in v)
    assert all(p.grad is None for p in t.model.parameters())


@pytest.mark.parametrize('damage',['selection','groups','frozen_encoder','adam','validation_label','gradient_dimension','forged_gradients','source_identity'])
def test_content_verifier_rejects_self_consistent_wrong_artifacts(damage,tmp_path,joint):
    j=m.jobs()[2];f,t,a,ref=joint(j['setting']);r=m.train_one(f,t,a,j,tmp_path,'a'*40,'cpu',ref,spec())
    top=m.read(tmp_path/'topology.json')
    if damage in ('selection','groups'):
        if damage=='selection':next(iter(top['selection'].values()))[0]['split']='validation'
        else:top['groups'][0].pop()
        (tmp_path/'topology.json').write_text(__import__('json').dumps(top),encoding='utf8');r['topology_sha256']=m.sha(tmp_path/'topology.json')
    elif damage in ('frozen_encoder','adam'):
        p=tmp_path/'best.pt';payload=torch.load(p,weights_only=True)
        if damage=='frozen_encoder':payload['model_state'][next(k for k in payload['model_state'] if k.startswith('encoder.'))].add_(1)
        else:next(iter(payload['optimizer_state']['state'].values()))['step'].add_(1)
        torch.save(payload,p);r['checkpoint_sha256']=m.sha(p)
    elif damage=='validation_label':
        p=tmp_path/'validation_epoch_001.json';rows=m.read(p);rows[0]['label']+=1;p.write_text(__import__('json').dumps(rows),encoding='utf8')
    elif damage in ('gradient_dimension','forged_gradients'):
        p=tmp_path/'gradients.pt';v=torch.load(p,weights_only=True)
        if damage=='gradient_dimension':v={k:x[:,:1] for k,x in v.items()};top['gradient_dimension']=1
        else:
            v={k:x+.1 for k,x in v.items()}
            top['relations']=m.group.similarities(v,top['selection'],list(v),spec()['grouping'])
            top['groups'],top['decision']=m.group.choose_groups(j['method'],top['relations'])
        torch.save(v,p);top['gradients_sha256']=m.sha(p)
        (tmp_path/'topology.json').write_text(__import__('json').dumps(top),encoding='utf8');r['topology_sha256']=m.sha(tmp_path/'topology.json')
    else:r['identity']['source']['scalers'][a.tasks[0]]['mean']+=1
    (tmp_path/'receipt.json').write_text(__import__('json').dumps(r),encoding='utf8')
    (tmp_path/'identity.json').write_text(__import__('json').dumps(r['identity']),encoding='utf8')
    f,t,a,ref=joint(j['setting'])
    with pytest.raises(ValueError):m.verify_job(f,t,a,j,tmp_path,'a'*40,ref,spec())


def test_warmup_trajectory_identical_across_three_methods(tmp_path,joint):
    rr=[]
    for method in m.METHODS:
        f,t,a,ref=joint('A');j=next(j for j in m.jobs() if j['setting']=='A' and j['method']==method)
        out=tmp_path/method;out.mkdir();rr.append(m.train_one(f,t,a,j,out,'a'*40,'cpu',ref,spec()))
    assert rr[0]['history'][0]==rr[1]['history'][0]==rr[2]['history'][0]
    assert rr[0]['parameters']==rr[1]['parameters']==rr[2]['parameters']


def test_dispatch_failure_preserves_claim_stops_before_formal_runs(tmp_path,monkeypatch):
    import p1d4_batch
    monkeypatch.setattr(m,'REPO',tmp_path);monkeypatch.setattr(m.s3.s2.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(m,'gate_check',lambda *a:None);monkeypatch.setattr(m.s3,'references',lambda *a:{})
    monkeypatch.setattr(m.s3,'copy_references',lambda *a:None);monkeypatch.setattr(p1d4_batch,'free_gpus',lambda a:a)
    monkeypatch.setattr(m.s3.s2.c0,'gpu_uuid',lambda *a:'GPU-fixture');calls=[]
    def fail(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=19)
    monkeypatch.setattr(m.subprocess,'run',fail)
    for role in ('wsl','server'):(tmp_path/role).mkdir()
    a=SimpleNamespace(commit='a'*40,gpu=0,output=tmp_path/'output',wsl_evidence=tmp_path/'wsl',server_evidence=tmp_path/'server',
                      reference_root=tmp_path/'reference',split_manifest=tmp_path/'split',source_lock=tmp_path/'source')
    with pytest.raises(ValueError):m.run(a)
    assert len(calls)==1 and (tmp_path/m.REGISTRY/'attempt.json').exists() and (a.output/'failed.json').exists()
    with pytest.raises(ValueError):m.run(a)
    assert len(calls)==1


def test_no_scene_collage_can_create_candidate_signal(monkeypatch):
    monkeypatch.setitem(m.SPEC,'warmup_epochs',1)
    old={j['id']:dict(selected=dict(macro_rmse=1.)) for j in m.s3.s2.jobs()}
    results=[dict(job=j,history=[dict(shared=True)],best_epoch=2,parameters=dict(adapters=8),selected=dict(macro_rmse=.9 if
                  j['method']==('SGTA' if j['setting']=='B' else 'TG_LORA_GRAPH') else 1.1)) for j in m.jobs()]
    assert m.compare(results,old)['all_scene_candidate_signals']==[]
    for r in results:r['selected']['macro_rmse']=.8 if r['job']['method']=='SGTA' else 1.1
    answer=m.compare(results,old);assert answer['all_scene_candidate_signals']==['SGTA'] and not answer['unified_superiority_confirmed']
