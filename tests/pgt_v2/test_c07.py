"""C07 necessary checks only: prior path, author recipe, effective HPO and handoff."""
import numpy as np
import torch

from pgt_v2.config import candidate, resolve_run
from pgt_v2.models import build_model, task_metadata
from pgt_v2.prior_prompts import PRIOR_METHODS, PriorPrompts
from pgt_v2.training import author_lr, observed_loss, seed_everything, train_model, read_checkpoint
from pgt_v2.hpo import REFERENCE, suggest_config, export_trials
from pgt_v2.artifacts import read_json
from tests.pgt_v2.test_redesign import TASKS, tiny_inputs, tiny_config


def test_prior_generation_and_four_model_paths():
    meta = list(task_metadata(TASKS).values())
    outputs, states = [], []
    # No empirical edges: semantic edges must supply the LD50 relation.
    for method in PRIOR_METHODS:
        seed_everything(42)
        model = build_model(method, TASKS, tiny_config('N1'), adjacency=np.eye(3)).eval()
        inputs = tiny_inputs(method)
        y = model(inputs.batch([0,1,2]))
        y.square().sum().backward()
        assert y.shape == (3,3) and torch.isfinite(y).all()
        assert model.prompt_ids is None and not hasattr(model, 'shared_head')
        assert model.prompt_bank.w1.weight.grad.norm() > 0
        assert model.prompt_bank.w2.weight.grad.norm() > 0
        assert model.layer_gates.grad.norm() > 0
        outputs.append(y.detach()); states.append(model)
    torch.testing.assert_close(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])
    assert states[2].prompt_bank.relation_weights[0,1] == .125
    perm = states[3].prompt_bank.permutation
    torch.testing.assert_close(states[3].prompt_bank.relation_weights,
        states[2].prompt_bank.relation_weights[perm][:,perm])
    for a,b in zip(states[2].parameters(),states[3].parameters()):
        torch.testing.assert_close(a,b)
    unknown = [{**m, 'organism':'mammal (species unspecified)', 'route':'unreported'} for m in meta]
    prior = PriorPrompts('N1P-Hybrid', unknown, np.eye(3), 16)
    assert prior.relation_weights.count_nonzero() == 0


def test_author_recipe_resume_and_raw_scale(tmp_path):
    assert [author_lr(e) for e in (0,19,20,39,40,50,60,70,80,100,119)] == [
        .001,.001,.0006,.0006,.00012,.000024,.000012,.000001,.0000001,.00000001,.00000001]
    p=torch.tensor([[1.,9.],[3.,8.]])
    y=torch.tensor([[0.,float('nan')],[1.,4.]])
    assert observed_loss(p,y).item() == 7
    cfg={**candidate('TOX_AUTHOR','AUTHOR'), 'max_epochs':2, 'min_epochs':2,
         'selection_tasks':[TASKS[-1]], 'auxiliary_selection_tasks':list(TASKS),
         'auxiliary_checkpoint':'best_all59.pt', 'auxiliary_role':'ALL59'}
    labels=np.array([[1.,2.,3.],[2.,np.nan,4.],[3.,4.,5.]])
    stats=dict(mean=np.array([2.,3.,4.]),scale=np.array([4.,5.,6.]),count=np.isfinite(labels).sum(0),adjacency=np.eye(3))
    identity=dict(task_names=list(TASKS), method_id='TOX_AUTHOR',config=cfg)
    def fit(path, stop=None, resume=False):
        seed_everything(42)
        model=build_model('TOX_AUTHOR',TASKS,cfg,adjacency=stats['adjacency'])
        train_model(model,tiny_inputs('TOX_AUTHOR'),labels,tiny_inputs('TOX_AUTHOR'),labels,stats,
            config=cfg,identity=identity,output=path,method='TOX_AUTHOR',resume=resume,stop_after_epoch=stop)
        return read_checkpoint(path/'last.pt')
    full=fit(tmp_path/'full')
    fit(tmp_path/'resume',stop=1)
    resumed=fit(tmp_path/'resume',resume=True)
    for k in full['model']:
        torch.testing.assert_close(full['model'][k],resumed['model'][k],atol=0,rtol=0)
    assert np.all(full['stats']['mean']==0) and np.all(full['stats']['scale']==1)
    assert full['optimizer']['param_groups'][0]['nesterov']
    assert (tmp_path/'full/best_all59.pt').exists()


def test_optuna_effective_config_and_persistent_selection(tmp_path):
    import optuna
    from pgt_v2.hpo import OPTUNA_VERSION
    assert optuna.__version__ == OPTUNA_VERSION
    study=optuna.create_study(direction='minimize', storage='sqlite:///'+(tmp_path/'study.db').as_posix(),
        study_name='synthetic_configuration_only',sampler=optuna.samplers.TPESampler(seed=20261028),
        pruner=optuna.pruners.NopPruner())
    study.enqueue_trial(REFERENCE)
    trial=study.ask(); cfg=suggest_config(trial,'N1',TASKS)
    assert cfg['ffn_dim']==192 and cfg['layers']==4 and 'prompt_rank' not in cfg
    study.tell(trial,.9)
    changed={**REFERENCE,'layers':2,'hidden_dim':128,'heads':8,'ffn_ratio':4}
    cfg2=suggest_config(optuna.trial.FixedTrial(changed),'N1S',TASKS)
    m=build_model('N1S',TASKS,cfg2)
    assert m.backbone.hidden_dim==128 and len(m.layer_gates)==2
    assert m.prompt_ids.shape==(1,128) and cfg2['ffn_dim']==512
    export_trials(study,tmp_path)
    assert read_json(tmp_path/'top2_validation.json')[0]['value']==.9
    loaded=optuna.load_study(study_name=study.study_name, storage='sqlite:///'+(tmp_path/'study.db').as_posix())
    assert len(loaded.trials)==1


def test_hpo_budget_and_interrupted_trial_reuse(tmp_path, monkeypatch):
    import pytest
    import pgt_v2.hpo as hpo
    import pgt_v2.artifacts as artifacts
    from architecture.toxacute_tasks import TOXACUTE_TASKS
    from molecular_features import PGT_FEATURE_SPEC
    names=list(TOXACUTE_TASKS)
    manifest=dict(fold='S0', task_names=names, manifest_sha256='synthetic', view_id='synthetic',
        source_sha256='synthetic', cache_id='synthetic', feature_spec_sha256=PGT_FEATURE_SPEC.sha256,
        files={'train.npz':'synthetic', 'validation.npz':'synthetic'})
    monkeypatch.setattr(hpo,'load_manifest',lambda path:manifest)
    monkeypatch.setattr(artifacts,'load_arrays',lambda root,name,meta:
        {'count':np.array([8,14]+[11]*18+[8]+[20]*38)} if name=='statistics.npz' else pytest.fail('unexpected label read'))
    artifacts.write_json(tmp_path/'groups.json',{'S21':names[:21]})
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append(read_json(args[0])['config'])
        raise KeyboardInterrupt()
    monkeypatch.setattr(hpo,'run_training',interrupted)
    args=('N1',tmp_path/'view',tmp_path/'groups.json',tmp_path/'cache',tmp_path/'search')
    with pytest.raises(KeyboardInterrupt):
        hpo.run_search(*args,device='cpu',trials=12)
    def simulated(run_file,*args,**kwargs):
        run=read_json(run_file)
        calls.append(run['config'])
        assert run['init_seed']==42 and run['config']['selection_tasks']==names[:21]
        return dict(terminal='COMPLETE', selection_score=float(run['config']['layers']),
            validation={'synthetic_only':True}, best_epoch=1, parameters={'total':1})
    monkeypatch.setattr(hpo,'run_training',simulated)
    result=hpo.run_search(*args,device='cpu',resume=True,trials=12)
    assert result['attempted']==12 and result['test_evaluations']==0
    assert len(calls)==13 and calls[0]==calls[1]
    assert len(read_json(tmp_path/'search/trials.json'))==12
