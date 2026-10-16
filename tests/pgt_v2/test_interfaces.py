import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from pgt_v2.artifacts import load_manifest, load_arrays, read_json, write_json, package_files
from pgt_v2.cache import file_sha256
from pgt_v2.chemistry import digest
from pgt_v2.config import candidate, git_identity, resolve_run
from pgt_v2.evaluation import select_config, sealed_predict, score_prediction
from pgt_v2.training import run_training

spec=importlib.util.spec_from_file_location('pgt_test_support',Path(__file__).with_name('support.py'))
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)


@pytest.fixture
def bundle(tmp_path):
    from architecture.toxacute_tasks import TOXACUTE_TASKS
    support.synthetic_bundle(tmp_path/'data',TOXACUTE_TASKS[:2])
    return tmp_path/'data'


def run_spec(bundle,path,config_id='C0',method='M6'):
    manifest=load_manifest(bundle/'view')
    run=resolve_run(dict(run_id='synthetic_'+method+'_'+config_id,phase='SYNTHETIC',method_id=method,
        config_id=config_id,fold='F0',init_seed=42,state='READY',code_commit=git_identity()),view_manifest=manifest)
    if method=='M0':
        run['config']['n_estimators']=3
    else:
        run['config'].update(hidden_dim=16,ffn_dim=24,heads=4,layers=1,dropout=0.,max_epochs=2,min_epochs=1,
                             patience=1,effective_batch=4,micro_batch=2)
    run['config_sha256']=digest(run['config'])
    run['run_sha256']=digest({k:v for k,v in run.items() if k!='run_sha256'})
    write_json(path,run)
    return run


def test_train_select_predict_score_separation_and_seal(bundle,tmp_path,monkeypatch,record_property):
    # Removing test truths proves both training and select are independent of
    # them. Separate test input file remains available for sealed prediction.
    test_file=bundle/'view/test.npz';hold=bundle/'view/withheld_test.npz'
    test_file.rename(hold)
    directories=[]
    for config in ('C0','C1','C2','C3'):
        path=tmp_path/(config+'.json');run_spec(bundle,path,config)
        output=tmp_path/config
        run_training(path,bundle/'view',bundle/'cache',output)
        directories.append(output)
    selected=select_config(directories,tmp_path/'selection.json')
    record_property('synthetic_optimizer_steps',sum(read_json(d/'summary.json')['optimizer_steps'] for d in directories))
    assert selected['seed42_reused'] and not selected['refit_train_plus_validation']
    best=Path(selected['seed42_checkpoint'])
    assert file_sha256(best)==selected['seed42_checkpoint_sha256']
    sealed=sealed_predict(best,bundle/'view',bundle/'cache',tmp_path/'predictions')
    with np.load(tmp_path/'predictions/predictions.npz') as z:
        assert 'labels' not in z.files and z['prediction'].shape==(3,2)
    with pytest.raises(FileNotFoundError):
        score_prediction(tmp_path/'predictions',bundle/'view',tmp_path/'score.json',expected_checkpoint_sha256=file_sha256(best))
    hold.rename(test_file)
    result=score_prediction(tmp_path/'predictions',bundle/'view',tmp_path/'score.json',expected_checkpoint_sha256=file_sha256(best))
    assert result['role']=='test' and np.isfinite(result['macro_rmse'])
    with pytest.raises(ValueError,match='checkpoint'):
        score_prediction(tmp_path/'predictions',bundle/'view',tmp_path/'wrong.json',expected_checkpoint_sha256='0'*64)
    # Tampering with sealed bytes fails before metric computation.
    prediction=tmp_path/'predictions/predictions.npz'
    prediction.write_bytes(prediction.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):
        score_prediction(tmp_path/'predictions',bundle/'view',tmp_path/'bad.json',expected_checkpoint_sha256=file_sha256(best))


def test_rf_original_scale_and_cli_package(bundle,tmp_path,record_property):
    path=tmp_path/'rf.json';run_spec(bundle,path,method='M0')
    run_training(path,bundle/'view',bundle/'cache',tmp_path/'rf')
    record_property('synthetic_rf_endpoint_fits',2)
    from pgt_v2.training import read_checkpoint
    checkpoint=read_checkpoint(tmp_path/'rf/best.pt')
    assert checkpoint['forest'].tasks==tuple(load_manifest(bundle/'view')['task_names'])
    sealed_predict(tmp_path/'rf/best.pt',bundle/'view',bundle/'cache',tmp_path/'pred')
    score_prediction(tmp_path/'pred',bundle/'view',tmp_path/'score.json',expected_checkpoint_sha256=file_sha256(tmp_path/'rf/best.pt'))
    files={'score.json':str(tmp_path/'score.json'),'pred/manifest.json':str(tmp_path/'pred/manifest.json'),
           'pred/predictions.npz':str(tmp_path/'pred/predictions.npz')}
    write_json(tmp_path/'files.json',files)
    result=subprocess.run([sys.executable,'-B','-m','pgt_v2','package','--files',str(tmp_path/'files.json'),
        '--output',str(tmp_path/'bundle.zip')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with zipfile.ZipFile(tmp_path/'bundle.zip') as archive:
        assert set(archive.namelist())==set(files)|{'PACKAGE_SHA256.json'}


def test_cli_train_predict_score_and_explain_no_test_labels(bundle,tmp_path,record_property):
    path=tmp_path/'run.json';run_spec(bundle,path,method='A2')
    def cli(*args):
        result=subprocess.run([sys.executable,'-B','-m','pgt_v2',*map(str,args)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    test=bundle/'view/test.npz';held=bundle/'view/held.npz';test.rename(held)
    cli('train','--run',path,'--view',bundle/'view','--cache',bundle/'cache','--output',tmp_path/'run')
    record_property('synthetic_optimizer_steps',read_json(tmp_path/'run/summary.json')['optimizer_steps'])
    cli('predict','--checkpoint',tmp_path/'run/best.pt','--view',bundle/'view','--cache',bundle/'cache','--output',tmp_path/'pred')
    task=load_manifest(bundle/'view')['task_names'][0]
    cli('explain','--checkpoint',tmp_path/'run/best.pt','--view',bundle/'view','--cache',bundle/'cache',
        '--reference',bundle/'reference.json','--record-id','synthetic_9','--task',task,'--output',tmp_path/'explain')
    assert read_json(tmp_path/'explain/provenance.json')['max_reconstruction_error']<1e-5
    held.rename(test)
    cli('score','--predictions',tmp_path/'pred','--view',bundle/'view','--output',tmp_path/'score.json',
        '--expected-checkpoint-sha256',file_sha256(tmp_path/'run/best.pt'))


def test_unreleased_or_tampered_run_cannot_train(bundle,tmp_path):
    path=tmp_path/'run.json';run=run_spec(bundle,path)
    run['state']='PLANNED_NOT_RELEASED';write_json(path,run)
    with pytest.raises(ValueError,match='released'):run_training(path,bundle/'view',bundle/'cache',tmp_path/'run')
    run['state']='READY';run['config']['lr']=.3;write_json(path,run)
    with pytest.raises(ValueError,match='hash'):run_training(path,bundle/'view',bundle/'cache',tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_concrete_config_resolution_inherits_m6_without_placeholder(bundle):
    manifest=load_manifest(bundle/'view')
    selection=dict(method_id='M6',fold='F0',config_id='C2')
    selection['selection_sha256']=digest(selection)
    row=dict(method_id='A1',fold='F0',config_id='INHERIT')
    run=resolve_run(row,view_manifest=manifest,selection=selection)
    assert run['config']==candidate('A1','C2') and run['config_id']=='C2'
    with pytest.raises(ValueError):resolve_run(row,view_manifest=manifest)
    with pytest.raises(ValueError,match='fold'):resolve_run({**row,'fold':'F1'},view_manifest=manifest,selection=selection)


def test_validation_tie_uses_lowest_config_and_reuses_exact_checkpoint(tmp_path):
    from pgt_v2.training import save_checkpoint
    outputs=[]
    for i in range(4):
        folder=tmp_path/f'C{i}';folder.mkdir();outputs.append(folder)
        identity=dict(method_id='M6',fold='F0',view_sha256='a'*64,code_commit='b'*40,task_names=['task'],
                      init_seed=42,config_id=f'C{i}',config=candidate('M6',f'C{i}'))
        score=1. + (3e-13 if i==0 else 0.)
        save_checkpoint(folder/'best.pt',dict(kind='best',identity=identity,best_epoch=7,best=score))
        write_json(folder/'summary.json',dict(identity=identity,terminal='COMPLETE',best_epoch=7,
            validation={'macro_rmse':score},best_sha256=file_sha256(folder/'best.pt')))
    result=select_config(outputs,tmp_path/'selection.json')
    assert result['config_id']=='C0' and result['best_epoch']==7
    assert result['seed42_checkpoint_sha256']==file_sha256(outputs[0]/'best.pt')
