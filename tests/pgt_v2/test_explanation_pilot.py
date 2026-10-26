"""Focused explanation checks with synthetic/random weights; no optimizer."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest
from rdkit import Chem
import torch

from pgt_v2.artifacts import load_arrays, load_manifest, read_json, write_json
from pgt_v2.cache import GraphCache
from pgt_v2.config import candidate
from pgt_v2.data import PGTCollator
from pgt_v2.explain import integrated_gradients
from pgt_v2.explanation_pilot import execute, predict_molecules
from pgt_v2.fragment_explanations import build_library, make_panel, compare_ranking, evaluate_fragments
from pgt_v2.models import build_model
from pgt_v2.training import save_checkpoint

TASKS=['mouse_oral_LD50','rat_oral_LD50','human_oral_TDLo','rat_subcutaneous_LD50']


def config(method):
    return {**candidate(method,'C0' if method=='M3' else 'C2'),
            'hidden_dim':16,'ffn_dim':24,'heads':4,'layers':1,'dropout':0.0}


def test_shared_substitution_panel_is_disjoint_and_preserves_context_mapping():
    from pgt_v2.chemistry import prepare_molecule
    training=['CCOc1ccccc1','CCCOc1ccccc1','CCCCOc1ccccc1','COc1ccccc1','COc1ccncc1',
              'CCOc1ccncc1','CCCOc1ccncc1','COc1ccc(F)cc1','COc1ccc(Cl)cc1',
              'COc1ccc(C)cc1','COc1cccc(C)c1','COc1ccc(CC)cc1','COc1ccc(N)cc1',
              'CC(C)Oc1ccccc1','CC(F)Oc1ccccc1','CC(Cl)Oc1ccccc1','NCCOc1ccccc1','OCCOc1ccccc1']
    library=build_library((f'train{i}',Chem.MolFromSmiles(s)) for i,s in enumerate(training))
    mol=prepare_molecule('CCOc1ccc(CC)cc1').mol
    panel=make_panel(mol,library,seed=20261026)
    other=make_panel(mol,library,seed=20261026)
    assert panel['candidates'] and len(panel['candidates'])<=96
    assert [(x['smiles'],x['set']) for x in panel['candidates']]==[(x['smiles'],x['set']) for x in other['candidates']]
    a={x['smiles'] for x in panel['candidates'] if x['set']=='A'}
    b={x['smiles'] for x in panel['candidates'] if x['set']=='B'}
    assert b and a.isdisjoint(b)
    for item in panel['candidates']:
        replacement=item['molecule'].mol
        assert all(atom.GetAtomicNum()>0 for atom in replacement.GetAtoms())
        removed=set(panel['fragments'][item['fragment_id']]['atoms'])
        mapping={int(i):j for i,j in item['retained_original_to_model'].items()}
        assert set(mapping)==set(range(mol.GetNumAtoms()))-removed
        for old,new in mapping.items():
            assert mol.GetAtomWithIdx(old).GetAtomicNum()==replacement.GetAtomWithIdx(new).GetAtomicNum()


def test_fidelity_is_scale_invariant_and_ranking_does_not_consume_b():
    fragments=[dict(fragment_id=i,size_bin=3,aromatic=False,atoms=[i]) for i in range(3)]
    score=compare_ranking([3.,2.,1.],[9.,2.,1.],fragments)
    assert score['capture_gain']==pytest.approx(9/12-1/3)
    assert compare_ranking([3.,2.,1.],[90.,20.,10.],fragments)['capture_gain']==pytest.approx(score['capture_gain'])
    assert compare_ranking([1.,1.,1.],[9.,2.,1.],fragments)['capture_gain']==pytest.approx(0.)
    assert compare_ranking([3.,2.,1.],[0.,0.,0.],fragments)['capture_gain'] is None
    panel=dict(fragments=fragments,candidates=[dict(fragment_id=i,set=s) for i in range(3) for s in ('A','B')])
    predictions=np.array([[0.],[3.],[9.],[2.],[2.],[1.],[1.]])
    original=evaluate_fragments(panel,predictions,task=0,scale=1)
    changed=predictions.copy();changed[[2,4,6],0]=[1,2,9]
    revised=evaluate_fragments(panel,changed,task=0,scale=1)
    assert original['a_mean_absolute_response']==revised['a_mean_absolute_response']
    assert original['metrics']['common_perturbation']['capture_gain']>revised['metrics']['common_perturbation']['capture_gain']


@pytest.mark.parametrize('method',['N1','N1S'])
def test_full_model_ig_endpoint_difference_and_padding(method):
    from pgt_v2.chemistry import graph_from_input,prepare_molecule
    torch.set_num_threads(1);torch.manual_seed(12)
    model=build_model(method,TASKS,config(method),counts=[10]*len(TASKS)).eval()
    graphs=[graph_from_input(prepare_molecule(s),sample_id=f'r{i}',raw_row_index=i)
            for i,s in enumerate(['CCO','C'])]
    batch=PGTCollator()(graphs)
    mean=np.array([.1,.2,.3,.4]);scale=np.array([1.,2.,3.,4.])
    left=integrated_gradients(model,batch,task=0,mean=mean,scale=scale)
    right=integrated_gradients(model,batch,task=1,mean=mean,scale=scale)
    diff=integrated_gradients(model,batch,task=0,other_task=1,mean=mean,scale=scale)
    torch.testing.assert_close(diff['attributions'],left['attributions']-right['attributions'],atol=2e-6,rtol=1e-4)
    assert torch.isfinite(diff['attributions']).all()
    assert torch.all(diff['attributions'][batch.padding_mask]==0)
    assert diff['completeness_residual'].abs().max()<1e-3


def test_entire_read_only_pilot_uses_existing_weights_without_test_labels(tmp_path):
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('pgt_support',Path(__file__).with_name('support.py'))
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    support.synthetic_bundle(tmp_path/'data',TASKS)
    view=tmp_path/'data/view';manifest=load_manifest(view)
    stats=load_arrays(view,'statistics.npz',manifest)
    inputs=load_arrays(view,'validation_inputs.npz',manifest)
    cache=GraphCache(tmp_path/'data/cache',source_sha256=manifest['source_sha256'])
    graph,mol=cache.read(int(inputs['cache_offset'][1]))
    registry=dict(task_id='SYNTHETIC_C06E_NO_TRAINING',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        task_names=TASKS,targets=TASKS[:2],view_manifest_sha256=manifest['manifest_sha256'],
        reference_record_ids=[f'synthetic_{i}' for i in range(6)],
        panel=[dict(record_id=graph.sample_id,cache_offset=int(inputs['cache_offset'][1]),chemical_id=graph.model_input_id,
            group_id=str(inputs['group_id'][1]),observations=[dict(task_index=0,truth=.5),dict(task_index=1,truth=.7)])],runs=[])
    for method in ('N1','N1S','M3'):
        model=build_model(method,TASKS,config(method),adjacency=stats['adjacency'],counts=stats['count']).eval()
        identity=dict(run_id='synthetic_'+method,method_id=method,init_seed=42,task_names=TASKS,config=config(method),
            view_sha256=manifest['manifest_sha256'])
        checkpoint=tmp_path/(method+'.pt')
        save_checkpoint(checkpoint,dict(kind='best',identity=identity,model=model.state_dict(),stats=stats))
        pred=predict_molecules(model,[graph],[mol],method,stats,'cpu')[0]
        registry['runs'].append(dict(run_id=identity['run_id'],method=method,seed=42,checkpoint=str(checkpoint),
            checkpoint_sha256=read_json(str(checkpoint)+'.sha256.json')['sha256'],expected_predictions={graph.sample_id:pred.tolist()}))
    write_json(tmp_path/'registry.json',registry)
    # The complete inference command does not need either test inputs or truths.
    (view/'test.npz').unlink();(view/'test_inputs.npz').unlink()
    result=execute(tmp_path/'registry.json',view,tmp_path/'data/cache',tmp_path/'result')
    assert result['status']=='COMPLETE' and result['counts']['new_fits']==0
    assert result['counts']['main_ig_sample_gradient_points']==512
    rows=read_json(tmp_path/'result/observations.json')
    assert len(rows)==6 and {row['method'] for row in rows}=={'N1','N1S','M3'}
    assert all(not row['ig'] for row in rows if row['method']=='M3')
    differences=read_json(tmp_path/'result/synthetic_N1'/graph.sample_id/'endpoint_differences.json')
    assert len(differences)==1 and differences[0]['ig']['zero']['attributions']
    assert read_json(tmp_path/'result/summary.json')['status']=='DESCRIPTIVE_DEVELOPMENT_EVIDENCE'
