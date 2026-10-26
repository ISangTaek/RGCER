"""Bounded G02I inference on nine accepted weights; never trains a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from rdkit import Chem, DataStructs
import torch

from .artifacts import load_arrays, load_manifest, read_json, write_json
from .cache import GraphCache
from .chemistry import graph_from_input
from .data import PGTCollator
from .evaluation import load_model_checkpoint
from .explain import integrated_gradients, training_embedding_mean
from .fragment_explanations import (MORGAN, build_library, make_panel, evaluate_fragments,
                                   correlation)
from .models import molecular_features


def molecular_batch(graphs, mols, method, device):
    if method == 'M3':
        return torch.as_tensor(molecular_features(mols, 'avalon'), device=device)
    return PGTCollator()(graphs).to(device)


@torch.no_grad()
def predict_molecules(model, graphs, mols, method, stats, device, batch_size=4):
    values = []
    for start in range(0, len(graphs), batch_size):
        batch = molecular_batch(graphs[start:start+batch_size], mols[start:start+batch_size], method, device)
        values.append(model(batch).cpu().numpy()*np.asarray(stats['scale'])+np.asarray(stats['mean']))
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite model prediction')
    return result


def check_reference_predictions(actual, expected):
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-4, rtol=1e-5):
        raise ValueError('Original prediction does not reproduce the accepted checkpoint result')
    return float(np.abs(actual-expected).max())


def endpoint_differences(rows):
    """Same-path IG is linear in the output; differences need no new gradients."""
    differences=[]
    for left,right in combinations(rows,2):
        if left['task'].rsplit('_',1)[-1] != right['task'].rsplit('_',1)[-1]:
            continue
        item=dict(left=left['task'],right=right['task'],prediction=left['prediction']-right['prediction'],
                  truth=left['truth']-right['truth'],ig={},
                  interpretation='Same-measurement endpoint difference; no population or species causal claim')
        for name in left['ig']:
            a,b=left['ig'][name],right['ig'][name]
            contribution=np.asarray(a['attributions'])-np.asarray(b['attributions'])
            baseline=a['baseline_prediction']-b['baseline_prediction']
            prediction=a['prediction']-b['prediction']
            residual=float(contribution.sum()-(prediction-baseline))
            item['ig'][name]=dict(attributions=contribution.tolist(),baseline_prediction=baseline,
                prediction=prediction,completeness_residual=residual,
                normalized_residual=abs(residual)/max(abs(prediction-baseline),1e-8),
                source='difference of two same-baseline, same-path full-model attributions')
        differences.append(item)
    return differences


def load_runs(registry, stats, device):
    runs = []
    for registered in registry['runs']:
        path = Path(registered['checkpoint'])
        # Existing small checkpoint check is done once per load, not per atom.
        if read_json(str(path)+'.sha256.json')['sha256'] != registered['checkpoint_sha256']:
            raise ValueError('Checkpoint sidecar differs from the accepted result')
        model, checkpoint = load_model_checkpoint(path, device=device)
        identity = checkpoint['identity']
        if (identity['run_id'] != registered['run_id'] or identity['method_id'] != registered['method'] or
                identity['init_seed'] != registered['seed'] or identity['task_names'] != registry['task_names'] or
                identity['view_sha256'] != registry['view_manifest_sha256']):
            raise ValueError('Wrong accepted run, task order, or training view')
        for key in ('mean', 'scale', 'count', 'adjacency'):
            if not np.array_equal(np.asarray(checkpoint['stats'][key]), stats[key]):
                raise ValueError('Checkpoint training scale/view statistics changed: '+key)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        runs.append(dict(registration=registered, model=model, stats=checkpoint['stats']))
    return runs


def summarize(rows, tasks):
    """Descriptive summaries; full rows and failures remain in returned files."""
    values = defaultdict(list)
    stability = []
    observations = defaultdict(list)
    for row in rows:
        observations[(row['method'],row['record_id'],row['task'])].append(row)
        for name, metric in row['fragment_evaluation']['metrics'].items():
            values[(row['method'],row['seed'],row['task'],name)].append(metric)
    per_task = []
    for (method,seed,task,name), items in sorted(values.items()):
        usable = [x['capture_gain'] for x in items if x['capture_gain'] is not None]
        per_task.append(dict(method=method,seed=seed,task=task,explainer=name,
            total_observations=len(items),evaluable_observations=len(usable),
            mean_capture_gain=float(np.mean(usable)) if usable else None))
    for (method,record,task), items in sorted(observations.items()):
        for left,right in combinations(sorted(items,key=lambda r:r['seed']),2):
            a = left['fragment_evaluation']['a_mean_absolute_response']
            b = right['fragment_evaluation']['a_mean_absolute_response']
            valid = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
            item = dict(method=method,record_id=record,task=task,seeds=[left['seed'],right['seed']],
                prediction_abs_difference=abs(left['prediction']-right['prediction']),
                common_fragment_rank_correlation=correlation([a[i] for i in valid],[b[i] for i in valid]),ig={})
            for baseline in left['ig']:
                x,y = [np.asarray(r['ig'][baseline]['attributions']) for r in (left,right)]
                k=max(1,int(np.ceil(.2*len(x))))
                topx=set(np.argsort(-np.abs(x),kind='stable')[:k].tolist())
                topy=set(np.argsort(-np.abs(y),kind='stable')[:k].tolist())
                nonzero=(np.abs(x)>1e-10)|(np.abs(y)>1e-10)
                item['ig'][baseline]=dict(rank_correlation=correlation(np.abs(x),np.abs(y)),
                    top20_jaccard=len(topx&topy)/len(topx|topy),
                    sign_agreement=float((np.sign(x[nonzero])==np.sign(y[nonzero])).mean()) if nonzero.any() else None,
                    either_completeness_flag=left['ig'][baseline]['unreliable'] or right['ig'][baseline]['unreliable'])
            stability.append(item)
    # All three methods and all three seeds must have a score for a paired item.
    by_obs=defaultdict(list)
    for row in rows:
        by_obs[(row['record_id'],row['task'])].append(row)
    common=[items for items in by_obs.values() if len(items)==9 and
            all(r['fragment_evaluation']['metrics']['common_perturbation']['capture_gain'] is not None for r in items)]
    paired=[]
    for method in ('N1','N1S'):
        for seed in (42,43,44):
            differences=defaultdict(list)
            for items in common:
                lookup={(r['method'],r['seed']):r for r in items}
                a,b=lookup[(method,seed)],lookup[('M3',seed)]
                differences[a['task']].append(a['fragment_evaluation']['metrics']['common_perturbation']['capture_gain']-
                    b['fragment_evaluation']['metrics']['common_perturbation']['capture_gain'])
            task_means={t:float(np.mean(differences[t])) if differences[t] else None for t in tasks}
            paired.append(dict(method=method,seed=seed,comparator='M3',per_task_difference=task_means,
                covered_tasks=sum(v is not None for v in task_means.values()),
                target6_macro_difference=float(np.mean(list(task_means.values()))) if all(v is not None for v in task_means.values()) else None))
    return dict(status='DESCRIPTIVE_DEVELOPMENT_EVIDENCE',per_task=per_task,stability=stability,
        shared_evaluable_observations=len(common),paired_common_explainer=paired,
        ig_flagged=sum(v['unreliable'] for r in rows for v in r['ig'].values()),
        warning='No formal test, causal mechanism, calibrated uncertainty, or automatic scientific acceptance.')


def execute(registry_path, view, cache_root, output, *, device='cpu', preflight=False):
    registry = read_json(registry_path)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    counter=dict(new_fits=0,main_ig_sample_gradient_points=0,perturbation_sample_forwards=0,
                 auxiliary_sample_forwards=0,reference_atom_encoder_samples=0)
    note=dict(task_id=registry['task_id'],stage='server_preflight' if preflight else 'development_explanation',
        python=sys.executable,hostname=platform.node(),device=device,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status='RUNNING',counts=counter,checkpoint_policy='One existing checksum per small checkpoint load; no full-cache rehash',
        training=False,test_partition_read=False)
    write_json(output/'execution.json',note)
    try:
        if note['code_commit'] != registry['code_commit']:
            raise ValueError('Use the released code commit')
        manifest=load_manifest(view)
        if manifest['manifest_sha256'] != registry['view_manifest_sha256']:
            raise ValueError('Use the accepted mini12 view')
        stats=load_arrays(view,'statistics.npz',manifest)
        train=load_arrays(view,'train_inputs.npz',manifest)
        cache=GraphCache(cache_root,source_sha256=manifest['source_sha256'],verify_database=False)
        if cache.manifest['cache_id'] != manifest['cache_id']:
            raise ValueError('Wrong accepted graph cache')
        runs=load_runs(registry,stats,device)
        if preflight:
            panel=registry['panel'][0]
            graph,mol=cache.read(panel['cache_offset'])
            checks=[]
            for run in runs:
                reg=run['registration']
                prediction=predict_molecules(run['model'],[graph],[mol],reg['method'],run['stats'],device)
                error=check_reference_predictions(prediction[0],reg['expected_predictions'][panel['record_id']])
                checks.append(dict(run_id=reg['run_id'],max_abs_prediction_difference=error))
            counter['auxiliary_sample_forwards']=len(runs)
            write_json(output/'checkpoint_reproduction.json',checks)
        else:
            reference_ids=set(registry['reference_record_ids'])
            reference_graphs=[]
            training_fps=[]
            def training_molecules():
                for record,offset in zip(train['record_id'],train['cache_offset']):
                    graph,mol=cache.read(int(offset))
                    if graph.sample_id != str(record):
                        raise ValueError('Training fragment registry differs from cache')
                    training_fps.append(MORGAN.GetFingerprint(mol))
                    if str(record) in reference_ids:
                        reference_graphs.append(graph)
                    yield str(record),mol
            library=build_library(training_molecules())
            if len(reference_graphs) != len(reference_ids):
                raise ValueError('Embedding reference must contain training records only')
            write_json(output/'training_reference.json',dict(record_ids=registry['reference_record_ids'],
                fragment_training_records=len(train['record_id']),library_fragments=sum(map(len,library.values())),
                library_environments=len(library),labels_used=False))
            for run in runs:
                reg=run['registration']
                if reg['method'] != 'M3':
                    batches=(PGTCollator()(reference_graphs[i:i+8]).to(device) for i in range(0,len(reference_graphs),8))
                    run['embedding_mean']=training_embedding_mean(run['model'],batches)
                    counter['reference_atom_encoder_samples']+=len(reference_graphs)
                    write_json(output/reg['run_id']/'embedding_reference.json',dict(record_ids=registry['reference_record_ids'],
                        mean=run['embedding_mean'].cpu().tolist()))
            rows=[]
            for panel_index, observation in enumerate(registry['panel']):
                graph,mol=cache.read(observation['cache_offset'])
                if graph.sample_id != observation['record_id'] or graph.model_input_id != observation['chemical_id']:
                    raise ValueError('Panel/cache record mismatch')
                panel=make_panel(mol,library,seed=20261026+panel_index)
                graphs=[graph]+[graph_from_input(c['molecule'],sample_id=f'perturb_{i}',raw_row_index=0)
                    for i,c in enumerate(panel['candidates'])]
                mols=[mol]+[c['molecule'].mol for c in panel['candidates']]
                for candidate in panel['candidates']:
                    fp=MORGAN.GetFingerprint(candidate['molecule'].mol)
                    candidate['max_training_similarity']=float(max(DataStructs.BulkTanimotoSimilarity(fp,training_fps)))
                meta={k:v for k,v in panel.items() if k!='candidates'}
                meta.update(record=observation,canonical_smiles=graph.canonical_smiles,
                    model_to_raw=graph.model_to_raw,raw_to_model=graph.raw_to_model,symmetry_classes=graph.symmetry_classes,
                    candidates=[{k:v for k,v in c.items() if k!='molecule'} for c in panel['candidates']])
                write_json(output/'structures'/f'{graph.sample_id}.json',meta)
                for run in runs:
                    reg,model=run['registration'],run['model']
                    predictions=predict_molecules(model,graphs,mols,reg['method'],run['stats'],device)
                    counter['perturbation_sample_forwards']+=len(graphs)
                    error=check_reference_predictions(predictions[0],reg['expected_predictions'][graph.sample_id])
                    path=output/reg['run_id']/graph.sample_id;path.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path/'predictions.npz',prediction=predictions)
                    local_rows=[]
                    batch=None;attention=None
                    if reg['method'] != 'M3':
                        batch=PGTCollator()([graph]).to(device)
                        with torch.no_grad():
                            _,aux=model(batch,return_aux=True)
                            attention=aux['prompt_atom_attention'][0].mean(1).cpu().numpy()
                        counter['auxiliary_sample_forwards']+=1
                    for task_info in observation['observations']:
                        task=task_info['task_index']; ig={}; atom_values={}
                        if batch is not None:
                            atom_values['attention']=attention[task].tolist()
                            for baseline,base in [('zero',None),('training_mean',run['embedding_mean'])]:
                                result=integrated_gradients(model,batch,task=task,mean=run['stats']['mean'],
                                    scale=run['stats']['scale'],baseline=base,points=64)
                                ig[baseline]={k:v.detach().cpu().numpy()[0].tolist() for k,v in result.items() if isinstance(v,torch.Tensor)}
                                ig[baseline]['path_power']=result['path_power']
                                atom_values['ig_'+baseline]=ig[baseline]['attributions']
                                counter['main_ig_sample_gradient_points']+=64
                                counter['auxiliary_sample_forwards']+=2
                        row=dict(run_id=reg['run_id'],method=reg['method'],seed=reg['seed'],record_id=graph.sample_id,
                            chemical_id=observation['chemical_id'],group_id=observation['group_id'],task=registry['task_names'][task],
                            truth=task_info['truth'],prediction=float(predictions[0,task]),prediction_reproduction_max_abs=error,
                            ig=ig,attention=atom_values.get('attention'),
                            fragment_evaluation=evaluate_fragments(panel,predictions,task=task,scale=run['stats']['scale'][task],
                                atom_explanations=atom_values))
                        local_rows.append(row);rows.append(row)
                    write_json(path/'explanations.json',local_rows)
                    write_json(path/'endpoint_differences.json',endpoint_differences(local_rows))
                note.update(completed_records=panel_index+1,completed_task_model_observations=len(rows),counts=counter,
                            elapsed_seconds=time.perf_counter()-started)
                write_json(output/'execution.json',note)
                print(f'{panel_index+1}/{len(registry["panel"])} records; {len(rows)} task-model observations',flush=True)
            write_json(output/'observations.json',rows)
            write_json(output/'summary.json',summarize(rows,registry['targets']))
            if counter['main_ig_sample_gradient_points'] > 41472 or counter['perturbation_sample_forwards'] > 47142:
                raise RuntimeError('Registered explanation budget exceeded')
        note.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-started,counts=counter)
        if str(device).startswith('cuda'):
            note['peak_cuda_bytes']=torch.cuda.max_memory_allocated(device)
        write_json(output/'execution.json',note)
        return note
    except Exception as exc:
        note.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),counts=counter,
                    elapsed_seconds=time.perf_counter()-started)
        write_json(output/'execution.json',note)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',required=True)
    parser.add_argument('--view',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    execute(args.registry,args.view,args.cache,args.output,device=args.device,preflight=args.preflight)


if __name__=='__main__':
    main()
