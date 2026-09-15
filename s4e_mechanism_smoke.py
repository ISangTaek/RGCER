"""S4E4 real-weight smoke only. Not a formal fifty-run mechanism evaluator."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from architecture.graphormer_backbone import MolecularGraphormerBackbone
from architecture.prediction_heads import TaskPredictionHead, decode_prediction
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
from s4e_mechanism_design import require, sha, read_json, write_json, unique, datastore_semantic_identity
from s4e_mechanism_metrics import hybrid_state, parameter_drift, linear_cka, functional_forgetting
from s4e_source_assets import load_verified_torch, _state_dict

PREFIX = 'encoder.backbone.'


class SourceModel(nn.Module):
    """Exact plain source inference topology; no adapters or target heads."""
    def __init__(self, architecture):
        super().__init__()
        a = architecture
        require(a['architecture'] == 'Graphormer' and a['prediction_mode'] == 'quantile'
                and a['task_names'] == list(ANIMAL_SOURCE_TASKS) and a['card_enabled'] is False,
                'unsupported source architecture')
        self.encoder = nn.Module()
        self.encoder.backbone = MolecularGraphormerBackbone(
            hidden_dim=a['hidden_dim'], num_heads=a['a_heads'], num_layers=a['a_layers'],
            ffn_dim=a['mid_dim'], dropout=0.1, spatial_pos_max_clip=a['spatial_pos_max_clip'],
            edge_bias_mode=a['edge_bias_mode'])
        self.decoders = nn.ModuleDict({t: TaskPredictionHead(a['hidden_dim'], mode='quantile',
            head_hidden_dim=a['head_hidden_dim'], dropout=a['head_dropout']) for t in ANIMAL_SOURCE_TASKS})

    def forward(self, batch):
        z = self.encoder.backbone(batch)
        return {t: decode_prediction(head(z), mode='quantile').median[:, 0]
                for t, head in self.decoders.items()}


def load_design(design, expected_sha):
    require(sha(design/'design_manifest.json') == expected_sha, 'design manifest SHA differs')
    manifest = read_json(design/'design_manifest.json')
    expected = {'mechanism_asset_lock.json', 'probe_manifest.json', 'probe_eligibility.json',
                'animal56_validation_manifest.json'}
    require(set(manifest['outputs']) == expected, 'design output set differs')
    for name, digest in manifest['outputs'].items():
        require(sha(design/name) == digest, f'design member differs: {name}')
    return manifest, {name: read_json(design/name) for name in expected}


def resolve_asset(repo, recorded):
    # Frozen paths refer to one original server checkout. Remapping only its
    # repository prefix supports WSL/server checkouts, never arbitrary paths.
    prefix = '/home/shangzeli/RGCER/'
    require(recorded.startswith(prefix), 'unknown historical repository root')
    p = (repo/recorded[len(prefix):]).resolve()
    require(p.is_relative_to(repo.resolve()), 'asset outside repository')
    return p


def state_from_asset(repo, asset):
    p = resolve_asset(repo, asset['path'])
    payload, receipt = load_verified_torch(p, asset['sha256'], repo,
        expected_size_bytes=asset['size_bytes'], label=asset.get('asset_id', 'best'))
    state = _state_dict(payload, raw_allowed=asset.get('kind') == 'init', label=str(p))
    if 'epoch' in asset:
        require(type(payload['epoch']) is int and payload['epoch'] == asset['epoch'], 'checkpoint epoch differs')
    return payload, state, receipt


def embeddings(model, batch):
    output, handles = {}, []
    def capture(name):
        def hook(module, inputs, value):
            require(value.ndim == 3, 'block output is not token sequence')
            output[name] = value[:, 0, :].detach().cpu().numpy().copy()
        return hook
    try:
        for i, block in enumerate(model.encoder.backbone.layers):
            handles.append(block.register_forward_hook(capture(f'block_{i}')))
        with torch.no_grad():
            z = model.encoder.backbone(batch)
        output['final_graph_token'] = z.detach().cpu().numpy().copy()
    finally:
        for h in handles: h.remove()
    return output


def predict(model, batches, device):
    values = {t: [] for t in ANIMAL_SOURCE_TASKS}
    with torch.no_grad():
        for b in batches:
            try:
                outputs = model(b.to(device))
                for t in values: values[t].append(outputs[t].detach().cpu())
            finally:
                b.cpu()
    return {t: torch.cat(v) for t,v in values.items()}


def smoke(repo, design, expected_design_sha, datastore, output, device_name):
    from dataset import DataCollator
    from preprocess_data import get_graph_data_from_smiles
    from toxacute_datastore import ToxAcuteDataStore
    from scripts.d6_prep_inits import _build_model
    require(not output.exists(), 'output exists; use a fresh run directory')
    require(device_name == 'cpu' or device_name.startswith('cuda:'), 'invalid device')
    manifest, docs = load_design(design, expected_design_sha)
    locks = docs['mechanism_asset_lock.json']
    assets = unique(locks['assets'], 'asset_id')
    all_probe = docs['probe_manifest.json']['ordered_samples']
    validation = docs['animal56_validation_manifest.json']['tasks']
    # Fixed smoke scope covers distinct seed42 50% and 100% initialization.
    runs = [r for r in locks['runs'] if r['seed'] == 42 and r['declared_fraction_percent'] in (50, 100)]
    require(len(runs) == 4 and {r['method'] for r in runs} == {'B1','RPT'}, 'smoke run matrix differs')
    selected = [next(x for x in all_probe if x['stratum'] == s) for s in
                ('human3_test', 'animal56_test_only', 'dataset115_external_test')]
    largest = max(all_probe, key=lambda x: x['input_record']['num_nodes'])
    if largest not in selected: selected.append(largest)
    require(len(selected) >= 3, 'smoke strata incomplete')
    require(datastore_semantic_identity(datastore) == manifest['datastore_semantic_identity'],
            'actual DataStore scientific content differs')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if device_name.startswith('cuda:'):
        require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    store = ToxAcuteDataStore(datastore)
    collator = DataCollator(spatial_pos_max_clip=20, max_node_filter=None)
    output.mkdir(parents=True)
    write_json(output/'scope.json', {'scope':'REAL_WEIGHT_SMOKE_ONLY', 'run_ids':[r['run_id'] for r in runs],
        'probe':selected, 'functional_rule':'first two frozen observations per Animal56 endpoint',
        'design_manifest_sha256':expected_design_sha, 'device':device_name,
        'torch_version':torch.__version__, 'formal_metrics':False,
        'commit':subprocess.run(['git','rev-parse','HEAD'],cwd=repo,check=True,capture_output=True,text=True).stdout.strip()})
    records, graphs = [], []
    graph_fields = ('x','in_degree','out_degree','spatial_pos','attn_edge_type','edge_input')
    try:
        for sample in selected:
            rec = sample['input_record']
            rebuilt = get_graph_data_from_smiles(rec['raw_smiles'], 0.0,
                sample_id=sample['sample_key'], max_path_distance=store.metadata['max_path_distance'])
            require(rebuilt.canonical_smiles == sample['canonical_smiles'], 'graph canonical differs from frozen identity')
            require(rebuilt.x.shape[0] == rec['num_nodes'], 'graph node count differs')
            if rec['source'] == 'toxacute':
                g = store.get_graph_data(rec['global_index'])
                require(g.sample_id == rec['sample_id'], 'probe graph ID differs')
                for field in graph_fields:
                    require(torch.equal(getattr(rebuilt, field), getattr(g, field)), f'raw/LMDB graph differs: {field}')
                g.sample_id = sample['sample_key']
            else:
                g = rebuilt
            graphs.append(g)
        batch = collator(graphs).to(device)
        ids = [s['sample_key'] for s in selected]
        require(list(batch.sample_id) == ids, 'probe collator reordered/dropped samples')
        # Use a shared graph for repeated endpoint observations; predictions
        # still retain their endpoint/sample identities and original scalers.
        obs_by_task = {v['task']: v['observations'][:2] for v in validation}
        global_ids = sorted({o['global_index'] for obs in obs_by_task.values() for o in obs})
        graph_pos = {g:i for i,g in enumerate(global_ids)}
        for task, obs in obs_by_task.items():
            for o in obs:
                require(store.sample_ids[o['global_index']] == o['sample_id']
                        and store.get_label(o['global_index'], task) == o['label_raw'], 'functional label/ID mismatch')
        eval_batches = [collator([store.get_graph_data(i) for i in global_ids[start:start+4]])
                        for start in range(0,len(global_ids),4)]
        source_cache = {}
        for run in runs:
            t = assets[run['teacher_asset_id']]; init = assets[run['init_asset_id']]
            payload, source_state, treceipt = state_from_asset(repo, t)
            _, init_state, ireceipt = state_from_asset(repo, init)
            _, best_state, breceipt = state_from_asset(repo, run['best'])
            # Validate current tensors independently of recorded MATCH flags.
            for k, v in init_state.items():
                if k.startswith('encoder.'):
                    require(k in source_state and torch.equal(v, source_state[k]), 'live init/source mismatch')
            require({k for k in init_state if k.startswith(PREFIX)} ==
                    {k for k in source_state if k.startswith(PREFIX)}, 'init backbone keys differ')
            model = SourceModel(t['architecture_config'])
            model.load_state_dict(source_state, strict=True)
            model.to(device).eval()
            scaler = unique(t['scalers'], 'task')
            for task in ANIMAL_SOURCE_TASKS:
                require(payload['task_scalers'][task]['mean'] == scaler[task]['mean']
                        and payload['task_scalers'][task]['std'] == scaler[task]['std']
                        and scaler[task]['std'] > 0, 'source scaler differs')
            # Compare the new inference topology with the project's original
            # model constructor, strict state loading, and decoder path.
            if t['asset_id'] not in source_cache:
                original, _, tasks = _build_model(t['seed'], 'animal56', device, t['configuration'])
                require(list(tasks) == list(ANIMAL_SOURCE_TASKS), 'original task order differs')
                original.load_state_dict(source_state, strict=True)
                original.to(device).eval()
                with torch.no_grad():
                    for eb in eval_batches:
                        try:
                            eb.to(device)
                            predictions = model(eb)
                            z = original.encoder(eb)
                            for task in tasks:
                                reference = decode_prediction(original.decoders[task](z), mode='quantile').median[:,0]
                                require(torch.allclose(reference, predictions[task], rtol=1e-6, atol=1e-7), 'new/original source forward differs')
                        finally:
                            eb.cpu()
                del original
                source_cache[t['asset_id']] = True
            before = predict(model, eval_batches, device)
            before_embed = embeddings(model, batch)
            candidate_bb = {k:v for k,v in best_state.items() if k.startswith(PREFIX)}
            reference_bb = {k:v for k,v in init_state.items() if k.startswith(PREFIX)}
            drift = parameter_drift({k:v.numpy() for k,v in reference_bb.items()}, {k:v.numpy() for k,v in candidate_bb.items()})
            model.load_state_dict(hybrid_state(source_state, candidate_bb), strict=True)
            model.eval()
            after = predict(model, eval_batches, device)
            after_embed = embeddings(model, batch)
            again = embeddings(model, batch)
            require(all(np.array_equal(after_embed[k], again[k]) for k in after_embed), 'repeated embedding differs')
            cka = {k:linear_cka(before_embed[k], after_embed[k], ids, ids) for k in before_embed}
            labels, sp, hp, expected_ids, prediction_rows = {}, {}, {}, {}, []
            for task in ANIMAL_SOURCE_TASKS:
                obs = obs_by_task[task]; pos = [graph_pos[o['global_index']] for o in obs]
                expected_ids[task] = [o['sample_id'] for o in obs]
                labels[task] = [o['label_raw'] for o in obs]
                # Decode in model FP32, inverse-scale in float64 for the saved
                # diagnostic values; original scaler values are not refit.
                a = before[task].detach().cpu().numpy().astype('f8')[pos]*scaler[task]['std']+scaler[task]['mean']
                b = after[task].detach().cpu().numpy().astype('f8')[pos]*scaler[task]['std']+scaler[task]['mean']
                sp[task], hp[task] = (expected_ids[task], a), (expected_ids[task], b)
                for o, va, vb in zip(obs,a,b):
                    prediction_rows.append(dict(o, task=task, source_prediction=float(va), hybrid_prediction=float(vb)))
            metrics = functional_forgetting(labels, sp, hp, list(ANIMAL_SOURCE_TASKS), expected_ids)
            if run['method'] == 'RPT':
                require(drift['difference_l2'] == 0, 'RPT frozen backbone changed')
                require(all(x['delta_rmse'] == 0 for x in metrics['tasks']), 'RPT source preservation failed')
            rd = output/run['run_id']; rd.mkdir()
            np.savez_compressed(rd/'embeddings.npz', **{'source_'+k:v for k,v in before_embed.items()},
                                **{'hybrid_'+k:v for k,v in after_embed.items()})
            write_json(rd/'predictions.json', prediction_rows)
            write_json(rd/'metrics.json', {'scope':'SMOKE_NOT_FIGURE', 'drift':drift,'cka':cka,'functional':metrics})
            records.append({'run_id':run['run_id'], 'source':treceipt,'init':ireceipt,'best':breceipt})
            del model
        write_json(output/'smoke_receipt.json', {'scope':'SMOKE_NOT_FIGURE','validation_status':'PASS',
            'acceptance_status':'PENDING_CODEX_REVIEW','runs':records,'source_forward_equivalence_count':len(source_cache)})
    finally:
        store.close()
    files = sorted(p for p in output.rglob('*') if p.is_file())
    write_json(output/'checksums.json', {p.relative_to(output).as_posix():sha(p) for p in files})


def verify_smoke(design, expected_design_sha, output):
    """Read-only recheck against the independently supplied frozen design."""
    _, docs = load_design(design, expected_design_sha)
    lock = docs['mechanism_asset_lock.json']
    runs = {r['run_id']:r for r in lock['runs'] if r['seed'] == 42 and r['declared_fraction_percent'] in (50,100)}
    expected_files = {'scope.json','smoke_receipt.json'} | {
        r+'/'+n for r in runs for n in ('predictions.json','metrics.json','embeddings.npz')}
    checks = read_json(output/'checksums.json')
    require(set(checks) == expected_files, 'smoke file set differs')
    for name in expected_files: require(sha(output/name) == checks[name], f'smoke file checksum differs: {name}')
    scope = read_json(output/'scope.json')
    all_probe = docs['probe_manifest.json']['ordered_samples']
    selected = [next(x for x in all_probe if x['stratum'] == s) for s in
                ('human3_test','animal56_test_only','dataset115_external_test')]
    largest = max(all_probe, key=lambda x:x['input_record']['num_nodes'])
    if largest not in selected: selected.append(largest)
    require(scope['probe'] == selected and set(scope['run_ids']) == set(runs)
            and len(scope['run_ids']) == 4 and scope['design_manifest_sha256'] == expected_design_sha
            and scope['formal_metrics'] is False, 'smoke scope differs')
    receipt = read_json(output/'smoke_receipt.json')
    rr = unique(receipt['runs'],'run_id')
    require(set(rr) == set(runs), 'receipt run set differs')
    require(receipt['source_forward_equivalence_count'] == len({r['teacher_asset_id'] for r in runs.values()}),
            'source forward equivalence coverage differs')
    assets = unique(lock['assets'],'asset_id')
    ids = [x['sample_key'] for x in selected]
    expected_obs = {v['task']:v['observations'][:2] for v in docs['animal56_validation_manifest.json']['tasks']}
    for run_id, run in runs.items():
        for key, a in [('source', assets[run['teacher_asset_id']]), ('init',assets[run['init_asset_id']]), ('best',run['best'])]:
            require(rr[run_id][key]['sha256'] == a['sha256'], 'receipt asset SHA differs')
        rd = output/run_id
        rows = read_json(rd/'predictions.json'); actual = {}
        for row in rows:
            key = (row['task'],row['sample_id'])
            require(key not in actual,'duplicate prediction'); actual[key]=row
        expected_keys = {(t,o['sample_id']) for t,obs in expected_obs.items() for o in obs}
        require(set(actual) == expected_keys,'prediction sample set differs')
        labels, sp, hp, eid = {}, {}, {}, {}
        for t, obs in expected_obs.items():
            eid[t] = [o['sample_id'] for o in obs]
            for o in obs:
                require(all(actual[t,o['sample_id']][k] == v for k,v in o.items()),'prediction label/identity differs')
            labels[t] = [o['label_raw'] for o in obs]
            sp[t] = (eid[t],[actual[t,i]['source_prediction'] for i in eid[t]])
            hp[t] = (eid[t],[actual[t,i]['hybrid_prediction'] for i in eid[t]])
        metrics = read_json(rd/'metrics.json')
        require(metrics['scope'] == 'SMOKE_NOT_FIGURE', 'wrong metric evidence level')
        require(functional_forgetting(labels,sp,hp,list(ANIMAL_SOURCE_TASKS),eid) == metrics['functional'], 'functional recomputation differs')
        layers = [f'block_{i}' for i in range(assets[run['teacher_asset_id']]['architecture_config']['a_layers'])]+['final_graph_token']
        with np.load(rd/'embeddings.npz',allow_pickle=False) as z:
            require(set(z.files) == {p+k for p in ('source_','hybrid_') for k in layers}, 'embedding layer set differs')
            ck = {k:linear_cka(z['source_'+k],z['hybrid_'+k],ids,ids) for k in layers}
        require(ck == metrics['cka'], 'CKA recomputation differs')
        if run['method'] == 'RPT':
            require(metrics['drift']['difference_l2'] == 0
                    and all(x['delta_rmse']==0 for x in metrics['functional']['tasks']), 'RPT invariant failed')
    return {'validation_status':'PASS','acceptance_status':'PENDING_CODEX_REVIEW',
            'scope':'SMOKE_NOT_FIGURE','run_count':len(runs),'probe_count':len(selected),
            'observations_per_run':len(expected_keys)}
