"""D2 train/validation adapter bound to reviewed five-seed identities.

PT/RT load source count metadata as a common clock, but never construct source
training datasets, fit source scalers, or fetch source graph/label batches.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from d2_control import Control, encoder
from reproducibility import state_dict_sha256
from v9_cost_probe import read, sha, digest, require


def extract_encoder(state, args, expected):
    from dataset115_smoke import state_digest
    values = {k[8:]: v for k, v in state.items() if k.startswith('encoder.')}
    require(values and all(torch.isfinite(v).all() for v in values.values()), 'finite source encoder')
    probe = encoder(args, 42)
    probe.load_state_dict(values, strict=True)
    require(state_digest(probe.state_dict()) == expected, 'trusted initial encoder identity')
    return {k: v.detach().cpu().clone() for k, v in values.items()}


def locked_source(factory, setting, seed, repo, source_lock, args, contract):
    """Read immutable weight/count metadata only, not a source label view."""
    if setting == 'A':
        from dataset115_contract import semantic_digest
        path = factory._output/'epoch_039.pt'
        bound = contract['source_identity']
        require(bound['seed'] == seed and sha(path) == bound['teacher_sha256'], 'same-seed A source bytes')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(type(payload['epoch']) is int and payload['epoch'] == 39 and payload['identity']['seed'] == seed,
                'A source epoch/seed')
        require(semantic_digest(payload['identity']) == payload['identity_sha256'] == bound['source_identity_sha256'], 'A source identity')
        scale = payload['identity']['scaler']['scaler']
        tasks = scale['task_names']
        counts = dict(zip(tasks, scale['counts']))
        require(len(tasks) == 104 and tasks == payload['identity']['task_names'], 'A source task count/order')
        require(sha(path) == bound['teacher_sha256'], 'A source changed during load')
        state = payload['model_state']
        binding = bound
    else:
        from dataset115_source import load_binding
        from s4e_mechanism_smoke import state_from_asset
        teacher, _ = load_binding(source_lock, seed)
        payload, state, _ = state_from_asset(repo, teacher)
        require(payload['configuration']['seed'] == seed and payload['epoch'] == 39, 'Animal source seed/epoch')
        tasks = teacher['task_names']
        counts = {t: payload['task_scalers'][t]['count'] for t in tasks}
        require(len(tasks) == 56, 'Animal56 count')
        binding = dict(seed=seed, teacher_sha256=teacher['sha256'], source_lock_sha256=sha(source_lock))
    require(len(counts) == len(tasks) and all(type(n) is int and n > 0 for n in counts.values()), 'source count metadata')
    if setting == 'ToxAcute':
        initial = {k[8:]: v for k, v in factory.initial.items() if k.startswith('encoder.')}
        require(initial and all(torch.equal(initial[k], state['encoder.'+k]) for k in initial), 'Tox source/init equality')
        probe = encoder(args, seed)
        probe.load_state_dict(initial, strict=True)
        from dataset115_smoke import state_digest
        expected = state_digest(probe.state_dict())
    else:
        expected = contract['initial_encoder']
    values = extract_encoder(state, args, expected)
    return values, tasks, counts, binding


class Data:
    def __init__(self, *, args, pretrained, target_datasets, target_rows, target_scalers,
                 source_tasks, source_counts, source, identity):
        self.args, self.pretrained = args, pretrained
        self.datasets, self.rows = target_datasets, target_rows
        self.target_tasks = list(target_datasets['train'])
        self.source_tasks, self.source_counts = list(source_tasks), source_counts
        self.target_counts = {t: len(ds) for t, ds in target_datasets['train'].items()}
        self.source = source
        self.scalers = dict(target=target_scalers, source={} if source is None else source.scalers)
        self.identity = deepcopy(identity)
        self.identity.update(target_train_sha256=digest(target_rows['train']),
                             target_validation_sha256=digest(target_rows['validation']),
                             target_scalers=target_scalers, target_counts=self.target_counts,
                             source_tasks=self.source_tasks, source_counts=source_counts,
                             source_training=None if source is None else source.identity,
                             architecture=vars(args))
        require(list(target_datasets['validation']) == self.target_tasks, 'target validation tasks/order')
        self.by_task = {split: {t: [r for r in rows if r['task'] == t] for t in self.target_tasks}
                        for split, rows in target_rows.items()}
        for split in ('train', 'validation'):
            for task, ds in target_datasets[split].items():
                require([r['sample_id'] for r in self.by_task[split][task]] ==
                        [str(ds.get_sample_id(i)) for i in range(len(ds))], 'target dataset identity/order')
        if source is not None:
            require(source.counts == source_counts and source.tasks == source_tasks, 'verified source clock counts')
        self.cache = OrderedDict()

    def model(self, job, device):
        return Control(self.args, self.pretrained, self.target_tasks, self.source_tasks,
                       job['condition'], job['seed']).to(device)

    def batch(self, role, split, task, indices, device):
        if role == 'source':
            require(split == 'train' and self.source is not None, 'source label access forbidden')
            return self.source.batch(task, indices, device)
        require(role == 'target' and split in ('train', 'validation'), 'train/validation target scope')
        from dataset import DataCollator
        graphs = []
        for index in indices:
            key = split, task, index
            if key not in self.cache:
                self.cache[key] = self.datasets[split][task][index]
            self.cache.move_to_end(key)
            graphs.append(self.cache[key])
            while len(self.cache) > 512:
                self.cache.popitem(last=False)
        batch = DataCollator()(graphs).to(device)
        expected = [self.by_task[split][task][i] for i in indices]
        require(not batch.is_empty and list(batch.sample_id) == [r['sample_id'] for r in expected]
                and list(batch.canonical_smiles) == [r['canonical'] for r in expected]
                and batch.y.numel() == len(expected), 'target graph population/chemistry')
        require(torch.equal(batch.y.reshape(-1).cpu(), torch.tensor([r['label'] for r in expected], dtype=batch.y.dtype)), 'target labels')
        return batch


def load(repo, setting, seed, condition, split_manifest, source_lock):
    from rdkit import rdBase
    from p1d4_identity import LOCK, contract_for
    from p1d4_runtime import factory_for
    from dataset115_adapter import GraphTaskView, TrainOnlyScaler
    from dataset115_contract import semantic_digest
    from dataset115_training import RouteBTrainer
    from architecture.prediction_heads import TaskPredictionHead
    from v9_s2_screen import observations
    from v9_s3_screen import check_populations
    import v9_joint_source
    repo = Path(repo)
    require(tuple(int(v) for v in rdBase.rdkitVersion.split('.')) == (2025, 9, 6), 'frozen RDKit 2025.09.6 required')
    require(sha(LOCK) == 'd3502bba88feeaffa884f28dd1fbaa866aed465b111210ec1afee49841bd5d61', 'reviewed five-seed lock')
    contract = contract_for(setting, seed)
    factory = factory_for(repo, setting=setting, seed=seed, split_manifest=split_manifest,
                          source_lock=source_lock, device='cpu')
    raw = contract['args'] if setting == 'ToxAcute' else contract['architecture']
    args = SimpleNamespace(**{k: raw[k] for k in ('hidden_dim', 'a_heads', 'a_layers', 'mid_dim',
                                                  'head_hidden_dim', 'head_dropout', 'edge_bias_mode', 'spatial_pos_clip')})
    pretrained, source_tasks, source_counts, binding = locked_source(factory, setting, seed, repo, source_lock, args, contract)
    if setting == 'ToxAcute':
        datasets = factory.datasets
        rows = {s: observations(factory, None, setting, s) for s in ('train', 'validation')}
        scalers = {}
        for t, ds in datasets['train'].items():
            y = torch.tensor([r['label'] for r in rows['train'] if r['task'] == t], dtype=torch.float32)
            scalers[t] = dict(mean=float(y.mean()), std=max(float(y.std(unbiased=False)), 1e-6), count=len(ds))
    else:
        views = {s: factory._table.view(setting, 'target', s) for s in ('train', 'validation')}
        datasets = {s: {t: GraphTaskView(v, t) for t in v.tasks} for s, v in views.items()}
        for s, v in views.items():
            require(semantic_digest(list(v.sample_ids)) == contract[s+'_ids'] and
                    RouteBTrainer._view_digest(v) == contract[s+'_observations'], 'target population contract')
        scaler = TrainOnlyScaler.fit(views['train'])
        require(scaler.to_dict() == contract['scaler'], 'target train scaler contract')
        scalers = scaler.trainer_scalers()
        rows = {s: observations(factory, SimpleNamespace(datasets=datasets), setting, s) for s in views}
    joint = condition in ('FJ', 'PJ', 'RJ')
    source = None
    if joint:
        # Legacy loader verifies full source tensors/scalers/population. Its
        # pretrained heads are used only for validation and are then discarded.
        base_encoder = encoder(args, seed)
        base_encoder.load_state_dict(pretrained, strict=True)
        base = SimpleNamespace(encoder=base_encoder, task_name=list(datasets['train']),
            decoders=nn.ModuleDict({t: TaskPredictionHead(args.hidden_dim, mode='quantile',
                head_hidden_dim=args.head_hidden_dim, dropout=args.head_dropout) for t in datasets['train']}))
        source = v9_joint_source.load(factory, SimpleNamespace(model=base), setting, repo, source_lock, seed=seed)
    populations = check_populations(setting, rows['train'], rows['validation'], rows['train'] if source is None else source.rows,
                                   excluded_canonical=factory._table.overlaps if setting == 'B' else None)
    populations['source_population_checked'] = source is not None
    return Data(args=args, pretrained=pretrained, target_datasets=datasets, target_rows=rows,
                target_scalers=scalers, source_tasks=source_tasks, source_counts=source_counts, source=source,
                identity=dict(setting=setting, seed=seed, condition=condition, contract_sha256=digest(contract),
                              source_binding=binding, populations=populations, source_label_datasets_constructed=joint,
                              historical_source_epochs=0 if condition in ('RT', 'RJ') else 40,
                              test_predictions_accessed=False, calibration_predictions_accessed=False))
