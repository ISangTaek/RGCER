"""Verified train-only source heads and data for the bounded V9-S3 comparison."""
from collections import OrderedDict
from copy import deepcopy
import math
import random

import torch

from v9_cost_probe import read, sha, digest, require
from reproducibility import state_dict_sha256


def restored_heads(base, tasks, state):
    """Copy the target head architecture, but require every pretrained tensor."""
    require(not set(tasks) & set(base.task_name), 'source/target task overlap')
    expected = {'decoders.'+t+'.'+k for t in tasks for k in next(iter(base.decoders.values())).state_dict()}
    require({k for k in state if k.startswith('decoders.')} == expected, 'source decoder keys/tasks')
    heads = torch.nn.ModuleDict()
    for task in tasks:
        head = deepcopy(next(iter(base.decoders.values()))).cpu()
        values = {k:state['decoders.'+task+'.'+k] for k in head.state_dict()}
        for k, v in head.state_dict().items():
            x = values[k]
            require(isinstance(x, torch.Tensor) and x.shape == v.shape and x.dtype == v.dtype
                    and torch.isfinite(x).all().item(), 'source decoder tensor '+task+'.'+k)
        head.load_state_dict(values, strict=True); heads[task] = head
    encoder = {k[8:]:v for k,v in state.items() if k.startswith('encoder.')}
    require(set(encoder) == set(base.encoder.state_dict()) and all(
        torch.equal(v.detach().cpu(), encoder[k].cpu()) for k,v in base.encoder.state_dict().items()),
        'auxiliary heads must come from identical source encoder')
    return heads


class Source:
    def __init__(self, datasets, rows, heads, scalers, provenance):
        self.datasets, self.rows, self.heads, self.scalers = datasets, rows, heads, scalers
        self.tasks = list(datasets); self.counts = {t:len(d) for t,d in datasets.items()}
        require(self.tasks == list(heads) and set(scalers) == set(self.tasks)
                and all(n > 0 for n in self.counts.values()), 'auxiliary task coverage')
        self.by_task = {t:[] for t in self.tasks}
        for row in rows:
            require(row['split'] == 'train' and row['task'] in self.by_task
                    and math.isfinite(row['label']), 'auxiliary train-only population')
            self.by_task[row['task']].append(row)
        require(len({(r['task'],r['sample_id']) for r in rows}) == len(rows), 'duplicate auxiliary observation')
        for task, ds in datasets.items():
            require([r['sample_id'] for r in self.by_task[task]] == [str(ds.get_sample_id(i)) for i in range(len(ds))],
                    'auxiliary data identity/order')
            s = scalers[task]
            require(type(s['count']) is int and s['count'] == len(ds) and
                    all(type(s[k]) in (int,float) and math.isfinite(s[k]) for k in ('mean','std'))
                    and s['std'] > 0, 'source train-only scaler/count')
        self.identity = dict(provenance=provenance, tasks=self.tasks, counts=self.counts, scalers=scalers,
                             heads_sha256=state_dict_sha256(heads), observations_sha256=digest(rows),
                             label_scale=provenance.get('label_scale','synthetic_fixture'), split='train')
        self.cache = OrderedDict()

    def batch(self, task, indices, device):
        from dataset import DataCollator
        graphs = []
        for i in indices:
            key = task,i
            if key not in self.cache: self.cache[key] = self.datasets[task][i]
            self.cache.move_to_end(key); graphs.append(self.cache[key])
            while len(self.cache) > 256: self.cache.popitem(last=False)
        batch = DataCollator()(graphs).to(device)
        expected = [self.by_task[task][i] for i in indices]
        require(not batch.is_empty and list(batch.sample_id) == [r['sample_id'] for r in expected]
                and list(batch.canonical_smiles) == [r['canonical'] for r in expected]
                and batch.y.numel() == len(expected), 'auxiliary graph identity/count')
        require(torch.equal(batch.y.reshape(-1).cpu(), torch.tensor([r['label'] for r in expected], dtype=batch.y.dtype)),
                'auxiliary graph labels')
        return batch


def load(factory, trainer, setting, repo, source_lock):
    base = trainer.model
    if setting == 'A':
        from dataset115_adapter import GraphTaskView, TrainOnlyScaler
        from dataset115_contract import semantic_digest
        path = factory._output/'epoch_039.pt'
        binding = factory.expected_identity['source_identity']
        require(sha(path) == binding['teacher_sha256'], 'Route A source checkpoint SHA')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(type(payload['epoch']) is int and payload['epoch'] == 39 and semantic_digest(payload['identity']) == payload['identity_sha256']
                == binding['source_identity_sha256'], 'Route A source identity')
        view = factory._table.view('A','source','train')
        require(len(view.tasks) == 104 and list(view.tasks) == payload['identity']['task_names'], 'Nonhuman104 scope')
        scaler = TrainOnlyScaler.fit(view)
        require(scaler.to_dict() == payload['identity']['scaler'], 'Route A train scaler identity')
        scalers = scaler.trainer_scalers()
        datasets = {t:GraphTaskView(view,t) for t in view.tasks}
        rows = [dict(task=t,sample_id=str(ds.get_sample_id(i)),split='train',canonical=view.canonical[j],
                     group=view.groups[j],label=float(view.labels[j,view.tasks.index(t)]))
                for t,ds in datasets.items() for i,j in enumerate(ds.indices)]
        state = payload['model_state']
        provenance = dict(kind='Nonhuman104',checkpoint_sha256=sha(path),binding=binding,
                          input_identity=view.input_identity,source_epoch=39,label_scale=scaler.to_dict()['label_scale'])
        require(provenance['checkpoint_sha256'] == binding['teacher_sha256'], 'Route A source changed during load')
    else:
        from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
        from dataset115_source import load_binding
        from s4e_mechanism_smoke import state_from_asset
        from p1d4_identity import LOCK
        from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset
        teacher,_ = load_binding(source_lock,42)
        payload,state,_ = state_from_asset(repo,teacher)
        store = factory.store if setting == 'ToxAcute' else ToxAcuteDataStore.resolve(repo/read(LOCK)['inputs']['datastore_relative'])
        require(all(payload['data_config'][k] == teacher['data_config'][k] == store.metadata[k]
                    for k in ('datastore_fingerprint','split_manifest_hash','feature_schema_version')), 'Animal56 datastore identity')
        require(payload['architecture_config'] == teacher['architecture_config'], 'Animal56 architecture identity')
        require(teacher['data_config']['max_nodes_filter'] == 512, 'Animal56 original max_nodes filter')
        datasets = {t:ToxAcuteTaskDataset(store,t,split='train',max_nodes=512) for t in ANIMAL_SOURCE_TASKS}
        manifest = {r['sample_id']:r for r in read(store.root/'split_manifest.json')['records']}
        rows = []
        for t,ds in datasets.items():
            for i,j in enumerate(ds.indices):
                sid = ds.get_sample_id(i); meta = manifest[sid]
                require(meta['split'] == 'train', 'Animal56 train membership')
                rows.append(dict(task=t,sample_id=sid,split='train',canonical=meta['canonical_smiles'],
                                 group=meta['split_group'],label=float(store.get_label(int(j),t))))
        scalers = deepcopy(payload['task_scalers']); locked = {s['task']:s for s in teacher['scalers']}
        for t,ds in datasets.items():
            require(scalers[t]['mean'] == locked[t]['mean'] and scalers[t]['std'] == locked[t]['std'], 'Animal56 locked scaler')
            values = torch.tensor([store.get_label(int(j),t) for j in ds.indices],dtype=torch.float32)
            require(math.isclose(scalers[t]['mean'],float(values.mean()),rel_tol=2e-6,abs_tol=2e-6)
                    and math.isclose(scalers[t]['std'],max(float(values.std(unbiased=False)),1e-6),rel_tol=2e-6,abs_tol=2e-6),
                    'Animal56 scaler must fit source train only')
        provenance = dict(kind='Animal56',checkpoint_sha256=teacher['sha256'],source_lock_sha256=sha(source_lock),
                          data_config=teacher['data_config'],source_epoch=39,label_scale='ToxAcute_frozen_native_labels_no_transform')
    return Source(datasets,rows,restored_heads(base,list(datasets),state),scalers,provenance)


class Schedule:
    """Balanced shuffled task cycles; per-task shuffled passes retain short tails."""
    def __init__(self, tasks, counts, batch_size):
        self.tasks, self.counts, self.batch_size = list(tasks), counts, batch_size
        require(len(set(tasks)) == len(tasks) and set(tasks) == set(counts) and
                all(type(n) is int and n > 0 for n in counts.values()) and batch_size > 0, 'auxiliary schedule inputs')
        self.rng = random.Random(42); self.order = []
        self.rngs = {t:random.Random(420001+i) for i,t in enumerate(tasks)}
        self.remaining = {t:[] for t in tasks}; self.step = 0

    def next(self):
        if not self.order:
            self.order = self.tasks.copy(); self.rng.shuffle(self.order)
        task = self.order.pop(0)
        if not self.remaining[task]:
            self.remaining[task] = list(range(self.counts[task])); self.rngs[task].shuffle(self.remaining[task])
        indices = self.remaining[task][:self.batch_size]; del self.remaining[task][:self.batch_size]
        row = dict(step=self.step,task=task,indices=indices); self.step += 1
        return row


def coverage(source, schedule):
    result = {}
    for task in source.tasks:
        batches = [r for r in schedule if r['task'] == task]
        indices = [i for r in batches for i in r['indices']]
        unique = {source.by_task[task][i]['canonical'] for i in indices}
        population = {r['canonical'] for r in source.by_task[task]}
        result[task] = dict(updates=len(batches),observations=len(indices),unique_observations=len(set(indices)),
                            unique_molecules=len(unique),population_molecules=len(population),
                            molecule_coverage=len(unique)/len(population))
    return result
