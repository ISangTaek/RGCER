"""Train/validation-only frozen features, task metadata and grouped episodes."""
from dataclasses import replace
import random

import torch

import v9_s3_screen as s3
from v10_functional_transfer import Episode, RowIdentity, FrozenFunctionBank, require

read, write, sha, digest = s3.read, s3.write, s3.sha, s3.digest


def task_metadata(tasks, source_tasks, setting):
    fields = {}
    for task in tasks:
        parts = task.rsplit('_', 2)
        require(len(parts) == 3 and all(parts), 'task name must encode population/route/endpoint')
        domain = 'ToxAcute' if setting == 'ToxAcute' or (setting == 'B' and task in source_tasks) else 'PubChem'
        fields[task] = dict(zip(('population', 'route', 'endpoint', 'domain'), (*parts, domain)))
    vocabulary = {k:sorted({v[k] for v in fields.values()}) for k in next(iter(fields.values()))}
    vectors = {t:[float(f[k] == value) for k, values in vocabulary.items() for value in values] for t,f in fields.items()}
    return dict(fields=fields, vocabulary=vocabulary, vectors=vectors)


def extract(factory, trainer, source, setting, device, *, source_indices=None):
    """No test data, no stochastic augmentations, no fitted graph parameters."""
    trainer.model.eval()
    bank = FrozenFunctionBank(trainer.model.encoder, source.heads).to(device)
    training = s3.s2.observations(factory, trainer, setting, 'train')
    validation = s3.s2.observations(factory, trainer, setting, 'validation')
    population = s3.population_check(factory, source, setting, training, validation)
    scalers = trainer.task_scalers if setting == 'ToxAcute' else trainer.scalers
    target_tasks = list(scalers)
    rows = []; hs = []; ps = []
    def collect(meta, get_batch):
        size = s3.s2.c0.BATCH_SIZES[setting]
        for start in range(0, len(meta), size):
            chunk = meta[start:start+size]; b = get_batch(start, len(chunk))
            require(list(b.sample_id) == [r['sample_id'] for r in chunk]
                    and list(b.canonical_smiles) == [r['canonical'] for r in chunk], 'feature graph identity')
            h, p = bank(b); hs.append(h.cpu()); ps.append(p.cpu()); rows.extend(chunk)
    for split, observations in (('train', training), ('validation', validation)):
        for task in target_tasks:
            meta = [r for r in observations if r['task'] == task]
            collect(meta, lambda start, n: s3.s2.batch_for(factory, trainer, setting, split, task, list(range(start,start+n)), device))
    for task in source.tasks:
        ids = list(range(source.counts[task])) if source_indices is None else source_indices[task]
        meta = [source.by_task[task][i] for i in ids]
        collect(meta, lambda start, n: source.batch(task, ids[start:start+n], device))
    raw_h, functions = torch.cat(hs), torch.cat(ps)
    identity = dict(setting=setting, initial_encoder=s3.state_dict_sha256(trainer.model.encoder), source=source.identity,
                    contract_sha256=digest(s3.s2.c0.contract(setting)), train_sha256=digest(training),
                    validation_sha256=digest(validation), source_train_sha256=digest(source.rows), population=population)
    return dict(identity=identity, rows=rows, h=raw_h, functions=functions, scalers=dict(scalers, **source.scalers),
                targets=target_tasks, sources=source.tasks, metadata=task_metadata(target_tasks+source.tasks, source.tasks, setting))


def pack(raw, commit):
    rows = raw['rows']; indices = [i for i,r in enumerate(rows) if r['split'] == 'train']
    h = raw['h']; mean = h[indices].double().mean(0).float()
    scale = h[indices].double().std(0, unbiased=False).float().clamp_min(1e-6)
    return dict(raw, mean=mean, scale=scale, commit=commit, schema='v10_frozen_features_v1',
                statistics_population_sha256=digest([rows[i] for i in indices]))


def save_cache(raw, folder, commit):
    folder.mkdir(parents=True, exist_ok=False); value = pack(raw, commit)
    with (folder/'cache.pt').open('xb') as stream: torch.save(value, stream)
    write(folder/'manifest.json', dict(commit=commit, sha256=sha(folder/'cache.pt'), identity=value['identity'],
                                     rows_sha256=digest(value['rows']), source_tasks=value['sources'], target_tasks=value['targets']))
    return value


def load_cache(folder, commit, identity=None):
    m = read(folder/'manifest.json')
    require(m['commit'] == commit and sha(folder/'cache.pt') == m['sha256'], 'feature cache SHA/commit')
    value = torch.load(folder/'cache.pt', map_location='cpu', weights_only=True)
    require(value['commit'] == commit and value['schema'] == 'v10_frozen_features_v1'
            and value['identity'] == m['identity'] and digest(value['rows']) == m['rows_sha256']
            and value['sources'] == m['source_tasks'] and value['targets'] == m['target_tasks'], 'cache manifest identity')
    if identity is not None: require(identity == value['identity'], 'trusted cache identity differs')
    return value


class Data:
    def __init__(self, value, device='cpu'):
        self.value, self.device = value, device
        self.rows = value['rows']; self.sources, self.targets = value['sources'], value['targets']
        require(set(self.sources).isdisjoint(self.targets), 'source/target task overlap')
        self.tasks = self.targets+self.sources
        require(len(set(self.tasks)) == len(self.tasks), 'duplicate tasks')
        self.by = {(t,s):[] for t in self.tasks for s in ('train','validation')}
        seen = set()
        for i,r in enumerate(self.rows):
            require(set(r) == {'task','sample_id','canonical','group','split','label'}
                    and r['split'] in ('train','validation') and r['task'] in self.tasks, 'cache row scope')
            key = (r['task'],r['sample_id']); require(key not in seen, 'cache duplicate observation'); seen.add(key)
            require(r['split'] == 'train' or r['task'] in self.targets, 'source validation prohibited')
            self.by[(r['task'],r['split'])].append(i)
        n = len(self.rows)
        require(value['h'].ndim == 2 and value['h'].shape[0] == n
                and value['functions'].shape == (n,len(self.sources)), 'cache tensor dimensions')
        for key in ('h','functions','mean','scale'):
            require(value[key].dtype == torch.float32 and bool(torch.isfinite(value[key]).all()), 'cache finite float tensors')
        require(value['mean'].shape == value['scale'].shape == (value['h'].shape[1],)
                and bool((value['scale'] > 0).all()), 'cache standardization')
        self.h = ((value['h']-value['mean'])/value['scale']).to(device)
        self.p = value['functions'].to(device)
        self.ids = [RowIdentity(**{k:r[k] for k in ('sample_id','canonical','group','split','task')}) for r in self.rows]
        for t in self.tasks:
            ids = self.by[t,'train']; s = value['scalers'][t]
            labels = torch.tensor([self.rows[i]['label'] for i in ids],dtype=torch.float64)
            require(ids and abs(s['mean']-float(labels.mean())) < 2e-5 and
                    abs(s['std']-max(float(labels.std(unbiased=False)),1e-6)) < 2e-5, 'scaler not train fitted')
        self.y = torch.tensor([[(r['label']-value['scalers'][r['task']]['mean'])/value['scalers'][r['task']]['std']]
                               for r in self.rows],dtype=torch.float32,device=device)
        require(bool(torch.isfinite(self.y).all()), 'finite scaled labels')

    def episode(self, task, query, *, context=None, cap=128, seed=42, no_functions=False, stochastic_neighbors=False):
        require(query and len(set(query)) == len(query)
                and all(self.rows[i]['task'] == task for i in query), 'episode query indices')
        if context is None:
            blocked = {k:{self.rows[i][k] for i in query} for k in ('sample_id','canonical','group')}
            context = [i for i in self.by[task,'train'] if all(self.rows[i][k] not in v for k,v in blocked.items())]
            random.Random(seed).shuffle(context)
            if stochastic_neighbors: cap = min(cap or len(context), max(2, math_ceil_half(len(context))))
            if cap is not None: context = context[:cap]
        require(len(context) >= 2, 'insufficient disjoint train context: '+task)
        metadata = torch.tensor([self.value['metadata']['vectors'][task]], dtype=self.h.dtype, device=self.device)
        pc, pq = self.p[context], self.p[query]
        if no_functions: pc, pq = torch.zeros_like(pc), torch.zeros_like(pq)
        e = Episode(task, [self.ids[i] for i in context], [self.ids[i] for i in query],
                    self.h[context], pc, self.y[context], self.h[query], pq, metadata)
        e.validate(self.h.shape[1], self.p.shape[1], metadata.shape[1]); return e


def math_ceil_half(n): return (n+1)//2


def query_batches(data, task, epoch, batch_size=16, limit=None):
    """Full target pass; split a batch further if it leaves too little context."""
    indices = data.by[task,'train'].copy(); random.Random(42000+epoch).shuffle(indices)
    if limit is not None:
        # Rotating shuffled pass avoids repeatedly selecting one easy source subset.
        fixed = data.by[task,'train'].copy(); random.Random(42100).shuffle(fixed)
        start = epoch*limit % len(fixed)
        indices = (fixed+fixed)[start:start+min(limit,len(fixed))]
    def safe(items):
        try:
            data.episode(task, items)
            return [items]
        except ValueError as exc:
            if 'insufficient disjoint' not in str(exc) or len(items) == 1: raise
            middle = len(items)//2
            return safe(items[:middle])+safe(items[middle:])
    return [q for i in range(0,len(indices),batch_size) for q in safe(indices[i:i+batch_size])]


def epoch_schedule(data, epoch, *, target_only=False, batch_size=16):
    target = [(t,q) for t in data.targets for q in query_batches(data,t,epoch,batch_size)]
    source = [] if target_only else [(t,q) for t in data.sources for q in query_batches(data,t,epoch,batch_size,limit=batch_size)]
    rng = random.Random(420000+epoch); rng.shuffle(target); rng.shuffle(source)
    result = [dict(target=(t,q), source=[]) for t,q in target]
    for i,item in enumerate(source): result[i % len(result)]['source'].append(item)
    return result
