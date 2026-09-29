"""V9 candidate A and a matched ReFine-style graph adaptation.

ReFine (arXiv:2505.11771) motivates frozen features plus an independent raw
input encoder. This is a graph/multi-endpoint adaptation, not author code or
an assertion that its original statistical theorem applies to this model.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from architecture.graphormer_backbone import MolecularGraphormerBackbone
from v9_cost_probe import digest, read, require

METHODS = ('REFINE_GRAPH', 'SRGT')
SPEC = dict(schema='v9_srgt_v1', target_width=48, target_layers=2,
    target_heads=4, target_ffn=64, dropout=.1, fusion='identity_source_zero_target_linear',
    fingerprint_radius=2, fingerprint_bits=2048, chirality=True,
    support='train_union_max_tanimoto_exclude_same_canonical_and_group',
    gates='independent_2sigmoid_global_task_support', gate_penalty=.01,
    adapter_lr=.0001, head_lr=.001, weight_decay=.00001, grad_clip=1.)


def training_records(factory, trainer, setting):
    """Metadata only. No source/validation/test row enters the reference pool."""
    if setting == 'ToxAcute':
        from toxacute_datastore import SPLIT_CODES
        manifest = read(factory.store.root / 'split_manifest.json')
        # DataStore V2 row_indices are dense graph positions. Manifest
        # row_index preserves raw CSV positions, including gaps after invalid
        # molecules. Their common identity is sample_id, never a row number.
        by_id = {}
        for record in manifest['records']:
            sid = record.get('sample_id')
            require(type(sid) is str and bool(sid) and sid not in by_id, 'manifest sample IDs')
            by_id[sid] = record
        store_ids = [str(sid) for sid in factory.store.sample_ids]
        require(len(store_ids) == len(set(store_ids)) and set(store_ids) == set(by_id), 'manifest/store sample ID bijection')
        require(len(factory.store.row_indices) == len(factory.store.split_codes)
                == len(store_ids), 'store metadata lengths')
        for index, sid in enumerate(store_ids):
            record = by_id[sid]
            require(int(factory.store.row_indices[index]) == index, 'dense DataStore graph index')
            require(record['split'] in SPLIT_CODES and SPLIT_CODES[record['split']]
                    == int(factory.store.split_codes[index]), 'manifest/store split identity')
        datasets = factory.datasets['train']
    else:
        require(setting in ('A', 'B') and trainer.train.split == 'train', 'target train required')
        datasets = trainer.datasets['train']
    records = []
    for task, ds in datasets.items():
        require(len(ds.indices) == len(set(int(i) for i in ds.indices)), 'duplicate task indices')
        for i, global_index in enumerate(ds.indices):
            sid = str(ds.get_sample_id(i))
            if setting == 'ToxAcute':
                require(ds.split == 'train', 'Tox reference split')
                require(0 <= int(global_index) < len(store_ids), 'Tox graph index bounds')
                require(store_ids[int(global_index)] == sid and sid in by_id, 'dataset/store sample identity')
                record = by_id[sid]
                require(record['sample_id'] == sid, 'manifest sample ID')
                require(record['split'] == 'train', 'manifest reference split')
                canonical, group = record['canonical_smiles'], record['split_group']
            else:
                require(ds.view.split == 'train' and ds.view.role == 'target', 'reference role/split')
                canonical, group = ds.view.canonical[global_index], ds.view.groups[global_index]
            records.append(dict(task=task, sample_id=sid, canonical=canonical, group=group, split='train'))
    return records


class TrainSupport:
    """Fixed training-only support; explicit exclusion of the whole own group."""
    def __init__(self, records):
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        require(type(records) is list and bool(records), 'training records required')
        self.records = sorted(deepcopy(records), key=lambda r: (r['task'], r['sample_id']))
        self.by_key, unique, id_identity = {}, {}, {}
        for row in self.records:
            require(type(row) is dict and set(row) == {'task','sample_id','canonical','group','split'}, 'reference fields')
            require(all(type(v) is str and v for v in row.values()) and row['split'] == 'train', 'train-only references')
            key = (row['task'], row['sample_id'])
            require(key not in self.by_key, 'duplicate task observation')
            require(row['canonical'] not in unique or unique[row['canonical']] == row['group'], 'canonical crosses group')
            pair = (row['canonical'], row['group'])
            require(row['sample_id'] not in id_identity or id_identity[row['sample_id']] == pair, 'sample identity drift')
            unique[row['canonical']] = row['group']; id_identity[row['sample_id']] = pair
            self.by_key[key] = row
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        fingerprints = {}
        for canonical in sorted(unique):
            mol = Chem.MolFromSmiles(canonical)
            require(mol is not None, 'invalid reference molecule')
            # The frozen data pipeline owns canonical identity. RDKit parsing
            # does not invent a new split or silently replace that identity.
            fingerprints[canonical] = generator.GetFingerprint(mol)
        values = {}
        for canonical, fp in fingerprints.items():
            others = [v for c,v in fingerprints.items() if c != canonical and unique[c] != unique[canonical]]
            values[canonical] = (max(DataStructs.BulkTanimotoSimilarity(fp, others)) if others else 0., float(bool(others)))
        self.values = values
        self.identity = dict(schema='v9_support_v1', records_sha256=digest(self.records),
            policy=SPEC['support'], radius=2, bits=2048, chirality=True,
            molecules=len(unique), groups=len(set(unique.values())),
            values_sha256=digest({k:list(v) for k,v in sorted(values.items())}))

    def for_train_batch(self, task, sample_ids, canonical):
        require(len(sample_ids) == len(canonical) and bool(sample_ids), 'support batch population')
        rows = []
        for sid, molecule in zip(sample_ids, canonical):
            key = (task, str(sid))
            require(key in self.by_key and self.by_key[key]['canonical'] == molecule, 'support sample/chemical identity')
            rows.append(self.values[molecule])
        return torch.tensor(rows, dtype=torch.float32)


class DualGraph(nn.Module):
    def __init__(self, base, method, counts, seed=42):
        super().__init__()
        require(method in METHODS and type(seed) is int and seed == 42, 'C1 method/seed')
        require(getattr(base, 'card', None) is None, 'unexpected existing adapter')
        self.method = method
        self.task_name = tuple(base.task_name)
        require(set(self.task_name) == set(counts) and all(type(n) is int and n > 0 for n in counts.values()), 'task counts')
        require(isinstance(base.encoder.readout, nn.Identity), 'parameterized source readout not supported')
        self.encoder = deepcopy(base.encoder)
        self.decoders = deepcopy(base.decoders)
        self.encoder.requires_grad_(False)
        hidden = base.encoder.backbone.hidden_dim
        self.hidden = hidden
        self.card = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.target = MolecularGraphormerBackbone(hidden_dim=48, num_heads=4, num_layers=2,
                ffn_dim=64, dropout=.1, spatial_pos_max_clip=base.encoder.backbone.spatial_pos_max_clip,
                edge_bias_mode=base.encoder.backbone.edge_bias_mode)
            self.fusion = nn.Linear(hidden+48, hidden, bias=False)
            with torch.no_grad():
                self.fusion.weight.zero_()
                self.fusion.weight[:, :hidden].copy_(torch.eye(hidden))
        if method == 'SRGT':
            self.global_logits = nn.Parameter(torch.zeros(2))
            self.task_logits = nn.Parameter(torch.zeros(len(self.task_name), 2))
            self.support_weights = nn.Parameter(torch.zeros(2, 2))
            # Less observed endpoints receive stronger shrinkage toward the
            # common gate. Weights have mean 1 and are fixed before training.
            weights = torch.tensor([1./counts[t] for t in self.task_name])
            self.register_buffer('shrink_weights', weights/weights.mean())
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def gates(self, task, support, reference):
        if self.method == 'REFINE_GRAPH':
            return reference.new_ones((reference.size(0), 2))
        require(isinstance(support, torch.Tensor) and tuple(support.shape) == (reference.size(0),2), 'support tensor shape')
        require(support.device == reference.device and torch.isfinite(support).all().item(), 'support tensor device/finite')
        require(((support[:,0]>=0)&(support[:,0]<=1)&((support[:,1]==0)|(support[:,1]==1))).all().item(), 'support bounds')
        values = torch.stack(((support[:,0]-.5)*support[:,1], support[:,1]), dim=1)
        logits = self.global_logits + self.task_logits[self.task_name.index(task)] + values @ self.support_weights.T
        return 2*torch.sigmoid(logits)

    def regularization(self):
        if self.method == 'REFINE_GRAPH':
            return self.fusion.weight.new_zeros(())
        return SPEC['gate_penalty']*(self.shrink_weights[:,None]*self.task_logits.square()).mean()

    def forward(self, inputs, task_name=None, return_aux=False):
        require(task_name in self.task_name, 'explicit known task required')
        require(not self.encoder.training and all(not p.requires_grad for p in self.encoder.parameters()), 'source must remain frozen/eval')
        with torch.no_grad():
            source = self.encoder(inputs)
        target = self.target(inputs)
        gates = self.gates(task_name, getattr(inputs, 'v9_support', None), source)
        features = torch.cat((gates[:,0:1]*source, gates[:,1:2]*target), dim=1)
        representation = self.fusion(features)
        raw = self.decoders[task_name](representation)
        output = {task_name: raw}
        if not return_aux:
            return output
        return output, dict(final_raw=raw, base_raw=raw, route_raw=raw, final_representation=representation,
                            source_representation=source, target_representation=target, gates=gates)


def optimizer_for(model):
    adapted = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('decoders.')]
    return torch.optim.AdamW([dict(params=adapted, lr=SPEC['adapter_lr'], scope='adaptation'),
                             dict(params=list(model.decoders.parameters()), lr=SPEC['head_lr'], scope='heads')],
                            weight_decay=SPEC['weight_decay'])


def save_smoke(path, model, optimizer, identity, updates):
    require(type(updates) is int and 1<=updates<=12, 'smoke update budget')
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    payload=dict(schema='v9_c1_smoke_state_v1', identity=identity, updates=updates,
                 model_state=state, optimizer_state=optimizer.state_dict())
    with Path(path).open('xb') as f:
        torch.save(payload,f)


def load_smoke(path, model, optimizer, identity, updates):
    payload=torch.load(path, map_location='cpu', weights_only=True)
    require(type(payload) is dict and set(payload)=={'schema','identity','updates','model_state','optimizer_state'}, 'checkpoint schema')
    require(payload['schema']=='v9_c1_smoke_state_v1' and digest(payload['identity'])==digest(identity), 'checkpoint identity')
    require(type(payload['updates']) is int and payload['updates']==updates, 'checkpoint update count')
    expected=model.state_dict(); actual=payload['model_state']
    require(set(actual)==set(expected), 'checkpoint keys')
    for key, value in expected.items():
        item=actual[key]
        require(isinstance(item,torch.Tensor) and item.dtype==value.dtype and item.shape==value.shape
                and torch.isfinite(item).all().item(), 'checkpoint tensor '+key)
        if key.startswith('encoder.') or key=='shrink_weights':
            require(torch.equal(item.cpu(),value.detach().cpu()), 'protected tensor '+key)
    opt=payload['optimizer_state']; expected_opt=optimizer.state_dict()
    require(type(opt) is dict and set(opt)=={'state','param_groups'} and opt['param_groups']==expected_opt['param_groups'], 'optimizer groups')
    parameters=[p for g in optimizer.param_groups for p in g['params']]
    task_updates=identity['task_updates']
    require(set(task_updates)==set(model.task_name) and all(type(n) is int and n>=0 for n in task_updates.values())
            and sum(task_updates.values())==updates, 'task update identity')
    names={id(p):n for n,p in model.named_parameters()}
    inactive={'path':'target.direct_bond_embeddings.', 'direct':'target.path_bond_embeddings.',
              'direct_plus_path':None}[model.target.edge_bias_mode]
    expected_steps={}
    for index,p in enumerate(parameters):
        name=names[id(p)]
        steps=task_updates[name.split('.')[1]] if name.startswith('decoders.') else updates
        if inactive is not None and name.startswith(inactive): steps=0
        if steps: expected_steps[index]=steps
    require(set(opt['state'])==set(expected_steps), 'optimizer IDs/states incomplete')
    for index,item in opt['state'].items():
        require(type(index) is int and set(item)=={'step','exp_avg','exp_avg_sq'}, 'Adam fields')
        require(all(isinstance(v,torch.Tensor) and torch.isfinite(v).all().item() for v in item.values()), 'Adam finite')
        require(item['step'].numel()==1 and item['step'].dtype==torch.float32
                and float(item['step'])==expected_steps[index], 'Adam steps')
        require(all(item[k].shape==parameters[index].shape and item[k].dtype==parameters[index].dtype for k in ('exp_avg','exp_avg_sq')), 'Adam shape/dtype')
    model.load_state_dict(actual,strict=True)
    optimizer.load_state_dict(opt)
    model.encoder.eval()
