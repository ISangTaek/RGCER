"""PGT point models: shared molecular inputs, explicit task order and readouts."""
from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools
import torch
from torch import nn

from architecture.molecule_adaptive_prompt import FactorizedTaskPromptBank, MoleculeAdaptiveTaskConditioner
from architecture.prediction_heads import TaskPredictionHead
from architecture.toxacute_tasks import parse_toxacute_task_name
from baselines.constants import CHEMPROP_COMMIT
from baselines.models.dmpnn import activate_chemprop, dmpnn_args
from baselines.models.toxacol import ToxACoLNet, endpoint_feature_matrix
from .data import make_backbone


def molecular_features(mols, kind):
    """Never parse display SMILES. Every caller supplies registered RDKit Mols."""
    if not mols or any(not isinstance(m, Chem.Mol) for m in mols):
        raise TypeError("Expected nonempty registered Mol list, not SMILES")
    width = {'morgan': 2048, 'avalon': 1024}[kind]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    result = np.zeros((len(mols), width), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = generator.GetFingerprint(mol) if kind == 'morgan' else pyAvalonTools.GetAvalonFP(mol, nBits=1024)
        DataStructs.ConvertToNumpyArray(fp, result[i])
    return result


def task_metadata(names):
    registered = json.loads((Path(__file__).resolve().parents[1] / 'configs/pgt_v2/task_metadata.json').read_text(encoding='utf-8'))
    output = {}
    for name in names:
        if name in registered:
            output[name] = registered[name]
            continue
        m = parse_toxacute_task_name(name)
        # Preserve C01's unknown populations; Human3 are populations of one organism.
        subject = name.rsplit('_', 2)[0]
        population = {'man': 'male', 'women': 'female', 'human': 'unspecified'}.get(subject, 'not_population_stratified')
        output[name] = dict(organism=m.organism, route=m.route, measurement=m.measurement, population=population)
    return output


class AtomReadout(nn.Module):
    def __init__(self, tasks, dim=96, heads=4, shared=False, metadata=False, additive=False):
        super().__init__()
        if dim % heads:
            raise ValueError('Readout dimension must divide into heads')
        self.tasks, self.dim, self.heads = tuple(tasks), dim, heads
        self.shared, self.additive = shared, additive
        self.prompts = None
        if metadata:
            self.prompts = FactorizedTaskPromptBank(tasks, dim, use_factorized_prompt=True,
                task_residual_scale=1.0, metadata_overrides=task_metadata(tasks))
            self.register_parameter('query', None)
        else:
            self.query = nn.Parameter(torch.randn(1 if shared else len(tasks), dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim, bias=not additive)
        if additive:
            self.u0 = nn.Parameter(torch.zeros(heads, dim // heads))
            self.u_proj = nn.Linear(dim, dim, bias=False)
            self.bias = nn.Parameter(torch.zeros(len(tasks)))
        else:
            self.out_proj = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, atoms, mask, *, query_override=None):
        if mask.dtype != torch.bool or not bool(mask.any(dim=1).all()):
            raise ValueError('Every graph must contain a valid atom')
        prompts = self.prompts() if self.prompts is not None else self.query
        if query_override is not None:
            if query_override.shape != prompts.shape:
                raise ValueError('Query replacement shape mismatch')
            prompts = query_override
        b, n, _ = atoms.shape
        d = self.dim // self.heads
        q = self.q_proj(prompts).reshape(len(prompts), self.heads, d)
        k = self.k_proj(atoms).reshape(b, n, self.heads, d)
        v = self.v_proj(atoms).reshape(b, n, self.heads, d)
        logits = torch.einsum('thd,bnhd->bthn', q, k) / d**0.5
        attention = logits.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
        if self.shared:
            # Evaluate once and broadcast, ensuring literally identical task
            # attention, including under different GEMM batching kernels.
            attention = attention.expand(-1, len(self.tasks), -1, -1)
        if self.additive:
            u = self.u0[None] + self.u_proj(prompts).reshape(len(self.tasks), self.heads, d)
            values = torch.einsum('thd,bnhd->bthn', u, v)
            contributions = (attention * values).sum(2)
            prediction = self.bias + contributions.sum(-1)
            return prediction, dict(attention=attention, values=values, contributions=contributions, bias=self.bias)
        representation = torch.einsum('bthn,bnhd->bthd', attention, v).reshape(b, len(self.tasks), self.dim)
        return self.norm(self.out_proj(representation)), dict(attention=attention)


class GraphPointModel(nn.Module):
    def __init__(self, method, tasks, config):
        super().__init__()
        self.method, self.tasks = method, tuple(tasks)
        d = config['hidden_dim']
        self.backbone = make_backbone(hidden_dim=d, ffn_dim=config['ffn_dim'], num_heads=config['heads'],
            num_layers=config['layers'], dropout=config['dropout'])
        if method in ('M5', 'M6', 'A1', 'A2'):
            self.readout = AtomReadout(tasks, d, config['heads'], shared=method == 'M5',
                metadata=method == 'A1', additive=method == 'A2')
        elif method == 'M4':
            self.conditioner = MoleculeAdaptiveTaskConditioner(tasks, d, use_factorized_prompt=True,
                task_residual_scale=0.1, prompt_layers=0, prompt_heads=config['heads'], prompt_ffn_dim=2*d,
                prompt_dropout=config['dropout'], router_mode='dynamic', router_top_k=8,
                router_temperature=1.0, exclude_target_from_sources=True, adapter_ratio=0.25, gate_init=-2.0)
        elif method == 'A3':
            self.linear = nn.Linear(d, len(tasks))
        elif method != 'M2':
            raise ValueError(method)
        if method not in ('A2', 'A3'):
            self.heads = nn.ModuleList([TaskPredictionHead(d, mode='point', head_hidden_dim=d,
                                                         dropout=config['dropout']) for _ in tasks])

    def forward(self, batch, *, return_aux=False, atom_embeddings=None, query_override=None):
        tokens = self.backbone.encode_tokens(batch, atom_embeddings=atom_embeddings)
        aux = dict(atom_mask=tokens.atom_mask)
        if self.method in ('M5', 'M6', 'A1', 'A2'):
            representation, extra = self.readout(tokens.atom_tokens, tokens.atom_mask, query_override=query_override)
            aux.update(extra)
            if self.method == 'A2':
                return (representation, aux) if return_aux else representation
        elif self.method == 'A3':
            attention = tokens.atom_mask / tokens.atom_mask.sum(-1, keepdim=True)
            values = torch.einsum('bnd,td->btn', tokens.atom_tokens, self.linear.weight)
            contribution = values * attention[:, None, :]
            prediction = self.linear.bias + contribution.sum(-1)
            aux.update(attention=attention[:, None, None, :].expand(-1, len(self.tasks), 1, -1),
                       values=values[:, :, None, :], contributions=contribution, bias=self.linear.bias)
            return (prediction, aux) if return_aux else prediction
        elif self.method == 'M4':
            reps, diagnostics = self.conditioner.condition_all(tokens.graph_token, return_aux=True)
            representation = torch.stack([reps[t] for t in self.tasks], dim=1)
            aux['routing'] = diagnostics
        else:
            representation = tokens.graph_token[:, None, :].expand(-1, len(self.tasks), -1)
        prediction = torch.cat([head(representation[:, i]) for i, head in enumerate(self.heads)], dim=-1)
        return (prediction, aux) if return_aux else prediction


class DMPNN(nn.Module):
    def __init__(self, tasks, config, source_root, device):
        super().__init__()
        if source_root is None:
            raise ValueError('M1 requires the locked Chemprop source root')
        activate_chemprop(source_root)
        module = importlib.import_module('chemprop')
        root = Path(source_root).resolve()
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('Loaded Chemprop is not from the locked source root')
        changed = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if changed.strip():
            raise RuntimeError('Tracked Chemprop sources are modified')
        from chemprop.features import get_atom_fdim, get_bond_fdim
        from chemprop.models.ffn import build_ffn
        from chemprop.models.mpn import MPNEncoder
        from chemprop.nn_utils import initialize_weights
        args = dmpnn_args(torch.device(device), config['dropout'])
        args.depth = config['depth']
        self.encoder = MPNEncoder(args, get_atom_fdim(), get_bond_fdim(atom_messages=False))
        self.head = build_ffn(first_linear_dim=300, hidden_size=300, num_layers=2, output_size=len(tasks),
            dropout=config['dropout'], activation='ReLU', dataset_type='regression')
        self.tasks = tuple(tasks)
        self.source_commit = CHEMPROP_COMMIT
        initialize_weights(self)

    def forward(self, mols):
        if any(not isinstance(m, Chem.Mol) for m in mols):
            raise TypeError('D-MPNN requires registered Mols')
        from chemprop.features import mol2graph
        return self.head(self.encoder(mol2graph(mols)))


class EndpointForest:
    """One forest per endpoint, trained directly on original label values."""
    def __init__(self, tasks, config, seed):
        from sklearn.ensemble import RandomForestRegressor
        self.tasks = tuple(tasks)
        self.models = [RandomForestRegressor(n_estimators=config['n_estimators'], max_features=config['max_features'],
            min_samples_leaf=config['min_samples_leaf'], random_state=seed, n_jobs=1) for _ in tasks]

    def fit(self, features, labels):
        for i, model in enumerate(self.models):
            mask = np.isfinite(labels[:, i])
            if not mask.any():
                raise ValueError('No visible labels for ' + self.tasks[i])
            model.fit(features[mask], labels[mask, i])
        return self

    def predict(self, features):
        return np.column_stack([m.predict(features) for m in self.models])


def build_model(method, tasks, config, *, adjacency=None, chemprop_root=None, device='cpu', seed=42):
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('Unique nonempty task registry required')
    if method == 'M0':
        return EndpointForest(tasks, config, seed)
    if method == 'M1':
        result = DMPNN(tasks, config, chemprop_root, device)
    elif method == 'M3':
        if adjacency is None or np.asarray(adjacency).shape != (len(tasks), len(tasks)):
            raise ValueError('M3 requires this exact visible training-view adjacency')
        features, _ = endpoint_feature_matrix(tasks)
        result = ToxACoLNet(adjacency, features, dropout=config['dropout'])
        result.tasks = tuple(tasks)
    else:
        result = GraphPointModel(method, tasks, config)
    return result.to(device)


def parameter_counts(model):
    if not isinstance(model, nn.Module):
        return {'trainable': None, 'total': None, 'kind': 'per_endpoint_forest'}
    return {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total': sum(p.numel() for p in model.parameters()),
            'prediction_heads': sum(p.numel() for p in getattr(model, 'heads', nn.ModuleList()).parameters())}
