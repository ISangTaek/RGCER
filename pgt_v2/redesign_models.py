"""G02R candidates: in-encoder task tokens, shared decoding and task regularization.

These are adaptations within PGT, not reproductions of published models. Atom
attention is diagnostic; it is not labelled a causal or complete explanation.
"""
from __future__ import annotations

import torch
from torch import nn

from architecture.prediction_heads import TaskPredictionHead
from .data import make_backbone

REDESIGN_METHODS = ('N0', 'N1', 'N1S', 'N2', 'N3', 'N4')
FINGERPRINT_METHODS = ('N0', 'N4')


def task_smoother(metadata, counts, strength=1.0, *, return_laplacian=False):
    """Solution of .5||D^.5(P-E)||² + .5*strength*tr(P'LP).

    Relations are defined before outcomes: equal measurement, and equal organism
    or route. Population identity remains in E; generic population labels never
    create graph edges. Counts use visible training labels only.
    """
    n = len(metadata)
    weights = torch.zeros(n, n, dtype=torch.float64)
    for i, left in enumerate(metadata):
        for j, right in enumerate(metadata):
            if i != j and left['measurement'] == right['measurement']:
                weights[i, j] = 0.5 * (left['organism'] == right['organism']) + 0.5 * (left['route'] == right['route'])
    degree = weights.sum(1)
    laplacian = (torch.diag(degree) - weights) / degree.max().clamp_min(1.0)
    count = torch.as_tensor(counts, dtype=torch.float64)
    if count.shape != (n,) or bool((count <= 0).any()):
        raise ValueError('Task smoothing requires positive visible training counts')
    precision = torch.diag(count / (count + 32.0))
    smoother = torch.linalg.solve(precision + strength * laplacian, precision)
    return (smoother.float(), laplacian.float()) if return_laplacian else smoother.float()


class SemanticPrompts(nn.Module):
    def __init__(self, metadata, dim, rank=8, counts=None, smoothing=False):
        super().__init__()
        self.fields = ('organism', 'route', 'measurement', 'population')
        self.embeddings = nn.ModuleDict()
        for field in self.fields:
            vocabulary = sorted({row[field] for row in metadata})
            indices = [vocabulary.index(row[field]) for row in metadata]
            self.register_buffer(field + '_indices', torch.tensor(indices, dtype=torch.long))
            table = nn.Embedding(len(vocabulary), rank)
            nn.init.normal_(table.weight, std=0.02)
            self.embeddings[field] = table
        self.residual = nn.Parameter(torch.randn(len(metadata), rank) * 0.02)
        self.projection = nn.Linear(rank, dim, bias=False)
        smoother, laplacian = (task_smoother(metadata, counts, return_laplacian=True) if smoothing else
                               (torch.eye(len(metadata)), torch.zeros(len(metadata), len(metadata))))
        self.register_buffer('smoother', smoother)
        self.register_buffer('laplacian', laplacian)

    def forward(self):
        factors = [self.embeddings[field](getattr(self, field + '_indices')) for field in self.fields]
        raw = sum(factors) / 2.0 + 0.1 * self.residual
        return self.projection(self.smoother @ raw)


class DeepPromptModel(nn.Module):
    """N0 late fusion; N1 deep IDs; N1S shared ID; N2 semantics; N3 prior; N4 fusion."""
    def __init__(self, method, tasks, config, metadata, counts=None):
        super().__init__()
        self.method, self.tasks = method, tuple(tasks)
        self.deep = method != 'N0'
        self.fusion = method in FINGERPRINT_METHODS
        dim = config['hidden_dim']
        self.backbone = make_backbone(hidden_dim=dim, ffn_dim=config['ffn_dim'],
            num_heads=config['heads'], num_layers=config['layers'], dropout=config['dropout'])
        if method in ('N1', 'N1S'):
            self.prompt_ids = nn.Parameter(torch.randn(len(tasks), dim) * 0.02)
            self.prompt_bank = None
        else:
            self.register_parameter('prompt_ids', None)
            self.prompt_bank = SemanticPrompts([metadata[name] for name in tasks], dim,
                rank=config.get('prompt_rank', 8), counts=counts, smoothing=method == 'N3')
        if self.deep:
            self.layer_gates = nn.Parameter(torch.full((config['layers'],), 0.1))
        self.readout_norm = nn.LayerNorm(dim)
        if method in ('N1', 'N1S'):
            self.heads = nn.ModuleList([TaskPredictionHead(dim, mode='point', head_hidden_dim=dim,
                dropout=config['dropout']) for _ in tasks])
        else:
            self.shared_head = TaskPredictionHead(dim, mode='point', head_hidden_dim=dim, dropout=config['dropout'])
            self.task_bias = nn.Parameter(torch.zeros(len(tasks)))
        if self.fusion:
            self.fingerprint_encoder = nn.Sequential(nn.Linear(1024, 256), nn.GELU(),
                nn.Dropout(config['dropout']), nn.Linear(256, dim), nn.LayerNorm(dim))
            self.fingerprint_head = TaskPredictionHead(dim, mode='point', head_hidden_dim=dim,
                dropout=config['dropout'])
            self.fusion_gate = nn.Linear(2 * dim, 1)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)
        if method == 'N1S':
            # Finish the full N1 initialization before tying prompts, preserving
            # all common initial weights and the RNG state for paired seeds.
            self.prompt_ids = nn.Parameter(self.prompt_ids[:1].detach().clone())

    def prompts(self):
        if self.method == 'N1S':
            return self.prompt_ids.expand(len(self.tasks), -1)
        return self.prompt_ids if self.prompt_bank is None else self.prompt_bank()

    def regularization_loss(self):
        prompt = self.prompts()
        if self.method != 'N3':
            return prompt.new_zeros(())
        # The learned raw prompts could undo an invertible smoother. Penalizing
        # the resulting prompts makes the task prior an actual objective term.
        return 0.01 * (prompt * (self.prompt_bank.laplacian @ prompt)).sum() / prompt.numel()

    def forward(self, inputs, *, return_aux=False, atom_embeddings=None, query_override=None):
        if self.fusion:
            batch, fingerprint = inputs
        else:
            batch = inputs
        prompt = self.prompts() if query_override is None else query_override
        if prompt.shape != (len(self.tasks), self.backbone.hidden_dim):
            raise ValueError('Prompt replacement must preserve the task registry')
        offsets = torch.tanh(self.layer_gates)[:, None, None] * prompt[None] if self.deep else None
        tokens = self.backbone.encode_tokens(batch, atom_embeddings=atom_embeddings,
            prompt_embeddings=prompt if self.deep else None, layer_prompt_offsets=offsets,
            return_attention=return_aux and self.deep)
        representation = self.readout_norm(tokens.graph_token[:, None, :] +
            (tokens.prompt_tokens if self.deep else prompt[None]))
        if self.method in ('N1', 'N1S'):
            prediction = torch.cat([head(representation[:, i]) for i, head in enumerate(self.heads)], dim=-1)
        else:
            prediction = self.shared_head(representation.reshape(-1, representation.shape[-1])).reshape(representation.shape[:2])
            prediction = prediction + self.task_bias
        aux = dict(atom_mask=tokens.atom_mask, task_prompts=prompt)
        if self.fusion:
            fp = self.fingerprint_encoder(fingerprint)[:, None, :] + prompt[None]
            fp_prediction = self.fingerprint_head(fp.reshape(-1, fp.shape[-1])).reshape(fp.shape[:2]) + self.task_bias
            gate = torch.sigmoid(self.fusion_gate(torch.cat([representation, fp], dim=-1))).squeeze(-1)
            graph_part, fingerprint_part = (1 - gate) * prediction, gate * fp_prediction
            prediction = graph_part + fingerprint_part
            aux.update(fusion_gate=gate, graph_prediction_part=graph_part, fingerprint_prediction_part=fingerprint_part)
        if return_aux and self.deep:
            n = tokens.atom_tokens.shape[1]
            aux['prompt_atom_attention'] = tokens.attentions[-1][:, :, n+1:, 1:n+1].transpose(1, 2)
            # This slice excludes graph/prompt destinations, so rows need not sum
            # to one. Do not renormalize it into an alleged complete attribution.
        return (prediction, aux) if return_aux else prediction
