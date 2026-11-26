"""Teacher-free Multi-View Biological Prompt Graph Transformer (MV-BioPGT) and controls.

Models supported under one unified interface:
- MV_BIOPGT_GINE      : Bond-aware GINE + Multi-View Chem + Factorized BioPrompt + Shared-Residual Head
- MV_BIOPGT_GRAPHORMER: Graphormer + Multi-View Chem + Factorized BioPrompt + Shared-Residual Head
- MV_BIOPGT_NOGRAPH   : Ablation removing graph channel (Multi-View Chem + Factorized BioPrompt)
- MV_BIOPGT_NOPROMPT  : Ablation removing factorized prompt & shared-residual readout (GINE/Graphormer + Multi-View Chem + plain heads)
- MV_MLP              : Critical negative control (Avalon + Morgan + 2D Descriptors MLP, no graph, no prompt)
- GINE_ONLY           : Pure bond-aware GINE graph baseline (repaired loss)
- GRAPHORMER_ONLY     : Pure Graphormer graph baseline (repaired loss)
- TOX_AUTHOR          : Exact locked author ToxACoLNet (Avalon + task correlation GCN)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.models.toxacol import ToxACoLNet, endpoint_feature_matrix
from molecular_features import PGT_FEATURE_SPEC
from ..data import make_backbone
from ..models import task_metadata


class GINEBlock(nn.Module):
    """Pre-LN residual bond-aware Graph Isomorphism Network layer in pure PyTorch."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm(h)
        agg = torch.zeros_like(x)
        if edge_src.numel() > 0:
            msg = F.gelu(x[edge_src] + edge_emb)
            agg.index_add_(0, edge_dst, msg)
        out = (1.0 + self.eps) * x + agg
        return h + self.mlp(out)


class GINEBackbone(nn.Module):
    """Pure-PyTorch multi-layer bond-aware GINE returning atom tokens and graph repr."""

    def __init__(
        self,
        hidden_dim: int = 192,
        num_layers: int = 5,
        dropout: float = 0.1,
        spec=PGT_FEATURE_SPEC,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, hidden_dim, padding_idx=0) for c in spec.atom_cardinalities]
        )
        self.bond_embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, hidden_dim, padding_idx=0) for c in spec.bond_cardinalities]
        )
        self.input_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [GINEBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, payload) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = sum(emb(payload.x_flat[:, j]) for j, emb in enumerate(self.atom_embeddings))
        h = self.input_drop(h)
        if payload.edge_src.numel() > 0:
            e = sum(emb(payload.edge_attr[:, j]) for j, emb in enumerate(self.bond_embeddings))
        else:
            e = h.new_zeros((0, self.hidden_dim))

        for layer in self.layers:
            h = layer(h, payload.edge_src, payload.edge_dst, e)
        h = self.final_norm(h)

        b = payload.batch_size
        max_n = payload.max_nodes
        atoms = h.new_zeros((b, max_n, self.hidden_dim))
        mask = torch.zeros((b, max_n), dtype=torch.bool, device=h.device)
        atoms[payload.graph_idx, payload.node_pos] = h
        mask[payload.graph_idx, payload.node_pos] = True

        # Mean + Max pooling (captures both global average and acute local toxicophores)
        lengths = payload.num_nodes.clamp_min(1).to(h.dtype).unsqueeze(-1)
        mean_pool = atoms.sum(dim=1) / lengths
        max_pool = atoms.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        graph_repr = self.pool_proj(torch.cat([mean_pool, max_pool], dim=-1))
        return graph_repr, atoms, mask


class BalancedChemEncoder(nn.Module):
    """Separate normalized projections for Avalon, Morgan, and 2D descriptors."""

    def __init__(
        self,
        *,
        avalon_dim: int = 1024,
        morgan_dim: int = 2048,
        desc_dim: int = 52,
        out_dim: int = 192,
        fp_hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.avalon_branch = nn.Sequential(
            nn.Linear(avalon_dim, fp_hidden_dim),
            nn.LayerNorm(fp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.morgan_branch = nn.Sequential(
            nn.Linear(morgan_dim, fp_hidden_dim),
            nn.LayerNorm(fp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.desc_branch = nn.Sequential(
            nn.Linear(desc_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        fused_in = fp_hidden_dim * 2 + 64
        mid_dim = max(384, out_dim * 2) if fp_hidden_dim > 256 else 384
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(fused_in, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        avalon: torch.Tensor,
        morgan: torch.Tensor,
        desc: torch.Tensor,
    ) -> torch.Tensor:
        a = self.avalon_branch(avalon)
        m = self.morgan_branch(morgan)
        d = self.desc_branch(desc)
        cat = torch.cat([a, m, d], dim=-1)
        return self.out_norm(self.fuse(cat) + self.skip(cat))


class FactorizedBioPromptBank(nn.Module):
    """Hierarchical biological endpoint prompts with residual empirical-graph propagation.

    p_t^(0) = LN(e_organism + e_route + e_measurement + e_population + sqrt(n_t/(n_t+tau)) * delta_t)
    p_t     = LN(p_t^(0) + tanh(g_rel) * GELU(A_train @ p^(0) @ W_rel))

    Data-rich endpoints leverage empirical correlation edges in A_train while scarce/human
    endpoints (which become isolated with degree 0 in A_train) remain connected through
    shared organism/route/measurement biological factors.
    """

    def __init__(
        self,
        tasks: tuple[str, ...],
        dim: int,
        counts: np.ndarray,
        *,
        adjacency: np.ndarray | None = None,
        shrinkage_tau: float = 32.0,
    ):
        super().__init__()
        self.tasks = tuple(tasks)
        self.dim = dim
        meta = task_metadata(tasks)
        self.fields = ("organism", "route", "measurement", "population")
        self.embeddings = nn.ModuleDict()
        for field in self.fields:
            vocab = sorted({meta[t][field] for t in tasks})
            indices = [vocab.index(meta[t][field]) for t in tasks]
            self.register_buffer(f"{field}_idx", torch.tensor(indices, dtype=torch.long))
            emb = nn.Embedding(len(vocab), dim)
            nn.init.normal_(emb.weight, std=0.02)
            self.embeddings[field] = emb
        count_t = torch.as_tensor(counts, dtype=torch.float32).clamp_min(1.0)
        self.register_buffer("shrinkage", (count_t / (count_t + float(shrinkage_tau))).sqrt())
        self.task_residual = nn.Parameter(torch.zeros(len(tasks), dim))
        self.base_norm = nn.LayerNorm(dim)
        if adjacency is not None:
            adj = np.asarray(adjacency, dtype=np.float32)
            if adj.shape != (len(tasks), len(tasks)):
                raise ValueError("adjacency shape must match number of tasks")
            self.register_buffer("adjacency", torch.from_numpy(adj.copy()))
            self.rel_proj = nn.Linear(dim, dim, bias=False)
            self.rel_gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("adjacency", None)
            self.rel_proj = None
            self.rel_gate = None
        self.norm = nn.LayerNorm(dim)

    def forward(self, *, factor_override: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        total = 0.0
        for field in self.fields:
            idx = (
                factor_override[field]
                if (factor_override is not None and field in factor_override)
                else getattr(self, f"{field}_idx")
            )
            total = total + self.embeddings[field](idx)
        base = self.base_norm(total + self.shrinkage[:, None] * self.task_residual)
        if self.adjacency is not None and self.rel_proj is not None:
            propagated = F.gelu(self.adjacency @ self.rel_proj(base))
            return self.norm(base + torch.tanh(self.rel_gate) * propagated)
        return self.norm(base)


class TaskAtomCrossAttention(nn.Module):
    """Endpoint prompts query molecular atom tokens to produce localized subgraph vectors."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        prompts: torch.Tensor,
        atoms: torch.Tensor,
        atom_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # prompts: [T, D], atoms: [B, N, D], atom_mask: [B, N] True=valid atom
        b, n, _ = atoms.shape
        t = prompts.shape[0]
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(prompts).view(t, h, d)
        k = self.k_proj(atoms).view(b, n, h, d)
        v = self.v_proj(atoms).view(b, n, h, d)
        logits = torch.einsum("thd,bnhd->bthn", q, k) * self.scale
        logits = logits.masked_fill(~atom_mask[:, None, None, :], -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn_drop = self.drop(attn)
        context = torch.einsum("bthn,bnhd->bthd", attn_drop, v).reshape(b, t, self.dim)
        out = torch.tanh(self.gate) * self.out_proj(context)
        return out, (attn if return_attention else None)


class MVBioPGT(nn.Module):
    """Unified multi-view graph + chemical context + factorized biological prompt model."""

    def __init__(
        self,
        tasks: tuple[str, ...],
        counts: np.ndarray,
        *,
        adjacency: np.ndarray | None = None,
        use_graph: bool = True,
        graph_type: str = "gine",
        use_chem: bool = True,
        use_prompt: bool = True,
        ordinal_dose: bool = False,
        chem_drop_prob: float = 0.0,
        fp_hidden_dim: int = 256,
        chem_dropout: float | None = None,
        hidden_dim: int = 192,
        graph_layers: int = 5,
        graph_heads: int = 4,
        dropout: float = 0.2,
        desc_dim: int = 52,
        shrinkage_tau: float = 32.0,
    ):
        super().__init__()
        if not use_graph and not use_chem:
            raise ValueError("At least one molecular modality (graph or chem) must be enabled")
        self.tasks = tuple(tasks)
        self.num_tasks = len(tasks)
        self.use_graph = bool(use_graph)
        self.graph_type = str(graph_type) if use_graph else "none"
        self.use_chem = bool(use_chem)
        self.use_prompt = bool(use_prompt)
        self.ordinal_dose = bool(ordinal_dose) and self.use_prompt
        self.chem_drop_prob = float(chem_drop_prob) if (self.use_graph and self.use_chem) else 0.0
        self.hidden_dim = int(hidden_dim)
        c_drop = float(dropout if chem_dropout is None else chem_dropout)

        if self.use_graph:
            if self.graph_type == "gine":
                self.graph_encoder = GINEBackbone(
                    hidden_dim=hidden_dim, num_layers=graph_layers, dropout=dropout * 0.5
                )
            elif self.graph_type == "graphormer":
                self.graph_encoder = make_backbone(
                    hidden_dim=hidden_dim,
                    ffn_dim=hidden_dim * 2,
                    num_heads=graph_heads,
                    num_layers=graph_layers,
                    dropout=dropout * 0.5,
                )
            else:
                raise ValueError(f"Unknown graph_type: {self.graph_type}")
            self.graph_out_norm = nn.LayerNorm(hidden_dim)
        else:
            self.graph_encoder = None
            self.graph_out_norm = None

        if self.use_chem:
            self.chem_encoder = BalancedChemEncoder(
                desc_dim=desc_dim,
                out_dim=hidden_dim,
                fp_hidden_dim=int(fp_hidden_dim),
                dropout=c_drop,
            )
        else:
            self.chem_encoder = None

        if self.use_graph and self.use_chem:
            self.modal_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.modal_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.modal_norm = nn.LayerNorm(hidden_dim)

        if self.use_prompt:
            self.prompt_bank = FactorizedBioPromptBank(
                self.tasks, hidden_dim, counts, adjacency=adjacency, shrinkage_tau=shrinkage_tau
            )
            if self.use_graph:
                self.atom_cross_attn = TaskAtomCrossAttention(
                    hidden_dim, num_heads=graph_heads, dropout=dropout * 0.5
                )
            else:
                self.atom_cross_attn = None
            # FiLM modulation (zero-init residual -> identity at init)
            self.film = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
            self.task_norm = nn.LayerNorm(hidden_dim)
            # Shared biological decoder (uses all 59 tasks jointly) + shrinkage linear residual
            self.shared_decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.endpoint_weight = nn.Parameter(torch.zeros(self.num_tasks, hidden_dim))
            self.endpoint_bias = nn.Parameter(torch.zeros(self.num_tasks))

            if self.ordinal_dose:
                meta = task_metadata(self.tasks)
                meas_ranks = []
                for t in self.tasks:
                    m_name = meta[t]["measurement"]
                    meas_ranks.append(2 if m_name == "TDLo" else (1 if m_name == "LDLo" else 0))
                self.register_buffer("meas_rank", torch.tensor(meas_ranks, dtype=torch.long))
                self.basal_prompt_norm = nn.LayerNorm(hidden_dim)
                self.basal_film = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )
                nn.init.zeros_(self.basal_film[-1].weight)
                nn.init.zeros_(self.basal_film[-1].bias)
                self.basal_norm = nn.LayerNorm(hidden_dim)
                self.basal_decoder = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                # Monotone potency ladder in -log10(mol/kg): 0 = LD50 <= LDLo <= TDLo
                # softplus(-2.48) ~= 0.080 (LDLo step), softplus(0.95) ~= 1.277 (TDLo step -> +1.36 total)
                self.ord_step_ldlo = nn.Parameter(torch.tensor(-2.48, dtype=torch.float32))
                self.ord_step_tdlo = nn.Parameter(torch.tensor(0.95, dtype=torch.float32))
                # Zero-init molecule-specific threshold adjustment on top of monotone ladder
                nn.init.zeros_(self.shared_decoder[-1].weight)
                nn.init.zeros_(self.shared_decoder[-1].bias)
        else:
            # Clean vectorized multi-task head without biological prompts
            self.plain_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_tasks),
            )

    def ordinal_threshold_ladder(self) -> torch.Tensor:
        """Return monotone 3-level potency offsets [tau_LD50, tau_LDLo, tau_TDLo] with 0 <= tau1 <= tau2."""
        zero = self.ord_step_ldlo.new_zeros(1)
        tau_ldlo = F.softplus(self.ord_step_ldlo).view(1)
        tau_tdlo = tau_ldlo + F.softplus(self.ord_step_tdlo).view(1)
        return torch.cat([zero, tau_ldlo, tau_tdlo], dim=0)

    def encode_molecule(self, payload):
        h_graph = None
        h_graph_norm = None
        atoms = None
        atom_mask = None
        if self.use_graph:
            if self.graph_type == "gine":
                h_graph, atoms, atom_mask = self.graph_encoder(payload)
            else:
                tokens = self.graph_encoder.encode_tokens(payload.graphormer_batch)
                h_graph = tokens.graph_token
                atoms = tokens.atom_tokens
                atom_mask = tokens.atom_mask
            h_graph_norm = self.graph_out_norm(h_graph)

        h_chem = (
            self.chem_encoder(payload.avalon, payload.morgan, payload.desc)
            if self.use_chem
            else None
        )

        if self.use_graph and self.use_chem:
            cat = torch.cat([h_graph, h_chem], dim=-1)
            g = self.modal_gate(cat)
            blended = g * h_graph + (1.0 - g) * h_chem
            fused = self.modal_norm(blended + self.modal_proj(cat))
            if self.training and self.chem_drop_prob > 0.0:
                drop_mask = (
                    torch.rand((h_graph.shape[0], 1), device=h_graph.device) < self.chem_drop_prob
                )
                h_mol = torch.where(drop_mask, h_graph_norm, fused)
            else:
                h_mol = fused
        elif self.use_graph:
            h_mol = h_graph
        else:
            h_mol = h_chem
        return h_mol, h_graph_norm, atoms, atom_mask

    def _decode_from_repr(
        self,
        h_mol: torch.Tensor,
        atoms: torch.Tensor | None,
        atom_mask: torch.Tensor | None,
        *,
        prompt_override: torch.Tensor | None = None,
        factor_override: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
        detach_prompts: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        aux = {}
        if not self.use_prompt:
            return self.plain_head(h_mol), aux

        prompts = (
            prompt_override
            if prompt_override is not None
            else self.prompt_bank(factor_override=factor_override)
        )
        if detach_prompts:
            prompts = prompts.detach()

        atom_ctx = None
        if self.atom_cross_attn is not None and atoms is not None and atom_mask is not None:
            atom_ctx, attn = self.atom_cross_attn(
                prompts, atoms, atom_mask, return_attention=return_aux
            )
            if return_aux:
                aux["atom_attention"] = attn
                aux["atom_mask"] = atom_mask

        film_params = self.film(prompts)  # [T, 2D]
        gamma, beta = film_params.chunk(2, dim=-1)
        z = (1.0 + gamma[None, :, :]) * h_mol[:, None, :] + beta[None, :, :]
        if atom_ctx is not None:
            z = z + atom_ctx
        z = self.task_norm(z)  # [B, T, D]

        p_exp = prompts[None, :, :].expand(z.shape[0], -1, -1)
        shrink = self.prompt_bank.shrinkage[None, :]  # [1, T]
        residual_pred = shrink * torch.einsum("btd,td->bt", z, self.endpoint_weight)

        if self.ordinal_dose:
            org_idx = (
                factor_override["organism"]
                if (factor_override is not None and "organism" in factor_override)
                else self.prompt_bank.organism_idx
            )
            route_idx = (
                factor_override["route"]
                if (factor_override is not None and "route" in factor_override)
                else self.prompt_bank.route_idx
            )
            p_sr = self.basal_prompt_norm(
                self.prompt_bank.embeddings["organism"](org_idx)
                + self.prompt_bank.embeddings["route"](route_idx)
            )
            if detach_prompts:
                p_sr = p_sr.detach()
            sr_gamma, sr_beta = self.basal_film(p_sr).chunk(2, dim=-1)
            z_sr = (1.0 + sr_gamma[None, :, :]) * h_mol[:, None, :] + sr_beta[None, :, :]
            if atom_ctx is not None:
                z_sr = z_sr + atom_ctx
            z_sr = self.basal_norm(z_sr)
            p_sr_exp = p_sr[None, :, :].expand(z_sr.shape[0], -1, -1)
            basal_pred = self.basal_decoder(torch.cat([z_sr, p_sr_exp], dim=-1)).squeeze(-1)

            ladder = self.ordinal_threshold_ladder()  # [3]
            tau_offset = ladder[self.meas_rank][None, :]  # [1, T]
            meas_adj = self.shared_decoder(torch.cat([z, p_exp - p_sr_exp], dim=-1)).squeeze(-1)
            pred = basal_pred + tau_offset + meas_adj + residual_pred + self.endpoint_bias[None, :]
            if return_aux:
                aux["ordinal_ladder"] = ladder.detach()
                aux["basal_pred"] = basal_pred.detach()
        else:
            shared_pred = self.shared_decoder(torch.cat([z, p_exp], dim=-1)).squeeze(-1)
            pred = shared_pred + residual_pred + self.endpoint_bias[None, :]

        if return_aux:
            aux["prompts"] = prompts
            aux["h_mol"] = h_mol
        return pred, aux

    def forward(
        self,
        payload,
        *,
        return_aux: bool = False,
        return_train_aux: bool = False,
        prompt_override: torch.Tensor | None = None,
        factor_override: dict[str, torch.Tensor] | None = None,
    ):
        h_mol, h_graph_norm, atoms, atom_mask = self.encode_molecule(payload)
        pred, aux = self._decode_from_repr(
            h_mol,
            atoms,
            atom_mask,
            prompt_override=prompt_override,
            factor_override=factor_override,
            return_aux=return_aux,
            detach_prompts=False,
        )
        if return_train_aux:
            train_aux = {}
            if self.use_graph and self.use_chem and h_graph_norm is not None:
                graph_pred, _ = self._decode_from_repr(
                    h_graph_norm,
                    atoms,
                    atom_mask,
                    prompt_override=prompt_override,
                    factor_override=factor_override,
                    return_aux=False,
                    detach_prompts=True,
                )
                train_aux["graph_pred"] = graph_pred
            return pred, train_aux
        return (pred, aux) if return_aux else pred


class FastToxACoLWrapper(nn.Module):
    """Wraps locked ToxACoLNet to accept BatchPayload directly from RAM."""

    def __init__(self, tasks: tuple[str, ...], adjacency: np.ndarray, dropout: float = 0.1):
        super().__init__()
        features, _ = endpoint_feature_matrix(tasks)
        self.net = ToxACoLNet(adjacency, features, dropout=dropout)
        self.tasks = tuple(tasks)

    def forward(self, payload):
        return self.net(payload.avalon)


def build_mv_model(
    method: str,
    tasks: tuple[str, ...],
    counts: np.ndarray,
    *,
    adjacency: np.ndarray | None = None,
    config: dict | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    cfg = dict(config or {})
    if method == "TOX_AUTHOR":
        if adjacency is None:
            raise ValueError("TOX_AUTHOR requires visible training-view adjacency")
        model = FastToxACoLWrapper(tasks, adjacency, dropout=float(cfg.get("dropout", 0.1)))
        return model.to(device)

    specs = {
        "MV_BIOPGT_GINE": dict(use_graph=True, graph_type="gine", use_chem=True, use_prompt=True),
        "MV_BIOPGT_GRAPHORMER": dict(
            use_graph=True, graph_type="graphormer", use_chem=True, use_prompt=True
        ),
        "MV_BIOPGT_NOGRAPH": dict(use_graph=False, graph_type="none", use_chem=True, use_prompt=True),
        "MV_BIOPGT_NOPROMPT_GINE": dict(
            use_graph=True, graph_type="gine", use_chem=True, use_prompt=False
        ),
        "MV_BIOPGT_NOPROMPT_GRAPHORMER": dict(
            use_graph=True, graph_type="graphormer", use_chem=True, use_prompt=False
        ),
        "MV_MLP": dict(use_graph=False, graph_type="none", use_chem=True, use_prompt=False),
        "GINE_ONLY": dict(use_graph=True, graph_type="gine", use_chem=False, use_prompt=False),
        "GRAPHORMER_ONLY": dict(
            use_graph=True, graph_type="graphormer", use_chem=False, use_prompt=False
        ),
        "MV_BIOPGT_V2": dict(
            use_graph=True,
            graph_type="gine",
            use_chem=True,
            use_prompt=True,
            ordinal_dose=True,
            chem_drop_prob=0.25,
            fp_hidden_dim=384,
            chem_dropout=0.30,
            default_dim=256,
        ),
        "MV_BIOPGT_V2_NOMDROP": dict(
            use_graph=True,
            graph_type="gine",
            use_chem=True,
            use_prompt=True,
            ordinal_dose=True,
            chem_drop_prob=0.0,
            fp_hidden_dim=384,
            chem_dropout=0.20,
            default_dim=256,
        ),
        "MV_BIOPGT_V2_NOGRAPH": dict(
            use_graph=False,
            graph_type="none",
            use_chem=True,
            use_prompt=True,
            ordinal_dose=True,
            chem_drop_prob=0.0,
            fp_hidden_dim=384,
            chem_dropout=0.30,
            default_dim=256,
        ),
        "MV_BIOPGT_V2_NOPROMPT": dict(
            use_graph=True,
            graph_type="gine",
            use_chem=True,
            use_prompt=False,
            ordinal_dose=False,
            chem_drop_prob=0.25,
            fp_hidden_dim=384,
            chem_dropout=0.30,
            default_dim=256,
        ),
    }
    if method not in specs:
        raise ValueError(f"Unknown MV method: {method}")
    flags = specs[method]
    default_dim = flags.get("default_dim", 128 if flags["graph_type"] == "graphormer" else 192)
    default_layers = 4 if flags["graph_type"] == "graphormer" else 5
    model = MVBioPGT(
        tasks=tuple(tasks),
        counts=counts,
        adjacency=adjacency,
        use_graph=flags["use_graph"],
        graph_type=flags["graph_type"],
        use_chem=flags["use_chem"],
        use_prompt=flags["use_prompt"],
        ordinal_dose=bool(cfg.get("ordinal_dose", flags.get("ordinal_dose", False))),
        chem_drop_prob=float(cfg.get("chem_drop_prob", flags.get("chem_drop_prob", 0.0))),
        fp_hidden_dim=int(cfg.get("fp_hidden_dim", flags.get("fp_hidden_dim", 256))),
        chem_dropout=cfg.get("chem_dropout", flags.get("chem_dropout", None)),
        hidden_dim=int(cfg.get("hidden_dim", default_dim)),
        graph_layers=int(cfg.get("graph_layers", default_layers)),
        graph_heads=int(cfg.get("graph_heads", 4)),
        dropout=float(cfg.get("dropout", 0.2)),
        desc_dim=int(cfg.get("desc_dim", 52)),
        shrinkage_tau=float(cfg.get("shrinkage_tau", 32.0)),
    )
    return model.to(device)


def method_input_needs(method: str) -> dict:
    if method == "TOX_AUTHOR":
        return dict(need_avalon=True, need_morgan=False, need_desc=False, graph_mode=None)
    if method == "RF":
        return dict(need_avalon=False, need_morgan=True, need_desc=True, graph_mode=None)
    if method in ("MV_MLP", "MV_BIOPGT_NOGRAPH", "MV_BIOPGT_V2_NOGRAPH"):
        return dict(need_avalon=True, need_morgan=True, need_desc=True, graph_mode=None)
    if method == "GINE_ONLY":
        return dict(need_avalon=False, need_morgan=False, need_desc=False, graph_mode="gine")
    if method == "GRAPHORMER_ONLY":
        return dict(need_avalon=False, need_morgan=False, need_desc=False, graph_mode="graphormer")
    if method in (
        "MV_BIOPGT_GINE",
        "MV_BIOPGT_NOPROMPT_GINE",
        "MV_BIOPGT_V2",
        "MV_BIOPGT_V2_NOMDROP",
        "MV_BIOPGT_V2_NOPROMPT",
    ):
        return dict(need_avalon=True, need_morgan=True, need_desc=True, graph_mode="gine")
    if method in ("MV_BIOPGT_GRAPHORMER", "MV_BIOPGT_NOPROMPT_GRAPHORMER"):
        return dict(need_avalon=True, need_morgan=True, need_desc=True, graph_mode="graphormer")
    raise ValueError(f"Unknown method: {method}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
