"""Representation-path audit metrics (D3 plan §5-§13).

Read-only computation of where the routed-path signal dies, layered exactly as
the plan asks: conditioning context -> FiLM -> adapter -> final routed
representation -> prediction.  Everything derives from tensors the model
already emits in its forward diagnostics plus the model's own submodules; no
module's forward or loss is touched.

Definitions (SharedFiLMAdapter, architecture/response_guided_router.py):
- h                    : base backbone representation (diagnostics["base_representation"])
- conditioned          : condition_norm((1 + gamma) * h + beta)  (FiLM output)
- film delta           : gamma * h + beta                          (FiLM's additive change to h)
- adapter output       : adapter_up(gelu(adapter_down(conditioned)))
- route representation : h + adapter_output
- source context (endpoint_router only) : conditional_weights @ value_projection(source tokens),
  recomputed exactly from diagnostics["response_profile"] + diagnostics["source_weights"].
"""

from __future__ import annotations

import torch

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS


def _norms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().norm(dim=-1)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1e-8)


@torch.no_grad()
def compute_representation_metrics(model, task_name: str, diagnostics: dict) -> dict:
    """Per-sample representation-path metrics for one task's forward batch.

    Returns 1-D tensors (length = batch size) keyed by metric name, or an
    empty dict when the batch carries no RGCER diagnostics (e.g. the plain
    Graphormer HPS model, which emits its own diagnostics flavour).
    """

    if not getattr(model, "is_rgcer", False):
        return {}
    h = diagnostics.get("base_representation")
    if h is None:
        return {}
    h_norm = _norms(h)
    metrics: dict[str, torch.Tensor] = {"h_norm": h_norm}

    conditioner = model.encoder.task_conditioner
    gamma = diagnostics.get("gamma")
    beta = diagnostics.get("beta")
    task_context = diagnostics.get("task_context")
    adapter_output = diagnostics.get("adapter_output")
    route_representation = diagnostics.get("route_representation")

    if task_context is not None:
        metrics["conditioning_context_norm"] = _norms(task_context)
        metrics["conditioning_context_ratio"] = _safe_ratio(
            metrics["conditioning_context_norm"], h_norm
        )

    # endpoint_router only: decompose the post-norm context back into the
    # pre-norm source context by recomputing the router's value path exactly.
    source_weights = diagnostics.get("source_weights")
    response_profile = diagnostics.get("response_profile")
    if getattr(conditioner, "router", None) is not None and source_weights is not None and response_profile is not None:
        prompts = conditioner.prompt_bank()
        tokens = prompts.unsqueeze(0).expand(h.size(0), -1, -1)
        if conditioner.router.use_source_response:
            tokens = tokens + conditioner.router.response_encoder(response_profile)
        values = conditioner.router.value_projection(tokens)
        source_context = torch.bmm(
            source_weights.float().unsqueeze(1), values.float()
        ).squeeze(1)
        metrics["source_context_norm"] = _norms(source_context)
        metrics["source_context_ratio"] = _safe_ratio(
            metrics["source_context_norm"], h_norm
        )
    if getattr(conditioner, "response_stacking", None) is not None and response_profile is not None:
        # D3-B: the stacked response feature before prompt addition/norm (§20).
        stacked = conditioner.response_stacking.response_network(
            response_profile.squeeze(-1).float()
        )
        metrics["stacked_response_norm"] = _norms(stacked)

    if gamma is not None and beta is not None and h is not None:
        film_delta = gamma.float() * h.float() + beta.float()
        metrics["film_delta_norm"] = _norms(film_delta)
        metrics["film_delta_ratio"] = _safe_ratio(metrics["film_delta_norm"], h_norm)
        metrics["gamma_abs_mean"] = gamma.float().abs().mean(dim=-1)
        metrics["beta_abs_mean"] = beta.float().abs().mean(dim=-1)

    if adapter_output is not None:
        metrics["adapter_delta_norm"] = _norms(adapter_output)
        metrics["adapter_delta_ratio"] = _safe_ratio(metrics["adapter_delta_norm"], h_norm)

    if route_representation is not None and h is not None:
        delta = (route_representation.float() - h.float()).norm(dim=-1)
        metrics["route_rep_delta_norm"] = delta
        metrics["route_rep_delta_ratio"] = _safe_ratio(delta, h_norm)

    entropy = diagnostics.get("routing_entropy")
    if entropy is not None:
        metrics["routing_entropy"] = entropy.float()
    if source_weights is not None:
        weights = source_weights.float()
        metrics["routing_variance"] = weights.var(dim=-1, unbiased=False)
        top_weight, top_index = weights.max(dim=-1)
        metrics["top1_weight"] = top_weight
        metrics["top1_index"] = top_index.float()

    return {key: value.detach().cpu() for key, value in metrics.items()}


REPRESENTATION_METRIC_FIELDS = (
    "h_norm",
    "conditioning_context_norm",
    "conditioning_context_ratio",
    "source_context_norm",
    "source_context_ratio",
    "stacked_response_norm",
    "film_delta_norm",
    "film_delta_ratio",
    "gamma_abs_mean",
    "beta_abs_mean",
    "adapter_delta_norm",
    "adapter_delta_ratio",
    "route_rep_delta_norm",
    "route_rep_delta_ratio",
    "routing_entropy",
    "routing_variance",
    "top1_weight",
)

# Aggregate columns written per (epoch, task) — mean over samples.
REPRESENTATION_AGGREGATE_FIELDS = tuple(f"{field}_mean" for field in REPRESENTATION_METRIC_FIELDS)

# Columns of the per-sample best-epoch audit CSV (plan §12; naming per §11).
AUDIT_PER_SAMPLE_FIELDS = (
    "seed",
    "sample_id",
    "row_index",
    "task",
    "label",
    "base_prediction",
    "route_prediction",
    "h_norm",
    "source_context_norm",
    "source_context_ratio",
    "film_delta_norm",
    "film_delta_ratio",
    "adapter_delta_norm",
    "adapter_delta_ratio",
    "route_rep_delta_norm",
    "route_rep_delta_ratio",
    "gamma_abs_mean",
    "beta_abs_mean",
    "routing_entropy",
    "routing_variance",
    "top1_source",
    "top1_weight",
)

AUDIT_SUMMARY_FIELDS = (
    "seed",
    "task",
    "n",
    "h_norm_mean",
    "source_context_norm_mean",
    "source_context_ratio_mean",
    "film_delta_norm_mean",
    "film_delta_ratio_mean",
    "adapter_delta_norm_mean",
    "adapter_delta_ratio_mean",
    "route_rep_delta_norm_mean",
    "route_rep_delta_ratio_mean",
    "gamma_abs_mean_mean",
    "beta_abs_mean_mean",
    "routing_entropy_mean",
    "routing_variance_mean",
    "top1_weight_mean",
    "route_minus_base_abs_mean",
)


def human_tasks_of(task_names) -> list[str]:
    return [task for task in task_names if task in set(HUMAN_TARGET_TASKS)]
