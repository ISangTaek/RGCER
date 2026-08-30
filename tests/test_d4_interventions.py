"""D4 Stage-1 intervention contracts (plan §66-§67).

Covers: hook identity (no-op hook == original forward), Base-only ==
NULL-forced-1, weight normalisation after uniform/permutation/deletion
interventions, and the validation-only loader contract of the audit passes.
"""

import csv

import torch

from architecture.response_guided_router import ResponseGuidedEndpointRouter
from scripts.d4_intervention_audit import (
    make_deletion,
    make_null_one,
    make_permutation,
    make_route_only,
    make_uniform,
    run_pass,
)


def _router(task_count=6, top_k=3):
    router = ResponseGuidedEndpointRouter(hidden_dim=8, router_dim=8, top_k=top_k)
    router.eval()
    return router


def _routing_inputs(batch=4, task_count=6):
    torch.manual_seed(7)
    molecular = torch.randn(batch, 8)
    prompts = torch.randn(task_count, 8)
    profile = torch.randn(batch, task_count, 1)
    return molecular, prompts, profile, 0


def _forward(router):
    molecular, prompts, profile, target = _routing_inputs()
    with torch.no_grad():
        return router(molecular, prompts, profile, target)


def test_noop_hook_is_identical_to_original_forward():
    router = _router()
    reference = _forward(router)

    def noop(**kwargs):
        return kwargs["conditional_weights"], kwargs["joint_source_weights"], kwargs["null_weight"], kwargs["entropy"]

    router.intervention = noop
    intervened = _forward(router)
    router.intervention = None

    assert torch.allclose(reference[0], intervened[0])
    assert torch.allclose(reference[3], intervened[3])


def _hook_outputs(router, hook):
    molecular, prompts, profile, target = _routing_inputs()
    router.intervention = hook
    try:
        with torch.no_grad():
            context, conditional, joint, null, entropy = router(molecular, prompts, profile, target)
    finally:
        router.intervention = None
    return context, conditional, joint, null, entropy


def test_route_only_forces_null_to_zero_and_keeps_conditional_sum():
    router = _router()
    learned = _forward(router)
    _, conditional, joint, null, _ = _hook_outputs(router, make_route_only(learned_null=learned[3][:1, :1]))

    assert float(null.abs().max()) == 0.0
    assert torch.allclose(conditional.sum(dim=-1), torch.ones(conditional.size(0)), atol=1e-5)


def test_null_one_hook():
    router = _router()
    _, _, joint, null, entropy = _hook_outputs(router, make_null_one())
    assert float((null - 1.0).abs().max()) == 0.0
    assert float(entropy.abs().max()) == 0.0
    assert torch.allclose(joint[:, :1], torch.ones_like(joint[:, :1]))


def test_uniform_intervention_sums_to_one_and_uses_active_set():
    router = _router()
    learned = _forward(router)
    allowed = learned[1][0] > 0
    _, conditional, _, null, _ = _hook_outputs(router, make_uniform(learned[3][:1, :1], keep_null=True))

    active_count = int(allowed.sum())
    assert torch.allclose(conditional, conditional[0].expand_as(conditional))  # endpoint-global
    assert torch.allclose(
        conditional[0][allowed], torch.full((active_count,), 1.0 / active_count), atol=1e-5
    )
    assert torch.allclose(conditional.sum(dim=-1), torch.ones(conditional.size(0)), atol=1e-5)


def test_permutation_keeps_values_and_sum():
    router = _router()
    learned = _forward(router)
    learned_conditional = learned[1][0].detach()
    learned_null = learned[3][:1, :1]
    _, conditional, _, null, _ = _hook_outputs(
        router,
        make_permutation(learned_conditional, learned_conditional > 0, learned_null, 1001),
    )
    assert torch.allclose(conditional.sum(dim=-1), torch.ones(conditional.size(0)), atol=1e-5)
    # Same multiset of weight values, different assignment.
    assert torch.allclose(
        conditional[0].sort().values, learned_conditional.sort().values, atol=1e-6
    )


def test_deletion_renormalises_to_one():
    router = _router()
    learned = _forward(router)
    learned_conditional = learned[1][0].detach()
    victim = int(learned_conditional.argmax())
    _, conditional, _, _, _ = _hook_outputs(
        router,
        make_deletion(learned_conditional, learned_conditional > 0, learned[3][:1, :1], victim, None),
    )
    assert float(conditional[0][victim]) == 0.0
    assert torch.allclose(conditional.sum(dim=-1), torch.ones(conditional.size(0)), atol=1e-5)


def test_run_pass_only_touches_the_requested_validation_loader(tmp_path):
    """§67 no-leakage: run_pass receives exactly one loader and never the
    calibration/test dicts."""

    class _ProbeLoader:
        def __init__(self):
            self.iterated = False

        def __iter__(self):
            self.iterated = True
            return iter([])

        def __len__(self):
            return 0

    val_loader = _ProbeLoader()
    calibration_loader = _ProbeLoader()
    test_loader = _ProbeLoader()
    loaders = {"val": val_loader, "calibration": calibration_loader, "test": test_loader}

    class _TrainerStub:
        model = None
        device = torch.device("cpu")
        sample_row_index = {}

        def _forward_task(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("must not forward on an empty loader")

        def decode_task_output(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    result = run_pass(_TrainerStub(), loaders["val"], "human_oral_TDLo", 0)
    assert val_loader.iterated is True
    assert calibration_loader.iterated is False
    assert test_loader.iterated is False
    assert result["final"].numel() == 0
