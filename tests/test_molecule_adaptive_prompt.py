import torch

from architecture.molecule_adaptive_prompt import MoleculeAdaptiveTaskConditioner
from architecture.toxacute_tasks import (
    ANIMAL_SOURCE_TASKS,
    HUMAN_TARGET_TASKS,
    TOXACUTE_TASKS,
    parse_toxacute_task_name,
)


TASKS = [
    "mouse_oral_LD50",
    "rat_oral_LD50",
    "rabbit_intravenous_LD50",
    "human_oral_TDLo",
]


def build_conditioner(router_mode="dynamic", router_top_k=2):
    torch.manual_seed(13)
    return MoleculeAdaptiveTaskConditioner(
        task_names=TASKS,
        hidden_dim=12,
        use_factorized_prompt=True,
        task_residual_scale=0.1,
        prompt_layers=1,
        prompt_heads=3,
        prompt_ffn_dim=24,
        prompt_dropout=0.0,
        router_mode=router_mode,
        router_dim=12,
        router_top_k=router_top_k,
        router_temperature=1.0,
        exclude_target_from_sources=True,
        adapter_ratio=0.25,
        gate_init=-2.0,
    )


def test_toxacute_registry_and_metadata():
    assert len(TOXACUTE_TASKS) == 59
    assert len(set(TOXACUTE_TASKS)) == 59
    assert len(ANIMAL_SOURCE_TASKS) == 56
    assert len(HUMAN_TARGET_TASKS) == 3
    metadata = parse_toxacute_task_name("women_oral_TDLo")
    assert metadata.organism == "human"
    assert metadata.population == "women"
    assert metadata.route == "oral"
    assert metadata.measurement == "TDLo"
    assert parse_toxacute_task_name("guinea pig_intravenous_LD50").organism == "guinea pig"


def test_dynamic_output_and_diagnostics_contract():
    conditioner = build_conditioner()
    h = torch.randn(5, 12)
    output, diagnostics = conditioner(h, "human_oral_TDLo", return_aux=True)
    assert output.shape == (5, 12)
    assert diagnostics.routing_weights.shape == (5, len(TASKS))
    assert diagnostics.transfer_gate.shape == (5, 1)
    assert diagnostics.routing_entropy.shape == (5,)
    assert diagnostics.task_context.shape == (5, 12)
    assert diagnostics.gamma.shape == (5, 12)
    assert diagnostics.beta.shape == (5, 12)


def test_dynamic_routing_depends_on_molecule_and_excludes_target():
    conditioner = build_conditioner().eval()
    h = torch.stack([torch.ones(12), -torch.ones(12)], dim=0)
    _, diagnostics = conditioner(h, "human_oral_TDLo", return_aux=True)
    target_index = TASKS.index("human_oral_TDLo")
    assert not torch.allclose(diagnostics.routing_weights[0], diagnostics.routing_weights[1])
    assert torch.allclose(diagnostics.routing_weights[:, target_index], torch.zeros(2))
    assert torch.allclose(diagnostics.routing_weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all((diagnostics.routing_weights > 0).sum(dim=-1) <= 2)
    assert torch.all((diagnostics.transfer_gate >= 0) & (diagnostics.transfer_gate <= 1))


def test_source_mask_can_disable_all_sources_without_nan():
    conditioner = build_conditioner().eval()
    h = torch.randn(4, 12)
    _, diagnostics = conditioner(
        h,
        "human_oral_TDLo",
        return_aux=True,
        source_mask=torch.zeros(len(TASKS), dtype=torch.bool),
    )
    assert torch.equal(diagnostics.routing_weights, torch.zeros_like(diagnostics.routing_weights))
    assert torch.equal(diagnostics.routing_entropy, torch.zeros_like(diagnostics.routing_entropy))
    assert torch.isfinite(diagnostics.task_context).all()


def test_prompt_router_adapter_gradients_and_no_cross_batch_state():
    conditioner = build_conditioner()
    h = torch.randn(5, 12)
    conditioner(h, "human_oral_TDLo").square().mean().backward()
    assert conditioner.prompt_bank.task_residual.grad is not None
    assert conditioner.router.query_projection[1].weight.grad is not None
    assert conditioner.film_adapter.adapter_down.weight.grad is not None

    conditioner.eval()
    h_a = torch.randn(4, 12)
    h_b = torch.randn(7, 12)
    first = conditioner(h_a, "human_oral_TDLo")
    conditioner(h_b, "rat_oral_LD50")
    second = conditioner(h_a, "human_oral_TDLo")
    assert torch.allclose(first, second)


def test_static_mode_has_zero_source_routes():
    conditioner = build_conditioner(router_mode="static").eval()
    _, diagnostics = conditioner(torch.randn(3, 12), "human_oral_TDLo", return_aux=True)
    assert torch.equal(diagnostics.routing_weights, torch.zeros_like(diagnostics.routing_weights))
