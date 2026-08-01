import torch

from architecture.task_prompt import RelationAwareTaskConditioner


def build_conditioner():
    torch.manual_seed(7)
    return RelationAwareTaskConditioner(
        task_names=["task_a", "task_b", "task_c"],
        hidden_dim=12,
        prompt_heads=3,
        prompt_layers=1,
        prompt_ffn_dim=24,
        prompt_dropout=0.0,
        adapter_ratio=0.25,
        gate_init=-2.0,
    )


def test_output_shape_and_different_tasks():
    conditioner = build_conditioner()
    h = torch.randn(5, 12)
    output_a = conditioner(h, "task_a")
    output_b = conditioner(h, "task_b")
    assert output_a.shape == (5, 12)
    assert output_b.shape == (5, 12)
    assert not torch.allclose(output_a, output_b)


def test_forward_order_and_batch_size_are_independent():
    conditioner = build_conditioner().eval()
    h = torch.randn(4, 12)
    first_a = conditioner(h, "task_a")
    first_b = conditioner(h, "task_b")
    second_b = conditioner(h, "task_b")
    second_a = conditioner(h, "task_a")
    assert torch.allclose(first_a, second_a)
    assert torch.allclose(first_b, second_b)
    assert conditioner(torch.randn(3, 12), "task_a").shape == (3, 12)
    assert conditioner(torch.randn(7, 12), "task_b").shape == (7, 12)


def test_prompt_parameters_receive_gradients():
    conditioner = build_conditioner()
    output = conditioner(torch.randn(5, 12), "task_a")
    output.square().mean().backward()
    grad = conditioner.relation_encoder.prompt_bank.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    assert conditioner.adapter.film_head.weight.grad.abs().sum() > 0
    assert conditioner.adapter.adapter_down.weight.grad.abs().sum() > 0
    assert conditioner.adapter.adapter_up.weight.grad.abs().sum() > 0


def test_attention_shape_and_unknown_task():
    conditioner = build_conditioner().eval()
    diagnostics = conditioner.get_prompt_diagnostics()
    assert diagnostics.raw_prompts.shape == (3, 12)
    assert diagnostics.contextual_prompts.shape == (3, 12)
    assert diagnostics.attention.shape == (1, 3, 3, 3)
    try:
        conditioner(torch.randn(2, 12), "missing_task")
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown task should raise KeyError")
