import torch

from architecture.response_guided_router import RGCERTaskConditioner, SharedFiLMAdapter


def test_shared_conditioner_flags_change_only_the_requested_computation():
    torch.manual_seed(17)
    h = torch.randn(4, 8)
    context = torch.randn(4, 8)

    full = SharedFiLMAdapter(8, adapter_ratio=0.5, dropout=0.0, use_film=True, use_adapter=True).eval()
    full_representation, full_aux = full(h, context)
    assert torch.any(full_aux["gamma"] != 0)
    assert torch.any(full_aux["beta"] != 0)
    assert torch.any(full_aux["adapter_output"] != 0)

    no_film = SharedFiLMAdapter(8, adapter_ratio=0.5, dropout=0.0, use_film=False, use_adapter=True).eval()
    no_film_representation, no_film_aux = no_film(h, context)
    assert torch.equal(no_film_aux["gamma"], torch.zeros_like(h))
    assert torch.equal(no_film_aux["beta"], torch.zeros_like(h))
    torch.testing.assert_close(no_film_representation, h + no_film_aux["adapter_output"])

    no_adapter = SharedFiLMAdapter(8, adapter_ratio=0.5, dropout=0.0, use_film=True, use_adapter=False).eval()
    no_adapter_representation, no_adapter_aux = no_adapter(h, context)
    assert torch.equal(no_adapter_aux["adapter_output"], torch.zeros_like(h))
    assert torch.any(no_adapter_aux["gamma"] != 0)
    torch.testing.assert_close(no_adapter_representation, no_adapter.condition_norm((1 + no_adapter_aux["gamma"]) * h + no_adapter_aux["beta"]))

    both_off = SharedFiLMAdapter(8, adapter_ratio=0.5, dropout=0.0, use_film=False, use_adapter=False).eval()
    both_off_representation, both_off_aux = both_off(h, context)
    torch.testing.assert_close(both_off_representation, h)
    assert torch.equal(both_off_aux["gamma"], torch.zeros_like(h))
    assert torch.equal(both_off_aux["beta"], torch.zeros_like(h))
    assert torch.equal(both_off_aux["adapter_output"], torch.zeros_like(h))


def test_target_only_and_response_stacking_do_not_report_endpoint_routing():
    torch.manual_seed(19)
    tasks = ["task_a", "task_b", "task_c"]
    h = torch.randn(2, 8)
    response = torch.randn(2, 3, 1)

    target_only = RGCERTaskConditioner(
        tasks,
        8,
        use_factorized_prompt=False,
        router_dim=8,
        router_top_k=2,
        dropout=0.0,
        transfer_mechanism="target_only",
    ).eval()
    target_rep, target_diag = target_only(h, "task_a", response, return_aux=True)
    assert target_rep.shape == h.shape
    assert torch.all(target_diag.null_weight == 0)
    assert torch.all(target_diag.source_weights == 0)
    assert torch.all(target_diag.joint_source_weights == 0)
    assert torch.all(target_diag.response_profile == 0)

    stacking = RGCERTaskConditioner(
        tasks,
        8,
        use_factorized_prompt=False,
        router_dim=8,
        router_top_k=2,
        dropout=0.0,
        transfer_mechanism="response_stacking",
    ).eval()
    stacking_rep, stacking_diag = stacking(h, "task_a", response, return_aux=True)
    assert stacking_rep.shape == h.shape
    assert torch.all(stacking_diag.null_weight == 0)
    assert torch.all(stacking_diag.source_weights == 0)
    assert torch.all(stacking_diag.joint_source_weights == 0)
