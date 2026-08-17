import pytest
import torch

from architecture.response_guided_router import ResponseGuidedEndpointRouter


def _router(**overrides):
    settings = {
        "hidden_dim": 12,
        "router_dim": 8,
        "top_k": 2,
        "temperature": 1.0,
        "exclude_target": True,
        "dropout": 0.0,
    }
    settings.update(overrides)
    return ResponseGuidedEndpointRouter(**settings).eval()


def _inputs():
    torch.manual_seed(11)
    return torch.randn(4, 12), torch.randn(5, 12), torch.randn(4, 5, 1)


def test_source_response_off_and_target_response_off_ignore_response_changes():
    h, prompts, response = _inputs()
    changed = response + torch.randn_like(response) * 4.0
    router = _router(use_source_response=False, use_target_response=False)
    first = router(h, prompts, response, target_index=2)
    second = router(h, prompts, changed, target_index=2)
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])
    torch.testing.assert_close(first[3], second[3])


def test_target_response_off_ignores_target_response_change():
    h, prompts, response = _inputs()
    changed = response.clone()
    changed[:, 2, :] += 7.0
    router = _router(use_target_response=False)
    first = router(h, prompts, response, target_index=2)
    second = router(h, prompts, changed, target_index=2)
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])
    torch.testing.assert_close(first[3], second[3])


def test_molecule_query_off_makes_routing_invariant_to_molecule_representation():
    h, prompts, response = _inputs()
    router = _router(
        use_molecule_query=False,
        use_source_response=False,
        use_target_response=False,
    )
    first = router(h, prompts, response, target_index=2)
    second = router(-h, prompts, response, target_index=2)
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])


def test_dense_ablation_can_keep_more_than_top_k_sources():
    h, prompts, response = _inputs()
    router = _router(use_sparse_routing=False, use_null_route=False)
    _, source, joint, null, _ = router(h, prompts, response, target_index=2)
    assert torch.all(null == 0)
    assert torch.allclose(source.sum(dim=-1), torch.ones(h.size(0)))
    assert torch.allclose(joint.sum(dim=-1), torch.ones(h.size(0)))
    assert torch.all((source > 0).sum(dim=-1) > 2)


def test_null_off_has_no_silent_hps_fallback_and_null_on_handles_no_sources():
    h, prompts, response = _inputs()
    no_null = _router(use_null_route=False)
    _, source, joint, null, _ = no_null(
        h, prompts, response, target_index=2, source_mask=torch.ones(5, dtype=torch.bool)
    )
    assert torch.all(null == 0)
    assert torch.allclose(source.sum(dim=-1), torch.ones(h.size(0)))
    assert torch.allclose(joint.sum(dim=-1), torch.ones(h.size(0)))
    with pytest.raises(ValueError, match="NULL route"):
        no_null(h, prompts, response, target_index=2, source_mask=torch.zeros(5, dtype=torch.bool))

    with_null = _router(use_null_route=True)
    _, source, joint, null, _ = with_null(
        h, prompts, response, target_index=2, source_mask=torch.zeros(5, dtype=torch.bool)
    )
    assert torch.all(source == 0)
    assert torch.all(joint == 0)
    assert torch.allclose(null, torch.ones_like(null))
