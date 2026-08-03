import torch
import pytest

from architecture.response_guided_router import ResponseGuidedEndpointRouter


def build_router():
    torch.manual_seed(3)
    return ResponseGuidedEndpointRouter(
        hidden_dim=12,
        router_dim=8,
        top_k=2,
        temperature=1.0,
        exclude_target=True,
        dropout=0.0,
    ).eval()


def test_response_router_shapes_dynamicity_and_null():
    router = build_router()
    h = torch.randn(5, 12)
    prompts = torch.randn(4, 12)
    responses = torch.randn(5, 4, 1)
    result = router(h, prompts, responses, target_index=3)
    context, source, joint, null, entropy = result
    assert context.shape == (5, 12)
    assert source.shape == joint.shape == (5, 4)
    assert null.shape == (5, 1)
    assert entropy.shape == (5,)
    assert torch.allclose(source.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(joint.sum(dim=-1) + null.squeeze(-1), torch.ones(5), atol=1e-6)
    assert torch.all((joint[:, 3] == 0))
    assert torch.all((joint > 0).sum(dim=-1) <= 2)
    _, source_other, _, _, _ = router(-h, prompts, responses, target_index=3)
    assert not torch.allclose(source, source_other)


def test_source_masks_T_and_BT_and_all_masked_null():
    router = build_router()
    h = torch.randn(3, 12)
    prompts = torch.randn(4, 12)
    responses = torch.randn(3, 4, 1)
    mask_t = torch.tensor([True, True, False, True])
    _, _, joint_t, null_t, _ = router(h, prompts, responses, 3, source_mask=mask_t)
    mask_bt = mask_t.unsqueeze(0).expand(3, -1).clone()
    _, _, joint_bt, null_bt, _ = router(h, prompts, responses, 3, source_mask=mask_bt)
    assert torch.allclose(joint_t, joint_bt)
    assert torch.allclose(null_t, null_bt)
    _, source, joint, null, _ = router(
        h, prompts, responses, 3, source_mask=torch.zeros(4, dtype=torch.bool)
    )
    assert torch.equal(source, torch.zeros_like(source))
    assert torch.equal(joint, torch.zeros_like(joint))
    assert torch.allclose(null, torch.ones_like(null))
    with pytest.raises(ValueError):
        router(h, prompts, responses, 3, source_mask=torch.ones(2, 2, dtype=torch.bool))
