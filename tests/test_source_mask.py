import torch

from architecture.response_guided_router import ResponseGuidedEndpointRouter


def test_source_mask_is_per_sample():
    router = ResponseGuidedEndpointRouter(
        8,
        router_dim=8,
        top_k=0,
        use_sparse_routing=False,
        dropout=0.0,
    ).eval()
    h = torch.randn(2, 8)
    prompts = torch.randn(3, 8)
    response = torch.randn(2, 3, 1)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    _, _, joint, _, _ = router(h, prompts, response, 2, source_mask=mask)
    assert joint[0, 1] == 0
    assert joint[1, 0] == 0
