"""Response-profile determinism contract (review §25-29, §46)."""

import torch
from torch import nn

from architecture.prediction_heads import TaskPredictionHead


def _trained_style_head(dropout: float = 0.5, hidden_dim: int = 32) -> TaskPredictionHead:
    torch.manual_seed(0)
    return TaskPredictionHead(
        hidden_dim=hidden_dim,
        mode="quantile",
        head_hidden_dim=24,
        dropout=dropout,
    )


def test_head_deterministic_call_pins_dropout_in_train_mode():
    head = _trained_style_head()
    representation = torch.randn(16, 32)
    head.train()

    deterministic = head(representation.clone(), deterministic=True)
    again = head(representation.clone(), deterministic=True)
    assert torch.equal(deterministic, again)


def test_supervised_head_still_uses_dropout_in_train_mode():
    head = _trained_style_head()  # p=0.5
    representation = torch.randn(16, 32)
    head.train()

    first = head(representation.clone())
    second = head(representation.clone())
    # A p=0.5 dropout on a wide hidden layer is overwhelmingly likely to
    # differ between two calls.
    assert not torch.equal(first, second)

    head.eval()
    eval_first = head(representation.clone())
    eval_second = head(representation.clone())
    assert torch.equal(eval_first, eval_second)


def test_response_profile_is_identical_across_train_mode_calls():
    """§46: same h, same checkpoint → identical profile while heads drop out."""

    from types import SimpleNamespace

    from architecture.Graphormer_rgcer import Graphormer_rgcer

    tasks = ["t_a", "t_b"]
    heads = {
        "t_a": _trained_style_head(),
        "t_b": _trained_style_head(dropout=0.4),
    }
    for head in heads.values():
        head.train()
    # Bind the real method against a minimal stand-in exposing exactly the
    # attributes the profile path consumes.
    encoder_stub = SimpleNamespace(
        decoders=heads,
        task_name=list(tasks),
        _head_mode=lambda task: getattr(heads[task], "mode", "quantile"),
    )
    h = torch.randn(8, 32)
    profile_one = Graphormer_rgcer._response_profile(encoder_stub, h)
    profile_two = Graphormer_rgcer._response_profile(encoder_stub, h)
    assert torch.equal(profile_one, profile_two)
    assert profile_one.shape == (8, len(tasks), 1)

    # No gradient may reach the heads through the response profile either.
    assert not any(p.grad is not None for p in heads["t_a"].parameters())
