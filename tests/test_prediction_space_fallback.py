import torch

from architecture.prediction_heads import (
    DecodedPrediction,
    blend_decoded_predictions,
    decode_prediction,
    encode_decoded_prediction,
)


def _decoded_point(median):
    return DecodedPrediction(median=median, lower=median, upper=median)


def test_point_prediction_space_blend_extremes_and_middle():
    base = _decoded_point(torch.tensor([[1.0], [2.0]]))
    route = _decoded_point(torch.tensor([[5.0], [8.0]]))

    for weight, expected in (
        (0.0, route.median),
        (1.0, base.median),
        (0.25, 0.25 * base.median + 0.75 * route.median),
    ):
        final = blend_decoded_predictions(base, route, torch.full((2, 1), weight))
        torch.testing.assert_close(final.median, expected)
        torch.testing.assert_close(encode_decoded_prediction(final, "point"), expected)


def test_quantile_prediction_space_blend_preserves_order_and_round_trip():
    base_raw = torch.tensor([[0.5, -0.2, 0.3], [1.0, 0.4, -0.1]])
    route_raw = torch.tensor([[2.0, 0.1, -0.4], [-1.0, -0.3, 0.7]])
    base = decode_prediction(base_raw, "quantile")
    route = decode_prediction(route_raw, "quantile")
    weight = torch.tensor([[0.25], [0.5]])
    final = blend_decoded_predictions(base, route, weight)

    torch.testing.assert_close(
        final.median,
        weight * base.median + (1.0 - weight) * route.median,
    )
    torch.testing.assert_close(
        final.lower,
        weight * base.lower + (1.0 - weight) * route.lower,
    )
    torch.testing.assert_close(
        final.upper,
        weight * base.upper + (1.0 - weight) * route.upper,
    )
    assert torch.all(final.lower <= final.median)
    assert torch.all(final.median <= final.upper)

    encoded = encode_decoded_prediction(final, "quantile")
    round_trip = decode_prediction(encoded, "quantile")
    torch.testing.assert_close(round_trip.median, final.median, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(round_trip.lower, final.lower, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(round_trip.upper, final.upper, atol=1e-6, rtol=1e-6)
