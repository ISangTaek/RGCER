from main import build_parser


def test_rgcer_flags_default_to_full_model_semantics():
    params = build_parser().parse_args([])
    assert params.rgcer_use_source_response is True
    assert params.rgcer_use_target_response is True
    assert params.rgcer_use_molecule_query is True
    assert params.rgcer_use_sparse_routing is True
    assert params.rgcer_use_null_route is True
    assert params.rgcer_use_film is True
    assert params.rgcer_use_adapter is True
    assert params.rgcer_use_base_aux_loss is True
    assert params.rgcer_fallback_space == "prediction"
    assert params.rgcer_transfer_mechanism == "endpoint_router"
    assert params.experiment_tag == "full"


def test_rgcer_boolean_flags_can_be_disabled_independently():
    params = build_parser().parse_args(
        [
            "--no-rgcer_use_source_response",
            "--no-rgcer_use_null_route",
            "--no-rgcer_use_film",
        ]
    )
    assert params.rgcer_use_source_response is False
    assert params.rgcer_use_null_route is False
    assert params.rgcer_use_film is False
    assert params.rgcer_use_target_response is True
    assert params.rgcer_use_adapter is True
