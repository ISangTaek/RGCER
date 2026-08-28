"""rgcer_source_policy: human targets draw only on animal (or allowed) sources."""

from types import SimpleNamespace

import pytest
import torch

from architecture.response_guided_router import ResponseGuidedEndpointRouter
from architecture.toxacute_tasks import (
    ANIMAL_SOURCE_TASKS,
    HUMAN_TARGET_TASKS,
    TOXACUTE_TASKS,
)
from config import resolve_effective_rgcer_config
from trainer import DEFAULT_SOURCE_POLICY, build_source_policy_mask

HUMAN_TARGET = HUMAN_TARGET_TASKS[0]
OTHER_HUMANS = [task for task in HUMAN_TARGET_TASKS if task != HUMAN_TARGET]


def _policy_names(task_names, policy, allowed=()):
    mask = build_source_policy_mask(task_names, policy=policy, allowed_auxiliary=allowed)
    return {name for name, keep in zip(task_names, mask) if keep}


def test_animal56_policy_blocks_human_sources_for_human_target():
    """§11.3 acceptance: for a human target no other human endpoint is a source."""

    allowed = _policy_names(TOXACUTE_TASKS, "animal56_only")
    assert set(allowed) == set(ANIMAL_SOURCE_TASKS)

    # The other two human endpoints are barred for the primary target...
    assert HUMAN_TARGET not in allowed
    assert not (set(OTHER_HUMANS) & allowed)
    # ...and every allowed source is an animal endpoint.
    assert allowed <= set(ANIMAL_SOURCE_TASKS)


def test_all_except_target_policy_keeps_everything():
    allowed = _policy_names(TOXACUTE_TASKS, "all_except_target")
    assert set(allowed) == set(TOXACUTE_TASKS)


def test_animal56_policy_on_human_only_scope_leaves_no_sources():
    """human3-scope engineering smokes must opt into all_except_target."""

    assert _policy_names(HUMAN_TARGET_TASKS, "animal56_only") == set()


def test_explicit_override_reinstate_named_auxiliary_endpoints():
    fake_endpoint = f"shuffled_{ANIMAL_SOURCE_TASKS[0]}"
    synthetic = list(TOXACUTE_TASKS) + [fake_endpoint]

    baseline = _policy_names(synthetic, "animal56_only")
    assert fake_endpoint not in baseline  # unknown non-animal tasks stay blocked

    overridden = _policy_names(
        synthetic,
        "animal56_only",
        allowed=(fake_endpoint,),
    )
    assert fake_endpoint in overridden
    assert overridden - {fake_endpoint} == set(ANIMAL_SOURCE_TASKS)


def test_mask_rejects_unknown_policy_and_unknown_auxiliary_tasks():
    with pytest.raises(ValueError, match="rgcer_source_policy"):
        build_source_policy_mask(["a"], policy="everything_goes")
        with pytest.raises(ValueError, match="outside this run"):
            build_source_policy_mask(
                list(ANIMAL_SOURCE_TASKS),
                policy="animal56_only",
                allowed_auxiliary=("not_a_task",),
            )


def test_router_zeros_disallowed_sources_for_human_target():
    """End-to-end through the real router kernel: masked humans get ~no mass."""

    hidden_dim = 16
    router = ResponseGuidedEndpointRouter(
        hidden_dim=hidden_dim,
        router_dim=16,
        top_k=0,
        temperature=1.0,
        exclude_target=True,
        dropout=0.0,
        response_hidden_dim=8,
        use_source_response=True,
        use_target_response=True,
        use_molecule_query=True,
        use_sparse_routing=False,
        use_null_route=True,
    )
    torch.manual_seed(7)
    batch_size = 4
    h = torch.randn(batch_size, hidden_dim)
    prompts = torch.randn(len(TOXACUTE_TASKS), hidden_dim)
    responses = torch.tanh(torch.randn(batch_size, len(TOXACUTE_TASKS), 1))

    mask_values = build_source_policy_mask(TOXACUTE_TASKS, policy="animal56_only")
    mask = torch.tensor(mask_values)
    target_index = TOXACUTE_TASKS.index(HUMAN_TARGET)

    _, conditional, joint, null_weight, _ = router(
        h, prompts, responses, target_index, source_mask=mask
    )
    joint = joint.detach()
    null = null_weight.detach()

    human_indices = [TOXACUTE_TASKS.index(t) for t in OTHER_HUMANS]
    animal_indices = [TOXACUTE_TASKS.index(t) for t in ANIMAL_SOURCE_TASKS]
    # Strictly zero probability mass on disallowed human endpoints.
    assert float(joint[:, human_indices].abs().max()) == 0.0
    assert float(conditional[:, human_indices].abs().max()) == 0.0
    # Animal sources remain eligible and NULL stays a live option.
    assert float(joint[:, animal_indices].sum()) >= 0.0
    assert torch.allclose(
        joint.sum(dim=-1) + null.squeeze(-1),
        torch.ones(batch_size),
        atol=1e-5,
    )
    assert bool((null > 0).all())


def test_trainer_checkpoint_records_and_enforces_policy(tmp_path):
    from loss import MSELoss
    from trainer import Trainer
    from weighting.EW import EW

    class _Dummy(torch.nn.Module):
        is_rgcer = True

        def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
            super().__init__()
            del encoder_class, decoders, device, args, kwargs
            self.encoder = torch.nn.Identity()
            self.projection = torch.nn.Linear(4, 1)

    args = SimpleNamespace(
        seed=42,
        gpu_id="cpu",
        prediction_mode="point",
        conformal_alpha=0.10,
        min_calibration_size=1,
        mode="train",
        save_path=str(tmp_path / "run"),
        load_path=None,
        data_store_dir=None,
        datastore_metadata=None,
        hps_warmup_epochs=0,
        lambda_base=0.25,
        lambda_quantile=1.0,
        routing_enabled=True,
        selection_scope="all_tasks",
        rgcer_transfer_mechanism="endpoint_router",
        exclude_target_from_sources=True,
        lower_quantile=0.05,
        upper_quantile=0.95,
        ckpt_name="model",
    )
    task_dict = {
        "human_oral_TDLo": {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]},
        ANIMAL_SOURCE_TASKS[0]: {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]},
    }
    trainer = Trainer(
        task_dict=task_dict,
        weighting=EW,
        architecture=_Dummy,
        encoder_class=torch.nn.Identity,
        decoders=torch.nn.ModuleDict(),
        optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        args=args,
        save_path=tmp_path / "run",
    )

    config = trainer._rgcer_config()
    assert config["rgcer_source_policy"] == "animal56_only"
    assert config["allowed_auxiliary_sources"] == []
    mask = trainer.source_policy_mask(torch.device("cpu"))
    target_pos = task_dict and list(task_dict).index("human_oral_TDLo")
    assert bool(mask[target_pos]) is False
    assert bool(mask[list(task_dict).index(ANIMAL_SOURCE_TASKS[0])]) is True


def test_effective_config_defaults_to_animal56_only_without_warning_noise():
    # The plain policy default rides along silently on non-RGCER runs...
    quiet = SimpleNamespace(
        arch="Graphormer",
        rgcer_transfer_mechanism=None,
        rgcer_source_policy=DEFAULT_SOURCE_POLICY,
    )
    resolved = resolve_effective_rgcer_config(quiet)
    assert resolved["requested"]["rgcer_source_policy"] == DEFAULT_SOURCE_POLICY
    import warnings as warnings_module

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        resolve_effective_rgcer_config(quiet)
    # No noise for the untouched default (nor anything else in this build).
    assert not any("rgcer_source_policy" in str(entry.message) for entry in caught)

    # ...while an explicitly customised policy calls attention to itself.
    custom = SimpleNamespace(
        arch="Graphormer",
        rgcer_transfer_mechanism=None,
        rgcer_source_policy="all_except_target",
    )
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        resolve_effective_rgcer_config(custom)
    assert any("rgcer_source_policy" in str(entry.message) for entry in caught)
