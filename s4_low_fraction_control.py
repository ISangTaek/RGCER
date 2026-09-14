"""Fail-closed controls for S4B low-fraction best-checkpoint inference.

This module is deliberately independent from the immutable S3C3 fifteen-asset
allowlist.  It validates the exact Codex-frozen S4B policy bytes, restores each
of the forty low-fraction checkpoints without metadata migration, validates
worker output against independently produced sample tables, and verifies the
ten inherited 100% aliases from the accepted 037 archive.

No function in this module trains, calibrates, rewrites a checkpoint, or opens
the real test/calibration split unless an explicit caller supplies the already
authorized server inputs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


HUMAN3_TASKS = (
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
)
S4B_TASK_ID = "S4B_LOW_FRACTION_STRICT_INFERENCE_20260914"
S4B_POLICY_SHA256 = (
    "9ea10d7341a0af11e7225c61cdbaad2a9be6e4aa115430039bf6d6796e80f7b5"
)
S4B_POLICY_ID = "S4B_LOW_FRACTION_STRICT_NO_MIGRATION_20260914"
S4B_BASE_COMMIT = "c4bdca6bead67351c5b7b96213586ffb55d50209"
S4A_OUTER_SHA256 = (
    "6f02e411d034b5be6dd92e9e90d157045017162127e364f60cb9c1118364429f"
)
S4A_CANONICAL_SHA256 = (
    "a3afa3886ad4d2005a07bbd436549be71b788311810e142bda9e7f8a03aa7edb"
)
CORE_POLICY_SHA256 = (
    "522ea73de9343b268c89197141221ed4fc87f87ea48eef24bc951831cbafae91"
)
S3C3_OUTER_SHA256 = (
    "314e8532c0d0a99c3a12703b3deaa612a0cdc877649b903c72172277ea49a85c"
)
S3C3_CANONICAL_SHA256 = (
    "44b428a26ed2ddc505ec101b1b302123c81785f01890c9a8ddd12636387ed3aa"
)
S3C3_CANONICAL_MEMBER = (
    "037_2026-09-14_S3C3R2核心恢复与test闭环/回传/"
    "S3C3R2_SERVER_20260914T013423Z.zip"
)
FROZEN_TEST_SAMPLE_PROJECTION_SHA256 = (
    "4c41890584c0af606efd9050e6f9355f4582128b759a6b25ddd682ce2c5ef258"
)

LOW_FRACTIONS = (10, 25, 50, 75)
METHODS = ("B1", "RPT")
SEEDS = (42, 43, 44, 45, 46)
ALLOWED_OPERATION = "batch_inference"
ALLOWED_SPLITS = ("validation", "test")
S4B_MATRIX = frozenset(
    (method, seed, fraction)
    for method in METHODS
    for seed in SEEDS
    for fraction in LOW_FRACTIONS
)
S4B_RUN_IDS = tuple(
    f"scaling_{method}_f{fraction}_s{seed}"
    for fraction in LOW_FRACTIONS
    for seed in SEEDS
    for method in METHODS
)

ADAPTATION_FIELDS = (
    "d6_candidate",
    "d7_candidate",
    "freeze_backbone_epochs",
    "backbone_lr_multiplier",
    "trainable_last_blocks",
    "retention_probe_per_task",
    "retention_damage_threshold",
    "feature_drift_probe_size",
    "feature_drift_epochs",
)
CONFIGURATION_FIELDS = (
    "arch",
    "dataset",
    "splitting",
    "split_seed",
    "toxacute_task_scope",
    "prediction_mode",
    "train_eval_scope",
    "fit_conformal",
    "epochs",
    "data_store_dir",
    "seed",
    "train_fraction",
    "init_state_path",
)
ARCHITECTURE_FIELDS = (
    "architecture",
    "task_names",
    "prediction_mode",
    "edge_bias_mode",
    "spatial_pos_max_clip",
    "hidden_dim",
    "a_layers",
    "a_heads",
    "mid_dim",
    "head_hidden_dim",
    "head_dropout",
    "adapter_ratio",
    "response_hidden_dim",
    "task_sampling",
    "conformal_scope",
    "router_top_k",
    "router_temperature",
    "exclude_target_from_sources",
    "use_factorized_prompt",
    "card_enabled",
    "card_lambda_delta",
    "card_bottleneck",
    "task_metadata",
    *ADAPTATION_FIELDS,
)
DATA_FIELDS = (
    "datastore_format_version",
    "datastore_build_id",
    "datastore_fingerprint",
    "raw_csv_sha256",
    "feature_schema_version",
    "max_path_distance",
    "splitting",
    "split_seed",
    "split_ratios",
    "split_manifest_hash",
    "max_nodes_filter",
    "task_names",
)
REPRODUCIBILITY_FIELDS = (
    "base_seed",
    "seed_policy_version",
    "epoch_seed_scheme",
    "loader_seed_scheme",
    "persistent_worker_policy",
    "initial_model_sha256",
)
S4A_FIELDS = (
    "checkpoint_version",
    "epoch",
    "configuration",
    "architecture_config",
    "task_names",
    "prediction_mode",
    "quantile_config",
    "task_scalers",
    "data_config",
    "split_manifest_sha256",
    "feature_schema_version",
    "rgcer_config",
    "effective_rgcer_config",
    "routing_enabled",
    "selection_state",
    "reproducibility",
)


ALIAS_PREDICTION_SHA256 = MappingProxyType(
    {
        ("B1", 42): "b8f81adf815001d4699f0e257ffecb1c59a8c6df36a94e44200d87070915b452",
        ("B1", 43): "348f8964cb2ffeb1794ee908f781e9cb9fd9be315277a8d8c0f892ecfbf33038",
        ("B1", 44): "75b73fc2fd5c8cf6d4d1e6663fe6277324dbbb229f6b2fef6608810cdbe46eea",
        ("B1", 45): "b116a1e4ec4c8e23621b6ac0dffc56a0629e2364918ac050eb95dcdc6cf60a91",
        ("B1", 46): "47d7f988db56dc38b942c034d46502f653eddbab05d191e15c0280417d24b002",
        ("RPT", 42): "b3f97ee9eb8214cd063358f07808926ce8c1b26daa2a66442b5382a9b4ff4bf8",
        ("RPT", 43): "7290e0b9f3ed114eaa176d77a7223ccbf7050bdb63c60c233276f4a7ef22e7f3",
        ("RPT", 44): "679e439c5853041c04b12e4c15b731667fd8dcd409a309d8cf3b616e52756eeb",
        ("RPT", 45): "0c534ed58c2c6f1b912c63da50e0c57ee4486af79e24181bb4dc647850d2d4a2",
        ("RPT", 46): "c0a7fbdb855f054412bad1c10283a57a640a575386c0e73831871d412f9a3d2d",
    }
)
ALIAS_METRICS_SHA256 = MappingProxyType(
    {
        ("B1", 42): "8eb56d44e96e5e394f5ba939a397c021d298853ef1dbfb2b27996f00ec6f226f",
        ("B1", 43): "03842b7e94e8d7dad227bf1e7d731b8922b4e7563e992f031b9ed9b534384785",
        ("B1", 44): "bcdfc2621ef3954018a7365ecd19df3113a2641bacd7b599dc836c155a26e9c4",
        ("B1", 45): "d81a63a7bc6ccd61555984f752b2cdbb3fa9f105366891086e2fa3fa251b0264",
        ("B1", 46): "410cfbb05a174ed5b7f5d3f500aa6ae8b89640fd1bafe5b356af54af0d7bfef0",
        ("RPT", 42): "e2c8ad103e61c7732229c89ebebe994700522e35162f3562208a4bb59c6e2c09",
        ("RPT", 43): "904e483acfa56f3e525f4fd8a316e2061978893b0ef8042f7e40c1bf83fc12d3",
        ("RPT", 44): "4ff5815f7012f1647209ccf3432e38a8c84d8407d5a34f5563544145c17bc0d8",
        ("RPT", 45): "2b363b0ed30f8d7fa3ade86d896d58ce13020d2fdbb6a24bd190a03a2628cda8",
        ("RPT", 46): "962e317c7dad7c49ac43071ee72064cec7a8be7f9c7809dd9dd613a91e8f93d7",
    }
)


class S4BControlError(ValueError):
    """An S4B identity, authorization, sample, metric, or archive failed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_identity(
    path: str | Path, *, expected_size_bytes: int, expected_sha256: str
) -> dict[str, Any]:
    """Verify size before hashing; callers may import torch only afterwards."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    _true_int(expected_size_bytes, "expected_size_bytes", minimum=1)
    if type(expected_sha256) is not str or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise S4BControlError("expected_sha256 is invalid")
    actual_size = path.stat().st_size
    if actual_size != expected_size_bytes:
        raise S4BControlError("file size differs before content load")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise S4BControlError("file SHA differs before content load")
    return {"size_bytes": actual_size, "sha256": actual_sha}


def strict_json_loads(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise S4BControlError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str):
        raise S4BControlError(f"non-finite JSON value: {value}")

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=reject_nonfinite,
    )


def read_json(path: str | Path) -> Any:
    try:
        return strict_json_loads(Path(path).read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise S4BControlError(f"JSON is not UTF-8: {path}") from exc


def identity_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    if type(expected) is float:
        return math.isfinite(actual) and actual == expected
    return actual == expected


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise S4BControlError(f"{context} must be a mapping")
    return value


def _exact_keys(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    if set(mapping) != set(keys):
        missing = sorted(set(keys) - set(mapping))
        extra = sorted(set(mapping) - set(keys))
        raise S4BControlError(
            f"{context} schema differs; missing={missing}, extra={extra}"
        )


def _require_exact(
    mapping: Mapping[str, Any], name: str, expected: Any, context: str
) -> None:
    if name not in mapping:
        raise S4BControlError(f"{context}.{name} is missing")
    if not _strict_equal(mapping[name], expected):
        raise S4BControlError(
            f"{context}.{name} differs from the frozen value/type "
            f"({mapping[name]!r} != {expected!r})"
        )


def _true_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise S4BControlError(f"{context} is not an authorized true integer")
    return value


def _finite_float(
    value: Any, context: str, *, positive: bool = False
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise S4BControlError(f"{context} is not a finite float")
    if positive and value <= 0:
        raise S4BControlError(f"{context} must be positive")
    return value


def _contains_unsupported_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "unsupported_type" in value:
            return True
        return any(_contains_unsupported_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unsupported_placeholder(item) for item in value)
    return False


@dataclass(frozen=True)
class LowFractionAsset:
    run_id: str
    method: str
    seed: int
    fraction_percent: int
    run_dir: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    best_epoch: int
    best_validation_macro_rmse: float
    expected_configuration: Mapping[str, Any]
    expected_adaptation: Mapping[str, Any]
    expected_architecture: Mapping[str, Any]
    expected_data_config: Mapping[str, Any]
    expected_scalers: Mapping[str, Any]
    expected_reproducibility: Mapping[str, Any]
    expected_model_tensor_shapes: Mapping[str, Any]
    provenance: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.method, self.seed, self.fraction_percent


@dataclass(frozen=True)
class LowFractionPolicy:
    path: str
    sha256: str
    policy_id: str
    base_commit: str
    assets: tuple[LowFractionAsset, ...]
    raw: Mapping[str, Any]

    def asset(self, run_id: str) -> LowFractionAsset:
        matches = [asset for asset in self.assets if asset.run_id == run_id]
        if len(matches) != 1:
            raise S4BControlError(
                f"run_id {run_id!r} is not uniquely authorized by S4B policy"
            )
        return matches[0]


def assert_core_policy_unchanged() -> None:
    from checkpoint_inference_compatibility import (
        PRODUCTION_POLICY,
        PRODUCTION_POLICY_SHA256,
    )

    if (
        PRODUCTION_POLICY_SHA256 != CORE_POLICY_SHA256
        or PRODUCTION_POLICY.computed_sha256() != CORE_POLICY_SHA256
    ):
        raise S4BControlError("the frozen S3C3 core policy identity changed")


def _validate_policy_asset(raw: Any) -> LowFractionAsset:
    asset = _mapping(raw, "policy.asset")
    _exact_keys(
        asset,
        (
            "run_id",
            "method",
            "seed",
            "fraction_percent",
            "run_dir",
            "checkpoint",
            "best_epoch",
            "best_validation_macro_rmse",
            "migration_required",
            "expected_configuration",
            "expected_adaptation",
            "expected_architecture",
            "expected_data_config",
            "expected_scalers",
            "expected_reproducibility",
            "expected_model_tensor_shapes",
            "provenance",
        ),
        "policy.asset",
    )
    run_id = asset["run_id"]
    method = asset["method"]
    seed = asset["seed"]
    fraction = asset["fraction_percent"]
    if type(run_id) is not str or not run_id:
        raise S4BControlError("policy asset run_id is invalid")
    if method not in METHODS or type(method) is not str:
        raise S4BControlError(f"policy method is invalid: {run_id}")
    _true_int(seed, f"{run_id}.seed")
    _true_int(fraction, f"{run_id}.fraction_percent")
    expected_run_id = f"scaling_{method}_f{fraction}_s{seed}"
    if run_id != expected_run_id or (method, seed, fraction) not in S4B_MATRIX:
        raise S4BControlError(f"policy matrix identity differs: {run_id}")
    if type(asset["run_dir"]) is not str or not asset["run_dir"]:
        raise S4BControlError(f"policy run_dir is invalid: {run_id}")
    if asset["migration_required"] is not False:
        raise S4BControlError(f"S4B migration must remain disabled: {run_id}")
    _true_int(asset["best_epoch"], f"{run_id}.best_epoch", minimum=0)
    _finite_float(
        asset["best_validation_macro_rmse"],
        f"{run_id}.best_validation_macro_rmse",
        positive=True,
    )

    checkpoint = _mapping(asset["checkpoint"], f"{run_id}.checkpoint")
    _exact_keys(checkpoint, ("path", "sha256", "size_bytes"), f"{run_id}.checkpoint")
    if type(checkpoint["path"]) is not str or not checkpoint["path"].endswith("_best.pt"):
        raise S4BControlError(f"checkpoint path is invalid: {run_id}")
    if type(checkpoint["sha256"]) is not str or re.fullmatch(
        r"[0-9a-f]{64}", checkpoint["sha256"]
    ) is None:
        raise S4BControlError(f"checkpoint SHA is invalid: {run_id}")
    _true_int(checkpoint["size_bytes"], f"{run_id}.checkpoint.size_bytes", minimum=1)

    configuration = _mapping(
        asset["expected_configuration"], f"{run_id}.expected_configuration"
    )
    _exact_keys(configuration, CONFIGURATION_FIELDS, f"{run_id}.expected_configuration")
    _require_exact(configuration, "seed", seed, f"{run_id}.expected_configuration")
    _require_exact(
        configuration,
        "train_fraction",
        fraction / 100.0,
        f"{run_id}.expected_configuration",
    )
    if type(configuration["init_state_path"]) is not str:
        raise S4BControlError(f"{run_id}.init_state_path must remain a string")

    adaptation = _mapping(
        asset["expected_adaptation"], f"{run_id}.expected_adaptation"
    )
    _exact_keys(adaptation, ADAPTATION_FIELDS, f"{run_id}.expected_adaptation")
    expected_method_adaptation = (
        ("b1", "none", 0) if method == "B1" else ("none", "s1", 40)
    )
    for name, expected in zip(
        ("d6_candidate", "d7_candidate", "freeze_backbone_epochs"),
        expected_method_adaptation,
    ):
        _require_exact(adaptation, name, expected, f"{run_id}.expected_adaptation")
    _require_exact(
        adaptation,
        "feature_drift_epochs",
        "0,5,10,15,19",
        f"{run_id}.expected_adaptation",
    )

    architecture = _mapping(
        asset["expected_architecture"], f"{run_id}.expected_architecture"
    )
    _exact_keys(architecture, ARCHITECTURE_FIELDS, f"{run_id}.expected_architecture")
    for name in ADAPTATION_FIELDS:
        _require_exact(
            architecture, name, adaptation[name], f"{run_id}.expected_architecture"
        )
    _require_exact(
        architecture,
        "task_names",
        list(HUMAN3_TASKS),
        f"{run_id}.expected_architecture",
    )

    data_config = _mapping(
        asset["expected_data_config"], f"{run_id}.expected_data_config"
    )
    _exact_keys(data_config, DATA_FIELDS, f"{run_id}.expected_data_config")
    _require_exact(
        data_config,
        "task_names",
        list(HUMAN3_TASKS),
        f"{run_id}.expected_data_config",
    )

    scalers = _mapping(asset["expected_scalers"], f"{run_id}.expected_scalers")
    _exact_keys(scalers, HUMAN3_TASKS, f"{run_id}.expected_scalers")
    for task in HUMAN3_TASKS:
        scaler = _mapping(scalers[task], f"{run_id}.expected_scalers.{task}")
        _exact_keys(scaler, ("mean", "std", "count"), f"{run_id}.expected_scalers.{task}")
        _finite_float(scaler["mean"], f"{run_id}.{task}.mean")
        _finite_float(scaler["std"], f"{run_id}.{task}.std", positive=True)
        _true_int(scaler["count"], f"{run_id}.{task}.count", minimum=1)

    reproducibility = _mapping(
        asset["expected_reproducibility"], f"{run_id}.expected_reproducibility"
    )
    _exact_keys(
        reproducibility,
        REPRODUCIBILITY_FIELDS,
        f"{run_id}.expected_reproducibility",
    )
    _require_exact(
        reproducibility, "base_seed", seed, f"{run_id}.expected_reproducibility"
    )
    if type(reproducibility["initial_model_sha256"]) is not str or re.fullmatch(
        r"[0-9a-f]{64}", reproducibility["initial_model_sha256"]
    ) is None:
        raise S4BControlError(f"{run_id}.initial_model_sha256 is invalid")

    tensors = _mapping(
        asset["expected_model_tensor_shapes"],
        f"{run_id}.expected_model_tensor_shapes",
    )
    if len(tensors) != 170:
        raise S4BControlError(f"{run_id} does not freeze exactly 170 model tensors")
    for name, signature in tensors.items():
        if type(name) is not str or not name:
            raise S4BControlError(f"{run_id} has an invalid tensor name")
        signature = _mapping(signature, f"{run_id}.tensor.{name}")
        _exact_keys(signature, ("shape", "dtype"), f"{run_id}.tensor.{name}")
        shape = signature["shape"]
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise S4BControlError(f"{run_id}.tensor.{name}.shape is invalid")
        if signature["dtype"] != "torch.float32":
            raise S4BControlError(f"{run_id}.tensor.{name}.dtype is not torch.float32")

    provenance = _mapping(asset["provenance"], f"{run_id}.provenance")
    _exact_keys(
        provenance,
        ("canonical_sha256", "metadata_member", "metadata_sha256"),
        f"{run_id}.provenance",
    )
    _require_exact(
        provenance, "canonical_sha256", S4A_CANONICAL_SHA256, f"{run_id}.provenance"
    )
    _require_exact(
        provenance,
        "metadata_member",
        f"{run_id}/checkpoint_metadata.json",
        f"{run_id}.provenance",
    )
    if type(provenance["metadata_sha256"]) is not str or re.fullmatch(
        r"[0-9a-f]{64}", provenance["metadata_sha256"]
    ) is None:
        raise S4BControlError(f"{run_id}.metadata_sha256 is invalid")

    return LowFractionAsset(
        run_id=run_id,
        method=method,
        seed=seed,
        fraction_percent=fraction,
        run_dir=asset["run_dir"],
        checkpoint_path=checkpoint["path"],
        checkpoint_sha256=checkpoint["sha256"],
        checkpoint_size_bytes=checkpoint["size_bytes"],
        best_epoch=asset["best_epoch"],
        best_validation_macro_rmse=asset["best_validation_macro_rmse"],
        expected_configuration=dict(configuration),
        expected_adaptation=dict(adaptation),
        expected_architecture=dict(architecture),
        expected_data_config=dict(data_config),
        expected_scalers=dict(scalers),
        expected_reproducibility=dict(reproducibility),
        expected_model_tensor_shapes=dict(tensors),
        provenance=dict(provenance),
        raw=dict(asset),
    )


def load_s4b_policy(path: str | Path) -> LowFractionPolicy:
    policy_path = Path(path).resolve()
    raw_bytes = policy_path.read_bytes()
    digest = sha256_bytes(raw_bytes)
    if digest != S4B_POLICY_SHA256:
        raise S4BControlError("S4B policy bytes differ from the frozen SHA256")
    document = strict_json_loads(raw_bytes.decode("utf-8-sig"))
    document = _mapping(document, "policy")
    _exact_keys(
        document,
        (
            "schema_version",
            "policy_id",
            "status",
            "base_commit",
            "source_outer_sha256",
            "source_canonical_sha256",
            "task_names",
            "mode",
            "allow_architecture_additions",
            "allow_checkpoint_rewrite",
            "core_policy_must_remain_unchanged",
            "expected_test_sample_projection_sha256",
            "assets",
        ),
        "policy",
    )
    fixed = {
        "schema_version": 1,
        "policy_id": S4B_POLICY_ID,
        "status": "SCIENTIFIC_RULES_FROZEN_IMPLEMENTATION_REQUIRED",
        "base_commit": S4B_BASE_COMMIT,
        "source_outer_sha256": S4A_OUTER_SHA256,
        "source_canonical_sha256": S4A_CANONICAL_SHA256,
        "task_names": list(HUMAN3_TASKS),
        "mode": "strict_original_metadata",
        "allow_architecture_additions": False,
        "allow_checkpoint_rewrite": False,
        "core_policy_must_remain_unchanged": CORE_POLICY_SHA256,
        "expected_test_sample_projection_sha256": FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
    }
    for name, expected in fixed.items():
        _require_exact(document, name, expected, "policy")
    raw_assets = document["assets"]
    if not isinstance(raw_assets, list) or len(raw_assets) != 40:
        raise S4BControlError("S4B policy must contain exactly forty assets")
    assets = tuple(_validate_policy_asset(item) for item in raw_assets)
    if [asset.run_id for asset in assets] != list(S4B_RUN_IDS):
        raise S4BControlError("S4B policy asset order/coverage differs")
    if {asset.identity for asset in assets} != S4B_MATRIX:
        raise S4BControlError("S4B policy identity matrix differs")
    if len({asset.checkpoint_sha256 for asset in assets}) != 40:
        raise S4BControlError("S4B checkpoint identities are not unique")
    by_pair: dict[tuple[int, int], dict[str, LowFractionAsset]] = {}
    for asset in assets:
        by_pair.setdefault((asset.seed, asset.fraction_percent), {})[asset.method] = asset
    for pair, methods in by_pair.items():
        if set(methods) != set(METHODS) or not _strict_equal(
            dict(methods["B1"].expected_scalers),
            dict(methods["RPT"].expected_scalers),
        ):
            raise S4BControlError(f"B1/RPT scaler pair differs for seed/fraction {pair}")
    assert_core_policy_unchanged()
    return LowFractionPolicy(
        path=str(policy_path),
        sha256=digest,
        policy_id=document["policy_id"],
        base_commit=document["base_commit"],
        assets=assets,
        raw=dict(document),
    )


def validate_request(
    policy: LowFractionPolicy,
    *,
    run_id: Any,
    method: Any,
    seed: Any,
    fraction_percent: Any,
    operation: Any,
    split: Any,
    expected_policy_sha256: Any,
) -> LowFractionAsset:
    if operation != ALLOWED_OPERATION or type(operation) is not str:
        raise S4BControlError("S4B authorizes batch_inference only")
    if split not in ALLOWED_SPLITS or type(split) is not str:
        raise S4BControlError("S4B authorizes validation/test splits only")
    if expected_policy_sha256 != S4B_POLICY_SHA256 or type(expected_policy_sha256) is not str:
        raise S4BControlError("requested S4B policy SHA differs")
    if type(run_id) is not str or type(method) is not str:
        raise S4BControlError("run_id/method types are invalid")
    if type(seed) is not int or type(fraction_percent) is not int:
        raise S4BControlError("seed/fraction types are invalid")
    asset = policy.asset(run_id)
    if (method, seed, fraction_percent) != asset.identity:
        raise S4BControlError("requested method/seed/fraction differs from policy asset")
    return asset


def _field_value(fields: Mapping[str, Any], name: str, *, present: bool = True) -> Any:
    projection = _mapping(fields.get(name), f"metadata.fields.{name}")
    _exact_keys(projection, ("present", "value"), f"metadata.fields.{name}")
    if type(projection["present"]) is not bool:
        raise S4BControlError(f"metadata.fields.{name}.present is not bool")
    if projection["present"] is not present:
        raise S4BControlError(f"metadata.fields.{name}.present differs")
    if not present and projection["value"] is not None:
        raise S4BControlError(f"metadata.fields.{name} absent marker carries a value")
    return projection["value"]


def validate_s4a_metadata(metadata: Any, asset: LowFractionAsset) -> dict[str, Any]:
    metadata = _mapping(metadata, f"metadata[{asset.run_id}]")
    _exact_keys(
        metadata,
        (
            "checkpoint",
            "sha256",
            "size_bytes",
            "fields",
            "model_tensor_shapes",
            "drift_state_present",
        ),
        f"metadata[{asset.run_id}]",
    )
    _require_exact(metadata, "checkpoint", asset.checkpoint_path, asset.run_id)
    _require_exact(metadata, "sha256", asset.checkpoint_sha256, asset.run_id)
    _require_exact(metadata, "size_bytes", asset.checkpoint_size_bytes, asset.run_id)
    fields = _mapping(metadata["fields"], f"metadata[{asset.run_id}].fields")
    _exact_keys(fields, S4A_FIELDS, f"metadata[{asset.run_id}].fields")
    _require_metadata_scalar = lambda name, expected: (
        None
        if _strict_equal(_field_value(fields, name), expected)
        else (_ for _ in ()).throw(
            S4BControlError(f"metadata {asset.run_id}.{name} differs")
        )
    )
    _require_metadata_scalar("checkpoint_version", 6)
    _require_metadata_scalar("epoch", asset.best_epoch)
    _require_metadata_scalar("task_names", list(HUMAN3_TASKS))
    _require_metadata_scalar("prediction_mode", "quantile")
    _require_metadata_scalar("quantile_config", {"lower": 0.05, "upper": 0.95})
    _require_metadata_scalar(
        "feature_schema_version", asset.expected_data_config["feature_schema_version"]
    )
    _require_metadata_scalar("routing_enabled", True)
    _field_value(fields, "split_manifest_sha256", present=False)

    configuration = _mapping(
        _field_value(fields, "configuration"), f"metadata.{asset.run_id}.configuration"
    )
    for name, expected in asset.expected_configuration.items():
        _require_exact(configuration, name, expected, f"metadata.{asset.run_id}.configuration")
    for name, expected in asset.expected_adaptation.items():
        _require_exact(configuration, name, expected, f"metadata.{asset.run_id}.configuration")

    architecture = _mapping(
        _field_value(fields, "architecture_config"),
        f"metadata.{asset.run_id}.architecture_config",
    )
    for name, expected in asset.expected_architecture.items():
        _require_exact(architecture, name, expected, f"metadata.{asset.run_id}.architecture_config")

    data_config = _mapping(
        _field_value(fields, "data_config"), f"metadata.{asset.run_id}.data_config"
    )
    for name, expected in asset.expected_data_config.items():
        _require_exact(data_config, name, expected, f"metadata.{asset.run_id}.data_config")
    scalers = _mapping(
        _field_value(fields, "task_scalers"), f"metadata.{asset.run_id}.task_scalers"
    )
    if not _strict_equal(dict(scalers), dict(asset.expected_scalers)):
        raise S4BControlError(f"metadata scaler identity differs: {asset.run_id}")
    reproducibility = _mapping(
        _field_value(fields, "reproducibility"),
        f"metadata.{asset.run_id}.reproducibility",
    )
    for name, expected in asset.expected_reproducibility.items():
        _require_exact(reproducibility, name, expected, f"metadata.{asset.run_id}.reproducibility")
    for required_mapping in ("rgcer_config", "effective_rgcer_config", "selection_state"):
        _mapping(_field_value(fields, required_mapping), f"metadata.{asset.run_id}.{required_mapping}")
    if not _strict_equal(
        metadata["model_tensor_shapes"], dict(asset.expected_model_tensor_shapes)
    ):
        raise S4BControlError(f"metadata tensor signatures differ: {asset.run_id}")
    if metadata["drift_state_present"] is not True:
        raise S4BControlError(f"metadata drift-state presence differs: {asset.run_id}")
    return {
        "run_id": asset.run_id,
        "status": "STRICT_ORIGINAL_METADATA_VERIFIED",
        "migration_applied": False,
        "architecture_additions": [],
        "tensor_count": len(asset.expected_model_tensor_shapes),
    }


def _safe_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise S4BControlError(f"unsafe ZIP member name: {name!r}")


def _parse_checksums(data: bytes) -> dict[str, str]:
    result = {}
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise S4BControlError("invalid checksums.sha256 line")
        digest, name = match.groups()
        _safe_member_name(name)
        if name in result:
            raise S4BControlError(f"duplicate checksum member: {name}")
        result[name] = digest
    return result


def verified_zip_members(data: bytes, *, expected_sha256: str) -> dict[str, bytes]:
    if sha256_bytes(data) != expected_sha256:
        raise S4BControlError("ZIP SHA256 differs from the frozen identity")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise S4BControlError("ZIP duplicate/CRC validation failed")
        file_names = [name for name in names if not name.endswith("/")]
        for name in file_names:
            _safe_member_name(name)
        checksum_names = [
            name for name in file_names if PurePosixPath(name).name == "checksums.sha256"
        ]
        if len(checksum_names) != 1:
            raise S4BControlError("ZIP must contain exactly one checksums.sha256")
        members = {name: archive.read(name) for name in file_names}
    checksum_name = checksum_names[0]
    checksums = _parse_checksums(members[checksum_name])
    if set(checksums) != set(file_names) - {checksum_name}:
        raise S4BControlError("ZIP checksum coverage is not exact")
    for name, digest in checksums.items():
        if sha256_bytes(members[name]) != digest:
            raise S4BControlError(f"ZIP member checksum differs: {name}")
    return members


def verify_s4a_metadata_archive(
    path: str | Path, policy: LowFractionPolicy
) -> dict[str, Any]:
    archive_bytes = Path(path).read_bytes()
    members = verified_zip_members(archive_bytes, expected_sha256=S4A_CANONICAL_SHA256)
    verified = []
    for asset in policy.assets:
        member = asset.provenance["metadata_member"]
        if member not in members:
            raise S4BControlError(f"S4A metadata member is missing: {member}")
        raw = members[member]
        if sha256_bytes(raw) != asset.provenance["metadata_sha256"]:
            raise S4BControlError(f"S4A metadata member SHA differs: {asset.run_id}")
        metadata = strict_json_loads(raw.decode("utf-8-sig"))
        verified.append(validate_s4a_metadata(metadata, asset))
    metadata_members = [name for name in members if name.endswith("/checkpoint_metadata.json")]
    if len(metadata_members) != 40 or set(metadata_members) != {
        asset.provenance["metadata_member"] for asset in policy.assets
    }:
        raise S4BControlError("S4A metadata archive coverage differs")
    return {
        "status": "PASS",
        "fixture_kind": "REAL_038_METADATA_NOT_REAL_CHECKPOINT_LOAD",
        "canonical_sha256": S4A_CANONICAL_SHA256,
        "verified_asset_count": len(verified),
        "migration_applied_count": 0,
        "architecture_addition_count": 0,
        "tensor_signatures_per_asset": 170,
    }


def _validate_payload_identity(
    payload: Any,
    asset: LowFractionAsset,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    checkpoint = _mapping(payload, "checkpoint")
    required_top = {
        "checkpoint_version",
        "epoch",
        "configuration",
        "architecture_config",
        "task_names",
        "prediction_mode",
        "quantile_config",
        "task_scalers",
        "data_config",
        "feature_schema_version",
        "routing_enabled",
        "selection_state",
        "reproducibility",
        "model_state",
        "split_manifest_hash",
        "rgcer_config",
        "effective_rgcer_config",
    }
    missing = required_top - set(checkpoint)
    if missing:
        raise S4BControlError(f"checkpoint required fields are missing: {sorted(missing)}")
    if "split_manifest_sha256" in checkpoint:
        raise S4BControlError("split_manifest_sha256 must remain absent in S4B checkpoints")
    _require_exact(checkpoint, "checkpoint_version", 6, "checkpoint")
    _require_exact(checkpoint, "epoch", asset.best_epoch, "checkpoint")
    _require_exact(checkpoint, "task_names", list(HUMAN3_TASKS), "checkpoint")
    _require_exact(checkpoint, "prediction_mode", "quantile", "checkpoint")
    _require_exact(
        checkpoint, "quantile_config", {"lower": 0.05, "upper": 0.95}, "checkpoint"
    )
    _require_exact(
        checkpoint,
        "feature_schema_version",
        asset.expected_data_config["feature_schema_version"],
        "checkpoint",
    )
    _require_exact(
        checkpoint,
        "split_manifest_hash",
        asset.expected_data_config["split_manifest_hash"],
        "checkpoint",
    )
    _require_exact(checkpoint, "routing_enabled", True, "checkpoint")
    for name in ("selection_state", "rgcer_config", "effective_rgcer_config"):
        _mapping(checkpoint[name], f"checkpoint.{name}")

    configuration = _mapping(checkpoint["configuration"], "checkpoint.configuration")
    for name, expected in asset.expected_configuration.items():
        _require_exact(configuration, name, expected, "checkpoint.configuration")
    for name, expected in asset.expected_adaptation.items():
        _require_exact(configuration, name, expected, "checkpoint.configuration")
    datastore_context = configuration.get("datastore_context")
    if isinstance(datastore_context, Mapping) and _contains_unsupported_placeholder(
        datastore_context
    ):
        raise S4BControlError(
            "collector unsupported_type placeholder cannot be written into checkpoint configuration"
        )

    architecture = _mapping(
        checkpoint["architecture_config"], "checkpoint.architecture_config"
    )
    for name, expected in asset.expected_architecture.items():
        _require_exact(architecture, name, expected, "checkpoint.architecture_config")
    for name in ADAPTATION_FIELDS:
        if name not in architecture or name not in configuration:
            raise S4BControlError("S4B strict restoration cannot add adaptation fields")

    data_config = _mapping(checkpoint["data_config"], "checkpoint.data_config")
    for name, expected in asset.expected_data_config.items():
        _require_exact(data_config, name, expected, "checkpoint.data_config")
    scalers = _mapping(checkpoint["task_scalers"], "checkpoint.task_scalers")
    if not _strict_equal(dict(scalers), dict(asset.expected_scalers)):
        raise S4BControlError("checkpoint per-run scaler identity differs")
    for task in HUMAN3_TASKS:
        scaler = _mapping(scalers[task], f"checkpoint.task_scalers.{task}")
        _finite_float(scaler.get("mean"), f"checkpoint.task_scalers.{task}.mean")
        _finite_float(
            scaler.get("std"), f"checkpoint.task_scalers.{task}.std", positive=True
        )
        _true_int(
            scaler.get("count"), f"checkpoint.task_scalers.{task}.count", minimum=1
        )
    reproducibility = _mapping(
        checkpoint["reproducibility"], "checkpoint.reproducibility"
    )
    for name, expected in asset.expected_reproducibility.items():
        _require_exact(reproducibility, name, expected, "checkpoint.reproducibility")

    model_state = _mapping(checkpoint["model_state"], "checkpoint.model_state")
    if set(model_state) != set(asset.expected_model_tensor_shapes):
        raise S4BControlError("checkpoint model tensor name coverage differs")
    for name, expected in asset.expected_model_tensor_shapes.items():
        tensor = model_state[name]
        if not isinstance(tensor, torch_module.Tensor):
            raise S4BControlError(f"checkpoint model value is not a tensor: {name}")
        if list(tensor.shape) != expected["shape"] or str(tensor.dtype) != expected["dtype"]:
            raise S4BControlError(f"checkpoint model tensor signature differs: {name}")
    return {
        "checkpoint_version": 6,
        "loaded_epoch": asset.best_epoch,
        "task_scalers": dict(scalers),
        "effective_configuration": {
            **{name: configuration[name] for name in CONFIGURATION_FIELDS},
            **{name: configuration[name] for name in ADAPTATION_FIELDS},
        },
        "architecture_config": {
            name: architecture[name] for name in ARCHITECTURE_FIELDS
        },
        "data_config": {name: data_config[name] for name in DATA_FIELDS},
        "reproducibility": {
            name: reproducibility[name] for name in REPRODUCIBILITY_FIELDS
        },
        "routing_enabled": True,
        "model_tensor_count": len(model_state),
        "migration_applied": False,
        "architecture_additions": [],
    }


@dataclass(frozen=True)
class PreparedLowFractionCheckpoint:
    payload: Mapping[str, Any]
    asset: LowFractionAsset
    policy_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    original_model_state_sha256: str
    strict_identity: Mapping[str, Any]


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def prepare_low_fraction_checkpoint(
    checkpoint_path: str | Path,
    *,
    policy: LowFractionPolicy,
    run_id: str,
    method: str,
    seed: int,
    fraction_percent: int,
    operation: str,
    split: str,
    expected_policy_sha256: str,
) -> PreparedLowFractionCheckpoint:
    asset = validate_request(
        policy,
        run_id=run_id,
        method=method,
        seed=seed,
        fraction_percent=fraction_percent,
        operation=operation,
        split=split,
        expected_policy_sha256=expected_policy_sha256,
    )
    path = Path(checkpoint_path).resolve()
    frozen_path = Path(asset.checkpoint_path).resolve()
    if path != frozen_path:
        raise S4BControlError("checkpoint path differs from the frozen asset path")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    file_identity = verify_file_identity(
        path,
        expected_size_bytes=asset.checkpoint_size_bytes,
        expected_sha256=asset.checkpoint_sha256,
    )
    size = file_identity["size_bytes"]
    digest = file_identity["sha256"]

    # Import is intentionally after path/size/SHA validation.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    strict_identity = _validate_payload_identity(payload, asset, torch_module=torch)
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise S4BControlError("checkpoint bytes changed during CPU validation")
    state_sha = tensor_state_sha256(payload["model_state"])
    return PreparedLowFractionCheckpoint(
        payload=payload,
        asset=asset,
        policy_sha256=policy.sha256,
        checkpoint_path=str(path),
        checkpoint_sha256=digest,
        checkpoint_size_bytes=size,
        original_model_state_sha256=state_sha,
        strict_identity=strict_identity,
    )


def validate_sample_table(samples: Any, *, split: str) -> list[dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise S4BControlError("sample table split is unauthorized")
    if not isinstance(samples, list) or not samples:
        raise S4BControlError("sample table must be a non-empty list")
    result = []
    sample_ids = []
    row_indices = []
    for index, sample in enumerate(samples):
        sample = _mapping(sample, f"samples[{index}]")
        _exact_keys(sample, ("sample_id", "row_index", "labels"), f"samples[{index}]")
        sample_id = sample["sample_id"]
        row_index = sample["row_index"]
        labels = sample["labels"]
        if type(sample_id) is not str or not sample_id:
            raise S4BControlError(f"sample_id is invalid at index {index}")
        _true_int(row_index, f"samples[{index}].row_index")
        if not isinstance(labels, list) or len(labels) != len(HUMAN3_TASKS):
            raise S4BControlError(f"labels are not Human3 ordered at index {index}")
        for task_index, label in enumerate(labels):
            if label is not None and (
                type(label) not in (int, float) or not math.isfinite(label)
            ):
                raise S4BControlError(
                    f"label is invalid at sample {index}, task {HUMAN3_TASKS[task_index]}"
                )
        result.append(
            {"sample_id": sample_id, "row_index": row_index, "labels": list(labels)}
        )
        sample_ids.append(sample_id)
        row_indices.append(row_index)
    if len(set(sample_ids)) != len(result) or len(set(row_indices)) != len(result):
        raise S4BControlError("sample table has duplicate sample_id or row_index")
    if [(row["row_index"], row["sample_id"]) for row in result] != sorted(
        (row["row_index"], row["sample_id"]) for row in result
    ):
        raise S4BControlError("sample table order is not row_index/sample_id canonical")
    return result


def validate_frozen_test_table(samples: Any) -> list[dict[str, Any]]:
    result = validate_sample_table(samples, split="test")
    if len(result) != 28:
        raise S4BControlError("frozen test table must contain exactly 28 molecules")
    counts = [
        sum(sample["labels"][task_index] is not None for sample in result)
        for task_index in range(len(HUMAN3_TASKS))
    ]
    if counts != [14, 13, 13]:
        raise S4BControlError("frozen test endpoint counts differ from 14/13/13")
    digest = identity_sha256(result)
    if digest != FROZEN_TEST_SAMPLE_PROJECTION_SHA256:
        raise S4BControlError("frozen test sample projection SHA differs")
    return result


def validate_datastore_task_selection(
    datastore_task_names: Any,
    requested_task_names: Any,
    selected_task_indices: Any | None = None,
) -> dict[str, Any]:
    """Keep the full DataStore catalogue separate from the Human3 request."""

    if type(datastore_task_names) is not list or not datastore_task_names:
        raise S4BControlError(
            "datastore.metadata.task_names must be a non-empty list"
        )
    if any(type(task) is not str or not task for task in datastore_task_names):
        raise S4BControlError(
            "datastore.metadata.task_names entries must be non-empty strings"
        )
    if len(set(datastore_task_names)) != len(datastore_task_names):
        raise S4BControlError("datastore.metadata.task_names contains duplicates")

    if type(requested_task_names) is not list or not _strict_equal(
        requested_task_names, list(HUMAN3_TASKS)
    ):
        raise S4BControlError(
            "requested_task_names differs from the frozen Human3 order/type"
        )
    missing = [task for task in requested_task_names if task not in datastore_task_names]
    if missing:
        raise S4BControlError(
            f"DataStore is missing requested Human3 task_names: {missing}"
        )
    resolved_indices = [datastore_task_names.index(task) for task in requested_task_names]

    if selected_task_indices is not None:
        if type(selected_task_indices) is not list or any(
            type(index) is not int for index in selected_task_indices
        ):
            raise S4BControlError("selected_task_indices must be true integers")
        if selected_task_indices != resolved_indices:
            raise S4BControlError(
                "selected_task_indices differs from the name-resolved Human3 columns"
            )

    return {
        "datastore_task_names": list(datastore_task_names),
        "requested_task_names": list(requested_task_names),
        "selected_task_indices": resolved_indices,
    }


def validate_datastore_metadata(
    metadata: Any, asset: LowFractionAsset
) -> dict[str, Any]:
    """Bind a live DataStore V2 metadata block to the frozen data identity."""

    metadata = _mapping(metadata, "datastore.metadata")
    requested_task_names = list(asset.expected_data_config["task_names"])
    selection = validate_datastore_task_selection(
        metadata.get("task_names"), requested_task_names
    )

    num_tasks = _true_int(
        metadata.get("num_tasks"), "datastore.metadata.num_tasks", minimum=1
    )
    if num_tasks != len(selection["datastore_task_names"]):
        raise S4BControlError(
            "datastore.metadata.num_tasks does not match the full task catalogue"
        )
    labels_shape = metadata.get("labels_shape")
    if type(labels_shape) is not list or len(labels_shape) != 2:
        raise S4BControlError("datastore.metadata.labels_shape must be a two-item list")
    num_samples = _true_int(
        metadata.get("num_samples"), "datastore.metadata.num_samples", minimum=1
    )
    labels_rows = _true_int(
        labels_shape[0], "datastore.metadata.labels_shape[0]", minimum=1
    )
    labels_columns = _true_int(
        labels_shape[1], "datastore.metadata.labels_shape[1]", minimum=1
    )
    if labels_rows != num_samples:
        raise S4BControlError(
            "datastore.metadata.labels_shape rows do not match num_samples"
        )
    if labels_columns != num_tasks:
        raise S4BControlError(
            "datastore.metadata.labels_shape columns do not match num_tasks"
        )

    try:
        from toxacute_datastore import (
            DATASTORE_FORMAT,
            GRAPH_RECORD_VERSION,
            compute_datastore_fingerprint,
        )
    except ImportError as exc:
        raise S4BControlError(
            "DataStore native fingerprint implementation is unavailable"
        ) from exc

    _require_exact(metadata, "format", DATASTORE_FORMAT, "datastore.metadata")
    _require_exact(
        metadata,
        "graph_record_version",
        GRAPH_RECORD_VERSION,
        "datastore.metadata",
    )
    key_map = {
        "format_version": "datastore_format_version",
        "build_id": "datastore_build_id",
        "datastore_fingerprint": "datastore_fingerprint",
        "raw_csv_sha256": "raw_csv_sha256",
        "feature_schema_version": "feature_schema_version",
        "max_path_distance": "max_path_distance",
        "splitting": "splitting",
        "split_seed": "split_seed",
        "split_ratios": "split_ratios",
        "split_manifest_hash": "split_manifest_hash",
    }
    for metadata_name, policy_name in key_map.items():
        _require_exact(
            metadata,
            metadata_name,
            asset.expected_data_config[policy_name],
            "datastore.metadata",
        )
    _require_exact(metadata, "build_complete", True, "datastore.metadata")
    computed_fingerprint = compute_datastore_fingerprint(
        raw_csv_sha256=metadata["raw_csv_sha256"],
        feature_schema_version=metadata["feature_schema_version"],
        graph_record_version=metadata["graph_record_version"],
        max_path_distance=metadata["max_path_distance"],
        task_names=selection["datastore_task_names"],
        split_manifest_hash=metadata["split_manifest_hash"],
    )
    if computed_fingerprint != metadata["datastore_fingerprint"]:
        raise S4BControlError(
            "datastore full ordered task catalogue does not reproduce the frozen fingerprint"
        )
    return {
        "format": metadata["format"],
        "datastore_format_version": metadata["format_version"],
        "graph_record_version": metadata["graph_record_version"],
        "datastore_build_id": metadata["build_id"],
        "datastore_fingerprint": metadata["datastore_fingerprint"],
        "raw_csv_sha256": metadata["raw_csv_sha256"],
        "feature_schema_version": metadata["feature_schema_version"],
        "max_path_distance": metadata["max_path_distance"],
        "splitting": metadata["splitting"],
        "split_seed": metadata["split_seed"],
        "split_ratios": metadata["split_ratios"],
        "split_manifest_hash": metadata["split_manifest_hash"],
        "num_samples": num_samples,
        "num_tasks": num_tasks,
        "labels_shape": list(labels_shape),
        **selection,
        "build_complete": True,
    }


def metric_values(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    count = len(pairs)
    if count == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "r2_reason": "no_finite_pairs",
        }
    target_mean = math.fsum(target for target, _ in pairs) / count
    squared_error = math.fsum((target - prediction) ** 2 for target, prediction in pairs)
    target_variation = math.fsum((target - target_mean) ** 2 for target, _ in pairs)
    return {
        "n": count,
        "rmse": math.sqrt(squared_error / count),
        "mae": math.fsum(abs(target - prediction) for target, prediction in pairs)
        / count,
        "r2": None if count < 2 or target_variation == 0 else 1 - squared_error / target_variation,
        "r2_reason": (
            "fewer_than_two_samples"
            if count < 2
            else "constant_targets"
            if target_variation == 0
            else None
        ),
    }


def _metric_equal(actual: Any, expected: Any, context: str) -> None:
    if expected is None or isinstance(expected, str):
        if actual != expected:
            raise S4BControlError(f"{context} differs")
        return
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise S4BControlError(f"{context} differs")
        return
    if type(actual) not in (int, float) or isinstance(actual, bool) or not math.isfinite(actual):
        raise S4BControlError(f"{context} is not finite numeric")
    if not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
        raise S4BControlError(f"{context} differs")


def validate_stored_metrics(stored: Any, calculated: Mapping[str, Any]) -> None:
    stored = _mapping(stored, "metrics")
    _exact_keys(stored, ("per_endpoint", "human3_macro_rmse"), "metrics")
    per_endpoint = _mapping(stored["per_endpoint"], "metrics.per_endpoint")
    _exact_keys(per_endpoint, HUMAN3_TASKS, "metrics.per_endpoint")
    for task in HUMAN3_TASKS:
        task_metrics = _mapping(per_endpoint[task], f"metrics.{task}")
        _exact_keys(
            task_metrics,
            ("n", "rmse", "mae", "r2", "r2_reason"),
            f"metrics.{task}",
        )
        for name, expected in calculated["per_endpoint"][task].items():
            _metric_equal(task_metrics[name], expected, f"metrics.{task}.{name}")
    _metric_equal(
        stored["human3_macro_rmse"],
        calculated["human3_macro_rmse"],
        "metrics.human3_macro_rmse",
    )


def validate_prediction_rows(
    rows: Any,
    expected_samples: Any,
    *,
    asset: LowFractionAsset,
    split: str,
) -> dict[str, Any]:
    samples = validate_sample_table(expected_samples, split=split)
    if split == "test":
        validate_frozen_test_table(samples)
    if not isinstance(rows, list):
        raise S4BControlError("prediction rows must be a list")
    expected_keys = [
        (sample["sample_id"], task)
        for sample in samples
        for task in HUMAN3_TASKS
    ]
    observed_keys = [
        (row.get("sample_id"), row.get("endpoint"))
        if isinstance(row, Mapping)
        else (None, None)
        for row in rows
    ]
    if observed_keys != expected_keys or len(expected_keys) != len(set(expected_keys)):
        raise S4BControlError("prediction coverage/order differs from independent samples")
    per_endpoint: dict[str, list[tuple[float, float]]] = {
        task: [] for task in HUMAN3_TASKS
    }
    required_fields = {
        "run_id",
        "method",
        "seed",
        "fraction_percent",
        "sample_id",
        "row_index",
        "split",
        "endpoint",
        "y_true",
        "y_pred",
        "mask",
    }
    for sample_index, sample in enumerate(samples):
        for task_index, task in enumerate(HUMAN3_TASKS):
            row = _mapping(rows[sample_index * len(HUMAN3_TASKS) + task_index], "prediction")
            missing = required_fields - set(row)
            if missing:
                raise S4BControlError(f"prediction required fields missing: {sorted(missing)}")
            identity = (
                row["run_id"],
                row["method"],
                row["seed"],
                row["fraction_percent"],
                row["split"],
            )
            expected_identity = (
                asset.run_id,
                asset.method,
                asset.seed,
                asset.fraction_percent,
                split,
            )
            if any(type(value) is not type(expected) for value, expected in zip(identity, expected_identity)) or identity != expected_identity:
                raise S4BControlError("prediction run/method/seed/fraction/split identity differs")
            if type(row["row_index"]) is not int or row["row_index"] != sample["row_index"]:
                raise S4BControlError("prediction row_index differs")
            target = sample["labels"][task_index]
            if type(row["y_true"]) is not type(target) or row["y_true"] != target:
                raise S4BControlError("prediction label value/type differs")
            mask = target is not None
            if type(row["mask"]) is not bool or row["mask"] is not mask:
                raise S4BControlError("prediction mask differs")
            prediction = row["y_pred"]
            if type(prediction) not in (int, float) or isinstance(prediction, bool) or not math.isfinite(prediction):
                raise S4BControlError("prediction is not a finite numeric value")
            if mask:
                per_endpoint[task].append((float(target), float(prediction)))
    calculated = {task: metric_values(values) for task, values in per_endpoint.items()}
    if any(result["n"] < 2 for result in calculated.values()):
        raise S4BControlError("an endpoint has fewer than two finite prediction pairs")
    return {
        "per_endpoint": calculated,
        "human3_macro_rmse": math.fsum(
            result["rmse"] for result in calculated.values()
        )
        / len(HUMAN3_TASKS),
    }


def parse_jsonl(data: bytes | str, *, context: str = "predictions") -> list[Any]:
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(strict_json_loads(line))
        except Exception as exc:
            raise S4BControlError(f"{context} JSONL line {line_number} is invalid") from exc
    return rows


def compare_validation_reference(
    rows: Sequence[Mapping[str, Any]],
    calculated_metrics: Mapping[str, Any],
    asset: LowFractionAsset,
    csv_path: str | Path,
) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.is_file():
        actual = calculated_metrics["human3_macro_rmse"]
        expected = asset.best_validation_macro_rmse
        if not math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-7):
            raise S4BControlError("macro-only validation reference differs")
        return {
            "status": "PASS",
            "reference_level": "FROZEN_MACRO_ONLY",
            "historical_csv_exists": False,
            "historical_csv_path": str(path),
            "expected_macro_rmse": expected,
            "actual_macro_rmse": actual,
            "abs_difference": abs(actual - expected),
            "sample_equivalence_claimed": False,
        }

    with path.open(encoding="utf-8-sig", newline="") as handle:
        historical = list(csv.DictReader(handle))

    def field(row: Mapping[str, str], names: Sequence[str], context: str) -> str:
        values = [row[name] for name in names if name in row and row[name] != ""]
        if len(values) != 1:
            raise S4BControlError(f"historical validation {context} is missing/ambiguous")
        return values[0]

    observed = [row for row in rows if row["mask"]]
    observed_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in observed:
        key = (row["sample_id"], row["endpoint"])
        if key in observed_by_key:
            raise S4BControlError("new validation predictions have duplicate keys")
        observed_by_key[key] = row
    historical_by_key: dict[tuple[str, str], tuple[float, float, int | None]] = {}
    compared_row_index_count = 0
    for row in historical:
        key = (
            field(row, ("sample_id", "id"), "sample_id"),
            field(row, ("endpoint", "task"), "endpoint"),
        )
        if key in historical_by_key:
            raise S4BControlError("historical validation has duplicate sample/endpoint keys")
        try:
            target = float(field(row, ("y_true", "label", "target"), "target"))
            prediction = float(
                field(
                    row,
                    ("y_pred", "final_prediction", "prediction"),
                    "prediction",
                )
            )
        except ValueError as exc:
            raise S4BControlError("historical validation has a non-numeric value") from exc
        if not math.isfinite(target) or not math.isfinite(prediction):
            raise S4BControlError("historical validation has a non-finite value")
        historical_row_index = None
        if "row_index" in row:
            raw_row_index = row["row_index"].strip()
            if re.fullmatch(r"0|[1-9][0-9]*", raw_row_index) is None:
                raise S4BControlError("historical validation row_index is invalid")
            historical_row_index = int(raw_row_index)
            compared_row_index_count += 1
        historical_by_key[key] = (target, prediction, historical_row_index)
    if set(historical_by_key) != set(observed_by_key):
        missing = sorted(set(observed_by_key) - set(historical_by_key))
        extra = sorted(set(historical_by_key) - set(observed_by_key))
        raise S4BControlError(
            f"historical validation key coverage differs: missing={missing}, extra={extra}"
        )
    max_target_difference = 0.0
    max_prediction_difference = 0.0
    for key, row in observed_by_key.items():
        target, prediction, historical_row_index = historical_by_key[key]
        if historical_row_index is not None and historical_row_index != row["row_index"]:
            raise S4BControlError("historical validation row_index differs")
        if not math.isclose(target, row["y_true"], abs_tol=1e-6, rel_tol=1e-7):
            raise S4BControlError("historical validation target differs")
        if not math.isclose(prediction, row["y_pred"], abs_tol=1e-6, rel_tol=1e-7):
            raise S4BControlError("historical validation prediction differs")
        max_target_difference = max(max_target_difference, abs(target - row["y_true"]))
        max_prediction_difference = max(
            max_prediction_difference, abs(prediction - row["y_pred"])
        )
    return {
        "status": "PASS",
        "reference_level": "HISTORICAL_SAMPLE_PREDICTIONS",
        "historical_csv_exists": True,
        "historical_csv_path": str(path),
        "historical_csv_sha256": sha256_file(path),
        "compared_finite_rows": len(observed),
        "compared_row_index_count": compared_row_index_count,
        "historical_row_order_ignored": True,
        "max_target_abs_difference": max_target_difference,
        "max_prediction_abs_difference": max_prediction_difference,
        "sample_equivalence_claimed": True,
    }


RESTORATION_FIELDS = (
    "schema_version",
    "run_id",
    "method",
    "seed",
    "fraction_percent",
    "policy_sha256",
    "implementation_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "loaded_epoch",
    "checkpoint_version",
    "scaler_identity_sha256",
    "task_scalers",
    "effective_configuration",
    "architecture_config",
    "runtime_overrides",
    "runtime_override_names",
    "model_state_sha_algorithm",
    "original_model_state_sha256",
    "model_state_sha256_before",
    "model_state_sha256_after",
    "checkpoint_sha256_before",
    "checkpoint_sha256_after",
    "migration_applied",
    "architecture_additions",
    "init_overlay_applied",
    "training_called",
    "calibration_called",
    "model_unchanged",
    "checkpoint_unchanged",
    "routing_enabled",
)
RUNTIME_OVERRIDE_NAMES = (
    "mode",
    "inference_split",
    "load_path",
    "save_path",
    "data_store_dir",
    "preprocessed_data_dir",
    "gpu_id",
    "num_loader_workers",
    "fit_conformal",
)
MODEL_STATE_SHA_ALGORITHM = "s4_tensor_state_sha256_v1"
VALIDATION_EVIDENCE_FILES = (
    "expected_samples.json",
    "predictions.jsonl",
    "metrics.json",
    "restoration.json",
    "sample_binding.json",
    "reference_comparison.json",
    "reference_source.json",
)


def _require_sha256(value: Any, context: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise S4BControlError(f"{context} is not a lowercase SHA256")
    return value


def validate_restoration_evidence(
    restoration: Any,
    *,
    asset: LowFractionAsset,
    policy: LowFractionPolicy,
    implementation_commit: str,
    split: str,
    expected_data_store_dir: str,
) -> dict[str, Any]:
    """Validate every restoration claim used by collection and offline review."""

    restoration = _mapping(restoration, "restoration")
    _exact_keys(restoration, RESTORATION_FIELDS, "restoration")
    expected_scalar = {
        "schema_version": 1,
        "run_id": asset.run_id,
        "method": asset.method,
        "seed": asset.seed,
        "fraction_percent": asset.fraction_percent,
        "policy_sha256": policy.sha256,
        "implementation_commit": implementation_commit,
        "checkpoint_path": asset.checkpoint_path,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "checkpoint_size_bytes": asset.checkpoint_size_bytes,
        "loaded_epoch": asset.best_epoch,
        "checkpoint_version": 6,
        "model_state_sha_algorithm": MODEL_STATE_SHA_ALGORITHM,
        "migration_applied": False,
        "architecture_additions": [],
        "init_overlay_applied": False,
        "training_called": False,
        "calibration_called": False,
        "model_unchanged": True,
        "checkpoint_unchanged": True,
        "routing_enabled": True,
    }
    for name, expected in expected_scalar.items():
        _require_exact(restoration, name, expected, "restoration")

    if not _strict_equal(dict(_mapping(restoration["task_scalers"], "restoration.task_scalers")), dict(asset.expected_scalers)):
        raise S4BControlError("restoration task_scalers differ from the frozen run")
    _require_exact(
        restoration,
        "scaler_identity_sha256",
        identity_sha256(asset.expected_scalers),
        "restoration",
    )
    expected_configuration = {
        **dict(asset.expected_configuration),
        **dict(asset.expected_adaptation),
    }
    if not _strict_equal(
        dict(_mapping(restoration["effective_configuration"], "restoration.effective_configuration")),
        expected_configuration,
    ):
        raise S4BControlError("restoration effective configuration differs")
    if not _strict_equal(
        dict(_mapping(restoration["architecture_config"], "restoration.architecture_config")),
        dict(asset.expected_architecture),
    ):
        raise S4BControlError("restoration architecture_config differs")

    _require_exact(
        restoration,
        "runtime_override_names",
        list(RUNTIME_OVERRIDE_NAMES),
        "restoration",
    )
    runtime = _mapping(restoration["runtime_overrides"], "restoration.runtime_overrides")
    _exact_keys(runtime, RUNTIME_OVERRIDE_NAMES, "restoration.runtime_overrides")
    expected_runtime = {
        "mode": ALLOWED_OPERATION,
        "inference_split": split,
        "load_path": asset.checkpoint_path,
        "save_path": None,
        "data_store_dir": expected_data_store_dir,
        "preprocessed_data_dir": None,
        "gpu_id": "0",
        "num_loader_workers": 0,
        "fit_conformal": False,
    }
    if not _strict_equal(dict(runtime), expected_runtime):
        raise S4BControlError("restoration runtime overrides differ")

    original = _require_sha256(
        restoration["original_model_state_sha256"],
        "restoration original model state SHA",
    )
    before = _require_sha256(
        restoration["model_state_sha256_before"],
        "restoration before model state SHA",
    )
    after = _require_sha256(
        restoration["model_state_sha256_after"],
        "restoration after model state SHA",
    )
    if original != before or before != after:
        raise S4BControlError("restoration original/before/after model state SHA differ")
    for name in ("checkpoint_sha256_before", "checkpoint_sha256_after"):
        _require_exact(restoration, name, asset.checkpoint_sha256, "restoration")
    return dict(restoration)


def validate_validation_result_evidence(result: Mapping[str, Any]) -> None:
    reference_level = result.get("reference_level")
    historical_exists = result.get("historical_csv_exists")
    if reference_level not in {
        "HISTORICAL_SAMPLE_PREDICTIONS",
        "FROZEN_MACRO_ONLY",
    }:
        raise S4BControlError("validation result reference level is invalid")
    if type(historical_exists) is not bool:
        raise S4BControlError("validation result historical CSV fact is invalid")
    if historical_exists is not (reference_level == "HISTORICAL_SAMPLE_PREDICTIONS"):
        raise S4BControlError("validation result reference level/fact conflict")
    evidence = _mapping(result.get("evidence_files"), "validation_result.evidence_files")
    expected_names = set(VALIDATION_EVIDENCE_FILES)
    if historical_exists:
        expected_names.add("historical_validation_original.csv")
    if set(evidence) != expected_names:
        raise S4BControlError("validation result evidence file coverage differs")
    for name, digest in evidence.items():
        _require_sha256(digest, f"validation evidence SHA for {name}")
    _require_exact(
        result,
        "evidence_identity_sha256",
        identity_sha256(dict(evidence)),
        "validation_result",
    )


AUTHORIZATION_FIELDS = (
    "schema_version",
    "authorization_id",
    "status",
    "implementation_commit",
    "policy_sha256",
    "authorized_run_ids",
    "allowed_splits",
    "output_run_id",
    "fixture_only",
)


def validate_authorization(
    authorization: Any,
    *,
    policy: LowFractionPolicy,
    implementation_commit: str,
    split: str,
    output_run_id: str,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    authorization = _mapping(authorization, "authorization")
    _exact_keys(authorization, AUTHORIZATION_FIELDS, "authorization")
    _require_exact(authorization, "schema_version", 1, "authorization")
    if type(authorization["authorization_id"]) is not str or not authorization["authorization_id"]:
        raise S4BControlError("authorization_id is invalid")
    fixture_only = authorization["fixture_only"]
    if type(fixture_only) is not bool:
        raise S4BControlError("authorization.fixture_only is not bool")
    expected_status = (
        "SYNTHETIC_FIXTURE_ONLY" if fixture_only else "READY_FOR_S4B_SERVER_EXECUTION"
    )
    _require_exact(authorization, "status", expected_status, "authorization")
    if fixture_only and not allow_fixture:
        raise S4BControlError("synthetic fixture authorization cannot open formal inference")
    if type(implementation_commit) is not str or re.fullmatch(
        r"[0-9a-f]{40}", implementation_commit
    ) is None:
        raise S4BControlError("implementation commit is not a full 40-hex SHA")
    _require_exact(
        authorization, "implementation_commit", implementation_commit, "authorization"
    )
    _require_exact(authorization, "policy_sha256", policy.sha256, "authorization")
    _require_exact(
        authorization,
        "authorized_run_ids",
        list(S4B_RUN_IDS),
        "authorization",
    )
    if (
        not isinstance(authorization["allowed_splits"], list)
        or not authorization["allowed_splits"]
        or len(set(authorization["allowed_splits"]))
        != len(authorization["allowed_splits"])
        or any(item not in ALLOWED_SPLITS for item in authorization["allowed_splits"])
    ):
        raise S4BControlError("authorization allowed_splits are invalid")
    if split not in authorization["allowed_splits"]:
        raise S4BControlError(f"authorization does not permit split={split}")
    _require_exact(authorization, "output_run_id", output_run_id, "authorization")
    return dict(authorization)


def load_authorization(
    path: str | Path | None,
    **validation: Any,
) -> tuple[dict[str, Any], str]:
    if path is None:
        raise S4BControlError("formal server inference requires a separate Codex authorization")
    authorization_path = Path(path)
    if not authorization_path.is_file():
        raise S4BControlError("Codex authorization file is missing")
    authorization = read_json(authorization_path)
    return validate_authorization(authorization, **validation), sha256_file(authorization_path)


def verify_wsl_receipt(
    receipt: Any,
    *,
    implementation_commit: str,
    policy_sha256: str,
    runner_sha256: str,
    control_sha256: str,
    trainer_sha256: str,
    output_run_id: str,
) -> dict[str, Any]:
    receipt = _mapping(receipt, "wsl_receipt")
    required = {
        "schema_version": 1,
        "task_id": S4B_TASK_ID,
        "receipt_kind": "S4B_WSL_PREFLIGHT",
        "implementation_commit": implementation_commit,
        "output_run_id": output_run_id,
        "policy_sha256": policy_sha256,
        "core_policy_sha256": CORE_POLICY_SHA256,
        "runner_sha256": runner_sha256,
        "control_sha256": control_sha256,
        "trainer_sha256": trainer_sha256,
        "test_status": "PASS",
        "asset_accessed": False,
        "real_data_accessed": False,
        "training_called": False,
    }
    for name, expected in required.items():
        _require_exact(receipt, name, expected, "wsl_receipt")
    if type(receipt.get("receipt_id")) is not str or not receipt["receipt_id"]:
        raise S4BControlError("WSL receipt_id is invalid")
    return dict(receipt)


def build_validation_gate(
    results: Sequence[Mapping[str, Any]],
    *,
    policy: LowFractionPolicy,
    implementation_commit: str,
    output_run_id: str,
    authorization_sha256: str,
    preflight_receipt_sha256: str,
    validation_log_identity_sha256: str,
) -> dict[str, Any]:
    for value, context in (
        (authorization_sha256, "authorization SHA"),
        (preflight_receipt_sha256, "preflight receipt SHA"),
        (validation_log_identity_sha256, "validation log identity SHA"),
    ):
        _require_sha256(value, context)
    if type(output_run_id) is not str or not output_run_id:
        raise S4BControlError("validation gate output_run_id is invalid")
    by_run: dict[str, Mapping[str, Any]] = {}
    failures = []
    for result in results:
        result = _mapping(result, "validation_result")
        run_id = result.get("run_id")
        if type(run_id) is not str or run_id not in S4B_RUN_IDS or run_id in by_run:
            raise S4BControlError("validation result has unknown/duplicate run_id")
        asset = policy.asset(run_id)
        expected = {
            "method": asset.method,
            "seed": asset.seed,
            "fraction_percent": asset.fraction_percent,
            "split": "validation",
            "checkpoint_sha256": asset.checkpoint_sha256,
            "policy_sha256": policy.sha256,
            "implementation_commit": implementation_commit,
            "output_run_id": output_run_id,
            "authorization_sha256": authorization_sha256,
            "preflight_receipt_sha256": preflight_receipt_sha256,
        }
        for name, value in expected.items():
            _require_exact(result, name, value, f"validation_result[{run_id}]")
        prediction_sha = result.get("prediction_sha256")
        if type(prediction_sha) is not str or re.fullmatch(
            r"[0-9a-f]{64}", prediction_sha
        ) is None:
            raise S4BControlError("validation result prediction SHA is invalid")
        _mapping(result.get("metrics"), f"validation_result[{run_id}].metrics")
        validate_validation_result_evidence(result)
        if result.get("status") != "PASS":
            failures.append(run_id)
        by_run[run_id] = result
    missing = [run_id for run_id in S4B_RUN_IDS if run_id not in by_run]
    passed = not failures and not missing and len(by_run) == 40
    ordered_results = [
        dict(by_run[run_id]) for run_id in S4B_RUN_IDS if run_id in by_run
    ]
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "test_unlocked": passed,
        "implementation_commit": implementation_commit,
        "policy_sha256": policy.sha256,
        "output_run_id": output_run_id,
        "authorization_sha256": authorization_sha256,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "validation_log_identity_sha256": validation_log_identity_sha256,
        "required_validation_count": 40,
        "passed_validation_count": sum(
            result.get("status") == "PASS" for result in by_run.values()
        ),
        "failed_run_ids": failures,
        "missing_run_ids": missing,
        "authorized_test_run_ids": list(S4B_RUN_IDS) if passed else [],
        "validation_result_identity_sha256": identity_sha256(ordered_results),
        "validation_results": ordered_results,
    }


def verify_validation_gate(
    gate: Any,
    *,
    policy: LowFractionPolicy,
    implementation_commit: str,
    output_run_id: str,
    authorization_sha256: str,
    preflight_receipt_sha256: str,
    validation_log_identity_sha256: str,
) -> dict[str, Any]:
    gate = _mapping(gate, "validation_gate")
    required = {
        "schema_version": 1,
        "status": "PASS",
        "test_unlocked": True,
        "implementation_commit": implementation_commit,
        "policy_sha256": policy.sha256,
        "output_run_id": output_run_id,
        "authorization_sha256": authorization_sha256,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "validation_log_identity_sha256": validation_log_identity_sha256,
        "required_validation_count": 40,
        "passed_validation_count": 40,
        "failed_run_ids": [],
        "missing_run_ids": [],
        "authorized_test_run_ids": list(S4B_RUN_IDS),
    }
    for name, expected in required.items():
        _require_exact(gate, name, expected, "validation_gate")
    if type(gate.get("validation_result_identity_sha256")) is not str or re.fullmatch(
        r"[0-9a-f]{64}", gate["validation_result_identity_sha256"]
    ) is None:
        raise S4BControlError("validation gate result identity SHA is invalid")
    results = gate.get("validation_results")
    if not isinstance(results, list):
        raise S4BControlError("validation gate does not carry its forty source results")
    rebuilt = build_validation_gate(
        results,
        policy=policy,
        implementation_commit=implementation_commit,
        output_run_id=output_run_id,
        authorization_sha256=authorization_sha256,
        preflight_receipt_sha256=preflight_receipt_sha256,
        validation_log_identity_sha256=validation_log_identity_sha256,
    )
    if not _strict_equal(dict(gate), rebuilt):
        raise S4BControlError("validation gate does not reproduce from source results")
    return dict(gate)


@dataclass(frozen=True)
class AliasArtifact:
    run_id: str
    method: str
    seed: int
    fraction_percent: int
    prediction_member: str
    prediction_sha256: str
    prediction_bytes: bytes
    metrics_member: str
    metrics_sha256: str
    metrics_bytes: bytes
    metrics: Mapping[str, Any]
    expected_samples_member: str


@dataclass(frozen=True)
class VerifiedAliases:
    outer_sha256: str
    canonical_sha256: str
    frozen_samples: tuple[Mapping[str, Any], ...]
    aliases: tuple[AliasArtifact, ...]
    selected_members: Mapping[str, bytes]


def validate_legacy_alias_rows(
    rows: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    method: str,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise S4BControlError("037 alias predictions are not a list")
    expected_keys = [
        (sample["sample_id"], task) for sample in samples for task in HUMAN3_TASKS
    ]
    if [(row.get("sample_id"), row.get("endpoint")) for row in rows] != expected_keys:
        raise S4BControlError("037 alias prediction order/coverage differs")
    per_endpoint: dict[str, list[tuple[float, float]]] = {
        task: [] for task in HUMAN3_TASKS
    }
    for sample_index, sample in enumerate(samples):
        for task_index, task in enumerate(HUMAN3_TASKS):
            row = _mapping(rows[sample_index * 3 + task_index], "037 prediction")
            if row.get("method") != method or type(row.get("seed")) is not int or row["seed"] != seed:
                raise S4BControlError("037 alias method/seed identity differs")
            if row.get("split") != "test":
                raise S4BControlError("037 alias split differs")
            if type(row.get("row_index")) is not int or row["row_index"] != sample["row_index"]:
                raise S4BControlError("037 alias row_index differs")
            target = sample["labels"][task_index]
            if type(row.get("y_true")) is not type(target) or row.get("y_true") != target:
                raise S4BControlError("037 alias target differs")
            expected_mask = target is not None
            if type(row.get("mask")) is not bool or row["mask"] is not expected_mask:
                raise S4BControlError("037 alias mask differs")
            prediction = row.get("y_pred")
            if type(prediction) not in (int, float) or isinstance(prediction, bool) or not math.isfinite(prediction):
                raise S4BControlError("037 alias prediction is not finite")
            if expected_mask:
                per_endpoint[task].append((float(target), float(prediction)))
    metrics = {task: metric_values(values) for task, values in per_endpoint.items()}
    return {
        "per_endpoint": metrics,
        "human3_macro_rmse": math.fsum(value["rmse"] for value in metrics.values())
        / len(HUMAN3_TASKS),
    }


def verify_037_aliases(path: str | Path) -> VerifiedAliases:
    outer = Path(path).read_bytes()
    if sha256_bytes(outer) != S3C3_OUTER_SHA256:
        raise S4BControlError("037 outer archive SHA differs")
    with zipfile.ZipFile(io.BytesIO(outer)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise S4BControlError("037 outer archive duplicate/CRC validation failed")
        for name in names:
            if not name.endswith("/"):
                _safe_member_name(name)
        if S3C3_CANONICAL_MEMBER not in names:
            raise S4BControlError("037 canonical member is missing")
        canonical = archive.read(S3C3_CANONICAL_MEMBER)
    members = verified_zip_members(canonical, expected_sha256=S3C3_CANONICAL_SHA256)
    selected: dict[str, bytes] = {}
    aliases = []
    frozen_samples: list[dict[str, Any]] | None = None
    for method in METHODS:
        for seed in SEEDS:
            run_id = f"D8_formal_{method}_s{seed}"
            prefix = f"test_{run_id}/"
            prediction_member = prefix + "predictions.jsonl"
            metrics_member = prefix + "metrics.json"
            expected_member = prefix + "expected_samples.json"
            restoration_member = prefix + "restoration.json"
            sample_verification_member = prefix + "frozen_test_sample_verification.json"
            for name in (
                prediction_member,
                metrics_member,
                expected_member,
                restoration_member,
                sample_verification_member,
            ):
                if name not in members:
                    raise S4BControlError(f"037 alias evidence is incomplete: {name}")
                selected[name] = members[name]
            if sha256_bytes(members[prediction_member]) != ALIAS_PREDICTION_SHA256[(method, seed)]:
                raise S4BControlError(f"037 alias prediction bytes differ: {run_id}")
            if sha256_bytes(members[metrics_member]) != ALIAS_METRICS_SHA256[(method, seed)]:
                raise S4BControlError(f"037 alias metrics bytes differ: {run_id}")
            expected = strict_json_loads(members[expected_member].decode("utf-8-sig"))
            expected = validate_frozen_test_table(expected)
            if frozen_samples is None:
                frozen_samples = expected
            elif not _strict_equal(expected, frozen_samples):
                raise S4BControlError("037 alias expected sample tables differ")
            rows = parse_jsonl(members[prediction_member], context=run_id)
            calculated = validate_legacy_alias_rows(rows, expected, method=method, seed=seed)
            stored_metrics = strict_json_loads(members[metrics_member].decode("utf-8-sig"))
            validate_stored_metrics(stored_metrics, calculated)
            restoration = _mapping(
                strict_json_loads(members[restoration_member].decode("utf-8-sig")),
                f"{run_id}.restoration",
            )
            restored_asset = _mapping(restoration.get("asset"), f"{run_id}.restoration.asset")
            if (
                restored_asset.get("run_id") != run_id
                or restored_asset.get("method") != method
                or type(restored_asset.get("seed")) is not int
                or restored_asset["seed"] != seed
                or restoration.get("checkpoint_unchanged") is not True
                or restoration.get("model_unchanged") is not True
                or restoration.get("training_called") is not False
                or restoration.get("calibration_called") is not False
            ):
                raise S4BControlError(f"037 alias restoration identity differs: {run_id}")
            sample_verification = _mapping(
                strict_json_loads(
                    members[sample_verification_member].decode("utf-8-sig")
                ),
                f"{run_id}.sample_verification",
            )
            if (
                sample_verification.get("status") != "VERIFIED"
                or sample_verification.get("frozen_test_sample_identity_sha256")
                != FROZEN_TEST_SAMPLE_PROJECTION_SHA256
                or sample_verification.get("exact_ordered_match") is not True
            ):
                raise S4BControlError(f"037 alias sample binding differs: {run_id}")
            aliases.append(
                AliasArtifact(
                    run_id=run_id,
                    method=method,
                    seed=seed,
                    fraction_percent=100,
                    prediction_member=prediction_member,
                    prediction_sha256=ALIAS_PREDICTION_SHA256[(method, seed)],
                    prediction_bytes=members[prediction_member],
                    metrics_member=metrics_member,
                    metrics_sha256=ALIAS_METRICS_SHA256[(method, seed)],
                    metrics_bytes=members[metrics_member],
                    metrics=stored_metrics,
                    expected_samples_member=expected_member,
                )
            )
    if frozen_samples is None or len(aliases) != 10:
        raise S4BControlError("037 alias coverage is not exactly ten")
    return VerifiedAliases(
        outer_sha256=S3C3_OUTER_SHA256,
        canonical_sha256=S3C3_CANONICAL_SHA256,
        frozen_samples=tuple(frozen_samples),
        aliases=tuple(aliases),
        selected_members=selected,
    )


def verify_analysis_manifest(path: str | Path, aliases: VerifiedAliases) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _mapping(read_json(manifest_path), "analysis_manifest")
    run_sources = manifest.get("run_sources")
    if not isinstance(run_sources, list):
        raise S4BControlError("analysis manifest run_sources are missing")
    relevant = {}
    for row in run_sources:
        if not isinstance(row, Mapping):
            continue
        identity = (row.get("method"), row.get("seed"))
        if identity in ALIAS_PREDICTION_SHA256:
            if identity in relevant:
                raise S4BControlError("analysis manifest has duplicate 100% identity")
            relevant[identity] = row
    if set(relevant) != set(ALIAS_PREDICTION_SHA256):
        raise S4BControlError("analysis manifest 100% identity coverage differs")
    for alias in aliases.aliases:
        row = relevant[(alias.method, alias.seed)]
        if (
            row.get("member") != alias.prediction_member
            or row.get("sha256") != alias.prediction_sha256
        ):
            raise S4BControlError("analysis manifest alias member/SHA differs")
    return {
        "status": "PASS",
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "verified_alias_count": len(relevant),
    }


def build_unified_50_metrics(
    low_fraction_results: Sequence[Mapping[str, Any]],
    aliases: VerifiedAliases,
    *,
    policy: LowFractionPolicy,
) -> list[dict[str, Any]]:
    by_identity = {}
    for row in low_fraction_results:
        row = _mapping(row, "low_fraction_result")
        run_id = row.get("run_id")
        if type(run_id) is not str:
            raise S4BControlError("low-fraction result run_id is invalid")
        asset = policy.asset(run_id)
        identity = (row.get("method"), row.get("seed"), row.get("fraction_percent"))
        if identity != asset.identity or any(
            type(value) is not type(expected)
            for value, expected in zip(identity, asset.identity)
        ):
            raise S4BControlError("low-fraction result identity differs")
        if identity in by_identity:
            raise S4BControlError("low-fraction result identity is duplicated")
        if row.get("status") != "PASS" or row.get("split") != "test":
            raise S4BControlError("low-fraction result is not a passed test export")
        _require_exact(row, "checkpoint_sha256", asset.checkpoint_sha256, run_id)
        _require_exact(row, "policy_sha256", policy.sha256, run_id)
        metrics = _mapping(row.get("metrics"), f"{run_id}.metrics")
        prediction_sha = row.get("prediction_sha256")
        if type(prediction_sha) is not str or re.fullmatch(
            r"[0-9a-f]{64}", prediction_sha
        ) is None:
            raise S4BControlError("low-fraction prediction SHA is invalid")
        by_identity[identity] = {
            "run_id": run_id,
            "method": asset.method,
            "seed": asset.seed,
            "fraction_percent": asset.fraction_percent,
            "source": "S4B_NEW_LOW_FRACTION_TEST",
            "metrics": dict(metrics),
            "prediction_sha256": prediction_sha,
        }
    if set(by_identity) != S4B_MATRIX:
        raise S4BControlError("low-fraction test matrix is not exactly forty")
    for alias in aliases.aliases:
        identity = (alias.method, alias.seed, 100)
        if identity in by_identity:
            raise S4BControlError("100% alias would overwrite a low-fraction result")
        by_identity[identity] = {
            "run_id": alias.run_id,
            "method": alias.method,
            "seed": alias.seed,
            "fraction_percent": 100,
            "source": "ALIAS_037_NO_RERUN",
            "metrics": dict(alias.metrics),
            "prediction_sha256": alias.prediction_sha256,
            "source_canonical_sha256": aliases.canonical_sha256,
            "source_prediction_member": alias.prediction_member,
            "source_metrics_member": alias.metrics_member,
        }
    expected = S4B_MATRIX | {
        (method, seed, 100) for method in METHODS for seed in SEEDS
    }
    if set(by_identity) != expected or len(by_identity) != 50:
        raise S4BControlError("unified label-fraction matrix is not exactly fifty")
    return [by_identity[key] for key in sorted(by_identity)]


def decode_synthetic_human3_batch(
    raw_by_task: Mapping[str, Any],
    scalers: Mapping[str, Any],
    model: Any,
) -> dict[str, list[float]]:
    """CPU smoke helper for quantile median decoding and per-run de-scaling.

    It snapshots the model before and after and performs no optimizer, backward,
    training-mode, calibration, or real-data action.
    """

    import torch
    from architecture.prediction_heads import decode_prediction

    if set(raw_by_task) != set(HUMAN3_TASKS) or set(scalers) != set(HUMAN3_TASKS):
        raise S4BControlError("synthetic batch must cover exactly Human3")
    before = tensor_state_sha256(model.state_dict())
    was_training = bool(model.training)
    model.eval()
    decoded = {}
    with torch.no_grad():
        batch_size = None
        for task in HUMAN3_TASKS:
            raw = raw_by_task[task]
            if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[1] != 3:
                raise S4BControlError("synthetic quantile output must have shape [B,3]")
            if batch_size is None:
                batch_size = raw.shape[0]
            elif raw.shape[0] != batch_size:
                raise S4BControlError("synthetic Human3 batch sizes differ")
            scaler = _mapping(scalers[task], f"synthetic.scaler.{task}")
            _finite_float(scaler.get("mean"), f"synthetic.scaler.{task}.mean")
            _finite_float(scaler.get("std"), f"synthetic.scaler.{task}.std", positive=True)
            median = decode_prediction(raw, mode="quantile").median
            values = median * scaler["std"] + scaler["mean"]
            decoded[task] = values.detach().cpu().reshape(-1).tolist()
    if was_training:
        model.train()
    after = tensor_state_sha256(model.state_dict())
    if before != after:
        raise S4BControlError("synthetic inference changed model tensors")
    return decoded


__all__ = [
    "ADAPTATION_FIELDS",
    "ALIAS_METRICS_SHA256",
    "ALIAS_PREDICTION_SHA256",
    "ALLOWED_OPERATION",
    "ALLOWED_SPLITS",
    "AliasArtifact",
    "CORE_POLICY_SHA256",
    "FROZEN_TEST_SAMPLE_PROJECTION_SHA256",
    "HUMAN3_TASKS",
    "LOW_FRACTIONS",
    "LowFractionAsset",
    "LowFractionPolicy",
    "METHODS",
    "MODEL_STATE_SHA_ALGORITHM",
    "PreparedLowFractionCheckpoint",
    "S3C3_CANONICAL_SHA256",
    "S3C3_OUTER_SHA256",
    "S4A_CANONICAL_SHA256",
    "S4B_BASE_COMMIT",
    "S4BControlError",
    "S4B_MATRIX",
    "S4B_POLICY_SHA256",
    "S4B_RUN_IDS",
    "SEEDS",
    "RUNTIME_OVERRIDE_NAMES",
    "VALIDATION_EVIDENCE_FILES",
    "VerifiedAliases",
    "assert_core_policy_unchanged",
    "build_unified_50_metrics",
    "build_validation_gate",
    "compare_validation_reference",
    "decode_synthetic_human3_batch",
    "identity_sha256",
    "load_authorization",
    "load_s4b_policy",
    "metric_values",
    "parse_jsonl",
    "prepare_low_fraction_checkpoint",
    "read_json",
    "sha256_bytes",
    "sha256_file",
    "strict_json_loads",
    "tensor_state_sha256",
    "validate_authorization",
    "validate_frozen_test_table",
    "validate_datastore_metadata",
    "validate_datastore_task_selection",
    "validate_legacy_alias_rows",
    "validate_prediction_rows",
    "validate_request",
    "validate_restoration_evidence",
    "validate_s4a_metadata",
    "validate_sample_table",
    "validate_stored_metrics",
    "validate_validation_result_evidence",
    "verify_037_aliases",
    "verify_analysis_manifest",
    "verify_s4a_metadata_archive",
    "verify_validation_gate",
    "verify_wsl_receipt",
    "verify_file_identity",
    "verified_zip_members",
]
