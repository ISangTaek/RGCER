"""Fail-closed, inference-only compatibility for the locked S3C3 checkpoints.

The historical seed 42/44/46 checkpoints predate nine adaptation fields in
``architecture_config``.  This module creates an in-memory architecture view
only after binding the request to an exact checkpoint file and validating the
audited missing-field pattern.  It never rewrites a checkpoint, changes a
state dict, or relaxes the normal training/resume checkpoint contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


HUMAN3_TASKS = (
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
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

CHECKPOINT_CONFIGURATION_FIELDS = (
    "d6_candidate",
    "d7_candidate",
    "freeze_backbone_epochs",
    "backbone_lr_multiplier",
    "feature_drift_probe_size",
    "feature_drift_epochs",
)

INFERENCE_COMPATIBILITY_CONSTANTS = MappingProxyType(
    {
        "trainable_last_blocks": 0,
        "retention_probe_per_task": 16,
        "retention_damage_threshold": 0.02,
    }
)

ALLOWED_OPERATION = "batch_inference"
ALLOWED_SPLITS = ("validation", "test")


class InferenceCompatibilityError(ValueError):
    """The checkpoint is outside the frozen inference compatibility policy."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return {"unsupported_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Value equality that rejects bool-as-int and all other type coercions."""

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


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceCompatibilityError(f"{context} must be a mapping")
    return value


def _require_exact(
    mapping: Mapping[str, Any], name: str, expected: Any, context: str
) -> None:
    if name not in mapping:
        raise InferenceCompatibilityError(f"{context}.{name} is missing")
    if not _strict_equal(mapping[name], expected):
        raise InferenceCompatibilityError(
            f"{context}.{name} does not match the frozen identity "
            f"({mapping[name]!r} != {expected!r}, including type)"
        )


@dataclass(frozen=True)
class AssetCompatibilityRule:
    run_id: str
    method: str
    seed: int
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    best_epoch: int
    migration_required: bool
    adaptation_config: Mapping[str, Any]

    def as_policy_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "method": self.method,
            "seed": self.seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "best_epoch": self.best_epoch,
            "migration_required": self.migration_required,
            "adaptation_config": dict(self.adaptation_config),
        }


@dataclass(frozen=True)
class InferenceCompatibilityPolicy:
    version: str
    assets: tuple[AssetCompatibilityRule, ...]
    task_names: tuple[str, ...]
    configuration_identity: Mapping[str, Any]
    model_identity: Mapping[str, Any]
    data_identity: Mapping[str, Any]
    compatibility_constants: Mapping[str, Any] = field(
        default_factory=lambda: INFERENCE_COMPATIBILITY_CONSTANTS
    )

    def as_policy_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "allowed_operation": ALLOWED_OPERATION,
            "allowed_splits": list(ALLOWED_SPLITS),
            "task_names": list(self.task_names),
            "configuration_identity": dict(self.configuration_identity),
            "model_identity": _json_safe(self.model_identity),
            "data_identity": _json_safe(self.data_identity),
            "compatibility_constants": dict(self.compatibility_constants),
            "assets": [asset.as_policy_dict() for asset in self.assets],
        }

    def computed_sha256(self) -> str:
        return sha256_json(self.as_policy_dict())

    def asset(self, run_id: str) -> AssetCompatibilityRule:
        matches = [asset for asset in self.assets if asset.run_id == run_id]
        if len(matches) != 1:
            raise InferenceCompatibilityError(
                f"run_id {run_id!r} is not uniquely authorized by the policy"
            )
        return matches[0]


def _adaptation_config(method: str, *, full_metadata: bool) -> Mapping[str, Any]:
    if method == "B0":
        d6_candidate, d7_candidate, freeze_epochs = "b0", "none", 0
    elif method == "B1":
        d6_candidate, d7_candidate, freeze_epochs = "b1", "none", 0
    elif method == "RPT":
        d6_candidate, d7_candidate, freeze_epochs = "none", "s1", 40
    else:  # pragma: no cover - construction guard
        raise AssertionError(method)
    # Audited new checkpoints have the newer default for B0/RPT.  B1 and all
    # audited legacy checkpoints record the original 40-epoch schedule.
    feature_epochs = (
        "0,5,10,15,19" if full_metadata and method in {"B0", "RPT"} else
        "0,5,10,15,20,25,30,35,39"
    )
    return MappingProxyType(
        {
            "d6_candidate": d6_candidate,
            "d7_candidate": d7_candidate,
            "freeze_backbone_epochs": freeze_epochs,
            "backbone_lr_multiplier": 1.0,
            "trainable_last_blocks": 0,
            "retention_probe_per_task": 16,
            "retention_damage_threshold": 0.02,
            "feature_drift_probe_size": 128,
            "feature_drift_epochs": feature_epochs,
        }
    )


_PRODUCTION_ASSET_ROWS = (
    ("D8_formal_B0_s42", "B0", 42, 2, "22008be9d70b3d88f0ae38ae72bfe3a8263bdb2753ac905e68cb8343f6e9bf74", 6957081),
    ("D8_formal_B0_s43", "B0", 43, 3, "7d80603040d04f6ed1fcf0f8bec81be52370ff1c6f563c426b9ee79040a0657e", 6957337),
    ("D8_formal_B0_s44", "B0", 44, 3, "d5fb2e9f32247e06683cc1ed2915335df7aecbe33d6077d837edf6beca6a922d", 6957081),
    ("D8_formal_B0_s45", "B0", 45, 4, "cd74f2c0cbc4a103f50f6a750ee5b489f403bb0da2715e2c5d8884b40b1396cd", 6957337),
    ("D8_formal_B0_s46", "B0", 46, 4, "b3f3d662f24a0d17da97ea74ddb2959c9ec4ec2e3a8151dc0c456d11f537daed", 6957081),
    ("D8_formal_B1_s42", "B1", 42, 0, "0e835e4fcb1b2da9e8b088b7e437acbe1a338ad370f7522e742f9a75dd39dfb3", 6957913),
    ("D8_formal_B1_s43", "B1", 43, 0, "22fb105ea3557ecc125a4ea79df90463fcc67f3ebd87b6cc5b4f986899081f91", 9225345),
    ("D8_formal_B1_s44", "B1", 44, 0, "2b1db3b4da22f31e624b502fe8032f6a9e496ef06bad20f10df5a81027040b3e", 6957913),
    ("D8_formal_B1_s45", "B1", 45, 0, "b6d2aa54e24cb58f0a5a8557a103eb66a7e7ac3aa66f2caa7120e0767b8c17dc", 9225345),
    ("D8_formal_B1_s46", "B1", 46, 0, "93faa21cf27581836484c69dae44ac4a2f2aeafffee813966321c3144984a9c6", 6957913),
    ("D8_formal_RPT_s42", "RPT", 42, 37, "abb0f295f24569b574ea80e97515bd437551b777e8f2a9b5d3d8176db907c327", 2577909),
    ("D8_formal_RPT_s43", "RPT", 43, 33, "d1dec49910ca89df830724725bd12384590fbcf8ceb007508665b202074548cb", 4849309),
    ("D8_formal_RPT_s44", "RPT", 44, 23, "03c113ba5bec232f553721580d70401ec4493857baf4e68d2b5252ea2c2dee63", 2577845),
    ("D8_formal_RPT_s45", "RPT", 45, 34, "e359dc8fdb6067f7d4045e294700bc232fd410a60c17063ba017c31ab10f6e7e", 4849437),
    ("D8_formal_RPT_s46", "RPT", 46, 19, "8b69c013c4ce41c6dd17c021c37d19d5f3d27f4dfc114c74b1abae061cb136aa", 2577845),
)


def _production_assets() -> tuple[AssetCompatibilityRule, ...]:
    result = []
    for run_id, method, seed, epoch, checkpoint_sha, size_bytes in _PRODUCTION_ASSET_ROWS:
        migration_required = seed in {42, 44, 46}
        result.append(
            AssetCompatibilityRule(
                run_id=run_id,
                method=method,
                seed=seed,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_size_bytes=size_bytes,
                best_epoch=epoch,
                migration_required=migration_required,
                adaptation_config=_adaptation_config(
                    method, full_metadata=not migration_required
                ),
            )
        )
    return tuple(result)


PRODUCTION_POLICY = InferenceCompatibilityPolicy(
    version="S3C3-R2-INFERENCE-COMPATIBILITY-v1",
    assets=_production_assets(),
    task_names=HUMAN3_TASKS,
    configuration_identity=MappingProxyType(
        {
            "arch": "Graphormer",
            "dataset": "toxacute",
            "splitting": "scaffold",
            "split_seed": 42,
            "toxacute_task_scope": "human3",
            "prediction_mode": "quantile",
            "train_eval_scope": "validation_only",
            "fit_conformal": False,
            "epochs": 40,
            "data_store_dir": "data/toxacute_datastore_v2",
        }
    ),
    model_identity=MappingProxyType(
        {
            "architecture": "Graphormer",
            "task_names": list(HUMAN3_TASKS),
            "prediction_mode": "quantile",
            "edge_bias_mode": "path",
            "spatial_pos_max_clip": 20,
            "hidden_dim": 96,
            "a_layers": 8,
            "a_heads": 4,
            "mid_dim": 128,
            "head_hidden_dim": 96,
            "head_dropout": 0.1,
            "adapter_ratio": 0.25,
            "response_hidden_dim": 96,
            "task_sampling": "proportional",
            "conformal_scope": "human3",
            "router_top_k": 8,
            "router_temperature": 1.0,
            "exclude_target_from_sources": True,
            "use_factorized_prompt": None,
            "card_enabled": False,
            "card_lambda_delta": 0.0,
            "card_bottleneck": 32,
            "task_metadata": [],
        }
    ),
    data_identity=MappingProxyType(
        {
            "datastore_format_version": 2,
            "datastore_build_id": "toxacute-v2-7b7bd62a6457",
            "datastore_fingerprint": "7b7bd62a6457501c8010f12b9bae851ec9c8b265539c93ddca583cf4b952be5c",
            "raw_csv_sha256": "47b406217dfbe916b0644a11ca071783791dd8a73f2f93685876560b0ab97eae",
            "feature_schema_version": "atom_v2_bond_v1_pathavg_v1",
            "max_path_distance": 8,
            "splitting": "scaffold",
            "split_seed": 42,
            "split_manifest_hash": "61a2e494469a4035447f237532272487fd897adb3fadae7879e0dc75d0b01085",
            "max_nodes_filter": 512,
            "task_names": list(HUMAN3_TASKS),
        }
    ),
)

# This value is intentionally checked at import.  Changing any policy asset,
# constant, allowed operation/split, or identity field requires an explicit
# source update rather than silently changing the meaning of an old runner.
PRODUCTION_POLICY_SHA256 = "522ea73de9343b268c89197141221ed4fc87f87ea48eef24bc951831cbafae91"
if PRODUCTION_POLICY.computed_sha256() != PRODUCTION_POLICY_SHA256:
    raise RuntimeError("S3C3-R2 production compatibility policy identity changed")


def _validate_request(
    *,
    run_id: Any,
    method: Any,
    seed: Any,
    operation: Any,
    split: Any,
    expected_policy_sha256: Any,
    policy: InferenceCompatibilityPolicy,
) -> AssetCompatibilityRule:
    if type(operation) is not str or operation != ALLOWED_OPERATION:
        raise InferenceCompatibilityError(
            "compatibility is authorized only for batch_inference"
        )
    if type(split) is not str or split not in ALLOWED_SPLITS:
        raise InferenceCompatibilityError(
            "compatibility is authorized only for validation/test inference"
        )
    if type(run_id) is not str or type(method) is not str or type(seed) is not int:
        raise InferenceCompatibilityError("run_id/method/seed have invalid types")
    actual_policy_sha = policy.computed_sha256()
    if (
        type(expected_policy_sha256) is not str
        or expected_policy_sha256 != actual_policy_sha
    ):
        raise InferenceCompatibilityError(
            "compatibility policy SHA does not match the expected frozen policy"
        )
    asset = policy.asset(run_id)
    if method != asset.method or seed != asset.seed:
        raise InferenceCompatibilityError(
            "requested method/seed do not match the authorized asset"
        )
    return asset


def _validate_payload_identity(
    payload: Any,
    *,
    asset: AssetCompatibilityRule,
    policy: InferenceCompatibilityPolicy,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    checkpoint = _require_mapping(payload, "checkpoint")
    _require_exact(checkpoint, "checkpoint_version", 6, "checkpoint")
    _require_exact(checkpoint, "epoch", asset.best_epoch, "checkpoint")
    _require_exact(checkpoint, "task_names", list(policy.task_names), "checkpoint")
    _require_exact(checkpoint, "prediction_mode", "quantile", "checkpoint")
    _require_exact(
        checkpoint,
        "feature_schema_version",
        policy.data_identity["feature_schema_version"],
        "checkpoint",
    )
    _require_exact(
        checkpoint,
        "quantile_config",
        {"lower": 0.05, "upper": 0.95},
        "checkpoint",
    )

    configuration = _require_mapping(
        checkpoint.get("configuration"), "checkpoint.configuration"
    )
    for name, expected in policy.configuration_identity.items():
        _require_exact(configuration, name, expected, "checkpoint.configuration")
    _require_exact(configuration, "seed", asset.seed, "checkpoint.configuration")

    data_config = _require_mapping(
        checkpoint.get("data_config"), "checkpoint.data_config"
    )
    for name, expected in policy.data_identity.items():
        _require_exact(data_config, name, expected, "checkpoint.data_config")

    reproducibility = _require_mapping(
        checkpoint.get("reproducibility"), "checkpoint.reproducibility"
    )
    _require_exact(reproducibility, "base_seed", asset.seed, "checkpoint.reproducibility")
    _require_exact(
        reproducibility, "seed_policy_version", 2, "checkpoint.reproducibility"
    )

    scalers = _require_mapping(
        checkpoint.get("task_scalers"), "checkpoint.task_scalers"
    )
    if set(scalers) != set(policy.task_names):
        raise InferenceCompatibilityError(
            "checkpoint.task_scalers do not cover exactly Human3"
        )
    for task_name in policy.task_names:
        scaler = _require_mapping(
            scalers[task_name], f"checkpoint.task_scalers.{task_name}"
        )
        if set(scaler) != {"mean", "std", "count"}:
            raise InferenceCompatibilityError(
                f"checkpoint.task_scalers.{task_name} has an invalid schema"
            )
        if (
            type(scaler["mean"]) is not float
            or not math.isfinite(scaler["mean"])
            or type(scaler["std"]) is not float
            or not math.isfinite(scaler["std"])
            or scaler["std"] <= 0
            or type(scaler["count"]) is not int
            or scaler["count"] < 0
        ):
            raise InferenceCompatibilityError(
                f"checkpoint.task_scalers.{task_name} values are invalid"
            )
    if type(checkpoint.get("routing_enabled")) is not bool:
        raise InferenceCompatibilityError("checkpoint.routing_enabled must be bool")
    if not isinstance(checkpoint.get("model_state"), Mapping) or not checkpoint["model_state"]:
        raise InferenceCompatibilityError("checkpoint.model_state is missing or empty")

    architecture = _require_mapping(
        checkpoint.get("architecture_config"), "checkpoint.architecture_config"
    )
    for name, expected in policy.model_identity.items():
        _require_exact(
            architecture, name, expected, "checkpoint.architecture_config"
        )
    return configuration, architecture


def _build_architecture_view(
    configuration: Mapping[str, Any],
    architecture: Mapping[str, Any],
    *,
    asset: AssetCompatibilityRule,
    policy: InferenceCompatibilityPolicy,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    expected = asset.adaptation_config
    for name in ADAPTATION_FIELDS:
        if name not in expected:
            raise InferenceCompatibilityError(
                f"policy adaptation config omits {name!r}"
            )

    if asset.migration_required:
        present = [name for name in ADAPTATION_FIELDS if name in architecture]
        if present:
            raise InferenceCompatibilityError(
                "legacy compatibility requires all nine architecture fields to "
                f"be absent; present={present}"
            )
        for name in CHECKPOINT_CONFIGURATION_FIELDS:
            _require_exact(
                configuration,
                name,
                expected[name],
                "checkpoint.configuration",
            )
        unexpected_constants = [
            name for name in policy.compatibility_constants if name in configuration
        ]
        if unexpected_constants:
            raise InferenceCompatibilityError(
                "legacy compatibility constants must be absent from the original "
                f"configuration; present={unexpected_constants}"
            )

        view = copy.deepcopy(dict(architecture))
        additions = []
        for name in ADAPTATION_FIELDS:
            source = (
                "checkpoint_configuration"
                if name in CHECKPOINT_CONFIGURATION_FIELDS
                else "codex_inference_compatibility_constant"
            )
            value = expected[name]
            if source == "codex_inference_compatibility_constant":
                _require_exact(
                    policy.compatibility_constants,
                    name,
                    value,
                    "compatibility_policy.constants",
                )
            view[name] = copy.deepcopy(value)
            additions.append(
                {
                    "field": name,
                    "before_present": False,
                    "before_value": None,
                    "after_value": _json_safe(value),
                    "source": source,
                }
            )
        return view, tuple(additions)

    # New checkpoints do not migrate at all.  Both metadata blocks must carry
    # every field with the exact audited type and value.
    view = copy.deepcopy(dict(architecture))
    for name in ADAPTATION_FIELDS:
        _require_exact(
            configuration, name, expected[name], "checkpoint.configuration"
        )
        _require_exact(
            architecture, name, expected[name], "checkpoint.architecture_config"
        )
    return view, ()


@dataclass(frozen=True)
class InferenceCompatibilityView:
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    run_id: str
    method: str
    seed: int
    operation: str
    split: str
    policy_version: str
    policy_sha256: str
    original_configuration: Mapping[str, Any]
    original_architecture_config: Mapping[str, Any]
    architecture_config_view: Mapping[str, Any]
    additions: tuple[Mapping[str, Any], ...]
    _policy: InferenceCompatibilityPolicy = field(repr=False, compare=False)

    @property
    def migration_applied(self) -> bool:
        return bool(self.additions)

    def architecture_for_trainer(
        self, path: str | Path, checkpoint: Any, args: Any
    ) -> dict[str, Any]:
        """Re-bind the view inside ``Trainer.load_checkpoint``.

        The trainer loads the file itself as before.  This method proves that
        the loaded bytes, payload identity, runtime mode/split, and current
        adaptation arguments still match the view before returning a copy for
        strict validation.
        """

        checkpoint_path = Path(path).resolve()
        if str(checkpoint_path) != self.checkpoint_path:
            raise InferenceCompatibilityError(
                "trainer checkpoint path differs from the compatibility binding"
            )
        if checkpoint_path.stat().st_size != self.checkpoint_size_bytes:
            raise InferenceCompatibilityError("trainer checkpoint size changed")
        if sha256_file(checkpoint_path) != self.checkpoint_sha256:
            raise InferenceCompatibilityError("trainer checkpoint SHA changed")
        asset = self._policy.asset(self.run_id)
        configuration, architecture = _validate_payload_identity(
            checkpoint, asset=asset, policy=self._policy
        )
        rebuilt, additions = _build_architecture_view(
            configuration, architecture, asset=asset, policy=self._policy
        )
        if not _strict_equal(rebuilt, dict(self.architecture_config_view)):
            raise InferenceCompatibilityError(
                "trainer payload does not reproduce the authorized compatibility view"
            )
        if not _strict_equal(list(additions), [dict(item) for item in self.additions]):
            raise InferenceCompatibilityError("compatibility audit additions changed")

        runtime = vars(args) if hasattr(args, "__dict__") else {}
        if runtime.get("mode") != ALLOWED_OPERATION:
            raise InferenceCompatibilityError(
                "trainer compatibility requires mode=batch_inference"
            )
        if runtime.get("inference_split") != self.split:
            raise InferenceCompatibilityError(
                "trainer inference_split differs from the compatibility binding"
            )
        _require_exact(runtime, "seed", self.seed, "runtime")
        for name in ADAPTATION_FIELDS:
            _require_exact(
                runtime,
                name,
                self.architecture_config_view[name],
                "runtime",
            )
        return copy.deepcopy(dict(self.architecture_config_view))

    def audit_record(self, *, code_identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "COMPATIBILITY_VIEW_PREPARED",
            "operation": self.operation,
            "split": self.split,
            "asset": {
                "run_id": self.run_id,
                "method": self.method,
                "seed": self.seed,
            },
            "checkpoint": {
                "path": self.checkpoint_path,
                "sha256": self.checkpoint_sha256,
                "size_bytes": self.checkpoint_size_bytes,
                "unchanged": True,
            },
            "policy": {
                "version": self.policy_version,
                "sha256": self.policy_sha256,
            },
            "migration_applied": self.migration_applied,
            "original_configuration": _json_safe(self.original_configuration),
            "original_architecture_config": _json_safe(
                self.original_architecture_config
            ),
            "compatibility_architecture_view": _json_safe(
                self.architecture_config_view
            ),
            "before_after_diff": [_json_safe(dict(item)) for item in self.additions],
            "code_identity": _json_safe(code_identity),
            "scientific_boundaries": {
                "checkpoint_rewritten": False,
                "model_tensors_changed": False,
                "init_overlay_applied": False,
                "training_authorized": False,
                "calibration_authorized": False,
            },
        }


@dataclass(frozen=True)
class PreparedInferenceCheckpoint:
    payload: Mapping[str, Any]
    compatibility: InferenceCompatibilityView


def _prepare_inference_checkpoint_with_policy(
    checkpoint_path: str | Path,
    *,
    run_id: str,
    method: str,
    seed: int,
    operation: str,
    split: str,
    expected_policy_sha256: str,
    policy: InferenceCompatibilityPolicy,
) -> PreparedInferenceCheckpoint:
    """Policy-parametric implementation used by isolated synthetic tests."""

    asset = _validate_request(
        run_id=run_id,
        method=method,
        seed=seed,
        operation=operation,
        split=split,
        expected_policy_sha256=expected_policy_sha256,
        policy=policy,
    )
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    size_bytes = path.stat().st_size
    if size_bytes != asset.checkpoint_size_bytes:
        raise InferenceCompatibilityError(
            f"checkpoint size does not match {asset.run_id}"
        )
    checkpoint_sha = sha256_file(path)
    if checkpoint_sha != asset.checkpoint_sha256:
        raise InferenceCompatibilityError(
            f"checkpoint SHA does not match {asset.run_id}"
        )

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    configuration, architecture = _validate_payload_identity(
        payload, asset=asset, policy=policy
    )
    view, additions = _build_architecture_view(
        configuration, architecture, asset=asset, policy=policy
    )
    compatibility = InferenceCompatibilityView(
        checkpoint_path=str(path),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_size_bytes=size_bytes,
        run_id=asset.run_id,
        method=asset.method,
        seed=asset.seed,
        operation=operation,
        split=split,
        policy_version=policy.version,
        policy_sha256=policy.computed_sha256(),
        original_configuration=_json_safe(configuration),
        original_architecture_config=_json_safe(architecture),
        architecture_config_view=copy.deepcopy(view),
        additions=tuple(copy.deepcopy(item) for item in additions),
        _policy=policy,
    )
    return PreparedInferenceCheckpoint(payload=payload, compatibility=compatibility)


def prepare_inference_checkpoint(
    checkpoint_path: str | Path,
    *,
    run_id: str,
    method: str,
    seed: int,
    operation: str,
    split: str,
    expected_policy_sha256: str,
) -> PreparedInferenceCheckpoint:
    """Load one of the exact 15 production checkpoints for controlled inference.

    The public entry point deliberately has no policy override.  Test policies
    use the private policy-parametric helper and therefore cannot accidentally
    broaden the production runner's asset allowlist.
    """

    return _prepare_inference_checkpoint_with_policy(
        checkpoint_path,
        run_id=run_id,
        method=method,
        seed=seed,
        operation=operation,
        split=split,
        expected_policy_sha256=expected_policy_sha256,
        policy=PRODUCTION_POLICY,
    )
