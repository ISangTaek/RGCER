"""Fail-closed authorization contract for fixed-checkpoint test inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
from typing import Any

from .checkpoint import validate_checkpoint_contract
from .code_identity import implementation_identity
from .constants import (
    BASELINE_TRAINING_GIT_COMMIT,
    BASELINE_TRAINING_IMPLEMENTATION_SHA256,
    CHEMPROP_COMMIT,
    DATASTORE_BUILD_ID,
    DATASTORE_FINGERPRINT,
    GROVER_COMMIT,
    METHODS,
    PROTOCOL_SHA256,
    SPLIT_MANIFEST_HASH,
)
from .utils import sha256_file


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_ROOT_FIELDS = frozenset({
    "schema_version",
    "task_id",
    "purpose",
    "allowed_splits",
    "inference_git_commit",
    "inference_implementation_sha256",
    "datastore_build_id",
    "datastore_fingerprint",
    "split_manifest_hash",
    "protocol_sha256",
    "assets",
})
_ASSET_FIELDS = frozenset({
    "asset_id",
    "method",
    "seed",
    "trial",
    "checkpoint_path",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "training_git_commit",
    "training_implementation_sha256",
})
_GRANT_MARKER = object()


class AuthorizationError(ValueError):
    """The authorization document or its bound request is invalid."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorizationError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AuthorizationError(f"Non-finite JSON number is forbidden: {value}")


def _require_exact_fields(payload: dict[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise AuthorizationError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuthorizationError(f"{field_name} must be a non-empty string")
    return value


def _require_integer(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise AuthorizationError(f"{field_name} must be an integer, not a boolean or another JSON type")
    if minimum is not None and value < minimum:
        raise AuthorizationError(f"{field_name} must be >= {minimum}")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    text = _require_string(value, field_name)
    if not _SHA256_RE.fullmatch(text):
        raise AuthorizationError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return text.lower()


def _require_commit(value: Any, field_name: str) -> str:
    text = _require_string(value, field_name)
    if not _COMMIT_RE.fullmatch(text):
        raise AuthorizationError(f"{field_name} must be a full 40-character hexadecimal Git commit")
    return text.lower()


def _is_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


@dataclass(frozen=True)
class AuthorizedAsset:
    asset_id: str
    method: str
    seed: int
    trial: int
    checkpoint_path: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    training_git_commit: str
    training_implementation_sha256: str


@dataclass(frozen=True)
class InferenceAuthorization:
    schema_version: int
    task_id: str
    purpose: str
    allowed_splits: tuple[str, ...]
    inference_git_commit: str
    inference_implementation_sha256: str
    datastore_build_id: str
    datastore_fingerprint: str
    split_manifest_hash: str
    protocol_sha256: str
    assets: tuple[AuthorizedAsset, ...]


class TestInferenceGrant:
    """Opaque capability issued only after the complete test preflight succeeds."""

    __slots__ = ("task_id", "asset_id", "method", "_marker")

    def __init__(self, task_id: str, asset_id: str, method: str, marker: object):
        if marker is not _GRANT_MARKER:
            raise AuthorizationError("TestInferenceGrant cannot be constructed directly")
        self.task_id = task_id
        self.asset_id = asset_id
        self.method = method
        self._marker = marker

    def require(self, *, split: str, method: str) -> None:
        if self._marker is not _GRANT_MARKER or split != "test" or method != self.method:
            raise PermissionError("The validated inference grant does not authorize this split/method")


@dataclass(frozen=True)
class AuthorizedInferenceContext:
    authorization: InferenceAuthorization
    authorization_path: str
    authorization_sha256: str
    asset: AuthorizedAsset
    checkpoint_sha256_before: str
    checkpoint_payload: dict[str, Any] = field(repr=False)
    scaler: Any = field(repr=False)
    inference_git_commit: str
    inference_implementation_sha256: str
    source_commit: str | None
    grant: TestInferenceGrant = field(repr=False)


def parse_authorization_text(text: str) -> InferenceAuthorization:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise AuthorizationError(f"Invalid authorization JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuthorizationError("Authorization JSON must be an object")
    _require_exact_fields(payload, _ROOT_FIELDS, "authorization")
    schema_version = _require_integer(payload["schema_version"], "schema_version")
    if schema_version != 1:
        raise AuthorizationError("schema_version must equal 1")
    purpose = _require_string(payload["purpose"], "purpose")
    if purpose != "fixed_checkpoint_test_export":
        raise AuthorizationError("purpose must equal 'fixed_checkpoint_test_export'")
    if payload["allowed_splits"] != ["test"]:
        raise AuthorizationError("allowed_splits must strictly equal ['test']")
    raw_assets = payload["assets"]
    if not isinstance(raw_assets, list) or not raw_assets:
        raise AuthorizationError("assets must be a non-empty list")
    assets: list[AuthorizedAsset] = []
    seen_asset_ids: set[str] = set()
    seen_method_seeds: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_assets):
        if not isinstance(raw, dict):
            raise AuthorizationError(f"assets[{index}] must be an object")
        _require_exact_fields(raw, _ASSET_FIELDS, f"assets[{index}]")
        asset_id = _require_string(raw["asset_id"], f"assets[{index}].asset_id")
        method = _require_string(raw["method"], f"assets[{index}].method")
        if method not in METHODS:
            raise AuthorizationError(f"assets[{index}].method is not an approved baseline: {method!r}")
        seed = _require_integer(raw["seed"], f"assets[{index}].seed")
        trial = _require_integer(raw["trial"], f"assets[{index}].trial", minimum=0)
        checkpoint_path = _require_string(raw["checkpoint_path"], f"assets[{index}].checkpoint_path")
        if not _is_absolute_path(checkpoint_path):
            raise AuthorizationError(f"assets[{index}].checkpoint_path must be absolute")
        asset = AuthorizedAsset(
            asset_id=asset_id,
            method=method,
            seed=seed,
            trial=trial,
            checkpoint_path=checkpoint_path,
            checkpoint_bytes=_require_integer(raw["checkpoint_bytes"], f"assets[{index}].checkpoint_bytes", minimum=1),
            checkpoint_sha256=_require_sha256(raw["checkpoint_sha256"], f"assets[{index}].checkpoint_sha256"),
            training_git_commit=_require_commit(raw["training_git_commit"], f"assets[{index}].training_git_commit"),
            training_implementation_sha256=_require_sha256(
                raw["training_implementation_sha256"],
                f"assets[{index}].training_implementation_sha256",
            ),
        )
        if asset.training_git_commit != BASELINE_TRAINING_GIT_COMMIT:
            raise AuthorizationError(
                f"assets[{index}].training_git_commit differs from the accepted Stage 3C1 training code"
            )
        if asset.training_implementation_sha256 != BASELINE_TRAINING_IMPLEMENTATION_SHA256:
            raise AuthorizationError(
                f"assets[{index}].training_implementation_sha256 differs from the accepted Stage 3C1 implementation"
            )
        if asset.asset_id in seen_asset_ids:
            raise AuthorizationError(f"Duplicate asset_id is forbidden: {asset.asset_id!r}")
        method_seed = (asset.method, asset.seed)
        if method_seed in seen_method_seeds:
            raise AuthorizationError(f"Duplicate method x seed asset is forbidden: {method_seed!r}")
        seen_asset_ids.add(asset.asset_id)
        seen_method_seeds.add(method_seed)
        assets.append(asset)
    return InferenceAuthorization(
        schema_version=schema_version,
        task_id=_require_string(payload["task_id"], "task_id"),
        purpose=purpose,
        allowed_splits=("test",),
        inference_git_commit=_require_commit(payload["inference_git_commit"], "inference_git_commit"),
        inference_implementation_sha256=_require_sha256(
            payload["inference_implementation_sha256"], "inference_implementation_sha256"
        ),
        datastore_build_id=_require_string(payload["datastore_build_id"], "datastore_build_id"),
        datastore_fingerprint=_require_sha256(payload["datastore_fingerprint"], "datastore_fingerprint"),
        split_manifest_hash=_require_sha256(payload["split_manifest_hash"], "split_manifest_hash"),
        protocol_sha256=_require_sha256(payload["protocol_sha256"], "protocol_sha256"),
        assets=tuple(assets),
    )


def load_authorization(path: str | Path, expected_sha256: str) -> tuple[InferenceAuthorization, str]:
    source = Path(path).resolve()
    expected = _require_sha256(expected_sha256, "authorization_sha256")
    actual = sha256_file(source)
    if actual != expected:
        raise AuthorizationError(f"Authorization SHA-256 mismatch: {actual} != {expected}")
    return parse_authorization_text(source.read_text(encoding="utf-8")), actual


def current_inference_identity(repository_root: str | Path) -> dict[str, str]:
    root = Path(repository_root).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise AuthorizationError(f"Current Git identity is not a full commit: {commit!r}")
    implementation = implementation_identity(root)["implementation_sha256"]
    return {"git_commit": commit, "implementation_sha256": implementation}


def _select_asset(authorization: InferenceAuthorization, asset_id: str) -> AuthorizedAsset:
    matches = [asset for asset in authorization.assets if asset.asset_id == asset_id]
    if len(matches) != 1:
        raise AuthorizationError(f"asset_id must select exactly one authorized asset: {asset_id!r}")
    return matches[0]


def _validate_checkpoint_payload(payload: dict[str, Any], asset: AuthorizedAsset) -> Any:
    scaler = validate_checkpoint_contract(payload, expected_method=asset.method)
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("role") != "FORMAL":
        raise AuthorizationError("Authorized test checkpoint must have FORMAL role")
    if type(payload.get("seed")) is not int or payload["seed"] != asset.seed:
        raise AuthorizationError("Checkpoint seed does not match the authorized asset")
    if type(config.get("seed")) is not int or config["seed"] != asset.seed:
        raise AuthorizationError("Checkpoint config seed does not match the authorized asset")
    if type(config.get("trial_index")) is not int or config["trial_index"] != asset.trial:
        raise AuthorizationError("Checkpoint trial does not match the authorized asset")
    if config.get("method") != asset.method:
        raise AuthorizationError("Checkpoint config method does not match the authorized asset")
    return scaler


def _source_commit(method: str, source_root: str | Path | None) -> str | None:
    expected = {"dmpnn": CHEMPROP_COMMIT, "grover": GROVER_COMMIT}.get(method)
    if expected is None:
        return None
    if source_root is None:
        raise AuthorizationError(f"{method} authorized test inference requires --source-root")
    root = Path(source_root).resolve()
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise AuthorizationError(f"{method} source commit mismatch: {actual!r} != {expected!r}")
    return actual


def authorize_test_inference(
    *,
    authorization_path: str | Path,
    authorization_sha256: str,
    task_id: str,
    asset_id: str,
    method: str,
    checkpoint: str | Path,
    repository_root: str | Path,
    source_root: str | Path | None = None,
) -> AuthorizedInferenceContext:
    authorization, actual_authorization_sha = load_authorization(authorization_path, authorization_sha256)
    if authorization.task_id != task_id:
        raise AuthorizationError("CLI task_id does not match authorization task_id")
    frozen_identity = {
        "datastore_build_id": DATASTORE_BUILD_ID,
        "datastore_fingerprint": DATASTORE_FINGERPRINT,
        "split_manifest_hash": SPLIT_MANIFEST_HASH,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    for field_name, expected in frozen_identity.items():
        if getattr(authorization, field_name) != expected:
            raise AuthorizationError(f"Authorization identity mismatch for {field_name}")
    current = current_inference_identity(repository_root)
    if authorization.inference_git_commit != current["git_commit"]:
        raise AuthorizationError("Authorization inference_git_commit does not match current checkout")
    if authorization.inference_implementation_sha256 != current["implementation_sha256"]:
        raise AuthorizationError("Authorization inference implementation identity does not match current code")
    asset = _select_asset(authorization, asset_id)
    if asset.method != method:
        raise AuthorizationError("CLI method does not match the authorized asset")
    checkpoint_path = Path(checkpoint).resolve()
    if os.path.normcase(str(checkpoint_path)) != os.path.normcase(asset.checkpoint_path):
        raise AuthorizationError("CLI checkpoint path does not match the authorized asset")
    if not checkpoint_path.is_file():
        raise AuthorizationError(f"Authorized checkpoint is not a file: {checkpoint_path}")
    actual_bytes = checkpoint_path.stat().st_size
    if actual_bytes != asset.checkpoint_bytes:
        raise AuthorizationError("Checkpoint byte size does not match the authorized asset")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != asset.checkpoint_sha256:
        raise AuthorizationError("Checkpoint SHA-256 does not match the authorized asset")
    from .inference import load_checkpoint

    payload, _ = load_checkpoint(checkpoint_path, expected_method=method)
    scaler = _validate_checkpoint_payload(payload, asset)
    source_commit = _source_commit(method, source_root)
    grant = TestInferenceGrant(task_id, asset_id, method, _GRANT_MARKER)
    return AuthorizedInferenceContext(
        authorization=authorization,
        authorization_path=str(Path(authorization_path).resolve()),
        authorization_sha256=actual_authorization_sha,
        asset=asset,
        checkpoint_sha256_before=checkpoint_sha,
        checkpoint_payload=payload,
        scaler=scaler,
        inference_git_commit=current["git_commit"],
        inference_implementation_sha256=current["implementation_sha256"],
        source_commit=source_commit,
        grant=grant,
    )
