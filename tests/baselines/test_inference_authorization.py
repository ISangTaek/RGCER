from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baselines.inference_authorization import (
    AuthorizationError,
    authorize_test_inference,
    parse_authorization_text,
)
from baselines.utils import sha256_file
from conftest import ROOT, authorization_payload


def _write_authorization(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def _authorize(path: Path, checkpoint: Path, **overrides):
    arguments = {
        "authorization_path": path,
        "authorization_sha256": sha256_file(path),
        "task_id": "S3C2A_SYNTHETIC_TEST",
        "asset_id": "rf-t02-s42",
        "method": "rf",
        "checkpoint": checkpoint,
        "repository_root": ROOT,
    }
    arguments.update(overrides)
    return authorize_test_inference(**arguments)


def test_valid_authorization_issues_test_grant(tmp_path, formal_rf_checkpoint):
    checkpoint, _ = formal_rf_checkpoint
    path = _write_authorization(tmp_path, authorization_payload(checkpoint))
    context = _authorize(path, checkpoint)
    context.grant.require(split="test", method="rf")
    assert context.asset.seed == 42
    assert context.asset.trial == 2
    assert context.checkpoint_sha256_before == sha256_file(checkpoint)


@pytest.mark.parametrize(
    "raw,match",
    [
        ('{"schema_version":1,"schema_version":1}', "Duplicate JSON key"),
        ('{"value":NaN}', "Non-finite JSON number"),
        ("[]", "must be an object"),
        ("{", "Invalid authorization JSON"),
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_and_malformed(raw, match):
    with pytest.raises(AuthorizationError, match=match):
        parse_authorization_text(raw)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(unknown=True), "unknown"),
        (lambda p: p.update(allowed_splits=["validation", "test"]), "strictly equal"),
        (lambda p: p.update(purpose="anything_else"), "purpose"),
        (lambda p: p["assets"][0].update(seed=True), "integer"),
        (lambda p: p["assets"][0].update(method="unknown"), "approved baseline"),
        (lambda p: p["assets"][0].update(extra="forbidden"), "unknown"),
        (lambda p: p["assets"].append(copy.deepcopy(p["assets"][0])), "Duplicate asset_id"),
        (
            lambda p: p["assets"].append({**copy.deepcopy(p["assets"][0]), "asset_id": "another"}),
            "Duplicate method x seed",
        ),
        (lambda p: p["assets"][0].update(training_git_commit="0" * 40), "accepted Stage 3C1"),
        (
            lambda p: p["assets"][0].update(training_implementation_sha256="0" * 64),
            "accepted Stage 3C1",
        ),
    ],
)
def test_authorization_schema_rejects_ambiguity(tmp_path, formal_rf_checkpoint, mutation, match):
    checkpoint, _ = formal_rf_checkpoint
    payload = authorization_payload(checkpoint)
    mutation(payload)
    path = _write_authorization(tmp_path, payload)
    with pytest.raises(AuthorizationError, match=match):
        _authorize(path, checkpoint)


@pytest.mark.parametrize(
    "mutation,override,match",
    [
        (lambda p: p.update(task_id="other"), {}, "task_id"),
        (lambda p: p.update(datastore_build_id="other"), {}, "datastore_build_id"),
        (lambda p: p.update(datastore_fingerprint="0" * 64), {}, "datastore_fingerprint"),
        (lambda p: p.update(split_manifest_hash="0" * 64), {}, "split_manifest_hash"),
        (lambda p: p.update(protocol_sha256="0" * 64), {}, "protocol_sha256"),
        (lambda p: p.update(inference_git_commit="0" * 40), {}, "inference_git_commit"),
        (
            lambda p: p.update(inference_implementation_sha256="0" * 64),
            {},
            "implementation identity",
        ),
        (lambda p: None, {"asset_id": "missing"}, "exactly one"),
        (lambda p: None, {"method": "afp"}, "CLI method"),
        (lambda p: p["assets"][0].update(seed=43), {}, "Checkpoint seed"),
        (lambda p: p["assets"][0].update(trial=3), {}, "Checkpoint trial"),
        (lambda p: p["assets"][0].update(checkpoint_bytes=1), {}, "byte size"),
        (lambda p: p["assets"][0].update(checkpoint_sha256="0" * 64), {}, "SHA-256"),
    ],
)
def test_each_bound_identity_mismatch_fails_closed(
    tmp_path,
    formal_rf_checkpoint,
    mutation,
    override,
    match,
):
    checkpoint, _ = formal_rf_checkpoint
    payload = authorization_payload(checkpoint)
    mutation(payload)
    path = _write_authorization(tmp_path, payload)
    with pytest.raises(AuthorizationError, match=match):
        _authorize(path, checkpoint, **override)


def test_authorization_file_sha_mismatch_fails_before_parsing(tmp_path, formal_rf_checkpoint):
    checkpoint, _ = formal_rf_checkpoint
    path = _write_authorization(tmp_path, authorization_payload(checkpoint))
    with pytest.raises(AuthorizationError, match="Authorization SHA-256 mismatch"):
        _authorize(path, checkpoint, authorization_sha256="0" * 64)


def test_checkpoint_path_mismatch_is_rejected(tmp_path, formal_rf_checkpoint):
    checkpoint, _ = formal_rf_checkpoint
    payload = authorization_payload(checkpoint)
    payload["assets"][0]["checkpoint_path"] = str((tmp_path / "other.joblib").resolve())
    path = _write_authorization(tmp_path, payload)
    with pytest.raises(AuthorizationError, match="checkpoint path"):
        _authorize(path, checkpoint)


def test_checkpoint_role_must_be_formal(tmp_path, formal_rf_checkpoint):
    checkpoint, payload = formal_rf_checkpoint
    payload = copy.deepcopy(payload)
    payload["config"]["role"] = "SMOKE"
    import joblib

    joblib.dump(payload, checkpoint)
    path = _write_authorization(tmp_path, authorization_payload(checkpoint))
    with pytest.raises(AuthorizationError, match="FORMAL role"):
        _authorize(path, checkpoint)


def test_invalid_authorization_never_reaches_test_reader(
    tmp_path,
    formal_rf_checkpoint,
    monkeypatch,
):
    checkpoint, _ = formal_rf_checkpoint
    payload = authorization_payload(checkpoint)
    payload["task_id"] = "wrong"
    path = _write_authorization(tmp_path, payload)

    import importlib.util

    script = ROOT / "scripts" / "baseline_predict.py"
    spec = importlib.util.spec_from_file_location("baseline_predict_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "load_authorized_inference_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("test reader was called")),
    )
    with pytest.raises(AuthorizationError, match="task_id"):
        module.main([
            "--checkpoint", str(checkpoint),
            "--method", "rf",
            "--datastore", "unused",
            "--output", str(tmp_path / "output"),
            "--split", "test",
            "--authorization", str(path),
            "--authorization-sha256", sha256_file(path),
            "--task-id", "S3C2A_SYNTHETIC_TEST",
            "--asset-id", "rf-t02-s42",
        ])
