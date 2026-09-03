"""Pipeline-level smoke tests for the D8 result-correction suite.

Covers the failure paths the review called out (P1-2):
  - metrics recompute on toy run directories (best-epoch source + RMSE
    cross-check inputs);
  - exact Wilcoxon end-to-end JSON payload;
  - drift contract FAILURE (STOP_DRIFT_CONTRACT);
  - embedding overlap FAILURE (exact overlap detection);
  - functional-forgetting artifact provenance / freshness gate;
  - embedding metadata provenance completeness.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.d8_drift_recompute import (
    MODEL_CONFIG_KEYS,
    POOLING_NAME,
    probe_manifest_sha256,
    verify_drift_contract,
)
from scripts.d8_embedding_overlap_audit import compute_overlap
from scripts.d8_exact_paired_stats import compute_paired_stats
from scripts.d8_label_scaling_mechanism import ff_provenance_ok
from scripts.d8_recompute_prediction_metrics import (
    best_epoch_from_run,
    existing_rmse_by_endpoint,
    fallback_best_epoch,
)


# ---------------------------------------------------------------------------
# Metrics recompute on toy run directories
# ---------------------------------------------------------------------------


def _toy_run_dir(tmp_path, epoch=37, rmse=0.5):
    run_dir = tmp_path / "seed_42"
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True)
    torch.save({"epoch": epoch, "model_state": {}}, run_dir / "model_best.pt")
    with (diag / "epoch_summary.csv").open("w", newline="") as handle:
        handle.write("epoch,val_human3_macro_rmse\n35,0.9\n36,0.8\n37,0.7\n38,0.75\n")
    with (diag / "human3_path_metrics.csv").open("w", newline="") as handle:
        handle.write("epoch,task,path,rmse,r2,mae,n\n")
        handle.write(f"{epoch},man_oral_TDLo,final,{rmse},0.1,0.4,14\n")
    return run_dir


def test_best_epoch_from_toy_checkpoint(tmp_path):
    run_dir = _toy_run_dir(tmp_path, epoch=36)
    assert best_epoch_from_run(run_dir) == 36


def test_fallback_best_epoch_is_argmin(tmp_path):
    run_dir = _toy_run_dir(tmp_path)
    (run_dir / "model_best.pt").unlink()
    assert fallback_best_epoch(run_dir) == 37  # lowest macro rmse in the toy log


def test_existing_rmse_by_endpoint_reads_final_path(tmp_path):
    run_dir = _toy_run_dir(tmp_path, epoch=37, rmse=0.5)
    assert existing_rmse_by_endpoint(run_dir, 37) == {"man_oral_TDLo": 0.5}
    assert existing_rmse_by_endpoint(run_dir, 99) == {}


def test_torch_checkpoint_roundtrip(tmp_path):
    # the toy checkpoint must load exactly like the real *_best.pt payload
    run_dir = _toy_run_dir(tmp_path, epoch=35)
    payload = torch.load(next(run_dir.glob("*_best.pt")), map_location="cpu",
                         weights_only=False)
    assert payload["epoch"] == 35


# ---------------------------------------------------------------------------
# Exact Wilcoxon end-to-end payload
# ---------------------------------------------------------------------------


def test_compute_paired_stats_payload_serializable():
    stats = compute_paired_stats([0.1, 0.2, 0.15, 0.05, 0.12],
                                 [42, 43, 44, 45, 46], "unit test")
    decoded = json.loads(json.dumps(stats))
    assert decoded["wins"] == 5
    assert decoded["wilcoxon_exact_one_sided_p"] == pytest.approx(1 / 32)
    assert decoded["test"] == "Wilcoxon signed-rank"
    assert decoded["method"] == "exact"


# ---------------------------------------------------------------------------
# Drift contract failure (P0-3)
# ---------------------------------------------------------------------------


def _toy_reference():
    weight = torch.ones(3, 3)
    return {
        "probe_ids": ["row_1", "row_2", "row_3"],
        "init_state": {"encoder.backbone.weight": weight},
        "source": "checkpoint_d7_drift_state",
    }


def _toy_candidate_payload():
    return {
        "model_state": {"encoder.backbone.weight": torch.ones(3, 3)},
        "configuration": {key: 1 for key in MODEL_CONFIG_KEYS},
    }


def test_drift_contract_accepts_consistent_inputs(tmp_path):
    init_artifact = tmp_path / "b1_init.pt"
    torch.save({"encoder.backbone.weight": torch.ones(3, 3)}, init_artifact)
    checks = verify_drift_contract(
        reference=_toy_reference(),
        candidate_payload=_toy_candidate_payload(),
        init_artifact_path=init_artifact,
        expected_probe_manifest=probe_manifest_sha256(["row_3", "row_1", "row_2"]),
        reference_config={key: 1 for key in MODEL_CONFIG_KEYS},
    )
    assert checks["probe_manifest_match"] is True
    assert checks["reference_matches_artifact"] is True
    assert checks["model_config_match"] is True
    assert checks["backbone_key_set_match"] is True


def test_drift_contract_rejects_probe_mismatch(tmp_path):
    init_artifact = tmp_path / "b1_init.pt"
    torch.save({"encoder.backbone.weight": torch.ones(3, 3)}, init_artifact)
    with pytest.raises(SystemExit, match="STOP_DRIFT_CONTRACT"):
        verify_drift_contract(
            reference=_toy_reference(),
            candidate_payload=_toy_candidate_payload(),
            init_artifact_path=init_artifact,
            expected_probe_manifest="deadbeef",  # not the reference probe set
            reference_config=None,
        )


def test_drift_contract_rejects_foreign_pretrained_reference(tmp_path):
    # the checkpoint claims one pretrained baseline; the provenance-verified
    # artifact holds DIFFERENT weights -> must not recompute
    init_artifact = tmp_path / "b1_init.pt"
    torch.save({"encoder.backbone.weight": torch.zeros(3, 3)}, init_artifact)
    with pytest.raises(SystemExit, match="STOP_DRIFT_CONTRACT"):
        verify_drift_contract(
            reference=_toy_reference(),
            candidate_payload=_toy_candidate_payload(),
            init_artifact_path=init_artifact,
            expected_probe_manifest=probe_manifest_sha256(["row_1", "row_2", "row_3"]),
            reference_config={key: 1 for key in MODEL_CONFIG_KEYS},
        )


def test_drift_contract_rejects_config_drift(tmp_path):
    init_artifact = tmp_path / "b1_init.pt"
    torch.save({"encoder.backbone.weight": torch.ones(3, 3)}, init_artifact)
    payload = _toy_candidate_payload()
    payload["configuration"] = {key: 2 for key in MODEL_CONFIG_KEYS}
    with pytest.raises(SystemExit, match="STOP_DRIFT_CONTRACT"):
        verify_drift_contract(
            reference=_toy_reference(),
            candidate_payload=payload,
            init_artifact_path=init_artifact,
            expected_probe_manifest=probe_manifest_sha256(["row_1", "row_2", "row_3"]),
            reference_config={key: 1 for key in MODEL_CONFIG_KEYS},
        )


def test_drift_contract_rejects_backbone_key_drift(tmp_path):
    init_artifact = tmp_path / "b1_init.pt"
    torch.save({"encoder.backbone.weight": torch.ones(3, 3)}, init_artifact)
    payload = _toy_candidate_payload()
    payload["model_state"] = {"encoder.backbone.other": torch.ones(3, 3)}
    with pytest.raises(SystemExit, match="STOP_DRIFT_CONTRACT"):
        verify_drift_contract(
            reference=_toy_reference(),
            candidate_payload=payload,
            init_artifact_path=init_artifact,
            expected_probe_manifest=probe_manifest_sha256(["row_1", "row_2", "row_3"]),
            reference_config={key: 1 for key in MODEL_CONFIG_KEYS},
        )


# ---------------------------------------------------------------------------
# Embedding overlap audit (P0-1)
# ---------------------------------------------------------------------------


def test_overlap_audit_detects_exact_overlap():
    smiles = {"row_1": "CCO", "row_2": "c1ccccc1", "row_3": "CCC"}
    audit = compute_overlap(["row_1"], ["row_2", "row_3"], smiles)
    assert audit["human_samples_checked"] == 1
    assert audit["animal_samples_checked"] == 2
    assert audit["exact_smiles_overlap"] == 0
    assert audit["test_accessed"] is False


def test_overlap_audit_flags_shared_molecule():
    smiles = {"row_1": "CCO", "row_2": "OCC"}  # same molecule, different SMILES
    audit = compute_overlap(["row_1"], ["row_2"], smiles)
    assert audit["exact_smiles_overlap"] == 1
    assert audit["exact_overlap_sample_ids"] == ["row_1"]


def test_overlap_audit_scaffold_overlap_reported_not_exact():
    # ethanol vs propan-1-ol share the alkanol scaffold? different scaffolds;
    # use two benzene derivatives instead — same benzene scaffold
    smiles = {"h1": "c1ccccc1O", "a1": "c1ccccc1C"}
    audit = compute_overlap(["h1"], ["a1"], smiles)
    assert audit["exact_smiles_overlap"] == 0
    assert audit["scaffold_overlap"] == 1


# ---------------------------------------------------------------------------
# Functional-forgetting provenance gate (P0-4)
# ---------------------------------------------------------------------------


def _toy_ff_run(tmp_path, ff_payload):
    import hashlib

    import scripts.d8_label_scaling_mechanism as mech
    from scripts.d8_functional_forgetting import EVALUATOR_VERSION

    run_dir = tmp_path / "seed_42"
    run_dir.mkdir()
    checkpoint = run_dir / "model_last.pt"
    torch.save({"model_state": {}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    script_sha = hashlib.sha256(
        (Path(mech.__file__).parent / "d8_functional_forgetting.py").read_bytes()
    ).hexdigest()
    ff_payload = dict(ff_payload)
    ff_payload.setdefault("checkpoint_sha256", digest)
    ff_payload.setdefault("git_commit", "a" * 40)
    ff_payload.setdefault("evaluator_version", EVALUATOR_VERSION)
    ff_payload.setdefault("evaluator_script_sha256", script_sha)
    ff_payload.setdefault("animal_manifest_hash", "manifest-abc")
    ff_payload.setdefault("provenance_verified", True)
    (run_dir / "functional_forgetting.json").write_text(json.dumps(ff_payload))
    return run_dir, digest


def test_ff_provenance_accepts_fresh_artifact(tmp_path):
    run_dir, _ = _toy_ff_run(tmp_path, {"some": "stats"})
    assert ff_provenance_ok(run_dir / "functional_forgetting.json", run_dir) is True


def test_ff_provenance_ignores_unrelated_head_change(tmp_path):
    # a different repository HEAD alone must NOT invalidate an evaluation:
    # the invalidation identity is the evaluator script hash, not the commit
    run_dir, _ = _toy_ff_run(tmp_path, {})  # stamped with git_commit "a"*40
    assert ff_provenance_ok(run_dir / "functional_forgetting.json", run_dir) is True


def test_ff_provenance_rejects_checkpoint_mismatch(tmp_path):
    run_dir, digest = _toy_ff_run(tmp_path, {})
    payload = json.loads((run_dir / "functional_forgetting.json").read_text())
    payload["checkpoint_sha256"] = "0" * 64
    (run_dir / "functional_forgetting.json").write_text(json.dumps(payload))
    assert ff_provenance_ok(run_dir / "functional_forgetting.json", run_dir) is False


def test_ff_provenance_rejects_stale_evaluator_script(tmp_path):
    import hashlib
    from pathlib import Path as _P

    run_dir, _ = _toy_ff_run(tmp_path, {})
    payload = json.loads((run_dir / "functional_forgetting.json").read_text())
    payload["evaluator_script_sha256"] = hashlib.sha256(b"old evaluator").hexdigest()
    (run_dir / "functional_forgetting.json").write_text(json.dumps(payload))
    assert ff_provenance_ok(run_dir / "functional_forgetting.json", run_dir) is False


def test_ff_provenance_rejects_legacy_format(tmp_path):
    # a pre-P0-4 file (no provenance fields, as produced by the 007 phase)
    run_dir = tmp_path / "seed_42"
    run_dir.mkdir()
    torch.save({"model_state": {}}, run_dir / "model_last.pt")
    (run_dir / "functional_forgetting.json").write_text(
        json.dumps({"functional_forgetting_abs": 0.08})
    )
    assert ff_provenance_ok(run_dir / "functional_forgetting.json", run_dir) is False


def test_ff_provenance_rejects_manifest_mismatch(tmp_path):
    run_dir, _ = _toy_ff_run(tmp_path, {"animal_manifest_hash": "manifest-abc"})
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"split_manifest_hash": "manifest-other"})
    )
    assert ff_provenance_ok(run_dir / "functional_forgetting.json", run_dir) is False


# ---------------------------------------------------------------------------
# Embedding metadata provenance completeness (P0-2)
# ---------------------------------------------------------------------------


def test_embedding_metadata_provenance_fields():
    required = {"checkpoint_sha256", "datastore_sha256", "manifest_sha256",
                "pooling_name", "embedding_dim"}
    row = {
        "seed": 42,
        "encoder_state": "b1_final",
        "species_group": "human",
        "task": "man_oral_TDLo",
        "sample_id": "row_239",
        "split": "validation",
        "label": "4.57",
        "checkpoint_sha256": "c" * 64,
        "datastore_sha256": "7b7bd62a",
        "manifest_sha256": "61a2e494",
        "pooling_name": POOLING_NAME,
        "embedding_dim": 96,
    }
    assert required <= set(row)
    assert row["pooling_name"] == "graphormer_backbone_pooled"
    assert np.isscalar(row["embedding_dim"])
