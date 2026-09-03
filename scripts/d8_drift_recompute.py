"""D8 result-correction Tasks C + D(drift): direct drift recompute.

Task C (§17-§23): S1 seeds 43/45 lack an epoch-39 feature-drift value in
their drift CSVs (the parameter-drift columns are present and exactly 0).
Because S1 freezes the backbone this is expected to be 0, but Figure 2
requires a direct recompute rather than an inference.  The recompute REUSES
the checkpoint's own ``d7_drift_state`` — the original 128 probe ids, the
pretrained reference representations and the frozen-init backbone tensors —
so nothing is re-sampled (§18) and the reference is the true Animal56
pretrained state (§19).

Task D (§24-§32): B1 label-scaling runs (10/25/50/75%) trained without
in-run drift logging.  Their final drift — and the 100% row as well — is
recomputed against the SAME per-seed Animal56-pretrained reference and the
seed's 100% probe, so every row of the Figure 4 drift table sits on one
scale.  (The D7-era in-run logs of seeds 42/44/46 are historical
measurements from older code and are not patched or mixed in.)

Machinery self-check: on B1 seed 43 — a current-code run whose checkpoint
embeds the original drift state and whose epoch-39 feature drift was
logged — the recompute must reproduce the logged value (verified to
~5e-9); otherwise STOP_DRIFT_SELFCHECK.

Outputs (under --output_dir):
    D8_S1_FINAL_DRIFT_PATCH.csv            (§22 fields + provenance)
    D8_REPRESENTATION_DRIFT_PATCHED.csv    original matrix + patched rows
    D8_LABEL_SCALING_B1_DRIFT.csv          fraction,seed,...,source
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SELF_CHECK_RTOL = 1e-3
SELF_CHECK_ATOL = 1e-4
CONFIG_KEYS = (
    "data_store_dir", "split_seed", "splitting", "vs", "calibration_size", "ts",
    "hidden_dim", "mid_dim", "a_layers", "a_heads", "edge_bias_mode",
    "spatial_pos_clip", "max_nodes_filter", "prediction_mode",
    "head_hidden_dim", "head_dropout",
)
# Result-correction review P0-3: model-defining keys that must agree between
# the candidate checkpoint and the reference source before any drift
# recompute may run.
MODEL_CONFIG_KEYS = (
    "hidden_dim", "mid_dim", "a_layers", "a_heads", "edge_bias_mode",
    "head_hidden_dim", "head_dropout", "prediction_mode",
)
POOLING_NAME = "graphormer_backbone_pooled"

_EXPECTED_PROBE_CACHE: dict[str, str] = {}


def expected_probe_manifest_cached(template_config: dict) -> str:
    """Deterministic §47 probe identity for a data configuration, cached —
    the probe rule (first 128 sorted full-train ids) is a pure function of
    (data_store_dir, split_seed, splitting, val/test sizes)."""
    key = json.dumps(
        {name: template_config.get(name) for name in (
            "data_store_dir", "split_seed", "splitting", "vs",
            "calibration_size", "ts",
        )},
        sort_keys=True, default=str,
    )
    if key not in _EXPECTED_PROBE_CACHE:
        from d7_diagnostics import select_probe_ids, train_sample_ids

        loaders = build_probe_loaders(template_config)
        ids = select_probe_ids(train_sample_ids(loaders["train"]), 128)
        _EXPECTED_PROBE_CACHE[key] = probe_manifest_sha256(ids)
    return _EXPECTED_PROBE_CACHE[key]


def verify_drift_contract(
    *,
    reference: dict,
    candidate_payload: dict,
    init_artifact_path: Path,
    expected_probe_manifest: str,
    reference_config: dict | None = None,
) -> dict:
    """P0-3 preflight: refuse to recompute drift when probe identity,
    pretrained-reference authenticity or model configuration disagree —
    otherwise a wrong-but-plausible drift number could slip through.

    Checks
      1. probe_manifest_match      reference probe ids == the deterministic
                                   §47 probe for the data configuration
      2. reference_matches_artifact checkpoint-embedded pretrained baseline
                                   tensors == the provenance-verified
                                   b1_init artifact (key-exact, bit-exact)
      3. model_config_match        candidate configuration == the reference
                                   source's configuration (when the
                                   reference came from a checkpoint)
      4. backbone_key_set_match    candidate backbone parameter names ==
                                   reference init_state names

    Raises SystemExit("STOP_DRIFT_CONTRACT") on any failure.
    """
    import torch

    candidate_config = dict(candidate_payload.get("configuration") or {})
    checks: dict = {}
    actual_manifest = probe_manifest_sha256(reference["probe_ids"])
    checks["probe_manifest_match"] = actual_manifest == expected_probe_manifest
    checks["probe_manifest_sha256"] = actual_manifest
    checks["expected_probe_manifest_sha256"] = expected_probe_manifest

    artifact_state = torch.load(init_artifact_path, map_location="cpu", weights_only=False)
    reference_state = reference["init_state"]
    artifact_backbone = {
        key: value for key, value in artifact_state.items() if key in reference_state
    }
    keys_match = set(artifact_backbone) == set(reference_state) and bool(reference_state)
    tensors_match = keys_match and all(
        torch.equal(
            torch.as_tensor(reference_state[key]).cpu(),
            torch.as_tensor(artifact_backbone[key]).cpu(),
        )
        for key in reference_state
    )
    checks["reference_matches_artifact"] = bool(tensors_match)

    if reference_config:
        checks["model_config_match"] = all(
            candidate_config.get(name) == reference_config.get(name)
            for name in MODEL_CONFIG_KEYS
        )
    else:
        checks["model_config_match"] = True  # derived from the artifact itself

    candidate_keys = {
        key for key in candidate_payload["model_state"]
        if key.startswith("encoder.backbone.")
    }
    checks["backbone_key_set_match"] = candidate_keys == set(reference_state)

    failures = [name for name, ok in checks.items() if ok is False]
    if failures:
        raise SystemExit(f"STOP_DRIFT_CONTRACT: failed checks={failures}")
    return checks


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_manifest_sha256(probe_ids) -> str:
    payload = "\n".join(sorted(str(value) for value in probe_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def last_checkpoint(run_dir: Path):
    import torch

    last_pts = sorted(Path(run_dir).glob("*_last.pt"))
    if len(last_pts) != 1:
        raise RuntimeError(f"expected exactly one *_last.pt in {run_dir}, found {len(last_pts)}")
    return torch.load(last_pts[0], map_location="cpu", weights_only=False), last_pts[0]


def _params_from_config(template_config: dict, train_fraction: float = 1.0):
    from main import build_parser

    params = build_parser().parse_args([])
    for key in CONFIG_KEYS:
        if key in template_config:
            setattr(params, key, template_config[key])
    setattr(params, "arch", "Graphormer")
    setattr(params, "toxacute_task_scope", "human3")
    setattr(params, "seed", int(template_config.get("seed", 42)))
    setattr(params, "fit_conformal", False)
    setattr(params, "train_eval_scope", "validation_only")
    setattr(params, "card_lambda_delta", 0.0)
    setattr(params, "train_fraction", float(train_fraction))
    return params


def build_model(template_config: dict, device):
    """Build the Graphormer architecture from a checkpoint's configuration."""
    from main import _build_model_components, task_names_for_params, validate_params

    params = _params_from_config(template_config)
    validate_params(params)
    task_names = task_names_for_params(params)
    encoder_class, architecture_class, decoders = _build_model_components(
        params, task_names, device
    )
    return architecture_class(task_names, encoder_class, decoders, device, params, **({}))


def load_model_from_state(state, template_config: dict, device):
    import torch

    model = build_model(template_config, device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_probe_loaders(template_config: dict):
    """Rebuild the human3 loaders (full train) exactly as the runs used."""
    from dataset import DataCollator
    from main import _loaders, task_names_for_params

    params = _params_from_config(template_config)
    collator = DataCollator(spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None)
    return _loaders(params, task_names_for_params(params), collator)


def build_probe_batches(template_config: dict, probe_ids, device):
    """Exact probe batches (probe order, the loaders' own collator)."""
    from d7_diagnostics import collect_probe_batches

    return collect_probe_batches(build_probe_loaders(template_config)["train"], probe_ids)


def derive_probe_and_reference(init_artifact: Path, template_config: dict, device) -> dict:
    """Deterministic probe (§47 rule: first 128 sorted full-train ids) and
    pretrained reference from the b1_init artifact — used when the run's
    checkpoint predates drift-state checkpointing."""
    import torch

    from d7_diagnostics import (
        pooled_representations, select_probe_ids, train_sample_ids,
    )
    from main import _loaders, task_names_for_params

    params = _params_from_config(template_config)
    from dataset import DataCollator

    collator = DataCollator(spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None)
    loaders = _loaders(params, task_names_for_params(params), collator)
    probe_ids = select_probe_ids(train_sample_ids(loaders["train"]), 128)
    batches = build_probe_batches(template_config, probe_ids, device)
    init_state = torch.load(init_artifact, map_location="cpu", weights_only=False)
    reference_model = load_model_from_state(init_state, template_config, device)
    reference = pooled_representations(reference_model, batches, device, expected_ids=probe_ids)
    backbone_state = {
        key: value.detach().cpu().clone()
        for key, value in init_state.items()
        if key.startswith("encoder.backbone.")
    }
    return {
        "probe_ids": [str(value) for value in probe_ids],
        "reference_representations": reference,
        "init_state": backbone_state,
        "source": f"derived_from_artifact:{Path(init_artifact).name}",
    }


def drift_state_or_derived(payload, init_artifact: Path, device) -> dict:
    """Prefer the checkpoint's original drift state; else derive it."""
    state = payload.get("d7_drift_state")
    if state and state.get("probe_ids"):
        return {
            "probe_ids": [str(value) for value in state["probe_ids"]],
            "reference_representations": dict(state["reference_representations"]),
            "init_state": dict(state["init_state"]),
            "source": "checkpoint_d7_drift_state",
        }
    return derive_probe_and_reference(init_artifact, dict(payload.get("configuration") or {}), device)


def recompute_for_run(run_dir: Path, reference: dict, device) -> dict:
    """Final-checkpoint drift of ``run_dir`` against ``reference``."""
    from d7_diagnostics import (
        backbone_parameter_drift, feature_drift, pooled_representations,
    )

    payload, ckpt_path = last_checkpoint(run_dir)
    model = load_model_from_state(
        payload["model_state"], dict(payload.get("configuration") or {}), device
    )
    probe_ids = reference["probe_ids"]
    batches = build_probe_batches(dict(payload.get("configuration") or {}), probe_ids, device)
    current = pooled_representations(model, batches, device, expected_ids=probe_ids)
    feature = feature_drift(current, reference["reference_representations"])
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    all_d, early_d, late_d = backbone_parameter_drift(state_dict, reference["init_state"])
    return {
        "epoch": int(payload.get("epoch", -1)),
        "backbone_param_drift": all_d,
        "early_block_drift": early_d,
        "late_block_drift": late_d,
        "feature_drift": feature,
        "candidate_checkpoint": ckpt_path,
    }


def read_drift_csv_row(run_dir: Path, epoch: int) -> dict | None:
    path = Path(run_dir) / "diagnostics" / "d7_representation_drift.csv"
    if not path.is_file():
        return None
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if int(row["epoch"]) == int(epoch):
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--scaling_root", default="artifacts/runs/d8_label_scaling")
    parser.add_argument("--inits_dir", default="artifacts/inits")
    parser.add_argument("--output_dir", default="artifacts/results/d8_result_correction/drift")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--fractions", nargs="+", type=int, default=[10, 25, 50, 75, 100])
    parser.add_argument("--skip_selfcheck", action="store_true")
    args = parser.parse_args()

    import torch

    if str(args.gpu_id) != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def formal_dir(method: str, seed: int) -> Path:
        if method == "s1":
            return Path(args.d7_root) / "d7_stage_b" / "s1" / "d7_s1_e40" / f"seed_{seed}"
        return Path(args.d6_root) / "d6_stage_b" / method / f"d6_{method}_e40" / f"seed_{seed}"

    def init_artifact_for(seed: int) -> Path:
        for candidate in (
            Path(args.inits_dir) / "d7" / f"b1_init_seed{seed}.pt",
            Path(args.inits_dir) / "d8_stage0" / f"b1_init_seed{seed}.pt",
            Path(args.inits_dir) / "d7_stage_b" / f"b1_init_seed{seed}.pt",
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no b1_init artifact for seed {seed}")

    # ---- machinery self-check: reproduce a logged epoch-39 value ----------
    # Anchor: B1 seed 43 — a CURRENT-code run whose checkpoint embeds the
    # original drift state AND whose epoch-39 feature drift was logged.
    # (The D7-era runs s42/44/46 logged drift with older code and carry no
    # embedded state; their logged values are not reproducible bit-exactly
    # and are treated as historical in-run measurements, never patched.)
    contract_audit: dict = {}
    if not args.skip_selfcheck:
        check_seed = 43
        payload, _ = last_checkpoint(formal_dir("b1", check_seed))
        reference = drift_state_or_derived(payload, init_artifact_for(check_seed), device)
        candidate_config = dict(payload.get("configuration") or {})
        contract_audit[f"selfcheck_b1_s{check_seed}"] = verify_drift_contract(
            reference=reference,
            candidate_payload=payload,
            init_artifact_path=init_artifact_for(check_seed),
            expected_probe_manifest=expected_probe_manifest_cached(candidate_config),
            reference_config=(
                candidate_config
                if reference["source"] == "checkpoint_d7_drift_state" else None
            ),
        )
        recomputed = recompute_for_run(formal_dir("b1", check_seed), reference, device)
        logged = read_drift_csv_row(formal_dir("b1", check_seed), 39)
        if logged is None or not logged.get("feature_drift"):
            print("SELF_CHECK skipped: no logged epoch-39 feature drift to compare")
        else:
            expected = float(logged["feature_drift"])
            diff = abs(recomputed["feature_drift"] - expected)
            ok = diff <= max(SELF_CHECK_ATOL, SELF_CHECK_RTOL * abs(expected))
            print(
                f"SELF_CHECK b1 s{check_seed}: logged={expected:.8f} "
                f"recomputed={recomputed['feature_drift']:.8f} diff={diff:.3e} ok={ok}"
            )
            if not ok:
                print("STOP_DRIFT_SELFCHECK: recompute machinery disagrees with the in-run log")
                raise SystemExit(2)

    # ---- Task C: S1 seeds with a missing epoch-39 feature-drift value -----
    patch_rows = []
    for seed in args.seeds:
        run_dir = formal_dir("s1", seed)
        payload, ckpt_path = last_checkpoint(run_dir)
        reference = drift_state_or_derived(payload, init_artifact_for(seed), device)
        candidate_config = dict(payload.get("configuration") or {})
        contract_audit[f"S1_s{seed}"] = verify_drift_contract(
            reference=reference,
            candidate_payload=payload,
            init_artifact_path=init_artifact_for(seed),
            expected_probe_manifest=expected_probe_manifest_cached(candidate_config),
            reference_config=(
                candidate_config
                if reference["source"] == "checkpoint_d7_drift_state" else None
            ),
        )
        result = recompute_for_run(run_dir, reference, device)
        logged = read_drift_csv_row(run_dir, result["epoch"])
        already_logged = bool(logged and logged.get("feature_drift") not in (None, ""))
        patch_rows.append(
            {
                "method": "S1",
                "seed": seed,
                "epoch": result["epoch"],
                "backbone_param_drift": f"{result['backbone_param_drift']:.8f}",
                "early_block_drift": f"{result['early_block_drift']:.8f}",
                "late_block_drift": f"{result['late_block_drift']:.8f}",
                "feature_drift": f"{result['feature_drift']:.8f}",
                "probe_manifest_sha256": probe_manifest_sha256(reference["probe_ids"]),
                "reference_source": reference["source"],
                "reference_checkpoint_sha256": sha256_file(init_artifact_for(seed)),
                "candidate_checkpoint_sha256": sha256_file(ckpt_path),
                "note": "already_logged_in_run" if already_logged else "patched_missing_value",
            }
        )
        print(
            f"TASK_C S1 s{seed} epoch{result['epoch']}: "
            f"param={result['backbone_param_drift']:.8f} feature={result['feature_drift']:.8f} "
            f"({'verified' if already_logged else 'patched'})"
        )

    with (output_dir / "D8_S1_FINAL_DRIFT_PATCH.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(patch_rows)
    print(f"wrote {output_dir / 'D8_S1_FINAL_DRIFT_PATCH.csv'}")

    # Task C invariant (§21): S1 param drift exactly 0, |feature drift| < 1e-7
    for row in patch_rows:
        if row["note"] != "patched_missing_value":
            continue
        if float(row["backbone_param_drift"]) != 0.0:
            raise SystemExit(
                f"STOP: S1 s{row['seed']} backbone_param_drift != 0 — probe/reference mismatch"
            )
        if abs(float(row["feature_drift"])) >= 1e-7:
            raise SystemExit(
                f"STOP: S1 s{row['seed']} epoch39 |feature drift| >= 1e-7 "
                f"({row['feature_drift']}) — check probe/reference mismatch (§21)"
            )

    # §23 merge: original unified matrix, patch values filled in, source column
    source_matrix = Path("artifacts/results/d8_formal/representation_drift.csv")
    patched_path = output_dir / "D8_REPRESENTATION_DRIFT_PATCHED.csv"
    patch_by_key = {
        (row["method"], int(row["seed"]), int(row["epoch"])): row["feature_drift"]
        for row in patch_rows
        if row["note"] == "patched_missing_value"
    }
    if source_matrix.is_file():
        with source_matrix.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames) + ["feature_drift_source"]
            out_rows = []
            for row in reader:
                key = (row["method"], int(row["seed"]), int(row["epoch"]))
                if key in patch_by_key and not row.get("feature_drift"):
                    row["feature_drift"] = patch_by_key[key]
                    row["feature_drift_source"] = "recompute_patch"
                else:
                    row["feature_drift_source"] = "training_log"
                out_rows.append(row)
        with patched_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"wrote {patched_path} ({len(out_rows)} rows)")
    else:
        print(f"WARNING: {source_matrix} not found — merge skipped (patch CSV stands alone)")

    # ---- Task D: B1 label-scaling drift -----------------------------------
    # ALL fractions (100% included) are recomputed with the same machinery
    # and reference so every row in the Figure 4 drift table is on one
    # scale.  (The D7-era in-run logs of s42/44/46 are not reproducible by
    # the current machinery — see the self-check comment — so using them
    # for 100% would mix scales within a seed's fraction trend.)
    drift_rows = []
    for seed in args.seeds:
        formal_payload, _ = last_checkpoint(formal_dir("b1", seed))
        reference = drift_state_or_derived(formal_payload, init_artifact_for(seed), device)
        formal_config = dict(formal_payload.get("configuration") or {})
        formal_reference_config = (
            formal_config if reference["source"] == "checkpoint_d7_drift_state" else None
        )
        for fraction in args.fractions:
            if fraction >= 100:
                run_dir = formal_dir("b1", seed)
            else:
                run_dir = Path(args.scaling_root) / "b1" / f"d8_b1_f{fraction}_e40" / f"seed_{seed}"
            payload, _ = last_checkpoint(run_dir)
            contract_audit[f"B1_f{fraction}_s{seed}"] = verify_drift_contract(
                reference=reference,
                candidate_payload=payload,
                init_artifact_path=init_artifact_for(seed),
                expected_probe_manifest=expected_probe_manifest_cached(
                    dict(payload.get("configuration") or {})
                ),
                reference_config=formal_reference_config,
            )
            result = recompute_for_run(run_dir, reference, device)
            drift_rows.append(
                {
                    "fraction": fraction / 100.0,
                    "seed": seed,
                    "backbone_param_drift": f"{result['backbone_param_drift']:.8f}",
                    "early_block_drift": f"{result['early_block_drift']:.8f}",
                    "late_block_drift": f"{result['late_block_drift']:.8f}",
                    "feature_drift": f"{result['feature_drift']:.8f}",
                    "source": "recompute_vs_pretrained_init",
                }
            )
            print(
                f"TASK_D b1 f{fraction} s{seed}: param={result['backbone_param_drift']:.6f} "
                f"feature={result['feature_drift']:.6f}"
            )

    with (output_dir / "D8_LABEL_SCALING_B1_DRIFT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(drift_rows[0].keys()))
        writer.writeheader()
        writer.writerows(drift_rows)
    print(f"wrote {output_dir / 'D8_LABEL_SCALING_B1_DRIFT.csv'} ({len(drift_rows)} rows)")

    contract_path = output_dir / "D8_DRIFT_CONTRACT_AUDIT.json"
    contract_path.write_text(
        json.dumps(contract_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {contract_path} ({len(contract_audit)} contracts verified)")


if __name__ == "__main__":
    main()
