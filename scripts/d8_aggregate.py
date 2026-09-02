"""D8 aggregation: 5-seed frozen-transfer lock, adaptation-locus screen, D8-B confirmation.

Modes (plan §10/§103-§108):
- d8-0 : B0/B1/S1 over seeds 42-46 (40 epochs).  Gates §16-§19: S1 vs B1 and
  S1 vs B0 paired StableRMSE gains (mean > 0.05, >= 4/5 seeds positive,
  median > 0), the endpoint gate (>= 2/3 endpoints positive, women > 0,
  human >= 3/5 positive) and the §19 SELECTION_SENSITIVE check.  Also
  collects functional source forgetting (§21-§28) and representation drift.
- d8-a : adaptation-locus screen — A1/A2/O6 (seeds 42/45, 20 epochs) vs S1
  (20 epochs).  Promotion §62-§66: mean gain >= 0.01 AND 2/2 seeds positive,
  >= 2/3 endpoints non-worse, women <= S1 + 0.02, human endpoint worsen
  > 0.05 blocks, feature drift <= 0.02, relative functional forgetting
  <= 0.05.  If nothing passes -> STOP_METHOD_ENGINEERING (§70/§106).
- d8-b : Top-1 5-seed confirmation vs S1 (§73-§79): mean gain >= 0.01 with
  >= 4/5 positive, >= 2/3 endpoints non-worse, women non-worse, human mean
  not worse by > 0.02, mechanism gates (feature drift / relative
  forgetting).  STRONG (>= 0.02) / MODEST (0.01-0.02).

Run layout:
    d8-a candidates : <d8_root>/d8_stage_a/<candidate>/d8_<candidate>_e20/seed_<seed>
    d8-b candidate  : <d8_root>/d8_stage_b/<candidate>/d8_<candidate>_e40/seed_<seed>
    S1 references   : <d7_root>/d7_stage_a|b/s1/d7_s1_e20|e40/seed_<seed>
    B0/B1 references: <d6_root>/d6_stage_b/<ref>/d6_<ref>_e40/seed_<seed>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scripts.d6_aggregate import (
    HUMAN_TASKS,
    _finite_mean,
    candidate_run_dir,
    summarize_run as summarize_d6_run,
    trajectory,
    _verify_stage_identity,
)
from scripts.d7_aggregate import summarize_run as summarize_d7_run

STABLE_WINDOW = 5
D8_CANDIDATES = ("a1", "a2", "o6")
# §97 protocol locks (mirror main.py).
D8_CANDIDATE_SCOPE = {
    "a1": {"trainable_last_blocks": 1, "multiplier": 0.01},
    "a2": {"trainable_last_blocks": 2, "multiplier": 0.01},
    "o6": {"trainable_last_blocks": 1, "multiplier": 0.02},
}
# D8-0 formal gate (§16-§18).
GATE_MEAN_VS_B1 = 0.05
GATE_MEAN_VS_B0 = 0.05
GATE_POSITIVE_FRACTION = 0.8  # >= 4/5
# D8-A promotion (§62-§66).
D8A_MIN_MEAN_GAIN = 0.01
D8A_WOMEN_MARGIN = 0.02
D8A_HUMAN_ENDPOINT_BLOCK = 0.05
D8A_FEATURE_DRIFT_MAX = 0.02
D8A_REL_FORGETTING_MAX = 0.05
# D8-B (§75-§78).
D8B_MIN_MEAN_GAIN = 0.01
D8B_STRONG_MEAN_GAIN = 0.02
D8B_HUMAN_MEAN_BLOCK = 0.02
D8B_FEATURE_DRIFT_MAX = 0.02
D8B_REL_FORGETTING_MAX = 0.05


def _read_json(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def check_d8_run_contract(run_dir: Path, candidate: str, seed: int, expected_last_epoch: int) -> list[str]:
    """D8 candidate contract: identity + the exact protocol locks (§39-§47)."""

    violations: list[str] = []
    args_payload = _read_json(run_dir / "args.json")
    metadata = _read_json(run_dir / "run_metadata.json")

    def _check(label: str, actual, expected) -> None:
        if actual != expected:
            violations.append(f"{label}: found {actual!r}, expected {expected!r}")

    _check("args.seed", args_payload.get("seed"), seed)
    _check("args.dataset", args_payload.get("dataset"), "toxacute")
    _check("args.toxacute_task_scope", args_payload.get("toxacute_task_scope"), "human3")
    _check("args.train_eval_scope", args_payload.get("train_eval_scope"), "validation_only")
    _check("args.fit_conformal", args_payload.get("fit_conformal"), False)
    _check("args.epochs", args_payload.get("epochs"), expected_last_epoch + 1)
    _check("run_metadata.d7_candidate", metadata.get("d7_candidate"), candidate)
    scope = D8_CANDIDATE_SCOPE[candidate]
    _check("args.freeze_backbone_epochs", args_payload.get("freeze_backbone_epochs"), 0)
    _check(
        "args.trainable_last_blocks",
        args_payload.get("trainable_last_blocks"),
        scope["trainable_last_blocks"],
    )
    multiplier = args_payload.get("backbone_lr_multiplier")
    if not (isinstance(multiplier, (int, float)) and math.isclose(float(multiplier), scope["multiplier"], rel_tol=0.0, abs_tol=1e-12)):
        violations.append(
            f"args.backbone_lr_multiplier: found {multiplier!r}, "
            f"expected exactly {scope['multiplier']}"
        )
    # Seventh-D8 review P0-2 (§26/§67): the final-epoch feature drift is part
    # of the run contract — a 40-epoch candidate without epoch 39 in its
    # feature_drift_epochs is a CONTRACT_VIOLATION, not a missing metric.
    feature_epochs = {
        int(token)
        for token in str(args_payload.get("feature_drift_epochs", "")).split(",")
        if str(token).strip()
    }
    if expected_last_epoch not in feature_epochs:
        violations.append(
            f"args.feature_drift_epochs lacks final epoch {expected_last_epoch}"
        )
    if candidate == "o6":
        # Seventh-D8 review P0-3 (§32) + P0-4 (§45): O6 protocol locks and
        # retention teacher identity must be recorded in run metadata.
        _check(
            "args.retention_probe_per_task",
            args_payload.get("retention_probe_per_task"),
            16,
        )
        threshold = args_payload.get("retention_damage_threshold")
        if not (
            isinstance(threshold, (int, float))
            and math.isclose(float(threshold), 0.02, rel_tol=0.0, abs_tol=1e-12)
        ):
            violations.append(
                f"args.retention_damage_threshold: found {threshold!r}, must be exactly 0.02"
            )
        for field in (
            "source_train_probe_manifest_sha256",
            "o6_retention_teacher_checkpoint_sha256",
            "o6_retention_teacher_initial_model_sha256",
        ):
            if not metadata.get(field):
                violations.append(f"run_metadata.{field} missing")
    contract = metadata.get("d7_artifact_contract")
    if not isinstance(contract, dict):
        violations.append("run_metadata.d7_artifact_contract missing")
    else:
        _check("d7_artifact_contract.mode", contract.get("mode"), "b1")
        _check("d7_artifact_contract.human_seed", contract.get("human_seed"), seed)
        _check(
            "d7_artifact_contract.split_manifest_hash",
            contract.get("split_manifest_hash"),
            metadata.get("manifest_sha256"),
        )
        _check(
            "d7_artifact_contract.datastore_fingerprint",
            contract.get("datastore_fingerprint"),
            metadata.get("datastore_fingerprint"),
        )
        _check(
            "d7_artifact_contract.feature_schema_version",
            contract.get("feature_schema_version"),
            metadata.get("feature_schema_version"),
        )
    return violations


def summarize_d8_run(run_dir: Path | None, candidate: str, seed: int, expected_last_epoch: int) -> dict:
    row: dict = {"candidate": candidate, "seed": seed, "expected_last_epoch": expected_last_epoch}
    if run_dir is None or not run_dir.exists():
        row["status"] = "MISSING_RUN"
        return row
    violations = check_d8_run_contract(run_dir, candidate, seed, expected_last_epoch)
    if violations:
        row["status"] = "CONTRACT_VIOLATION"
        row["contract_violations"] = "; ".join(violations)
        return row
    args_payload = _read_json(run_dir / "args.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    row["identity"] = {
        "manifest_sha256": metadata.get("manifest_sha256"),
        "datastore_fingerprint": metadata.get("datastore_fingerprint"),
        "split_seed": metadata.get("split_seed"),
    }
    trajectory_rows = trajectory(run_dir, expected_last_epoch)
    if not trajectory_rows:
        row["status"] = "INCOMPLETE_TRAJECTORY"
        return row
    observed = [entry["epoch"] for entry in trajectory_rows]
    duplicates = sorted({epoch for epoch in observed if observed.count(epoch) > 1})
    if duplicates:
        row["status"] = "DUPLICATE_EPOCHS"
        row["duplicate_epochs"] = duplicates
        return row
    missing = sorted(set(range(expected_last_epoch + 1)) - set(observed))
    if missing:
        row["status"] = "INCOMPLETE_TRAJECTORY"
        row["missing_epochs"] = f"{missing[0]}..{missing[-1]}"
        return row
    values = [entry["val"] for entry in trajectory_rows]
    stable_epochs = {entry["epoch"] for entry in trajectory_rows[-5:]}
    from scripts.d6_aggregate import endpoint_stable

    endpoints = endpoint_stable(run_dir, stable_epochs)
    row.update(
        {
            "status": "PASS",
            "stable_rmse": _finite_mean(values[-5:]),
            "best_rmse": min(values),
            "best_epoch": trajectory_rows[values.index(min(values))]["epoch"],
            "man_rmse": endpoints[HUMAN_TASKS[0]],
            "women_rmse": endpoints[HUMAN_TASKS[1]],
            "human_rmse": endpoints[HUMAN_TASKS[2]],
        }
    )
    if any(
        not math.isfinite(float(row[key]))
        for key in ("stable_rmse", "man_rmse", "women_rmse", "human_rmse")
    ):
        row["status"] = "FAILED_NON_FINITE"
    return row


def _summarize_reference(kind: str, root: Path, reference: str, seed: int, expected_last_epoch: int) -> dict:
    if kind == "d6":
        run_dir = candidate_run_dir(root / "d6_stage_b", reference, f"d6_{reference}_e{expected_last_epoch + 1:02d}", seed)
        return summarize_d6_run(run_dir, reference, seed, expected_last_epoch)
    run_dir = candidate_run_dir(root / f"d7_stage_{'a' if expected_last_epoch == 19 else 'b'}", reference, f"d7_{reference}_e{expected_last_epoch + 1:02d}", seed)
    return summarize_d7_run(run_dir, reference, seed, expected_last_epoch)


def _endpoint_columns(row: dict) -> dict:
    return {task: row.get(f"{task.split('_')[0]}_rmse") for task in HUMAN_TASKS}


def _paired_stats(gains: list[float]) -> dict:
    ordered = sorted(gains)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    variance = sum((value - _finite_mean(gains)) ** 2 for value in gains) / max(n - 1, 1)
    return {
        "mean": _finite_mean(gains),
        "std": math.sqrt(variance) if n > 1 else 0.0,
        "median": median,
        "positive": sum(1 for value in gains if value > 0),
        "w/t/l": (
            sum(1 for value in gains if value > 1e-9),
            sum(1 for value in gains if abs(value) <= 1e-9),
            sum(1 for value in gains if value < -1e-9),
        ),
    }


def _final_feature_drift(run_dir: Path):
    drift_csv = run_dir / "diagnostics" / "d7_representation_drift.csv"
    if not drift_csv.is_file():
        return None
    rows = [row for row in csv.DictReader(drift_csv.open(encoding="utf-8")) if row.get("state") == "post_epoch"]
    if not rows:
        return None
    value = rows[-1].get("feature_drift")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _final_param_drift(run_dir: Path):
    """P1-4: final post-epoch backbone parameter drift (diagnostic output for
    ChatGPT review — deliberately NOT a gate threshold)."""

    drift_csv = run_dir / "diagnostics" / "d7_representation_drift.csv"
    if not drift_csv.is_file():
        return None
    rows = [row for row in csv.DictReader(drift_csv.open(encoding="utf-8")) if row.get("state") == "post_epoch"]
    if not rows:
        return None
    value = rows[-1].get("backbone_param_drift")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _functional_forgetting(run_dir: Path):
    data = _read_json(run_dir / "functional_forgetting.json")
    if not data:
        return None
    # Fifth-D8 review P0-6 (§35): only evaluator output with VERIFIED matched
    # teacher provenance may participate in mechanism gates.
    if data.get("provenance_verified") is not True:
        return None
    # Seventh-D8 review P1-1 (§47-§50): bind the JSON to THIS run — a stale or
    # copied file from another run/seed must never enter a mechanism gate.
    metadata = _read_json(run_dir / "run_metadata.json")
    from scripts.d8_functional_forgetting import _resolve_b1_artifact_contract

    try:
        contract, _ = _resolve_b1_artifact_contract(metadata)
    except ValueError:
        return None
    identity_checks = {
        "candidate_run_seed": metadata.get("seed"),
        "split_manifest_hash": metadata.get("manifest_sha256"),
        "datastore_fingerprint": metadata.get("datastore_fingerprint"),
        "feature_schema_version": metadata.get("feature_schema_version"),
        "artifact_sha256": contract.get("artifact_sha256"),
    }
    for field, expected in identity_checks.items():
        if expected is None:
            continue
        if data.get(field) != expected:
            print(
                f"INVALID_PROVENANCE: {run_dir} functional_forgetting.json "
                f"{field}={data.get(field)!r} != run {expected!r} — excluded"
            )
            return None
    return {
        "relative": data.get("functional_forgetting_relative"),
        "abs": data.get("functional_forgetting_abs"),
        "exact_zero": data.get("functional_forgetting_exact_zero"),
        "per_task_delta": data.get("per_task_delta", {}),
        "tasks_worse": data.get("animal_tasks_worsened"),
        "tasks_better": data.get("animal_tasks_improved"),
        "tasks_unchanged": data.get("animal_tasks_unchanged"),
        "delta_q25": data.get("delta_q25"),
        "delta_median": data.get("delta_median"),
        "delta_q75": data.get("delta_q75"),
        "delta_q90": data.get("delta_q90"),
        "delta_max": data.get("delta_max"),
    }


def _write_csv(path: Path, fields, rows_list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)
    print(f"wrote {path}")


def d8_0(d6_root: Path, d7_root: Path, seeds: list[int], output_dir: Path, trace: dict) -> dict:
    expected_last_epoch = 39
    references = {"b0": {}, "b1": {}, "s1": {}}
    for seed in seeds:
        references["b0"][seed] = _summarize_reference("d6", d6_root, "b0", seed, expected_last_epoch)
        references["b1"][seed] = _summarize_reference("d6", d6_root, "b1", seed, expected_last_epoch)
        references["s1"][seed] = _summarize_reference("d7", d7_root, "s1", seed, expected_last_epoch)
    identity_rows = [
        row
        for by_seed in references.values()
        for row in by_seed.values()
        if row.get("identity")
    ]
    _verify_stage_identity(identity_rows, "D8-0")
    for by_seed in references.values():
        for seed, row in by_seed.items():
            if row.get("status") != "PASS":
                raise SystemExit(
                    f"D8-0 reference {row['candidate']} seed {seed} unusable ({row.get('status')})"
                )

    gains_b1 = {seed: references["b1"][seed]["stable_rmse"] - references["s1"][seed]["stable_rmse"] for seed in seeds}
    gains_b0 = {seed: references["b0"][seed]["stable_rmse"] - references["s1"][seed]["stable_rmse"] for seed in seeds}
    stats_b1 = _paired_stats(list(gains_b1.values()))
    stats_b0 = _paired_stats(list(gains_b0.values()))

    endpoint_rows = []
    endpoint_gate = {}
    for task in HUMAN_TASKS:
        per_seed = []
        for seed in seeds:
            s1_ep = references["s1"][seed].get(f"{task.split('_')[0]}_rmse")
            b1_ep = references["b1"][seed].get(f"{task.split('_')[0]}_rmse")
            if s1_ep is None or b1_ep is None:
                per_seed.append(float("nan"))
            else:
                per_seed.append(float(b1_ep) - float(s1_ep))
        mean_gain = _finite_mean(per_seed)
        positive = sum(1 for value in per_seed if value > 0)
        endpoint_gate[task] = {"mean_gain_vs_b1": mean_gain, "positive_seeds": positive}
        for seed in seeds:
            endpoint_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "b1_rmse": references["b1"][seed].get(f"{task.split('_')[0]}_rmse"),
                    "s1_rmse": references["s1"][seed].get(f"{task.split('_')[0]}_rmse"),
                    "gain_vs_b1": per_seed[seeds.index(seed)],
                }
            )

    # §18 endpoint gate: >= 2/3 endpoints mean-positive, women > 0,
    # human_oral_TDLo >= 3/5 positive.
    positive_endpoints = sum(1 for task in HUMAN_TASKS if endpoint_gate[task]["mean_gain_vs_b1"] > 0)
    women_ok = endpoint_gate[HUMAN_TASKS[1]]["mean_gain_vs_b1"] > 0
    human_ok = endpoint_gate[HUMAN_TASKS[2]]["positive_seeds"] >= 3

    # §19: Stable vs Best direction agreement.
    s1_best_mean = _finite_mean([references["s1"][seed]["best_rmse"] for seed in seeds])
    b1_best_mean = _finite_mean([references["b1"][seed]["best_rmse"] for seed in seeds])
    selection_sensitive = None
    if stats_b1["mean"] > 0 and s1_best_mean > b1_best_mean:
        selection_sensitive = True

    performance_gate_pass = bool(
        stats_b1["mean"] > GATE_MEAN_VS_B1
        and stats_b1["positive"] >= math.ceil(GATE_POSITIVE_FRACTION * len(seeds))
        and stats_b1["median"] > 0
        and stats_b0["mean"] > GATE_MEAN_VS_B0
        and stats_b0["positive"] >= math.ceil(GATE_POSITIVE_FRACTION * len(seeds))
        and positive_endpoints >= 2
        and women_ok
        and human_ok
    )

    # Fifth-D8 review P1-3 (§49-§54): the functional forgetting audit is a
    # REQUIRED D8-0 deliverable — B1 and S1 need complete provenance-verified
    # evidence on every seed, and every S1 seed must show the §25 exact-zero
    # invariant.  Kept OUT of the performance gate so diagnostics never change
    # method selection; D8-A may only start when BOTH pass.
    audit_by_ref: dict[str, dict[int, dict]] = {"b1": {}, "s1": {}}
    functional_audit_complete = True
    functional_invariant_failure = False
    for reference in ("b1", "s1"):
        stage_dir = "d6_stage_b" if reference == "b1" else "d7_stage_b"
        tag = f"d6_{reference}_e40" if reference == "b1" else f"d7_{reference}_e40"
        root = d6_root if reference == "b1" else d7_root
        for seed in seeds:
            run_dir = candidate_run_dir(root / stage_dir, reference, tag, seed)
            forgetting = _functional_forgetting(run_dir)
            if forgetting is None:
                functional_audit_complete = False
                continue
            audit_by_ref[reference][seed] = forgetting
    for seed in seeds:
        entry = audit_by_ref["s1"].get(seed)
        if entry is None or entry.get("exact_zero") is not True:
            functional_invariant_failure = True
            functional_audit_complete = False
    gate_pass = bool(performance_gate_pass and functional_audit_complete)
    if not performance_gate_pass:
        stop_reason = "S1 failed the 5-seed formal gate (§16-§18); STOP before D8-A"
    elif not functional_audit_complete:
        stop_reason = (
            "functional forgetting audit incomplete"
            + (" / FUNCTIONAL_INVARIANT_FAILURE" if functional_invariant_failure else "")
            + " — complete the audit before D8-A"
        )
    else:
        stop_reason = None

    summary_rows = []
    for reference, by_seed in references.items():
        for seed in seeds:
            row = by_seed[seed]
            summary_rows.append(
                {
                    "candidate": reference,
                    "seed": seed,
                    "status": row.get("status"),
                    "stable_rmse": row.get("stable_rmse"),
                    "best_rmse": row.get("best_rmse"),
                    "best_epoch": row.get("best_epoch"),
                }
            )
    paired_rows = []
    for seed in seeds:
        paired_rows.append(
            {
                "seed": seed,
                "b0_stable_rmse": references["b0"][seed]["stable_rmse"],
                "b1_stable_rmse": references["b1"][seed]["stable_rmse"],
                "s1_stable_rmse": references["s1"][seed]["stable_rmse"],
                "gain_s1_vs_b1": gains_b1[seed],
                "gain_s1_vs_b0": gains_b0[seed],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "D8_0_5SEED_SUMMARY.csv",
        ["candidate", "seed", "status", "stable_rmse", "best_rmse", "best_epoch"],
        summary_rows,
    )
    _write_csv(
        output_dir / "D8_0_PAIRED_COMPARISON.csv",
        ["seed", "b0_stable_rmse", "b1_stable_rmse", "s1_stable_rmse", "gain_s1_vs_b1", "gain_s1_vs_b0"],
        paired_rows,
    )
    _write_csv(
        output_dir / "D8_0_ENDPOINT_SUMMARY.csv",
        ["task", "seed", "b1_rmse", "s1_rmse", "gain_vs_b1"],
        endpoint_rows,
    )

    ff_rows = []
    for reference, stage_dir, tag in (
        ("b0", "d6_stage_b", "d6_b0_e40"),
        ("b1", "d6_stage_b", "d6_b1_e40"),
        ("s1", "d7_stage_b", "d7_s1_e40"),
    ):
        root = d6_root if reference in ("b0", "b1") else d7_root
        for seed in seeds:
            run_dir = candidate_run_dir(root / stage_dir, reference, tag, seed)
            forgetting = _functional_forgetting(run_dir)
            if forgetting is None:
                continue
            ff_rows.append(
                {
                    "candidate": reference,
                    "seed": seed,
                    "functional_forgetting_abs": forgetting["abs"],
                    "functional_forgetting_relative": forgetting["relative"],
                    "exact_zero": forgetting["exact_zero"],
                    "tasks_improved": forgetting["tasks_better"],
                    "tasks_unchanged": forgetting["tasks_unchanged"],
                    "tasks_worsened": forgetting["tasks_worse"],
                    "delta_q25": forgetting["delta_q25"],
                    "delta_median": forgetting["delta_median"],
                    "delta_q75": forgetting["delta_q75"],
                    "delta_q90": forgetting["delta_q90"],
                    "delta_max": forgetting["delta_max"],
                }
            )
    # P1-3: completeness + invariant are first-class gate fields.
    ff_complete = {
        "functional_audit_complete": functional_audit_complete,
        "functional_invariant_failure": functional_invariant_failure,
        "b1_files_found": len(audit_by_ref["b1"]),
        "s1_files_found": len(audit_by_ref["s1"]),
        "required_files_per_reference": len(seeds),
    }
    _write_csv(
        output_dir / "D8_0_FUNCTIONAL_FORGETTING.csv",
        ["candidate", "seed", "functional_forgetting_abs", "functional_forgetting_relative",
         "exact_zero", "tasks_improved", "tasks_unchanged", "tasks_worsened",
         "delta_q25", "delta_median", "delta_q75", "delta_q90", "delta_max"],
        ff_rows,
    )
    animal_rows = []
    for reference in ("b1", "s1"):
        for seed in seeds:
            run_dir = (
                candidate_run_dir(d6_root / "d6_stage_b", reference, "d6_b1_e40", seed)
                if reference == "b1"
                else candidate_run_dir(d7_root / "d7_stage_b", reference, "d7_s1_e40", seed)
            )
            forgetting = _functional_forgetting(run_dir)
            for task, delta in (forgetting or {}).get("per_task_delta", {}).items():
                animal_rows.append(
                    {"candidate": reference, "seed": seed, "task": task, "delta": delta}
                )
    _write_csv(
        output_dir / "D8_0_ANIMAL_ENDPOINT_FORGETTING.csv",
        ["candidate", "seed", "task", "delta"],
        animal_rows,
    )

    drift_rows = []
    for candidate, root, stage in (("b1", d6_root, "d6_stage_b"), ("s1", d7_root, "d7_stage_b")):
        for seed in seeds:
            run_dir = root / stage / candidate / (f"d6_{candidate}_e40" if candidate == "b1" else f"d7_{candidate}_e40") / f"seed_{seed}"
            drift_csv = run_dir / "diagnostics" / "d7_representation_drift.csv"
            if not drift_csv.is_file():
                continue
            with drift_csv.open(newline="", encoding="utf-8") as handle:
                for entry in csv.DictReader(handle):
                    drift_rows.append({"candidate": candidate, "seed": seed, **entry})
    _write_csv(
        output_dir / "D8_0_REPRESENTATION_DRIFT.csv",
        ["candidate", "seed", "epoch", "state", "drift_anchor_type",
         "backbone_param_drift", "early_block_drift", "late_block_drift",
         "feature_drift", "human3_rmse"],
        drift_rows,
    )

    trace["d8_0"] = {
        "seeds": seeds,
        "gate_pass": gate_pass,
        "stop_reason": stop_reason,
        "s1_vs_b1": stats_b1,
        "s1_vs_b0": stats_b0,
        "endpoint_gate": endpoint_gate,
        "women_gate": women_ok,
        "human_endpoint_gate": human_ok,
        "selection_sensitive": selection_sensitive,
        "performance_gate_pass": performance_gate_pass,
        "functional_audit_complete": functional_audit_complete,
        "functional_invariant_failure": functional_invariant_failure,
        "s1_best_mean": s1_best_mean,
        "b1_best_mean": b1_best_mean,
        "b0_stable_by_seed": {seed: references["b0"][seed]["stable_rmse"] for seed in seeds},
        "b1_stable_by_seed": {seed: references["b1"][seed]["stable_rmse"] for seed in seeds},
        "s1_stable_by_seed": {seed: references["s1"][seed]["stable_rmse"] for seed in seeds},
    }
    _write_csv(
        output_dir / "D8_0_AUDIT_COMPLETENESS.csv",
        ["functional_audit_complete", "functional_invariant_failure",
         "b1_files_found", "s1_files_found", "required_files_per_reference"],
        [ff_complete],
    )
    (output_dir / "D8_0_GATE.json").write_text(
        json.dumps(trace["d8_0"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return trace["d8_0"]


def d8_a(d8_root: Path, d7_root: Path, seeds: list[int], output_dir: Path, trace: dict) -> dict:
    expected_last_epoch = 19
    s1_refs = {}
    for seed in seeds:
        s1_refs[seed] = _summarize_reference("d7", d7_root, "s1", seed, expected_last_epoch)
        if s1_refs[seed].get("status") != "PASS":
            raise SystemExit(
                f"D8-A S1 reference seed {seed} unusable ({s1_refs[seed].get('status')}); "
                "plan §60 requires a matching 20-epoch S1 run"
            )
    identity_rows = [row for row in s1_refs.values() if row.get("identity")]
    candidate_rows: dict[str, dict[int, dict]] = {}
    summary_rows = []
    for candidate in D8_CANDIDATES:
        candidate_rows[candidate] = {}
        for seed in seeds:
            run_dir = candidate_run_dir(
                d8_root / "d8_stage_a", candidate, f"d8_{candidate}_e20", seed
            )
            row = summarize_d8_run(run_dir, candidate, seed, expected_last_epoch)
            candidate_rows[candidate][seed] = row
            summary_rows.append(row)
            if row.get("identity"):
                identity_rows.append(row)
    _verify_stage_identity(identity_rows, "D8-A")

    gates = {}
    for candidate in D8_CANDIDATES:
        per_seed = candidate_rows[candidate]
        passing_seeds = [seed for seed in seeds if per_seed[seed].get("status") == "PASS"]
        gains = {
            seed: float(s1_refs[seed]["stable_rmse"]) - float(per_seed[seed]["stable_rmse"])
            for seed in passing_seeds
        }
        complete = len(gains) == len(seeds)
        mean_gain = _finite_mean(list(gains.values()))
        positive = sum(1 for value in gains.values() if value > 0)

        # Sixth-D8 review P0-2 (§12-§13): the endpoint gate is defined on
        # CROSS-SEED endpoint MEANS, never on per-seed wins.
        endpoint_gate = {}
        endpoint_non_worse = 0
        for task in HUMAN_TASKS:
            key = f"{task.split('_')[0]}_rmse"
            candidate_mean = _finite_mean([float(per_seed[seed][key]) for seed in passing_seeds])
            s1_mean = _finite_mean([float(s1_refs[seed][key]) for seed in passing_seeds])
            non_worse = candidate_mean <= s1_mean
            endpoint_gate[task] = {
                "candidate_mean": candidate_mean,
                "s1_mean": s1_mean,
                "gain_vs_s1": s1_mean - candidate_mean,
                "non_worse": non_worse,
            }
            if non_worse:
                endpoint_non_worse += 1
        endpoint_gate_pass = endpoint_non_worse >= 2
        women_entry = endpoint_gate[HUMAN_TASKS[1]]
        women_gate = bool(
            women_entry["candidate_mean"] <= women_entry["s1_mean"] + D8A_WOMEN_MARGIN
        )
        human_entry = endpoint_gate[HUMAN_TASKS[2]]
        human_worsen = human_entry["candidate_mean"] - human_entry["s1_mean"]
        human_gate = human_worsen <= D8A_HUMAN_ENDPOINT_BLOCK

        # Sixth-D8 review P0-3 (§17-§18): feature drift must be present and
        # within threshold for BOTH development seeds — never just the first.
        feature_drift_by_seed = {}
        param_drift_by_seed = {}
        for seed in seeds:
            run_dir = d8_root / "d8_stage_a" / candidate / f"d8_{candidate}_e20" / f"seed_{seed}"
            feature_value = _final_feature_drift(run_dir)
            if feature_value is not None:
                feature_drift_by_seed[seed] = feature_value
            param_value = _final_param_drift(run_dir)
            if param_value is not None:
                param_drift_by_seed[seed] = param_value
        complete_feature_drift = len(feature_drift_by_seed) == len(seeds)
        representation_gate = bool(
            complete_feature_drift
            and all(value <= D8A_FEATURE_DRIFT_MAX for value in feature_drift_by_seed.values())
        )

        # Sixth-D8 review P0-5 (§25): functional forgetting evidence must be
        # complete over both seeds as well.
        forgetting_by_seed = {}
        for seed in seeds:
            run_dir = d8_root / "d8_stage_a" / candidate / f"d8_{candidate}_e20" / f"seed_{seed}"
            forgetting = _functional_forgetting(run_dir)
            if forgetting and forgetting.get("relative") is not None:
                forgetting_by_seed[seed] = float(forgetting["relative"])
        complete_forgetting = len(forgetting_by_seed) == len(seeds)
        functional_gate = bool(
            complete_forgetting
            and all(value <= D8A_REL_FORGETTING_MAX for value in forgetting_by_seed.values())
        )

        gate_pass = bool(
            complete
            and mean_gain >= D8A_MIN_MEAN_GAIN
            and positive == len(seeds)
            and endpoint_gate_pass
            and women_gate
            and human_gate
            and representation_gate
            and functional_gate
        )
        gates[candidate] = {
            "complete_seed_pairs": complete,
            "mean_gain_vs_s1": mean_gain,
            "positive_seeds": positive,
            "endpoint_gate": endpoint_gate,
            "endpoint_non_worse": endpoint_non_worse,
            "endpoint_gate_pass": endpoint_gate_pass,
            "women_gate": women_gate,
            "human_worsen": human_worsen,
            "human_gate": human_gate,
            "feature_drift_by_seed": feature_drift_by_seed,
            "max_feature_drift": max(feature_drift_by_seed.values()) if feature_drift_by_seed else None,
            "mean_feature_drift": _finite_mean(list(feature_drift_by_seed.values())) if feature_drift_by_seed else None,
            "complete_feature_drift": complete_feature_drift,
            "representation_gate": representation_gate,
            "final_backbone_param_drift_by_seed": param_drift_by_seed,
            "forgetting_by_seed": forgetting_by_seed,
            "complete_forgetting": complete_forgetting,
            "functional_gate": functional_gate,
            "gate_pass": gate_pass,
        }

    passing = [candidate for candidate in D8_CANDIDATES if gates[candidate]["gate_pass"]]
    if passing:
        # §72: keep exactly one Top-1 — the highest mean Gain_vs_S1.
        top1 = max(passing, key=lambda candidate: gates[candidate]["mean_gain_vs_s1"])
        selection_mode = "PROMOTED"
        stop_reason = None
    else:
        top1 = None
        selection_mode = "STOP_METHOD_ENGINEERING"
        stop_reason = (
            "no D8-A candidate beats S1 by >=0.01 on both development seeds — "
            "S1 is the final transfer strategy; STOP method engineering (§70/§106)"
        )

    trace["d8_a"] = {
        "seeds": seeds,
        "gates": gates,
        "top1": top1,
        "selection_mode": selection_mode,
        "stop_reason": stop_reason,
        "s1_stable_by_seed": {seed: s1_refs[seed]["stable_rmse"] for seed in seeds},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "D8A_MINIMAL_ADAPTATION_SUMMARY.csv",
        ["candidate", "seed", "status", "stable_rmse", "best_rmse", "best_epoch",
         "contract_violations"],
        summary_rows,
    )
    endpoint_rows = []
    for candidate in D8_CANDIDATES:
        for seed in seeds:
            row = candidate_rows[candidate][seed]
            for task in HUMAN_TASKS:
                endpoint_rows.append(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "task": task,
                        "stable_rmse": row.get(f"{task.split('_')[0]}_rmse"),
                        "s1_stable_rmse": s1_refs[seed].get(f"{task.split('_')[0]}_rmse"),
                    }
                )
    _write_csv(
        output_dir / "D8A_ENDPOINT_SUMMARY.csv",
        ["candidate", "seed", "task", "stable_rmse", "s1_stable_rmse"],
        endpoint_rows,
    )
    drift_rows = []
    for candidate in D8_CANDIDATES:
        for seed in seeds:
            run_dir = d8_root / "d8_stage_a" / candidate / f"d8_{candidate}_e20" / f"seed_{seed}"
            drift_csv = run_dir / "diagnostics" / "d7_representation_drift.csv"
            if not drift_csv.is_file():
                continue
            with drift_csv.open(newline="", encoding="utf-8") as handle:
                for entry in csv.DictReader(handle):
                    drift_rows.append({"candidate": candidate, "seed": seed, **entry})
    _write_csv(
        output_dir / "D8A_REPRESENTATION_DRIFT.csv",
        ["candidate", "seed", "epoch", "state", "drift_anchor_type",
         "backbone_param_drift", "early_block_drift", "late_block_drift",
         "feature_drift", "human3_rmse"],
        drift_rows,
    )
    ff_rows = []
    for candidate in D8_CANDIDATES:
        for seed in seeds:
            run_dir = d8_root / "d8_stage_a" / candidate / f"d8_{candidate}_e20" / f"seed_{seed}"
            forgetting = _functional_forgetting(run_dir)
            if forgetting is None:
                continue
            ff_rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "functional_forgetting_abs": forgetting["abs"],
                    "functional_forgetting_relative": forgetting["relative"],
                    "exact_zero": forgetting["exact_zero"],
                    "tasks_worsened": forgetting["tasks_worse"],
                    "delta_max": forgetting["delta_max"],
                }
            )
    _write_csv(
        output_dir / "D8A_FUNCTIONAL_FORGETTING.csv",
        ["candidate", "seed", "functional_forgetting_abs",
         "functional_forgetting_relative", "exact_zero", "tasks_worsened", "delta_max"],
        ff_rows,
    )
    trigger_rows = []
    for candidate in D8_CANDIDATES:
        for seed in seeds:
            run_dir = d8_root / "d8_stage_a" / candidate / f"d8_{candidate}_e20" / f"seed_{seed}"
            trigger_csv = run_dir / "diagnostics" / "d8_retention_trigger.csv"
            if not trigger_csv.is_file():
                continue
            with trigger_csv.open(newline="", encoding="utf-8") as handle:
                for entry in csv.DictReader(handle):
                    trigger_rows.append({"candidate": candidate, "seed": seed, **entry})
    _write_csv(
        output_dir / "D8A_SOURCE_RETENTION_TRIGGER.csv",
        ["candidate", "seed", "epoch", "state", "probe_rmse_teacher",
         "probe_rmse_current", "retention_damage_train", "trigger_threshold",
         "backbone_trainable", "triggered", "trigger_fired_this_epoch"],
        trigger_rows,
    )
    (output_dir / "D8A_SELECTION_TRACE.json").write_text(
        json.dumps(trace["d8_a"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return trace["d8_a"]


def d8_b(d8_root: Path, d7_root: Path, d6_root: Path, candidate: str, seeds: list[int], output_dir: Path, trace: dict) -> dict:
    expected_last_epoch = 39
    s1_refs = {}
    candidate_rows = {}
    identity_rows = []
    for seed in seeds:
        s1_refs[seed] = _summarize_reference("d7", d7_root, "s1", seed, expected_last_epoch)
        run_dir = candidate_run_dir(d8_root / "d8_stage_b", candidate, f"d8_{candidate}_e40", seed)
        candidate_rows[seed] = summarize_d8_run(run_dir, candidate, seed, expected_last_epoch)
        identity_rows.extend(row for row in (s1_refs[seed], candidate_rows[seed]) if row.get("identity"))
        if s1_refs[seed].get("status") != "PASS":
            raise SystemExit(f"D8-B S1 reference seed {seed} unusable")
    _verify_stage_identity(identity_rows, "D8-B")

    gains = {}
    endpoint_non_worse = 0
    endpoint_gate = {}
    women_stats = {}
    human_stats = {}
    for task in HUMAN_TASKS:
        candidate_mean = _finite_mean(
            [float(candidate_rows[seed].get(f"{task.split('_')[0]}_rmse")) for seed in seeds]
        )
        s1_mean = _finite_mean(
            [float(s1_refs[seed].get(f"{task.split('_')[0]}_rmse")) for seed in seeds]
        )
        endpoint_gate[task] = {
            "candidate_mean": candidate_mean,
            "s1_mean": s1_mean,
            "non_worse": candidate_mean <= s1_mean,
        }
        if task == HUMAN_TASKS[1]:
            women_stats = {
                "candidate_women_mean": candidate_mean,
                "s1_women_mean": s1_mean,
                "women_non_worse": candidate_mean <= s1_mean,
            }
        if task == HUMAN_TASKS[2]:
            human_stats = {
                "candidate_human_mean": candidate_mean,
                "s1_human_mean": s1_mean,
                "human_worsen": candidate_mean - s1_mean,
            }
        if candidate_mean <= s1_mean:
            endpoint_non_worse += 1
    for seed in seeds:
        if candidate_rows[seed].get("status") == "PASS":
            gains[seed] = float(s1_refs[seed]["stable_rmse"]) - float(
                candidate_rows[seed]["stable_rmse"]
            )
    complete = len(gains) == len(seeds)
    stats = _paired_stats(list(gains.values()))
    women_non_worse = women_stats.get("women_non_worse", False)
    human_ok = human_stats.get("human_worsen", 0.0) <= D8B_HUMAN_MEAN_BLOCK
    # Sixth-D8 review P0-4 (§19-§21): ALL five seeds need mechanism evidence —
    # one good seed must never carry a 5-seed confirmation.
    feature_drift_by_seed = {}
    param_drift_by_seed = {}
    for seed in seeds:
        run_dir = d8_root / "d8_stage_b" / candidate / f"d8_{candidate}_e40" / f"seed_{seed}"
        feature_value = _final_feature_drift(run_dir)
        if feature_value is not None:
            feature_drift_by_seed[seed] = feature_value
        param_value = _final_param_drift(run_dir)
        if param_value is not None:
            param_drift_by_seed[seed] = param_value
    complete_feature_drift = len(feature_drift_by_seed) == len(seeds)
    forgetting_by_seed = {}
    for seed in seeds:
        run_dir = d8_root / "d8_stage_b" / candidate / f"d8_{candidate}_e40" / f"seed_{seed}"
        forgetting = _functional_forgetting(run_dir)
        if forgetting and forgetting.get("relative") is not None:
            forgetting_by_seed[seed] = float(forgetting["relative"])
    complete_forgetting = len(forgetting_by_seed) == len(seeds)
    mechanism_gate = bool(
        complete_feature_drift
        and all(value <= D8B_FEATURE_DRIFT_MAX for value in feature_drift_by_seed.values())
        and complete_forgetting
        and all(value <= D8B_REL_FORGETTING_MAX for value in forgetting_by_seed.values())
    )
    gate_pass = bool(
        complete
        and stats["mean"] >= D8B_MIN_MEAN_GAIN
        and stats["positive"] >= math.ceil(0.8 * len(seeds))
        and endpoint_non_worse >= 2
        and women_non_worse
        and human_ok
        and mechanism_gate
    )
    strength = "STRONG" if stats["mean"] >= D8B_STRONG_MEAN_GAIN else "MODEST"
    trace["d8_b"] = {
        "candidate": candidate,
        "seeds": seeds,
        "gate_pass": gate_pass,
        "strength": strength,
        "mean_gain_vs_s1": stats["mean"],
        "std": stats["std"],
        "median": stats["median"],
        "positive_seeds": stats["positive"],
        "endpoint_non_worse": endpoint_non_worse,
        "endpoint_gate": endpoint_gate,
        "women_non_worse": women_non_worse,
        "human_worsen": human_stats.get("human_worsen"),
        "feature_drift_by_seed": feature_drift_by_seed,
        "max_feature_drift": max(feature_drift_by_seed.values()) if feature_drift_by_seed else None,
        "complete_feature_drift": complete_feature_drift,
        "final_backbone_param_drift_by_seed": param_drift_by_seed,
        "forgetting_by_seed": forgetting_by_seed,
        "complete_forgetting": complete_forgetting,
        "mechanism_gate": mechanism_gate,
        "s1_stable_by_seed": {seed: s1_refs[seed]["stable_rmse"] for seed in seeds},
    }
    if not gate_pass:
        trace["d8_b"]["stop_reason"] = (
            f"{candidate} failed the D8-B 5-seed gate — final transfer strategy is S1; "
            "stop searching for new methods (§79)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        row = candidate_rows[seed]
        rows.append(
            {
                "seed": seed,
                "candidate_stable_rmse": row.get("stable_rmse"),
                "s1_stable_rmse": s1_refs[seed]["stable_rmse"],
                "gain_vs_s1": (
                    float(s1_refs[seed]["stable_rmse"]) - float(row["stable_rmse"])
                    if row.get("status") == "PASS"
                    else ""
                ),
            }
        )
    _write_csv(
        output_dir / "D8B_TOP1_5SEED_PAIRED.csv",
        ["seed", "candidate_stable_rmse", "s1_stable_rmse", "gain_vs_s1"],
        rows,
    )
    (output_dir / "D8B_SELECTION_TRACE.json").write_text(
        json.dumps(trace["d8_b"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return trace["d8_b"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["d8-0", "d8-a", "d8-b"])
    parser.add_argument("--d8_root", default="artifacts/runs/d8")
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--top1", default=None, help="D8-B candidate (from D8-A)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trace_path = output_dir / "D8_TRACE.json"
    trace = {}
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))

    if args.mode == "d8-0":
        result = d8_0(Path(args.d6_root), Path(args.d7_root), args.seeds, output_dir, trace)
        print(f"D8-0 gate_pass={result['gate_pass']} s1_vs_b1_mean={result['s1_vs_b1']['mean']:.4f}")
        if not result["gate_pass"]:
            print(f"STOP: {result['stop_reason']}")
    elif args.mode == "d8-a":
        result = d8_a(Path(args.d8_root), Path(args.d7_root), args.seeds, output_dir, trace)
        print(f"D8-A top1={result['top1']} mode={result['selection_mode']}")
        if result["stop_reason"]:
            print(f"STOP: {result['stop_reason']}")
    else:
        if not args.top1:
            raise SystemExit("d8-b requires --top1")
        result = d8_b(
            Path(args.d8_root),
            Path(args.d7_root),
            Path(args.d6_root),
            args.top1,
            args.seeds,
            output_dir,
            trace,
        )
        print(
            f"D8-B {args.top1}: pass={result['gate_pass']} strength={result['strength']} "
            f"mean_gain={result['mean_gain_vs_s1']:.4f}"
        )
        print("STOP after D8-B — formal test/CQR only after review (§80/§118)")


if __name__ == "__main__":
    main()
