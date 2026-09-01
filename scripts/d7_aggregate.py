"""D7 representation-preservation micro-screening aggregation (plan §41-§68, §73-§74, §82).

Modes:
- stage-a : seed42 screen of S1/S2/S3/O4/O5 against the D6 references
  (B0/B1/B0A, reused when protocol-identical per plan §38-§39).  Primary
  metric is the last-5 Stable Human3 RMSE.  Applies the §49 elimination, the
  §50/§82 promotion tiers, the §51 women protection, the §52 anchor causal
  gate for O4/O5, and the §54/§55 Top-2 rule (at least one S-family candidate
  unless all three fail).
- stage-b : three-seed (42/44/46, 40 epochs) confirmation of the Top-2 with
  Gain_vs_B1, the §64 strong gate, the §65 endpoint gate and per-candidate
  drift summaries.  Stage C is deliberately NOT implemented (plan §68-§70):
  after Stage B the pipeline STOPs for review.

Run layout:
    <stage_root>/<candidate>/<tag>/seed_<seed>   (tag convention: d7_<cand>_e20 / d7_<cand>_e40)
    <d6_root>/d6_stage_a/<reference>/d6_<ref>_e20/seed_<seed>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow direct execution: `python scripts/d7_aggregate.py stage-a ...`
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.d6_aggregate import (
    HUMAN_TASKS,
    STABLE_WINDOW,
    _finite_mean,
    _mean,
    _std,
    candidate_run_dir,
    endpoint_stable,
    summarize_run as summarize_d6_run,
    trajectory,
    _verify_stage_identity,
)

STAGE_LAST_EPOCH = {"stage-a": 19, "stage-b": 39}
D7_CANDIDATES = ("s1", "s2", "s3", "o4", "o5")
PRESERVATION_FAMILY = ("s1", "s2", "s3")
CSDT_FAMILY = ("o4", "o5")
# Fifth-review P0-1: every D7 candidate consumes an initialization artifact;
# the S-family MUST use the seed-matched B1 (sequential-transfer) init.
ARTIFACT_CANDIDATES = {
    "s1": "b1",
    "s2": "b1",
    "s3": "b1",
    "o4": "csdt",
    "o5": "csdt_bounded",
}
# §82: promotion needs a >=0.01 win over B1; a positive-but-smaller win is
# BORDERLINE; losing to B1 by >=0.02 excludes the candidate entirely.
STRONG_GAIN_VS_B1 = 0.01
EXCLUDE_VS_B1 = 0.02
# §49: absolute elimination thresholds versus B0.
ELIMINATE_MACRO_MARGIN = 0.05
ELIMINATE_ENDPOINT_MARGIN = 0.05
# §51: women protection — an otherwise-better candidate that degrades women by
# more than this cannot be Top-1 (Stage A vs B0; Stage B gate vs B1 below).
WOMEN_PROTECTION = 0.08
# Fifth review §54: Stage-B women gate — the candidate's women mean may sit at
# most this far above the B1 women mean.
WOMEN_GATE_MARGIN = 0.03
# Sixth-review P0: S2/S3 backbone LR is part of the experiment definition —
# exactly 0.1x the head LR; anything else is not S2/S3.
D7_BACKBONE_LR_MULTIPLIER = 0.1


def _read_json(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def check_d7_run_contract(run_dir: Path, candidate: str, seed: int, expected_last_epoch: int) -> list[str]:
    """Plan §93: the selector cannot trust a directory name.  Every D7 run
    must prove identity (seed/scope/no-CQR/epochs/d7_candidate) and the
    freeze/LR schedule must match the candidate definition (plan §8/§11/§15)."""

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
    freeze = args_payload.get("freeze_backbone_epochs")
    multiplier = args_payload.get("backbone_lr_multiplier")

    def _multiplier_matches(value, expected: float) -> bool:
        return isinstance(value, (int, float)) and math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-12
        )

    if candidate == "s1":
        _check("args.freeze_backbone_epochs", freeze, expected_last_epoch + 1)
        if not _multiplier_matches(multiplier, 1.0):
            violations.append(
                f"args.backbone_lr_multiplier: found {multiplier!r}, expected 1.0"
            )
    if candidate == "s2":
        _check("args.freeze_backbone_epochs", freeze, 0)
        if not _multiplier_matches(multiplier, D7_BACKBONE_LR_MULTIPLIER):
            violations.append(
                "args.backbone_lr_multiplier: found "
                f"{multiplier!r}, expected exactly {D7_BACKBONE_LR_MULTIPLIER} "
                "(protocol lock, not an LR grid search)"
            )
    if candidate == "s3":
        _check("args.freeze_backbone_epochs", freeze, 5)
        if not _multiplier_matches(multiplier, D7_BACKBONE_LR_MULTIPLIER):
            violations.append(
                "args.backbone_lr_multiplier: found "
                f"{multiplier!r}, expected exactly {D7_BACKBONE_LR_MULTIPLIER} "
                "(protocol lock, not an LR grid search)"
            )
    if candidate in CSDT_FAMILY:
        _check("args.freeze_backbone_epochs", freeze, 0)
        _check("args.backbone_lr_multiplier", multiplier, 1.0)
    # Fifth-review P0-1 (§12): EVERY D7 candidate must record a validated
    # artifact contract — S-family mode=b1, CSDT family per its variant —
    # bound to this run's data identity.
    if candidate in ARTIFACT_CANDIDATES:
        contract = metadata.get("d7_artifact_contract")
        if not isinstance(contract, dict):
            violations.append("run_metadata.d7_artifact_contract missing")
        else:
            _check("d7_artifact_contract.mode", contract.get("mode"), ARTIFACT_CANDIDATES[candidate])
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


def summarize_run(run_dir: Path | None, candidate: str, seed: int, expected_last_epoch: int) -> dict:
    row: dict = {
        "candidate": candidate,
        "seed": seed,
        "expected_last_epoch": expected_last_epoch,
    }
    if run_dir is None or not run_dir.exists():
        row["status"] = "MISSING_RUN"
        return row
    violations = check_d7_run_contract(run_dir, candidate, seed, expected_last_epoch)
    if violations:
        row["status"] = "CONTRACT_VIOLATION"
        row["contract_violations"] = "; ".join(violations)
        return row
    args_payload = _read_json(run_dir / "args.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    # Fifth-review P1-6: carry the data identity for the stage cross-check.
    row["identity"] = {
        "manifest_sha256": metadata.get("manifest_sha256"),
        "datastore_fingerprint": metadata.get("datastore_fingerprint"),
        "split_seed": args_payload.get("split_seed"),
    }
    trajectory_rows = trajectory(run_dir, expected_last_epoch)
    if not trajectory_rows:
        row["status"] = "INCOMPLETE_TRAJECTORY"
        return row
    observed = [entry["epoch"] for entry in trajectory_rows]
    # Fifth-review P1-7: duplicates (e.g. after a resume) would poison the
    # last-5 StableRMSE window even when the epoch SET looks complete.
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
    stable = _mean(values[-STABLE_WINDOW:])
    best = min(values)
    best_index = values.index(best)
    epochs = {entry["epoch"] for entry in trajectory_rows[-STABLE_WINDOW:]}
    endpoints = endpoint_stable(run_dir, epochs)
    row.update(
        {
            "status": "PASS",
            "stable_rmse": stable,
            "best_rmse": best,
            "best_epoch": trajectory_rows[best_index]["epoch"],
            "best_sharpness": _mean(values[max(0, best_index - 2) : best_index + 3]) - best,
            "trajectory_std": _std(values),
            "man_rmse": endpoints[HUMAN_TASKS[0]],
            "women_rmse": endpoints[HUMAN_TASKS[1]],
            "human_rmse": endpoints[HUMAN_TASKS[2]],
        }
    )
    if any(not math.isfinite(float(row[key])) for key in ("stable_rmse", "man_rmse", "women_rmse", "human_rmse")):
        row["status"] = "FAILED_NON_FINITE"
    return row


def _load_reference(d6_root: Path, reference: str, seed: int, expected_last_epoch: int, stage_dir: str = "d6_stage_a") -> dict:
    """Fifth-review §51: D6 references are summarised by the D6 summarizer
    itself, so they automatically inherit the D6 run contract, data identity,
    duplicate-epoch guard and endpoint completeness — no parallel reimplementation."""

    run_dir = candidate_run_dir(
        d6_root / stage_dir, reference, f"d6_{reference}_e{expected_last_epoch + 1:02d}", seed
    )
    return summarize_d6_run(run_dir, reference, seed, expected_last_epoch)


def stage_a(d7_root: Path, d6_root: Path, seed: int, output_dir: Path, trace: dict) -> dict:
    expected_last_epoch = STAGE_LAST_EPOCH["stage-a"]
    b0 = _load_reference(d6_root, "b0", seed, expected_last_epoch)
    b1 = _load_reference(d6_root, "b1", seed, expected_last_epoch)
    b0a = _load_reference(d6_root, "b0a", seed, expected_last_epoch)
    for reference in (b0, b1, b0a):
        if reference.get("status") != "PASS":
            raise SystemExit(
                f"Stage A reference {reference['candidate']} is not usable "
                f"({reference.get('status')}); plan §38 requires B0/B1/B0A"
            )

    rows = []
    for candidate in D7_CANDIDATES:
        run_dir = candidate_run_dir(d7_root / "d7_stage_a", candidate, f"d7_{candidate}_e20", seed)
        row = summarize_run(run_dir, candidate, seed, expected_last_epoch)
        if row.get("status") == "PASS":
            stable = float(row["stable_rmse"])
            row["stable_gain_vs_b0"] = float(b0["stable_rmse"]) - stable
            row["gain_vs_b1"] = float(b1["stable_rmse"]) - stable
            # §49 elimination (only when the run itself is otherwise valid).
            worse_endpoints = sum(
                1
                for task, key in zip(HUMAN_TASKS, ("man_rmse", "women_rmse", "human_rmse"))
                if float(row[key]) >= float(b0[f"{task.split('_')[0]}_rmse"]) + ELIMINATE_ENDPOINT_MARGIN
            )
            if stable >= float(b0["stable_rmse"]) + ELIMINATE_MACRO_MARGIN:
                row["status"] = "ELIMINATED"
                row["elimination_reason"] = "stable_rmse worse than B0 by >=0.05 (§49)"
            elif worse_endpoints >= 2:
                row["status"] = "ELIMINATED"
                row["elimination_reason"] = ">=2 endpoints worse than B0 by >=0.05 (§49)"
            else:
                # §52: the CSDT family must beat the anchor-only control.
                if candidate in CSDT_FAMILY:
                    anchor_gain = float(b0a["stable_rmse"]) - stable
                    row["anchor_semantic_gain"] = anchor_gain
                    if anchor_gain <= 0:
                        row["status"] = "ANCHOR_GATE_FAILED"
                        row["elimination_reason"] = "AnchorSemanticGain <= 0 (§52)"
                # §82 tiers.
                if row.get("status") == "PASS":
                    gain_vs_b1 = row["gain_vs_b1"]
                    if gain_vs_b1 >= STRONG_GAIN_VS_B1:
                        row["tier"] = "PROMOTED"
                    elif gain_vs_b1 > 0:
                        row["tier"] = "BORDERLINE"
                    elif gain_vs_b1 <= -EXCLUDE_VS_B1:
                        row["status"] = "EXCLUDED_VS_B1"
                        row["elimination_reason"] = "worse than B1 by >=0.02 (§82)"
                    else:
                        row["tier"] = "WORSE_THAN_B1"
                # §51 women protection.
                if row.get("status") == "PASS":
                    women_delta = float(row["women_rmse"]) - float(b0["women_rmse"])
                    row["women_degradation_vs_b0"] = women_delta
                    if women_delta > WOMEN_PROTECTION:
                        row["women_flag"] = f"women worse than B0 by {women_delta:.3f} (>0.08, §51)"
        rows.append(row)

    # Fifth-review P1-6: references and every PASS candidate must share one
    # data identity (manifest / datastore / split_seed) — never rank a mix.
    identity_rows = [row for row in ([b0, b1, b0a] + rows) if row.get("identity")]
    _verify_stage_identity(identity_rows, "D7 Stage A")

    passed = [row for row in rows if row.get("status") == "PASS"]
    promoted = sorted(
        (row for row in passed if row.get("tier") == "PROMOTED"),
        key=lambda row: float(row["stable_rmse"]),
    )
    borderline = sorted(
        (row for row in passed if row.get("tier") == "BORDERLINE"),
        key=lambda row: float(row["stable_rmse"]),
    )
    s_family_alive = [row["candidate"] for row in passed if row["candidate"] in PRESERVATION_FAMILY]

    if promoted:
        top2 = [row["candidate"] for row in promoted[:2]]
        selection_mode = "PROMOTED"
        if len(top2) < 2 and borderline:
            top2.append(borderline[0]["candidate"])
        # §54: at least one S-family candidate unless all three clearly failed.
        if top2 and not any(candidate in PRESERVATION_FAMILY for candidate in top2) and s_family_alive:
            best_s = min(
                (row for row in passed if row["candidate"] in PRESERVATION_FAMILY),
                key=lambda row: float(row["stable_rmse"]),
            )
            if best_s["candidate"] not in top2:
                top2 = [top2[0], best_s["candidate"]]
                selection_mode = "PROMOTED_WITH_S_FAMILY_RESERVE"
        stop_reason = None
    elif borderline:
        # §82: no candidate clears the +0.01 bar — take the two best
        # BORDERLINE rows but mark the selection BORDERLINE.
        top2 = [row["candidate"] for row in borderline[:2]]
        selection_mode = "BORDERLINE"
        stop_reason = None
    else:
        # §56/§82: nothing beats B1 — fall back to B1 3-seed confirmation only.
        top2 = []
        selection_mode = "B1_CONFIRMATION_ONLY"
        stop_reason = (
            "no D7 candidate improves on B1; proceed to B1 3-seed confirmation only (§56/§82)"
        )

    # §51: women protection cannot be overridden by the ranking.
    top1 = None
    for candidate in top2:
        row = next(row for row in passed if row["candidate"] == candidate)
        if not row.get("women_flag"):
            top1 = candidate
            break

    trace["stage_a"] = {
        "seed": seed,
        "b0_stable_rmse": b0["stable_rmse"],
        "b1_stable_rmse": b1["stable_rmse"],
        "b0a_stable_rmse": b0a["stable_rmse"],
        "ranking": [
            {
                "candidate": row["candidate"],
                "status": row.get("status"),
                "tier": row.get("tier"),
                "stable_rmse": row.get("stable_rmse"),
                "gain_vs_b1": row.get("gain_vs_b1"),
                "anchor_semantic_gain": row.get("anchor_semantic_gain"),
                "women_flag": row.get("women_flag"),
            }
            for row in rows
        ],
        "top2": top2,
        "top1": top1,
        "selection_mode": selection_mode,
        "s_family_alive": s_family_alive,
        "stop_reason": stop_reason,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "D7A_MICROSCREEN_SUMMARY.csv",
        ["candidate", "seed", "status", "tier", "stable_rmse", "best_rmse", "best_epoch",
         "best_sharpness", "trajectory_std", "man_rmse", "women_rmse", "human_rmse",
         "stable_gain_vs_b0", "gain_vs_b1", "anchor_semantic_gain",
         "women_degradation_vs_b0", "women_flag", "elimination_reason", "contract_violations"],
        rows,
    )
    endpoint_rows = []
    for row in rows:
        for task, key in zip(HUMAN_TASKS, ("man_rmse", "women_rmse", "human_rmse")):
            endpoint_rows.append(
                {
                    "candidate": row["candidate"],
                    "seed": seed,
                    "task": task,
                    "stable_rmse": row.get(key),
                    "b0_stable_rmse": b0[f"{task.split('_')[0]}_rmse"],
                    "b1_stable_rmse": b1[f"{task.split('_')[0]}_rmse"],
                }
            )
    _write_csv(
        output_dir / "D7A_ENDPOINT_SUMMARY.csv",
        ["candidate", "seed", "task", "stable_rmse", "b0_stable_rmse", "b1_stable_rmse"],
        endpoint_rows,
    )
    stability_rows = [
        {
            "candidate": row["candidate"],
            "seed": seed,
            "best_rmse": row.get("best_rmse"),
            "best_epoch": row.get("best_epoch"),
            "best_sharpness": row.get("best_sharpness"),
            "trajectory_std": row.get("trajectory_std"),
            "stable_rmse": row.get("stable_rmse"),
        }
        for row in rows
        if row.get("status") == "PASS"
    ]
    _write_csv(
        output_dir / "D7A_TRAJECTORY_STABILITY.csv",
        ["candidate", "seed", "best_rmse", "best_epoch", "best_sharpness", "trajectory_std", "stable_rmse"],
        stability_rows,
    )
    drift_sources = [
        (candidate, seed, d7_root / "d7_stage_a" / candidate / f"d7_{candidate}_e20" / f"seed_{seed}")
        for candidate in D7_CANDIDATES
    ]
    # Fifth-review P1-3: include the B1 reference trajectory when the
    # reference was rerun with --d7_drift_reference b1 (absent otherwise).
    drift_sources.append(
        ("b1", seed, d6_root / "d6_stage_a" / "b1" / "d6_b1_e20" / f"seed_{seed}")
    )
    drift_rows = _collect_drift_rows(drift_sources)
    _write_csv(output_dir / "D7A_REPRESENTATION_DRIFT.csv", DRIFT_CSV_FIELDS, drift_rows)
    trace_path = output_dir / "D7A_SELECTION_TRACE.json"
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    return {"top2": top2, "top1": top1, "selection_mode": selection_mode}


DRIFT_CSV_FIELDS = [
    "candidate",
    "seed",
    "epoch",
    "state",
    "drift_anchor_type",
    "backbone_param_drift",
    "early_block_drift",
    "late_block_drift",
    "feature_drift",
    "human3_rmse",
]


def _read_run_drift(run_dir: Path) -> list[dict]:
    drift_csv = run_dir / "diagnostics" / "d7_representation_drift.csv"
    if not drift_csv.is_file():
        return []
    with drift_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _epoch_human_rmse(run_dir: Path, epoch: int):
    summary = run_dir / "diagnostics" / "epoch_summary.csv"
    if not summary.is_file():
        return ""
    with summary.open(newline="", encoding="utf-8") as handle:
        for metric_row in csv.DictReader(handle):
            try:
                if int(metric_row["epoch"]) == int(epoch):
                    return metric_row.get("val_human3_macro_rmse", "")
            except (TypeError, ValueError):
                continue
    return ""


def _collect_drift_rows(sources: list[tuple[str, int, Path]]) -> list[dict]:
    """Fifth-review P1-3: collect drift rows for candidates AND references
    (e.g. a B1 run rerun with --d7_drift_reference b1) so the mechanism
    comparison StableRMSE-vs-drift has a reference trajectory."""

    rows = []
    for candidate, seed, run_dir in sources:
        for entry in _read_run_drift(run_dir):
            human_rmse = ""
            if str(entry.get("state", "post_epoch")) == "post_epoch":
                human_rmse = _epoch_human_rmse(run_dir, entry["epoch"])
            rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "epoch": entry.get("epoch"),
                    "state": entry.get("state", ""),
                    "drift_anchor_type": entry.get("drift_anchor_type", ""),
                    "backbone_param_drift": entry.get("backbone_param_drift"),
                    "early_block_drift": entry.get("early_block_drift"),
                    "late_block_drift": entry.get("late_block_drift"),
                    "feature_drift": entry.get("feature_drift"),
                    "human3_rmse": human_rmse,
                }
            )
    return rows


def stage_b(d7_root: Path, d6_root: Path, candidates: list[str], seeds: list[int], output_dir: Path, trace: dict) -> dict:
    expected_last_epoch = STAGE_LAST_EPOCH["stage-b"]
    # §92 anticipates two candidates, but when §82 demotes the field to a
    # single survivor (or B1-only confirmation) the 3-seed machinery must
    # still run for that one candidate.
    if len(candidates) not in (1, 2):
        raise SystemExit(f"Stage B requires one or two Top-2 candidates, got {candidates}")
    if len(set(candidates)) != len(candidates):
        raise SystemExit(f"Stage-B candidates must be unique: {candidates}")
    for candidate in candidates:
        if candidate not in D7_CANDIDATES:
            raise SystemExit(f"unknown Stage-B candidate: {candidate!r}")
    b1_by_seed: dict[int, float] = {}
    b0_by_seed: dict[int, float] = {}
    b0a_by_seed: dict[int, float] = {}
    endpoint_by_ref: dict[tuple[str, int], dict] = {}
    identity_rows = []
    # Fifth-review P1-1: B0A is only required when the Top-2 actually contains
    # an O4/O5 candidate — a pure S-family Top-2 must not pay for extra
    # 40-epoch B0A runs (plan §61).
    needs_anchor = any(candidate in CSDT_FAMILY for candidate in candidates)
    for seed in seeds:
        b0 = _load_reference(d6_root, "b0", seed, expected_last_epoch, stage_dir="d6_stage_b")
        b1 = _load_reference(d6_root, "b1", seed, expected_last_epoch, stage_dir="d6_stage_b")
        references = [b0, b1]
        if needs_anchor:
            b0a = _load_reference(d6_root, "b0a", seed, expected_last_epoch, stage_dir="d6_stage_b")
            references.append(b0a)
        # Stage B references run at the Stage-B budget; if the D6 root only
        # carries the 20-epoch seed42 artifacts the caller must generate and
        # point at 40-epoch reference runs instead (plan §59-§60).
        for reference in references:
            if reference.get("status") != "PASS":
                raise SystemExit(
                    f"Stage B reference {reference['candidate']} seed {seed} unusable "
                    f"({reference.get('status')}); plan §60 requires 40-epoch B0/B1 "
                    "at seeds 42/44/46"
                )
        b0_by_seed[seed] = float(b0["stable_rmse"])
        b1_by_seed[seed] = float(b1["stable_rmse"])
        if needs_anchor:
            b0a_by_seed[seed] = float(b0a["stable_rmse"])
        for reference in references:
            if reference.get("identity"):
                identity_rows.append(reference)
        for reference in (b0, b1):
            endpoint_by_ref[(reference["candidate"], seed)] = {
                task: float(reference[f"{task.split('_')[0]}_rmse"]) for task in HUMAN_TASKS
            }

    candidate_rows: dict[str, dict[int, dict]] = {}
    summary_rows = []
    for candidate in candidates:
        candidate_rows[candidate] = {}
        for seed in seeds:
            run_dir = candidate_run_dir(
                d7_root / "d7_stage_b", candidate, f"d7_{candidate}_e40", seed
            )
            row = summarize_run(run_dir, candidate, seed, expected_last_epoch)
            candidate_rows[candidate][seed] = row
            summary_rows.append(row)
            if row.get("identity"):
                identity_rows.append(row)
    # Fifth-review P1-6: references and candidates must share one identity.
    _verify_stage_identity(identity_rows, "D7 Stage B")

    gates = {}
    for candidate in candidates:
        per_seed = candidate_rows[candidate]
        gains = {}
        for seed in seeds:
            if per_seed[seed].get("status") != "PASS":
                continue
            gains[seed] = b1_by_seed[seed] - float(per_seed[seed]["stable_rmse"])
        complete = len(gains) == len(seeds)
        mean_gain = _finite_mean(list(gains.values()))
        positive = sum(1 for value in gains.values() if value > 0)
        # §64 strong gate; §65 endpoint gate (3-seed mean, >=2/3 non-worse).
        endpoint_non_worse = 0
        anchor_gains = []
        women_gate_pass = None
        women_stats = {}
        if complete:
            for task, key in zip(HUMAN_TASKS, ("man_rmse", "women_rmse", "human_rmse")):
                candidate_mean = _finite_mean(
                    [float(per_seed[seed][key]) for seed in seeds]
                )
                reference_mean = _finite_mean(
                    [endpoint_by_ref[("b1", seed)][task] for seed in seeds]
                )
                if task == HUMAN_TASKS[1]:
                    # Fifth-review P1-8 (§51/§54): D7 exists to repair B1's
                    # women-endpoint damage — the candidate's women mean may
                    # exceed the B1 women mean by at most WOMEN_GATE_MARGIN.
                    women_degradation = candidate_mean - reference_mean
                    women_gate_pass = bool(women_degradation <= WOMEN_GATE_MARGIN)
                    women_stats = {
                        "candidate_women_mean": candidate_mean,
                        "b1_women_mean": reference_mean,
                        "women_degradation_vs_b1": women_degradation,
                        "women_gate_pass": women_gate_pass,
                    }
                if candidate_mean <= reference_mean:
                    endpoint_non_worse += 1
            if candidate in CSDT_FAMILY and needs_anchor:
                anchor_gains = [
                    b0a_by_seed[seed] - float(per_seed[seed]["stable_rmse"]) for seed in seeds
                ]
        gate_pass = bool(
            complete
            and mean_gain > 0
            and positive >= 2
            and endpoint_non_worse >= 2
            and (women_gate_pass is not False)
            and (candidate not in CSDT_FAMILY or _finite_mean(anchor_gains) > 0)
        )
        gates[candidate] = {
            "complete_seed_pairs": complete,
            "mean_gain_vs_b1": mean_gain,
            "positive_seeds": positive,
            "endpoint_non_worse": endpoint_non_worse,
            "anchor_semantic_gain_mean": _finite_mean(anchor_gains) if anchor_gains else None,
            "women_gate_pass": women_gate_pass,
            **women_stats,
            "gate_pass": gate_pass,
            "strong_margin": bool(mean_gain >= 0.02),
        }

    trace["stage_b"] = {
        "seeds": seeds,
        "candidates": candidates,
        "gates": gates,
        "b1_stable_by_seed": b1_by_seed,
        "b0_stable_by_seed": b0_by_seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    for candidate in candidates:
        for seed in seeds:
            row = candidate_rows[candidate][seed]
            paired_rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "status": row.get("status"),
                    "stable_rmse": row.get("stable_rmse"),
                    "b1_stable_rmse": b1_by_seed.get(seed),
                    "gain_vs_b1": (
                        b1_by_seed[seed] - float(row["stable_rmse"])
                        if row.get("status") == "PASS"
                        else ""
                    ),
                }
            )
    _write_csv(
        output_dir / "D7B_PAIRED_COMPARISON.csv",
        ["candidate", "seed", "status", "stable_rmse", "b1_stable_rmse", "gain_vs_b1"],
        paired_rows,
    )
    _write_csv(
        output_dir / "D7B_TOP2_3SEED_SUMMARY.csv",
        ["candidate", "seed", "status", "stable_rmse", "best_rmse", "best_epoch",
         "man_rmse", "women_rmse", "human_rmse"],
        summary_rows,
    )
    drift_sources = [
        (candidate, seed, d7_root / "d7_stage_b" / candidate / f"d7_{candidate}_e40" / f"seed_{seed}")
        for candidate in candidates
        for seed in seeds
    ]
    # Fifth-review P1-3: include B1 reference drift when rerun with
    # --d7_drift_reference b1.
    drift_sources.extend(
        ("b1", seed, d6_root / "d6_stage_b" / "b1" / "d6_b1_e40" / f"seed_{seed}")
        for seed in seeds
    )
    drift_rows = _collect_drift_rows(drift_sources)
    _write_csv(output_dir / "D7B_REPRESENTATION_DRIFT.csv", DRIFT_CSV_FIELDS, drift_rows)
    endpoint_rows = []
    for candidate in candidates:
        for seed in seeds:
            row = candidate_rows[candidate][seed]
            for task, key in zip(HUMAN_TASKS, ("man_rmse", "women_rmse", "human_rmse")):
                endpoint_rows.append(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "task": task,
                        "stable_rmse": row.get(key),
                        "b1_stable_rmse": endpoint_by_ref.get(("b1", seed), {}).get(task),
                    }
                )
    _write_csv(
        output_dir / "D7B_ENDPOINT_SUMMARY.csv",
        ["candidate", "seed", "task", "stable_rmse", "b1_stable_rmse"],
        endpoint_rows,
    )
    trace_path = output_dir / "D7B_SELECTION_TRACE.json"
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    return {"gates": gates}


def b1_confirmation(d6_root: Path, seeds: list[int], output_dir: Path, trace: dict) -> dict:
    """Fifth-review P1-2 (plan §56/§82/§83): when no D7 candidate survives
    Stage A, the protocol falls back to confirming B1 itself at 40 epochs over
    seeds 42/44/46.  StableGain_B1 = B0 - B1 per seed; gate: mean gain > 0 and
    >= 2/3 seeds positive (strong: mean >= 0.02).  A failed confirmation means
    STOP_ANIMAL_TO_HUMAN_TRANSFER_ENGINEERING (§83) — no S4/S5-style
    continuation."""

    expected_last_epoch = STAGE_LAST_EPOCH["stage-b"]
    rows = []
    gains = {}
    identity_rows = []
    for seed in seeds:
        b0 = _load_reference(d6_root, "b0", seed, expected_last_epoch, stage_dir="d6_stage_b")
        b1 = _load_reference(d6_root, "b1", seed, expected_last_epoch, stage_dir="d6_stage_b")
        for reference in (b0, b1):
            if reference.get("status") != "PASS":
                raise SystemExit(
                    f"B1 confirmation reference {reference['candidate']} seed {seed} "
                    f"unusable ({reference.get('status')})"
                )
            if reference.get("identity"):
                identity_rows.append(reference)
        _verify_stage_identity(identity_rows, "D7 B1 confirmation")
        gain = float(b0["stable_rmse"]) - float(b1["stable_rmse"])
        gains[seed] = gain
        rows.append(
            {
                "seed": seed,
                "b0_stable_rmse": b0["stable_rmse"],
                "b1_stable_rmse": b1["stable_rmse"],
                "stable_gain_b1": gain,
                "b0_best_epoch": b0.get("best_epoch"),
                "b1_best_epoch": b1.get("best_epoch"),
                "b0_women_rmse": b0.get("women_rmse"),
                "b1_women_rmse": b1.get("women_rmse"),
            }
        )

    complete = len(gains) == len(seeds)
    mean_gain = _finite_mean(list(gains.values()))
    positive = sum(1 for value in gains.values() if value > 0)
    gate_pass = bool(complete and mean_gain > 0 and positive >= 2)
    strong = bool(mean_gain >= 0.02)
    trace["b1_confirmation"] = {
        "seeds": seeds,
        "gains_by_seed": gains,
        "mean_stable_gain": mean_gain,
        "positive_seeds": positive,
        "complete": complete,
        "gate_pass": gate_pass,
        "strong_margin": strong,
        "stop_reason": (
            None
            if gate_pass
            else "STOP_ANIMAL_TO_HUMAN_TRANSFER_ENGINEERING: B1 is not a stable "
            "positive transfer under the 42/44/46 protocol (plan §83) — stop "
            "micro-direction engineering, pivot to data/external pretraining"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "D7B_B1_CONFIRMATION.csv",
        ["seed", "b0_stable_rmse", "b1_stable_rmse", "stable_gain_b1",
         "b0_best_epoch", "b1_best_epoch", "b0_women_rmse", "b1_women_rmse"],
        rows,
    )
    (output_dir / "D7B_B1_CONFIRMATION_TRACE.json").write_text(
        json.dumps(trace["b1_confirmation"], indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"B1 confirmation: mean_gain={mean_gain:.4f} positive={positive}/{len(seeds)} "
        f"gate_pass={gate_pass} strong={strong}"
    )
    if not gate_pass:
        print(trace["b1_confirmation"]["stop_reason"])
    return trace["b1_confirmation"]


def _write_csv(path: Path, fields, rows_list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage-a", "stage-b", "b1-confirm"])
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top2", nargs="+", default=None, help="Stage-B candidates (from Stage A)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trace = {}
    trace_path = output_dir / "D7A_SELECTION_TRACE.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))

    if args.mode == "stage-a":
        result = stage_a(Path(args.d7_root), Path(args.d6_root), args.seed, output_dir, trace)
        print(f"Stage A Top-2: {result['top2']} (mode={result['selection_mode']}, top1={result['top1']})")
        if result["selection_mode"] == "B1_CONFIRMATION_ONLY":
            print("STOP: proceed to B1 3-seed confirmation only (plan §56/§82)")
    elif args.mode == "stage-b":
        if not args.top2 or not (1 <= len(args.top2) <= 2):
            raise SystemExit("stage-b requires --top2 with one or two candidates")
        result = stage_b(
            Path(args.d7_root),
            Path(args.d6_root),
            list(args.top2),
            [42, 44, 46],
            output_dir,
            trace,
        )
        print("Stage B gates:")
        for candidate, gate in result["gates"].items():
            print(f"  {candidate}: pass={gate['gate_pass']} mean_gain={gate['mean_gain_vs_b1']:.4f}")
        print("STOP after Stage B — no Stage C without review (plan §68-§70)")
    else:
        b1_confirmation(Path(args.d6_root), [42, 44, 46], output_dir, trace)
        print("B1 confirmation complete — report to review before any further stage (plan §68-§69)")


if __name__ == "__main__":
    main()
