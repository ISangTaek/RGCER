"""D6 Progressive Micro-Screening aggregation and auditable selection (§34, §73-§76; review P0-2/3/4, P1-7/8/9/10/11).

Modes:
- stage-a : rank B1/O1/O2/O3 by StableRMSE_20 (mean of the last five epochs)
  against the B0 Human3-only reference; apply the §31 elimination and §33
  originality minimum gates; select Top-2 per §32.
- stage-b : three-seed (42/44/46) confirmation of the Stage-A Top-2 with the
  B0 paired reference; StableGain gates §43-§44; Top-1 per §45-§46.
- stage-c : five-seed full-budget confirmation of the Stage-B Top-1 with the
  §51-§52 gates (paired StableGain sign per §42/§52 and the endpoint gate).

Selection contract (review P0-2): every run directory is resolved as an exact
``runs_root/candidate/tag/seed_<seed>`` match — Stage B/C therefore read each
seed independently, and incomplete trajectories (missing or duplicate epochs
up to the stage's expected last epoch) are rejected instead of ranked (P1-7).
Every decision is written to D6_SELECTION_TRACE.json (§76).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import json

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
STABLE_WINDOW = 5
CANDIDATES = {
    "b0": {"originality_eligible": False},
    # Review §22/§71: B0A is the anchor-only diagnostic control for O1; it is
    # never ranked and never becomes a Stage-B/C candidate.
    "b0a": {"originality_eligible": False, "control": True},
    "b1": {"originality_eligible": False},
    "o1": {"originality_eligible": True},
    "o2": {"originality_eligible": True},
    "o3": {"originality_eligible": True},
}
# Review P0-6/§34: within this StableRMSE margin an original candidate and B1
# are treated as tied, so originality/simplicity may decide; beyond it B1
# dominates and the novelty candidates must not be promoted.
ORIGINAL_VS_B1_TOLERANCE = 0.01
# Review §25: O1 must beat the anchor-only control by more than zero
# (strong: >= 0.01) to attribute any gain to the semantic delta itself.
ANCHOR_SEMANTIC_GAIN_MIN = 0.0
ANCHOR_SEMANTIC_GAIN_STRONG = 0.01
# §46: minimal-change preference order for near ties.
PREFERENCE_ORDER = ["o1", "o2", "o3"]
ELIMINATE_MACRO_MARGIN = 0.05
ELIMINATE_ENDPOINT_MARGIN = 0.05
ORIGINAL_MIN_GAIN = 0.01
STRONG_GAIN = 0.02
TIE_MARGIN = 0.01
STAGE_LAST_EPOCH = {"stage-a": 19, "stage-b": 39, "stage-c": 99}


def _read_json(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def check_d6_run_contract(
    run_dir: Path,
    candidate: str,
    seed: int,
    expected_last_epoch: int,
) -> tuple[list[str], dict]:
    """Review P1-5/§31-§32: the selector cannot trust a directory name.

    Every D6 run must carry an args.json/run_metadata.json pair proving it is
    the requested candidate at the requested seed with the validation-only,
    no-CQR protocol; it must also record its data identity so the stage can
    verify manifest/datastore consistency across runs.
    """

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
    _check("run_metadata.d6_candidate", metadata.get("d6_candidate"), candidate)

    identity = {
        "manifest_sha256": metadata.get("manifest_sha256"),
        "datastore_fingerprint": metadata.get("datastore_fingerprint"),
        "split_seed": metadata.get("split_seed"),
    }
    return violations, identity


def _read_csv(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}")


def _finite_mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def candidate_run_dir(runs_root: Path, candidate: str, tag: str, seed: int) -> Path | None:
    """Review P0-2: exact per-seed run directory, no glob fallback."""

    path = runs_root / candidate / tag / f"seed_{int(seed)}"
    return path if path.exists() else None


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1))


def trajectory(run_dir: Path, expected_last_epoch: int) -> list[dict]:
    rows = _read_csv(run_dir / "diagnostics" / "epoch_summary.csv")
    parsed = [
        {"epoch": int(row["epoch"]), "val": float(row["val_human3_macro_rmse"])}
        for row in rows
        if row.get("val_human3_macro_rmse") not in (None, "", "nan")
    ]
    return sorted(parsed, key=lambda row: row["epoch"])


def summarize_run(
    run_dir: Path | None,
    candidate: str,
    seed: int,
    expected_last_epoch: int,
) -> dict:
    row: dict = {
        "candidate": candidate,
        "originality_eligible": CANDIDATES.get(candidate, {}).get("originality_eligible", False),
        "seed": seed,
        "expected_last_epoch": expected_last_epoch,
    }
    if run_dir is None or not run_dir.exists():
        row["status"] = "MISSING_RUN"
        return row
    # Review P1-5/§31-32: run identity/config contract is checked before any
    # metric computation — a mislabelled run must never be ranked.
    violations, identity = check_d6_run_contract(
        run_dir, candidate, seed, expected_last_epoch
    )
    if violations:
        row["status"] = "CONTRACT_VIOLATION"
        row["contract_violations"] = "; ".join(violations)
        return row
    row["identity"] = identity
    trajectory_rows = trajectory(run_dir, expected_last_epoch)
    if not trajectory_rows:
        row["status"] = "INCOMPLETE_TRAJECTORY"
        row["missing_epochs"] = f"0..{expected_last_epoch}"
        return row
    observed = [entry["epoch"] for entry in trajectory_rows]
    duplicates = sorted({epoch for epoch in observed if observed.count(epoch) > 1})
    if duplicates:
        row["status"] = "DUPLICATE_EPOCHS"
        row["duplicate_epochs"] = duplicates
        return row
    missing = sorted(set(range(expected_last_epoch + 1)) - set(observed))
    if missing:
        # Review P1-7: a crashed run must never enter the ranking.
        row["status"] = "INCOMPLETE_TRAJECTORY"
        row["missing_epochs"] = (
            f"{missing[0]}..{missing[-1]}" if len(missing) > 1 else str(missing[0])
        )
        row["n_missing_epochs"] = len(missing)
        return row

    values = [entry["val"] for entry in trajectory_rows]
    window = values[-STABLE_WINDOW:]
    stable = _mean(window)
    best = min(values)
    best_index = values.index(best)
    local = values[max(0, best_index - 2) : best_index + 3]
    epochs = {entry["epoch"] for entry in trajectory_rows[-STABLE_WINDOW:]}
    endpoints = endpoint_stable(run_dir, epochs)
    row.update(
        {
            "status": "PASS",
            "stable_rmse": stable,
            "best_rmse": best,
            "best_epoch": trajectory_rows[best_index]["epoch"],
            "best_sharpness": _mean(local) - best,
            "trajectory_std": _std(values),
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


def endpoint_stable(run_dir: Path, epochs: set[int]) -> dict[str, float]:
    rows = [
        row
        for row in _read_csv(run_dir / "diagnostics" / "human3_path_metrics.csv")
        if row["path"] == "final" and int(row["epoch"]) in epochs
    ]
    result = {}
    for task in HUMAN_TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        observed_epochs = {int(row["epoch"]) for row in task_rows}
        # Review P1-9: every endpoint must have all stable-window epochs; a
        # partial endpoint trajectory must never produce a finite mean.
        if observed_epochs != epochs:
            raise SystemExit(
                "INCOMPLETE_ENDPOINT_TRAJECTORY: "
                f"{run_dir} task {task} observed epochs {sorted(observed_epochs)}, "
                f"expected {sorted(epochs)}"
            )
        result[task] = _finite_mean([row["rmse"] for row in task_rows])
    return result


def _endpoint_means(rows: list[dict], candidate: str) -> dict[str, float]:
    passed = [
        row for row in rows if row["candidate"] == candidate and row.get("status") == "PASS"
    ]
    return {
        task: _finite_mean([row.get(f"{task.split('_')[0]}_rmse") for row in passed])
        for task in HUMAN_TASKS
    }


def _verify_stage_identity(rows: list[dict], stage: str) -> None:
    """Review §32: all runs inside one stage must share the same data identity."""

    identities = {
        json.dumps(row.get("identity") or {}, sort_keys=True)
        for row in rows
        if row.get("identity")
    }
    if len(identities) > 1:
        raise SystemExit(
            f"{stage}: runs disagree on data identity (manifest/datastore/split_seed): "
            f"{identities}"
        )


def stage_a(runs_root: Path, seed: int, output_dir: Path, trace: dict) -> list[str]:
    expected_last_epoch = STAGE_LAST_EPOCH["stage-a"]
    rows = []
    b0_row = None
    b0a_row = None
    for candidate in CANDIDATES:
        run_dir = candidate_run_dir(runs_root, candidate, f"d6_{candidate}_e20", seed)
        row = summarize_run(run_dir, candidate, seed, expected_last_epoch)
        rows.append(row)
        if row.get("status") != "PASS":
            continue
        if candidate == "b0":
            b0_row = row
        if candidate == "b0a":
            b0a_row = row

    _verify_stage_identity(rows, "Stage A")
    if b0_row is None:
        raise SystemExit("Stage A requires a PASS B0 Human3-only reference run")
    b0_stable = float(b0_row["stable_rmse"])
    b0_endpoints = {task: float(b0_row[f"{task.split('_')[0]}_rmse"]) for task in HUMAN_TASKS}

    for row in rows:
        if row.get("status") != "PASS":
            continue
        endpoints = {task: float(row[f"{task.split('_')[0]}_rmse"]) for task in HUMAN_TASKS}
        worse_endpoints = sum(
            1
            for task in HUMAN_TASKS
            if endpoints[task] >= b0_endpoints[task] + ELIMINATE_ENDPOINT_MARGIN
        )
        row["stable_gain_vs_human3_only"] = b0_stable - float(row["stable_rmse"])
        reasons = []
        if float(row["stable_rmse"]) >= b0_stable + ELIMINATE_MACRO_MARGIN:
            reasons.append("stable_rmse worse than B0 by >=0.05")
        if worse_endpoints >= 2:
            reasons.append(">=2 endpoints worse by >=0.05")
        if reasons:
            row["status"] = "ELIMINATED"
            row["elimination_reason"] = "; ".join(reasons)
        elif CANDIDATES[row["candidate"]]["originality_eligible"]:
            endpoint_improved = sum(1 for task in HUMAN_TASKS if endpoints[task] < b0_endpoints[task])
            gain = b0_stable - float(row["stable_rmse"])
            meets = bool(gain >= ORIGINAL_MIN_GAIN or endpoint_improved >= 2)
            if row["candidate"] == "o1":
                # Review P0-4/§22/§25: O1 must additionally beat the anchor-only
                # control, otherwise its gain is not attributable to the
                # semantic delta (it could be the anchor initialisation alone).
                if b0a_row is None:
                    meets = False
                    row["anchor_note"] = "B0A anchor control missing/failed"
                else:
                    anchor_gain = float(b0a_row["stable_rmse"]) - float(row["stable_rmse"])
                    row["anchor_semantic_gain"] = anchor_gain
                    if anchor_gain <= ANCHOR_SEMANTIC_GAIN_MIN:
                        meets = False
                        row["anchor_note"] = "AnchorSemanticGain <= 0 (§25)"
                    elif anchor_gain >= ANCHOR_SEMANTIC_GAIN_STRONG:
                        row["anchor_note"] = "AnchorSemanticGain strong (>=0.01)"
            row["meets_original_minimum"] = meets

    ranked = sorted(
        (
            row
            for row in rows
            if row.get("status") == "PASS"
            and row["candidate"] not in ("b0", "b0a")  # references are never ranked
        ),
        key=lambda row: row["stable_rmse"],
    )
    eligible = [
        row
        for row in ranked
        if not CANDIDATES[row["candidate"]]["originality_eligible"]
        or row.get("meets_original_minimum")
    ]
    top2 = [row["candidate"] for row in eligible[:2]]

    trace["stage_a"] = {
        "seed": seed,
        "stable_window": "last 5 epochs of 20 (epochs 15-19)",
        "expected_last_epoch": expected_last_epoch,
        "b0_stable_rmse": b0_stable,
        "ranking": [
            {
                "candidate": row["candidate"],
                "stable_rmse": row.get("stable_rmse"),
                "status": row.get("status"),
                "meets_original_minimum": row.get("meets_original_minimum"),
            }
            for row in ranked
        ],
        "thresholds": {
            "eliminate_macro_margin": ELIMINATE_MACRO_MARGIN,
            "eliminate_endpoint_margin": ELIMINATE_ENDPOINT_MARGIN,
            "original_min_gain": ORIGINAL_MIN_GAIN,
        },
        "top2": top2,
        "top2_reason": "§32: top-2 by StableRMSE among B1/O1/O2/O3 that pass the §33 original minimum",
        "top2_insufficient": len(top2) < 2,
    }
    if len(top2) < 2:
        # Review P1-8: never fabricate placeholder candidates.
        trace["stage_a"]["stop_reason"] = (
            "fewer than two eligible Stage-A candidates (§32/§33); STOP before Stage B"
        )
        print(f"Stage A Top-2 incomplete: {top2} — {trace['stage_a']['stop_reason']}")

    fields = ["candidate", "originality_eligible", "seed", "status", "elimination_reason",
              "missing_epochs", "stable_rmse", "best_rmse", "best_epoch", "best_sharpness",
              "trajectory_std", "man_rmse", "women_rmse", "human_rmse",
              "stable_gain_vs_human3_only", "meets_original_minimum"]
    _write_csv(output_dir / "D6A_MICROSCREEN_SUMMARY.csv", fields, rows)

    endpoint_rows = []
    for row in rows:
        for task in HUMAN_TASKS:
            endpoint_rows.append(
                {
                    "candidate": row["candidate"],
                    "seed": row["seed"],
                    "task": task,
                    "stable_rmse": row.get(f"{task.split('_')[0]}_rmse"),
                    "b0_stable_rmse": b0_endpoints[task],
                    "delta_vs_b0": (
                        b0_endpoints[task] - row[f"{task.split('_')[0]}_rmse"]
                    )
                    if row.get("status") == "PASS"
                    else float("nan"),
                }
            )
    _write_csv(
        output_dir / "D6A_ENDPOINT_RESULTS.csv",
        ["candidate", "seed", "task", "stable_rmse", "b0_stable_rmse", "delta_vs_b0"],
        endpoint_rows,
    )
    return top2


def stage_b(runs_root: Path, candidates: list[str], seeds: list[int], output_dir: Path, trace: dict) -> str | None:
    # Review P1-7: Stage B requires exactly the two Stage-A candidates.
    if len(candidates) != 2:
        raise SystemExit(
            f"Stage B requires exactly two Stage-A candidates, got {candidates}"
        )
    for candidate in candidates:
        if candidate not in CANDIDATES:
            raise SystemExit(f"unknown Stage-B candidate: {candidate!r}")
    if len(set(candidates)) != len(candidates):
        raise SystemExit(f"Stage-B candidates must be unique: {candidates}")

    expected_last_epoch = STAGE_LAST_EPOCH["stage-b"]
    summary_rows = []
    paired_rows = []
    stable_by_candidate: dict[str, dict[int, float]] = {}
    b0_by_seed: dict[int, float] = {}
    b0a_by_seed: dict[int, float] = {}
    # Review P0-4/§26: when O1 is confirmed, its anchor-only control B0A must
    # be summarised over the same seeds to keep the causal evidence alive.
    references = ["b0"] + (["b0a"] if "o1" in candidates else [])
    for candidate in candidates + references:
        for seed in seeds:
            run_dir = candidate_run_dir(runs_root, candidate, f"d6_{candidate}_e40", seed)
            row = summarize_run(run_dir, candidate, seed, expected_last_epoch)
            summary_rows.append(row)
            if row.get("status") == "PASS":
                stable_by_candidate.setdefault(candidate, {})[seed] = float(row["stable_rmse"])
                if candidate == "b0":
                    b0_by_seed[seed] = float(row["stable_rmse"])
                if candidate == "b0a":
                    b0a_by_seed[seed] = float(row["stable_rmse"])

    _verify_stage_identity(summary_rows, "Stage B")
    gate_rows = []
    verdicts = []
    tie_break = None
    for candidate in candidates:
        per_seed = stable_by_candidate.get(candidate, {})
        paired = [
            {
                "candidate": candidate,
                "seed": seed,
                "b0_stable_rmse": b0_by_seed[seed],
                "candidate_stable_rmse": per_seed[seed],
                # §42/§52: StableGain = Human3Only - Candidate (RMSE lower is better).
                "stable_gain": b0_by_seed[seed] - per_seed[seed],
            }
            for seed in seeds
            if seed in per_seed and seed in b0_by_seed
        ]
        paired_rows.extend(paired)
        gains = [entry["stable_gain"] for entry in paired]
        mean_gain = _finite_mean(gains)
        positive = sum(1 for gain in gains if gain > 0)
        candidate_endpoint_means = _endpoint_means(summary_rows, candidate)
        b0_endpoint_means = _endpoint_means(summary_rows, "b0")
        endpoints_non_worse = sum(
            1
            for task in HUMAN_TASKS
            if candidate_endpoint_means[task] <= b0_endpoint_means[task] + 1e-9
        )
        gate_pass = bool(
            mean_gain > 0
            and positive >= 2
            and endpoints_non_worse >= 2
            and len(gains) == len(seeds)
        )
        anchor_info = {}
        if candidate == "o1" and gate_pass:
            # Review P0-4/§26: O1's Stage-B gate additionally requires the
            # anchor-only control (mean AnchorSemanticGain > 0).
            anchor_gains = [
                b0a_by_seed[seed] - per_seed[seed]
                for seed in seeds
                if seed in per_seed and seed in b0a_by_seed
            ]
            mean_anchor = _finite_mean(anchor_gains)
            gate_pass = bool(len(anchor_gains) == len(seeds) and mean_anchor > ANCHOR_SEMANTIC_GAIN_MIN)
            anchor_info = {
                "mean_anchor_semantic_gain": mean_anchor,
                "anchor_gate_pass": gate_pass,
                "anchor_strong": bool(mean_anchor >= ANCHOR_SEMANTIC_GAIN_STRONG),
            }
        verdict = (
            "STRONG"
            if gate_pass and mean_gain >= STRONG_GAIN
            else ("BORDERLINE" if gate_pass else "FAIL")
        )
        verdicts.append((candidate, mean_gain, positive, endpoints_non_worse, gate_pass, verdict))
        gate_rows.append(
            {
                "candidate": candidate,
                "mean_stable_gain": mean_gain,
                "positive_seeds": positive,
                "seeds_compared": len(gains),
                "endpoints_non_worse": endpoints_non_worse,
                **anchor_info,
                "gate": verdict,
            }
        )

    # Review P0-6/§32-§35: B1 dominance guard — a clearly stronger sequential
    # transfer baseline stops the novelty line.
    b1_mean_stable = _finite_mean(list(stable_by_candidate.get("b1", {}).values()))
    top1 = None
    original_pass = [entry for entry in verdicts if CANDIDATES[entry[0]]["originality_eligible"] and entry[4]]
    if original_pass:
        pool = [entry for entry in original_pass if entry[1] is not None and entry[1] > 0]
        strong = [entry for entry in pool if entry[5] == "STRONG"]
        if strong:
            pool = strong
        means = {
            candidate: _finite_mean(list(stable_by_candidate.get(candidate, {}).values()))
            for candidate, *_ in pool
        }
        pool_sorted = sorted(pool, key=lambda entry: means[entry[0]])
        top1 = pool_sorted[0][0]
        if len(pool_sorted) >= 2:
            best_mean = means[pool_sorted[0][0]]
            second_mean = means[pool_sorted[1][0]]
            if abs(best_mean - second_mean) < TIE_MARGIN:
                shared_preferences = [
                    name
                    for name in PREFERENCE_ORDER
                    if name in {entry[0] for entry in pool_sorted[:2]}
                ]
                if shared_preferences:
                    top1 = shared_preferences[0]
                    tie_break = {
                        "means": means,
                        "applied_preference": PREFERENCE_ORDER,
                        "top1": top1,
                    }

        b1_dominates = False
        if math.isfinite(b1_mean_stable):
            best_original_mean = means[top1]
            if best_original_mean > b1_mean_stable + ORIGINAL_VS_B1_TOLERANCE:
                b1_dominates = True
        if b1_dominates:
            top1 = None

    trace["stage_b"] = {
        "seeds": seeds,
        "expected_last_epoch": expected_last_epoch,
        "ranking": [
            {
                "candidate": candidate,
                "mean_stable_gain": mean_gain,
                "positive_seeds": positive,
                "endpoints_non_worse": endpoints_non_worse,
                "gate": verdict,
            }
            for candidate, mean_gain, positive, endpoints_non_worse, _gate_pass, verdict in verdicts
        ],
        "thresholds": {"strong_gain": STRONG_GAIN, "tie_margin": TIE_MARGIN,
                        "preference_order": PREFERENCE_ORDER,
                        "original_vs_b1_tolerance": ORIGINAL_VS_B1_TOLERANCE},
        "tie_break": tie_break,
        "b1_mean_stable_rmse": b1_mean_stable,
        "b1_dominates": top1 is None and bool(
            original_pass and math.isfinite(b1_mean_stable)
        ),
        "top1": top1,
        "top1_reason": (
            "§43-§46" if top1 else (
                "B1 sequential transfer dominates all originality-eligible candidates "
                "(review P0-6/§83 STOP)" if original_pass and top1 is None else
                "no original candidate passed the Stage-B gate (§81 STOP)"
            )
        ),
        "novelty_candidates_failed": top1 is None,
    }

    _write_csv(
        output_dir / "D6B_TOP2_3SEED_SUMMARY.csv",
        ["candidate", "seed", "status", "stable_rmse", "best_rmse", "best_epoch",
         "best_sharpness", "trajectory_std", "man_rmse", "women_rmse", "human_rmse"],
        summary_rows,
    )
    _write_csv(
        output_dir / "D6B_PAIRED_COMPARISON.csv",
        ["candidate", "seed", "b0_stable_rmse", "candidate_stable_rmse", "stable_gain"],
        paired_rows,
    )
    _write_csv(
        output_dir / "D6B_GATE.csv",
        ["candidate", "mean_stable_gain", "positive_seeds", "seeds_compared",
         "endpoints_non_worse", "gate"],
        gate_rows,
    )
    return top1


def stage_c(runs_root: Path, top1: str, seeds: list[int], output_dir: Path, trace: dict) -> None:
    # Review P1-9: Stage C only accepts an originality-eligible winner.
    if top1 not in CANDIDATES:
        raise SystemExit(f"unknown Stage-C candidate: {top1!r}")
    if not CANDIDATES[top1]["originality_eligible"]:
        raise SystemExit("Stage C only accepts an originality-eligible Stage-B winner")

    expected_last_epoch = STAGE_LAST_EPOCH["stage-c"]
    summary_rows = []
    endpoint_rows = []
    candidate_stable: dict[int, float] = {}
    b0_stable: dict[int, float] = {}
    b0a_stable: dict[int, float] = {}
    # Review P0-4/§26: when the winner is O1, its anchor-only control B0A runs
    # the same five seeds so the CSDT-vs-anchor causal evidence survives.
    references = ("b0", "b0a") if top1 == "o1" else ("b0",)
    for candidate in (top1,) + references:
        for seed in seeds:
            run_dir = candidate_run_dir(runs_root, candidate, f"d6_{candidate}_e100", seed)
            row = summarize_run(run_dir, candidate, seed, expected_last_epoch)
            summary_rows.append(row)
            if row.get("status") != "PASS":
                continue
            stable = float(row["stable_rmse"])
            if candidate == top1:
                candidate_stable[seed] = stable
            elif candidate == "b0":
                b0_stable[seed] = stable
            elif candidate == "b0a":
                b0a_stable[seed] = stable
            for task in HUMAN_TASKS:
                endpoint_rows.append(
                    {
                        "model": candidate,
                        "seed": seed,
                        "task": task,
                        "stable_rmse": float(row[f"{task.split('_')[0]}_rmse"]),
                    }
                )

    # §42/§52 gain sign: Human3Only_StableRMSE - Candidate_StableRMSE.
    gains = [
        b0_stable[seed] - candidate_stable[seed]
        for seed in seeds
        if seed in candidate_stable and seed in b0_stable
    ]
    mean_gain = _finite_mean(gains)
    positive_seeds = sum(1 for gain in gains if gain > 0)

    candidate_endpoint_means = _endpoint_means(summary_rows, top1)
    b0_endpoint_means = _endpoint_means(summary_rows, "b0")
    endpoints_non_worse = sum(
        1
        for task in HUMAN_TASKS
        if candidate_endpoint_means[task] <= b0_endpoint_means[task] + 1e-9
    )
    _verify_stage_identity(summary_rows, "Stage C")
    # Review P0-5: Stage C is the 5-seed final validation — every seed pair
    # must be present before the gate can pass.
    complete_seed_pairs = len(gains) == len(seeds)
    # Review P0-4/§51-§52: mean gain, seed count AND the endpoint gate.
    gate_pass = bool(
        complete_seed_pairs
        and mean_gain > 0
        and endpoints_non_worse >= 2
        and (
            positive_seeds >= 4
            or (positive_seeds >= 3 and mean_gain >= STRONG_GAIN)
        )
    )
    gate = {
        "paired_mean_stable_gain": mean_gain,
        "positive_seeds": positive_seeds,
        "seeds_compared": len(gains),
        "expected_seed_count": len(seeds),
        "complete_seed_pairs": complete_seed_pairs,
        "endpoints_non_worse": endpoints_non_worse,
        "endpoint_means_candidate": candidate_endpoint_means,
        "endpoint_means_b0": b0_endpoint_means,
        "gate_pass": gate_pass,
    }
    if top1 == "o1":
        # Review P0-2/§15: the anchor causal gate must hold at Stage C too —
        # O1 passing B0 is not enough if it has lost to the anchor-only
        # control (the gain would come from the anchor initialisation, not
        # from the semantic delta).  Strong evidence (>=0.01) is reported but
        # only >0 is required, consistent with Stage B.
        anchor_gains = [
            b0a_stable[seed] - candidate_stable[seed]
            for seed in seeds
            if seed in candidate_stable and seed in b0a_stable
        ]
        complete_anchor_pairs = len(anchor_gains) == len(seeds)
        mean_anchor_gain = _finite_mean(anchor_gains)
        anchor_gate_pass = bool(
            complete_anchor_pairs and mean_anchor_gain > ANCHOR_SEMANTIC_GAIN_MIN
        )
        gate.update(
            {
                "anchor_semantic_gain_mean": mean_anchor_gain,
                "anchor_seed_pairs": len(anchor_gains),
                "complete_anchor_pairs": complete_anchor_pairs,
                "anchor_gate_pass": anchor_gate_pass,
                "anchor_strong_evidence": bool(
                    mean_anchor_gain >= ANCHOR_SEMANTIC_GAIN_STRONG
                ),
            }
        )
        gate["gate_pass"] = bool(gate["gate_pass"] and anchor_gate_pass)
    trace["stage_c"] = {"top1": top1, "seeds": seeds, "gate": gate}

    _write_csv(
        output_dir / "D6C_FINAL_5SEED_SUMMARY.csv",
        ["candidate", "seed", "status", "stable_rmse", "best_rmse", "best_epoch",
         "best_sharpness", "trajectory_std", "man_rmse", "women_rmse", "human_rmse"],
        summary_rows,
    )
    _write_csv(
        output_dir / "D6C_ENDPOINT_SUMMARY.csv",
        ["model", "seed", "task", "stable_rmse"],
        endpoint_rows,
    )
    # Review P1-11: stability rows expose the "model" field properly.
    _write_csv(
        output_dir / "D6C_STABILITY_SUMMARY.csv",
        ["model", "seed", "stable_rmse", "best_rmse", "best_epoch", "best_sharpness", "trajectory_std"],
        [
            {
                "model": row.get("candidate"),
                "seed": row.get("seed"),
                "stable_rmse": row.get("stable_rmse"),
                "best_rmse": row.get("best_rmse"),
                "best_epoch": row.get("best_epoch"),
                "best_sharpness": row.get("best_sharpness"),
                "trajectory_std": row.get("trajectory_std"),
            }
            for row in summary_rows
        ],
    )
    print(json.dumps({"stage_c_gate": gate}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage-a", "stage-b", "stage-c"])
    parser.add_argument("--runs_root", default="artifacts/runs")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top2", nargs="+", default=None, help="Stage-B candidates (from Stage A)")
    parser.add_argument("--top1", default=None, help="Stage-C candidate (from Stage B)")
    parser.add_argument("--csdt_csv", default=None)
    parser.add_argument("--clst_csv", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "D6_SELECTION_TRACE.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}

    if args.mode == "stage-a":
        runs_root = Path(args.runs_root) / "d6_stage_a"
        top2 = stage_a(runs_root, args.seed, output_dir, trace)
        print(f"Stage A Top-2: {top2}")
        if len(top2) < 2:
            print("STOP: " + trace["stage_a"].get("stop_reason", "insufficient eligible candidates"))
    elif args.mode == "stage-b":
        candidates = args.top2 or []
        runs_root = Path(args.runs_root) / "d6_stage_b"
        top1 = stage_b(runs_root, candidates, [42, 44, 46], output_dir, trace)
        print(f"Stage B Top-1: {top1}")
        if top1 is None:
            print("STOP: no original candidate passed the Stage-B gate (plan §81).")
    else:
        if not args.top1:
            raise SystemExit("stage-c requires --top1 from Stage B")
        runs_root = Path(args.runs_root) / "d6_stage_c"
        stage_c(runs_root, args.top1, [42, 43, 44, 45, 46], output_dir, trace)
        print("Stage C complete; STOP before any test/CQR (plan §82).")

    artifacts = trace.setdefault("artifacts", {})
    if args.csdt_csv:
        artifacts["csdt_parameter_delta_csv"] = args.csdt_csv
    if args.clst_csv:
        artifacts["clst_layer_score_csv"] = args.clst_csv
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    print(f"updated {trace_path}")


if __name__ == "__main__":
    main()
