"""Stage 1 Go/No-Go Gate runner (S0 FULL + S0 S21_P10) for RTX A6000 server.

Usage:
    python -m pgt_v2.mv_prompt.gate build-cache --cache <GraphCache> --view <S0/FULL> --out <mv_cache.pt>
    python -m pgt_v2.mv_prompt.gate preflight --mv-cache <mv_cache.pt> --view <S0/FULL> --out <preflight.json> --device cuda:0
    python -m pgt_v2.mv_prompt.gate run-gate --mv-cache <mv_cache.pt> --views-root <views/S0> --groups <endpoint_groups.json> --work <work_dir> --gpus 0 1 2 3
    python -m pgt_v2.mv_prompt.gate summarize --work <work_dir> --views-root <views/S0> --groups <endpoint_groups.json>
    python -m pgt_v2.mv_prompt.gate collect --work <work_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from ..artifacts import load_arrays, load_manifest, read_json, write_json
from ..cache import file_sha256
from ..config import git_identity
from .features import build_mv_cache, fit_descriptor_scaler, load_mv_cache, transform_descriptors
from .models import build_mv_model, count_parameters, method_input_needs
from .training import seed_all, train_and_evaluate


def analyze_toxacol_adjacency(view_root: Path, endpoint_groups: dict) -> dict:
    """Count how many endpoints are isolated (degree=0 cross-edges) in ToxACoL's empirical PCC graph."""
    manifest = load_manifest(view_root)
    tasks = list(manifest["task_names"])
    stats = load_arrays(view_root, "statistics.npz", manifest)
    adj = stats["adjacency"]
    # Off-diagonal positive entries indicate empirical correlation edges (min_shared>=15, pcc>=0.75)
    off_diag = (adj > 0).astype(np.int32)
    np.fill_diagonal(off_diag, 0)
    degrees = off_diag.sum(axis=1)
    by_task = {t: int(degrees[i]) for i, t in enumerate(tasks)}
    groups = {
        "ALL59": tasks,
        "L11": endpoint_groups["L11"],
        "M27": endpoint_groups["M27"],
        "S21": endpoint_groups["S21"],
        "Human3": ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"],
    }
    report = {}
    for gname, members in groups.items():
        degs = [by_task[t] for t in members]
        report[gname] = {
            "tasks": len(members),
            "isolated_zero_edge_count": int(sum(d == 0 for d in degs)),
            "mean_degree": round(float(np.mean(degs)), 2),
            "median_degree": float(np.median(degs)),
        }
    return {"fold": manifest["fold"], "view_id": manifest["view_id"], "groups": report, "per_task_degree": by_task}


def run_preflight(mv_cache_path: Path, view_root: Path, output_path: Path, device: str) -> dict:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Server preflight requires an available CUDA device")
    store = load_mv_cache(mv_cache_path)
    manifest = load_manifest(view_root)
    train_arr = load_arrays(view_root, "train.npz", manifest)
    stats_arr = load_arrays(view_root, "statistics.npz", manifest)
    store.verify_partition(train_arr, manifest)

    train_off = train_arr["cache_offset"].astype(np.int64)
    active = np.where(np.isfinite(train_arr["labels"]).any(axis=1))[0][:16]
    scaler = fit_descriptor_scaler(store.desc_raw, store.desc_missing, train_off[active])
    desc = transform_descriptors(store.desc_raw, store.desc_missing, train_off[active], scaler)

    checked = []
    for method in (
        "TOX_AUTHOR",
        "MV_MLP",
        "GINE_ONLY",
        "GRAPHORMER_ONLY",
        "MV_BIOPGT_NOGRAPH",
        "MV_BIOPGT_NOPROMPT_GINE",
        "MV_BIOPGT_GINE",
        "MV_BIOPGT_GRAPHORMER",
    ):
        seed_all(42)
        needs = method_input_needs(method)
        model = build_mv_model(
            method,
            tuple(manifest["task_names"]),
            stats_arr["count"],
            adjacency=stats_arr["adjacency"],
            device=device,
        )
        model.train()
        payload = store.build_batch(
            train_off[active],
            desc_array=desc if needs["need_desc"] else None,
            need_avalon=needs["need_avalon"],
            need_morgan=needs["need_morgan"],
            need_desc=needs["need_desc"],
            graph_mode=needs["graph_mode"],
            device=device,
        )
        out = model(payload)
        if tuple(out.shape) != (len(active), 59) or not torch.isfinite(out).all():
            raise RuntimeError(f"Invalid forward output for {method}: {tuple(out.shape)}")
        loss = out.square().mean()
        loss.backward()
        grad_ok = all(
            (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        if not grad_ok:
            raise RuntimeError(f"Nonfinite gradient in preflight for {method}")
        checked.append(
            {
                "method": method,
                "parameters": count_parameters(model),
                "output_shape": list(out.shape),
                "finite_forward_and_grad": True,
            }
        )
        del model
        torch.cuda.empty_cache()

    report = {
        "status": "PREFLIGHT_PASS",
        "code_commit": git_identity(),
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "torch": str(torch.__version__),
        "models_checked": checked,
    }
    write_json(output_path, report)
    return report


def _dispatch_jobs(
    jobs: list[dict],
    *,
    mv_cache_path: Path,
    views_root: Path,
    groups_path: Path,
    runs_dir: Path,
    logs_dir: Path,
    gpus: list[int],
) -> None:
    """Run neural jobs in parallel across GPUs and CPU RF jobs alongside."""
    pending_gpu = [j for j in jobs if j["method"] != "RF"]
    pending_cpu = [j for j in jobs if j["method"] == "RF"]
    active_gpu: dict[int, tuple[subprocess.Popen, object, dict]] = {}
    active_cpu: tuple[subprocess.Popen, object, dict] | None = None
    failed = []

    def launch(job: dict, gpu_id: int | None):
        run_dir = runs_dir / job["run_id"]
        if (run_dir / "summary.json").is_file():
            try:
                if read_json(run_dir / "summary.json").get("status") == "COMPLETE":
                    print(f"[gate] SKIP completed {job['run_id']}", flush=True)
                    return None
            except Exception:
                pass
        cfg_file = runs_dir / "_configs" / f"{job['run_id']}.json"
        write_json(cfg_file, job["config"])
        cmd = [
            sys.executable,
            "-B",
            "-m",
            "pgt_v2.mv_prompt.gate",
            "run-one",
            "--mv-cache",
            str(mv_cache_path),
            "--view",
            str(views_root / job["budget"]),
            "--groups",
            str(groups_path),
            "--method",
            job["method"],
            "--seed",
            str(job["seed"]),
            "--config-json",
            str(cfg_file),
            "--out",
            str(run_dir),
            "--device",
            "cpu" if gpu_id is None else "cuda:0",
        ]
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4" if gpu_id is None else "2",
            "MKL_NUM_THREADS": "4" if gpu_id is None else "2",
        }
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_fh = (logs_dir / f"{job['run_id']}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        tag = "CPU" if gpu_id is None else f"GPU{gpu_id}"
        print(f"[gate] START {tag} {job['run_id']} ({job['method']} / {job['budget']})", flush=True)
        return (proc, log_fh, job)

    while pending_gpu or pending_cpu or active_gpu or active_cpu is not None:
        if not failed:
            for g in gpus:
                if g not in active_gpu and pending_gpu:
                    job = pending_gpu.pop(0)
                    started = launch(job, g)
                    if started is not None:
                        active_gpu[g] = started
            if active_cpu is None and pending_cpu:
                job = pending_cpu.pop(0)
                started = launch(job, None)
                if started is not None:
                    active_cpu = started

        for g, (proc, fh, job) in list(active_gpu.items()):
            rc = proc.poll()
            if rc is not None:
                fh.close()
                del active_gpu[g]
                if rc != 0:
                    failed.append((job["run_id"], rc))
                    print(f"[gate] FAILED {job['run_id']} exit={rc}", flush=True)
                else:
                    print(f"[gate] DONE   {job['run_id']}", flush=True)

        if active_cpu is not None:
            proc, fh, job = active_cpu
            rc = proc.poll()
            if rc is not None:
                fh.close()
                active_cpu = None
                if rc != 0:
                    failed.append((job["run_id"], rc))
                    print(f"[gate] FAILED {job['run_id']} exit={rc}", flush=True)
                else:
                    print(f"[gate] DONE   {job['run_id']}", flush=True)

        if active_gpu or active_cpu is not None:
            time.sleep(2.0)

    if failed:
        raise RuntimeError(f"Gate batch failed on runs: {failed}")


def run_gate(
    *,
    mv_cache_path: Path,
    views_root: Path,
    groups_path: Path,
    work_dir: Path,
    gpus: list[int],
    seed: int = 42,
) -> dict:
    runs_dir = work_dir / "runs"
    logs_dir = work_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Wave 1: S0/FULL gamma ablation (3) + independent baselines on FULL & S21_P10 (8)
    wave1 = [
        {"run_id": f"FULL_MV_BIOPGT_GINE_g000_s{seed}", "method": "MV_BIOPGT_GINE", "budget": "FULL", "seed": seed, "config": {"gamma": 0.0}},
        {"run_id": f"FULL_MV_BIOPGT_GINE_g025_s{seed}", "method": "MV_BIOPGT_GINE", "budget": "FULL", "seed": seed, "config": {"gamma": 0.25}},
        {"run_id": f"FULL_MV_BIOPGT_GINE_g050_s{seed}", "method": "MV_BIOPGT_GINE", "budget": "FULL", "seed": seed, "config": {"gamma": 0.50}},
        {"run_id": f"FULL_TOX_AUTHOR_s{seed}", "method": "TOX_AUTHOR", "budget": "FULL", "seed": seed, "config": {}},
        {"run_id": f"FULL_MV_MLP_s{seed}", "method": "MV_MLP", "budget": "FULL", "seed": seed, "config": {"gamma": 0.25}},
        {"run_id": f"FULL_GINE_ONLY_s{seed}", "method": "GINE_ONLY", "budget": "FULL", "seed": seed, "config": {"gamma": 0.25}},
        {"run_id": f"FULL_GRAPHORMER_ONLY_s{seed}", "method": "GRAPHORMER_ONLY", "budget": "FULL", "seed": seed, "config": {"gamma": 0.25}},
        {"run_id": f"FULL_RF_s{seed}", "method": "RF", "budget": "FULL", "seed": seed, "config": {"n_estimators": 300, "n_jobs": 8}},
        {"run_id": f"P10_TOX_AUTHOR_s{seed}", "method": "TOX_AUTHOR", "budget": "S21_P10", "seed": seed, "config": {}},
        {"run_id": f"P10_MV_MLP_s{seed}", "method": "MV_MLP", "budget": "S21_P10", "seed": seed, "config": {"gamma": 0.25}},
        {"run_id": f"P10_RF_s{seed}", "method": "RF", "budget": "S21_P10", "seed": seed, "config": {"n_estimators": 300, "n_jobs": 8}},
    ]
    _dispatch_jobs(
        wave1,
        mv_cache_path=mv_cache_path,
        views_root=views_root,
        groups_path=groups_path,
        runs_dir=runs_dir,
        logs_dir=logs_dir,
        gpus=gpus,
    )

    # Validation-only selection of best gamma
    gamma_candidates = [
        (0.0, f"FULL_MV_BIOPGT_GINE_g000_s{seed}"),
        (0.25, f"FULL_MV_BIOPGT_GINE_g025_s{seed}"),
        (0.50, f"FULL_MV_BIOPGT_GINE_g050_s{seed}"),
    ]
    gamma_scores = []
    for g_val, rid in gamma_candidates:
        val_m = read_json(runs_dir / rid / "validation_metrics.json")
        gamma_scores.append((float(val_m["groups"]["ALL59"]["macro_rmse"]), g_val, rid))
    gamma_scores.sort(key=lambda x: x[0])
    best_gamma = gamma_scores[0][1]
    best_gine_full_id = gamma_scores[0][2]
    print(f"[gate] Selected best_gamma={best_gamma} on S0/FULL validation: {gamma_scores}", flush=True)

    # Wave 2: Graphormer counterpart + FULL ablations at best_gamma (+ MV_MLP at best_gamma if different)
    wave2 = [
        {"run_id": f"FULL_MV_BIOPGT_GRAPHORMER_gbest_s{seed}", "method": "MV_BIOPGT_GRAPHORMER", "budget": "FULL", "seed": seed, "config": {"gamma": best_gamma}},
        {"run_id": f"FULL_MV_BIOPGT_NOGRAPH_gbest_s{seed}", "method": "MV_BIOPGT_NOGRAPH", "budget": "FULL", "seed": seed, "config": {"gamma": best_gamma}},
        {"run_id": f"FULL_MV_BIOPGT_NOPROMPT_GINE_gbest_s{seed}", "method": "MV_BIOPGT_NOPROMPT_GINE", "budget": "FULL", "seed": seed, "config": {"gamma": best_gamma}},
    ]
    if abs(best_gamma - 0.25) > 1e-9:
        wave2.append(
            {"run_id": f"FULL_MV_MLP_gbest_s{seed}", "method": "MV_MLP", "budget": "FULL", "seed": seed, "config": {"gamma": best_gamma}}
        )
    _dispatch_jobs(
        wave2,
        mv_cache_path=mv_cache_path,
        views_root=views_root,
        groups_path=groups_path,
        runs_dir=runs_dir,
        logs_dir=logs_dir,
        gpus=gpus,
    )

    # Validation-only selection between GINE and Graphormer backbone
    gine_val = float(read_json(runs_dir / best_gine_full_id / "validation_metrics.json")["groups"]["ALL59"]["macro_rmse"])
    gformer_id = f"FULL_MV_BIOPGT_GRAPHORMER_gbest_s{seed}"
    gformer_val = float(read_json(runs_dir / gformer_id / "validation_metrics.json")["groups"]["ALL59"]["macro_rmse"])
    if gine_val <= gformer_val:
        best_backbone = "GINE"
        best_method = "MV_BIOPGT_GINE"
        noprompt_method = "MV_BIOPGT_NOPROMPT_GINE"
        best_full_run_id = best_gine_full_id
    else:
        best_backbone = "GRAPHORMER"
        best_method = "MV_BIOPGT_GRAPHORMER"
        noprompt_method = "MV_BIOPGT_NOPROMPT_GRAPHORMER"
        best_full_run_id = gformer_id

    selection = {
        "best_gamma": best_gamma,
        "gamma_validation_scores": [{"gamma": g, "run_id": r, "val_all59_rmse": s} for s, g, r in gamma_scores],
        "best_backbone": best_backbone,
        "best_method": best_method,
        "best_full_run_id": best_full_run_id,
        "gine_val_all59_rmse": gine_val,
        "graphormer_val_all59_rmse": gformer_val,
    }
    write_json(work_dir / "validation_selection.json", selection)
    print(f"[gate] Selected best_backbone={best_backbone} (GINE={gine_val:.5f}, Graphormer={gformer_val:.5f})", flush=True)

    # Wave 3: S21_P10 sparse stress runs for winning backbone + ablations
    wave3 = [
        {"run_id": f"P10_MV_BIOPGT_BEST_s{seed}", "method": best_method, "budget": "S21_P10", "seed": seed, "config": {"gamma": best_gamma}},
        {"run_id": f"P10_MV_BIOPGT_NOGRAPH_s{seed}", "method": "MV_BIOPGT_NOGRAPH", "budget": "S21_P10", "seed": seed, "config": {"gamma": best_gamma}},
        {"run_id": f"P10_MV_BIOPGT_NOPROMPT_s{seed}", "method": noprompt_method, "budget": "S21_P10", "seed": seed, "config": {"gamma": best_gamma}},
    ]
    if best_backbone == "GRAPHORMER":
        wave3.append(
            {"run_id": f"FULL_MV_BIOPGT_NOPROMPT_GRAPHORMER_gbest_s{seed}", "method": "MV_BIOPGT_NOPROMPT_GRAPHORMER", "budget": "FULL", "seed": seed, "config": {"gamma": best_gamma}}
        )
    if abs(best_gamma - 0.25) > 1e-9:
        wave3.append(
            {"run_id": f"P10_MV_MLP_gbest_s{seed}", "method": "MV_MLP", "budget": "S21_P10", "seed": seed, "config": {"gamma": best_gamma}}
        )
    _dispatch_jobs(
        wave3,
        mv_cache_path=mv_cache_path,
        views_root=views_root,
        groups_path=groups_path,
        runs_dir=runs_dir,
        logs_dir=logs_dir,
        gpus=gpus,
    )
    return summarize_gate(work_dir=work_dir, views_root=views_root, groups_path=groups_path)


def summarize_gate(*, work_dir: Path, views_root: Path, groups_path: Path) -> dict:
    runs_dir = work_dir / "runs"
    endpoint_groups = read_json(groups_path)
    selection = read_json(work_dir / "validation_selection.json")
    rows = []
    by_id = {}
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        s = read_json(summary_path)
        rid = summary_path.parent.name
        by_id[rid] = s
        budget = "FULL" if rid.startswith("FULL_") else "S21_P10"
        tg = s["test_groups"]
        vg = s["validation_groups"]
        rows.append(
            {
                "run_id": rid,
                "budget": budget,
                "method": s["method"],
                "gamma": s.get("gamma"),
                "best_epoch": s["best_epoch"],
                "seconds": s["elapsed_seconds"],
                "params": (s.get("parameters") or {}).get("total"),
                "val_ALL59_rmse": vg["ALL59"]["macro_rmse"],
                "test_ALL59_rmse": tg["ALL59"]["macro_rmse"],
                "test_ALL59_r2": tg["ALL59"]["macro_r2"],
                "test_L11_rmse": tg["L11"]["macro_rmse"],
                "test_M27_rmse": tg["M27"]["macro_rmse"],
                "test_S21_rmse": tg["S21"]["macro_rmse"],
                "test_S21_r2": tg["S21"]["macro_r2"],
                "test_Human3_rmse": tg["Human3"]["macro_rmse"],
                "test_Human3_r2": tg["Human3"]["macro_r2"],
            }
        )

    csv_path = work_dir / "gate_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Evaluate pre-registered Go/No-Go criteria on sealed S0 test
    seed = next(iter(by_id.values()))["seed"]
    best_full = by_id[selection["best_full_run_id"]]
    tox_full = by_id[f"FULL_TOX_AUTHOR_s{seed}"]
    mv_mlp_full = by_id.get(f"FULL_MV_MLP_gbest_s{seed}", by_id[f"FULL_MV_MLP_s{seed}"])
    nograph_full = by_id[f"FULL_MV_BIOPGT_NOGRAPH_gbest_s{seed}"]
    noprompt_full = by_id.get(
        f"FULL_MV_BIOPGT_NOPROMPT_{selection['best_backbone']}_gbest_s{seed}",
        by_id[f"FULL_MV_BIOPGT_NOPROMPT_GINE_gbest_s{seed}"],
    )

    best_p10 = by_id[f"P10_MV_BIOPGT_BEST_s{seed}"]
    tox_p10 = by_id[f"P10_TOX_AUTHOR_s{seed}"]

    def rel_improve(cand: float, ref: float) -> float:
        return 100.0 * (ref - cand) / ref

    full_vs_tox_pct = rel_improve(
        best_full["test_groups"]["ALL59"]["macro_rmse"],
        tox_full["test_groups"]["ALL59"]["macro_rmse"],
    )
    full_r2_better_tox = (
        best_full["test_groups"]["ALL59"]["macro_r2"] > tox_full["test_groups"]["ALL59"]["macro_r2"]
    )
    groups_won_vs_tox = sum(
        best_full["test_groups"][g]["macro_rmse"] < tox_full["test_groups"][g]["macro_rmse"]
        for g in ("ALL59", "L11", "M27", "S21")
    )
    p10_vs_tox_pct = rel_improve(
        best_p10["test_groups"]["ALL59"]["macro_rmse"],
        tox_p10["test_groups"]["ALL59"]["macro_rmse"],
    )
    full_vs_mvmlp_pct = rel_improve(
        best_full["test_groups"]["ALL59"]["macro_rmse"],
        mv_mlp_full["test_groups"]["ALL59"]["macro_rmse"],
    )
    full_vs_nograph_pct = rel_improve(
        best_full["test_groups"]["ALL59"]["macro_rmse"],
        nograph_full["test_groups"]["ALL59"]["macro_rmse"],
    )
    full_vs_noprompt_pct = rel_improve(
        best_full["test_groups"]["ALL59"]["macro_rmse"],
        noprompt_full["test_groups"]["ALL59"]["macro_rmse"],
    )

    cond1 = (
        full_vs_tox_pct >= 3.0
        and full_r2_better_tox
        and groups_won_vs_tox >= 3
        and p10_vs_tox_pct > 0.0
    )
    cond2_prompt = full_vs_mvmlp_pct > 0.0 and full_vs_noprompt_pct > 0.0
    cond2_graph = full_vs_nograph_pct > 0.0

    if cond1 and cond2_prompt and cond2_graph:
        verdict = "PASS"
    elif cond1 and cond2_prompt and not cond2_graph:
        verdict = "PARTIAL_PASS_NO_GRAPH_GAIN"
    else:
        verdict = "FAIL_FALLBACK_ROUTE1"

    adj_diag = {
        "FULL": analyze_toxacol_adjacency(views_root / "FULL", endpoint_groups),
        "S21_P10": analyze_toxacol_adjacency(views_root / "S21_P10", endpoint_groups),
    }

    decision = {
        "verdict": verdict,
        "validation_selection": selection,
        "criteria": {
            "cond1_beats_toxacol": cond1,
            "cond2_prompt_helps": cond2_prompt,
            "cond2_graph_helps": cond2_graph,
            "full_vs_toxacol_pct": round(full_vs_tox_pct, 3),
            "full_r2_better_than_toxacol": full_r2_better_tox,
            "groups_won_vs_toxacol": groups_won_vs_tox,
            "p10_vs_toxacol_pct": round(p10_vs_tox_pct, 3),
            "full_vs_mv_mlp_pct": round(full_vs_mvmlp_pct, 3),
            "full_vs_nograph_pct": round(full_vs_nograph_pct, 3),
            "full_vs_noprompt_pct": round(full_vs_noprompt_pct, 3),
        },
        "toxacol_adjacency_isolation": adj_diag,
    }
    write_json(work_dir / "gate_decision.json", decision)

    lines = [
        f"# Stage 1 Gate Report — Verdict: **{verdict}**",
        "",
        f"- Selected on S0/FULL validation: backbone = `{selection['best_backbone']}` (`{selection['best_method']}`), `gamma = {selection['best_gamma']}`",
        f"- FULL vs ToxACoL: **{full_vs_tox_pct:+.2f}%** ALL59 RMSE, R2 better={full_r2_better_tox}, groups won={groups_won_vs_tox}/4",
        f"- S21_P10 vs ToxACoL: **{p10_vs_tox_pct:+.2f}%** ALL59 RMSE",
        f"- FULL vs MV-MLP control: **{full_vs_mvmlp_pct:+.2f}%** | vs noPrompt: **{full_vs_noprompt_pct:+.2f}%** | vs noGraph: **{full_vs_nograph_pct:+.2f}%**",
        "",
        "## ToxACoL Empirical Adjacency Isolation (Zero Cross-Endpoint Edges)",
        "",
        "| View | ALL59 isolated | S21 isolated | Human3 isolated |",
        "|---|---:|---:|---:|",
        f"| S0/FULL | {adj_diag['FULL']['groups']['ALL59']['isolated_zero_edge_count']}/59 | {adj_diag['FULL']['groups']['S21']['isolated_zero_edge_count']}/21 | {adj_diag['FULL']['groups']['Human3']['isolated_zero_edge_count']}/3 |",
        f"| S0/S21_P10 | {adj_diag['S21_P10']['groups']['ALL59']['isolated_zero_edge_count']}/59 | {adj_diag['S21_P10']['groups']['S21']['isolated_zero_edge_count']}/21 | {adj_diag['S21_P10']['groups']['Human3']['isolated_zero_edge_count']}/3 |",
        "",
        "## All Runs (Sealed S0 Test)",
        "",
        "| Run ID | Budget | Method | Gamma | Val ALL59 | Test ALL59 | Test R2 | L11 | M27 | S21 | Human3 | Time(s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["budget"], x["test_ALL59_rmse"])):
        g_str = "-" if r["gamma"] is None else f"{r['gamma']:.2f}"
        lines.append(
            f"| {r['run_id']} | {r['budget']} | {r['method']} | {g_str} | "
            f"{r['val_ALL59_rmse']:.5f} | {r['test_ALL59_rmse']:.5f} | {r['test_ALL59_r2']:.4f} | "
            f"{r['test_L11_rmse']:.5f} | {r['test_M27_rmse']:.5f} | {r['test_S21_rmse']:.5f} | "
            f"{r['test_Human3_rmse']:.5f} | {r['seconds']:.1f} |"
        )
    (work_dir / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]), flush=True)
    return decision


def collect_gate(work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = work_dir / f"PGTV2_MV_GATE_RESULTS_{stamp}.zip"
    include_suffixes = {".json", ".csv", ".md", ".log", ".xml", ".npz"}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(work_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(work_dir).as_posix()
            if rel.startswith("mv_cache/") or path.suffix == ".pt" or path.suffix == ".zip":
                continue
            if path.suffix in include_suffixes:
                zf.write(path, rel)
    sha = file_sha256(archive)
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(
        f"{sha}  {archive.name}\n", encoding="utf-8"
    )
    print(f"[gate] Packaged {archive} ({archive.stat().st_size / 1e6:.2f} MB) sha256={sha}", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cache = sub.add_parser("build-cache")
    p_cache.add_argument("--cache", required=True)
    p_cache.add_argument("--view", required=True)
    p_cache.add_argument("--out", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--mv-cache", required=True)
    p_pre.add_argument("--view", required=True)
    p_pre.add_argument("--out", required=True)
    p_pre.add_argument("--device", default="cuda:0")

    p_one = sub.add_parser("run-one")
    p_one.add_argument("--mv-cache", required=True)
    p_one.add_argument("--view", required=True)
    p_one.add_argument("--groups", required=True)
    p_one.add_argument("--method", required=True)
    p_one.add_argument("--seed", type=int, default=42)
    p_one.add_argument("--config-json", required=True)
    p_one.add_argument("--out", required=True)
    p_one.add_argument("--device", default="cuda:0")

    p_gate = sub.add_parser("run-gate")
    p_gate.add_argument("--mv-cache", required=True)
    p_gate.add_argument("--views-root", required=True)
    p_gate.add_argument("--groups", required=True)
    p_gate.add_argument("--work", required=True)
    p_gate.add_argument("--gpus", type=int, nargs="+", required=True)
    p_gate.add_argument("--seed", type=int, default=42)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--work", required=True)
    p_sum.add_argument("--views-root", required=True)
    p_sum.add_argument("--groups", required=True)

    p_col = sub.add_parser("collect")
    p_col.add_argument("--work", required=True)

    args = parser.parse_args()
    if args.cmd == "build-cache":
        manifest = load_manifest(args.view)
        build_mv_cache(args.cache, args.out, source_sha256=manifest["source_sha256"])
    elif args.cmd == "preflight":
        run_preflight(Path(args.mv_cache), Path(args.view), Path(args.out), args.device)
    elif args.cmd == "run-one":
        store = load_mv_cache(args.mv_cache)
        groups = read_json(args.groups)
        cfg = read_json(args.config_json)
        train_and_evaluate(
            store=store,
            view_root=Path(args.view),
            endpoint_groups=groups,
            method=args.method,
            seed=args.seed,
            config=cfg,
            output_dir=Path(args.out),
            device=args.device,
        )
    elif args.cmd == "run-gate":
        run_gate(
            mv_cache_path=Path(args.mv_cache),
            views_root=Path(args.views_root),
            groups_path=Path(args.groups),
            work_dir=Path(args.work),
            gpus=args.gpus,
            seed=args.seed,
        )
    elif args.cmd == "summarize":
        summarize_gate(
            work_dir=Path(args.work),
            views_root=Path(args.views_root),
            groups_path=Path(args.groups),
        )
    elif args.cmd == "collect":
        collect_gate(Path(args.work))


if __name__ == "__main__":
    main()
