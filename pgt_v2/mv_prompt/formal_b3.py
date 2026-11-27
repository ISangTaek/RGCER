"""Formal Batch 3: External Baselines, Full S21 Curve, R0-R4 Random CV, Prompt-Learning Suite,
PubChem 115-Endpoint External Benchmark (Random + Scaffold 5-Fold), and 59->36 Double Zero-Shot.

Usage:
    python -m pgt_v2.mv_prompt.formal_b3 preflight --mv-cache <mv_cache.pt> --view <S0/FULL> --out <preflight_b3.json> --device cuda:0
    python -m pgt_v2.mv_prompt.formal_b3 prep-views --views-root <views59> --groups <groups59.json> --pub115-csv <pub115.csv.gz> --pub115-random-dir <random_split> --work <work_dir>
    python -m pgt_v2.mv_prompt.formal_b3 run-b3 --mv-cache <mv_cache.pt> --groups <groups59.json> --work <work_dir> --gpus 0 1 2 3 [--prev-b1-dir <dir>] [--prev-b2-dir <dir>]
    python -m pgt_v2.mv_prompt.formal_b3 interpret --mv-cache <mv_cache.pt> --graph-cache <GraphCache> --work <work_dir> --groups <groups59.json> --device cuda:0 [--prev-b1-dir <dir>] [--prev-b2-dir <dir>]
    python -m pgt_v2.mv_prompt.formal_b3 ext-zeroshot --mv-cache <mv_cache.pt> --graph-cache <GraphCache> --pub115-csv <pub115.csv.gz> --work <work_dir> --device cuda:0 [--prev-b1-dir <dir>] [--prev-b2-dir <dir>]
    python -m pgt_v2.mv_prompt.formal_b3 summarize --work <work_dir> --groups <groups59.json>
    python -m pgt_v2.mv_prompt.formal_b3 collect --work <work_dir>
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import silhouette_score
import torch

from ..artifacts import load_arrays, load_manifest, read_json, write_json
from ..cache import GraphCache, file_sha256
from ..config import git_identity
from ..models import task_metadata
from .features import MVDataStore, fit_descriptor_scaler, load_mv_cache, transform_descriptors
from .formal_b1 import S4_CASE_RECORDS, _predict_with_override, _try_import_s0_run
from .formal_b2 import (
    LEO5_ANIMAL_TASKS,
    LEO6_TASKS,
    _donor_z_transfer_baseline,
    _eval_subset_metrics,
    build_all_leo_views,
)
from .models import build_mv_model, count_parameters, method_input_needs
from .probe_v2 import _subdomain_metrics
from .pubchem115 import (
    build_nested_s21_view,
    build_pubchem115_mv_cache,
    build_pubchem115_views,
    find_external_zeroshot_subset,
)
from .training import (
    compute_group_metrics,
    endpoint_weights_for_gamma,
    fit_factor_prior_stats,
    seed_all,
    tempered_mse_loss,
)

RDLogger.DisableLog("rdApp.*")

SCAFFOLD_FOLDS = ("S0", "S1", "S2", "S3", "S4")
RANDOM_FOLDS = ("R0", "R1", "R2", "R3", "R4")
PUB115_RANDOM_FOLDS = ("PUB115_R0", "PUB115_R1", "PUB115_R2", "PUB115_R3", "PUB115_R4")
PUB115_SCAFFOLD_FOLDS = ("PUB115_S0", "PUB115_S1", "PUB115_S2", "PUB115_S3", "PUB115_S4")
S21_NEW_LEVELS = (
    ("S21_P80", "P80", 0.8),
    ("S21_P50", "P50", 0.5),
    ("S21_P40", "P40", 0.4),
    ("S21_P30", "P30", 0.3),
    ("S21_P20", "P20", 0.2),
)

STRUCTURAL_ALERT_SMARTS = {
    "quaternary_ammonium": "[NX4+;!$([N+]-[O-])]",
    "aromatic_nitro": "[c][$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "organophosphate": "[PX4](=[O,S])([O,S,N,C])[O,S,N]",
    "halogenated_aromatic": "[c][F,Cl,Br,I]",
    "michael_acceptor": "[CX3]=[CX3]-[CX3]=[OX1]",
    "nitrile_cyanide": "[CX2]#[NX1]",
}

CROSS_SPECIES_PAIRS = (
    ("mouse_vs_rat_oral", "mouse_oral_LD50", "rat_oral_LD50"),
    ("mouse_oral_vs_iv", "mouse_oral_LD50", "mouse_intravenous_LD50"),
    ("mouse_vs_chicken_oral", "mouse_oral_LD50", "chicken_oral_LD50"),
    ("mouse_vs_quail_oral", "mouse_oral_LD50", "quail_oral_LD50"),
    ("rat_oral_vs_human_tdlo", "rat_oral_LD50", "human_oral_TDLo"),
)


def _safe_float(val) -> float | None:
    if val is None:
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _resolve_checkpoint(
    run_id: str,
    runs_dir: Path,
    search_dirs: list[Path | None],
) -> Path | None:
    """Locate best.pt either inside runs_dir or via reused_ckpt_path / prev directories."""
    direct = runs_dir / run_id / "best.pt"
    if direct.is_file():
        return direct
    s_file = runs_dir / run_id / "summary.json"
    if s_file.is_file():
        try:
            rec = read_json(s_file)
            rp = rec.get("reused_ckpt_path")
            if rp and Path(rp).is_file():
                return Path(rp)
        except Exception:
            pass
    for base in search_dirs:
        if base is None:
            continue
        for cand in (base / "runs" / run_id / "best.pt", base / run_id / "best.pt"):
            if cand.is_file():
                return cand
    return None


def _load_ckpt_model(ckpt_path: Path, device: str | torch.device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    method = str(blob["method"])
    tasks = tuple(blob["task_names"])
    model = build_mv_model(
        method,
        tasks,
        blob["counts"],
        adjacency=blob["adjacency"],
        config=blob.get("config") or {},
        device=device,
    )
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


def run_preflight_b3(
    mv_cache_path: Path,
    view_root: Path,
    output_path: Path,
    device: str,
) -> dict:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Server preflight requires an available CUDA device")

    import lightgbm as lgb
    import xgboost as xgb

    store = load_mv_cache(mv_cache_path)
    manifest = load_manifest(view_root)
    tasks = list(manifest["task_names"])
    train_arr = load_arrays(view_root, "train.npz", manifest)
    stats_arr = load_arrays(view_root, "statistics.npz", manifest)
    store.verify_partition(train_arr, manifest)

    train_off = train_arr["cache_offset"].astype(np.int64)
    active = np.where(np.isfinite(train_arr["labels"]).any(axis=1))[0][:16]
    scaler = fit_descriptor_scaler(store.desc_raw, store.desc_missing, train_off[active])
    desc = transform_descriptors(store.desc_raw, store.desc_missing, train_off[active], scaler)
    ep_w = torch.from_numpy(endpoint_weights_for_gamma(stats_arr["count"], 0.50)).to(device)

    checked = []
    for method in (
        "DMPNN",
        "ATTENTIVEFP",
        "MV_BIOPGT_FINAL_RDROP",
        "MV_BIOPGT_FLAT_PROMPT",
        "TOX_AUTHOR",
    ):
        seed_all(42)
        needs = method_input_needs(method)
        model = build_mv_model(
            method,
            tuple(tasks),
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
        if method in ("MV_BIOPGT_FINAL_RDROP", "MV_BIOPGT_FLAT_PROMPT"):
            out, train_aux = model(payload, return_train_aux=True)
        else:
            out, train_aux = model(payload), {}
        if tuple(out.shape) != (len(active), len(tasks)) or not torch.isfinite(out).all():
            raise RuntimeError(f"Invalid forward output for {method}: {tuple(out.shape)}")
        target = torch.from_numpy(train_arr["labels"][active].astype(np.float32)).to(device)
        loss = tempered_mse_loss(out, target, ep_w)
        if "graph_pred" in train_aux:
            loss = loss + 0.25 * tempered_mse_loss(train_aux["graph_pred"], target, ep_w)
        loss.backward()
        grad_ok = all((p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
        if not grad_ok:
            raise RuntimeError(f"Nonfinite gradient in preflight_b3 for {method}")

        extra_info = {}
        if method == "MV_BIOPGT_FINAL_RDROP":
            model.eval()
            with torch.no_grad():
                ext_p = model.prompt_bank.compose_external_prompts(
                    ["human_oral_LDLo", "dog_intraperitoneal_LD50"]
                )
                ext_out = model(payload, prompt_override=ext_p, external_zero_shrink=True)
            if tuple(ext_out.shape) != (len(active), 2) or not torch.isfinite(ext_out).all():
                raise RuntimeError("External zero-shot prompt composition check failed")
            extra_info["external_prompt_check"] = True
        if method == "TOX_AUTHOR":
            model.eval()
            with torch.no_grad():
                y_pay = model(payload)
                y_av = model.forward_avalon(payload.avalon)
            if not torch.allclose(y_pay, y_av, atol=1e-6):
                raise RuntimeError("TOX_AUTHOR preload_avalon equivalence check failed")
            extra_info["avalon_preload_exact"] = True

        checked.append(
            {
                "method": method,
                "parameters": count_parameters(model),
                "finite_forward_and_grad": True,
                **extra_info,
            }
        )
        del model
        torch.cuda.empty_cache()

    # Quick GBDT smoke check
    x_sm = np.concatenate([store.unpack_morgan(train_off[active]), desc], axis=1)
    y_sm = np.nan_to_num(train_arr["labels"][active, 0], nan=3.0)
    xr = xgb.XGBRegressor(n_estimators=5, max_depth=3, tree_method="hist", device="cuda", verbosity=0)
    xr.fit(x_sm, y_sm)
    lr = lgb.LGBMRegressor(n_estimators=5, num_leaves=7, min_child_samples=2, verbosity=-1)
    lr.fit(x_sm, y_sm)
    if not (np.isfinite(xr.predict(x_sm)).all() and np.isfinite(lr.predict(x_sm)).all()):
        raise RuntimeError("XGBoost / LightGBM smoke check produced nonfinite predictions")

    report = {
        "status": "PREFLIGHT_B3_PASS",
        "code_commit": git_identity(),
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "torch": str(torch.__version__),
        "xgboost": str(xgb.__version__),
        "lightgbm": str(lgb.__version__),
        "models_checked": checked,
    }
    write_json(output_path, report)
    return report


def prepare_b3_assets(
    *,
    views59_root: Path,
    groups59_path: Path,
    pub115_csv: Path | None,
    pub115_random_dir: Path | None,
    work_dir: Path,
    prev_b2_dir: Path | None = None,
    max_workers: int | None = None,
) -> dict:
    """Prepare all nested S21 views, LEO6 views, unified 59 views, and PubChem 115 cache + 10 views."""
    views59_root = Path(views59_root)
    work_dir = Path(work_dir)
    groups59 = read_json(groups59_path)
    s21_tasks = list(groups59["S21"])

    unified59 = work_dir / "views_all_59"
    # Link/copy existing FULL & S21_P30 & S21_P10 views for S0-S4 and R0-R4
    for fold in list(SCAFFOLD_FOLDS) + list(RANDOM_FOLDS):
        for budget in ("FULL", "S21_P30", "S21_P10"):
            src = views59_root / fold / budget
            if (src / "manifest.json").is_file():
                dst = unified59 / fold / budget
                if not (dst / "manifest.json").is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        dst.symlink_to(src.resolve(), target_is_directory=True)
                    except OSError:
                        shutil.copytree(src, dst, dirs_exist_ok=True)

    # Build strictly nested S21_P80, S21_P50, S21_P40, S21_P20 (and S21_P30 if not already present)
    nested_counts = {}
    for fold in SCAFFOLD_FOLDS:
        src_full = unified59 / fold / "FULL"
        if not (src_full / "manifest.json").is_file():
            continue
        prev_mask = None
        for budget, _, frac in sorted(S21_NEW_LEVELS, key=lambda x: x[2]):
            dst = unified59 / fold / budget
            if not (dst / "manifest.json").is_file():
                build_nested_s21_view(
                    src_full,
                    dst,
                    target_tasks=s21_tasks,
                    fraction=frac,
                    budget_id=budget,
                    subset_seed=1042,
                )
            tr_y = np.load(dst / "train.npz")["labels"]
            cur_mask = np.isfinite(tr_y)
            if prev_mask is not None and np.any(prev_mask & ~cur_mask):
                raise AssertionError(f"Nested S21 monotonicity violated at {fold}/{budget}")
            prev_mask = cur_mask
            nested_counts[f"{fold}_{budget}"] = int(cur_mask.sum())

    # Reuse bit-exact LEO6 views from prev_b2_dir if available, else build from FULL
    leo_root = work_dir / "leo_views"
    missing_leo = []
    for fold in SCAFFOLD_FOLDS:
        b2_leo = (Path(prev_b2_dir) / "leo_views" / fold / "LEO6") if prev_b2_dir is not None else None
        dst_leo = unified59 / fold / "LEO6"
        if b2_leo is not None and (b2_leo / "manifest.json").is_file():
            if not (dst_leo / "manifest.json").is_file():
                dst_leo.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst_leo.symlink_to(b2_leo.resolve(), target_is_directory=True)
                except OSError:
                    shutil.copytree(b2_leo, dst_leo, dirs_exist_ok=True)
        else:
            missing_leo.append(fold)
    if missing_leo:
        build_all_leo_views(views59_root, leo_root, folds=tuple(missing_leo), held_out_tasks=LEO6_TASKS)
        for fold in missing_leo:
            src_leo = leo_root / fold / "LEO6"
            if (src_leo / "manifest.json").is_file():
                dst_leo = unified59 / fold / "LEO6"
                if not (dst_leo / "manifest.json").is_file():
                    dst_leo.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        dst_leo.symlink_to(src_leo.resolve(), target_is_directory=True)
                    except OSError:
                        shutil.copytree(src_leo, dst_leo, dirs_exist_ok=True)

    pub115_summary = None
    if pub115_csv is not None and Path(pub115_csv).is_file():
        pub_dir = work_dir / "pubchem115"
        pub_cache = pub_dir / "mv_cache_115.pt"
        build_pubchem115_mv_cache(pub115_csv, pub_cache, max_workers=max_workers)
        pub115_summary = build_pubchem115_views(
            cache_path=pub_cache,
            csv_path=pub115_csv,
            random_split_dir=Path(pub115_random_dir),
            views_out_root=pub_dir / "views",
            groups_out_path=pub_dir / "endpoint_groups_115.json",
        )

    out = {
        "status": "PREP_B3_COMPLETE",
        "unified59_dir": str(unified59),
        "nested_s21_visible_labels": nested_counts,
        "pubchem115": pub115_summary,
    }
    write_json(work_dir / "prep_b3_summary.json", out)
    return out


def build_b3_jobs(
    *,
    include_pub115: bool = True,
    seed: int = 42,
    s0_rdrop_control: bool = True,
) -> list[dict]:
    """Build Formal Batch 3 job queue (145 new units) ordered longest-first across 4 GPUs + CPU."""
    jobs: list[dict] = []

    def add(dataset: str, fold: str, budget: str, btag: str, method: str, cfg: dict, device_kind: str = "gpu"):
        jobs.append(
            {
                "run_id": f"{fold}_{btag}_{method}_s{seed}",
                "dataset": dataset,
                "fold": fold,
                "budget": budget,
                "method": method,
                "seed": seed,
                "config": dict(cfg),
                "device_kind": device_kind,
            }
        )

    # Priority 1: PubChem 115-endpoint GINE models (~18-22 min each)
    if include_pub115:
        for fold in list(PUB115_RANDOM_FOLDS) + list(PUB115_SCAFFOLD_FOLDS):
            add("115", fold, "FULL", "FULL", "MV_BIOPGT_FINAL", {"gamma": 0.50})
        for fold in list(PUB115_RANDOM_FOLDS) + list(PUB115_SCAFFOLD_FOLDS):
            add("115", fold, "FULL", "FULL", "MV_BIOPGT_FINAL_NOPROMPT", {"gamma": 0.50})

    # Priority 2: 59-endpoint GNN/MV_BIOPGT runs (~15-18 min each)
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "FULL", "FULL", "ATTENTIVEFP", {"gamma": 0.50})
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "FULL", "FULL", "DMPNN", {"gamma": 0.50})
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "LEO6", "LEO6", "MV_BIOPGT_FINAL_RDROP", {"gamma": 0.50, "residual_drop_prob": 0.5})
    if s0_rdrop_control:
        add("59", "S0", "FULL", "FULL", "MV_BIOPGT_FINAL_RDROP", {"gamma": 0.50, "residual_drop_prob": 0.5})
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "FULL", "FULL", "MV_BIOPGT_FLAT_PROMPT", {"gamma": 0.50})
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "LEO6", "LEO6", "MV_BIOPGT_FLAT_PROMPT", {"gamma": 0.50})

    for fold in ("R1", "R2", "R3", "R4"):
        add("59", fold, "FULL", "FULL", "MV_BIOPGT_FINAL", {"gamma": 0.50})

    for budget, btag, _ in S21_NEW_LEVELS:
        for fold in SCAFFOLD_FOLDS:
            add("59", fold, budget, btag, "MV_BIOPGT_FINAL", {"gamma": 0.50})
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "S21_P30", "P30", "MV_BIOPGT_FINAL_NOPROMPT", {"gamma": 0.50})

    # Priority 3: Fast GPU-preloaded TOX_AUTHOR (~4-6 min each)
    if include_pub115:
        for fold in list(PUB115_RANDOM_FOLDS) + list(PUB115_SCAFFOLD_FOLDS):
            add("115", fold, "FULL", "FULL", "TOX_AUTHOR", {"preload_avalon_gpu": True})
    for fold in ("R1", "R2", "R3", "R4"):
        add("59", fold, "FULL", "FULL", "TOX_AUTHOR", {"preload_avalon_gpu": True})
    for budget, btag, _ in S21_NEW_LEVELS:
        for fold in SCAFFOLD_FOLDS:
            add("59", fold, budget, btag, "TOX_AUTHOR", {"preload_avalon_gpu": True})

    # Priority 4: Very fast GPU runs (XGBOOST CUDA hist + MV_MLP ~2 min each)
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "FULL", "FULL", "XGBOOST", {"n_estimators": 600, "early_stopping_rounds": 30})
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "S21_P30", "P30", "MV_MLP", {"gamma": 0.50})

    # CPU queue (runs concurrently alongside GPU workers!)
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "FULL", "FULL", "LIGHTGBM", {"n_estimators": 600, "early_stopping_rounds": 30, "n_jobs": 8}, device_kind="cpu")
    for fold in SCAFFOLD_FOLDS:
        add("59", fold, "S21_P30", "P30", "RF", {"n_estimators": 300, "n_jobs": 8}, device_kind="cpu")
    for fold in ("R1", "R2", "R3", "R4"):
        add("59", fold, "FULL", "FULL", "RF", {"n_estimators": 300, "n_jobs": 8}, device_kind="cpu")
    if include_pub115:
        for fold in list(PUB115_RANDOM_FOLDS) + list(PUB115_SCAFFOLD_FOLDS):
            add("115", fold, "FULL", "FULL", "RF", {"n_estimators": 300, "n_jobs": 8}, device_kind="cpu")

    return jobs


def _dispatch_b3_jobs(
    jobs: list[dict],
    *,
    mv_cache_59: Path,
    views59_root: Path,
    groups59_path: Path,
    mv_cache_115: Path | None,
    views115_root: Path | None,
    groups115_path: Path | None,
    runs_dir: Path,
    logs_dir: Path,
    gpus: list[int],
    cpu_workers: int = 2,
) -> None:
    pending_gpu = [j for j in jobs if j["device_kind"] == "gpu"]
    pending_cpu = [j for j in jobs if j["device_kind"] == "cpu"]
    active_gpu: dict[int, tuple[subprocess.Popen, object, dict]] = {}
    active_cpu: dict[int, tuple[subprocess.Popen, object, dict]] = {}
    failed: list[tuple[str, int]] = []

    def launch(job: dict, gpu_id: int | None, cpu_slot: int | None = None):
        run_dir = runs_dir / job["run_id"]
        if (run_dir / "summary.json").is_file():
            try:
                if read_json(run_dir / "summary.json").get("status") == "COMPLETE":
                    print(f"[b3] SKIP completed {job['run_id']}", flush=True)
                    return None
            except Exception:
                pass
        if job["dataset"] == "115":
            cache_p, vroot, gpath = mv_cache_115, views115_root, groups115_path
        else:
            cache_p, vroot, gpath = mv_cache_59, views59_root, groups59_path
        cfg_file = runs_dir / "_configs" / f"{job['run_id']}.json"
        write_json(cfg_file, job["config"])
        cmd = [
            sys.executable,
            "-B",
            "-m",
            "pgt_v2.mv_prompt.gate",
            "run-one",
            "--mv-cache",
            str(cache_p),
            "--view",
            str(Path(vroot) / job["fold"] / job["budget"]),
            "--groups",
            str(gpath),
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
            "OMP_NUM_THREADS": "8" if gpu_id is None else "2",
            "MKL_NUM_THREADS": "8" if gpu_id is None else "2",
        }
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_fh = (logs_dir / f"{job['run_id']}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        tag = f"CPU{cpu_slot}" if gpu_id is None else f"GPU{gpu_id}"
        print(
            f"[b3] START {tag} {job['run_id']} ({job['dataset']} {job['fold']}/{job['budget']} {job['method']})",
            flush=True,
        )
        return (proc, log_fh, job)

    cpu_slots = list(range(max(1, cpu_workers)))
    while pending_gpu or pending_cpu or active_gpu or active_cpu:
        if not failed:
            for g in gpus:
                if g not in active_gpu and pending_gpu:
                    job = pending_gpu.pop(0)
                    started = launch(job, g)
                    if started is not None:
                        active_gpu[g] = started
            for cslot in cpu_slots:
                if cslot not in active_cpu and pending_cpu:
                    job = pending_cpu.pop(0)
                    started = launch(job, None, cpu_slot=cslot)
                    if started is not None:
                        active_cpu[cslot] = started

        for g, (proc, fh, job) in list(active_gpu.items()):
            rc = proc.poll()
            if rc is not None:
                fh.close()
                del active_gpu[g]
                if rc != 0:
                    failed.append((job["run_id"], rc))
                    print(f"[b3] FAILED {job['run_id']} exit={rc}", flush=True)
                else:
                    print(f"[b3] DONE   {job['run_id']}", flush=True)

        for cslot, (proc, fh, job) in list(active_cpu.items()):
            rc = proc.poll()
            if rc is not None:
                fh.close()
                del active_cpu[cslot]
                if rc != 0:
                    failed.append((job["run_id"], rc))
                    print(f"[b3] FAILED {job['run_id']} exit={rc}", flush=True)
                else:
                    print(f"[b3] DONE   {job['run_id']}", flush=True)

        if active_gpu or active_cpu:
            time.sleep(2.0)

    if failed:
        raise RuntimeError(f"Formal Batch 3 failed on runs: {failed}")


def _import_reused_runs(
    *,
    unified59: Path,
    runs_dir: Path,
    prev_b1_dir: Path | None,
    prev_b2_dir: Path | None,
    seed: int = 42,
) -> list[str]:
    reused = []
    # Reuse S0-S4 FULL & P10 from B1, and R0 FULL & S0-S4 LEO6 from B2
    sources = []
    if prev_b1_dir is not None:
        sources.append(Path(prev_b1_dir) / "runs")
    if prev_b2_dir is not None:
        sources.append(Path(prev_b2_dir) / "runs")

    candidates = []
    for fold in SCAFFOLD_FOLDS:
        for m in ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "MV_BIOPGT_FINAL_NOPROMPT", "MV_BIOPGT_FINAL_NOMDROP", "MV_BIOPGT_FINAL_NOGRAPH", "MV_MLP", "GINE_ONLY", "RF"):
            candidates.append((fold, "FULL", "FULL", m))
        for m in ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "MV_BIOPGT_FINAL_NOPROMPT", "MV_BIOPGT_FINAL_NOGRAPH", "MV_MLP", "RF"):
            candidates.append((fold, "S21_P10", "P10", m))
        for m in ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "MV_BIOPGT_FINAL_NOPROMPT", "MV_BIOPGT_FINAL_NOGRAPH"):
            candidates.append((fold, "LEO6", "LEO6", m))
    for m in ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "RF"):
        candidates.append(("R0", "FULL", "FULL", m))

    for fold, budget, btag, method in candidates:
        vdir = unified59 / fold / budget
        if not (vdir / "manifest.json").is_file():
            continue
        vsha = load_manifest(vdir)["manifest_sha256"]
        rid = f"{fold}_{btag}_{method}_s{seed}"
        dst = runs_dir / rid
        for sroot in sources:
            src = sroot / rid
            if not src.is_dir():
                continue
            if _try_import_s0_run(src_dir=src, dst_dir=dst, expected_view_sha=vsha, canonical_method=method):
                reused.append(rid)
                break
    return reused


# ---------------------------------------------------------------------------
# Module 6: Prompt-Learning & Cross-Species Interpretability Suite
# ---------------------------------------------------------------------------

def _prompt_geometry_metrics(prompts: np.ndarray, tasks: list[str]) -> dict:
    pn = prompts / (np.linalg.norm(prompts, axis=1, keepdims=True) + 1e-12)
    cos = pn @ pn.T
    meta = task_metadata(tuple(tasks))
    orgs = [meta[t]["organism"] for t in tasks]
    rts = [meta[t]["route"] for t in tasks]
    meass = [meta[t]["measurement"] for t in tasks]

    def _within_between(labels: list[str]) -> dict:
        w, b = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                (w if labels[i] == labels[j] else b).append(float(cos[i, j]))
        # Silhouette over groups with >= 2 members
        counts = {}
        for lb in labels:
            counts[lb] = counts.get(lb, 0) + 1
        valid_idx = [i for i, lb in enumerate(labels) if counts[lb] >= 2]
        valid_lbls = [labels[i] for i in valid_idx]
        sil = None
        if len(set(valid_lbls)) >= 2 and len(valid_idx) > len(set(valid_lbls)):
            sil = float(silhouette_score(pn[valid_idx], valid_lbls, metric="cosine"))
        return {
            "within_cos": _safe_float(np.mean(w)) if w else None,
            "between_cos": _safe_float(np.mean(b)) if b else None,
            "gap": _safe_float((np.mean(w) - np.mean(b))) if (w and b) else None,
            "cosine_silhouette": _safe_float(sil),
        }

    off_diag = cos[~np.eye(len(tasks), dtype=bool)]
    return {
        "n_tasks": len(tasks),
        "global_mean_offdiag_cos": _safe_float(np.mean(off_diag)),
        "global_sd_offdiag_cos": _safe_float(np.std(off_diag)),
        "organism": _within_between(orgs),
        "route": _within_between(rts),
        "measurement": _within_between(meass),
    }


def _integrated_gradients_atoms(
    model,
    payload,
    *,
    target_fn,
    steps: int = 16,
) -> np.ndarray:
    """Exact Integrated Gradients w.r.t. GINE continuous atom input embeddings h0 (zero baseline)."""
    model.eval()
    with torch.no_grad():
        h0_full, e_emb = model.graph_encoder.encode_atom_inputs(payload)
    h0_full = h0_full.detach()
    e_emb = e_emb.detach()
    grad_acc = torch.zeros_like(h0_full)

    for s in range(1, steps + 1):
        alpha = float(s) / float(steps)
        h_interp = (h0_full * alpha).detach().requires_grad_(True)
        out = model(payload, atom_h0_override=h_interp, edge_emb_override=e_emb)
        scalar = target_fn(out)
        grads = torch.autograd.grad(scalar, h_interp, retain_graph=False, create_graph=False)[0]
        grad_acc = grad_acc + grads.detach()

    ig_vec = (h0_full * (grad_acc / float(steps))).sum(dim=-1)  # [total_nodes]
    return ig_vec.detach().cpu().numpy().astype(np.float32)


def run_prompt_interpretability_suite(
    *,
    mv_cache_59: Path,
    graph_cache_59: Path | None,
    work_dir: Path,
    groups59_path: Path,
    device: str = "cuda:0",
    prev_b1_dir: Path | None = None,
    prev_b2_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """Run P1 (prompt geometry), P2 (counterfactual prompt editing), P3 (IG, deletion, gate, h_mol)."""
    store = load_mv_cache(mv_cache_59)
    runs_dir = work_dir / "runs"
    unified59 = work_dir / "views_all_59"
    interp_dir = work_dir / "interpretability_b3"
    interp_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(
        device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    search_dirs = [prev_b1_dir, prev_b2_dir]

    # ---- P1: Prompt Geometry & Factor Tables ----
    p1_report = {}
    geom_targets = [
        ("S0_FULL_MV_BIOPGT_FINAL", f"S0_FULL_MV_BIOPGT_FINAL_s{seed}"),
        ("S0_FULL_MV_BIOPGT_FINAL_RDROP", f"S0_FULL_MV_BIOPGT_FINAL_RDROP_s{seed}"),
        ("S0_FULL_MV_BIOPGT_FLAT_PROMPT", f"S0_FULL_MV_BIOPGT_FLAT_PROMPT_s{seed}"),
        ("PUB115_S0_FULL_MV_BIOPGT_FINAL", f"PUB115_S0_FULL_MV_BIOPGT_FINAL_s{seed}"),
    ]
    for tag, rid in geom_targets:
        ckpt = _resolve_checkpoint(rid, runs_dir, search_dirs)
        if ckpt is None:
            continue
        model, blob = _load_ckpt_model(ckpt, dev)
        if not getattr(model, "use_prompt", False) or model.prompt_bank is None:
            continue
        tasks = list(blob["task_names"])
        with torch.no_grad():
            p_final, p_pre = model.prompt_bank(return_pre_graph=True)
            p_final_np = p_final.detach().cpu().numpy()
            p_pre_np = p_pre.detach().cpu().numpy()
            shrink_np = model.prompt_bank.shrinkage.detach().cpu().numpy()
            rel_gate = float(torch.tanh(model.prompt_bank.rel_gate).item()) if model.prompt_bank.rel_gate is not None else 0.0
            attn_gate = float(torch.tanh(model.atom_cross_attn.gate).item()) if model.atom_cross_attn is not None else 0.0
            ep_w_norm = float(model.endpoint_weight.detach().norm().item())
        ftables = model.prompt_bank.export_factor_tables()
        save_dict = {
            "task_names": np.asarray(tasks),
            "prompts_final": p_final_np,
            "prompts_pre_graph": p_pre_np,
            "shrinkage": shrink_np,
            "rel_gate": np.asarray([rel_gate], dtype=np.float32),
            "attn_gate": np.asarray([attn_gate], dtype=np.float32),
        }
        for fk, fd in ftables.items():
            save_dict[f"factor_{fk}_vocab"] = np.asarray(fd["vocab"])
            save_dict[f"factor_{fk}_weight"] = fd["weight"]
        np.savez_compressed(interp_dir / f"prompt_geometry_{tag}.npz", **save_dict)

        p1_report[tag] = {
            "prompt_mode": model.prompt_bank.prompt_mode,
            "rel_gate_tanh": round(rel_gate, 5),
            "atom_attn_gate_tanh": round(attn_gate, 5),
            "endpoint_weight_l2": round(ep_w_norm, 4),
            "pre_graph_geometry": _prompt_geometry_metrics(p_pre_np, tasks),
            "post_graph_geometry": _prompt_geometry_metrics(p_final_np, tasks),
        }
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    write_json(interp_dir / "prompt_geometry_b3.json", p1_report)

    # ---- P2: Counterfactual Biological Prompt Editing on S0/FULL (FINAL vs RDROP) ----
    vdir_s0 = unified59 / "S0" / "FULL"
    cf_rows = []
    emb_payload: dict[str, np.ndarray] = {}
    if (vdir_s0 / "manifest.json").is_file():
        m_s0 = load_manifest(vdir_s0)
        tasks59 = list(m_s0["task_names"])
        test_arr = load_arrays(vdir_s0, "test.npz", m_s0)
        test_off = test_arr["cache_offset"].astype(np.int64)
        test_y = test_arr["labels"].astype(np.float64)
        ep_groups = read_json(groups59_path)
        meta59 = task_metadata(tuple(tasks59))
        emb_payload["record_id"] = test_arr["record_id"]
        emb_payload["test_labels"] = test_y.astype(np.float32)

        for tag, rid in (
            ("MV_BIOPGT_FINAL", f"S0_FULL_MV_BIOPGT_FINAL_s{seed}"),
            ("MV_BIOPGT_FINAL_RDROP", f"S0_FULL_MV_BIOPGT_FINAL_RDROP_s{seed}"),
        ):
            ckpt = _resolve_checkpoint(rid, runs_dir, search_dirs)
            if ckpt is None:
                continue
            model, blob = _load_ckpt_model(ckpt, dev)
            test_desc = transform_descriptors(
                store.desc_raw, store.desc_missing, test_off, blob["desc_scaler"]
            )
            mean = np.asarray(blob["target_mean"], dtype=np.float64)
            scale = np.asarray(blob["target_scale"], dtype=np.float64)

            # Export h_mol, modal_gate, shared_pred, residual_pred on S0 test set
            h_list, g_list, sp_list, rp_list = [], [], [], []
            needs = method_input_needs(str(blob["method"]))
            with torch.no_grad():
                for st in range(0, len(test_off), 256):
                    en = min(st + 256, len(test_off))
                    pay = store.build_batch(
                        test_off[st:en],
                        desc_array=test_desc[st:en],
                        need_avalon=needs["need_avalon"],
                        need_morgan=needs["need_morgan"],
                        need_desc=needs["need_desc"],
                        graph_mode=needs["graph_mode"],
                        device=dev,
                    )
                    _, aux = model(pay, return_aux=True)
                    h_list.append(aux["h_mol"].detach().cpu().numpy().astype(np.float16))
                    if "modal_gate" in aux:
                        g_list.append(aux["modal_gate"].mean(dim=-1).cpu().numpy().astype(np.float32))
                    if "shared_pred" in aux:
                        sp_list.append(aux["shared_pred"].cpu().numpy().astype(np.float32))
                    if "residual_pred" in aux:
                        rp_list.append(aux["residual_pred"].cpu().numpy().astype(np.float32))

            emb_payload[f"{tag}_h_mol"] = np.concatenate(h_list, axis=0)
            if g_list:
                emb_payload[f"{tag}_modal_gate_mean"] = np.concatenate(g_list, axis=0)
            shared_var_frac = None
            if sp_list and rp_list:
                sp_all = np.concatenate(sp_list, axis=0)
                rp_all = np.concatenate(rp_list, axis=0)
                v_s = float(np.var(sp_all))
                v_r = float(np.var(rp_all))
                shared_var_frac = v_s / max(1e-12, v_s + v_r)

            prompts = model.prompt_bank().detach()
            org_vocab = list(model.prompt_bank.vocabs["organism"])
            rt_vocab = list(model.prompt_bank.vocabs["route"])
            ms_vocab = list(model.prompt_bank.vocabs["measurement"])

            orig_rt = model.prompt_bank.route_idx.clone()
            swap_rt = orig_rt.clone()
            if "oral" in rt_vocab and "intravenous" in rt_vocab:
                i_or, i_iv = rt_vocab.index("oral"), rt_vocab.index("intravenous")
                swap_rt[orig_rt == i_or] = i_iv
                swap_rt[orig_rt == i_iv] = i_or

            orig_org = model.prompt_bank.organism_idx.clone()
            shift_org = (orig_org + 1) % max(1, len(org_vocab))
            avian_org = orig_org.clone()
            if "chicken" in org_vocab:
                avian_org[:] = org_vocab.index("chicken")

            gen = torch.Generator(device="cpu").manual_seed(20261127)
            perm_p = prompts[torch.randperm(len(tasks59), generator=gen).to(prompts.device)]

            # Pure prompt-path (zero residual) predictions to test counterfactual biological ordering
            def _predict_shared_only(f_ov: dict[str, torch.Tensor] | None):
                out_chunks = []
                with torch.no_grad():
                    for st in range(0, len(test_off), 256):
                        en = min(st + 256, len(test_off))
                        pay = store.build_batch(
                            test_off[st:en],
                            desc_array=test_desc[st:en],
                            need_avalon=True,
                            need_morgan=True,
                            need_desc=True,
                            graph_mode="gine",
                            device=dev,
                        )
                        p_ov = model.prompt_bank(factor_override=f_ov) if f_ov is not None else None
                        pr = model(pay, prompt_override=p_ov, external_zero_shrink=True)
                        out_chunks.append(pr.detach().cpu().numpy().astype(np.float32))
                return np.concatenate(out_chunks, axis=0)

            # Test indicator ladder (LD50 -> LDLo -> TDLo) & route bioavailability (oral -> IV) on pure prompt decoder
            ladder_info = {}
            if all(k in ms_vocab for k in ("LD50", "LDLo", "TDLo")):
                z_ld50 = _predict_shared_only({"measurement": torch.full_like(model.prompt_bank.measurement_idx, ms_vocab.index("LD50"))})
                z_ldlo = _predict_shared_only({"measurement": torch.full_like(model.prompt_bank.measurement_idx, ms_vocab.index("LDLo"))})
                z_tdlo = _predict_shared_only({"measurement": torch.full_like(model.prompt_bank.measurement_idx, ms_vocab.index("TDLo"))})
                ladder_info = {
                    "prompt_only_mean_z_LD50": float(np.mean(z_ld50)),
                    "prompt_only_mean_z_LDLo": float(np.mean(z_ldlo)),
                    "prompt_only_mean_z_TDLo": float(np.mean(z_tdlo)),
                    "frac_z_TDLo_gt_LD50": float(np.mean(z_tdlo > z_ld50)),
                }
            if "oral" in rt_vocab and "intravenous" in rt_vocab:
                z_oral = _predict_shared_only({"route": torch.full_like(orig_rt, rt_vocab.index("oral"))})
                z_iv = _predict_shared_only({"route": torch.full_like(orig_rt, rt_vocab.index("intravenous"))})
                ladder_info["prompt_only_mean_z_oral"] = float(np.mean(z_oral))
                ladder_info["prompt_only_mean_z_iv"] = float(np.mean(z_iv))
                ladder_info["frac_z_iv_gt_oral"] = float(np.mean(z_iv > z_oral))

            pred_intact = _predict_with_override(model, store, test_off, test_desc, mean, scale, device=dev)
            for cname, p_ov, f_ov in (
                ("intact", None, None),
                ("perm_all", perm_p, None),
                ("swap_oral_iv", None, {"route": swap_rt}),
                ("shift_species", None, {"organism": shift_org}),
                ("mammal_to_chicken", None, {"organism": avian_org}),
            ):
                pred_c = (
                    pred_intact
                    if cname == "intact"
                    else _predict_with_override(
                        model, store, test_off, test_desc, mean, scale, device=dev,
                        prompt_override=p_ov, factor_override=f_ov,
                    )
                )
                rep = compute_group_metrics(test_y, pred_c, tasks59, ep_groups, groups=test_arr["group_id"])
                z_mad = float(np.mean(np.abs((pred_c - pred_intact) / scale[None, :])))
                cf_rows.append(
                    {
                        "model": tag,
                        "condition": cname,
                        "ALL59_rmse": rep["groups"]["ALL59"]["macro_rmse"],
                        "ALL59_r2": rep["groups"]["ALL59"]["macro_r2"],
                        "S21_rmse": rep["groups"]["S21"]["macro_rmse"],
                        "Human3_rmse": rep["groups"]["Human3"]["macro_rmse"],
                        "z_shift_mad": z_mad,
                        "shared_var_fraction": shared_var_frac,
                        **{k: v for k, v in ladder_info.items() if cname == "intact"},
                    }
                )
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        if emb_payload:
            np.savez_compressed(interp_dir / "s0_test_embeddings_and_gates.npz", **emb_payload)

    # Also run LEO6 prompt interventions on S0-S4 MV_BIOPGT_FINAL_RDROP
    leo_rdrop_rows = []
    for fold in SCAFFOLD_FOLDS:
        vdir = unified59 / fold / "LEO6"
        ckpt = _resolve_checkpoint(f"{fold}_LEO6_MV_BIOPGT_FINAL_RDROP_s{seed}", runs_dir, search_dirs)
        if not (vdir / "manifest.json").is_file() or ckpt is None:
            continue
        model, blob = _load_ckpt_model(ckpt, dev)
        m_leo = load_manifest(vdir)
        tasks = list(m_leo["task_names"])
        test_arr = load_arrays(vdir, "test.npz", m_leo)
        test_off = test_arr["cache_offset"].astype(np.int64)
        test_desc = transform_descriptors(store.desc_raw, store.desc_missing, test_off, blob["desc_scaler"])
        mean = np.asarray(blob["target_mean"], dtype=np.float64)
        scale = np.asarray(blob["target_scale"], dtype=np.float64)
        prompts = model.prompt_bank().detach()
        rt_vocab = list(model.prompt_bank.vocabs["route"])
        org_vocab = list(model.prompt_bank.vocabs["organism"])
        orig_rt = model.prompt_bank.route_idx.clone()
        swap_rt = orig_rt.clone()
        if "oral" in rt_vocab and "intravenous" in rt_vocab:
            i_or, i_iv = rt_vocab.index("oral"), rt_vocab.index("intravenous")
            swap_rt[orig_rt == i_or] = i_iv
            swap_rt[orig_rt == i_iv] = i_or
        shift_org = (model.prompt_bank.organism_idx.clone() + 1) % max(1, len(org_vocab))
        gen = torch.Generator(device="cpu").manual_seed(20261127)
        perm_p = prompts[torch.randperm(len(tasks), generator=gen).to(prompts.device)]

        pred_intact = _predict_with_override(model, store, test_off, test_desc, mean, scale, device=dev)
        active_held = [t for t in LEO6_TASKS if t in tasks]
        held_idx = [tasks.index(t) for t in active_held]
        for cname, p_ov, f_ov in (
            ("intact", None, None),
            ("perm_all", perm_p, None),
            ("swap_oral_iv", None, {"route": swap_rt}),
            ("shift_species", None, {"organism": shift_org}),
        ):
            pred_c = (
                pred_intact
                if cname == "intact"
                else _predict_with_override(
                    model, store, test_off, test_desc, mean, scale, device=dev,
                    prompt_override=p_ov, factor_override=f_ov,
                )
            )
            z_mad_leo = float(np.mean(np.abs((pred_c[:, held_idx] - pred_intact[:, held_idx]) / scale[None, held_idx])))
            sub = _eval_subset_metrics(test_arr["labels"], pred_c, tasks, tuple(active_held))
            leo_rdrop_rows.append(
                {
                    "fold": fold,
                    "model": "MV_BIOPGT_FINAL_RDROP",
                    "condition": cname,
                    "LEO6_rmse": sub.get("macro_rmse"),
                    "LEO6_r2": sub.get("macro_r2"),
                    "LEO6_pearson_r": sub.get("macro_pearson_r"),
                    "LEO6_z_shift_mad": z_mad_leo,
                }
            )
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if cf_rows:
        keys = list(dict.fromkeys(k for r in cf_rows for k in r.keys()))
        with (interp_dir / "prompt_counterfactual_b3.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(cf_rows)
    if leo_rdrop_rows:
        with (interp_dir / "leo_rdrop_prompt_interventions.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(leo_rdrop_rows[0].keys()))
            w.writeheader()
            w.writerows(leo_rdrop_rows)

    # ---- P3: Faithful Integrated Gradients, Structural Alert Enrichment, Cross-Species Diff & Deletion ----
    alert_report: dict[str, dict] = {}
    cross_species_report: dict[str, list] = {}
    deletion_report: dict[str, dict] = {}

    if (vdir_s0 / "manifest.json").is_file() and graph_cache_59 is not None and Path(graph_cache_59).is_dir():
        m_s0 = load_manifest(vdir_s0)
        tasks59 = list(m_s0["task_names"])
        test_arr = load_arrays(vdir_s0, "test.npz", m_s0)
        test_off = test_arr["cache_offset"].astype(np.int64)
        test_y = test_arr["labels"].astype(np.float64)
        gcache = GraphCache(graph_cache_59, source_sha256=m_s0["source_sha256"], verify_database=False)

        # Match structural alert SMARTS once on S0 test molecules with >=1 observed label
        smarts_mols = {name: Chem.MolFromSmarts(pat) for name, pat in STRUCTURAL_ALERT_SMARTS.items()}
        alert_matches: dict[str, list[tuple[int, int, list[int]]]] = {k: [] for k in smarts_mols}
        obs_ep_per_row = [int(np.flatnonzero(np.isfinite(test_y[i]))[0]) if np.isfinite(test_y[i]).any() else -1 for i in range(len(test_off))]

        for row_i, off in enumerate(test_off):
            if all(len(v) >= 40 for v in alert_matches.values()):
                break
            ep_idx = obs_ep_per_row[row_i]
            if ep_idx < 0:
                continue
            _, mol = gcache.read(int(off))
            n_atoms = mol.GetNumAtoms()
            if n_atoms < 6:
                continue
            for aname, qmol in smarts_mols.items():
                if len(alert_matches[aname]) >= 40 or qmol is None:
                    continue
                hits = mol.GetSubstructMatches(qmol)
                if hits:
                    atom_set = sorted({a for match in hits for a in match if 0 <= a < n_atoms})
                    if 0 < len(atom_set) < n_atoms:
                        alert_matches[aname].append((row_i, ep_idx, atom_set))

        # Select molecules for cross-species differential attribution & deletion faithfulness
        pair_mols: dict[str, list[int]] = {}
        for pname, t_a, t_b in CROSS_SPECIES_PAIRS:
            if t_a in tasks59 and t_b in tasks59:
                ia, ib = tasks59.index(t_a), tasks59.index(t_b)
                cand = np.flatnonzero(np.isfinite(test_y[:, ia]) & np.isfinite(test_y[:, ib]))
                # Keep molecules with >= 6 atoms
                valid_c = [int(i) for i in cand if int(store.num_nodes[test_off[i]]) >= 6][:35]
                pair_mols[pname] = valid_c

        # Deletion faithfulness test molecules (up to 80 test molecules with mouse_oral_LD50 or rat_oral_LD50)
        del_candidates: list[tuple[int, int]] = []
        for tname in ("mouse_oral_LD50", "rat_oral_LD50"):
            if tname not in tasks59:
                continue
            ti = tasks59.index(tname)
            for i in np.flatnonzero(np.isfinite(test_y[:, ti])):
                if len(del_candidates) >= 80:
                    break
                if int(store.num_nodes[test_off[i]]) >= 10:
                    del_candidates.append((int(i), ti))

        for tag, rid in (
            ("MV_BIOPGT_FINAL", f"S0_FULL_MV_BIOPGT_FINAL_s{seed}"),
            ("MV_BIOPGT_FINAL_RDROP", f"S0_FULL_MV_BIOPGT_FINAL_RDROP_s{seed}"),
        ):
            ckpt = _resolve_checkpoint(rid, runs_dir, search_dirs)
            if ckpt is None:
                continue
            model, blob = _load_ckpt_model(ckpt, dev)
            test_desc = transform_descriptors(store.desc_raw, store.desc_missing, test_off, blob["desc_scaler"])

            # 1) Structural alert IG enrichment
            alert_report[tag] = {}
            for aname, items in alert_matches.items():
                al_means, bg_means, ratios = [], [], []
                for row_i, ep_idx, atom_set in items:
                    pay = store.build_batch(
                        test_off[row_i : row_i + 1],
                        desc_array=test_desc[row_i : row_i + 1],
                        need_avalon=True,
                        need_morgan=True,
                        need_desc=True,
                        graph_mode="gine",
                        device=dev,
                    )
                    ig = np.abs(_integrated_gradients_atoms(model, pay, target_fn=lambda o, e=ep_idx: o[0, e], steps=16))
                    mask_a = np.zeros(len(ig), dtype=bool)
                    mask_a[atom_set] = True
                    m_al = float(np.mean(ig[mask_a]))
                    m_bg = float(np.mean(ig[~mask_a]))
                    al_means.append(m_al)
                    bg_means.append(m_bg)
                    ratios.append(m_al / max(1e-12, m_bg))
                pval = None
                if len(al_means) >= 6:
                    try:
                        pval = float(wilcoxon(al_means, bg_means, alternative="greater").pvalue)
                    except Exception:
                        pval = None
                alert_report[tag][aname] = {
                    "n_molecules": len(al_means),
                    "mean_alert_ig": _safe_float(np.mean(al_means)) if al_means else None,
                    "mean_bg_ig": _safe_float(np.mean(bg_means)) if bg_means else None,
                    "median_enrichment_ratio": _safe_float(np.median(ratios)) if ratios else None,
                    "frac_alert_gt_bg": _safe_float(np.mean([a > b for a, b in zip(al_means, bg_means)])) if al_means else None,
                    "wilcoxon_p_greater": _safe_float(pval),
                }

            # 2) Cross-species differential IG + Raw TaskAtomCrossAttention L1 comparison
            cs_rows = []
            for pname, t_a, t_b in CROSS_SPECIES_PAIRS:
                cands = pair_mols.get(pname, [])
                if not cands:
                    continue
                ia, ib = tasks59.index(t_a), tasks59.index(t_b)
                attn_l1s, ig_cos_list, ig_diff_norms = [], [], []
                for row_i in cands:
                    pay = store.build_batch(
                        test_off[row_i : row_i + 1],
                        desc_array=test_desc[row_i : row_i + 1],
                        need_avalon=True,
                        need_morgan=True,
                        need_desc=True,
                        graph_mode="gine",
                        device=dev,
                    )
                    with torch.no_grad():
                        _, aux = model(pay, return_aux=True)
                    n_v = int(pay.num_nodes[0].item())
                    if "atom_attention" in aux:
                        at = aux["atom_attention"][0, :, :, :n_v].mean(dim=1).cpu().numpy()  # [T, n_v]
                        attn_l1s.append(float(np.abs(at[ia] - at[ib]).sum()))
                    ig_a = _integrated_gradients_atoms(model, pay, target_fn=lambda o, e=ia: o[0, e], steps=12)
                    ig_b = _integrated_gradients_atoms(model, pay, target_fn=lambda o, e=ib: o[0, e], steps=12)
                    ig_d = ig_a - ig_b
                    denom = (np.linalg.norm(ig_a) * np.linalg.norm(ig_b)) + 1e-12
                    ig_cos_list.append(float(np.dot(ig_a, ig_b) / denom))
                    ig_diff_norms.append(float(np.linalg.norm(ig_d) / max(1e-12, 0.5 * (np.linalg.norm(ig_a) + np.linalg.norm(ig_b)))))
                cs_rows.append(
                    {
                        "pair": pname,
                        "task_a": t_a,
                        "task_b": t_b,
                        "n_molecules": len(cands),
                        "raw_attention_L1_mean": _safe_float(np.mean(attn_l1s)) if attn_l1s else None,
                        "ig_cosine_AB_mean": _safe_float(np.mean(ig_cos_list)) if ig_cos_list else None,
                        "ig_rel_diff_norm_mean": _safe_float(np.mean(ig_diff_norms)) if ig_diff_norms else None,
                    }
                )
            cross_species_report[tag] = cs_rows

            # 3) Top-k Atom Deletion Faithfulness Test (IG-top vs random atom masking in GINE h0)
            rng_del = np.random.default_rng(20261127)
            del_stats = {}
            for frac in (0.10, 0.20, 0.30):
                top_drops, rand_drops = [], []
                for row_i, ti in del_candidates:
                    pay = store.build_batch(
                        test_off[row_i : row_i + 1],
                        desc_array=test_desc[row_i : row_i + 1],
                        need_avalon=True,
                        need_morgan=True,
                        need_desc=True,
                        graph_mode="gine",
                        device=dev,
                    )
                    ig_abs = np.abs(_integrated_gradients_atoms(model, pay, target_fn=lambda o, e=ti: o[0, e], steps=12))
                    n_at = len(ig_abs)
                    k_mask = max(1, int(round(frac * n_at)))
                    top_idx = np.argsort(-ig_abs)[:k_mask]
                    with torch.no_grad():
                        h0, e_emb = model.graph_encoder.encode_atom_inputs(pay)
                        y_base = float(model(pay, atom_h0_override=h0, edge_emb_override=e_emb)[0, ti].item())
                        h_top = h0.clone()
                        h_top[top_idx] = 0.0
                        y_top = float(model(pay, atom_h0_override=h_top, edge_emb_override=e_emb)[0, ti].item())
                        r_diffs = []
                        for _ in range(3):
                            r_idx = rng_del.choice(n_at, size=k_mask, replace=False)
                            h_r = h0.clone()
                            h_r[r_idx] = 0.0
                            y_r = float(model(pay, atom_h0_override=h_r, edge_emb_override=e_emb)[0, ti].item())
                            r_diffs.append(abs(y_base - y_r))
                    top_drops.append(abs(y_base - y_top))
                    rand_drops.append(float(np.mean(r_diffs)))
                pval = None
                if len(top_drops) >= 6:
                    try:
                        pval = float(wilcoxon(top_drops, rand_drops, alternative="greater").pvalue)
                    except Exception:
                        pval = None
                del_stats[f"mask_{int(frac * 100)}pct"] = {
                    "n_molecules": len(top_drops),
                    "top_ig_abs_delta_z": _safe_float(np.mean(top_drops)) if top_drops else None,
                    "random_abs_delta_z": _safe_float(np.mean(rand_drops)) if rand_drops else None,
                    "ratio_top_vs_random": _safe_float(np.mean(top_drops) / max(1e-12, np.mean(rand_drops))) if top_drops else None,
                    "frac_top_gt_random": _safe_float(np.mean([a > b for a, b in zip(top_drops, rand_drops)])) if top_drops else None,
                    "wilcoxon_p_greater": _safe_float(pval),
                }
            deletion_report[tag] = del_stats

            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        # 4) S4 held-out cases faithful IG (Aconitum C8-ester hydrolysis pairs & cardiac glycosides)
        vdir_s4 = unified59 / "S4" / "FULL"
        ckpt_s4 = _resolve_checkpoint(f"S4_FULL_MV_BIOPGT_FINAL_s{seed}", runs_dir, search_dirs)
        if (vdir_s4 / "manifest.json").is_file() and ckpt_s4 is not None:
            m_s4 = load_manifest(vdir_s4)
            t_s4 = list(m_s4["task_names"])
            te_s4 = load_arrays(vdir_s4, "test.npz", m_s4)
            rec_map = {str(r): idx for idx, r in enumerate(te_s4["record_id"].tolist())}
            case_rows = [rec_map[r] for r in S4_CASE_RECORDS if r in rec_map]
            if case_rows:
                model_s4, blob_s4 = _load_ckpt_model(ckpt_s4, dev)
                offs_s4 = te_s4["cache_offset"].astype(np.int64)[case_rows]
                desc_s4 = transform_descriptors(store.desc_raw, store.desc_missing, offs_s4, blob_s4["desc_scaler"])
                key_eps = [e for e in ("mouse_intravenous_LD50", "mouse_oral_LD50", "rat_oral_LD50", "guinea pig_intravenous_LD50", "human_oral_TDLo") if e in t_s4]
                ig_mat = {}
                for ci, row_i in enumerate(case_rows):
                    rid_case = str(te_s4["record_id"][row_i])
                    pay = store.build_batch(
                        offs_s4[ci : ci + 1],
                        desc_array=desc_s4[ci : ci + 1],
                        need_avalon=True,
                        need_morgan=True,
                        need_desc=True,
                        graph_mode="gine",
                        device=dev,
                    )
                    arr_eps = []
                    for ep in key_eps:
                        ei = t_s4.index(ep)
                        arr_eps.append(_integrated_gradients_atoms(model_s4, pay, target_fn=lambda o, e=ei: o[0, e], steps=16))
                    ig_mat[f"ig_{rid_case}"] = np.stack(arr_eps, axis=0)
                np.savez_compressed(
                    interp_dir / "s4_case_faithful_ig.npz",
                    key_endpoints=np.asarray(key_eps),
                    **ig_mat,
                )
                del model_s4
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    write_json(interp_dir / "alert_ig_enrichment.json", alert_report)
    write_json(interp_dir / "cross_species_attribution.json", cross_species_report)
    write_json(interp_dir / "deletion_faithfulness.json", deletion_report)

    summary = {
        "status": "INTERPRET_B3_COMPLETE",
        "prompt_geometry": p1_report,
        "counterfactual_rows": cf_rows,
        "leo_rdrop_interventions": leo_rdrop_rows,
        "alert_ig_enrichment": alert_report,
        "cross_species_attribution": cross_species_report,
        "deletion_faithfulness": deletion_report,
    }
    write_json(interp_dir / "interpretability_b3_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Module 7: External 59 -> 36 Unseen PubChem Endpoints Double Zero-Shot
# ---------------------------------------------------------------------------

def _prompt_kernel_z_transfer(
    obs_pred: np.ndarray,
    obs_prompts: np.ndarray,
    ext_prompts: np.ndarray,
    obs_mean: np.ndarray,
    obs_scale: np.ndarray,
    ext_mean: np.ndarray,
    ext_scale: np.ndarray,
    *,
    temperature: float = 10.0,
) -> np.ndarray:
    """Transfer observed Z-scores to unseen external endpoints weighted by learned prompt cosine similarity."""
    z_obs = (obs_pred - obs_mean[None, :]) / obs_scale[None, :]
    p_o = obs_prompts / (np.linalg.norm(obs_prompts, axis=1, keepdims=True) + 1e-12)
    p_e = ext_prompts / (np.linalg.norm(ext_prompts, axis=1, keepdims=True) + 1e-12)
    sim = p_e @ p_o.T  # [T_ext, T_obs]
    logits = (sim - sim.max(axis=1, keepdims=True)) * float(temperature)
    w = np.exp(logits)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)  # [T_ext, T_obs]
    z_ext = z_obs @ w.T
    return (z_ext * ext_scale[None, :] + ext_mean[None, :]).astype(np.float32)


def _external_donor_z_transfer(
    obs_pred: np.ndarray,
    obs_tasks: list[str],
    ext_tasks: list[str],
    obs_mean: np.ndarray,
    obs_scale: np.ndarray,
    ext_mean: np.ndarray,
    ext_scale: np.ndarray,
) -> np.ndarray:
    meta_obs = task_metadata(tuple(obs_tasks))
    meta_ext = task_metadata(tuple(ext_tasks))
    z_obs = (obs_pred - obs_mean[None, :]) / obs_scale[None, :]
    out = np.zeros((obs_pred.shape[0], len(ext_tasks)), dtype=np.float32)
    for ti, t in enumerate(ext_tasks):
        mt = meta_ext[t]
        same_org = [j for j, ot in enumerate(obs_tasks) if meta_obs[ot]["organism"] == mt["organism"]]
        same_rt = [j for j, ot in enumerate(obs_tasks) if meta_obs[ot]["route"] == mt["route"]]
        donors = sorted(set(same_org) & set(same_rt))
        if not donors:
            donors = same_org if same_org else (same_rt if same_rt else list(range(len(obs_tasks))))
        z_est = np.mean(z_obs[:, donors], axis=1)
        out[:, ti] = (z_est * ext_scale[ti] + ext_mean[ti]).astype(np.float32)
    return out


def _collect_seen_noniso_smiles_from_graph_cache(graph_cache_59: Path, source_sha256: str) -> set[str]:
    gcache = GraphCache(graph_cache_59, source_sha256=source_sha256, verify_database=False)
    n_rec = int(gcache.manifest["record_count"])
    seen: set[str] = set()
    for off in range(n_rec):
        _, mol = gcache.read(off)
        seen.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    return seen


@torch.no_grad()
def run_external_zeroshot_36(
    *,
    graph_cache_59: Path,
    pub115_csv: Path,
    work_dir: Path,
    device: str = "cuda:0",
    prev_b1_dir: Path | None = None,
    prev_b2_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """Zero-retraining external double zero-shot evaluation: frozen 59-ep models -> 36 unseen PubChem endpoints."""
    runs_dir = work_dir / "runs"
    unified59 = work_dir / "views_all_59"
    pub_cache_path = work_dir / "pubchem115" / "mv_cache_115.pt"
    out_dir = work_dir / "external_zeroshot_36"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(
        device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    search_dirs = [prev_b1_dir, prev_b2_dir]

    vdir_s0 = unified59 / "S0" / "FULL"
    m_s0 = load_manifest(vdir_s0)
    tasks59 = list(m_s0["task_names"])
    seen_smiles = _collect_seen_noniso_smiles_from_graph_cache(graph_cache_59, m_s0["source_sha256"])

    subset = find_external_zeroshot_subset(
        pub115_cache_path=pub_cache_path,
        pub115_csv_path=pub115_csv,
        tasks_59=tasks59,
        seen_noniso_smiles=seen_smiles,
    )
    ext_tasks = list(subset["unseen_tasks"])
    ext_off = subset["cache_offsets"]
    ext_y = subset["labels"]
    if not ext_tasks or len(ext_off) == 0:
        raise RuntimeError("No external zero-shot endpoints/molecules found")

    pub_store = load_mv_cache(pub_cache_path)
    human_ext = tuple(t for t in ext_tasks if any(t.startswith(p) for p in ("human_", "man_", "women_")))
    animal_ext = tuple(t for t in ext_tasks if t not in set(human_ext))

    # Evaluate frozen S0..S4 MV_BIOPGT_FINAL checkpoints and S0 MV_BIOPGT_FINAL_RDROP
    preds_by_method: dict[str, list[np.ndarray]] = {}
    const_preds: list[np.ndarray] = []

    def _eval_ckpt(ckpt_path: Path, prefix: str):
        model, blob = _load_ckpt_model(ckpt_path, dev)
        obs_tasks = list(blob["task_names"])
        obs_counts = np.asarray(blob["counts"], dtype=np.float64)
        obs_mean = np.asarray(blob["target_mean"], dtype=np.float64)
        obs_scale = np.asarray(blob["target_scale"], dtype=np.float64)

        # Leak-free factor-prior (mean, scale) for the 36 unseen external endpoints strictly from the 59 observed tasks!
        comb_tasks = obs_tasks + ext_tasks
        comb_counts = np.concatenate([obs_counts, np.zeros(len(ext_tasks), dtype=np.float64)])
        comb_mean = np.concatenate([obs_mean, np.zeros(len(ext_tasks), dtype=np.float64)])
        comb_scale = np.concatenate([obs_scale, np.ones(len(ext_tasks), dtype=np.float64)])
        imp_mean, imp_scale = fit_factor_prior_stats(comb_tasks, comb_counts, comb_mean, comb_scale)
        ext_mean = imp_mean[len(obs_tasks) :]
        ext_scale = imp_scale[len(obs_tasks) :]

        ext_desc = transform_descriptors(
            pub_store.desc_raw, pub_store.desc_missing, ext_off, blob["desc_scaler"]
        )
        ext_prompts = model.prompt_bank.compose_external_prompts(ext_tasks)
        obs_prompts_np = model.prompt_bank().detach().cpu().numpy()
        ext_prompts_np = ext_prompts.detach().cpu().numpy()

        direct_chunks, obs_chunks = [], []
        for st in range(0, len(ext_off), 256):
            en = min(st + 256, len(ext_off))
            pay = pub_store.build_batch(
                ext_off[st:en],
                desc_array=ext_desc[st:en],
                need_avalon=True,
                need_morgan=True,
                need_desc=True,
                graph_mode="gine",
                device=dev,
            )
            z_ext = model(pay, prompt_override=ext_prompts, external_zero_shrink=True).detach().cpu().numpy()
            z_obs = model(pay).detach().cpu().numpy()
            direct_chunks.append((z_ext * ext_scale[None, :] + ext_mean[None, :]).astype(np.float32))
            obs_chunks.append((z_obs * obs_scale[None, :] + obs_mean[None, :]).astype(np.float32))

        pred_direct = np.concatenate(direct_chunks, axis=0)
        pred_obs = np.concatenate(obs_chunks, axis=0)
        pred_pkernel = _prompt_kernel_z_transfer(
            pred_obs, obs_prompts_np, ext_prompts_np, obs_mean, obs_scale, ext_mean, ext_scale, temperature=10.0
        )
        pred_donor = _external_donor_z_transfer(
            pred_obs, obs_tasks, ext_tasks, obs_mean, obs_scale, ext_mean, ext_scale
        )
        preds_by_method.setdefault(f"{prefix}_DIRECT", []).append(pred_direct)
        preds_by_method.setdefault(f"{prefix}_PROMPT_KERNEL_Z", []).append(pred_pkernel)
        preds_by_method.setdefault(f"{prefix}_DONOR_Z", []).append(pred_donor)
        if prefix == "MV_BIOPGT_FINAL":
            const_preds.append(np.tile(ext_mean.astype(np.float32)[None, :], (len(ext_off), 1)))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    for fold in SCAFFOLD_FOLDS:
        ck = _resolve_checkpoint(f"{fold}_FULL_MV_BIOPGT_FINAL_s{seed}", runs_dir, search_dirs)
        if ck is not None:
            _eval_ckpt(ck, "MV_BIOPGT_FINAL")

    ck_rdrop = _resolve_checkpoint(f"S0_FULL_MV_BIOPGT_FINAL_RDROP_s{seed}", runs_dir, search_dirs)
    if ck_rdrop is not None:
        _eval_ckpt(ck_rdrop, "MV_BIOPGT_FINAL_RDROP_S0")

    if const_preds:
        preds_by_method["FACTOR_PRIOR_CONST"] = const_preds

    summary_rows = []
    per_ep_rows = []
    saved_preds = {"cids": subset["cids"], "unseen_tasks": np.asarray(ext_tasks), "labels": ext_y.astype(np.float32)}
    for mname, plist in preds_by_method.items():
        ens_pred = np.mean(np.stack(plist, axis=0), axis=0).astype(np.float32)
        saved_preds[f"pred_{mname}"] = ens_pred
        all_m = _eval_subset_metrics(ext_y, ens_pred, ext_tasks, tuple(ext_tasks))
        ani_m = _eval_subset_metrics(ext_y, ens_pred, ext_tasks, animal_ext)
        hum_m = _eval_subset_metrics(ext_y, ens_pred, ext_tasks, human_ext)
        summary_rows.append(
            {
                "method": mname,
                "n_models_ensembled": len(plist),
                "ALL_UNSEEN_tasks": all_m["tasks"],
                "ALL_UNSEEN_rmse": all_m.get("macro_rmse"),
                "ALL_UNSEEN_r2": all_m.get("macro_r2"),
                "ALL_UNSEEN_pearson_r": all_m.get("macro_pearson_r"),
                "ALL_UNSEEN_spearman_rho": all_m.get("macro_spearman_rho"),
                "ANIMAL_UNSEEN_rmse": ani_m.get("macro_rmse"),
                "ANIMAL_UNSEEN_r2": ani_m.get("macro_r2"),
                "ANIMAL_UNSEEN_pearson_r": ani_m.get("macro_pearson_r"),
                "HUMAN_UNSEEN_tasks": hum_m["tasks"],
                "HUMAN_UNSEEN_rmse": hum_m.get("macro_rmse"),
                "HUMAN_UNSEEN_r2": hum_m.get("macro_r2"),
                "HUMAN_UNSEEN_pearson_r": hum_m.get("macro_pearson_r"),
            }
        )
        for ep_r in all_m["endpoints"]:
            per_ep_rows.append({"method": mname, **ep_r})

    np.savez_compressed(out_dir / "external_zeroshot_36_predictions.npz", **saved_preds)
    if summary_rows:
        with (out_dir / "external_zeroshot_36_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    if per_ep_rows:
        with (out_dir / "external_zeroshot_36_per_endpoint.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_ep_rows[0].keys()))
            w.writeheader()
            w.writerows(per_ep_rows)

    out = {
        "status": "EXT_ZEROSHOT_COMPLETE",
        "n_unseen_endpoints": len(ext_tasks),
        "n_unseen_human_endpoints": len(human_ext),
        "n_total_with_label": subset["n_total_with_label"],
        "n_overlap_removed": subset["n_overlap_removed"],
        "n_novel_molecules": subset["n_novel_molecules"],
        "n_novel_measurements": subset["n_novel_measurements"],
        "summary": summary_rows,
    }
    write_json(out_dir / "external_zeroshot_36_summary.json", out)
    return out


# ---------------------------------------------------------------------------
# Summarize & Collect
# ---------------------------------------------------------------------------

def _agg_runs_table(runs_dir: Path, folds: tuple[str, ...], btag: str, methods: tuple[str, ...], seed: int, dataset: str = "59") -> list[dict]:
    rows = []
    for method in methods:
        vals_rmse, vals_r2, vals_s, vals_h, vals_ld50, vals_av = [], [], [], [], [], []
        for fold in folds:
            rdir = runs_dir / f"{fold}_{btag}_{method}_s{seed}"
            if not (rdir / "summary.json").is_file():
                continue
            s = read_json(rdir / "summary.json")
            if s.get("status") != "COMPLETE":
                continue
            tg = s["test_groups"]
            if dataset == "115":
                vals_rmse.append(tg["ALL115"]["macro_rmse"])
                vals_r2.append(tg["ALL115"]["macro_r2"])
                vals_s.append(tg["Small68"]["macro_rmse"])
                vals_h.append(tg["Human11"]["macro_rmse"])
            else:
                vals_rmse.append(tg["ALL59"]["macro_rmse"])
                vals_r2.append(tg["ALL59"]["macro_r2"])
                vals_s.append(tg["S21"]["macro_rmse"])
                vals_h.append(tg["Human3"]["macro_rmse"])
                tm = read_json(rdir / "test_metrics.json")
                sub = _subdomain_metrics(tm)
                vals_ld50.append(sub["LD50_37"])
                vals_av.append(sub["Avian_5"])
        if not vals_rmse:
            continue
        entry = {
            "dataset": dataset,
            "split_group": f"{folds[0]}..{folds[-1]}",
            "budget": btag,
            "method": method,
            "n_folds": len(vals_rmse),
            "ALL_rmse_mean": float(np.mean(vals_rmse)),
            "ALL_rmse_sd": float(np.std(vals_rmse, ddof=1)) if len(vals_rmse) > 1 else 0.0,
            "ALL_r2_mean": float(np.mean(vals_r2)),
            "ALL_r2_sd": float(np.std(vals_r2, ddof=1)) if len(vals_r2) > 1 else 0.0,
            "Small_rmse_mean": float(np.mean(vals_s)),
            "Human_rmse_mean": float(np.mean(vals_h)),
        }
        if dataset == "59":
            entry["LD50_37_rmse_mean"] = float(np.mean(vals_ld50)) if vals_ld50 else None
            entry["Avian_5_rmse_mean"] = float(np.mean(vals_av)) if vals_av else None
        rows.append(entry)
    return rows


def summarize_formal_b3(*, work_dir: Path, groups59_path: Path, seed: int = 42) -> dict:
    runs_dir = work_dir / "runs"
    unified59 = work_dir / "views_all_59"

    # 1. 59-endpoint S0-S4 FULL (all baselines + ablations)
    s_full_rows = _agg_runs_table(
        runs_dir,
        SCAFFOLD_FOLDS,
        "FULL",
        (
            "MV_BIOPGT_FINAL",
            "TOX_AUTHOR",
            "DMPNN",
            "ATTENTIVEFP",
            "XGBOOST",
            "LIGHTGBM",
            "MV_BIOPGT_FLAT_PROMPT",
            "MV_BIOPGT_FINAL_NOPROMPT",
            "MV_BIOPGT_FINAL_NOMDROP",
            "MV_BIOPGT_FINAL_NOGRAPH",
            "MV_MLP",
            "GINE_ONLY",
            "RF",
        ),
        seed,
        dataset="59",
    )

    # 2. 59-endpoint S21 full 7-point downsampling curve
    s21_curve_rows = []
    for btag in ("FULL", "P80", "P50", "P40", "P30", "P20", "P10"):
        s21_curve_rows.extend(
            _agg_runs_table(
                runs_dir,
                SCAFFOLD_FOLDS,
                btag,
                ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "MV_BIOPGT_FINAL_NOPROMPT", "MV_MLP", "RF"),
                seed,
                dataset="59",
            )
        )

    # 3. 59-endpoint R0-R4 Random 5-fold CV
    r_full_rows = _agg_runs_table(
        runs_dir, RANDOM_FOLDS, "FULL", ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "RF"), seed, dataset="59"
    )

    # 4. S0 Prompt-learning diagnostics (FULL & LEO6: FINAL vs RDROP vs FLAT_PROMPT vs NOPROMPT)
    leo_rows = []
    for method in (
        "MV_BIOPGT_FINAL",
        "MV_BIOPGT_FINAL_RDROP",
        "MV_BIOPGT_FLAT_PROMPT",
        "MV_BIOPGT_FINAL_NOPROMPT",
        "MV_BIOPGT_FINAL_NOGRAPH",
        "TOX_AUTHOR",
    ):
        rmses6, r2s6, rs6, rmses5, obs_rmses = [], [], [], [], []
        for fold in SCAFFOLD_FOLDS:
            vdir = unified59 / fold / "LEO6"
            rdir = runs_dir / f"{fold}_LEO6_{method}_s{seed}"
            if not (vdir / "manifest.json").is_file() or not (rdir / "test_predictions.npz").is_file():
                continue
            m = load_manifest(vdir)
            tasks = list(m["task_names"])
            te = load_arrays(vdir, "test.npz", m)
            pr = np.load(rdir / "test_predictions.npz")["prediction"]
            m6 = _eval_subset_metrics(te["labels"], pr, tasks, LEO6_TASKS)
            m5 = _eval_subset_metrics(te["labels"], pr, tasks, LEO5_ANIMAL_TASKS)
            obs_t = tuple(t for t in tasks if t not in set(LEO6_TASKS))
            mo = _eval_subset_metrics(te["labels"], pr, tasks, obs_t)
            rmses6.append(m6["macro_rmse"])
            r2s6.append(m6["macro_r2"])
            rs6.append(m6["macro_pearson_r"])
            rmses5.append(m5["macro_rmse"])
            obs_rmses.append(mo["macro_rmse"])
        if rmses6:
            leo_rows.append(
                {
                    "method": method,
                    "n_folds": len(rmses6),
                    "LEO6_rmse_mean": float(np.mean(rmses6)),
                    "LEO6_rmse_sd": float(np.std(rmses6, ddof=1)) if len(rmses6) > 1 else 0.0,
                    "LEO6_r2_mean": float(np.mean(r2s6)),
                    "LEO6_pearson_r_mean": float(np.mean(rs6)),
                    "LEO5_Animal_rmse_mean": float(np.mean(rmses5)),
                    "OBS53_rmse_mean": float(np.mean(obs_rmses)),
                }
            )

    s0_prompt_cmp = []
    for method in ("MV_BIOPGT_FINAL", "MV_BIOPGT_FINAL_RDROP", "MV_BIOPGT_FLAT_PROMPT", "MV_BIOPGT_FINAL_NOPROMPT"):
        rdir = runs_dir / f"S0_FULL_{method}_s{seed}"
        if (rdir / "summary.json").is_file():
            s = read_json(rdir / "summary.json")
            tm = read_json(rdir / "test_metrics.json")
            sub = _subdomain_metrics(tm)
            s0_prompt_cmp.append(
                {
                    "fold": "S0",
                    "method": method,
                    "ALL59_rmse": s["test_groups"]["ALL59"]["macro_rmse"],
                    "ALL59_r2": s["test_groups"]["ALL59"]["macro_r2"],
                    "S21_rmse": s["test_groups"]["S21"]["macro_rmse"],
                    "Human3_rmse": s["test_groups"]["Human3"]["macro_rmse"],
                    "LD50_37_rmse": sub["LD50_37"],
                    "Avian_5_rmse": sub["Avian_5"],
                }
            )

    # 5. PubChem 115-endpoint Random 5-fold & Scaffold 5-fold
    pub115_rows = _agg_runs_table(
        runs_dir,
        PUB115_RANDOM_FOLDS,
        "FULL",
        ("MV_BIOPGT_FINAL", "MV_BIOPGT_FINAL_NOPROMPT", "TOX_AUTHOR", "RF"),
        seed,
        dataset="115",
    ) + _agg_runs_table(
        runs_dir,
        PUB115_SCAFFOLD_FOLDS,
        "FULL",
        ("MV_BIOPGT_FINAL", "MV_BIOPGT_FINAL_NOPROMPT", "TOX_AUTHOR", "RF"),
        seed,
        dataset="115",
    )

    for fname, rows in (
        ("formal_b3_scaffold59_baselines.csv", s_full_rows),
        ("formal_b3_s21_downsampling_curve.csv", s21_curve_rows),
        ("formal_b3_random59_5fold.csv", r_full_rows),
        ("formal_b3_leo6_rdrop_summary.csv", leo_rows),
        ("formal_b3_s0_prompt_ablation.csv", s0_prompt_cmp),
        ("formal_b3_pubchem115_summary.csv", pub115_rows),
    ):
        if rows:
            with (work_dir / fname).open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    ext_zs = None
    ext_zs_path = work_dir / "external_zeroshot_36" / "external_zeroshot_36_summary.json"
    if ext_zs_path.is_file():
        ext_zs = read_json(ext_zs_path)

    interp_sum = None
    interp_path = work_dir / "interpretability_b3" / "interpretability_b3_summary.json"
    if interp_path.is_file():
        interp_sum = read_json(interp_path)

    # Render concise Markdown report (all sealed TEST set metrics!)
    lines = [
        "# Formal Batch 3 Summary Report (Sealed Test Metrics)",
        "",
        f"- Code commit: `{git_identity()}`",
        "",
        "## 1. 59-Endpoint 5-Fold Scaffold Benchmark (`S0–S4 / FULL`, External Baselines & Ablations)",
        "| Method | Folds | ALL59 RMSE | ALL59 R2 | S21 RMSE | Human3 RMSE | LD50(37) RMSE | Avian(5) RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s_full_rows:
        lines.append(
            f"| {r['method']} | {r['n_folds']} | {r['ALL_rmse_mean']:.4f} ± {r['ALL_rmse_sd']:.4f} | "
            f"{r['ALL_r2_mean']:.4f} ± {r['ALL_r2_sd']:.4f} | {r['Small_rmse_mean']:.4f} | "
            f"{r['Human_rmse_mean']:.4f} | {r['LD50_37_rmse_mean']:.4f} | {r['Avian_5_rmse_mean']:.4f} |"
        )

    lines += [
        "",
        "## 2. 59-Endpoint 5-Fold Random Split (`R0–R4 / FULL`)",
        "| Method | Folds | ALL59 RMSE | ALL59 R2 | S21 RMSE | Human3 RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in r_full_rows:
        lines.append(
            f"| {r['method']} | {r['n_folds']} | {r['ALL_rmse_mean']:.4f} ± {r['ALL_rmse_sd']:.4f} | "
            f"{r['ALL_r2_mean']:.4f} ± {r['ALL_r2_sd']:.4f} | {r['Small_rmse_mean']:.4f} | {r['Human_rmse_mean']:.4f} |"
        )

    lines += [
        "",
        "## 3. 21 Small-Endpoint Downsampling Curve (`S0–S4`, 100% -> 10%)",
        "| Budget | Method | Folds | ALL59 RMSE | S21 RMSE | Human3 RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in s21_curve_rows:
        lines.append(
            f"| {r['budget']} | {r['method']} | {r['n_folds']} | {r['ALL_rmse_mean']:.4f} | "
            f"{r['Small_rmse_mean']:.4f} | {r['Human_rmse_mean']:.4f} |"
        )

    lines += [
        "",
        "## 4. Prompt-Learning Diagnostics (`S0/FULL` & `S0–S4/LEO6`)",
        "| Method | Folds | LEO6 RMSE | LEO6 R2 | LEO6 Pearson r | LEO5 Animal RMSE | OBS53 RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in leo_rows:
        lines.append(
            f"| {r['method']} | {r['n_folds']} | {r['LEO6_rmse_mean']:.4f} ± {r['LEO6_rmse_sd']:.4f} | "
            f"{r['LEO6_r2_mean']:.4f} | {r['LEO6_pearson_r_mean']:.4f} | "
            f"{r['LEO5_Animal_rmse_mean']:.4f} | {r['OBS53_rmse_mean']:.4f} |"
        )

    lines += [
        "",
        "## 5. External PubChem 115-Endpoint Benchmark (Random 5-Fold & Scaffold 5-Fold)",
        "| Split | Method | Folds | ALL115 RMSE | ALL115 R2 | Small68 RMSE | Human11 RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in pub115_rows:
        lines.append(
            f"| {r['split_group']} | {r['method']} | {r['n_folds']} | "
            f"{r['ALL_rmse_mean']:.4f} ± {r['ALL_rmse_sd']:.4f} | {r['ALL_r2_mean']:.4f} ± {r['ALL_r2_sd']:.4f} | "
            f"{r['Small_rmse_mean']:.4f} | {r['Human_rmse_mean']:.4f} |"
        )

    if ext_zs is not None:
        lines += [
            "",
            f"## 6. External 59 -> {ext_zs['n_unseen_endpoints']} Unseen PubChem Endpoints Double Zero-Shot "
            f"({ext_zs['n_novel_molecules']} strictly novel molecules, {ext_zs['n_novel_measurements']} measurements)",
            "| Method | Unseen All RMSE | Unseen All R2 | Unseen All r | Animal RMSE | Human(6) RMSE | Human(6) r |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in ext_zs.get("summary", []):
            def _f(x):
                return "NA" if x is None else f"{x:.4f}"
            lines.append(
                f"| {r['method']} | {_f(r['ALL_UNSEEN_rmse'])} | {_f(r['ALL_UNSEEN_r2'])} | "
                f"{_f(r['ALL_UNSEEN_pearson_r'])} | {_f(r['ANIMAL_UNSEEN_rmse'])} | "
                f"{_f(r['HUMAN_UNSEEN_rmse'])} | {_f(r['HUMAN_UNSEEN_pearson_r'])} |"
            )

    (work_dir / "formal_b3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "code_commit": git_identity(),
        "scaffold59_baselines": s_full_rows,
        "s21_downsampling_curve": s21_curve_rows,
        "random59_5fold": r_full_rows,
        "leo6_rdrop_summary": leo_rows,
        "s0_prompt_ablation": s0_prompt_cmp,
        "pubchem115_summary": pub115_rows,
        "external_zeroshot_36": ext_zs,
        "has_interpretability_b3": interp_sum is not None,
    }
    write_json(work_dir / "formal_b3_summary.json", summary)
    return summary


def collect_formal_b3(work_dir: Path) -> Path:
    """Package canonical Formal Batch 3 results ZIP excluding caches, views, and .pt weights."""
    work_dir = Path(work_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = work_dir / f"PGTV2_FORMAL_B3_RESULTS_{stamp}.zip"
    skip_dirs = {"mv_cache", "leo_views", "views_all_59", "pubchem115", "release", "_configs"}

    members = []
    for p in sorted(work_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(work_dir)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if p.suffix in (".pt", ".zip") or p.name.endswith(".zip.sha256"):
            continue
        members.append((p, rel.as_posix()))

    # Include lightweight pubchem115 summary JSONs (not the 150MB .pt cache or raw views)
    for extra in (
        work_dir / "pubchem115" / "endpoint_groups_115.json",
        work_dir / "pubchem115" / "mv_cache_115.meta.json",
        work_dir / "pubchem115" / "views" / "pubchem115_views_summary.json",
    ):
        if extra.is_file():
            members.append((extra, extra.relative_to(work_dir).as_posix()))

    chk_lines = [f"{file_sha256(p)}  {arc}" for p, arc in members]
    chk_path = work_dir / "checksums.sha256"
    chk_path.write_text("\n".join(chk_lines) + "\n", encoding="utf-8")

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p, arc in members:
            zf.write(p, arcname=arc)
        zf.write(chk_path, arcname="checksums.sha256")

    sha = file_sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
    print(f"[b3] Packaged {archive} ({archive.stat().st_size / 1e6:.2f} MB) sha256={sha}", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--mv-cache", required=True)
    p_pre.add_argument("--view", required=True)
    p_pre.add_argument("--out", required=True)
    p_pre.add_argument("--device", default="cuda:0")

    p_prep = sub.add_parser("prep-views")
    p_prep.add_argument("--views-root", required=True)
    p_prep.add_argument("--groups", required=True)
    p_prep.add_argument("--pub115-csv", default=None)
    p_prep.add_argument("--pub115-random-dir", default=None)
    p_prep.add_argument("--prev-b2-dir", default=None)
    p_prep.add_argument("--work", required=True)
    p_prep.add_argument("--workers", type=int, default=None)

    p_run = sub.add_parser("run-b3")
    p_run.add_argument("--mv-cache", required=True)
    p_run.add_argument("--groups", required=True)
    p_run.add_argument("--work", required=True)
    p_run.add_argument("--gpus", type=int, nargs="+", required=True)
    p_run.add_argument("--cpu-workers", type=int, default=2)
    p_run.add_argument("--prev-b1-dir", default=None)
    p_run.add_argument("--prev-b2-dir", default=None)
    p_run.add_argument("--seed", type=int, default=42)

    p_int = sub.add_parser("interpret")
    p_int.add_argument("--mv-cache", required=True)
    p_int.add_argument("--graph-cache", default=None)
    p_int.add_argument("--groups", required=True)
    p_int.add_argument("--work", required=True)
    p_int.add_argument("--device", default="cuda:0")
    p_int.add_argument("--prev-b1-dir", default=None)
    p_int.add_argument("--prev-b2-dir", default=None)
    p_int.add_argument("--seed", type=int, default=42)

    p_zs = sub.add_parser("ext-zeroshot")
    p_zs.add_argument("--graph-cache", required=True)
    p_zs.add_argument("--pub115-csv", required=True)
    p_zs.add_argument("--work", required=True)
    p_zs.add_argument("--device", default="cuda:0")
    p_zs.add_argument("--prev-b1-dir", default=None)
    p_zs.add_argument("--prev-b2-dir", default=None)
    p_zs.add_argument("--seed", type=int, default=42)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--work", required=True)
    p_sum.add_argument("--groups", required=True)
    p_sum.add_argument("--seed", type=int, default=42)

    p_col = sub.add_parser("collect")
    p_col.add_argument("--work", required=True)

    args = parser.parse_args()
    if args.cmd == "preflight":
        run_preflight_b3(Path(args.mv_cache), Path(args.view), Path(args.out), args.device)
    elif args.cmd == "prep-views":
        prepare_b3_assets(
            views59_root=Path(args.views_root),
            groups59_path=Path(args.groups),
            pub115_csv=Path(args.pub115_csv) if args.pub115_csv else None,
            pub115_random_dir=Path(args.pub115_random_dir) if args.pub115_random_dir else None,
            work_dir=Path(args.work),
            prev_b2_dir=Path(args.prev_b2_dir) if args.prev_b2_dir else None,
            max_workers=args.workers,
        )
    elif args.cmd == "run-b3":
        work_dir = Path(args.work)
        unified59 = work_dir / "views_all_59"
        runs_dir = work_dir / "runs"
        logs_dir = work_dir / "logs"
        reused = _import_reused_runs(
            unified59=unified59,
            runs_dir=runs_dir,
            prev_b1_dir=Path(args.prev_b1_dir) if args.prev_b1_dir else None,
            prev_b2_dir=Path(args.prev_b2_dir) if args.prev_b2_dir else None,
            seed=args.seed,
        )
        print(f"[b3] Reused {len(reused)} verified prior runs", flush=True)
        pub_cache = work_dir / "pubchem115" / "mv_cache_115.pt"
        pub_views = work_dir / "pubchem115" / "views"
        pub_groups = work_dir / "pubchem115" / "endpoint_groups_115.json"
        has_pub = pub_cache.is_file() and pub_groups.is_file()
        jobs = build_b3_jobs(include_pub115=has_pub, seed=args.seed)
        _dispatch_b3_jobs(
            jobs,
            mv_cache_59=Path(args.mv_cache),
            views59_root=unified59,
            groups59_path=Path(args.groups),
            mv_cache_115=pub_cache if has_pub else None,
            views115_root=pub_views if has_pub else None,
            groups115_path=pub_groups if has_pub else None,
            runs_dir=runs_dir,
            logs_dir=logs_dir,
            gpus=args.gpus,
            cpu_workers=args.cpu_workers,
        )
    elif args.cmd == "interpret":
        run_prompt_interpretability_suite(
            mv_cache_59=Path(args.mv_cache),
            graph_cache_59=Path(args.graph_cache) if args.graph_cache else None,
            work_dir=Path(args.work),
            groups59_path=Path(args.groups),
            device=args.device,
            prev_b1_dir=Path(args.prev_b1_dir) if args.prev_b1_dir else None,
            prev_b2_dir=Path(args.prev_b2_dir) if args.prev_b2_dir else None,
            seed=args.seed,
        )
    elif args.cmd == "ext-zeroshot":
        run_external_zeroshot_36(
            graph_cache_59=Path(args.graph_cache),
            pub115_csv=Path(args.pub115_csv),
            work_dir=Path(args.work),
            device=args.device,
            prev_b1_dir=Path(args.prev_b1_dir) if args.prev_b1_dir else None,
            prev_b2_dir=Path(args.prev_b2_dir) if args.prev_b2_dir else None,
            seed=args.seed,
        )
    elif args.cmd == "summarize":
        summarize_formal_b3(
            work_dir=Path(args.work),
            groups59_path=Path(args.groups),
            seed=args.seed,
        )
    elif args.cmd == "collect":
        collect_formal_b3(Path(args.work))


if __name__ == "__main__":
    main()
