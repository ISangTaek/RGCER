"""Formal Batch 2 (5-Fold x 3-Seed Benchmark, Zero-Shot Leave-Endpoint-Out & R0 Split Control).

Usage:
    python -m pgt_v2.mv_prompt.formal_b2 preflight --mv-cache <mv_cache.pt> --views-root <views> --work <work_dir> --device cuda:0
    python -m pgt_v2.mv_prompt.formal_b2 run-b2 --mv-cache <mv_cache.pt> --views-root <views> --groups <endpoint_groups.json> --work <work_dir> --gpus 0 1 2 3 [--prev-b1-dir <dir>]
    python -m pgt_v2.mv_prompt.formal_b2 summarize --work <work_dir> --views-root <views> --groups <endpoint_groups.json>
    python -m pgt_v2.mv_prompt.formal_b2 collect --work <work_dir>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

from baselines.models.toxacol import task_adjacency
from ..artifacts import load_arrays, load_manifest, read_json, seal_manifest, write_json
from ..cache import file_sha256
from ..config import git_identity
from ..models import task_metadata
from .features import MVDataStore, fit_descriptor_scaler, load_mv_cache, transform_descriptors
from .formal_b1 import _dispatch_formal_jobs, _predict_with_override, _try_import_s0_run
from .models import build_mv_model, count_parameters, method_input_needs
from .probe_v2 import _subdomain_metrics
from .training import (
    compute_group_metrics,
    endpoint_weights_for_gamma,
    fit_factor_prior_stats,
    seed_all,
    tempered_mse_loss,
)

SCAFFOLD_FOLDS = ("S0", "S1", "S2", "S3", "S4")
MULTISEED_LIST = (42, 43, 44)

LEO6_TASKS = (
    "rabbit_oral_LD50",
    "dog_intravenous_LD50",
    "rat_skin_LD50",
    "guinea pig_intraperitoneal_LD50",
    "mouse_subcutaneous_LDLo",
    "human_oral_TDLo",
)
LEO5_ANIMAL_TASKS = LEO6_TASKS[:5]


def build_leo_view(
    src_full_view: Path,
    dst_leo_view: Path,
    *,
    held_out_tasks: tuple[str, ...] = LEO6_TASKS,
    budget_id: str = "LEO6",
) -> dict:
    """Create a leak-free Leave-Endpoint-Out view masking held-out tasks in train & validation."""
    src_full_view = Path(src_full_view)
    dst_leo_view = Path(dst_leo_view)
    if (dst_leo_view / "manifest.json").is_file():
        return load_manifest(dst_leo_view)

    src_manifest = load_manifest(src_full_view)
    tasks = list(src_manifest["task_names"])
    missing = [t for t in held_out_tasks if t not in tasks]
    if missing:
        raise ValueError(f"Held-out tasks not found in view task_names: {missing}")
    held_idx = [tasks.index(t) for t in held_out_tasks]

    train_arr = load_arrays(src_full_view, "train.npz", src_manifest)
    val_arr = load_arrays(src_full_view, "validation.npz", src_manifest)
    test_arr = load_arrays(src_full_view, "test.npz", src_manifest)

    # Zero-leakage masking: neither training nor validation early-stopping sees held-out tasks!
    train_labels = train_arr["labels"].astype(np.float64).copy()
    train_labels[:, held_idx] = np.nan
    val_labels = val_arr["labels"].astype(np.float64).copy()
    val_labels[:, held_idx] = np.nan
    test_labels = test_arr["labels"].astype(np.float64).copy()

    # Recompute statistics STRICTLY from masked train_labels (never touch src statistics[held_idx])
    counts = np.isfinite(train_labels).sum(axis=0).astype(np.int64)
    if not np.all(counts[held_idx] == 0):
        raise AssertionError("Held-out tasks must have exact zero visible training count")
    pos = counts > 0
    if not np.any(pos):
        raise ValueError("LEO view must retain at least one observed endpoint")

    raw_mean = np.zeros(len(tasks), dtype=np.float64)
    raw_scale = np.ones(len(tasks), dtype=np.float64)
    scale_fallback = np.zeros(len(tasks), dtype=bool)
    group_count = np.zeros(len(tasks), dtype=np.int64)
    train_grps = np.asarray(train_arr["group_id"])

    for j in range(len(tasks)):
        mask_j = np.isfinite(train_labels[:, j])
        if not bool(mask_j.any()):
            scale_fallback[j] = True
            continue
        vals = train_labels[mask_j, j]
        raw_mean[j] = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        if sd > 1e-8:
            raw_scale[j] = sd
        else:
            raw_scale[j] = 1.0
            scale_fallback[j] = True
        group_count[j] = int(len(set(train_grps[mask_j].tolist())))

    imp_mean, imp_scale = fit_factor_prior_stats(tasks, counts, raw_mean, raw_scale)
    adj, _ = task_adjacency(train_labels)

    dst_leo_view.mkdir(parents=True, exist_ok=True)
    common_keys = ("record_id", "cache_offset", "raw_row_index", "group_id", "chemical_id")
    files = []
    for role, arr, y_mat in (
        ("train", train_arr, train_labels),
        ("validation", val_arr, val_labels),
        ("test", test_arr, test_labels),
    ):
        part = {k: arr[k] for k in common_keys}
        np.savez_compressed(dst_leo_view / f"{role}.npz", **part, labels=y_mat)
        files.append(f"{role}.npz")
        np.savez_compressed(dst_leo_view / f"{role}_inputs.npz", **part)
        files.append(f"{role}_inputs.npz")

    np.savez_compressed(
        dst_leo_view / "statistics.npz",
        mean=imp_mean.astype(np.float64),
        scale=imp_scale.astype(np.float64),
        count=counts,
        group_count=group_count,
        scale_fallback=scale_fallback,
        adjacency=adj,
    )
    files.append("statistics.npz")

    meta = {
        "format": src_manifest["format"],
        "view_id": f"{src_manifest['fold']}_{budget_id}",
        "fold": str(src_manifest["fold"]),
        "task_names": tasks,
        "held_out_tasks": list(held_out_tasks),
        "parent_view_sha256": src_manifest["manifest_sha256"],
        "source_sha256": src_manifest["source_sha256"],
        "feature_spec_sha256": src_manifest["feature_spec_sha256"],
        "cache_id": src_manifest["cache_id"],
        "active_train_rows": int(np.isfinite(train_labels).any(axis=1).sum()),
        "visible_labels": int(counts.sum()),
    }
    return seal_manifest(dst_leo_view, meta, files)


def build_all_leo_views(
    views_root: Path,
    leo_views_root: Path,
    *,
    folds: tuple[str, ...] = SCAFFOLD_FOLDS,
    held_out_tasks: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    if held_out_tasks is None:
        held_out_tasks = LEO6_TASKS
    out = {}
    for fold in folds:
        src = Path(views_root) / fold / "FULL"
        if not (src / "manifest.json").is_file():
            continue
        dst = Path(leo_views_root) / fold / "LEO6"
        build_leo_view(src, dst, held_out_tasks=held_out_tasks, budget_id="LEO6")
        out[fold] = dst
    return out


def run_preflight_b2(
    mv_cache_path: Path,
    views_root: Path,
    work_dir: Path,
    output_path: Path,
    device: str,
) -> dict:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Server preflight requires an available CUDA device")
    store = load_mv_cache(mv_cache_path)
    leo_root = work_dir / "leo_views"
    build_leo_view(views_root / "S0" / "FULL", leo_root / "S0" / "LEO6")

    manifest = load_manifest(leo_root / "S0" / "LEO6")
    train_arr = load_arrays(leo_root / "S0" / "LEO6", "train.npz", manifest)
    stats_arr = load_arrays(leo_root / "S0" / "LEO6", "statistics.npz", manifest)
    store.verify_partition(train_arr, manifest)

    tasks = list(manifest["task_names"])
    held_idx = [tasks.index(t) for t in LEO6_TASKS]
    if not np.all(stats_arr["count"][held_idx] == 0):
        raise RuntimeError("LEO6 preflight count check failed")

    train_off = train_arr["cache_offset"].astype(np.int64)
    active = np.where(np.isfinite(train_arr["labels"]).any(axis=1))[0][:16]
    scaler = fit_descriptor_scaler(store.desc_raw, store.desc_missing, train_off[active])
    desc = transform_descriptors(store.desc_raw, store.desc_missing, train_off[active], scaler)
    ep_w = torch.from_numpy(endpoint_weights_for_gamma(stats_arr["count"], 0.50)).to(device)

    checked = []
    for method in (
        "MV_BIOPGT_FINAL",
        "MV_BIOPGT_FINAL_NOPROMPT",
        "MV_BIOPGT_FINAL_NOGRAPH",
        "TOX_AUTHOR",
    ):
        seed_all(43)
        needs = method_input_needs(method)
        model = build_mv_model(
            method,
            tuple(tasks),
            stats_arr["count"],
            adjacency=stats_arr["adjacency"],
            device=device,
        )
        if hasattr(model, "prompt_bank") and model.prompt_bank is not None:
            sh = model.prompt_bank.shrinkage.detach().cpu().numpy()
            if not np.all(sh[held_idx] == 0.0):
                raise RuntimeError(f"Non-zero shrinkage on held-out LEO6 tasks for {method}")
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
        if method in ("MV_BIOPGT_FINAL", "MV_BIOPGT_FINAL_NOPROMPT"):
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
            raise RuntimeError(f"Nonfinite gradient in preflight_b2 for {method}")
        checked.append(
            {
                "method": method,
                "parameters": count_parameters(model),
                "leo_zero_shrinkage_verified": True,
                "finite_forward_and_grad": True,
            }
        )
        del model
        torch.cuda.empty_cache()

    report = {
        "status": "PREFLIGHT_B2_PASS",
        "code_commit": git_identity(),
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "torch": str(torch.__version__),
        "r0_view_present": (views_root / "R0" / "FULL" / "manifest.json").is_file(),
        "models_checked": checked,
    }
    write_json(output_path, report)
    return report


def build_b2_jobs(
    *,
    folds: tuple[str, ...] = SCAFFOLD_FOLDS,
    extra_seeds: tuple[int, ...] = (43, 44),
    leo_seed: int = 42,
    include_r0: bool = True,
) -> list[dict]:
    """Build Formal Batch 2 job queue ordered longest-first across 4 GPUs + CPU."""
    jobs = []
    # Priority 1: Slowest reference baseline TOX_AUTHOR (~37 min each)
    for seed in extra_seeds:
        for fold in folds:
            jobs.append(
                {
                    "run_id": f"{fold}_FULL_TOX_AUTHOR_s{seed}",
                    "fold": fold,
                    "budget": "FULL",
                    "view_source": "main",
                    "method": "TOX_AUTHOR",
                    "seed": seed,
                    "config": {},
                }
            )
    for fold in folds:
        jobs.append(
            {
                "run_id": f"{fold}_LEO6_TOX_AUTHOR_s{leo_seed}",
                "fold": fold,
                "budget": "LEO6",
                "view_source": "leo",
                "method": "TOX_AUTHOR",
                "seed": leo_seed,
                "config": {},
            }
        )
    if include_r0:
        jobs.append(
            {
                "run_id": f"R0_FULL_TOX_AUTHOR_s{leo_seed}",
                "fold": "R0",
                "budget": "FULL",
                "view_source": "main",
                "method": "TOX_AUTHOR",
                "seed": leo_seed,
                "config": {},
            }
        )

    # Priority 2: Locked primary single model MV_BIOPGT_FINAL (~20 min each)
    for seed in extra_seeds:
        for fold in folds:
            jobs.append(
                {
                    "run_id": f"{fold}_FULL_MV_BIOPGT_FINAL_s{seed}",
                    "fold": fold,
                    "budget": "FULL",
                    "view_source": "main",
                    "method": "MV_BIOPGT_FINAL",
                    "seed": seed,
                    "config": {"gamma": 0.50},
                }
            )
    for fold in folds:
        jobs.append(
            {
                "run_id": f"{fold}_LEO6_MV_BIOPGT_FINAL_s{leo_seed}",
                "fold": fold,
                "budget": "LEO6",
                "view_source": "leo",
                "method": "MV_BIOPGT_FINAL",
                "seed": leo_seed,
                "config": {"gamma": 0.50},
            }
        )
    if include_r0:
        jobs.append(
            {
                "run_id": f"R0_FULL_MV_BIOPGT_FINAL_s{leo_seed}",
                "fold": "R0",
                "budget": "FULL",
                "view_source": "main",
                "method": "MV_BIOPGT_FINAL",
                "seed": leo_seed,
                "config": {"gamma": 0.50},
            }
        )

    # Priority 3: LEO6 unprompted & no-graph ablations + R0 CPU RF
    for fold in folds:
        jobs.append(
            {
                "run_id": f"{fold}_LEO6_MV_BIOPGT_FINAL_NOPROMPT_s{leo_seed}",
                "fold": fold,
                "budget": "LEO6",
                "view_source": "leo",
                "method": "MV_BIOPGT_FINAL_NOPROMPT",
                "seed": leo_seed,
                "config": {"gamma": 0.50},
            }
        )
    for fold in folds:
        jobs.append(
            {
                "run_id": f"{fold}_LEO6_MV_BIOPGT_FINAL_NOGRAPH_s{leo_seed}",
                "fold": fold,
                "budget": "LEO6",
                "view_source": "leo",
                "method": "MV_BIOPGT_FINAL_NOGRAPH",
                "seed": leo_seed,
                "config": {"gamma": 0.50},
            }
        )
    if include_r0:
        jobs.append(
            {
                "run_id": f"R0_FULL_RF_s{leo_seed}",
                "fold": "R0",
                "budget": "FULL",
                "view_source": "main",
                "method": "RF",
                "seed": leo_seed,
                "config": {"n_estimators": 300, "n_jobs": 8},
            }
        )
    return jobs


def prepare_unified_views_dir(
    views_root: Path,
    work_dir: Path,
    folds: tuple[str, ...],
    *,
    held_out_tasks: tuple[str, ...] | None = None,
) -> Path:
    """Create unified view symlinks/copies under work_dir/views_all/{fold}/{FULL|LEO6}."""
    unified = work_dir / "views_all"
    leo_root = work_dir / "leo_views"
    build_all_leo_views(views_root, leo_root, folds=folds, held_out_tasks=held_out_tasks)

    for fold in list(folds) + ["R0"]:
        src_full = views_root / fold / "FULL"
        if (src_full / "manifest.json").is_file():
            dst_full = unified / fold / "FULL"
            if not (dst_full / "manifest.json").is_file():
                dst_full.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst_full.symlink_to(src_full.resolve(), target_is_directory=True)
                except OSError:
                    shutil.copytree(src_full, dst_full, dirs_exist_ok=True)
        src_leo = leo_root / fold / "LEO6"
        if (src_leo / "manifest.json").is_file():
            dst_leo = unified / fold / "LEO6"
            if not (dst_leo / "manifest.json").is_file():
                dst_leo.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst_leo.symlink_to(src_leo.resolve(), target_is_directory=True)
                except OSError:
                    shutil.copytree(src_leo, dst_leo, dirs_exist_ok=True)
    return unified


def _eval_subset_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    tasks: list[str],
    subset_tasks: tuple[str, ...],
) -> dict:
    per_ep = []
    for t in subset_tasks:
        if t not in tasks:
            continue
        j = tasks.index(t)
        mask = np.isfinite(truth[:, j])
        n = int(mask.sum())
        if n < 2:
            continue
        yt = truth[mask, j].astype(np.float64)
        yp = pred[mask, j].astype(np.float64)
        err = yp - yt
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        denom = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 0 else None
        if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
            pr = float(pearsonr(yt, yp)[0])
            sr = float(spearmanr(yt, yp).statistic)
        else:
            pr = 0.0
            sr = 0.0
        per_ep.append(
            {
                "task": t,
                "n": n,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pearson_r": pr,
                "spearman_rho": sr,
            }
        )
    if not per_ep:
        return {"tasks": 0, "endpoints": []}
    r2_list = [x["r2"] for x in per_ep if x["r2"] is not None]
    return {
        "tasks": len(per_ep),
        "macro_rmse": float(np.mean([x["rmse"] for x in per_ep])),
        "macro_mae": float(np.mean([x["mae"] for x in per_ep])),
        "macro_r2": float(np.mean(r2_list)) if r2_list else None,
        "positive_r2_count": int(sum(v > 0 for v in r2_list)),
        "macro_pearson_r": float(np.mean([x["pearson_r"] for x in per_ep])),
        "macro_spearman_rho": float(np.mean([x["spearman_rho"] for x in per_ep])),
        "endpoints": per_ep,
    }


@torch.no_grad()
def export_leo_interventions(
    *,
    mv_cache_path: Path,
    unified_views_root: Path,
    work_dir: Path,
    device: str = "cuda:0",
    seed: int = 42,
    folds: tuple[str, ...] = SCAFFOLD_FOLDS,
    held_out_tasks: tuple[str, ...] = LEO6_TASKS,
) -> dict:
    """Zero-training prompt factor interventions on unseen LEO6 endpoints (shrinkage == 0)."""
    store = load_mv_cache(mv_cache_path)
    runs_dir = work_dir / "runs"
    mech_dir = work_dir / "mechanistic_b2"
    mech_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(
        device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )

    rows = []
    for fold in folds:
        vdir = unified_views_root / fold / "LEO6"
        ckpt_path = runs_dir / f"{fold}_LEO6_MV_BIOPGT_FINAL_s{seed}" / "best.pt"
        if not (vdir / "manifest.json").is_file() or not ckpt_path.is_file():
            continue
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_mv_model(
            "MV_BIOPGT_FINAL",
            tuple(blob["task_names"]),
            blob["counts"],
            adjacency=blob["adjacency"],
            config=blob.get("config") or {},
            device=dev,
        )
        model.load_state_dict(blob["model_state"])
        model.eval()

        manifest = load_manifest(vdir)
        tasks = list(manifest["task_names"])
        test_arr = load_arrays(vdir, "test.npz", manifest)
        test_off = test_arr["cache_offset"].astype(np.int64)
        test_desc = transform_descriptors(
            store.desc_raw, store.desc_missing, test_off, blob["desc_scaler"]
        )
        mean = np.asarray(blob["target_mean"], dtype=np.float64)
        scale = np.asarray(blob["target_scale"], dtype=np.float64)

        prompts = model.prompt_bank().detach()
        np.savez_compressed(
            mech_dir / f"leo_prompts_{fold}.npz",
            task_names=np.asarray(tasks),
            prompts=prompts.cpu().numpy(),
            shrinkage=model.prompt_bank.shrinkage.detach().cpu().numpy(),
        )

        meta = task_metadata(tuple(tasks))
        route_vocab = sorted({meta[t]["route"] for t in tasks})
        org_vocab = sorted({meta[t]["organism"] for t in tasks})
        orig_route = model.prompt_bank.route_idx.clone()
        swap_route = orig_route.clone()
        if "oral" in route_vocab and "intravenous" in route_vocab:
            i_oral = route_vocab.index("oral")
            i_iv = route_vocab.index("intravenous")
            swap_route[orig_route == i_oral] = i_iv
            swap_route[orig_route == i_iv] = i_oral
        orig_org = model.prompt_bank.organism_idx.clone()
        shift_org = (orig_org + 1) % max(1, len(org_vocab))

        gen = torch.Generator(device="cpu").manual_seed(20261127)
        perm_idx = torch.randperm(len(tasks), generator=gen).to(prompts.device)
        perm_prompts = prompts[perm_idx]

        # Measure z-score shift and correlation shift on pure zero-shot LEO6 heads (shrinkage=0)!
        pred_intact = _predict_with_override(
            model, store, test_off, test_desc, mean, scale, device=dev
        )
        active_held = [t for t in held_out_tasks if t in tasks]
        held_idx = [tasks.index(t) for t in active_held]

        for cname, p_ov, f_ov in (
            ("intact", None, None),
            ("perm_all", perm_prompts, None),
            ("swap_oral_iv", None, {"route": swap_route}),
            ("shift_species", None, {"organism": shift_org}),
        ):
            pred_phys = (
                pred_intact
                if cname == "intact"
                else _predict_with_override(
                    model,
                    store,
                    test_off,
                    test_desc,
                    mean,
                    scale,
                    device=dev,
                    prompt_override=p_ov,
                    factor_override=f_ov,
                )
            )
            z_mad_leo = (
                float(
                    np.mean(
                        np.abs(
                            (pred_phys[:, held_idx] - pred_intact[:, held_idx])
                            / scale[None, held_idx]
                        )
                    )
                )
                if held_idx
                else 0.0
            )
            sub = _eval_subset_metrics(test_arr["labels"], pred_phys, tasks, tuple(active_held))
            rows.append(
                {
                    "fold": fold,
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

    if rows:
        with (mech_dir / "leo_prompt_interventions.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    out = {"leo_interventions": rows}
    write_json(mech_dir / "leo_prompt_interventions.json", out)
    return out


def _donor_z_transfer_baseline(
    obs_pred: np.ndarray,
    stats_arr: dict,
    tasks: list[str],
    held_out_tasks: tuple[str, ...],
) -> np.ndarray:
    """Non-parametric toxicological transfer baseline: average z-scores of same-species/route donors."""
    meta = task_metadata(tuple(tasks))
    counts = np.asarray(stats_arr["count"], dtype=np.float64)
    mean = np.asarray(stats_arr["mean"], dtype=np.float64)
    scale = np.asarray(stats_arr["scale"], dtype=np.float64)
    pos_idx = [i for i, c in enumerate(counts) if c > 0]
    z_obs = (obs_pred - mean[None, :]) / scale[None, :]
    out = obs_pred.copy()
    for t in held_out_tasks:
        if t not in tasks:
            continue
        ti = tasks.index(t)
        mt = meta[t]
        same_org = [j for j in pos_idx if meta[tasks[j]]["organism"] == mt["organism"]]
        same_rt = [j for j in pos_idx if meta[tasks[j]]["route"] == mt["route"]]
        donors = sorted(set(same_org) & set(same_rt))
        if not donors:
            donors = same_org if same_org else (same_rt if same_rt else pos_idx)
        z_est = np.mean(z_obs[:, donors], axis=1)
        out[:, ti] = (z_est * scale[ti] + mean[ti]).astype(np.float32)
    return out


def summarize_formal_b2(
    *,
    work_dir: Path,
    unified_views_root: Path,
    groups_path: Path,
    held_out_tasks: tuple[str, ...] = LEO6_TASKS,
) -> dict:
    runs_dir = work_dir / "runs"
    endpoint_groups = read_json(groups_path)

    # 1. Multi-seed 5-fold FULL benchmark (5 folds x 3 seeds + 3-seed ensemble)
    ms_rows = []
    ens_rows = []
    for method in ("MV_BIOPGT_FINAL", "TOX_AUTHOR"):
        per_run = []
        for fold in SCAFFOLD_FOLDS:
            vdir = unified_views_root / fold / "FULL"
            if not (vdir / "manifest.json").is_file():
                continue
            manifest = load_manifest(vdir)
            tasks = list(manifest["task_names"])
            test_arr = load_arrays(vdir, "test.npz", manifest)
            seed_preds = []
            for seed in MULTISEED_LIST:
                rdir = runs_dir / f"{fold}_FULL_{method}_s{seed}"
                if not (rdir / "summary.json").is_file():
                    continue
                s = read_json(rdir / "summary.json")
                if s.get("status") != "COMPLETE":
                    continue
                tm = read_json(rdir / "test_metrics.json")
                sub = _subdomain_metrics(tm)
                per_run.append(
                    {
                        "method": method,
                        "fold": fold,
                        "seed": seed,
                        "ALL59_rmse": s["test_groups"]["ALL59"]["macro_rmse"],
                        "ALL59_r2": s["test_groups"]["ALL59"]["macro_r2"],
                        "LD50_rmse": sub["LD50_37"],
                        "LDLo_rmse": sub["LDLo_19"],
                        "Human3_rmse": s["test_groups"]["Human3"]["macro_rmse"],
                        "Avian_rmse": sub["Avian_5"],
                        "NonRodent_rmse": sub["NonRodent_21"],
                        "Oral_rmse": sub["Oral_19"],
                    }
                )
                seed_preds.append(np.load(rdir / "test_predictions.npz")["prediction"])
            if len(seed_preds) >= 2:
                ens_pred = np.mean(np.stack(seed_preds, axis=0), axis=0)
                ens_rep = compute_group_metrics(
                    test_arr["labels"], ens_pred, tasks, endpoint_groups, groups=test_arr["group_id"]
                )
                ens_sub = _subdomain_metrics(ens_rep)
                ens_rows.append(
                    {
                        "method": f"{method}_ENS{len(seed_preds)}",
                        "fold": fold,
                        "n_seeds": len(seed_preds),
                        "ALL59_rmse": ens_rep["groups"]["ALL59"]["macro_rmse"],
                        "ALL59_r2": ens_rep["groups"]["ALL59"]["macro_r2"],
                        "LD50_rmse": ens_sub["LD50_37"],
                        "LDLo_rmse": ens_sub["LDLo_19"],
                        "Human3_rmse": ens_rep["groups"]["Human3"]["macro_rmse"],
                        "Avian_rmse": ens_sub["Avian_5"],
                        "NonRodent_rmse": ens_sub["NonRodent_21"],
                        "Oral_rmse": ens_sub["Oral_19"],
                    }
                )
        ms_rows.extend(per_run)

    ms_agg = []
    metric_keys = (
        "ALL59_rmse",
        "ALL59_r2",
        "LD50_rmse",
        "LDLo_rmse",
        "Human3_rmse",
        "Avian_rmse",
        "NonRodent_rmse",
        "Oral_rmse",
    )
    for label, pool in (
        ("MV_BIOPGT_FINAL_15runs", [r for r in ms_rows if r["method"] == "MV_BIOPGT_FINAL"]),
        ("TOX_AUTHOR_15runs", [r for r in ms_rows if r["method"] == "TOX_AUTHOR"]),
        ("MV_BIOPGT_FINAL_ENS", [r for r in ens_rows if r["method"].startswith("MV_BIOPGT_FINAL")]),
        ("TOX_AUTHOR_ENS", [r for r in ens_rows if r["method"].startswith("TOX_AUTHOR")]),
    ):
        if not pool:
            continue
        entry = {"summary_id": label, "n_units": len(pool)}
        for mk in metric_keys:
            vals = [x[mk] for x in pool if x[mk] is not None]
            entry[f"{mk}_mean"] = float(np.mean(vals)) if vals else None
            entry[f"{mk}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ms_agg.append(entry)

    if ms_rows:
        with (work_dir / "formal_b2_multiseed_runs.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ms_rows[0].keys()))
            w.writeheader()
            w.writerows(ms_rows)
    if ms_agg:
        with (work_dir / "formal_b2_multiseed_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ms_agg[0].keys()))
            w.writeheader()
            w.writerows(ms_agg)

    # 2. Zero-Shot Leave-Endpoint-Out (LEO6 & LEO5_Animal) across S0-S4
    leo_fold_rows = []
    leo_ep_rows = []
    leo_methods = (
        "MV_BIOPGT_FINAL",
        "MV_BIOPGT_FINAL_NOGRAPH",
        "TOX_AUTHOR",
        "MV_BIOPGT_FINAL_NOPROMPT",
    )
    for fold in SCAFFOLD_FOLDS:
        vdir = unified_views_root / fold / "LEO6"
        if not (vdir / "manifest.json").is_file():
            continue
        manifest = load_manifest(vdir)
        tasks = list(manifest["task_names"])
        test_arr = load_arrays(vdir, "test.npz", manifest)
        stats_arr = load_arrays(vdir, "statistics.npz", manifest)
        active_leo6 = tuple(t for t in held_out_tasks if t in tasks)
        active_leo5 = tuple(t for t in LEO5_ANIMAL_TASKS if t in tasks)
        obs_tasks = tuple(t for t in tasks if t not in active_leo6)

        pred_map = {}
        for m in leo_methods:
            rdir = runs_dir / f"{fold}_LEO6_{m}_s42"
            if (rdir / "test_predictions.npz").is_file():
                pred_map[m] = np.load(rdir / "test_predictions.npz")["prediction"]

        # Add non-parametric factor-prior constant & observed-donor z-transfer baselines
        const_pred = np.tile(stats_arr["mean"].astype(np.float32)[None, :], (len(test_arr["labels"]), 1))
        pred_map["FACTOR_PRIOR_CONST"] = const_pred
        if "MV_BIOPGT_FINAL_NOPROMPT" in pred_map:
            pred_map["NOPROMPT_DONOR_Z_AVG"] = _donor_z_transfer_baseline(
                pred_map["MV_BIOPGT_FINAL_NOPROMPT"], stats_arr, tasks, active_leo6
            )

        for m, pr in pred_map.items():
            m6 = _eval_subset_metrics(test_arr["labels"], pr, tasks, active_leo6)
            m5 = _eval_subset_metrics(test_arr["labels"], pr, tasks, active_leo5)
            m_obs = _eval_subset_metrics(test_arr["labels"], pr, tasks, obs_tasks)
            leo_fold_rows.append(
                {
                    "fold": fold,
                    "method": m,
                    "LEO6_rmse": m6.get("macro_rmse"),
                    "LEO6_mae": m6.get("macro_mae"),
                    "LEO6_r2": m6.get("macro_r2"),
                    "LEO6_pos_r2": m6.get("positive_r2_count"),
                    "LEO6_pearson_r": m6.get("macro_pearson_r"),
                    "LEO6_spearman_rho": m6.get("macro_spearman_rho"),
                    "LEO5_Animal_rmse": m5.get("macro_rmse"),
                    "LEO5_Animal_r2": m5.get("macro_r2"),
                    "LEO5_Animal_pearson_r": m5.get("macro_pearson_r"),
                    "OBS53_rmse": m_obs.get("macro_rmse"),
                }
            )
            for ep in m6.get("endpoints", []):
                leo_ep_rows.append({"fold": fold, "method": m, **ep})

    leo_agg = []
    leo_mkeys = (
        "LEO6_rmse",
        "LEO6_mae",
        "LEO6_r2",
        "LEO6_pearson_r",
        "LEO6_spearman_rho",
        "LEO5_Animal_rmse",
        "LEO5_Animal_r2",
        "LEO5_Animal_pearson_r",
        "OBS53_rmse",
    )
    for m in (
        "MV_BIOPGT_FINAL",
        "MV_BIOPGT_FINAL_NOGRAPH",
        "NOPROMPT_DONOR_Z_AVG",
        "TOX_AUTHOR",
        "MV_BIOPGT_FINAL_NOPROMPT",
        "FACTOR_PRIOR_CONST",
    ):
        sub = [r for r in leo_fold_rows if r["method"] == m]
        if not sub:
            continue
        entry = {"method": m, "n_folds": len(sub)}
        for mk in leo_mkeys:
            vals = [x[mk] for x in sub if x[mk] is not None]
            entry[f"{mk}_mean"] = float(np.mean(vals)) if vals else None
            entry[f"{mk}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        leo_agg.append(entry)

    if leo_agg:
        with (work_dir / "formal_b2_leo6_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(leo_agg[0].keys()))
            w.writeheader()
            w.writerows(leo_agg)
    if leo_ep_rows:
        with (work_dir / "formal_b2_leo6_per_endpoint.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(leo_ep_rows[0].keys()))
            w.writeheader()
            w.writerows(leo_ep_rows)

    # 3. Random Split (R0) vs Scaffold Split (S0) Control
    r0_rows = []
    for m in ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "RF"):
        for fold in ("R0", "S0"):
            rdir = runs_dir / f"{fold}_FULL_{m}_s42"
            if not (rdir / "summary.json").is_file():
                continue
            s = read_json(rdir / "summary.json")
            if s.get("status") != "COMPLETE":
                continue
            r0_rows.append(
                {
                    "method": m,
                    "split": fold,
                    "ALL59_rmse": s["test_groups"]["ALL59"]["macro_rmse"],
                    "ALL59_r2": s["test_groups"]["ALL59"]["macro_r2"],
                    "Human3_rmse": s["test_groups"]["Human3"]["macro_rmse"],
                }
            )
    if r0_rows:
        with (work_dir / "formal_b2_r0_vs_s0.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(r0_rows[0].keys()))
            w.writeheader()
            w.writerows(r0_rows)

    def _fms(d: dict, key: str) -> str:
        mv = d.get(f"{key}_mean")
        sv = d.get(f"{key}_sd", 0.0)
        if mv is None:
            return "NA"
        return f"{mv:.4f}±{0.0 if sv is None else sv:.4f}"

    # Markdown summary report
    lines = [
        "# Formal Batch 2 Closing Report (Multi-Seed, Zero-Shot LEO6 & R0 Split Control)",
        "",
        f"- Commit: `{git_identity()}`",
        "",
        "## 1. Five-Fold x Three-Seed (`seeds 42, 43, 44`) & 3-Seed Ensemble on `FULL`",
        "",
        "| Summary ID | Units | ALL59 RMSE | ALL59 R2 | LD50(37) | LDLo(19) | Human3 | Avian(5) | NonRod(21) | Oral(19) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ms_agg:
        lines.append(
            f"| {r['summary_id']} | {r['n_units']} | "
            f"{_fms(r, 'ALL59_rmse')} | {_fms(r, 'ALL59_r2')} | "
            f"{_fms(r, 'LD50_rmse')} | {_fms(r, 'LDLo_rmse')} | "
            f"{_fms(r, 'Human3_rmse')} | {_fms(r, 'Avian_rmse')} | "
            f"{_fms(r, 'NonRodent_rmse')} | {_fms(r, 'Oral_rmse')} |"
        )
    lines.extend(
        [
            "",
            "## 2. Zero-Shot Unseen Endpoint Extrapolation (`S0-S4 / LEO6`, `n_t = 0`)",
            "",
            "| Method | Folds | LEO6 RMSE | LEO6 R2 | LEO6 Pearson r | LEO6 Spearman rho | LEO5 Animal RMSE | LEO5 Pearson r | OBS53 RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in leo_agg:
        lines.append(
            f"| {r['method']} | {r['n_folds']} | "
            f"{_fms(r, 'LEO6_rmse')} | {_fms(r, 'LEO6_r2')} | "
            f"{_fms(r, 'LEO6_pearson_r')} | {_fms(r, 'LEO6_spearman_rho')} | "
            f"{_fms(r, 'LEO5_Animal_rmse')} | {_fms(r, 'LEO5_Animal_pearson_r')} | "
            f"{_fms(r, 'OBS53_rmse')} |"
        )
    if r0_rows:
        lines.extend(
            [
                "",
                "## 3. Random Split (`R0`) vs Scaffold Split (`S0`) Optimism Control",
                "",
                "| Method | Split | ALL59 RMSE | ALL59 R2 | Human3 RMSE |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for r in r0_rows:
            lines.append(
                f"| {r['method']} | {r['split']} | {r['ALL59_rmse']:.4f} | {r['ALL59_r2']:.4f} | {r['Human3_rmse']:.4f} |"
            )

    report_md = "\n".join(lines) + "\n"
    (work_dir / "formal_b2_report.md").write_text(report_md, encoding="utf-8")

    payload = {
        "code_commit": git_identity(),
        "multiseed_summary": ms_agg,
        "ensemble_per_fold": ens_rows,
        "leo6_summary": leo_agg,
        "r0_vs_s0": r0_rows,
    }
    write_json(work_dir / "formal_b2_summary.json", payload)
    print(report_md, flush=True)
    return payload


def collect_formal_b2(work_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_path = work_dir / f"PGTV2_FORMAL_B2_RESULTS_{stamp}.zip"
    excluded_top = {"mv_cache", "leo_views", "views_all"}
    files_to_pack = []
    for p in sorted(work_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(work_dir)
        if rel.parts[0] in excluded_top:
            continue
        if len(rel.parts) >= 2 and rel.parts[0] == "release" and rel.parts[1] == "views":
            continue
        if p.suffix in (".pt", ".zip", ".sha256"):
            continue
        files_to_pack.append((p, rel.as_posix()))

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_to_pack:
            zf.write(src, arcname=arcname)

    sha = file_sha256(archive_path)
    (archive_path.with_suffix(archive_path.suffix + ".sha256")).write_text(
        f"{sha}  {archive_path.name}\n", encoding="utf-8"
    )
    return archive_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formal Batch 2 closing runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--mv-cache", type=Path, required=True)
    p_pre.add_argument("--views-root", type=Path, required=True)
    p_pre.add_argument("--work", type=Path, required=True)
    p_pre.add_argument("--out", type=Path, required=True)
    p_pre.add_argument("--device", default="cuda:0")

    p_run = sub.add_parser("run-b2")
    p_run.add_argument("--mv-cache", type=Path, required=True)
    p_run.add_argument("--views-root", type=Path, required=True)
    p_run.add_argument("--groups", type=Path, required=True)
    p_run.add_argument("--work", type=Path, required=True)
    p_run.add_argument("--gpus", type=int, nargs="+", default=[0])
    p_run.add_argument("--prev-b1-dir", type=Path, default=None)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--work", type=Path, required=True)
    p_sum.add_argument("--views-root", type=Path, required=True)
    p_sum.add_argument("--groups", type=Path, required=True)

    p_col = sub.add_parser("collect")
    p_col.add_argument("--work", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.cmd == "preflight":
        run_preflight_b2(args.mv_cache, args.views_root, args.work, args.out, args.device)
        return 0

    if args.cmd == "run-b2":
        runs_dir = args.work / "runs"
        logs_dir = args.work / "logs"
        unified_views = prepare_unified_views_dir(args.views_root, args.work, SCAFFOLD_FOLDS)

        # Read-only import of verified seed=42 S0-S4 FULL runs from Formal Batch 1
        if args.prev_b1_dir is not None and (args.prev_b1_dir / "runs").is_dir():
            prev_runs = args.prev_b1_dir / "runs"
            for fold in SCAFFOLD_FOLDS:
                vfull = unified_views / fold / "FULL"
                if not (vfull / "manifest.json").is_file():
                    continue
                vsha = load_manifest(vfull)["manifest_sha256"]
                for m in ("MV_BIOPGT_FINAL", "TOX_AUTHOR", "RF"):
                    rid = f"{fold}_FULL_{m}_s42"
                    _try_import_s0_run(
                        src_dir=prev_runs / rid,
                        dst_dir=runs_dir / rid,
                        expected_view_sha=vsha,
                        canonical_method=m,
                    )

        has_r0 = (unified_views / "R0" / "FULL" / "manifest.json").is_file()
        jobs = build_b2_jobs(folds=SCAFFOLD_FOLDS, extra_seeds=(43, 44), leo_seed=42, include_r0=has_r0)
        _dispatch_formal_jobs(
            jobs,
            mv_cache_path=args.mv_cache,
            views_root=unified_views,
            groups_path=args.groups,
            runs_dir=runs_dir,
            logs_dir=logs_dir,
            gpus=args.gpus,
        )
        export_leo_interventions(
            mv_cache_path=args.mv_cache,
            unified_views_root=unified_views,
            work_dir=args.work,
            device=f"cuda:{args.gpus[0]}",
            seed=42,
            folds=SCAFFOLD_FOLDS,
        )
        summarize_formal_b2(
            work_dir=args.work,
            unified_views_root=unified_views,
            groups_path=args.groups,
        )
        return 0

    if args.cmd == "summarize":
        unified_views = prepare_unified_views_dir(args.views_root, args.work, SCAFFOLD_FOLDS)
        summarize_formal_b2(
            work_dir=args.work,
            unified_views_root=unified_views,
            groups_path=args.groups,
        )
        return 0

    if args.cmd == "collect":
        out = collect_formal_b2(args.work)
        print(f"ARCHIVE={out}", flush=True)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
