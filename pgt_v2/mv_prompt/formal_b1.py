"""Formal Batch 1 (5-Fold S0-S4 Main Benchmark, Matched Z-Score Ablations & Mechanistic Exports).

Usage:
    python -m pgt_v2.mv_prompt.formal_b1 preflight --mv-cache <mv_cache.pt> --view <S0/FULL> --out <preflight_b1.json> --device cuda:0
    python -m pgt_v2.mv_prompt.formal_b1 run-b1 --mv-cache <mv_cache.pt> --views-root <views> --groups <endpoint_groups.json> --work <work_dir> --gpus 0 1 2 3 [--gate-runs-dir <dir>] [--probe-runs-dir <dir>]
    python -m pgt_v2.mv_prompt.formal_b1 mechanistic --mv-cache <mv_cache.pt> --views-root <views> --groups <endpoint_groups.json> --work <work_dir> --device cuda:0 [--probe-runs-dir <dir>]
    python -m pgt_v2.mv_prompt.formal_b1 summarize --work <work_dir> --groups <endpoint_groups.json>
    python -m pgt_v2.mv_prompt.formal_b1 collect --work <work_dir>
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from ..artifacts import load_arrays, load_manifest, read_json, write_json
from ..cache import file_sha256
from ..config import git_identity
from ..models import task_metadata
from .features import MVDataStore, fit_descriptor_scaler, load_mv_cache, transform_descriptors
from .models import build_mv_model, count_parameters, method_input_needs
from .probe_v2 import _subdomain_metrics
from .training import compute_group_metrics, seed_all, tempered_mse_loss

SCAFFOLD_FOLDS = ("S0", "S1", "S2", "S3", "S4")

S4_CASE_RECORDS = (
    "row_59",
    "row_4318",
    "row_9135",
    "row_38535",
    "row_13552",
    "row_78620",
    "row_4572",
    "row_70055",
)
ACONITUM_PAIRS = (
    ("aconitine_to_benzoylaconine", "row_59", "row_4318", 2.31178),
    ("mesaconitine_to_benzoylmesaconine", "row_9135", "row_38535", 2.35213),
    ("hypaconitine_to_benzoylhypaconine", "row_13552", "row_78620", 1.72034),
)


def run_preflight_b1(mv_cache_path: Path, view_root: Path, output_path: Path, device: str) -> dict:
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
    ep_w = torch.ones(len(manifest["task_names"]), device=device)

    checked = []
    for method in (
        "MV_BIOPGT_FINAL",
        "MV_BIOPGT_FINAL_NOGRAPH",
        "MV_BIOPGT_FINAL_NOPROMPT",
        "MV_BIOPGT_FINAL_NOMDROP",
        "TOX_AUTHOR",
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
        if method.startswith("MV_BIOPGT_FINAL"):
            out, train_aux = model(payload, return_train_aux=True)
        else:
            out, train_aux = model(payload), {}
        if tuple(out.shape) != (len(active), len(manifest["task_names"])) or not torch.isfinite(out).all():
            raise RuntimeError(f"Invalid forward output for {method}: {tuple(out.shape)}")
        target = torch.from_numpy(train_arr["labels"][active].astype(np.float32)).to(device)
        loss = tempered_mse_loss(out, target, ep_w)
        if "graph_pred" in train_aux:
            loss = loss + 0.25 * tempered_mse_loss(train_aux["graph_pred"], target, ep_w)
        loss.backward()
        grad_ok = all((p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
        if not grad_ok:
            raise RuntimeError(f"Nonfinite gradient in preflight_b1 for {method}")
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
        "status": "PREFLIGHT_B1_PASS",
        "code_commit": git_identity(),
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "torch": str(torch.__version__),
        "models_checked": checked,
    }
    write_json(output_path, report)
    return report


def _try_import_s0_run(
    *,
    src_dir: Path | None,
    dst_dir: Path,
    expected_view_sha: str,
    canonical_method: str,
) -> bool:
    """Safely import an already-completed S0 run if view_sha256 matches and files are intact."""
    if src_dir is None or not (src_dir / "summary.json").is_file():
        return False
    if (dst_dir / "summary.json").is_file():
        try:
            if read_json(dst_dir / "summary.json").get("status") == "COMPLETE":
                return True
        except Exception:
            pass
    required = [
        "summary.json",
        "validation_metrics.json",
        "test_metrics.json",
        "validation_predictions.npz",
        "test_predictions.npz",
    ]
    if not all((src_dir / f).is_file() for f in required):
        return False
    s = read_json(src_dir / "summary.json")
    if s.get("status") != "COMPLETE" or s.get("view_sha256") != expected_view_sha:
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in required + ["history.json"]:
        if (src_dir / fname).is_file():
            shutil.copy2(src_dir / fname, dst_dir / fname)
    s_new = dict(s)
    s_new["method"] = canonical_method
    s_new["reused_from"] = str(src_dir)
    if (src_dir / "best.pt").is_file():
        s_new["reused_ckpt_path"] = str((src_dir / "best.pt").resolve())
    write_json(dst_dir / "summary.json", s_new)
    print(f"[b1] REUSED verified S0 run {dst_dir.name} <- {src_dir}", flush=True)
    return True


def _dispatch_formal_jobs(
    jobs: list[dict],
    *,
    mv_cache_path: Path,
    views_root: Path,
    groups_path: Path,
    runs_dir: Path,
    logs_dir: Path,
    gpus: list[int],
) -> None:
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
                    print(f"[b1] SKIP completed {job['run_id']}", flush=True)
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
            str(views_root / job["fold"] / job["budget"]),
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
        print(
            f"[b1] START {tag} {job['run_id']} ({job['fold']}/{job['budget']} {job['method']})",
            flush=True,
        )
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
                    print(f"[b1] FAILED {job['run_id']} exit={rc}", flush=True)
                else:
                    print(f"[b1] DONE   {job['run_id']}", flush=True)

        if active_cpu is not None:
            proc, fh, job = active_cpu
            rc = proc.poll()
            if rc is not None:
                fh.close()
                active_cpu = None
                if rc != 0:
                    failed.append((job["run_id"], rc))
                    print(f"[b1] FAILED {job['run_id']} exit={rc}", flush=True)
                else:
                    print(f"[b1] DONE   {job['run_id']}", flush=True)

        if active_gpu or active_cpu is not None:
            time.sleep(2.0)

    if failed:
        raise RuntimeError(f"Formal Batch 1 failed on runs: {failed}")


def build_b1_jobs(folds: tuple[str, ...] = SCAFFOLD_FOLDS, seed: int = 42) -> list[dict]:
    """Build Formal Batch 1 job queue ordered longest-first to minimize multi-GPU tail idle."""
    jobs = []
    # Priority 1: Slowest reference baseline TOX_AUTHOR (~36 min each)
    for budget, btag in (("FULL", "FULL"), ("S21_P10", "P10")):
        for fold in folds:
            jobs.append(
                {
                    "run_id": f"{fold}_{btag}_TOX_AUTHOR_s{seed}",
                    "fold": fold,
                    "budget": budget,
                    "method": "TOX_AUTHOR",
                    "seed": seed,
                    "config": {},
                }
            )
    # Priority 2: Primary locked single model MV_BIOPGT_FINAL (~22 min each)
    for budget, btag in (("FULL", "FULL"), ("S21_P10", "P10")):
        for fold in folds:
            jobs.append(
                {
                    "run_id": f"{fold}_{btag}_MV_BIOPGT_FINAL_s{seed}",
                    "fold": fold,
                    "budget": budget,
                    "method": "MV_BIOPGT_FINAL",
                    "seed": seed,
                    "config": {"gamma": 0.50},
                }
            )
    # Priority 3: Matched Z-Score GINE ablations (~16-21 min each)
    for fold in folds:
        jobs.append(
            {
                "run_id": f"{fold}_FULL_MV_BIOPGT_FINAL_NOPROMPT_s{seed}",
                "fold": fold,
                "budget": "FULL",
                "method": "MV_BIOPGT_FINAL_NOPROMPT",
                "seed": seed,
                "config": {"gamma": 0.50},
            }
        )
        jobs.append(
            {
                "run_id": f"{fold}_P10_MV_BIOPGT_FINAL_NOPROMPT_s{seed}",
                "fold": fold,
                "budget": "S21_P10",
                "method": "MV_BIOPGT_FINAL_NOPROMPT",
                "seed": seed,
                "config": {"gamma": 0.50},
            }
        )
        jobs.append(
            {
                "run_id": f"{fold}_FULL_MV_BIOPGT_FINAL_NOMDROP_s{seed}",
                "fold": fold,
                "budget": "FULL",
                "method": "MV_BIOPGT_FINAL_NOMDROP",
                "seed": seed,
                "config": {"gamma": 0.50},
            }
        )
        jobs.append(
            {
                "run_id": f"{fold}_FULL_GINE_ONLY_s{seed}",
                "fold": fold,
                "budget": "FULL",
                "method": "GINE_ONLY",
                "seed": seed,
                "config": {"gamma": 0.50},
            }
        )
    # Priority 4: Fast non-graph neural ablations & baselines (~1-4 min each) + CPU RF
    for budget, btag in (("FULL", "FULL"), ("S21_P10", "P10")):
        for fold in folds:
            jobs.append(
                {
                    "run_id": f"{fold}_{btag}_MV_BIOPGT_FINAL_NOGRAPH_s{seed}",
                    "fold": fold,
                    "budget": budget,
                    "method": "MV_BIOPGT_FINAL_NOGRAPH",
                    "seed": seed,
                    "config": {"gamma": 0.50},
                }
            )
            jobs.append(
                {
                    "run_id": f"{fold}_{btag}_MV_MLP_s{seed}",
                    "fold": fold,
                    "budget": budget,
                    "method": "MV_MLP",
                    "seed": seed,
                    "config": {"gamma": 0.50},
                }
            )
            jobs.append(
                {
                    "run_id": f"{fold}_{btag}_RF_s{seed}",
                    "fold": fold,
                    "budget": budget,
                    "method": "RF",
                    "seed": seed,
                    "config": {"n_estimators": 300, "n_jobs": 8},
                }
            )
    return jobs


def _load_final_ckpt_for_fold(
    fold: str,
    runs_dir: Path,
    probe_runs_dir: Path | None,
    seed: int,
    device: str | torch.device,
):
    run_dir = runs_dir / f"{fold}_FULL_MV_BIOPGT_FINAL_s{seed}"
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file() and (run_dir / "summary.json").is_file():
        s = read_json(run_dir / "summary.json")
        if s.get("reused_ckpt_path") and Path(s["reused_ckpt_path"]).is_file():
            ckpt_path = Path(s["reused_ckpt_path"])
    if (
        not ckpt_path.is_file()
        and fold == "S0"
        and probe_runs_dir is not None
        and (probe_runs_dir / f"FULL_MV_BIOPGT_V2_ZSCORE_s{seed}" / "best.pt").is_file()
    ):
        ckpt_path = probe_runs_dir / f"FULL_MV_BIOPGT_V2_ZSCORE_s{seed}" / "best.pt"
    if not ckpt_path.is_file():
        return None
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_mv_model(
        "MV_BIOPGT_FINAL",
        tuple(blob["task_names"]),
        blob["counts"],
        adjacency=blob["adjacency"],
        config=blob.get("config") or {},
        device=device,
    )
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


@torch.no_grad()
def _predict_with_override(
    model,
    store: MVDataStore,
    offsets: np.ndarray,
    desc_array: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int = 256,
    prompt_override: torch.Tensor | None = None,
    factor_override: dict[str, torch.Tensor] | None = None,
) -> np.ndarray:
    needs = method_input_needs("MV_BIOPGT_FINAL")
    pieces = []
    for start in range(0, len(offsets), batch_size):
        end = min(start + batch_size, len(offsets))
        payload = store.build_batch(
            offsets[start:end],
            desc_array=desc_array[start:end],
            device=device,
            **needs,
        )
        out = (
            model(
                payload,
                prompt_override=prompt_override,
                factor_override=factor_override,
            )
            .detach()
            .cpu()
            .numpy()
            * scale[None, :]
            + mean[None, :]
        )
        pieces.append(out.astype(np.float32))
    return np.concatenate(pieces, axis=0)


@torch.no_grad()
def export_mechanistic(
    *,
    mv_cache_path: Path,
    views_root: Path,
    groups_path: Path,
    work_dir: Path,
    device: str = "cuda:0",
    seed: int = 42,
    folds: tuple[str, ...] = SCAFFOLD_FOLDS,
    probe_runs_dir: Path | None = None,
) -> dict:
    """Run zero-training prompt interventions across folds + S4 held-out case attributions."""
    store = load_mv_cache(mv_cache_path)
    endpoint_groups = read_json(groups_path)
    runs_dir = work_dir / "runs"
    mech_dir = work_dir / "mechanistic"
    mech_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")

    intervention_rows = []
    for fold in folds:
        view_dir = views_root / fold / "FULL"
        if not (view_dir / "manifest.json").is_file():
            continue
        loaded = _load_final_ckpt_for_fold(fold, runs_dir, probe_runs_dir, seed, dev)
        if loaded is None:
            continue
        model, blob = loaded
        manifest = load_manifest(view_dir)
        tasks = list(manifest["task_names"])
        test_arr = load_arrays(view_dir, "test.npz", manifest)
        test_off = test_arr["cache_offset"].astype(np.int64)
        test_desc = transform_descriptors(
            store.desc_raw, store.desc_missing, test_off, blob["desc_scaler"]
        )
        mean = np.asarray(blob["target_mean"], dtype=np.float64)
        scale = np.asarray(blob["target_scale"], dtype=np.float64)

        prompts = model.prompt_bank().detach()
        rel_gate_val = (
            float(torch.tanh(model.prompt_bank.rel_gate).detach().cpu().item())
            if model.prompt_bank.rel_gate is not None
            else 0.0
        )
        np.savez_compressed(
            mech_dir / f"prompts_{fold}.npz",
            task_names=np.asarray(tasks),
            prompts=prompts.cpu().numpy(),
            shrinkage=model.prompt_bank.shrinkage.detach().cpu().numpy(),
            rel_gate=np.asarray([rel_gate_val], dtype=np.float32),
        )

        # Build deterministic factor overrides
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

        gen = torch.Generator(device="cpu").manual_seed(20261126)
        perm_idx = torch.randperm(len(tasks), generator=gen).to(prompts.device)
        perm_prompts = prompts[perm_idx]

        conds = [
            ("intact", None, None),
            ("perm_all", perm_prompts, None),
            ("swap_oral_iv", None, {"route": swap_route}),
            ("shift_species", None, {"organism": shift_org}),
        ]
        for cname, p_ov, f_ov in conds:
            pred = _predict_with_override(
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
            rep = compute_group_metrics(test_arr["labels"], pred, tasks, endpoint_groups)
            sub = _subdomain_metrics(rep)
            intervention_rows.append(
                {
                    "fold": fold,
                    "condition": cname,
                    "ALL59_rmse": rep["groups"]["ALL59"]["macro_rmse"],
                    "ALL59_r2": rep["groups"]["ALL59"]["macro_r2"],
                    "S21_rmse": rep["groups"]["S21"]["macro_rmse"],
                    "Human3_rmse": rep["groups"]["Human3"]["macro_rmse"],
                    "Oral_19_rmse": sub["Oral_19"],
                }
            )

        # S4 held-out toxicology cases & C8-hydrolysis channel counterfactuals
        if fold == "S4":
            rec_ids = [str(r) for r in test_arr["record_id"]]
            rec_map = {r: i for i, r in enumerate(rec_ids)}
            present = [r for r in S4_CASE_RECORDS if r in rec_map]
            if present:
                case_rows = [rec_map[r] for r in present]
                case_off = test_off[case_rows]
                case_desc = test_desc[case_rows]
                needs = method_input_needs("MV_BIOPGT_FINAL")
                payload = store.build_batch(case_off, desc_array=case_desc, device=dev, **needs)
                raw_out, aux = model(payload, return_aux=True)
                case_pred_final = raw_out.detach().cpu().numpy() * scale[None, :] + mean[None, :]
                attn_np = aux["atom_attention"].detach().cpu().numpy()  # [C, T, H, N]
                mask_np = aux["atom_mask"].detach().cpu().numpy()  # [C, N]

                # Collect predictions from other S4 methods for direct comparison
                other_preds = {"MV_BIOPGT_FINAL": case_pred_final.tolist()}
                for m_other in (
                    "TOX_AUTHOR",
                    "MV_BIOPGT_FINAL_NOGRAPH",
                    "MV_BIOPGT_FINAL_NOPROMPT",
                    "GINE_ONLY",
                    "RF",
                ):
                    odir = runs_dir / f"S4_FULL_{m_other}_s{seed}"
                    if (odir / "test_predictions.npz").is_file():
                        op = np.load(odir / "test_predictions.npz")["prediction"]
                        other_preds[m_other] = op[case_rows].tolist()

                # Channel counterfactuals for the 3 Aconitum C8-hydrolysis pairs
                pair_counterfactuals = []
                p_idx_map = {r: k for k, r in enumerate(present)}
                h_graph_all, atoms_all, mask_all = model.graph_encoder(payload)
                h_chem_all = model.chem_encoder(payload.avalon, payload.morgan, payload.desc)

                def _fuse_and_decode(hg, hc, at, am):
                    cat = torch.cat([hg, hc], dim=-1)
                    g = model.modal_gate(cat)
                    fused = model.modal_norm(g * hg + (1.0 - g) * hc + model.modal_proj(cat))
                    pr, _ = model._decode_from_repr(fused, at, am)
                    return pr.detach().cpu().numpy() * scale[None, :] + mean[None, :]

                key_tasks = [
                    t
                    for t in (
                        "mouse_intravenous_LD50",
                        "mouse_oral_LD50",
                        "rat_oral_LD50",
                        "guinea pig_intravenous_LD50",
                        "human_oral_TDLo",
                    )
                    if t in tasks
                ]
                for pname, rec_a, rec_b, lit_delta in ACONITUM_PAIRS:
                    if rec_a in p_idx_map and rec_b in p_idx_map:
                        ia, ib = p_idx_map[rec_a], p_idx_map[rec_b]
                        y_aa = _fuse_and_decode(
                            h_graph_all[ia : ia + 1],
                            h_chem_all[ia : ia + 1],
                            atoms_all[ia : ia + 1],
                            mask_all[ia : ia + 1],
                        )[0]
                        y_bb = _fuse_and_decode(
                            h_graph_all[ib : ib + 1],
                            h_chem_all[ib : ib + 1],
                            atoms_all[ib : ib + 1],
                            mask_all[ib : ib + 1],
                        )[0]
                        y_ba = _fuse_and_decode(
                            h_graph_all[ib : ib + 1],
                            h_chem_all[ia : ia + 1],
                            atoms_all[ib : ib + 1],
                            mask_all[ib : ib + 1],
                        )[0]
                        y_ab = _fuse_and_decode(
                            h_graph_all[ia : ia + 1],
                            h_chem_all[ib : ib + 1],
                            atoms_all[ia : ia + 1],
                            mask_all[ia : ia + 1],
                        )[0]
                        by_kt = {}
                        for kt in key_tasks:
                            ti = tasks.index(kt)
                            by_kt[kt] = {
                                "parent_pred": round(float(y_aa[ti]), 5),
                                "hydro_pred": round(float(y_bb[ti]), 5),
                                "delta_full": round(float(y_aa[ti] - y_bb[ti]), 5),
                                "delta_graph_only": round(float(y_aa[ti] - y_ba[ti]), 5),
                                "delta_chem_only": round(float(y_aa[ti] - y_ab[ti]), 5),
                                "parent_truth": (
                                    round(float(test_arr["labels"][case_rows[ia], ti]), 5)
                                    if np.isfinite(test_arr["labels"][case_rows[ia], ti])
                                    else None
                                ),
                                "hydro_truth": (
                                    round(float(test_arr["labels"][case_rows[ib], ti]), 5)
                                    if np.isfinite(test_arr["labels"][case_rows[ib], ti])
                                    else None
                                ),
                            }
                        pair_counterfactuals.append(
                            {
                                "pair": pname,
                                "parent_record": rec_a,
                                "hydro_record": rec_b,
                                "sato1979_mouse_iv_expected_delta": lit_delta,
                                "endpoints": by_kt,
                            }
                        )

                np.savez_compressed(
                    mech_dir / "s4_case_attentions.npz",
                    record_id=np.asarray(present),
                    task_names=np.asarray(tasks),
                    atom_attention=attn_np,
                    atom_mask=mask_np,
                    truth=test_arr["labels"][case_rows],
                )
                write_json(
                    mech_dir / "s4_cases.json",
                    {
                        "fold": "S4",
                        "records": present,
                        "task_names": tasks,
                        "model_predictions": other_preds,
                        "aconitum_c8_counterfactuals": pair_counterfactuals,
                    },
                )

        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if intervention_rows:
        with (mech_dir / "prompt_interventions.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(intervention_rows[0].keys()))
            writer.writeheader()
            writer.writerows(intervention_rows)
    out_summary = {"interventions": intervention_rows}
    write_json(mech_dir / "prompt_interventions.json", out_summary)
    return out_summary


def summarize_formal_b1(*, work_dir: Path, groups_path: Path) -> dict:
    runs_dir = work_dir / "runs"
    per_fold_rows = []
    grouped: dict[tuple[str, str], list[dict]] = {}

    for sp in sorted(runs_dir.glob("*/summary.json")):
        rdir = sp.parent
        s = read_json(sp)
        if s.get("status") != "COMPLETE":
            continue
        tm = read_json(rdir / "test_metrics.json")
        rid = rdir.name
        parts = rid.split("_", 2)
        fold = s["fold"]
        budget = "FULL" if parts[1] == "FULL" else "S21_P10"
        method = s["method"]
        tg = s["test_groups"]
        vg = s["validation_groups"]
        sub = _subdomain_metrics(tm)
        row = {
            "run_id": rid,
            "fold": fold,
            "budget": budget,
            "method": method,
            "best_epoch": s["best_epoch"],
            "seconds": s["elapsed_seconds"],
            "reused": bool(s.get("reused_from")),
            "val_ALL59_rmse": vg["ALL59"]["macro_rmse"],
            "test_ALL59_rmse": tg["ALL59"]["macro_rmse"],
            "test_ALL59_r2": tg["ALL59"]["macro_r2"],
            "test_L11_rmse": tg["L11"]["macro_rmse"],
            "test_M27_rmse": tg["M27"]["macro_rmse"],
            "test_S21_rmse": tg["S21"]["macro_rmse"],
            "test_Human3_rmse": tg["Human3"]["macro_rmse"],
            "test_LD50_rmse": sub["LD50_37"],
            "test_LDLo_rmse": sub["LDLo_19"],
            "test_TDLo_rmse": sub["TDLo_3"],
            "test_Avian_rmse": sub["Avian_5"],
            "test_NonRodent_rmse": sub["NonRodent_21"],
            "test_Oral_rmse": sub["Oral_19"],
        }
        per_fold_rows.append(row)
        grouped.setdefault((budget, method), []).append(row)

    if not per_fold_rows:
        raise RuntimeError("No completed runs found to summarize")

    with (work_dir / "formal_b1_per_fold.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_fold_rows[0].keys()))
        w.writeheader()
        w.writerows(per_fold_rows)

    metric_cols = [
        "test_ALL59_rmse",
        "test_ALL59_r2",
        "test_L11_rmse",
        "test_M27_rmse",
        "test_S21_rmse",
        "test_Human3_rmse",
        "test_LD50_rmse",
        "test_LDLo_rmse",
        "test_TDLo_rmse",
        "test_Avian_rmse",
        "test_NonRodent_rmse",
        "test_Oral_rmse",
    ]
    agg_rows = []
    for (budget, method), items in sorted(grouped.items()):
        for scope_name, filt in (
            ("5fold_S0_S4", lambda x: True),
            ("blind4_S1_S4", lambda x: x["fold"] != "S0"),
        ):
            sub_items = [x for x in items if filt(x)]
            if not sub_items:
                continue
            entry = {
                "scope": scope_name,
                "budget": budget,
                "method": method,
                "n_folds": len(sub_items),
            }
            for mc in metric_cols:
                vals = [x[mc] for x in sub_items if x[mc] is not None]
                entry[f"{mc}_mean"] = float(np.mean(vals)) if vals else None
                entry[f"{mc}_sd"] = (
                    float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
                )
            agg_rows.append(entry)

    with (work_dir / "formal_b1_5fold_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    report = {
        "code_commit": git_identity(),
        "total_runs": len(per_fold_rows),
        "aggregated": agg_rows,
    }
    write_json(work_dir / "formal_b1_summary.json", report)

    lines = [
        "# Formal Batch 1 Report (5-Fold S0-S4 & Blind 4-Fold S1-S4)",
        "",
        f"- Commit: `{report['code_commit']}` | Total completed run units: `{len(per_fold_rows)}`",
        "",
        "| Scope | Budget | Method | Folds | ALL59 RMSE | R2 | LD50(37) | LDLo(19) | Human3 | Avian(5) | NonRod(21) | Oral(19) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(
        agg_rows,
        key=lambda x: (x["scope"], x["budget"], x["test_ALL59_rmse_mean"] or 999.0),
    ):
        fmt = lambda k: "-" if r[f"{k}_mean"] is None else f"{r[f'{k}_mean']:.4f}±{r[f'{k}_sd']:.4f}"
        lines.append(
            f"| {r['scope']} | {r['budget']} | {r['method']} | {r['n_folds']} | "
            f"{fmt('test_ALL59_rmse')} | {fmt('test_ALL59_r2')} | {fmt('test_LD50_rmse')} | "
            f"{fmt('test_LDLo_rmse')} | {fmt('test_Human3_rmse')} | {fmt('test_Avian_rmse')} | "
            f"{fmt('test_NonRodent_rmse')} | {fmt('test_Oral_rmse')} |"
        )
    (work_dir / "formal_b1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def run_b1(
    *,
    mv_cache_path: Path,
    views_root: Path,
    groups_path: Path,
    work_dir: Path,
    gpus: list[int],
    seed: int = 42,
    folds: tuple[str, ...] = SCAFFOLD_FOLDS,
    gate_runs_dir: Path | None = None,
    probe_runs_dir: Path | None = None,
) -> dict:
    runs_dir = work_dir / "runs"
    logs_dir = work_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if "S0" in folds:
        s0_full_sha = load_manifest(views_root / "S0" / "FULL")["manifest_sha256"]
        s0_p10_sha = load_manifest(views_root / "S0" / "S21_P10")["manifest_sha256"]
        if probe_runs_dir is not None:
            _try_import_s0_run(
                src_dir=probe_runs_dir / f"FULL_MV_BIOPGT_V2_ZSCORE_s{seed}",
                dst_dir=runs_dir / f"S0_FULL_MV_BIOPGT_FINAL_s{seed}",
                expected_view_sha=s0_full_sha,
                canonical_method="MV_BIOPGT_FINAL",
            )
        if gate_runs_dir is not None:
            _try_import_s0_run(
                src_dir=gate_runs_dir / f"FULL_TOX_AUTHOR_s{seed}",
                dst_dir=runs_dir / f"S0_FULL_TOX_AUTHOR_s{seed}",
                expected_view_sha=s0_full_sha,
                canonical_method="TOX_AUTHOR",
            )
            _try_import_s0_run(
                src_dir=gate_runs_dir / f"P10_TOX_AUTHOR_s{seed}",
                dst_dir=runs_dir / f"S0_P10_TOX_AUTHOR_s{seed}",
                expected_view_sha=s0_p10_sha,
                canonical_method="TOX_AUTHOR",
            )
            _try_import_s0_run(
                src_dir=gate_runs_dir / f"FULL_RF_s{seed}",
                dst_dir=runs_dir / f"S0_FULL_RF_s{seed}",
                expected_view_sha=s0_full_sha,
                canonical_method="RF",
            )
            _try_import_s0_run(
                src_dir=gate_runs_dir / f"P10_RF_s{seed}",
                dst_dir=runs_dir / f"S0_P10_RF_s{seed}",
                expected_view_sha=s0_p10_sha,
                canonical_method="RF",
            )

    jobs = build_b1_jobs(folds=folds, seed=seed)
    _dispatch_formal_jobs(
        jobs,
        mv_cache_path=mv_cache_path,
        views_root=views_root,
        groups_path=groups_path,
        runs_dir=runs_dir,
        logs_dir=logs_dir,
        gpus=gpus,
    )
    dev_str = f"cuda:{gpus[0]}" if (gpus and torch.cuda.is_available()) else "cpu"
    export_mechanistic(
        mv_cache_path=mv_cache_path,
        views_root=views_root,
        groups_path=groups_path,
        work_dir=work_dir,
        device=dev_str,
        seed=seed,
        folds=folds,
        probe_runs_dir=probe_runs_dir,
    )
    return summarize_formal_b1(work_dir=work_dir, groups_path=groups_path)


def collect_formal_b1(work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = work_dir / f"PGTV2_FORMAL_B1_RESULTS_{stamp}.zip"
    include_suffixes = {".json", ".csv", ".md", ".log", ".xml", ".npz"}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(work_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(work_dir).as_posix()
            if (
                rel.startswith("mv_cache/")
                or rel.startswith("release/views/")
                or path.suffix in (".pt", ".zip")
            ):
                continue
            if path.suffix in include_suffixes:
                zf.write(path, rel)
    sha = file_sha256(archive)
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(
        f"{sha}  {archive.name}\n", encoding="utf-8"
    )
    print(f"[b1] Packaged {archive} ({archive.stat().st_size / 1e6:.2f} MB) sha256={sha}", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--mv-cache", required=True)
    p_pre.add_argument("--view", required=True)
    p_pre.add_argument("--out", required=True)
    p_pre.add_argument("--device", default="cuda:0")

    p_run = sub.add_parser("run-b1")
    p_run.add_argument("--mv-cache", required=True)
    p_run.add_argument("--views-root", required=True)
    p_run.add_argument("--groups", required=True)
    p_run.add_argument("--work", required=True)
    p_run.add_argument("--gpus", type=int, nargs="+", required=True)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--folds", nargs="+", default=list(SCAFFOLD_FOLDS))
    p_run.add_argument("--gate-runs-dir", default=None)
    p_run.add_argument("--probe-runs-dir", default=None)

    p_mech = sub.add_parser("mechanistic")
    p_mech.add_argument("--mv-cache", required=True)
    p_mech.add_argument("--views-root", required=True)
    p_mech.add_argument("--groups", required=True)
    p_mech.add_argument("--work", required=True)
    p_mech.add_argument("--device", default="cuda:0")
    p_mech.add_argument("--seed", type=int, default=42)
    p_mech.add_argument("--folds", nargs="+", default=list(SCAFFOLD_FOLDS))
    p_mech.add_argument("--probe-runs-dir", default=None)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--work", required=True)
    p_sum.add_argument("--groups", required=True)

    p_col = sub.add_parser("collect")
    p_col.add_argument("--work", required=True)

    args = parser.parse_args()
    gate_dir = Path(args.gate_runs_dir) if getattr(args, "gate_runs_dir", None) else None
    probe_dir = Path(args.probe_runs_dir) if getattr(args, "probe_runs_dir", None) else None

    if args.cmd == "preflight":
        run_preflight_b1(Path(args.mv_cache), Path(args.view), Path(args.out), args.device)
    elif args.cmd == "run-b1":
        run_b1(
            mv_cache_path=Path(args.mv_cache),
            views_root=Path(args.views_root),
            groups_path=Path(args.groups),
            work_dir=Path(args.work),
            gpus=args.gpus,
            seed=args.seed,
            folds=tuple(args.folds),
            gate_runs_dir=gate_dir,
            probe_runs_dir=probe_dir,
        )
    elif args.cmd == "mechanistic":
        export_mechanistic(
            mv_cache_path=Path(args.mv_cache),
            views_root=Path(args.views_root),
            groups_path=Path(args.groups),
            work_dir=Path(args.work),
            device=args.device,
            seed=args.seed,
            folds=tuple(args.folds),
            probe_runs_dir=probe_dir,
        )
    elif args.cmd == "summarize":
        summarize_formal_b1(work_dir=Path(args.work), groups_path=Path(args.groups))
    elif args.cmd == "collect":
        collect_formal_b1(Path(args.work))


if __name__ == "__main__":
    main()
