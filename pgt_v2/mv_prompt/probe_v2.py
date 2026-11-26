"""S0 single-model V2 probe runner (Physical Dose Ordinal Offset + Training ModalDrop).

Usage:
    python -m pgt_v2.mv_prompt.probe_v2 preflight --mv-cache <mv_cache.pt> --view <S0/FULL> --out <preflight_v2.json> --device cuda:0
    python -m pgt_v2.mv_prompt.probe_v2 run-probe --mv-cache <mv_cache.pt> --views-root <views/S0> --groups <endpoint_groups.json> --work <work_dir> --gpus 0 1 2 3 [--ref-runs-dir <gate_runs_dir>]
    python -m pgt_v2.mv_prompt.probe_v2 summarize --work <work_dir> --groups <endpoint_groups.json> [--ref-runs-dir <gate_runs_dir>]
    python -m pgt_v2.mv_prompt.probe_v2 collect --work <work_dir>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from ..artifacts import load_arrays, load_manifest, read_json, write_json
from ..cache import file_sha256
from ..config import git_identity
from .features import fit_descriptor_scaler, load_mv_cache, transform_descriptors
from .gate import _dispatch_jobs
from .models import build_mv_model, count_parameters, method_input_needs
from .training import seed_all, tempered_huber_loss


def run_preflight_v2(mv_cache_path: Path, view_root: Path, output_path: Path, device: str) -> dict:
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
        "MV_BIOPGT_V2",
        "MV_BIOPGT_V2_NOMDROP",
        "MV_BIOPGT_V2_NOGRAPH",
        "MV_BIOPGT_V2_NOPROMPT",
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
        out, train_aux = model(payload, return_train_aux=True)
        if tuple(out.shape) != (len(active), 59) or not torch.isfinite(out).all():
            raise RuntimeError(f"Invalid forward output for {method}: {tuple(out.shape)}")
        target = torch.from_numpy(train_arr["labels"][active].astype(np.float32)).to(device)
        loss = tempered_huber_loss(out, target, ep_w, beta=1.0)
        if "graph_pred" in train_aux:
            loss = loss + 0.25 * tempered_huber_loss(train_aux["graph_pred"], target, ep_w, beta=1.0)
        loss.backward()
        grad_ok = all(
            (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        if not grad_ok:
            raise RuntimeError(f"Nonfinite gradient in preflight_v2 for {method}")
        model.eval()
        with torch.no_grad():
            eval_out = model(payload)
            if not torch.isfinite(eval_out).all():
                raise RuntimeError(f"Nonfinite eval output for {method}")
        checked.append(
            {
                "method": method,
                "parameters": count_parameters(model),
                "output_shape": list(out.shape),
                "has_aux_graph": "graph_pred" in train_aux,
                "finite_forward_and_grad": True,
            }
        )
        del model
        torch.cuda.empty_cache()

    report = {
        "status": "PREFLIGHT_V2_PASS",
        "code_commit": git_identity(),
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "torch": str(torch.__version__),
        "models_checked": checked,
    }
    write_json(output_path, report)
    return report


def _subdomain_metrics(test_metrics: dict) -> dict:
    endpoints = test_metrics["endpoints"]
    ld50 = [r["rmse"] for r in endpoints if r["task"].endswith("_LD50")]
    ldlo = [r["rmse"] for r in endpoints if r["task"].endswith("_LDLo")]
    tdlo = [r["rmse"] for r in endpoints if r["task"].endswith("_TDLo")]
    avian = [
        r["rmse"]
        for r in endpoints
        if r["task"].rsplit("_", 2)[0] in ("bird-wild", "quail", "duck", "chicken", "frog")
    ]
    nonrod = [
        r["rmse"]
        for r in endpoints
        if r["task"].rsplit("_", 2)[0] in ("rabbit", "dog", "cat", "mammal (species unspecified)")
    ]
    rodent = [
        r["rmse"]
        for r in endpoints
        if r["task"].rsplit("_", 2)[0] in ("mouse", "rat", "guinea pig")
    ]
    oral = [r["rmse"] for r in endpoints if r["task"].rsplit("_", 2)[1] == "oral"]
    return {
        "LD50_37": float(np.mean(ld50)) if ld50 else None,
        "LDLo_19": float(np.mean(ldlo)) if ldlo else None,
        "TDLo_3": float(np.mean(tdlo)) if tdlo else None,
        "Avian_5": float(np.mean(avian)) if avian else None,
        "NonRodent_21": float(np.mean(nonrod)) if nonrod else None,
        "Rodent_30": float(np.mean(rodent)) if rodent else None,
        "Oral_19": float(np.mean(oral)) if oral else None,
    }


def run_probe_v2(
    *,
    mv_cache_path: Path,
    views_root: Path,
    groups_path: Path,
    work_dir: Path,
    gpus: list[int],
    seed: int = 42,
    ref_runs_dir: Path | None = None,
) -> dict:
    runs_dir = work_dir / "runs"
    logs_dir = work_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Wave 1 (4 GPUs in parallel, ~22-28 min): Full V2 on FULL & P10 + 2 mechanism ablations
    wave1 = [
        {
            "run_id": f"FULL_MV_BIOPGT_V2_s{seed}",
            "method": "MV_BIOPGT_V2",
            "budget": "FULL",
            "seed": seed,
            "config": {"gamma": 0.50},
        },
        {
            "run_id": f"P10_MV_BIOPGT_V2_s{seed}",
            "method": "MV_BIOPGT_V2",
            "budget": "S21_P10",
            "seed": seed,
            "config": {"gamma": 0.50},
        },
        {
            "run_id": f"FULL_MV_BIOPGT_V2_NOMDROP_s{seed}",
            "method": "MV_BIOPGT_V2_NOMDROP",
            "budget": "FULL",
            "seed": seed,
            "config": {"gamma": 0.50, "aux_graph_weight": 0.0, "chem_weight_decay": 1e-3},
        },
        {
            "run_id": f"FULL_MV_BIOPGT_V2_ZSCORE_s{seed}",
            "method": "MV_BIOPGT_V2",
            "budget": "FULL",
            "seed": seed,
            "config": {
                "gamma": 0.50,
                "label_scaling": "per_task",
                "loss_type": "mse",
                "ordinal_dose": False,
            },
        },
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

    # Wave 2 (4 GPUs in parallel, ~8-12 min): Matched gamma=0.50 NOGRAPH & NOPROMPT ablations
    wave2 = [
        {
            "run_id": f"FULL_MV_BIOPGT_V2_NOGRAPH_s{seed}",
            "method": "MV_BIOPGT_V2_NOGRAPH",
            "budget": "FULL",
            "seed": seed,
            "config": {"gamma": 0.50},
        },
        {
            "run_id": f"FULL_MV_BIOPGT_V2_NOPROMPT_s{seed}",
            "method": "MV_BIOPGT_V2_NOPROMPT",
            "budget": "FULL",
            "seed": seed,
            "config": {"gamma": 0.50},
        },
        {
            "run_id": f"P10_MV_BIOPGT_V2_NOGRAPH_s{seed}",
            "method": "MV_BIOPGT_V2_NOGRAPH",
            "budget": "S21_P10",
            "seed": seed,
            "config": {"gamma": 0.50},
        },
        {
            "run_id": f"P10_MV_BIOPGT_V2_NOPROMPT_s{seed}",
            "method": "MV_BIOPGT_V2_NOPROMPT",
            "budget": "S21_P10",
            "seed": seed,
            "config": {"gamma": 0.50},
        },
    ]
    _dispatch_jobs(
        wave2,
        mv_cache_path=mv_cache_path,
        views_root=views_root,
        groups_path=groups_path,
        runs_dir=runs_dir,
        logs_dir=logs_dir,
        gpus=gpus,
    )
    return summarize_probe_v2(work_dir=work_dir, groups_path=groups_path, ref_runs_dir=ref_runs_dir)


def summarize_probe_v2(
    *,
    work_dir: Path,
    groups_path: Path,
    ref_runs_dir: Path | None = None,
) -> dict:
    runs_dir = work_dir / "runs"
    rows = []
    by_id = {}

    run_dirs = [p.parent for p in sorted(runs_dir.glob("*/summary.json"))]
    if ref_runs_dir is not None and ref_runs_dir.is_dir():
        for ref_name in (
            "FULL_TOX_AUTHOR_s42",
            "FULL_MV_BIOPGT_GINE_g050_s42",
            "P10_TOX_AUTHOR_s42",
            "P10_MV_BIOPGT_BEST_s42",
        ):
            cand = ref_runs_dir / ref_name
            if (cand / "summary.json").is_file() and (cand / "test_metrics.json").is_file():
                run_dirs.append(cand)

    for rdir in run_dirs:
        s = read_json(rdir / "summary.json")
        tm = read_json(rdir / "test_metrics.json")
        rid = rdir.name
        by_id[rid] = (s, tm)
        budget = "FULL" if rid.startswith("FULL_") else "S21_P10"
        tg = s["test_groups"]
        vg = s["validation_groups"]
        sub = _subdomain_metrics(tm)
        rows.append(
            {
                "run_id": rid,
                "budget": budget,
                "method": s["method"],
                "gamma": s.get("gamma"),
                "label_scaling": s.get("label_scaling", "per_task"),
                "loss_type": s.get("loss_type", "mse"),
                "best_epoch": s["best_epoch"],
                "seconds": s["elapsed_seconds"],
                "params": (s.get("parameters") or {}).get("total"),
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
                "ordinal_ladder": (
                    "|".join(str(x) for x in s["ordinal_ladder"])
                    if s.get("ordinal_ladder")
                    else ""
                ),
            }
        )

    csv_path = work_dir / "probe_v2_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def rel_pct(cand: float | None, ref: float | None) -> float | None:
        if cand is None or ref is None or ref == 0:
            return None
        return round(100.0 * (ref - cand) / ref, 3)

    comparisons = {}
    if "FULL_MV_BIOPGT_V2_s42" in by_id:
        v2_s, v2_tm = by_id["FULL_MV_BIOPGT_V2_s42"]
        v2_sub = _subdomain_metrics(v2_tm)
        comparisons["FULL_V2_ordinal_ladder"] = v2_s.get("ordinal_ladder")
        if "FULL_MV_BIOPGT_V2_NOGRAPH_s42" in by_id:
            ng_s, _ = by_id["FULL_MV_BIOPGT_V2_NOGRAPH_s42"]
            comparisons["FULL_vs_V2_NOGRAPH_pct"] = rel_pct(
                v2_s["test_groups"]["ALL59"]["macro_rmse"],
                ng_s["test_groups"]["ALL59"]["macro_rmse"],
            )
        if "FULL_MV_BIOPGT_V2_NOPROMPT_s42" in by_id:
            np_s, _ = by_id["FULL_MV_BIOPGT_V2_NOPROMPT_s42"]
            comparisons["FULL_vs_V2_NOPROMPT_pct"] = rel_pct(
                v2_s["test_groups"]["ALL59"]["macro_rmse"],
                np_s["test_groups"]["ALL59"]["macro_rmse"],
            )
        if "FULL_MV_BIOPGT_V2_NOMDROP_s42" in by_id:
            nm_s, _ = by_id["FULL_MV_BIOPGT_V2_NOMDROP_s42"]
            comparisons["FULL_vs_V2_NOMDROP_pct"] = rel_pct(
                v2_s["test_groups"]["ALL59"]["macro_rmse"],
                nm_s["test_groups"]["ALL59"]["macro_rmse"],
            )
        if "FULL_MV_BIOPGT_V2_ZSCORE_s42" in by_id:
            zs_s, _ = by_id["FULL_MV_BIOPGT_V2_ZSCORE_s42"]
            comparisons["FULL_vs_V2_ZSCORE_pct"] = rel_pct(
                v2_s["test_groups"]["ALL59"]["macro_rmse"],
                zs_s["test_groups"]["ALL59"]["macro_rmse"],
            )
        if "FULL_TOX_AUTHOR_s42" in by_id:
            tox_s, tox_tm = by_id["FULL_TOX_AUTHOR_s42"]
            tox_sub = _subdomain_metrics(tox_tm)
            comparisons["FULL_vs_ToxACoL_ALL59_pct"] = rel_pct(
                v2_s["test_groups"]["ALL59"]["macro_rmse"],
                tox_s["test_groups"]["ALL59"]["macro_rmse"],
            )
            comparisons["FULL_vs_ToxACoL_LD50_pct"] = rel_pct(
                v2_sub["LD50_37"], tox_sub["LD50_37"]
            )
            comparisons["FULL_vs_ToxACoL_LDLo_pct"] = rel_pct(
                v2_sub["LDLo_19"], tox_sub["LDLo_19"]
            )
            comparisons["FULL_vs_ToxACoL_TDLo_pct"] = rel_pct(
                v2_sub["TDLo_3"], tox_sub["TDLo_3"]
            )
            comparisons["FULL_endpoint_wins_vs_ToxACoL"] = int(
                sum(
                    a["rmse"] < b["rmse"]
                    for a, b in zip(v2_tm["endpoints"], tox_tm["endpoints"])
                )
            )

    if "P10_MV_BIOPGT_V2_s42" in by_id and "P10_TOX_AUTHOR_s42" in by_id:
        v2_p10, _ = by_id["P10_MV_BIOPGT_V2_s42"]
        tox_p10, _ = by_id["P10_TOX_AUTHOR_s42"]
        comparisons["P10_vs_ToxACoL_ALL59_pct"] = rel_pct(
            v2_p10["test_groups"]["ALL59"]["macro_rmse"],
            tox_p10["test_groups"]["ALL59"]["macro_rmse"],
        )

    report_obj = {
        "code_commit": git_identity(),
        "comparisons": comparisons,
        "runs": rows,
    }
    write_json(work_dir / "probe_v2_decision.json", report_obj)

    lines = [
        "# S0 Single-Model V2 Probe Report (Physical Ordinal Offset + ModalDrop)",
        "",
        f"- Commit: `{report_obj['code_commit']}`",
        f"- Key comparisons: `{comparisons}`",
        "",
        "| Run ID | Budget | Val ALL59 | Test ALL59 | R2 | LD50(37) | LDLo(19) | TDLo(3) | Avian(5) | NonRod(21) | Oral(19) | Ep | Time(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["budget"], x["test_ALL59_rmse"])):
        fmt = lambda v: "-" if v is None else f"{v:.4f}"
        lines.append(
            f"| {r['run_id']} | {r['budget']} | {r['val_ALL59_rmse']:.4f} | "
            f"{r['test_ALL59_rmse']:.4f} | {r['test_ALL59_r2']:.4f} | "
            f"{fmt(r['test_LD50_rmse'])} | {fmt(r['test_LDLo_rmse'])} | {fmt(r['test_TDLo_rmse'])} | "
            f"{fmt(r['test_Avian_rmse'])} | {fmt(r['test_NonRodent_rmse'])} | {fmt(r['test_Oral_rmse'])} | "
            f"{r['best_epoch']} | {r['seconds']:.1f} |"
        )
    (work_dir / "probe_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report_obj


def collect_probe_v2(work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = work_dir / f"PGTV2_MV_V2_S0_RESULTS_{stamp}.zip"
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
    print(f"[probe_v2] Packaged {archive} ({archive.stat().st_size / 1e6:.2f} MB) sha256={sha}", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--mv-cache", required=True)
    p_pre.add_argument("--view", required=True)
    p_pre.add_argument("--out", required=True)
    p_pre.add_argument("--device", default="cuda:0")

    p_run = sub.add_parser("run-probe")
    p_run.add_argument("--mv-cache", required=True)
    p_run.add_argument("--views-root", required=True)
    p_run.add_argument("--groups", required=True)
    p_run.add_argument("--work", required=True)
    p_run.add_argument("--gpus", type=int, nargs="+", required=True)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--ref-runs-dir", default=None)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--work", required=True)
    p_sum.add_argument("--groups", required=True)
    p_sum.add_argument("--ref-runs-dir", default=None)

    p_col = sub.add_parser("collect")
    p_col.add_argument("--work", required=True)

    args = parser.parse_args()
    ref_dir = Path(args.ref_runs_dir) if getattr(args, "ref_runs_dir", None) else None
    if args.cmd == "preflight":
        run_preflight_v2(Path(args.mv_cache), Path(args.view), Path(args.out), args.device)
    elif args.cmd == "run-probe":
        run_probe_v2(
            mv_cache_path=Path(args.mv_cache),
            views_root=Path(args.views_root),
            groups_path=Path(args.groups),
            work_dir=Path(args.work),
            gpus=args.gpus,
            seed=args.seed,
            ref_runs_dir=ref_dir,
        )
    elif args.cmd == "summarize":
        summarize_probe_v2(
            work_dir=Path(args.work),
            groups_path=Path(args.groups),
            ref_runs_dir=ref_dir,
        )
    elif args.cmd == "collect":
        collect_probe_v2(Path(args.work))


if __name__ == "__main__":
    main()
