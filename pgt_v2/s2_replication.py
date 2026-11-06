"""Six frozen S2 fits, paired prediction exports, and the registered explanation panel."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import platform
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from .artifacts import load_arrays, load_manifest, read_json, write_json
from .cache import GraphCache, file_sha256
from .chemistry import digest
from .config import git_identity
from .evaluation import sealed_predict, score_prediction
from .training import run_training


METHODS = ("M2", "N1S")
SEEDS = (42, 43, 44)


def unit(method, seed, fold="S2"):
    return f"{fold}_{method}_S{seed}"


def run_name(a, method, seed):
    return unit(method, seed, load_manifest(a.release/"view")["fold"])


def csv_write(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def release_check(release):
    """WSL uses only the delivered JSON files; no cache, fitting, or prediction."""
    info = read_json(release/"release.json")
    if info["code_commit"] != git_identity():
        raise ValueError("Use the full commit in release/COMMIT")
    panel = read_json(release/"explanation_panel.json")
    view = load_manifest(release/"view")
    if (info["new_fits"] != 6 or view["fold"] not in ("S1","S2","S3","S4") or
            len(panel["panel"]) != panel["expected_panel"]["records"] or len(view["task_names"]) != 59):
        raise ValueError("Unexpected scaffold matrix or panel")
    if view["manifest_sha256"] != info["view_sha256"] or panel["view_sha256"] != info["view_sha256"]:
        raise ValueError("Release and panel use different views")
    expected = [unit(m,s,view["fold"]) for s in SEEDS for m in METHODS]
    if info["run_ids"] != expected:
        raise ValueError("Expected two models and three seeds")
    for name in expected:
        run = read_json(release/"runs"/(name+".json"))
        if (run["run_id"] != name or run["code_commit"] != info["code_commit"] or
                run["view_sha256"] != info["view_sha256"] or run["config"] != info["config"] or
                digest({k:v for k,v in run.items() if k != "run_sha256"}) != run["run_sha256"]):
            raise ValueError("Run differs from the frozen release: "+name)
    return info


def completed(a, method, seed):
    name = run_name(a,method,seed)
    result = read_json(a.work/"results"/name/"training/summary.json")
    expected = read_json(a.release/"runs"/(name+".json"))
    if result["terminal"] != "COMPLETE" or result["identity"]["run_sha256"] != expected["run_sha256"]:
        raise ValueError("Fit is incomplete or uses another configuration: "+name)
    return result


def preflight(a, info):
    view = load_manifest(a.release/"view")
    cache = GraphCache(a.cache,source_sha256=view["source_sha256"],verify_database=False)
    if cache.manifest["cache_id"] != view["cache_id"]:
        raise ValueError("Use the accepted G01R1 graph cache")
    train = load_arrays(a.release/"view","train.npz",view)
    stats = load_arrays(a.release/"view","statistics.npz",view)
    counts = np.isfinite(train["labels"]).sum(0)
    if not np.array_equal(counts,stats["count"]):
        raise ValueError("S2 training supervision and statistics differ")
    selection = [view["task_names"].index(t) for t in info["config"]["selection_tasks"]]
    registered = next(r for r in info["counts"] if r["role"] == "train")
    if (len(train["record_id"]),int(counts.sum()),int(counts[selection].sum())) != (
            registered["records"],registered["labels"],registered["S21_labels"]):
        raise ValueError("Unexpected training population")
    graph,_ = cache.read(int(train["cache_offset"][0]))
    if graph.sample_id != train["record_id"][0]:
        raise ValueError("Cache record differs from the frozen view")
    torch.zeros(1,device=a.device)
    report = dict(status="SERVER_INPUTS_READY",host=platform.node(),code_commit=git_identity(),
        device=a.device,cuda_name=torch.cuda.get_device_name(0),training_rows=len(train["record_id"]),
        visible_labels=int(counts.sum()),s21_labels=int(counts[selection].sum()),optimizer_steps=0)
    write_json(a.work/"operations/server/input_check.json",report)
    return report


def fit(a, info):
    name = run_name(a,a.method,a.seed)
    out = a.work/"results"/name/"training"
    if (out/"summary.json").exists() and read_json(out/"summary.json")["terminal"] == "COMPLETE":
        result = completed(a,a.method,a.seed)
        return dict(status="REUSED_COMPLETE",unit=name,best_epoch=result["best_epoch"])
    run_training(a.release/"runs"/(name+".json"),a.release/"view",a.cache,out,
        device=a.device,resume=(out/"last.pt").exists(),verify_database=False)
    result = completed(a,a.method,a.seed)
    return dict(status="FIT_COMPLETE",unit=name,best_epoch=result["best_epoch"])


def evaluate(a, info):
    result = completed(a,a.method,a.seed)
    name = run_name(a,a.method,a.seed)
    out = a.work/"results"/name
    for role in ("validation","test"):
        predictions, report = out/role, out/(role+"_metrics.json")
        if not (predictions/"manifest.json").exists():
            sealed_predict(out/"training/best.pt",a.release/"view",a.cache,predictions,
                role=role,device=a.device,verify_database=False)
        if not report.exists():
            score_prediction(predictions,a.release/"view",report,expected_checkpoint_sha256=result["best_sha256"])
        saved = read_json(report)
        if saved["checkpoint_sha256"] != result["best_sha256"]:
            raise ValueError("Prediction uses another checkpoint")
        if role == "validation" and abs(saved["macro_rmse"]-result["validation"]["macro_rmse"]) > 1e-5:
            raise ValueError("Validation replay differs from the selected fit")
    write_json(out/"complete.json",dict(unit=name,checkpoint_sha256=result["best_sha256"],roles=["validation","test"]))
    return dict(status="EVALUATION_COMPLETE",unit=name)


def explain(a, info):
    from .i01 import check_registry, execute

    reg = read_json(a.panel or a.release/"explanation_panel.json")
    records = {point["record_id"] for point in reg["panel"]}
    reg["runs"] = []
    for seed in SEEDS:
        for method in METHODS:
            name = run_name(a,method,seed)
            result = completed(a,method,seed)
            root = a.work/"results"/name
            if read_json(root/"complete.json")["checkpoint_sha256"] != result["best_sha256"]:
                raise ValueError("Evaluate the frozen best checkpoint before explaining it")
            seal = load_manifest(root/"test")
            predictions = load_arrays(root/"test","predictions.npz",seal)
            original = {str(r):p.tolist() for r,p in zip(predictions["record_id"],predictions["prediction"]) if r in records}
            if set(original) != records:
                raise ValueError("The preselected S2 panel is not covered by test predictions")
            reg["runs"].append(dict(unit=name,method=method,seed=seed,
                run_sha256=result["identity"]["run_sha256"],training_commit=info["code_commit"],
                checkpoint_sha256=result["best_sha256"],original_predictions=original))
    check_registry(reg)
    registry_name = "explanation_registry.json" if a.explanation_name == "explanation" else a.explanation_name+"_registry.json"
    write_json(a.work/registry_name,reg)
    args = SimpleNamespace(view=a.release/"view",cache=a.cache,a00=a.work,h02=a.work,
                           out=a.work/a.explanation_name,device=a.device)
    execute(args,reg)
    result = read_json(args.out/"execution.json")
    if result["task_model_observations"] != 6*sum(len(p["observations"]) for p in reg["panel"]):
        raise ValueError("Expected all registered observations under six weights")
    return result


def collect(a, info):
    groups = read_json(a.release/"endpoint_groups.json")
    groups = {k:groups[k] for k in ("ALL59","S21","M27","L11","Human3","Original6")}
    rows, endpoint_rows, fits, predictions = [], [], [], []
    for seed in SEEDS:
        for method in METHODS:
            name = run_name(a,method,seed)
            root = a.work/"results"/name
            if (root/"training/summary.json").exists() and read_json(root/"training/summary.json")["terminal"] == "COMPLETE":
                completed(a,method,seed)
                fits.append(name)
            if (root/"complete.json").exists():
                predictions.append(name)
            for role in ("validation","test"):
                file = root/(role+"_metrics.json")
                if not file.exists():
                    continue
                endpoints = {r["task"]:r for r in read_json(file)["endpoints"]}
                for task,r in endpoints.items():
                    endpoint_rows.append(dict(method=method,seed=seed,role=role,task=task,
                        rmse=r["rmse"],mae=r["mae"],r2=r["r2"]))
                for group,tasks in groups.items():
                    vals = [endpoints[t] for t in tasks]
                    row = dict(method=method,seed=seed,role=role,group=group,tasks=len(vals))
                    for metric in ("rmse","mae","r2"):
                        usable = [r[metric] for r in vals if r[metric] is not None]
                        row["macro_"+metric] = float(np.mean(usable)) if usable else None
                        row[metric+"_tasks"] = len(usable)
                    rows.append(row)
    csv_write(a.work/"group_metrics.csv",rows,["method","seed","role","group","tasks",
        "macro_rmse","rmse_tasks","macro_mae","mae_tasks","macro_r2","r2_tasks"])
    csv_write(a.work/"endpoint_metrics.csv",endpoint_rows,["method","seed","role","task","rmse","mae","r2"])
    averages = []
    for method,role,group in sorted({(r["method"],r["role"],r["group"]) for r in rows}):
        members = [r for r in rows if (r["method"],r["role"],r["group"]) == (method,role,group)]
        row = dict(method=method,role=role,group=group,completed_seeds=len(members))
        for metric in ("rmse","mae","r2"):
            values = [r["macro_"+metric] for r in members if r["macro_"+metric] is not None]
            row[metric+"_mean"] = float(np.mean(values)) if values else None
            row[metric+"_sd"] = float(np.std(values,ddof=1)) if len(values)>1 else None
        averages.append(row)
    csv_write(a.work/"group_seed_summary.csv",averages,["method","role","group","completed_seeds",
        "rmse_mean","rmse_sd","mae_mean","mae_sd","r2_mean","r2_sd"])
    explanation_names = info.get("required_explanations",["explanation"])
    explanation_done = all((a.work/n/"execution.json").exists() and
        read_json(a.work/n/"execution.json")["status"] == "COMPLETE" for n in explanation_names)
    fold = load_manifest(a.release/"view")["fold"]
    receipt = dict(experiment=fold,code_commit=info["code_commit"],new_fits=len(fits),
        completed_fits=fits,evaluated_runs=predictions,explanation_complete=explanation_done,
        status="COMPLETE" if len(predictions)==6 and explanation_done else "PARTIAL",scientific_review="PENDING")
    write_json(a.work/"return_receipt.json",receipt)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = a.work/f"PGTV2_{fold}_RESULTS_{stamp}.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for name in ("return_receipt.json","group_metrics.csv","endpoint_metrics.csv","group_seed_summary.csv","explanation_registry.json")+tuple(
                n+"_registry.json" for n in explanation_names if n != "explanation"):
            if (a.work/name).exists():
                package.write(a.work/name,name)
        for name,folder in [("release",a.release)]+[(n,a.work/n) for n in ["results","operations"]+explanation_names]:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and not path.name.startswith("last.pt") and "__pycache__" not in path.parts:
                    package.write(path,(Path(name)/path.relative_to(folder)).as_posix())
    identity = dict(file=archive.name,bytes=archive.stat().st_size,sha256=file_sha256(archive),**receipt)
    archive.with_name(archive.name+".sha256").write_text(identity["sha256"]+"  "+archive.name+"\n",encoding="utf-8")
    write_json(a.work/"return_package_identity.json",identity)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("check","preflight","fit","evaluate","explain","collect"))
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--work",type=Path)
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--panel",type=Path)
    parser.add_argument("--explanation-name",default="explanation")
    a = parser.parse_args()
    torch.set_num_threads(1)
    info = release_check(a.release)
    if a.action == "check":
        print(dict(status="RELEASE_CHECK_PASS",new_fits=6,explanation_observations=info["explanation_observations"],optimizer_steps=0),flush=True)
        return
    if a.work is None:
        parser.error("--work is required for execution")
    a.work.mkdir(parents=True,exist_ok=True)
    print(globals()[a.action](a,info),flush=True)


if __name__ == "__main__":
    main()
