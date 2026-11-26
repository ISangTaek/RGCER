"""Fast, leak-free training, tempered multi-task loss, and full group evaluation."""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch

from baselines.models.toxacol import toxacol_learning_rate
from ..artifacts import load_arrays, load_manifest, write_json
from ..cache import file_sha256
from ..training import metrics as base_metrics
from .features import MVDataStore, fit_descriptor_scaler, transform_descriptors
from .models import build_mv_model, count_parameters, method_input_needs

HUMAN3_TASKS = ("man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo")
ORIGINAL6_TASKS = (
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
    "mammal (species unspecified)_subcutaneous_LD50",
    "frog_subcutaneous_LDLo",
    "guinea pig_intravenous_LDLo",
)


def seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def author_lr(epoch: int) -> float:
    return toxacol_learning_rate(
        epoch,
        [20, 40, 50, 60, 70, 80, 100],
        [0.001, 0.0006, 0.00012, 0.000024, 0.000012, 0.000001, 0.0000001, 0.00000001],
    )


def cosine_warmup_lr(
    epoch: int, *, max_epochs: int, base_lr: float, warmup_epochs: int = 5, min_lr: float = 1e-5
) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def endpoint_weights_for_gamma(counts: np.ndarray, gamma: float) -> np.ndarray:
    """Compute per-endpoint weights w_t proportional to n_t^(-gamma).

    - gamma = 0.0: exact observed-label equal weighting (all labels weight 1)
    - gamma = 1.0: exact equal-endpoint weighting (sum_i w_t = const per endpoint)
    - 0 < gamma < 1: smooth interpolation balancing stable encoder gradients and sparse tasks
    Normalized so sum_t (n_t * w_t) = sum_t n_t, preserving loss scale across gammas.
    """
    c = np.asarray(counts, dtype=np.float64)
    if np.any(c <= 0):
        raise ValueError("All endpoints must have positive visible training counts")
    raw = c ** (-float(gamma))
    norm = float(c.sum()) / float(np.sum(c * raw))
    return (raw * norm).astype(np.float32)


def tempered_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    endpoint_weights: torch.Tensor,
) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    sq_err = (prediction - safe_target).square()
    weighted = sq_err * endpoint_weights[None, :]
    weight_sum = (mask.to(prediction.dtype) * endpoint_weights[None, :]).sum().clamp_min(1e-8)
    return weighted.masked_fill(~mask, 0.0).sum() / weight_sum


def tempered_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    endpoint_weights: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Tempered SmoothL1/Huber loss scaled by 2*beta so |err|<=beta equals exact MSE."""
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    elem = 2.0 * float(beta) * torch.nn.functional.smooth_l1_loss(
        prediction, safe_target, reduction="none", beta=float(beta)
    )
    weighted = elem * endpoint_weights[None, :]
    weight_sum = (mask.to(prediction.dtype) * endpoint_weights[None, :]).sum().clamp_min(1e-8)
    return weighted.masked_fill(~mask, 0.0).sum() / weight_sum


def compute_group_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    tasks: list[str] | tuple[str, ...],
    endpoint_groups: dict,
    groups: np.ndarray | None = None,
) -> dict:
    report = base_metrics(truth, prediction, tasks, groups=groups)
    by_task = {row["task"]: row for row in report["endpoints"]}
    group_defs = {
        "ALL59": list(tasks),
        "L11": list(endpoint_groups["L11"]),
        "M27": list(endpoint_groups["M27"]),
        "S21": list(endpoint_groups["S21"]),
        "Human3": [t for t in HUMAN3_TASKS if t in by_task],
        "Original6": [t for t in ORIGINAL6_TASKS if t in by_task],
    }
    summary = {}
    for name, members in group_defs.items():
        rows = [by_task[t] for t in members]
        r2_vals = [r["r2"] for r in rows if r["r2"] is not None]
        summary[name] = {
            "tasks": len(rows),
            "macro_rmse": float(np.mean([r["rmse"] for r in rows])),
            "macro_mae": float(np.mean([r["mae"] for r in rows])),
            "macro_r2": float(np.mean(r2_vals)) if r2_vals else None,
            "positive_r2_count": int(sum(v > 0 for v in r2_vals)),
        }
    report["groups"] = summary
    return report


def _partition_indices(indices: list[int], batch_size: int, *, drop_single_tail: bool) -> list[list[int]]:
    parts = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    if drop_single_tail and len(parts) >= 2 and len(parts[-1]) == 1:
        parts[-2].extend(parts.pop())
    return parts


@torch.no_grad()
def predict_neural(
    model: torch.nn.Module,
    store: MVDataStore,
    offsets: np.ndarray,
    *,
    desc_array: np.ndarray | None,
    needs: dict,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    model.eval()
    n = len(offsets)
    pieces = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_desc = desc_array[start:end] if desc_array is not None else None
        payload = store.build_batch(
            offsets[start:end],
            desc_array=batch_desc,
            need_avalon=needs["need_avalon"],
            need_morgan=needs["need_morgan"],
            need_desc=needs["need_desc"],
            graph_mode=needs["graph_mode"],
            device=device,
        )
        out = model(payload).detach().cpu().numpy() * scale[None, :] + mean[None, :]
        if not np.isfinite(out).all():
            raise FloatingPointError("Nonfinite prediction encountered")
        pieces.append(out.astype(np.float32))
    return np.concatenate(pieces, axis=0)


def _train_rf(
    store: MVDataStore,
    view_root: Path,
    endpoint_groups: dict,
    output_dir: Path,
    *,
    seed: int,
    config: dict,
) -> dict:
    started = time.perf_counter()
    manifest = load_manifest(view_root)
    tasks = list(manifest["task_names"])
    train_arr = load_arrays(view_root, "train.npz", manifest)
    val_arr = load_arrays(view_root, "validation.npz", manifest)
    test_arr = load_arrays(view_root, "test.npz", manifest)
    for arr in (train_arr, val_arr, test_arr):
        store.verify_partition(arr, manifest)

    train_off = train_arr["cache_offset"].astype(np.int64)
    val_off = val_arr["cache_offset"].astype(np.int64)
    test_off = test_arr["cache_offset"].astype(np.int64)
    train_y = train_arr["labels"].astype(np.float64)
    active_mask = np.isfinite(train_y).any(axis=1)
    scaler = fit_descriptor_scaler(store.desc_raw, store.desc_missing, train_off[active_mask])

    x_train = np.concatenate(
        [store.unpack_morgan(train_off), transform_descriptors(store.desc_raw, store.desc_missing, train_off, scaler)],
        axis=1,
    )
    x_val = np.concatenate(
        [store.unpack_morgan(val_off), transform_descriptors(store.desc_raw, store.desc_missing, val_off, scaler)],
        axis=1,
    )
    x_test = np.concatenate(
        [store.unpack_morgan(test_off), transform_descriptors(store.desc_raw, store.desc_missing, test_off, scaler)],
        axis=1,
    )

    n_estimators = int(config.get("n_estimators", 300))
    max_features = config.get("max_features", "sqrt")
    min_samples_leaf = int(config.get("min_samples_leaf", 2))
    n_jobs = int(config.get("n_jobs", 8))

    val_pred = np.zeros((len(val_off), len(tasks)), dtype=np.float32)
    test_pred = np.zeros((len(test_off), len(tasks)), dtype=np.float32)
    for j in range(len(tasks)):
        mask = np.isfinite(train_y[:, j])
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=seed + j,
            n_jobs=n_jobs,
        )
        rf.fit(x_train[mask], train_y[mask, j])
        val_pred[:, j] = rf.predict(x_val).astype(np.float32)
        test_pred[:, j] = rf.predict(x_test).astype(np.float32)

    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        record_id=val_arr["record_id"],
        prediction=val_pred,
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        record_id=test_arr["record_id"],
        prediction=test_pred,
    )
    val_report = compute_group_metrics(
        val_arr["labels"], val_pred, tasks, endpoint_groups, groups=val_arr["group_id"]
    )
    test_report = compute_group_metrics(
        test_arr["labels"], test_pred, tasks, endpoint_groups, groups=test_arr["group_id"]
    )
    write_json(output_dir / "validation_metrics.json", val_report)
    write_json(output_dir / "test_metrics.json", test_report)
    summary = {
        "status": "COMPLETE",
        "method": "RF",
        "seed": seed,
        "fold": manifest["fold"],
        "view_id": manifest["view_id"],
        "view_sha256": manifest["manifest_sha256"],
        "config": config,
        "parameters": {"total": None, "trainable": None, "kind": "per_endpoint_rf"},
        "best_epoch": 1,
        "epochs_run": 1,
        "elapsed_seconds": round(elapsed, 2),
        "validation_groups": val_report["groups"],
        "test_groups": test_report["groups"],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def train_and_evaluate(
    *,
    store: MVDataStore,
    view_root: str | Path,
    endpoint_groups: dict,
    method: str,
    seed: int,
    config: dict,
    output_dir: str | Path,
    device: str | torch.device = "cuda:0",
    save_checkpoint_file: bool = True,
) -> dict:
    """Train one model on train.npz, select epoch on validation.npz, evaluate sealed test.npz."""
    output_dir = Path(output_dir)
    if (output_dir / "summary.json").is_file():
        existing = load_existing_summary(output_dir / "summary.json")
        if existing and existing.get("status") == "COMPLETE":
            return existing

    view_root = Path(view_root)
    if method == "RF":
        return _train_rf(
            store, view_root, endpoint_groups, output_dir, seed=seed, config=config
        )

    seed_all(seed)
    started = time.perf_counter()
    manifest = load_manifest(view_root)
    tasks = list(manifest["task_names"])
    train_arr = load_arrays(view_root, "train.npz", manifest)
    val_arr = load_arrays(view_root, "validation.npz", manifest)
    test_arr = load_arrays(view_root, "test.npz", manifest)
    stats_arr = load_arrays(view_root, "statistics.npz", manifest)
    for arr in (train_arr, val_arr, test_arr):
        store.verify_partition(arr, manifest)

    train_off = train_arr["cache_offset"].astype(np.int64)
    val_off = val_arr["cache_offset"].astype(np.int64)
    test_off = test_arr["cache_offset"].astype(np.int64)
    train_y = train_arr["labels"].astype(np.float64)
    active_idx = np.where(np.isfinite(train_y).any(axis=1))[0].tolist()
    counts = stats_arr["count"].astype(np.float64)
    if not np.array_equal(np.isfinite(train_y).sum(axis=0), counts.astype(np.int64)):
        raise ValueError("Training visible label count mismatch with statistics.npz")

    needs = method_input_needs(method)
    desc_scaler = None
    train_desc = val_desc = test_desc = None
    if needs["need_desc"]:
        desc_scaler = fit_descriptor_scaler(
            store.desc_raw, store.desc_missing, train_off[active_idx], clip_value=5.0
        )
        train_desc = transform_descriptors(store.desc_raw, store.desc_missing, train_off, desc_scaler)
        val_desc = transform_descriptors(store.desc_raw, store.desc_missing, val_off, desc_scaler)
        test_desc = transform_descriptors(store.desc_raw, store.desc_missing, test_off, desc_scaler)

    author = method == "TOX_AUTHOR"
    is_v2 = method.startswith("MV_BIOPGT_V2")
    is_final = method.startswith("MV_BIOPGT_FINAL")
    label_scaling = str(
        config.get("label_scaling", "raw" if author else ("global_center" if is_v2 else "per_task"))
    )
    if label_scaling == "raw":
        mean = np.zeros_like(stats_arr["mean"], dtype=np.float64)
        scale = np.ones_like(stats_arr["scale"], dtype=np.float64)
    elif label_scaling == "global_center":
        global_mean = float(np.nanmean(train_y))
        mean = np.full_like(stats_arr["mean"], global_mean, dtype=np.float64)
        scale = np.ones_like(stats_arr["scale"], dtype=np.float64)
    elif label_scaling == "per_task":
        mean = stats_arr["mean"].astype(np.float64)
        scale = stats_arr["scale"].astype(np.float64)
    else:
        raise ValueError(f"Unknown label_scaling: {label_scaling}")
    std_train_y = ((train_y - mean[None, :]) / scale[None, :]).astype(np.float32)

    dev = torch.device(device)
    model = build_mv_model(
        method,
        tuple(tasks),
        counts,
        adjacency=stats_arr["adjacency"],
        config=config,
        device=dev,
    )
    params = count_parameters(model)

    if author:
        max_epochs = int(config.get("max_epochs", 120))
        min_epochs = int(config.get("min_epochs", 120))
        patience = int(config.get("patience", 121))
        batch_size = int(config.get("batch_size", 32))
        eval_batch = int(config.get("eval_batch", 512))
        grad_clip = None
        gamma = 0.0
        loss_type = "mse"
        huber_beta = 1.0
        aux_graph_weight = 0.0
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config.get("lr", 0.001)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
            nesterov=True,
        )
    else:
        max_epochs = int(config.get("max_epochs", 90 if (is_v2 or is_final) else 100))
        min_epochs = int(config.get("min_epochs", 25))
        patience = int(config.get("patience", 18 if (is_v2 or is_final) else 20))
        default_bs = 64 if needs["graph_mode"] == "graphormer" else (128 if needs["graph_mode"] == "gine" else 256)
        batch_size = int(config.get("batch_size", default_bs))
        eval_batch = int(config.get("eval_batch", default_bs * 2))
        grad_clip = float(config.get("grad_clip", 5.0))
        gamma = float(config.get("gamma", 0.50 if (is_v2 or is_final) else 0.25))
        loss_type = str(config.get("loss_type", "huber" if is_v2 else "mse"))
        huber_beta = float(config.get("huber_beta", 1.0))
        default_aux_w = (
            0.25
            if method
            in (
                "MV_BIOPGT_V2",
                "MV_BIOPGT_V2_NOPROMPT",
                "MV_BIOPGT_FINAL",
                "MV_BIOPGT_FINAL_NOPROMPT",
            )
            else 0.0
        )
        aux_graph_weight = float(config.get("aux_graph_weight", default_aux_w))
        base_lr = float(
            config.get("lr", 5e-4 if needs["graph_mode"] == "graphormer" else 1e-3)
        )
        base_wd = float(config.get("weight_decay", 1e-3))
        default_chem_wd = (
            3e-3
            if (
                is_v2
                or method
                in ("MV_BIOPGT_FINAL", "MV_BIOPGT_FINAL_NOGRAPH", "MV_BIOPGT_FINAL_NOPROMPT")
            )
            else base_wd
        )
        chem_wd = float(config.get("chem_weight_decay", default_chem_wd))
        if abs(chem_wd - base_wd) > 1e-12 and getattr(model, "chem_encoder", None) is not None:
            chem_ids = {id(p) for p in model.chem_encoder.parameters()}
            param_groups = [
                {
                    "params": [
                        p for p in model.parameters() if p.requires_grad and id(p) not in chem_ids
                    ],
                    "weight_decay": base_wd,
                },
                {
                    "params": [
                        p for p in model.parameters() if p.requires_grad and id(p) in chem_ids
                    ],
                    "weight_decay": chem_wd,
                },
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=base_lr,
                weight_decay=base_wd,
            )

    ep_weights = torch.from_numpy(endpoint_weights_for_gamma(counts, gamma)).to(dev)

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    best_val_pred = None
    best_val_report = None
    stale = 0
    history = []

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for epoch in range(max_epochs):
        if epoch >= min_epochs and stale >= patience:
            break
        ep_t0 = time.perf_counter()
        model.train()
        if author:
            lr_now = author_lr(epoch)
        else:
            lr_now = cosine_warmup_lr(
                epoch,
                max_epochs=max_epochs,
                base_lr=base_lr,
                warmup_epochs=int(config.get("warmup_epochs", 5)),
            )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        perm = torch.randperm(len(active_idx)).tolist()
        order = [active_idx[i] for i in perm]
        batches = _partition_indices(order, batch_size, drop_single_tail=author)
        running_loss = 0.0
        seen_rows = 0

        for idx_batch in batches:
            optimizer.zero_grad(set_to_none=True)
            b_off = train_off[idx_batch]
            b_desc = train_desc[idx_batch] if train_desc is not None else None
            payload = store.build_batch(
                b_off,
                desc_array=b_desc,
                need_avalon=needs["need_avalon"],
                need_morgan=needs["need_morgan"],
                need_desc=needs["need_desc"],
                graph_mode=needs["graph_mode"],
                device=dev,
            )
            if aux_graph_weight > 0.0:
                pred, train_aux = model(payload, return_train_aux=True)
            else:
                pred = model(payload)
                train_aux = {}
            target = torch.from_numpy(std_train_y[idx_batch]).to(dev, non_blocking=True)
            if loss_type == "huber":
                loss = tempered_huber_loss(pred, target, ep_weights, beta=huber_beta)
                if aux_graph_weight > 0.0 and "graph_pred" in train_aux:
                    loss = loss + aux_graph_weight * tempered_huber_loss(
                        train_aux["graph_pred"], target, ep_weights, beta=huber_beta
                    )
            else:
                loss = tempered_mse_loss(pred, target, ep_weights)
                if aux_graph_weight > 0.0 and "graph_pred" in train_aux:
                    loss = loss + aux_graph_weight * tempered_mse_loss(
                        train_aux["graph_pred"], target, ep_weights
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}")
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.detach()) * len(idx_batch)
            seen_rows += len(idx_batch)

        val_pred = predict_neural(
            model,
            store,
            val_off,
            desc_array=val_desc,
            needs=needs,
            mean=mean,
            scale=scale,
            batch_size=eval_batch,
            device=dev,
        )
        val_report = compute_group_metrics(
            val_arr["labels"], val_pred, tasks, endpoint_groups, groups=val_arr["group_id"]
        )
        val_score = float(val_report["groups"]["ALL59"]["macro_rmse"])
        improved = val_score < best_score - 1e-7
        if improved:
            best_score = val_score
            best_epoch = epoch + 1
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_pred = val_pred.copy()
            best_val_report = val_report
        else:
            stale += 1

        history.append(
            {
                "epoch": epoch + 1,
                "lr": lr_now,
                "train_loss": running_loss / max(1, seen_rows),
                "val_all59_rmse": val_score,
                "val_all59_r2": val_report["groups"]["ALL59"]["macro_r2"],
                "val_s21_rmse": val_report["groups"]["S21"]["macro_rmse"],
                "best_epoch": best_epoch,
                "stale": stale,
                "seconds": round(time.perf_counter() - ep_t0, 2),
            }
        )

    if best_state is None:
        raise RuntimeError("No valid checkpoint selected during training")

    model.load_state_dict(best_state)
    test_pred = predict_neural(
        model,
        store,
        test_off,
        desc_array=test_desc,
        needs=needs,
        mean=mean,
        scale=scale,
        batch_size=eval_batch,
        device=dev,
    )
    test_report = compute_group_metrics(
        test_arr["labels"], test_pred, tasks, endpoint_groups, groups=test_arr["group_id"]
    )
    elapsed = time.perf_counter() - started
    peak_mem_mb = (
        round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 1)
        if dev.type == "cuda"
        else 0.0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        record_id=val_arr["record_id"],
        prediction=best_val_pred,
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        record_id=test_arr["record_id"],
        prediction=test_pred,
    )
    write_json(output_dir / "validation_metrics.json", best_val_report)
    write_json(output_dir / "test_metrics.json", test_report)
    write_json(output_dir / "history.json", history)

    ckpt_sha = None
    if save_checkpoint_file:
        ckpt_path = output_dir / "best.pt"
        torch.save(
            {
                "method": method,
                "seed": seed,
                "fold": manifest["fold"],
                "view_id": manifest["view_id"],
                "view_sha256": manifest["manifest_sha256"],
                "task_names": tasks,
                "config": config,
                "best_epoch": best_epoch,
                "model_state": best_state,
                "target_mean": mean,
                "target_scale": scale,
                "counts": counts,
                "adjacency": stats_arr["adjacency"],
                "desc_scaler": desc_scaler,
            },
            ckpt_path,
        )
        ckpt_sha = file_sha256(ckpt_path)

    ordinal_ladder = None
    if getattr(model, "ordinal_dose", False):
        ordinal_ladder = [
            round(float(x), 5) for x in model.ordinal_threshold_ladder().detach().cpu().tolist()
        ]

    summary = {
        "status": "COMPLETE",
        "method": method,
        "seed": seed,
        "fold": manifest["fold"],
        "view_id": manifest["view_id"],
        "view_sha256": manifest["manifest_sha256"],
        "config": config,
        "gamma": gamma,
        "label_scaling": label_scaling,
        "loss_type": loss_type,
        "aux_graph_weight": aux_graph_weight,
        "ordinal_ladder": ordinal_ladder,
        "parameters": params,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "elapsed_seconds": round(elapsed, 2),
        "peak_cuda_mb": peak_mem_mb,
        "checkpoint_sha256": ckpt_sha,
        "validation_groups": best_val_report["groups"],
        "test_groups": test_report["groups"],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def load_existing_summary(path: Path) -> dict | None:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
