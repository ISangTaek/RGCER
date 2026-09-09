"""Configuration-driven training core for formal and local smoke baseline runs."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.data import Batch

from .artifacts import endpoint_prediction_rows, model_sha256, run_manifest_payload, smoke_samples_payload, state_dict_sha256
from .checkpoint import StrictBest, protocol_identity, validate_checkpoint_contract
from .code_identity import implementation_identity
from .config import resolve_training_config
from .constants import DEFAULT_SEED, HUMAN3_TASKS, METHODS, TOXACUTE_TASKS
from .data import BaselineData, load_formal_train_validation, load_human3_smoke
from .features import afp_graph, afp_schema, avalon_matrix, morgan_matrix
from .inference import predict_partition
from .metrics import regression_metrics
from .models.afp import AttentiveFPRegressor
from .models.dmpnn import ChempropDMPNN, NoamLikeScheduler
from .models.grover import build_grover, grover_batch, optimizer_with_coverage
from .models.toxacol import ToxACoLNet, endpoint_feature_matrix, task_adjacency, toxacol_learning_rate
from .scaling import TaskScaler
from .utils import environment_snapshot, require_nonexistent_output, set_global_seed, sha256_file, write_json


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    denominator = mask.sum()
    if denominator.item() <= 0:
        return None
    return (((prediction - target) ** 2) * mask).sum() / denominator


def _batches(length: int, batch_size: int, *, shuffle: bool, seed: int):
    indices = np.arange(length)
    if shuffle:
        np.random.RandomState(seed).shuffle(indices)
    for start in range(0, length, batch_size):
        yield indices[start:start + batch_size]


def _checkpoint_payload(method: str, model, config: dict, scaler: TaskScaler, epoch: int, metric: float) -> dict:
    return {
        "format_version": 2,
        "method": method,
        "model_type": config["model_type"],
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler": scaler.to_dict(),
        "task_names": list(config["task_names"]),
        "feature_schema": config["feature_schema"],
        "seed": int(config["seed"]),
        "epoch": int(epoch),
        "best_validation_human3_macro_rmse": float(metric),
        **protocol_identity(),
    }


def _validation_metrics(data: BaselineData, predictions: np.ndarray) -> dict:
    return regression_metrics(data.validation.labels_human3, predictions, HUMAN3_TASKS)


def _run_rf(data, output: Path, config: dict, log: Callable[[str], None]):
    training = config["training"]
    scaler = TaskScaler.fit(data.train.labels_human3, HUMAN3_TASKS, allow_empty=False)
    train_x = morgan_matrix(data.train.smiles, include_chirality=True)
    validation_x = morgan_matrix(data.validation.smiles, include_chirality=True)
    models = []
    checkpoint = output / "checkpoint.joblib"
    for column, task in enumerate(HUMAN3_TASKS):
        mask = np.isfinite(data.train.labels_human3[:, column])
        model = RandomForestRegressor(
            n_estimators=int(training["n_estimators"]),
            max_depth=training["max_depth"],
            min_samples_leaf=int(training["min_samples_leaf"]),
            max_features=training["max_features"],
            n_jobs=int(training.get("n_jobs", 1)),
            random_state=config["seed"],
        )
        model.fit(train_x[mask], data.train.labels_human3[mask, column])
        models.append(model)
        log(f"fit task={task} n={int(mask.sum())} n_estimators={training['n_estimators']}")
    predictions = np.column_stack([model.predict(validation_x) for model in models])
    metrics = _validation_metrics(data, predictions)
    payload = {
        "format_version": 2,
        "method": "rf",
        "model_type": config["model_type"],
        "models": models,
        "config": config,
        "scaler": scaler.to_dict(),
        "task_names": list(config["task_names"]),
        "feature_schema": config["feature_schema"],
        "seed": config["seed"],
        "epoch": 0,
        "best_validation_human3_macro_rmse": metrics["macro_rmse"],
        **protocol_identity(),
    }
    joblib.dump(payload, checkpoint)
    validate_checkpoint_contract(joblib.load(checkpoint), expected_method="rf")
    return scaler, predictions, metrics, [{"epoch": 0, **metrics}], checkpoint, {
        "features": config["feature_schema"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_update": {
            "passed": all(len(model.estimators_) == int(training["n_estimators"]) for model in models),
            "evidence": f"three fitted forests each contain {training['n_estimators']} estimators",
        },
    }


def _afp_predict(model, records, scaler, device, batch_size):
    model.eval()
    rows = []
    with torch.no_grad():
        for indices in _batches(len(records), batch_size, shuffle=False, seed=0):
            batch = Batch.from_data_list([afp_graph(records[int(index)]) for index in indices]).to(device)
            rows.append(model(batch).cpu().numpy())
    return scaler.inverse_transform(np.concatenate(rows, axis=0))


def _run_afp(data, output, config, device, log):
    training = config["training"]
    scaler = TaskScaler.fit(data.train.labels_human3, HUMAN3_TASKS, allow_empty=False)
    train_y, train_mask = scaler.transform(data.train.labels_human3)
    schema = afp_schema()
    model = AttentiveFPRegressor(
        schema.atom_dim, schema.bond_dim, dropout=float(training["dropout"])
    ).to(device)
    initial_state_sha256 = model_sha256(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["lr"]), weight_decay=float(training["weight_decay"])
    )
    best = StrictBest(output / "checkpoint.pt")
    best_predictions = None
    history = []
    epochs_without_improvement = 0
    for epoch in range(int(training["epochs"])):
        model.train()
        losses = []
        for indices in _batches(len(data.train.sample_ids), int(training["batch_size"]), shuffle=True, seed=config["seed"] + epoch):
            graphs = [afp_graph(data.train.graph_records[int(index)], train_y[int(index)], train_mask[int(index)]) for index in indices]
            batch = Batch.from_data_list(graphs).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_mse(model(batch), batch.y, batch.label_mask)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        predictions = _afp_predict(model, data.validation.graph_records, scaler, device, int(training["batch_size"]))
        metrics = _validation_metrics(data, predictions)
        metric = metrics["macro_rmse"]
        improved = best.consider(metric, epoch, lambda path, e=epoch, m=metric: torch.save(
            _checkpoint_payload("afp", model, config, scaler, e, m), path
        ))
        if improved:
            best_predictions = predictions.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        log(f"epoch={epoch} train_loss={np.mean(losses):.6f} validation_macro_rmse={metric:.6f}")
        patience = training.get("patience")
        if patience is not None and epochs_without_improvement >= int(patience):
            log(f"early_stop epoch={epoch} patience={patience}")
            break
    payload = torch.load(best.path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(payload, expected_method="afp")
    metrics = _validation_metrics(data, best_predictions)
    return scaler, best_predictions, metrics, history, best.path, {
        "afp_schema_hash": schema.schema_hash,
        "atom_dim": schema.atom_dim,
        "bond_dim": schema.bond_dim,
        "training_update": {
            "passed": initial_state_sha256 != state_dict_sha256(payload["model_state_dict"]),
            "initial_state_sha256": initial_state_sha256,
            "best_state_sha256": state_dict_sha256(payload["model_state_dict"]),
        },
    }


def _smiles_predict(model, smiles, scaler, device, batch_size, forward):
    model.eval()
    rows = []
    with torch.no_grad():
        for indices in _batches(len(smiles), batch_size, shuffle=False, seed=0):
            batch_smiles = [smiles[int(index)] for index in indices]
            rows.append(forward(model, batch_smiles).detach().cpu().numpy())
    return scaler.inverse_transform(np.concatenate(rows, axis=0))


def _run_dmpnn(data, output, config, device, source_root, log):
    if source_root is None:
        raise ValueError("D-MPNN requires --source-root pointing to official Chemprop 1.6.1 source")
    training = config["training"]
    scaler = TaskScaler.fit(data.train.labels_human3, HUMAN3_TASKS, allow_empty=False)
    train_y, train_mask = scaler.transform(data.train.labels_human3)
    model = ChempropDMPNN(source_root, device, dropout=float(training["dropout"])).to(device)
    initial_state_sha256 = model_sha256(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["init_lr"]), weight_decay=float(training.get("weight_decay", 0.0))
    )
    steps_per_epoch = math.ceil(len(data.train.smiles) / int(training["batch_size"]))
    scheduler = NoamLikeScheduler(
        optimizer,
        warmup_epochs=float(training["warmup_epochs"]),
        total_epochs=int(training["epochs"]),
        steps_per_epoch=steps_per_epoch,
        init_lr=float(training["init_lr"]),
        max_lr=float(training["max_lr"]),
        final_lr=float(training["final_lr"]),
    )
    best = StrictBest(output / "checkpoint.pt")
    best_predictions = None
    history = []
    for epoch in range(int(training["epochs"])):
        model.train()
        losses = []
        for indices in _batches(len(data.train.smiles), int(training["batch_size"]), shuffle=True, seed=config["seed"] + epoch):
            target = torch.as_tensor(train_y[indices], device=device)
            mask = torch.as_tensor(train_mask[indices], device=device)
            smiles = [data.train.smiles[int(index)] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_mse(model(smiles), target, mask)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        predictions = _smiles_predict(model, data.validation.smiles, scaler, device, int(training["batch_size"]), lambda m, s: m(s))
        metrics = _validation_metrics(data, predictions)
        metric = metrics["macro_rmse"]
        if best.consider(metric, epoch, lambda path, e=epoch, m=metric: torch.save(_checkpoint_payload("dmpnn", model, config, scaler, e, m), path)):
            best_predictions = predictions.copy()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "last_lr": optimizer.param_groups[0]["lr"], **metrics})
        log(f"epoch={epoch} train_loss={np.mean(losses):.6f} validation_macro_rmse={metric:.6f}")
    payload = torch.load(best.path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(payload, expected_method="dmpnn")
    return scaler, best_predictions, _validation_metrics(data, best_predictions), history, best.path, {
        "chemprop_version": "1.6.1",
        "ffn_num_layers": 2,
        "ffn_hidden_size": 300,
        "training_update": {
            "passed": initial_state_sha256 != state_dict_sha256(payload["model_state_dict"]),
            "initial_state_sha256": initial_state_sha256,
            "best_state_sha256": state_dict_sha256(payload["model_state_dict"]),
        },
    }


def _run_grover(data, output, config, device, source_root, weights, log):
    if source_root is None or weights is None:
        raise ValueError("GROVER training requires official --source-root and --pretrained GROVER_base weights")
    training = config["training"]
    scaler = TaskScaler.fit(data.train.labels_human3, HUMAN3_TASKS, allow_empty=False)
    train_y, train_mask = scaler.transform(data.train.labels_human3)
    model, args, load_audit = build_grover(source_root, weights, device, dropout=float(training["dropout"]))
    initial_state_sha256 = model_sha256(model)
    optimizer, coverage = optimizer_with_coverage(
        model,
        init_lr=float(training["init_lr"]),
        weight_decay=float(training["weight_decay"]),
        fine_tune_coff=float(training.get("fine_tune_coff", 1.0)),
    )
    scheduler = NoamLikeScheduler(
        optimizer,
        warmup_epochs=float(training["warmup_epochs"]),
        total_epochs=int(training["epochs"]),
        steps_per_epoch=math.ceil(len(data.train.smiles) / int(training["batch_size"])),
        init_lr=float(training["init_lr"]),
        max_lr=float(training["max_lr"]),
        final_lr=float(training["final_lr"]),
    )
    best = StrictBest(output / "checkpoint.pt")
    best_predictions = None
    history = []
    forward = lambda m, smiles: m(grover_batch(smiles, args, device), [None] * len(smiles))
    for epoch in range(int(training["epochs"])):
        model.train()
        losses = []
        for indices in _batches(len(data.train.smiles), int(training["batch_size"]), shuffle=True, seed=config["seed"] + epoch):
            smiles = [data.train.smiles[int(index)] for index in indices]
            target = torch.as_tensor(train_y[indices], device=device)
            mask = torch.as_tensor(train_mask[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            atom_pred, bond_pred = forward(model, smiles)
            denominator = mask.sum()
            if denominator.item() <= 0:
                continue
            loss = ((((atom_pred - target) ** 2 + (bond_pred - target) ** 2 + 0.1 * (atom_pred - bond_pred) ** 2) * mask).sum() / denominator)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        predictions = _smiles_predict(model, data.validation.smiles, scaler, device, int(training["batch_size"]), forward)
        metrics = _validation_metrics(data, predictions)
        metric = metrics["macro_rmse"]
        if best.consider(metric, epoch, lambda path, e=epoch, m=metric: torch.save(_checkpoint_payload("grover", model, config, scaler, e, m), path)):
            best_predictions = predictions.copy()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        log(f"epoch={epoch} train_loss={np.mean(losses):.6f} validation_macro_rmse={metric:.6f}")
    payload = torch.load(best.path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(payload, expected_method="grover")
    return scaler, best_predictions, _validation_metrics(data, best_predictions), history, best.path, {
        **load_audit,
        "optimizer_coverage": coverage,
        "training_update": {
            "passed": initial_state_sha256 != state_dict_sha256(payload["model_state_dict"]),
            "initial_state_sha256": initial_state_sha256,
            "best_state_sha256": state_dict_sha256(payload["model_state_dict"]),
        },
    }


def _tox_predict(model, features, scaler, device):
    model.eval()
    with torch.no_grad():
        scaled = model(torch.as_tensor(features, device=device)).cpu().numpy()
    all_predictions = scaler.inverse_transform(scaled)
    columns = [TOXACUTE_TASKS.index(task) for task in HUMAN3_TASKS]
    return all_predictions[:, columns]


def _run_toxacol(data, output, config, device, log):
    training = config["training"]
    scaler = TaskScaler.fit(data.train.labels_all, TOXACUTE_TASKS, allow_empty=config["role"] == "SMOKE")
    train_y, train_mask = scaler.transform(data.train.labels_all)
    train_x = avalon_matrix(data.train.smiles)
    validation_x = avalon_matrix(data.validation.smiles)
    adjacency, graph_audit = task_adjacency(data.train.labels_all)
    endpoint_features, endpoint_schema = endpoint_feature_matrix()
    model = ToxACoLNet(adjacency, endpoint_features, dropout=float(training["dropout"])).to(device)
    initial_state_sha256 = model_sha256(model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["lr_lut"][0][1]),
        momentum=float(training["momentum"]),
        nesterov=bool(training["nesterov"]),
        weight_decay=float(training["weight_decay"]),
    )
    bounds = [int(row[0]) for row in training["lr_lut"][:-1]]
    values = [float(row[1]) for row in training["lr_lut"]]
    multiplier = float(training.get("lr_multiplier", 1.0))
    best = StrictBest(output / "checkpoint.pt")
    best_predictions = None
    history = []
    for epoch in range(int(training["epochs"])):
        lr = toxacol_learning_rate(epoch, bounds, values) * multiplier
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses = []
        for indices in _batches(len(train_x), int(training["batch_size"]), shuffle=True, seed=config["seed"] + epoch):
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.train(len(indices) > 1)
            x = torch.as_tensor(train_x[indices], device=device)
            target = torch.as_tensor(train_y[indices], device=device)
            mask = torch.as_tensor(train_mask[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_mse(model(x), target, mask)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        predictions = _tox_predict(model, validation_x, scaler, device)
        metrics = _validation_metrics(data, predictions)
        metric = metrics["macro_rmse"]
        if best.consider(metric, epoch, lambda path, e=epoch, m=metric: torch.save(_checkpoint_payload("toxacol", model, config, scaler, e, m), path)):
            best_predictions = predictions.copy()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": lr, **metrics})
        log(f"epoch={epoch} train_loss={np.mean(losses):.6f} validation_macro_rmse={metric:.6f}")
    payload = torch.load(best.path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(payload, expected_method="toxacol")
    return scaler, best_predictions, _validation_metrics(data, best_predictions), history, best.path, {
        "source_commit": ToxACoLNet.source_commit,
        "task_graph": graph_audit,
        "endpoint_schema": endpoint_schema,
        "empty_train_tasks": [TOXACUTE_TASKS[index] for index, count in enumerate(scaler.counts) if count == 0],
        "sampler": "seeded_random_permutation_without_replacement_full_dataset_drop_last_false",
        "upstream_epoch_size_metadata": training.get("epoch_size"),
        "upstream_epoch_size_effect": "reporting_only_not_sampling",
        "training_update": {
            "passed": initial_state_sha256 != state_dict_sha256(payload["model_state_dict"]),
            "initial_state_sha256": initial_state_sha256,
            "best_state_sha256": state_dict_sha256(payload["model_state_dict"]),
        },
    }


def run_training(
    method: str,
    datastore: str | Path,
    output_dir: str | Path,
    *,
    config: dict,
    source_root: str | Path | None = None,
    pretrained: str | Path | None = None,
) -> dict:
    method = method.lower()
    if method not in METHODS or config.get("method") != method:
        raise ValueError("Method and resolved config do not match an approved baseline")
    device_name = str(config["device"])
    if device_name != "cpu" and not (device_name.startswith("cuda") and torch.cuda.is_available()):
        raise ValueError(f"Requested unavailable device: {device_name}")
    output = require_nonexistent_output(output_dir)
    started = time.time()
    log_lines = []
    data = None

    def log(message: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    try:
        set_global_seed(int(config["seed"]))
        data = (
            load_human3_smoke(datastore, lazy_graphs=method == "afp")
            if config["role"] == "SMOKE"
            else load_formal_train_validation(datastore, method)
        )
        if method == "toxacol" and config["role"] == "FORMAL" and data.training_task_scope != "joint59":
            raise AssertionError("Formal TOXACol must use the complete joint59 train union")
        resolved = {
            **config,
            "source_root": str(Path(source_root).resolve()) if source_root else None,
            "pretrained": str(Path(pretrained).resolve()) if pretrained else None,
            "effective_train_count": len(data.train.sample_ids),
            "effective_validation_count": len(data.validation.sample_ids),
        }
        config.clear()
        config.update(resolved)
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.warning")
        environment = environment_snapshot()
        write_json(output / "config_resolved.json", config)
        write_json(output / "resolved_config.json", config)
        write_json(output / "environment.json", environment)
        write_json(output / "data_manifest.json", data.to_manifest())
        if config["role"] == "SMOKE":
            write_json(output / "smoke_samples.json", smoke_samples_payload(data))
        command = {"argv": sys.argv, "cwd": str(Path.cwd()), "started_unix": started, "exit_code": None}
        write_json(output / "command.json", command)
        runner = {
            "rf": lambda: _run_rf(data, output, config, log),
            "afp": lambda: _run_afp(data, output, config, torch.device(device_name), log),
            "dmpnn": lambda: _run_dmpnn(data, output, config, torch.device(device_name), source_root, log),
            "grover": lambda: _run_grover(data, output, config, torch.device(device_name), source_root, pretrained, log),
            "toxacol": lambda: _run_toxacol(data, output, config, torch.device(device_name), log),
        }[method]
        scaler, predictions, metrics, history, checkpoint, audit = runner()
        independent, _, independent_scaler = predict_partition(
            checkpoint,
            data.validation,
            device=device_name,
            source_root=source_root,
            expected_method=method,
        )
        tolerance = {"atol": 0.0, "rtol": 0.0} if method == "rf" else {"atol": 1e-5, "rtol": 1e-5}
        equivalent = bool(np.allclose(predictions, independent, **tolerance))
        if tuple(independent_scaler.task_names) != tuple(config["task_names"]):
            raise AssertionError("Independent inference did not restore the checkpoint scaler task order")
        if not equivalent:
            raise AssertionError("Independent checkpoint predictions differ from training-time best predictions")
        max_difference = float(np.max(np.abs(predictions - independent)))
        reload_audit = {
            "passed": True,
            "independent_inference_module": "baselines.inference.predict_partition",
            "restored_scaler_from_checkpoint": True,
            "restored_config_from_checkpoint": True,
            **tolerance,
            "max_abs_difference": max_difference,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        metrics = {**metrics, "reload": {"allclose": True, **tolerance, "max_abs_difference": max_difference}, "training_update": audit["training_update"]}
        write_json(output / "scaler.json", scaler.to_dict())
        write_json(output / "metrics.json", metrics)
        write_json(output / "history.json", history)
        _write_jsonl(output / "validation_predictions.jsonl", endpoint_prediction_rows(data, predictions, scaler))
        write_json(output / "model_audit.json", audit)
        write_json(output / "reload_verification.json", reload_audit)
        manifest = run_manifest_payload(config=config, data=data, environment=environment, checkpoint=checkpoint, metrics=metrics, audit=audit, status="PASS", reason=None)
        if config["role"] == "SMOKE":
            manifest["input_manifest_sha256"] = sha256_file(output / "smoke_samples.json")
        manifest["code_identity"] = {**manifest["code_identity"], **implementation_identity(Path(__file__).resolve().parents[1])}
        write_json(output / "run_manifest.json", manifest)
        command.update({"finished_unix": time.time(), "exit_code": 0})
        write_json(output / "command.json", command)
        (output / "exit_code.txt").write_text("0\n", encoding="utf-8")
        log(f"PASS role={config['role']} method={method} checkpoint={checkpoint.name}")
        result = {"method": method, "role": config["role"], "status": "PASS", "metrics": metrics, "output": str(output), "elapsed_seconds": time.time() - started}
        write_json(output / "run_result.json", result)
        return result
    except Exception as exc:
        write_json(output / "run_result.json", {"method": method, "status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.time() - started})
        if "command" in locals():
            command.update({"finished_unix": time.time(), "exit_code": 1})
            write_json(output / "command.json", command)
        (output / "exit_code.txt").write_text("1\n", encoding="utf-8")
        (output / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    finally:
        if data is not None:
            data.close()
        (output / "stdout.log").write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
        (output / "stderr.log").touch(exist_ok=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def run_smoke(
    method: str,
    datastore: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
    source_root: str | Path | None = None,
    pretrained: str | Path | None = None,
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    method = "afp" if method.lower() == "attentivefp" else method.lower()
    config = resolve_training_config(
        method,
        role="smoke",
        seed=seed,
        device=device,
        config_path=config_path,
        overrides=overrides,
    )
    return run_training(
        method,
        datastore,
        output_dir,
        config=config,
        source_root=source_root,
        pretrained=pretrained,
    )
