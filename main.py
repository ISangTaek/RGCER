"""Command-line entry point for HPS, Prompt-only, and RGCER experiments."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from rdkit import RDLogger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import architecture as architecture_method
import weighting as weighting_method
from architecture.Graphormer import Encoder as Encoder_Graphormer
from architecture.Graphormer_prompt import Encoder as Encoder_Graphormer_prompt
from architecture.Graphormer_rgcer import Encoder as Encoder_Graphormer_rgcer
from architecture.prediction_heads import TaskPredictionHead
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS, TOXACUTE_TASKS
from config import prepare_args, resolve_effective_rgcer_config
from dataset import DataCollator, DataloaderWrapper, PreprocessedDatasetWrapper, SubsetSequentialSampler
from experiment_config import (
    TOXACUTE_DATASTORE_DIR,
    TOXACUTE_MAIN_RUN_DIR,
    TOXACUTE_PHASE0_CHECKPOINT_NAME,
    TOXACUTE_PREPROCESSED_DIR,
    TOXACUTE_RAW_CSV,
    TOXACUTE_SMOKE_RUN_DIR,
)
from metric import ClsMetric, RegMetric
from loss import BCELoss, MSELoss
from preprocess_data import convert_to_single_emb_offline, get_graph_data_from_smiles
from trainer import Trainer
from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset
from utils import calculate_mgkg

RDLogger.DisableLog("rdApp.*")


CLASSIFICATION_DATASETS = {"hiv", "bace", "bbbp", "muv", "tox21", "sider", "clintox"}


RGCER_FLAG_NAMES = (
    "rgcer_use_source_response",
    "rgcer_use_target_response",
    "rgcer_use_molecule_query",
    "rgcer_use_sparse_routing",
    "rgcer_use_null_route",
    "rgcer_use_film",
    "rgcer_use_adapter",
    "rgcer_use_base_aux_loss",
    "rgcer_fallback_space",
    "rgcer_transfer_mechanism",
    "rgcer_source_policy",
)


def _configure_run_identity(params):
    """Give every experiment tag/seed its own reproducible output directory."""

    if not getattr(params, "save_path", None):
        return
    tag = str(getattr(params, "experiment_tag", "full"))
    seed = int(getattr(params, "seed", 42))
    params.save_path = str(Path(params.save_path) / tag / f"seed_{seed}")
    if getattr(params, "ckpt_name", "toxacute_rgcer") == "toxacute_rgcer":
        prefix = "rgcer" if getattr(params, "arch", "Graphormer_rgcer") == "Graphormer_rgcer" else str(params.arch).lower()
        params.ckpt_name = f"{prefix}_{tag}_seed{seed}"
    Path(params.save_path).mkdir(parents=True, exist_ok=True)


def _write_run_metadata(params, task_names, trainer):
    if not getattr(params, "save_path", None):
        return
    run_path = Path(params.save_path)
    run_path.mkdir(parents=True, exist_ok=True)
    with (run_path / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(params), handle, indent=2, sort_keys=True, default=str)

    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    lines = [
        f"arch: {params.arch}",
        f"task count: {len(task_names)}",
        f"parameter count: {total_parameters}",
        f"trainable parameter count: {trainable_parameters}",
        f"prediction mode: {effective_prediction_mode(params)}",
        f"split mode: {params.splitting}",
        f"seed: {params.seed}",
        f"experiment tag: {getattr(params, 'experiment_tag', 'full')}",
        f"split seed: {getattr(params, 'split_seed', '<unset>')}",
        f"selection scope: {getattr(params, 'selection_scope', 'human3')}",
    ]
    data_metadata = getattr(trainer, "data_metadata", None)
    if data_metadata:
        lines.extend(
            [
                f"datastore build id: {data_metadata.get('build_id', '<unset>')}",
                f"datastore fingerprint: {data_metadata.get('datastore_fingerprint', '<unset>')}",
                f"raw csv sha256: {data_metadata.get('raw_csv_sha256', '<unset>')}",
                f"split manifest hash: {data_metadata.get('split_manifest_hash', '<unset>')}",
                f"feature schema version: {data_metadata.get('feature_schema_version', '<unset>')}",
                f"max path distance: {data_metadata.get('max_path_distance', '<unset>')}",
            ]
        )
    lines.extend(f"{name}: {getattr(params, name, '<unset>')}" for name in RGCER_FLAG_NAMES)
    with (run_path / "architecture_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    effective = getattr(params, "effective_rgcer_config", None)
    if effective is not None:
        _write_json(run_path / "effective_config.json", effective)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str, allow_nan=True)


def effective_prediction_mode(params):
    """Return a head mode compatible with the selected dataset.

    The common regression default is quantile, but classification labels have
    one logit per sample and therefore must use the point head.
    """

    requested = getattr(params, "prediction_mode", "point")
    if getattr(params, "dataset", None) in CLASSIFICATION_DATASETS and requested == "quantile":
        return "point"
    return requested


def task_names_for_params(params):
    if params.dataset == "toxacute":
        scopes = {"human3": HUMAN_TARGET_TASKS, "animal56": ANIMAL_SOURCE_TASKS, "all59": TOXACUTE_TASKS}
        scope = getattr(params, "toxacute_task_scope", "all59")
        if scope not in scopes:
            raise ValueError(f"Unknown toxacute task scope: {scope!r}")
        return list(scopes[scope])
    if params.dataset in {"hiv", "bace", "bbbp", "qm7", "esol", "freesolv", "lipophilicity"}:
        return ["task"]
    if params.dataset == "muv":
        return ["MUV-858", "MUV-859"]
    if params.dataset == "tox21":
        return ["SR-MMP", "SR-p53"]
    if params.dataset == "sider":
        return ["Nervous system disorders", "Injury, poisoning and procedural complications"]
    if params.dataset == "clintox":
        return ["FDA_APPROVED", "CT_TOX"]
    if params.dataset == "qm8":
        return ["f1-CAM", "f2-CAM"]
    if params.dataset == "qm9":
        return ["homo", "lumo"]
    raise ValueError(f"No support dataset {params.dataset}")


def build_task_dict(params, task_names):
    if params.dataset in CLASSIFICATION_DATASETS:
        task_dict = {
            task: {"metrics": ["AUROC", "AUPRC"], "metrics_fn": ClsMetric(), "loss_fn": BCELoss(), "weight": [1, 1]}
            for task in task_names
        }
    else:
        task_dict = {
            task: {"metrics": ["RMSE", "R2"], "metrics_fn": RegMetric(), "loss_fn": MSELoss(), "weight": [-1, 1]}
            for task in task_names
        }
    for task_name, spec in getattr(params, "auxiliary_task_specs", {}).items():
        if task_name in task_dict:
            task_dict[task_name].update(spec)
    return task_dict


def _device_from_params(params):
    if params.gpu_id != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{params.gpu_id}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _resolve_data_store(params, task_names, *, required=False):
    """Resolve the formal V2 root while retaining a deprecated V1 alias."""

    if getattr(params, "dataset", None) != "toxacute":
        return None
    requested = getattr(params, "data_store_dir", None)
    legacy = getattr(params, "preprocessed_data_dir", None)
    default_root = str(TOXACUTE_DATASTORE_DIR)
    if legacy:
        if requested and str(requested) not in {default_root, str(legacy)}:
            raise ValueError("--data_store_dir and --preprocessed_data_dir point to different roots")
        warnings.warn(
            "--preprocessed_data_dir is a legacy alias; use --data_store_dir for DataStore V2.",
            DeprecationWarning,
            stacklevel=2,
        )
        requested = legacy
    if not requested:
        if required:
            raise ValueError("--data_store_dir is required for ToxAcute train/test/data_check")
        return None
    store = ToxAcuteDataStore.resolve(requested)
    expected_max_path = getattr(params, "max_path_distance", None)
    auxiliary_names = set(getattr(params, "auxiliary_task_names", []))
    expected_tasks = [task for task in task_names if task not in auxiliary_names]
    store.validate(
        strict=False,
        expected_task_names=expected_tasks,
        expected_max_path_distance=expected_max_path,
    )
    params.data_store_dir = str(requested)
    params.preprocessed_data_dir = None
    params.max_path_distance = int(store.metadata["max_path_distance"])
    params.datastore_metadata = store.metadata
    params.datastore_context = store.context
    return store


def _attach_shuffled_endpoint(params, store, task_names):
    source_task = getattr(params, "shuffled_endpoint_source", None)
    if not source_task:
        return list(task_names)
    from auxiliary_labels import ShuffledEndpointOverlay

    overlay = ShuffledEndpointOverlay(
        store,
        source_task,
        seed=getattr(params, "shuffled_endpoint_seed", 42),
    )
    params.label_providers = {overlay.task_name: overlay}
    params.auxiliary_metadata_overrides = {
        overlay.task_name: overlay.metadata_override()
    }
    if getattr(params, "use_factorized_prompt", None) is None:
        params.use_factorized_prompt = True
    params.auxiliary_task_specs = {
        overlay.task_name: {
            "include_in_macro": False,
            "fit_conformal": False,
            "auxiliary_only": True,
        }
    }
    params.auxiliary_task_names = [overlay.task_name]
    return list(task_names) + [overlay.task_name]


def _loaders(params, task_names, collator):
    store = _resolve_data_store(params, task_names, required=True)
    wrapper = DataloaderWrapper(
        task_list=task_names,
        data_store=store,
        batch_size=params.bs,
        splitting=params.splitting,
        valid_size=params.vs,
        calibration_size=params.calibration_size,
        test_size=params.ts,
        num_workers=params.num_loader_workers,
        collate_fn_for_loader=collator,
        split_seed=params.split_seed,
        max_nodes_filter=params.max_nodes_filter,
        label_providers=getattr(params, "label_providers", None),
    )
    all_loaders = wrapper.get_data_loaders()
    result = {name: {} for name in ("train", "val", "calibration", "test")}
    for task in task_names:
        loaders = all_loaders[task]
        result["train"][task] = loaders["train"]
        result["val"][task] = loaders["val"]
        result["calibration"][task] = loaders["calibration"]
        result["test"][task] = loaders["test"]
    return result


def _build_model_components(params, task_names, device):
    encoder_map = {
        "Graphormer": Encoder_Graphormer,
        "Graphormer_prompt": Encoder_Graphormer_prompt,
        "Graphormer_rgcer": Encoder_Graphormer_rgcer,
    }
    if params.arch not in encoder_map or not hasattr(architecture_method, params.arch):
        raise ValueError(f"Unsupported architecture {params.arch!r}")
    decoders = nn.ModuleDict(
        {
            task: TaskPredictionHead(
                hidden_dim=params.hidden_dim,
                mode=effective_prediction_mode(params),
                head_hidden_dim=params.head_hidden_dim,
                dropout=params.head_dropout,
            )
            for task in task_names
        }
    )
    return encoder_map[params.arch], getattr(architecture_method, params.arch), decoders


def _prediction_records(params, trainer, batch, task_names):
    batch = batch.to(trainer.device)
    with torch.no_grad():
        result = trainer.predict_all_tasks(batch, return_aux=True)
    predictions, diagnostics = result if isinstance(result, tuple) else (result, None)
    rows = []
    batch_smiles = list(getattr(batch, "smiles", [""] * batch.y.size(0)))
    for sample_index, smiles in enumerate(batch_smiles):
        row = {"smiles": smiles}
        for task in task_names:
            decoded = trainer.decode_task_output(task, predictions[task][sample_index : sample_index + 1])
            median = float(decoded["median"].item())
            lower = float(decoded["lower"].item())
            upper = float(decoded["upper"].item())
            row[f"{task}__median_log"] = median
            row[f"{task}__lower_log"] = lower
            row[f"{task}__upper_log"] = upper
            row[f"{task}__median_mgkg"] = calculate_mgkg(smiles, median)
            row[f"{task}__lower_mgkg"] = calculate_mgkg(smiles, upper)
            row[f"{task}__upper_mgkg"] = calculate_mgkg(smiles, lower)
            if isinstance(diagnostics, dict) and task in diagnostics:
                diag = diagnostics[task]
                row[f"{task}__null_weight"] = float(diag["null_weight"][sample_index, 0].detach().cpu())
        rows.append(row)
    return rows


def main(params):
    _configure_run_identity(params)
    selected_mode = effective_prediction_mode(params)
    if selected_mode != getattr(params, "prediction_mode", selected_mode):
        warnings.warn(
            "Classification datasets use the point prediction head; prediction_mode='quantile' was overridden.",
            RuntimeWarning,
            stacklevel=2,
        )
        params.prediction_mode = selected_mode
    task_names = task_names_for_params(params)
    formal_task_names = list(task_names)
    params.effective_rgcer_config = resolve_effective_rgcer_config(params)
    if params.mode == "data_check":
        store = _resolve_data_store(params, formal_task_names, required=True)
        from data_preflight import run_datastore_preflight

        report = run_datastore_preflight(
            store,
            formal_task_names,
            max_nodes_filter=params.max_nodes_filter,
            min_calibration_size=params.min_calibration_size,
            require_calibration=bool(params.fit_conformal and effective_prediction_mode(params) == "quantile"),
        )
        output_root = Path(params.save_path or ".")
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / "data_preflight.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if params.mode in {"train", "test", "batch_inference"}:
        store = _resolve_data_store(params, formal_task_names, required=True)
        task_names = _attach_shuffled_endpoint(params, store, task_names)
        if getattr(params, "explicitly_allowed_auxiliary_sources", None):
            unknown = [
                name
                for name in params.explicitly_allowed_auxiliary_sources
                if name not in task_names
            ]
            if unknown:
                raise ValueError(
                    "--explicitly_allowed_auxiliary_sources names tasks outside this run: "
                    f"{unknown}; available={task_names}"
                )
        if params.arch == "Graphormer_rgcer" and task_names:
            # Fail before any training I/O when the policy contradicts the run.
            from trainer import build_source_policy_mask

            build_source_policy_mask(
                task_names,
                policy=params.rgcer_source_policy,
                allowed_auxiliary=params.explicitly_allowed_auxiliary_sources,
            )
        if params.mode in {"train", "test"}:
            from data_preflight import run_datastore_preflight

            preflight = run_datastore_preflight(
                store,
                formal_task_names,
                max_nodes_filter=params.max_nodes_filter,
                min_calibration_size=params.min_calibration_size,
                require_calibration=bool(
                    params.fit_conformal and effective_prediction_mode(params) == "quantile"
                ),
            )
            if params.save_path:
                _write_json(Path(params.save_path) / "data_preflight.json", preflight)
    task_dict = build_task_dict(params, task_names)
    device = _device_from_params(params)
    kwargs, optim_param = prepare_args(params)
    encoder_class, architecture_class, decoders = _build_model_components(params, task_names, device)
    collator = DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip,
        # V2 applies max_nodes_filter while constructing task indices.  The
        # collator keeps the legacy option only for direct compatibility tests.
        max_node_filter=None,
    )
    trainer = Trainer(
        task_dict=task_dict,
        weighting=weighting_method.__dict__[params.weighting],
        architecture=architecture_class,
        encoder_class=encoder_class,
        decoders=decoders,
        optim_param=optim_param,
        args=params,
        save_path=params.save_path,
        load_path=params.load_path,
        **kwargs,
    )
    _write_run_metadata(params, task_names, trainer)
    print(f"Using device: {trainer.device}; tasks={len(task_names)}; architecture={params.arch}")

    if params.mode in {"train", "test"}:
        loaders = _loaders(params, task_names, collator)
        if params.mode == "train":
            history = trainer.train(
                train_dataloaders_dict=loaders["train"],
                val_dataloaders_dict=loaders["val"],
                calibration_dataloaders_dict=loaders["calibration"],
                test_dataloaders_dict=loaders["test"],
                epochs=params.epochs,
                params_main=params,
            )
            if params.save_path:
                metrics_payload = {"history": history}
                metrics_payload["conformal"] = trainer._conformal_validity()
                if trainer.final_test_result is not None:
                    metrics_payload["test"] = trainer.final_test_result
                _write_json(Path(params.save_path) / "metrics.json", metrics_payload)
                routing_result = trainer.final_test_result
                if routing_result is None and history:
                    routing_result = history[-1].get("validation", {})
                if routing_result is not None:
                    _write_json(
                        Path(params.save_path) / "routing_summary.json",
                        {
                            **Trainer.aggregate_routing_summary(
                                routing_result.get("routing", {})
                            ),
                            "tasks": routing_result.get("routing", {}),
                        },
                    )
        else:
            result = trainer.test(loaders["test"])
            if params.save_path:
                _write_json(
                    Path(params.save_path) / "metrics.json",
                    {"test": result, "conformal": trainer._conformal_validity()},
                )
                _write_json(
                    Path(params.save_path) / "routing_summary.json",
                    {
                        **Trainer.aggregate_routing_summary(result.get("routing", {})),
                        "tasks": result.get("routing", {}),
                    },
                )
        return

    if params.mode == "single_inference":
        if not params.smiles:
            raise ValueError("--smiles is required for single_inference")
        graph = get_graph_data_from_smiles(
            params.smiles,
            0.0,
            convert_to_single_emb_offline,
            task_name=None,
            max_path_distance=getattr(trainer, "max_path_distance", getattr(params, "max_path_distance", 8) or 8),
        )
        batch = collator([graph]).to(trainer.device)
        with torch.no_grad():
            predictions = trainer.predict_all_tasks(batch)
        print("--- Predictions ---")
        for task in task_names:
            decoded = trainer.decode_task_output(task, predictions[task])
            print(
                f"{task}: median={decoded['median'].item():.6f}, "
                f"lower={decoded['lower'].item():.6f}, upper={decoded['upper'].item():.6f}, "
                f"median_mgkg={calculate_mgkg(params.smiles, decoded['median'].item()):.6f}, "
                f"lower_mgkg={calculate_mgkg(params.smiles, decoded['upper'].item()):.6f}, "
                f"upper_mgkg={calculate_mgkg(params.smiles, decoded['lower'].item()):.6f}"
            )
        return

    if params.mode == "batch_inference":
        if not params.inference_task:
            raise ValueError("--inference_task is required")
        store = _resolve_data_store(params, task_names, required=True)
        dataset = ToxAcuteTaskDataset(
            store,
            params.inference_task,
            split=params.inference_split,
            max_nodes=params.max_nodes_filter,
            label_provider=getattr(params, "label_providers", {}).get(params.inference_task),
        )
        loader = DataLoader(
            dataset,
            batch_size=params.bs,
            shuffle=False,
            num_workers=params.num_loader_workers,
            collate_fn=collator,
            persistent_workers=params.num_loader_workers > 0,
        )
        rows = []
        trainer.model.eval()
        for batch in tqdm(loader, desc="Running Inference", unit="batch"):
            if batch.get("is_empty", False):
                continue
            rows.extend(_prediction_records(params, trainer, batch, task_names))
        output_path = params.inference_output_path or f"./{params.dataset}_{params.inference_task}_predictions.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Batch inference complete: {output_path}")
        return
    raise ValueError(f"Unknown mode {params.mode!r}")


def build_parser():
    parser = argparse.ArgumentParser(description="Multitask molecular Graphormer framework")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "single_inference", "batch_inference", "data_check"],
        default="train",
    )
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--save_path", default=TOXACUTE_MAIN_RUN_DIR)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--ckpt_name", default="toxacute_rgcer")
    parser.add_argument("--data_store_dir", default=TOXACUTE_DATASTORE_DIR)
    parser.add_argument("--preprocessed_data_dir", default=None)
    parser.add_argument("--num_loader_workers", type=int, default=0)
    parser.add_argument("--spatial_pos_clip", type=int, default=20)
    parser.add_argument("--max_path_distance", type=int, default=None)
    parser.add_argument("--max_nodes_filter", type=int, default=512)
    parser.add_argument("--smiles", default=None)
    parser.add_argument("--inference_task", default="human_oral_TDLo")
    parser.add_argument("--inference_split", choices=["train", "validation", "calibration", "test"], default=None)
    parser.add_argument("--shuffled_endpoint_source", default=None)
    parser.add_argument("--shuffled_endpoint_seed", type=int, default=42)
    parser.add_argument("--inference_output_path", default="./artifacts/results/toxacute_predictions.csv")

    parser.add_argument("--arch", choices=["Graphormer", "Graphormer_prompt", "Graphormer_rgcer"], default="Graphormer_rgcer")
    parser.add_argument("--a_layers", type=int, default=8)
    parser.add_argument("--a_heads", type=int, default=4)
    parser.add_argument("--t_layers", type=int, default=1)
    parser.add_argument("--t_heads", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--mid_dim", type=int, default=128)
    parser.add_argument("--edge_bias_mode", choices=["path", "direct", "direct_plus_path"], default="path")
    parser.add_argument("--prediction_mode", choices=["point", "quantile"], default="quantile")
    parser.add_argument("--head_hidden_dim", type=int, default=96)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--response_hidden_dim", type=int, default=96)
    parser.add_argument(
        "--use_factorized_prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use ToxAcute endpoint factors; auto-detect when omitted.",
    )
    parser.add_argument("--task_residual_scale", type=float, default=0.1)
    parser.add_argument("--prompt_layers", type=int, default=0)
    parser.add_argument("--prompt_heads", type=int, default=4)
    parser.add_argument("--prompt_ffn_dim", type=int, default=192)
    parser.add_argument("--prompt_dropout", type=float, default=0.1)
    parser.add_argument("--router_mode", choices=["static", "dynamic"], default="dynamic")
    parser.add_argument("--router_dim", type=int, default=96)
    parser.add_argument("--router_top_k", type=int, default=8)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--exclude_target_from_sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter_ratio", type=float, default=0.25)
    parser.add_argument("--prompt_gate_init", type=float, default=-2.0)
    parser.add_argument(
        "--rgcer_use_source_response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include each source endpoint's preliminary response in source tokens.",
    )
    parser.add_argument(
        "--rgcer_use_target_response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the target endpoint preliminary response in the router query.",
    )
    parser.add_argument(
        "--rgcer_use_molecule_query",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition endpoint routing weights on the current molecule representation.",
    )
    parser.add_argument(
        "--rgcer_use_sparse_routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply top-k sparsification to real source endpoints.",
    )
    parser.add_argument(
        "--rgcer_use_null_route",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the explicit no-transfer NULL option.",
    )
    parser.add_argument("--rgcer_use_film", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgcer_use_adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rgcer_use_base_aux_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the HPS/base auxiliary loss on the routed RGCER path.",
    )
    parser.add_argument(
        "--rgcer_fallback_space",
        choices=["prediction", "representation"],
        default="prediction",
        help="Space in which the NULL fallback blends HPS and routed outputs.",
    )
    parser.add_argument(
        "--rgcer_transfer_mechanism",
        choices=["endpoint_router", "response_stacking", "target_only"],
        default="endpoint_router",
        help="Mutually exclusive RGCER transfer mechanism or baseline.",
    )

    parser.add_argument("--hps_warmup_epochs", type=int, default=10)
    parser.add_argument("--lambda_base", type=float, default=0.25)
    parser.add_argument("--lambda_quantile", type=float, default=1.0)
    parser.add_argument("--lower_quantile", type=float, default=0.05)
    parser.add_argument("--upper_quantile", type=float, default=0.95)
    parser.add_argument("--conformal_alpha", type=float, default=0.10)
    parser.add_argument("--fit_conformal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_calibration_size", type=int, default=30)
    parser.add_argument("--calibration_size", type=float, default=0.10)
    parser.add_argument("--tasks_per_update", type=int, default=1)
    parser.add_argument("--routing_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--toxacute_task_scope", choices=["human3", "animal56", "all59"], default="all59")
    parser.add_argument(
        "--selection_scope",
        choices=["human3", "all_tasks"],
        default="human3",
        help="Task scope for best-checkpoint selection. human3 selects on the three "
        "human target endpoints (falls back to all_tasks when the run has none).",
    )
    parser.add_argument(
        "--rgcer_source_policy",
        choices=["all_except_target", "animal56_only"],
        default="animal56_only",
        help="Router source policy for Graphormer_rgcer. animal56_only keeps the formal "
        "animal-to-human claim: human targets draw only on animal endpoints, plus any "
        "auxiliary endpoint explicitly allowed below. The router always excludes the "
        "target itself.",
    )
    parser.add_argument(
        "--explicitly_allowed_auxiliary_sources",
        type=str,
        default="",
        help="Comma-separated auxiliary (e.g. shuffled) endpoint names that stay usable "
        "as router sources under rgcer_source_policy=animal56_only.",
    )
    parser.add_argument("--experiment_tag", default="full")

    parser.add_argument("--weighting", choices=["EW", "UW", "DWA"], default="EW")
    parser.add_argument("--optim", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dataset", default="toxacute")
    parser.add_argument("--splitting", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--vs", type=float, default=0.1)
    parser.add_argument("--ts", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def validate_params(params):
    if params.hidden_dim % params.a_heads != 0:
        raise ValueError("hidden_dim must be divisible by a_heads")
    raw_allowed_sources = str(
        getattr(params, "explicitly_allowed_auxiliary_sources", "") or ""
    )
    params.explicitly_allowed_auxiliary_sources = tuple(
        name.strip() for name in raw_allowed_sources.split(",") if name.strip()
    )
    if params.prompt_heads <= 0 or params.hidden_dim % params.prompt_heads != 0:
        raise ValueError("hidden_dim must be divisible by prompt_heads")
    if params.router_dim <= 0 or params.router_top_k < 0 or params.router_temperature <= 0:
        raise ValueError("Invalid router configuration")
    if getattr(params, "rgcer_use_sparse_routing", True) and params.router_top_k <= 0:
        raise ValueError("Sparse routing requires router_top_k > 0")
    if params.hps_warmup_epochs < 0 or params.tasks_per_update <= 0:
        raise ValueError("Invalid warmup or tasks_per_update")
    if params.weighting == "DWA" and params.tasks_per_update == 1:
        raise ValueError("DWA requires tasks_per_update > 1")
    if params.vs + params.calibration_size + params.ts >= 1.0:
        raise ValueError("validation + calibration + test ratios must be less than 1")
    if params.split_seed is None:
        raise ValueError("split_seed must be an integer")
    if not 0.0 < params.adapter_ratio <= 1.0:
        raise ValueError("adapter_ratio must be in (0, 1]")
    if params.arch == "Graphormer_rgcer" and params.router_mode != "dynamic":
        warnings.warn(
            "--router_mode belongs to Graphormer_prompt and is ignored by Graphormer_rgcer; "
            "use --rgcer_* switches instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    if params.arch != "Graphormer_rgcer":
        defaults = {
            "rgcer_use_source_response": True,
            "rgcer_use_target_response": True,
            "rgcer_use_molecule_query": True,
            "rgcer_use_sparse_routing": True,
            "rgcer_use_null_route": True,
            "rgcer_use_film": True,
            "rgcer_use_adapter": True,
            "rgcer_use_base_aux_loss": True,
            "rgcer_fallback_space": "prediction",
            "rgcer_transfer_mechanism": "endpoint_router",
        }
        changed = [name for name, default in defaults.items() if getattr(params, name, default) != default]
        if changed:
            warnings.warn(
                f"RGCER flags {changed} are ignored by architecture {params.arch}.",
                RuntimeWarning,
                stacklevel=2,
            )
    params.effective_rgcer_config = resolve_effective_rgcer_config(params)


if __name__ == "__main__":
    parser = build_parser()
    params = parser.parse_args()
    validate_params(params)
    if params.save_path:
        os.makedirs(params.save_path, exist_ok=True)
    main(params)
