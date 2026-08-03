"""Command-line entry point for HPS, Prompt-only, and RGCER experiments."""

from __future__ import annotations

import argparse
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
from config import prepare_args
from dataset import DataCollator, DataloaderWrapper, PreprocessedDatasetWrapper, SubsetSequentialSampler
from experiment_config import (
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
from utils import calculate_mgkg

RDLogger.DisableLog("rdApp.*")


CLASSIFICATION_DATASETS = {"hiv", "bace", "bbbp", "muv", "tox21", "sider", "clintox"}


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
        return {
            task: {"metrics": ["AUROC", "AUPRC"], "metrics_fn": ClsMetric(), "loss_fn": BCELoss(), "weight": [1, 1]}
            for task in task_names
        }
    return {
        task: {"metrics": ["RMSE", "R2"], "metrics_fn": RegMetric(), "loss_fn": MSELoss(), "weight": [-1, 1]}
        for task in task_names
    }


def _device_from_params(params):
    if params.gpu_id != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{params.gpu_id}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _loaders(params, task_names, collator):
    wrapper = DataloaderWrapper(
        task_list=task_names,
        preprocessed_data_base_dir=params.preprocessed_data_dir,
        batch_size=params.bs,
        splitting=params.splitting,
        valid_size=params.vs,
        calibration_size=params.calibration_size,
        test_size=params.ts,
        num_workers=params.num_loader_workers,
        collate_fn_for_loader=collator,
        split_seed=params.seed,
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
        result = trainer.model(batch, return_all_tasks=True, return_aux=True)
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
    selected_mode = effective_prediction_mode(params)
    if selected_mode != getattr(params, "prediction_mode", selected_mode):
        warnings.warn(
            "Classification datasets use the point prediction head; prediction_mode='quantile' was overridden.",
            RuntimeWarning,
            stacklevel=2,
        )
        params.prediction_mode = selected_mode
    task_names = task_names_for_params(params)
    task_dict = build_task_dict(params, task_names)
    device = _device_from_params(params)
    kwargs, optim_param = prepare_args(params)
    encoder_class, architecture_class, decoders = _build_model_components(params, task_names, device)
    collator = DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip,
        max_node_filter=params.max_nodes_filter,
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
    print(f"Using device: {trainer.device}; tasks={len(task_names)}; architecture={params.arch}")

    if params.mode in {"train", "test"}:
        if not params.preprocessed_data_dir:
            raise ValueError("--preprocessed_data_dir is required for train/test")
        loaders = _loaders(params, task_names, collator)
        if params.mode == "train":
            trainer.train(
                train_dataloaders_dict=loaders["train"],
                val_dataloaders_dict=loaders["val"],
                calibration_dataloaders_dict=loaders["calibration"],
                test_dataloaders_dict=loaders["test"],
                epochs=params.epochs,
                params_main=params,
            )
        else:
            trainer.test(loaders["test"])
        return

    if params.mode == "single_inference":
        if not params.smiles:
            raise ValueError("--smiles is required for single_inference")
        graph = get_graph_data_from_smiles(
            params.smiles,
            0.0,
            convert_to_single_emb_offline,
            task_name=None,
        )
        batch = collator([graph]).to(trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            predictions = trainer.model(batch, return_all_tasks=True)
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
        if not params.preprocessed_data_dir or not params.inference_task:
            raise ValueError("--preprocessed_data_dir and --inference_task are required")
        dataset = PreprocessedDatasetWrapper(Path(params.preprocessed_data_dir) / params.inference_task)
        loader = DataLoader(
            dataset,
            batch_size=params.bs,
            sampler=SubsetSequentialSampler(list(range(len(dataset)))),
            num_workers=params.num_loader_workers,
            collate_fn=collator,
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
    parser.add_argument("--mode", choices=["train", "test", "single_inference", "batch_inference"], default="train")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--save_path", default=TOXACUTE_MAIN_RUN_DIR)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--ckpt_name", default="toxacute_rgcer")
    parser.add_argument("--preprocessed_data_dir", default=TOXACUTE_PREPROCESSED_DIR)
    parser.add_argument("--num_loader_workers", type=int, default=0)
    parser.add_argument("--spatial_pos_clip", type=int, default=20)
    parser.add_argument("--max_nodes_filter", type=int, default=512)
    parser.add_argument("--smiles", default=None)
    parser.add_argument("--inference_task", default="human_oral_TDLo")
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
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def validate_params(params):
    if params.hidden_dim % params.a_heads != 0:
        raise ValueError("hidden_dim must be divisible by a_heads")
    if params.prompt_heads <= 0 or params.hidden_dim % params.prompt_heads != 0:
        raise ValueError("hidden_dim must be divisible by prompt_heads")
    if params.router_dim <= 0 or params.router_top_k < 0 or params.router_temperature <= 0:
        raise ValueError("Invalid router configuration")
    if params.hps_warmup_epochs < 0 or params.tasks_per_update <= 0:
        raise ValueError("Invalid warmup or tasks_per_update")
    if params.weighting == "DWA" and params.tasks_per_update == 1:
        raise ValueError("DWA requires tasks_per_update > 1")
    if params.vs + params.calibration_size + params.ts >= 1.0:
        raise ValueError("validation + calibration + test ratios must be less than 1")
    if not 0.0 < params.adapter_ratio <= 1.0:
        raise ValueError("adapter_ratio must be in (0, 1]")


if __name__ == "__main__":
    parser = build_parser()
    params = parser.parse_args()
    validate_params(params)
    if params.save_path:
        os.makedirs(params.save_path, exist_ok=True)
    main(params)
