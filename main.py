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
from reproducibility import seed_everything, state_dict_sha256
from conformal import resolve_conformal_tasks
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


def _git_commit_hash() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:
        return "<unknown>"


def _sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    initial_model_sha256 = getattr(trainer, "initial_model_sha256", None)
    if initial_model_sha256 is None:
        initial_model_sha256 = state_dict_sha256(trainer.model)
    lines = [
        f"arch: {params.arch}",
        f"task count: {len(task_names)}",
        f"parameter count: {total_parameters}",
        f"trainable parameter count: {trainable_parameters}",
        f"initial model sha256: {initial_model_sha256}",
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
    # Review-2 §51: structured provenance for multi-seed audit — identical
    # seeds must regenerate this hash, different seeds must not.  The
    # reproducibility/worker policy is recorded so formal runs are auditable.
    data_metadata = getattr(trainer, "data_metadata", None) or {}
    from reproducibility import PERSISTENT_WORKER_POLICY, SEED_POLICY_VERSION

    metadata = {
        "seed": int(getattr(params, "seed", 42)),
        "seed_policy_version": SEED_POLICY_VERSION,
        "num_loader_workers": int(getattr(params, "num_loader_workers", 0)),
        "persistent_worker_policy": PERSISTENT_WORKER_POLICY,
        "train_eval_scope": getattr(params, "train_eval_scope", "full"),
        "evaluation_scope": getattr(params, "train_eval_scope", "full"),
        "git_commit": _git_commit_hash(),
        # Review §30: D6 aggregator second-line identity verification.
        "d6_candidate": getattr(params, "d6_candidate", "none"),
        # D6 shuffle/counterfactual provenance (review P1-5, §51).
        "shuffle_animal_train_labels": bool(
            getattr(params, "shuffle_animal_train_labels", False)
        ),
        "animal_shuffle_seed": getattr(params, "animal_shuffle_seed", None),
        "animal_shuffle_mapping_sha256": getattr(
            params, "animal_shuffle_mapping_sha256", None
        ),
        "init_overlay": getattr(params, "init_overlay_provenance", None),
        "initial_model_sha256": initial_model_sha256,
        "manifest_sha256": data_metadata.get("split_manifest_hash"),
        "datastore_fingerprint": data_metadata.get("datastore_fingerprint"),
        "checkpoint_version": 6 if data_metadata else 4,
    }
    # Review P1-5: formal CARD runs must pin the delta-table identity.
    if float(getattr(params, "card_lambda_delta", 0.0)) > 0:
        card_path = Path(params.card_delta_table)
        sidecar = Path(str(card_path) + ".provenance.json")
        if not sidecar.is_file():
            raise RuntimeError(
                "Formal D6 CARD run requires the delta-table provenance sidecar: "
                f"{sidecar}"
            )
        metadata["card_delta_table"] = str(card_path)
        metadata["card_delta_table_sha256"] = _sha256_file(card_path)
        metadata["card_delta_provenance_path"] = str(sidecar)
        metadata["card_delta_provenance_sha256"] = _sha256_file(sidecar)
    _write_json(run_path / "run_metadata.json", metadata)
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


def _priority_tasks(params, task_names):
    """Endpoints the formal split/preflight must cover with human floors."""

    if getattr(params, "conformal_scope", "human3") == "human3":
        return [name for name in HUMAN_TARGET_TASKS if name in task_names]
    return []


def _conformal_scope_tasks(params, task_names):
    """Endpoints that must reach the conformal finite-rank calibration floor."""

    if not getattr(params, "fit_conformal", True):
        return []
    return resolve_conformal_tasks(getattr(params, "conformal_scope", "human3"), task_names)


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


def _scoped_loader_dicts(loaders, train_eval_scope):
    """Plan §5.2: which loader dicts reach ``Trainer.train`` for this scope.

    ``validation_only`` must hand over only train+validation; calibration and
    test are passed as ``None`` so the trainer can neither fit CQR on them nor
    evaluate them.
    """

    if train_eval_scope == "validation_only":
        return loaders["train"], loaders["val"], None, None
    return loaders["train"], loaders["val"], loaders["calibration"], loaders["test"]


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
        loader_seed=params.seed,
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


def _apply_conformal_for_task(params, trainer, task) -> bool:
    """Conformal applies only where a fitted qhat state exists (review §23).

    Under ``conformal_scope=human3`` checkpoints, animal endpoints have no
    qhat; requesting conformal there used to raise.  They must still decode
    as point/raw-quantile outputs instead of crashing.
    """

    return bool(getattr(params, "fit_conformal", True)) and (
        task in trainer.conformal_calibrator.states
    )


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
            apply_conformal = _apply_conformal_for_task(params, trainer, task)
            decoded = trainer.decode_task_output(
                task,
                predictions[task][sample_index : sample_index + 1],
                apply_conformal=apply_conformal,
            )
            median = float(decoded["median"].item())
            lower = float(decoded["lower"].item())
            upper = float(decoded["upper"].item())
            row[f"{task}__median_log"] = median
            row[f"{task}__lower_log"] = lower
            row[f"{task}__upper_log"] = upper
            row[f"{task}__median_mgkg"] = calculate_mgkg(smiles, median)
            row[f"{task}__lower_mgkg"] = calculate_mgkg(smiles, upper)
            row[f"{task}__upper_mgkg"] = calculate_mgkg(smiles, lower)
            row[f"{task}__interval_type"] = "conformal" if apply_conformal else "raw_quantile"
            if isinstance(diagnostics, dict) and task in diagnostics:
                diag = diagnostics[task]
                row[f"{task}__null_weight"] = float(diag["null_weight"][sample_index, 0].detach().cpu())
        rows.append(row)
    return rows


def main(params):
    # P0 reproducibility gate: the seed must be active before ANY nn.Module
    # is constructed (review §5) — otherwise decoder initial weights are not
    # a function of --seed.  Trainer's internal re-seed is defensive only.
    seed_everything(params.seed)
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
            conformal_alpha=(
                params.conformal_alpha if params.fit_conformal else None
            ),
            conformal_task_names=_conformal_scope_tasks(params, formal_task_names),
            priority_task_names=_priority_tasks(params, formal_task_names),
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
                conformal_alpha=(
                    params.conformal_alpha if params.fit_conformal else None
                ),
                conformal_task_names=_conformal_scope_tasks(params, formal_task_names),
                priority_task_names=_priority_tasks(params, formal_task_names),
            )
            if params.save_path:
                _write_json(Path(params.save_path) / "data_preflight.json", preflight)
    task_dict = build_task_dict(params, task_names)
    device = _device_from_params(params)
    if getattr(params, "shuffle_animal_train_labels", False) and params.mode in {"train", "test"}:
        # D6 §61-§63: counterfactual Animal56 teacher.  Only train-split labels
        # of the animal endpoints are permuted (fixed task-local mapping);
        # validation stays real and human labels are never touched.
        from shuffled_animal_labels import ANIMAL_SHUFFLE_SEED, ShuffledAnimalTrainLabels

        animal_tasks = [name for name in task_names if name in set(ANIMAL_SOURCE_TASKS)]
        if store is not None and animal_tasks:
            provider = ShuffledAnimalTrainLabels(
                store,
                animal_tasks,
                shuffle_seed=getattr(params, "animal_shuffle_seed", ANIMAL_SHUFFLE_SEED),
            )
            params.label_providers = {name: provider for name in animal_tasks}
            print(
                f"shuffled-animal teacher: permuted train labels for {len(animal_tasks)} endpoints"
            )
            # Review P1-5: persist the exact shuffle mapping + sanity table so
            # the counterfactual alignment is auditable per run.
            mapping_sha = provider.mapping_sha256()
            params.animal_shuffle_mapping_sha256 = mapping_sha
            if params.save_path:
                manifest = {
                    "animal_shuffle_seed": getattr(params, "animal_shuffle_seed", ANIMAL_SHUFFLE_SEED),
                    "animal_shuffle_mapping_sha256": mapping_sha,
                    "tasks": sorted(animal_tasks),
                }
                _write_json(Path(params.save_path) / "D6_ANIMAL_SHUFFLE_MANIFEST.json", manifest)
                _write_csv(
                    Path(params.save_path) / "D6_SHUFFLE_SANITY.csv",
                    ["task", "n", "mean_before", "mean_after", "std_before", "std_after",
                     "label_multiset_equal", "mapping_hash"],
                    provider.sanity_rows(),
                )
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
    if getattr(params, "init_state_path", None):
        # D6 CSDT/CLST/sequential-transfer initialisation (plan §64-§65): the
        # fresh seed-matched model receives a prepared state dict before any
        # training step; provenance hashes are recorded post-overlay
        # (review P1-12/§51).
        from reproducibility import state_dict_sha256

        pre_overlay_model_sha256 = trainer.initial_model_sha256
        state = torch.load(params.init_state_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(state, strict=True)
        post_overlay_model_sha256 = state_dict_sha256(trainer.model)
        trainer.initial_model_sha256 = post_overlay_model_sha256
        init_provenance_path = Path(str(params.init_state_path) + ".provenance.json")
        if getattr(params, "require_init_provenance", False) and not init_provenance_path.is_file():
            raise FileNotFoundError(
                "D6 formal candidate is missing its init provenance sidecar: "
                f"{init_provenance_path}"
            )
        # Review §29: actually parse the sidecar and verify it matches this
        # candidate run (seed, artifact identity, data identity).
        init_provenance = None
        if init_provenance_path.is_file():
            init_provenance = json.loads(init_provenance_path.read_text(encoding="utf-8"))
            if int(init_provenance.get("human_seed", -1)) != int(params.seed):
                raise ValueError(
                    f"Init provenance human_seed {init_provenance.get('human_seed')!r} "
                    f"does not match --seed {params.seed}"
                )
            if (
                init_provenance.get("output_sha256")
                and init_provenance["output_sha256"] != _sha256_file(params.init_state_path)
            ):
                raise ValueError(
                    "Init provenance output_sha256 does not match the init state file"
                )
        params.init_overlay_provenance = {
            "pre_overlay_model_sha256": pre_overlay_model_sha256,
            "post_overlay_model_sha256": post_overlay_model_sha256,
            "init_state_path": str(params.init_state_path),
            "init_state_sha256": _sha256_file(Path(params.init_state_path)),
            "init_provenance_path": (
                str(init_provenance_path) if init_provenance_path.exists() else None
            ),
        }
        print(f"applied init state overlay from {params.init_state_path}")
        _write_run_metadata(params, task_names, trainer)
    print(f"Using device: {trainer.device}; tasks={len(task_names)}; architecture={params.arch}")

    if params.mode in {"train", "test"}:
        loaders = _loaders(params, task_names, collator)
        if store is not None:
            # Per-sample diagnostics need the DataStore sample_id -> raw CSV row
            # mapping; batch indices alone cannot be aligned across runs.
            trainer.sample_row_index = {
                str(sample_id): int(row_index)
                for sample_id, row_index in zip(
                    store.sample_ids.tolist(), store.row_indices.tolist()
                )
            }
        if params.mode == "train":
            eval_scope = getattr(params, "train_eval_scope", "full")
            if eval_scope == "validation_only" and bool(getattr(params, "fit_conformal", True)):
                warnings.warn(
                    "--train_eval_scope validation_only never passes calibration to the "
                    "trainer, so CQR will not be fitted; pass --no-fit_conformal to make "
                    "this explicit.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            train_loaders, val_loaders, calibration_loaders, test_loaders = _scoped_loader_dicts(
                loaders, eval_scope
            )
            history = trainer.train(
                train_dataloaders_dict=train_loaders,
                val_dataloaders_dict=val_loaders,
                calibration_dataloaders_dict=calibration_loaders,
                test_dataloaders_dict=test_loaders,
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
            apply_conformal = _apply_conformal_for_task(params, trainer, task)
            decoded = trainer.decode_task_output(
                task, predictions[task], apply_conformal=apply_conformal
            )
            interval_type = "conformal" if apply_conformal else "raw_quantile"
            print(
                f"{task}: median={decoded['median'].item():.6f}, "
                f"lower={decoded['lower'].item():.6f}, upper={decoded['upper'].item():.6f}, "
                f"interval_type={interval_type}, "
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
        "--train_eval_scope",
        choices=["validation_only", "full"],
        default="full",
        help="Diagnostic-phase data hygiene (plan §5): validation_only hands the "
        "Trainer only train+validation loaders — calibration/test are never "
        "passed in, CQR is not fitted, and no test evaluation runs. Keep the "
        "default full for formal runs.",
    )
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write read-only per-epoch diagnostic artifacts (path metrics, "
        "routing aggregates, gradient norms, best-epoch per-sample CSVs) under "
        "<save_path>/diagnostics. Never changes model math.",
    )
    parser.add_argument(
        "--conformal_scope",
        choices=["human3", "all_tasks"],
        default="human3",
        help="Which regression tasks receive CQR states: primary conformal "
        "evaluation targets the human endpoints while point metrics still cover "
        "every task.",
    )
    parser.add_argument(
        "--task_sampling",
        choices=["proportional", "human_target_floor"],
        default="proportional",
        help="Schedule exposure (review §39). proportional keeps the historical "
        "data-proportional mix for every model; human_target_floor adds one "
        "full extra pass of each human target loader per epoch and is meant "
        "for ablations only. Per-epoch exposure is recorded in "
        "schedule_diagnostics either way.",
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

    # D6 counterfactual transfer switches (plan §61-§67, §77).
    parser.add_argument(
        "--shuffle_animal_train_labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Animal56 teacher mode: replace train-split labels of every animal "
        "endpoint with a fixed task-local permutation (ANIMAL_SHUFFLE_SEED). "
        "Validation labels stay real; human labels are never touched.",
    )
    parser.add_argument("--animal_shuffle_seed", type=int, default=20260831)
    parser.add_argument(
        "--init_state_path",
        default=None,
        help="Optional plain model state_dict applied to the freshly built model "
        "before training (CSDT/CLST/sequential-transfer initialisation).",
    )
    parser.add_argument(
        "--require_init_provenance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="D6 formal candidates: abort when the init state lacks its "
        ".provenance.json sidecar (review P1-6).",
    )
    parser.add_argument(
        "--card_lambda_delta",
        type=float,
        default=0.0,
        help="D6 O3 CARD distillation weight; >0 enables the zero-init residual adapter.",
    )
    parser.add_argument("--card_bottleneck", type=int, default=32)
    parser.add_argument(
        "--card_delta_table",
        default=None,
        help="npz (ids, delta) with per-sample real-vs-shuffled teacher representation deltas.",
    )
    parser.add_argument(
        "--d6_candidate",
        choices=["none", "b0", "b0a", "b1", "o1", "o2", "o3"],
        default="none",
        help="D6 micro-screen candidate identity; enables the per-candidate "
        "init/provenance contracts (review P1-4, §27-§30).",
    )

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
    # D6 fail-fast guards (review P1-6, §62): illegal CARD / shuffle
    # combinations must abort before any model or dataloader is built.
    card_lambda = getattr(params, "card_lambda_delta", 0.0)
    if card_lambda < 0:
        raise ValueError("card_lambda_delta must be >= 0")
    if card_lambda > 0:
        if params.arch != "Graphormer":
            raise ValueError("CARD micro-screen uses the HPS Graphormer architecture")
        if getattr(params, "toxacute_task_scope", "human3") != "human3":
            raise ValueError("CARD is defined only for Human3 fine-tuning")
        if not getattr(params, "card_delta_table", None):
            raise ValueError("--card_delta_table is required when CARD is enabled")
        if not Path(params.card_delta_table).is_file():
            raise FileNotFoundError(f"CARD delta table not found: {params.card_delta_table}")
        if getattr(params, "shuffle_animal_train_labels", False):
            raise ValueError(
                "CARD Human3 runs must not enable --shuffle_animal_train_labels"
            )
    if getattr(params, "shuffle_animal_train_labels", False) and getattr(
        params, "toxacute_task_scope", "all59"
    ) != "animal56":
        raise ValueError(
            "--shuffle_animal_train_labels is only valid with --toxacute_task_scope animal56"
        )
    # D6 candidate-specific contracts (review P1-4, §27-§30).
    d6_candidate = getattr(params, "d6_candidate", "none")
    init_state = getattr(params, "init_state_path", None)
    if d6_candidate != "none":
        if params.dataset != "toxacute":
            raise ValueError("D6 candidates run on the toxacute dataset")
        if params.toxacute_task_scope != "human3":
            raise ValueError("D6 candidates fine-tune the human3 task scope only")
        if params.arch != "Graphormer":
            raise ValueError("D6 candidates use the plain Graphormer architecture")
        if getattr(params, "fit_conformal", True):
            raise ValueError("D6 candidates must run with --no-fit_conformal")
        if getattr(params, "train_eval_scope", "full") != "validation_only":
            raise ValueError("D6 candidates must run with validation-only evaluation")
        if getattr(params, "shuffle_animal_train_labels", False):
            raise ValueError("D6 human candidates must not shuffle animal labels")
    if d6_candidate in {"b0a", "b1", "o1", "o2"}:
        if not init_state:
            raise ValueError(f"--d6_candidate {d6_candidate} requires --init_state_path")
        sidecar = Path(str(init_state) + ".provenance.json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Candidate init provenance sidecar missing: {sidecar}")
        try:
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Init provenance sidecar is not valid JSON: {sidecar}") from exc
        expected_modes = {"b0a": "anchor", "b1": "b1", "o1": "csdt", "o2": "clst"}
        if provenance.get("mode") != expected_modes[d6_candidate]:
            raise ValueError(
                f"--d6_candidate {d6_candidate} requires init mode "
                f"{expected_modes[d6_candidate]!r}, found {provenance.get('mode')!r}"
            )
        if int(provenance.get("human_seed", -1)) != int(params.seed):
            raise ValueError("Init provenance human_seed does not match --seed")
        params.require_init_provenance = True
    if d6_candidate == "b0":
        if init_state:
            raise ValueError("--d6_candidate b0 is a Human3-only scratch run: no --init_state_path")
        if card_lambda > 0:
            raise ValueError("--d6_candidate b0 must not enable CARD")
    if d6_candidate == "o3":
        if init_state:
            raise ValueError("--d6_candidate o3 starts from scratch: no --init_state_path")
        if card_lambda <= 0:
            raise ValueError("--d6_candidate o3 requires --card_lambda_delta > 0")
    params.effective_rgcer_config = resolve_effective_rgcer_config(params)


if __name__ == "__main__":
    parser = build_parser()
    params = parser.parse_args()
    validate_params(params)
    if params.save_path:
        os.makedirs(params.save_path, exist_ok=True)
    main(params)
