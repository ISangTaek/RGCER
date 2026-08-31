"""D6 candidate initialisation utilities (plan §7-§27, §63-§67; review P0-1, P1-2/3/4/12).

Modes:
- csdt : theta_init = theta0 + alpha * (theta_real - theta_shuffle), shared
         Graphormer backbone only; heads stay seed-matched fresh (§64).
- b1   : standard sequential transfer initialisation - copy the real
         teacher's shared backbone into theta0 (diagnostic baseline).
- clst : layer-selective transfer - rank backbone blocks by the
         real-vs-shuffled representation discrepancy S_l computed on human3
         TRAIN molecules only, deduplicated across endpoints (§11-§12, §66,
         review P0-1), and initialise the top-k blocks from the real teacher.
- card_table : precompute the per-sample pooled representation delta
         h_real(x) - h_shuffle(x) for human3 train molecules (§15, §67) and
         save it as the CARD distillation table.

Teacher contract (review P1-2/P1-3): every consumed teacher checkpoint is a
strict last-epoch ``*_last.pt`` whose epoch equals ``--expected_teacher_epoch``,
whose task scope is exactly the 56 animal endpoints, whose architecture is the
plain Graphormer, and whose real/shuffle pair matches on initialisation hash,
base seed, architecture config, task names, split manifest hash, datastore
fingerprint and feature schema.  Only ``shuffle_animal_train_labels`` (and the
weights/metrics it causes) may differ.  Every mode writes a
``<output>.provenance.json`` sidecar (review P1-12).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS
from config import prepare_args
from dataset import DataCollator
from main import (
    _build_model_components,
    _loaders,
    _resolve_data_store,
    build_parser,
    task_names_for_params,
    validate_params,
)
from reproducibility import seed_everything, state_dict_sha256

BACKBONE_PREFIX = "encoder.backbone."
# Review P1-4: architecture/data keys inherited from the teacher checkpoint's
# recorded configuration instead of parser defaults.
ALLOWED_TEMPLATE_KEYS = (
    "data_store_dir",
    "split_seed",
    "splitting",
    "vs",
    "calibration_size",
    "ts",
    "hidden_dim",
    "mid_dim",
    "a_layers",
    "a_heads",
    "edge_bias_mode",
    "spatial_pos_clip",
    "max_nodes_filter",
    "prediction_mode",
    "head_hidden_dim",
    "head_dropout",
)
# The single permitted real-vs-shuffle configuration difference is whether
# the teacher trained on shuffled animal labels (plus the shuffle seed that
# defines the mapping).  Review P0-3: matched-config equality is checked on a
# scientific-field whitelist, never on the whole vars(args) dict —
# orchestration/runtime fields (save_path, experiment_tag, ckpt_name,
# label_providers, mapping hashes, provenance objects) legitimately differ
# between two real teacher runs.
SHUFFLE_ONLY_KEYS = {"shuffle_animal_train_labels", "animal_shuffle_seed"}
MATCHED_TEACHER_CONFIG_KEYS = (
    "arch",
    "dataset",
    "toxacute_task_scope",
    "hidden_dim",
    "mid_dim",
    "a_layers",
    "a_heads",
    "edge_bias_mode",
    "spatial_pos_clip",
    "max_nodes_filter",
    "prediction_mode",
    "head_hidden_dim",
    "head_dropout",
    "lower_quantile",
    "upper_quantile",
    "lambda_quantile",
    "weighting",
    "optim",
    "lr",
    "weight_decay",
    "grad_clip",
    "bs",
    "epochs",
    "task_sampling",
    "num_loader_workers",
    "splitting",
    "vs",
    "calibration_size",
    "ts",
    "split_seed",
    "fit_conformal",
    "train_eval_scope",
    "selection_scope",
    "seed",
)
EXPECTED_ANIMAL_SHUFFLE_SEED = 20260831


def _build_model(seed: int, task_scope: str, device, template_config: dict | None = None):
    """Build the plain-Graphormer model; architecture/data hyper-parameters are
    inherited from the teacher checkpoint configuration when provided (P1-4)."""

    params = build_parser().parse_args([])
    template = template_config or {}
    for key in ALLOWED_TEMPLATE_KEYS:
        if key in template:
            setattr(params, key, template[key])
    setattr(params, "arch", "Graphormer")
    setattr(params, "toxacute_task_scope", task_scope)
    setattr(params, "seed", seed)
    setattr(params, "fit_conformal", False)
    setattr(params, "train_eval_scope", "validation_only")
    setattr(params, "card_lambda_delta", 0.0)
    validate_params(params)
    seed_everything(seed)
    task_names = task_names_for_params(params)
    kwargs, _ = prepare_args(params)
    encoder_class, architecture_class, decoders = _build_model_components(params, task_names, device)
    model = architecture_class(
        task_names, encoder_class, decoders, device, params, **kwargs.get("arch_args", {})
    )
    return model, params, task_names


def _teacher_state_path(run_dir: Path) -> Path:
    """Stage teachers take the FIXED last-epoch parameters (plan §24/§38)."""

    last = sorted(Path(run_dir).glob("*_last.pt"))
    if not last:
        raise FileNotFoundError(f"no *_last.pt under {run_dir}")
    return last[0]


def _load_teacher_checkpoint(run_dir: Path, expected_epoch: int) -> tuple[Path, dict]:
    """Review P1-3: strict checkpoint-level teacher contract.

    D6 teachers must come from validation-only runs without conformal
    fitting — a teacher that has already seen calibration/test (or was
    selected with CQR) must never enter a counterfactual pipeline.
    """

    checkpoint_path = _teacher_state_path(run_dir)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_state") is None:
        raise SystemExit(f"teacher checkpoint {checkpoint_path} has no model_state")
    if int(payload.get("checkpoint_version", -1)) != 6:
        raise SystemExit(
            f"teacher checkpoint {checkpoint_path} is not a strict v6 checkpoint "
            f"(version={payload.get('checkpoint_version')!r})"
        )
    observed_epoch = int(payload.get("epoch", -1))
    if observed_epoch != int(expected_epoch):
        raise SystemExit(
            f"teacher checkpoint epoch mismatch: {checkpoint_path} has epoch "
            f"{observed_epoch}, expected {expected_epoch} (fixed last-epoch contract)"
        )
    configuration = dict(payload.get("configuration") or {})
    if configuration.get("train_eval_scope") != "validation_only":
        raise SystemExit(
            "D6 teacher must be trained with validation-only evaluation scope, found "
            f"{configuration.get('train_eval_scope')!r} (review P1-3)"
        )
    if bool(configuration.get("fit_conformal", True)):
        raise SystemExit("D6 teacher must use --no-fit_conformal (review P1-3)")
    task_names = list(payload.get("task_names", []))
    if sorted(task_names) != sorted(ANIMAL_SOURCE_TASKS):
        human_present = [name for name in task_names if name in HUMAN_TARGET_TASKS]
        raise SystemExit(
            "teacher task scope must be exactly the 56 animal endpoints; found "
            f"{len(task_names)} tasks"
            + (f" (human endpoints present: {human_present})" if human_present else "")
        )
    if payload.get("configuration", {}).get("arch") != "Graphormer":
        raise SystemExit(
            "teacher architecture must be the plain Graphormer, found "
            f"{payload.get('configuration', {}).get('arch')!r}"
        )
    return checkpoint_path, payload


def _verify_teacher_pair(
    real_dir: Path,
    shuffle_dir: Path,
    expected_epoch: int,
    expected_model_seed: int,
) -> dict:
    """Review P1-2/P1-3: matched real/shuffle teacher pair contract."""

    real_path, real_payload = _load_teacher_checkpoint(real_dir, expected_epoch)
    shuffle_path, shuffle_payload = _load_teacher_checkpoint(shuffle_dir, expected_epoch)

    mismatch = []

    def require_equal(field, real_value, shuffle_value):
        if real_value != shuffle_value:
            mismatch.append(f"{field}: real={real_value!r} shuffle={shuffle_value!r}")

    require_equal("task_names", list(real_payload["task_names"]), list(shuffle_payload["task_names"]))
    require_equal(
        "architecture_config",
        real_payload.get("architecture_config"),
        shuffle_payload.get("architecture_config"),
    )
    require_equal(
        "split_manifest_hash",
        real_payload.get("split_manifest_hash"),
        shuffle_payload.get("split_manifest_hash"),
    )
    require_equal(
        "feature_schema_version",
        real_payload.get("feature_schema_version"),
        shuffle_payload.get("feature_schema_version"),
    )
    real_data = dict(real_payload.get("data_config") or {})
    shuffle_data = dict(shuffle_payload.get("data_config") or {})
    require_equal(
        "datastore_fingerprint",
        real_data.get("datastore_fingerprint"),
        shuffle_data.get("datastore_fingerprint"),
    )
    require_equal(
        "reproducibility.base_seed",
        (real_payload.get("reproducibility") or {}).get("base_seed"),
        (shuffle_payload.get("reproducibility") or {}).get("base_seed"),
    )
    # Review P1-1: the teacher pair must also be trained at the same model
    # seed as the human run (Stage B/C seed-paired protocol).
    for label in ("real", "shuffle"):
        payload = real_payload if label == "real" else shuffle_payload
        pair_seed = (payload.get("reproducibility") or {}).get("base_seed")
        if pair_seed is not None and int(pair_seed) != int(expected_model_seed):
            mismatch.append(
                f"{label} teacher base seed {pair_seed} != expected model seed {expected_model_seed}"
            )
    real_configuration = dict(real_payload.get("configuration") or {})
    shuffle_configuration = dict(shuffle_payload.get("configuration") or {})
    # Review P0-3: compare only the scientific training fields; orchestration
    # fields (save_path, experiment_tag, ckpt_name, label_providers, mapping
    # hashes, ...) legitimately differ between two real teacher runs.
    for key in MATCHED_TEACHER_CONFIG_KEYS:
        require_equal(
            f"configuration.{key}",
            real_configuration.get(key),
            shuffle_configuration.get(key),
        )
    if real_configuration.get("shuffle_animal_train_labels") is not False:
        mismatch.append(
            "real teacher must be trained on real labels (shuffle_animal_train_labels=False), found "
            f"{real_configuration.get('shuffle_animal_train_labels')!r}"
        )
    if shuffle_configuration.get("shuffle_animal_train_labels") is not True:
        mismatch.append(
            "shuffle teacher must be trained with shuffle_animal_train_labels=True, found "
            f"{shuffle_configuration.get('shuffle_animal_train_labels')!r}"
        )
    if int(shuffle_configuration.get("animal_shuffle_seed", -1)) != EXPECTED_ANIMAL_SHUFFLE_SEED:
        mismatch.append(
            f"shuffle teacher animal_shuffle_seed must be {EXPECTED_ANIMAL_SHUFFLE_SEED}, found "
            f"{shuffle_configuration.get('animal_shuffle_seed')!r}"
        )

    init_hashes = {}
    metadata_by_label = {}
    for label, run_dir in (("real", real_dir), ("shuffle", shuffle_dir)):
        metadata_path = Path(run_dir) / "run_metadata.json"
        if not metadata_path.exists():
            raise SystemExit(f"missing run_metadata.json under {run_dir}")
        metadata_by_label[label] = json.loads(metadata_path.read_text(encoding="utf-8"))
        init_hashes[label] = metadata_by_label[label].get("initial_model_sha256")
    # Review P1-2: a missing initial hash on either side makes the
    # counterfactual delta unverifiable — fail fast instead of passing on
    # None == None.
    for label in ("real", "shuffle"):
        if not init_hashes[label]:
            mismatch.append(f"{label} teacher lacks initial_model_sha256")
    require_equal("initial_model_sha256", init_hashes["real"], init_hashes["shuffle"])

    # Review P1-11: cross-check the shuffle manifest file against the hash
    # recorded in the run metadata — provenance is more than "a string".
    shuffle_manifest_path = Path(shuffle_dir) / "D6_ANIMAL_SHUFFLE_MANIFEST.json"
    if shuffle_manifest_path.exists():
        manifest = json.loads(shuffle_manifest_path.read_text(encoding="utf-8"))
        require_equal(
            "D6_ANIMAL_SHUFFLE_MANIFEST.animal_shuffle_mapping_sha256",
            manifest.get("animal_shuffle_mapping_sha256"),
            metadata_by_label["shuffle"].get("animal_shuffle_mapping_sha256"),
        )
        require_equal(
            "D6_ANIMAL_SHUFFLE_MANIFEST.animal_shuffle_seed",
            manifest.get("animal_shuffle_seed"),
            metadata_by_label["shuffle"].get("animal_shuffle_seed"),
        )

    # Review P1-2: the shuffle mapping identity is part of the counterfactual
    # contract — the shuffle teacher must record it.
    shuffle_metadata = metadata_by_label["shuffle"]
    mapping_hash = shuffle_metadata.get("animal_shuffle_mapping_sha256")
    if not mapping_hash:
        mismatch.append(
            "shuffle teacher run_metadata.json lacks animal_shuffle_mapping_sha256"
        )

    if mismatch:
        raise SystemExit(
            "teacher pair is not matched (review §59-§60):\n  - " + "\n  - ".join(mismatch)
        )

    return {
        "teacher_real_checkpoint": str(real_path),
        "teacher_real_checkpoint_sha256": _sha256_file(real_path),
        "teacher_real_epoch": int(real_payload["epoch"]),
        "teacher_shuffle_checkpoint": str(shuffle_path),
        "teacher_shuffle_checkpoint_sha256": _sha256_file(shuffle_path),
        "teacher_shuffle_epoch": int(shuffle_payload["epoch"]),
        "teacher_initial_model_sha256": init_hashes["real"],
        "split_manifest_hash": real_payload.get("split_manifest_hash"),
        "feature_schema_version": real_payload.get("feature_schema_version"),
        "datastore_fingerprint": real_data.get("datastore_fingerprint"),
        "animal_shuffle_seed": shuffle_configuration.get("animal_shuffle_seed"),
        "animal_shuffle_mapping_sha256": mapping_hash,
        "teacher_configuration": real_configuration,
        "architecture_config": real_payload.get("architecture_config"),
    }


def _b1_teacher_contract(real_payload: dict, real_dir: Path, human_seed: int, expected_epoch: int) -> None:
    """Review §31: B1 skips the shuffle teacher but still checks scope/seed/epoch."""

    task_names = list(real_payload.get("task_names", []))
    if sorted(task_names) != sorted(ANIMAL_SOURCE_TASKS):
        raise SystemExit("B1 teacher task scope must be exactly the 56 animal endpoints")
    if real_payload.get("configuration", {}).get("arch") != "Graphormer":
        raise SystemExit("B1 teacher architecture must be the plain Graphormer")
    if int(real_payload.get("epoch", -1)) != int(expected_epoch):
        raise SystemExit(
            f"B1 teacher epoch mismatch: found {real_payload.get('epoch')}, "
            f"expected {expected_epoch}"
        )
    base_seed = (real_payload.get("reproducibility") or {}).get("base_seed")
    if base_seed is not None and int(base_seed) != int(human_seed):
        raise SystemExit(
            f"B1 teacher base seed {base_seed} != human seed {human_seed} (review §31)"
        )
    metadata_path = Path(real_dir) / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("initial_model_sha256") is None:
            raise SystemExit("B1 teacher run_metadata.json lacks initial_model_sha256")


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(output: Path, provenance: dict) -> Path:
    """Review P1-12: full provenance sidecar for every generated init artifact."""

    provenance = dict(provenance)
    provenance["output"] = str(output)
    provenance["output_sha256"] = _sha256_file(output)
    sidecar = output.with_name(output.name + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {sidecar}")
    return sidecar


def _write_rows(path: str | None, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def apply_csdt(theta0: dict, real: dict, shuffle: dict, alpha: float) -> dict:
    """theta_init = theta0 + alpha * (real - shuffle) on backbone params (§64).

    Review P0-4: ``theta0`` must be the *teacher-coordinate anchor* (see
    ``build_anchor_state``), never the human3-native random initialisation —
    weight-space task vectors are only additive within one random coordinate
    system.
    """

    merged = {key: value.detach().clone() for key, value in theta0.items()}
    for key in merged:
        if key.startswith(BACKBONE_PREFIX) and key in real and key in shuffle:
            merged[key] = theta0[key] + alpha * (real[key].float() - shuffle[key].float())
    return merged


def apply_b1(theta0: dict, real: dict) -> dict:
    """Sequential transfer: copy the real teacher's whole shared backbone."""

    merged = {key: value.detach().clone() for key, value in theta0.items()}
    for key in merged:
        if key.startswith(BACKBONE_PREFIX) and key in real:
            merged[key] = real[key].detach().clone()
    return merged


def _regenerate_teacher_anchor(human_seed: int, template_config: dict | None, expected_initial_model_sha256: str):
    """Review P0-4 (§19): rebuild the teachers' true common starting point.

    The Animal56 teacher construction consumes a different amount of RNG than
    the human3 construction (56 vs 3 decoders first), so a same-seed human3
    model does NOT share the teacher backbone coordinate.  This helper
    reproduces the teacher's initial model exactly and fails loudly when the
    reproduction is imperfect.
    """

    animal_init_model, _, _ = _build_model(human_seed, "animal56", torch.device("cpu"), template_config)
    observed_hash = state_dict_sha256(animal_init_model)
    if observed_hash != expected_initial_model_sha256:
        raise SystemExit(
            "Could not reproduce teacher initial model exactly "
            f"(expected {expected_initial_model_sha256}, observed {observed_hash})"
        )
    return animal_init_model


def build_anchor_state(model_human, animal_init_model) -> dict:
    """Review §20: human heads stay fresh; backbone = teacher initial coordinate."""

    animal_state = animal_init_model.state_dict()
    anchor = {key: value.detach().clone() for key, value in model_human.state_dict().items()}
    for key in anchor:
        if key.startswith(BACKBONE_PREFIX) and key in animal_state:
            anchor[key] = animal_state[key].detach().clone()
    return anchor


@torch.no_grad()
def clst_layer_scores(model_real, model_shuffle, train_loaders: dict, device) -> tuple[list[dict], int]:
    """S_l = E||h_real - h_shuffle|| / (E||h_real|| + eps) per block.

    Review P0-1: consumes the human3 **train** loader dict directly (never
    val/calibration/test), deduplicates molecules shared across endpoints, and
    always removes the representation hooks.
    """

    layers = model_real.encoder.backbone.layers
    n_layers = len(layers)
    acts = {"real": {}, "shuffle": {}}
    hooks_real, hooks_shuffle = [], []

    def make_hook(store, layer_index):
        def hook(_module, _inputs, output):
            store[layer_index] = output[:, 0, :].detach().float().cpu()
        return hook

    for index, layer in enumerate(layers):
        hooks_real.append(layer.register_forward_hook(make_hook(acts["real"], index)))
        hooks_shuffle.append(layer.register_forward_hook(make_hook(acts["shuffle"], index)))

    diff_sums = np.zeros(n_layers)
    real_norm_sums = np.zeros(n_layers)
    seen_sample_ids: set[str] = set()
    model_real.eval()
    model_shuffle.eval()
    try:
        for task in HUMAN_TARGET_TASKS:
            loader = train_loaders.get(task)
            if loader is None:
                continue
            for batch in loader:
                if getattr(batch, "get", lambda *_: None)("is_empty", False):
                    continue
                sample_ids = [str(value) for value in (getattr(batch, "sample_id", None) or [])]
                unseen_positions = [
                    position
                    for position, sample_id in enumerate(sample_ids)
                    if sample_id not in seen_sample_ids
                ]
                if not unseen_positions:
                    continue
                batch = batch.to(device)
                model_real.encoder.backbone(batch)
                model_shuffle.encoder.backbone(batch)
                index_tensor = torch.tensor(unseen_positions, dtype=torch.long)
                for layer_index in range(n_layers):
                    real_pooled = acts["real"][layer_index][index_tensor]
                    shuffle_pooled = acts["shuffle"][layer_index][index_tensor]
                    diff_sums[layer_index] += float((real_pooled - shuffle_pooled).norm(dim=-1).sum())
                    real_norm_sums[layer_index] += float(real_pooled.norm(dim=-1).sum())
                for position in unseen_positions:
                    seen_sample_ids.add(sample_ids[position])
    finally:
        for hook in hooks_real + hooks_shuffle:
            hook.remove()

    rows = []
    unique = max(len(seen_sample_ids), 1)
    for layer_index in range(n_layers):
        score = (diff_sums[layer_index] / unique) / (real_norm_sums[layer_index] / unique + 1e-8)
        rows.append(
            {
                "layer": layer_index,
                "real_minus_shuffle_norm_mean": float(diff_sums[layer_index] / unique),
                "real_rep_norm_mean": float(real_norm_sums[layer_index] / unique),
                "semantic_layer_score": float(score),
            }
        )
    rows.sort(key=lambda row: row["semantic_layer_score"], reverse=True)
    for rank, row in enumerate(rows):
        row["rank"] = rank
    return rows, len(seen_sample_ids)


@torch.no_grad()
def card_delta_table(model_real, model_shuffle, train_loaders: dict, task_names, device) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample pooled-representation delta over human3 train molecules (§15)."""

    model_real.eval()
    model_shuffle.eval()
    ids: list[str] = []
    deltas: list[np.ndarray] = []
    seen: set[str] = set()
    for task in task_names:
        loader = train_loaders.get(task)
        if loader is None:
            continue
        for batch in loader:
            if getattr(batch, "get", lambda *_: None)("is_empty", False):
                continue
            batch = batch.to(device)
            sample_ids = [str(value) for value in (getattr(batch, "sample_id", None) or [])]
            real_repr = model_real.encoder(batch)
            shuffle_repr = model_shuffle.encoder(batch)
            delta = (real_repr - shuffle_repr).detach().float().cpu().numpy()
            for index, sample_id in enumerate(sample_ids):
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                ids.append(sample_id)
                deltas.append(delta[index])
    if not deltas:
        raise RuntimeError("card_delta_table produced no rows - human3 train loaders were empty")
    return np.asarray(ids), np.stack(deltas)


def _human_loaders(params):
    """Build the human3 train/val loader dict (review P0-1: single helper so
    the CLST/CARD branches cannot reference an undefined collator)."""

    collator = DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip,
        max_node_filter=None,
    )
    return _loaders(params, list(HUMAN_TARGET_TASKS), collator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["anchor", "csdt", "clst", "b1", "card_table"]
    )
    parser.add_argument("--teacher_real_dir", required=True)
    parser.add_argument("--teacher_shuffle_dir", default=None)
    parser.add_argument(
        "--expected_teacher_epoch",
        type=int,
        required=True,
        help="Fixed last-epoch teacher contract: Stage A=29, Stage B=39 (review §30).",
    )
    parser.add_argument("--human_seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--clst_top_k", type=int, default=2)
    parser.add_argument("--gpu_id", default="cpu")
    parser.add_argument("--output", required=True, help="init state .pt (or CARD npz table)")
    parser.add_argument("--delta_csv", default=None, help="CSDT/CLST per-layer diagnostic CSV")
    args = parser.parse_args()

    device = torch.device(args.gpu_id if args.gpu_id != "cpu" and torch.cuda.is_available() else "cpu")
    teacher_real_dir = Path(args.teacher_real_dir)
    teacher_shuffle_dir = Path(args.teacher_shuffle_dir) if args.teacher_shuffle_dir else None
    if args.mode in {"csdt", "clst", "card_table"} and teacher_shuffle_dir is None:
        raise SystemExit(f"{args.mode} requires --teacher_shuffle_dir")
    if args.mode == "card_table" and not str(args.output).endswith(".npz"):
        # Review P1-4: np.savez silently appends .npz, which would break the
        # provenance sidecar path.
        raise SystemExit("CARD table --output must end with .npz")

    provenance: dict = {
        "mode": args.mode,
        "human_seed": args.human_seed,
        "alpha": args.alpha,
        "clst_top_k": args.clst_top_k,
        "expected_teacher_epoch": args.expected_teacher_epoch,
        "teacher_real_run_dir": str(teacher_real_dir),
        "teacher_shuffle_run_dir": str(teacher_shuffle_dir) if teacher_shuffle_dir else None,
    }

    if args.mode == "card_table":
        # Review P1-2: CARD deltas are only meaningful for a matched pair.
        matched = _verify_teacher_pair(
            teacher_real_dir,
            teacher_shuffle_dir,
            args.expected_teacher_epoch,
            expected_model_seed=args.human_seed,
        )
        provenance.update(matched)
        template_config = matched["teacher_configuration"]
        model_real, _, teacher_tasks = _build_model(args.human_seed, "animal56", device, template_config)
        model_real.load_state_dict(
            _load_teacher_checkpoint(teacher_real_dir, args.expected_teacher_epoch)[1]["model_state"],
            strict=True,
        )
        model_real.to(device).eval()
        model_shuffle, _, _ = _build_model(args.human_seed, "animal56", device, template_config)
        model_shuffle.load_state_dict(
            _load_teacher_checkpoint(teacher_shuffle_dir, args.expected_teacher_epoch)[1]["model_state"],
            strict=True,
        )
        model_shuffle.to(device).eval()

        params = build_parser().parse_args([])
        template = matched["teacher_configuration"]
        for key in ALLOWED_TEMPLATE_KEYS:
            if key in template:
                setattr(params, key, template[key])
        setattr(params, "arch", "Graphormer")
        setattr(params, "toxacute_task_scope", "human3")
        setattr(params, "seed", args.human_seed)
        setattr(params, "fit_conformal", False)
        setattr(params, "train_eval_scope", "validation_only")
        setattr(params, "card_lambda_delta", 0.0)
        validate_params(params)
        # Review P0-2: iterate the human3 train loaders (keyed by the three
        # human endpoints) — the 56 animal task names must never be used as
        # loader keys here.
        loaders = _human_loaders(params)
        ids, deltas = card_delta_table(
            model_real, model_shuffle, loaders["train"], list(HUMAN_TARGET_TASKS), device
        )
        output = Path(args.output)
        np.savez(output, ids=ids, delta=deltas)
        print(f"card delta table: {len(ids)} samples x {deltas.shape[1]} dims -> {output}")
        _write_provenance(output, provenance)
        return

    # theta0: fresh seed-matched human3 model.  Architecture/data settings are
    # inherited from the real teacher checkpoint configuration (review P1-4).
    real_checkpoint_path, real_checkpoint = _load_teacher_checkpoint(
        teacher_real_dir, args.expected_teacher_epoch
    )
    template_config = dict(real_checkpoint.get("configuration") or {})
    provenance["teacher_real_checkpoint"] = str(real_checkpoint_path)
    provenance["teacher_real_epoch"] = int(real_checkpoint["epoch"])
    provenance["teacher_real_checkpoint_sha256"] = _sha256_file(real_checkpoint_path)
    matched_init_hash = None

    if args.mode != "b1":
        matched = _verify_teacher_pair(
            teacher_real_dir,
            teacher_shuffle_dir,
            args.expected_teacher_epoch,
            expected_model_seed=args.human_seed,
        )
        provenance.update(matched)
        matched_init_hash = matched["teacher_initial_model_sha256"]
        template_config = matched["teacher_configuration"]
    else:
        _b1_teacher_contract(real_checkpoint, teacher_real_dir, args.human_seed, args.expected_teacher_epoch)
        metadata_path = teacher_real_dir / "run_metadata.json"
        # Review P1-3: formal teachers must come from auditable runs.
        if not metadata_path.exists():
            raise SystemExit(f"B1 teacher is missing run_metadata.json under {teacher_real_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("initial_model_sha256"):
            raise SystemExit("B1 teacher run_metadata.json lacks initial_model_sha256")
        provenance["teacher_initial_model_sha256"] = metadata["initial_model_sha256"]

    shuffle_checkpoint_path = None
    if teacher_shuffle_dir is not None:
        shuffle_checkpoint_path, shuffle_checkpoint = _load_teacher_checkpoint(
            teacher_shuffle_dir, args.expected_teacher_epoch
        )
        provenance["teacher_shuffle_checkpoint"] = str(shuffle_checkpoint_path)
        provenance["teacher_shuffle_checkpoint_sha256"] = _sha256_file(shuffle_checkpoint_path)

    model_human, human_params, _ = _build_model(
        args.human_seed, "human3", torch.device("cpu"), template_config
    )
    theta0 = {key: value.detach().clone() for key, value in model_human.state_dict().items()}

    # Review P0-4: regenerate the teachers' common initial backbone and build
    # the teacher-coordinate anchor (heads fresh, backbone = teacher init).
    theta_anchor = None
    if args.mode in {"anchor", "csdt"}:
        expected_init_hash = (
            matched_init_hash
            if args.mode == "csdt"
            else json.loads((teacher_real_dir / "run_metadata.json").read_text(encoding="utf-8")).get(
                "initial_model_sha256"
            )
        )
        if not expected_init_hash:
            raise SystemExit("teacher run_metadata.json lacks initial_model_sha256")
        animal_init_model = _regenerate_teacher_anchor(
            args.human_seed, template_config, expected_init_hash
        )
        theta_anchor = build_anchor_state(model_human, animal_init_model)
        provenance["teacher_anchor_initial_model_sha256"] = expected_init_hash

    model_real, _, _ = _build_model(args.human_seed, "animal56", torch.device("cpu"), template_config)
    model_real.load_state_dict(real_checkpoint["model_state"], strict=True)
    state_real = model_real.state_dict()

    if args.mode == "anchor":
        # §22-§23: B0A anchor-only control — no semantic delta applied.
        merged = theta_anchor
    elif args.mode == "csdt":
        # Review P0-4/§21: CSDT must apply the counterfactual delta in the
        # teacher coordinate (alpha=0 reproduces the anchor exactly).
        model_shuffle, _, _ = _build_model(args.human_seed, "animal56", torch.device("cpu"), template_config)
        model_shuffle.load_state_dict(shuffle_checkpoint["model_state"], strict=True)
        state_shuffle = model_shuffle.state_dict()
        merged = apply_csdt(theta_anchor, state_real, state_shuffle, args.alpha)

        if args.delta_csv:
            # Review P2: the semantic delta is applied on top of the anchor,
            # so the ratio denominator is the anchor backbone norm.
            rows = []
            for key in sorted(theta_anchor):
                if not key.startswith(BACKBONE_PREFIX):
                    continue
                anchor_norm = float(theta_anchor[key].float().norm())
                real_norm = float(state_real[key].float().norm())
                shuffle_norm = float(state_shuffle[key].float().norm())
                delta_norm = float((state_real[key].float() - state_shuffle[key].float()).norm())
                rows.append(
                    {
                        "layer": key,
                        "parameter_group": key,
                        "anchor_norm": anchor_norm,
                        "real_norm": real_norm,
                        "shuffle_norm": shuffle_norm,
                        "semantic_delta_norm": delta_norm,
                        "semantic_delta_ratio": (delta_norm / anchor_norm) if anchor_norm > 0 else float("nan"),
                        "alpha": args.alpha,
                    }
                )
            _write_rows(args.delta_csv, rows)
            anomalous = [row for row in rows if row["semantic_delta_ratio"] > 1.0]
            provenance["anomalous_delta_ratio_tensors"] = len(anomalous)
            print(f"csdt: {len(rows)} backbone tensors; anomalous ratio>1: {len(anomalous)}")
    elif args.mode == "clst":
        model_shuffle, _, _ = _build_model(args.human_seed, "animal56", torch.device("cpu"), template_config)
        model_shuffle.load_state_dict(shuffle_checkpoint["model_state"], strict=True)

        seed_everything(args.human_seed)
        loaders = _human_loaders(human_params)
        model_real.to(device)
        model_shuffle.to(device)
        score_rows, n = clst_layer_scores(model_real, model_shuffle, loaders["train"], device)
        selected = {row["layer"] for row in score_rows[: args.clst_top_k]}

        merged = {key: value.detach().clone() for key, value in theta0.items()}
        for key in merged:
            if key.startswith(BACKBONE_PREFIX + "layers."):
                layer_index = int(key.split(".")[3])
                if layer_index in selected:
                    merged[key] = state_real[key].detach().clone()
        for row in score_rows:
            row["selected"] = int(row["layer"] in selected)
        if args.delta_csv:
            _write_rows(
                args.delta_csv,
                [
                    {
                        "seed": args.human_seed,
                        "layer": row["layer"],
                        "real_rep_norm": row["real_rep_norm_mean"],
                        "real_minus_shuffle_norm": row["real_minus_shuffle_norm_mean"],
                        "semantic_layer_score": row["semantic_layer_score"],
                        "rank": row["rank"],
                        "selected": row["selected"],
                    }
                    for row in score_rows
                ],
            )
        provenance["clst_unique_train_molecules"] = n
        provenance["clst_selected_layers"] = sorted(selected)
        print(
            f"clst: scored {len(score_rows)} layers on {n} unique train molecules; "
            f"selected {sorted(selected)}"
        )
    else:  # b1
        merged = apply_b1(theta0, state_real)

    model_human.load_state_dict(merged, strict=True)
    torch.save(model_human.state_dict(), args.output)
    provenance["init_state_sha256"] = state_dict_sha256(model_human)
    _write_provenance(Path(args.output), provenance)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "human_seed": args.human_seed,
                "alpha": args.alpha,
                "teacher_matched_init_hash": matched_init_hash,
                "init_state_sha256": provenance["init_state_sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
