"""D8 result-correction Task E: joint/reference-projected embedding analysis.

The formal-phase embeddings compared B1/S1 in INDEPENDENT PCA fits over
Human3 validation molecules only — coordinates are not comparable across
fits and no animal molecules were included, so the earlier artefacts cannot
support any "cross-species space" claim.  This script rebuilds the analysis
properly (§43-§57):

Selection (deterministic, §45-§46)
    Animal56 representative molecules: per animal endpoint at most 20
    unique validation molecules, globally capped at 500 unique ids, ranked
    by sha256(sample_id); manifest CSV + JSON with the manifest hash.

Extraction (§47)
    Per seed, three encoder states over the SAME pooled backbone layer:
      animal_pretrained : the seed's b1_init artifact (pre-finetuning)
      s1_final          : the S1 formal run's final checkpoint
      b1_final          : the B1 formal run's final checkpoint
    Animal probe molecules are embedded through animal_pretrained; Human3
    validation molecules through all three states.

Analysis
    - Reference-projected PCA (§48): fit on ALL molecules embedded through
      the animal_pretrained state, then transform every group into that
      same space (no per-state fits).
    - Joint PCA as a secondary fit (§50).
    - Human representation movement (§52): ||h_B1 - h_pre||, cosine, and
      the S1 counterpart (expected ~0).
    - Animal-neighborhood retention (§53/§55): k=10 nearest animal
      molecules in pretrained space vs B1 space, Jaccard overlap.
    - §54 cross-species label similarity is SKIPPED: human TDLo and animal
      LD50/LDLo units have no validated mapping (plan allows skipping).

Outputs (under --output_dir/embedding):
    D8_EMBEDDING_PROBE_MANIFEST.csv/.json
    D8_JOINT_EMBEDDING_MATRIX.npy / D8_JOINT_EMBEDDING_METADATA.csv
    D8_REFERENCE_PCA_COORDINATES.csv / D8_REFERENCE_PCA_MODEL.pkl
    D8_JOINT_PCA_COORDINATES.csv
    D8_HUMAN_REPRESENTATION_MOVEMENT.csv
    D8_CROSS_SPECIES_NEIGHBOR_RETENTION.csv

Figure PDFs are rendered OFFLINE (the server has no matplotlib); the
coordinate CSVs above are the server-side deliverables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_PER_TASK = 20
MAX_TOTAL = 500
K_NEIGHBORS = 10
MIN_ANIMAL_POOL = 30


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def select_animal_probe(store, animal_tasks) -> list[dict]:
    """Deterministic Animal56 validation representative selection (§45)."""
    import numpy as np

    candidates: dict[str, dict] = {}
    for task in animal_tasks:
        indices = store.get_task_indices(task, split="validation")
        for global_index in indices:
            sample_id = store.sample_ids[int(global_index)]
            sample_id = sample_id.decode() if isinstance(sample_id, bytes) else str(sample_id)
            if sample_id in candidates:
                candidates[sample_id]["tasks"].add(task)
                continue
            label = store.get_label(int(global_index), task)
            candidates[sample_id] = {
                "sample_id": sample_id,
                "tasks": {task},
                "split": "validation",
                "label": float(label),
                "rank_key": sha256_text(sample_id),
            }
    ordered = sorted(candidates.values(), key=lambda item: (item["rank_key"], item["sample_id"]))
    selected, seen = [], set()
    per_task_counts = {task: 0 for task in animal_tasks}
    for candidate in ordered:
        if len(selected) >= MAX_TOTAL:
            break
        task_added = False
        for task in sorted(candidate["tasks"]):
            if per_task_counts[task] >= MAX_PER_TASK:
                continue
            per_task_counts[task] += 1
            task_added = True
            break
        if not task_added:
            continue
        selected.append(candidate)
        seen.add(candidate["sample_id"])
    rows = []
    for rank, candidate in enumerate(selected):
        rows.append(
            {
                "species_group": "animal",
                "task": "|".join(sorted(candidate["tasks"])),
                "sample_id": candidate["sample_id"],
                "split": candidate["split"],
                "selection_rank": rank,
                "label": candidate["label"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--inits_dir", default="artifacts/inits")
    parser.add_argument(
        "--output_dir", default="artifacts/results/d8_result_correction/embedding"
    )
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    import numpy as np
    import torch
    from sklearn.decomposition import PCA

    from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS
    from d7_diagnostics import pooled_representations

    if str(args.gpu_id) != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- selection manifest (stateless across seeds) ----------------------
    from toxacute_datastore import ToxAcuteDataStore

    from scripts.d8_drift_recompute import (
        POOLING_NAME, _params_from_config, last_checkpoint, sha256_file,
    )

    # a scaling run's args supplies the shared data conventions
    args_json = Path(
        "artifacts/runs/d8_label_scaling/b1/d8_b1_f50_e40/seed_42/args.json"
    )
    run_args = json.loads(args_json.read_text(encoding="utf-8"))
    store = ToxAcuteDataStore.resolve(run_args.get("data_store_dir"))

    probe_rows = select_animal_probe(store, list(ANIMAL_SOURCE_TASKS))
    probe_ids = [row["sample_id"] for row in probe_rows]
    manifest_csv = output_dir / "D8_EMBEDDING_PROBE_MANIFEST.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["species_group", "task", "sample_id", "split",
                        "selection_rank", "label"],
        )
        writer.writeheader()
        writer.writerows(probe_rows)
    manifest_json = {
        "manifest_sha256": sha256_bytes(manifest_csv.read_bytes()),
        "selection_rule": (
            f"Animal56 validation molecules, ranked by sha256(sample_id), "
            f"<= {MAX_PER_TASK} per endpoint, global cap {MAX_TOTAL} unique ids"
        ),
        "max_per_task": MAX_PER_TASK,
        "max_total": MAX_TOTAL,
        "n_selected": len(probe_rows),
        "n_tasks": len(ANIMAL_SOURCE_TASKS),
        "test_accessed": False,
    }
    (output_dir / "D8_EMBEDDING_PROBE_MANIFEST.json").write_text(
        json.dumps(manifest_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"animal probe: {len(probe_rows)} molecules "
          f"({len(set(probe_ids))} unique) -> manifest written")

    # ---- loaders for the two molecule pools (validation only) -------------
    from dataset import DataCollator
    from main import _loaders, task_names_for_params

    params = _params_from_config(run_args)
    collator = DataCollator(spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None)

    human_params = _params_from_config(run_args)
    human_loaders = _loaders(
        human_params, task_names_for_params(human_params), collator
    )["val"]
    animal_params = _params_from_config(run_args)
    setattr(animal_params, "toxacute_task_scope", "animal56")
    animal_loaders = _loaders(animal_params, list(ANIMAL_SOURCE_TASKS), collator)["val"]

    def collect_ids_and_items(loaders_by_task, wanted=None):
        items, labels = {}, {}
        for task in sorted(loaders_by_task):
            loader = loaders_by_task[task]
            dataset = loader.dataset
            for index in range(len(dataset)):
                sample_id = (
                    dataset.get_sample_id(index)
                    if hasattr(dataset, "get_sample_id")
                    else str(getattr(dataset[index], "sample_id"))
                )
                sample_id = str(sample_id)
                if wanted is not None and sample_id not in wanted:
                    continue
                if sample_id in items:
                    labels[sample_id].add(task)
                    continue
                items[sample_id] = dataset[index]
                labels[sample_id] = {task}
        return items, labels

    animal_items, animal_tasks_by_id = collect_ids_and_items(
        animal_loaders, wanted=set(probe_ids)
    )
    missing = set(probe_ids) - set(animal_items)
    if missing:
        raise RuntimeError(f"animal probe molecules not found in val loaders: {sorted(missing)[:3]}")
    human_items, human_tasks_by_id = collect_ids_and_items(human_loaders)
    print(f"pools: animal={len(animal_items)} human_val={len(human_items)}")

    # ---- review P0-1: source/target overlap audit BEFORE any embedding ----
    from experiment_config import TOXACUTE_RAW_CSV
    from scripts.d8_embedding_overlap_audit import run_overlap_audit
    from scripts.d8_label_scaling_subset_audit import load_smiles_by_row

    smiles_by_row = load_smiles_by_row(Path(TOXACUTE_RAW_CSV))
    smiles_by_id = {
        sample_id: smiles_by_row.get(sample_id, "")
        for sample_id in set(probe_ids) | set(human_items)
    }
    run_overlap_audit(
        human_ids=sorted(human_items),
        animal_ids=probe_ids,
        smiles_by_id=smiles_by_id,
        output_path=output_dir / "D8_EMBEDDING_OVERLAP_AUDIT.json",
    )

    # ---- review P0-2: shared provenance for every embedding row -----------
    datastore_sha256 = store.fingerprint
    manifest_sha256 = store.split_manifest_hash

    human_ids = sorted(human_items)
    animal_batch = collator([animal_items[sid] for sid in probe_ids])
    human_batch = collator([human_items[sid] for sid in human_ids])

    def embed_state(model, batch, ids):
        reps = pooled_representations(model, [batch], device, expected_ids=ids)
        return np.stack([reps[sid].numpy() for sid in ids])

    def init_artifact_for(seed: int) -> Path:
        for candidate in (
            Path(args.inits_dir) / "d7" / f"b1_init_seed{seed}.pt",
            Path(args.inits_dir) / "d8_stage0" / f"b1_init_seed{seed}.pt",
            Path(args.inits_dir) / "d7_stage_b" / f"b1_init_seed{seed}.pt",
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no b1_init artifact for seed {seed}")

    def run_payload(method: str, seed: int):
        if method == "s1":
            run_dir = Path(args.d7_root) / "d7_stage_b" / "s1" / "d7_s1_e40" / f"seed_{seed}"
        else:
            run_dir = Path(args.d6_root) / "d6_stage_b" / method / f"d6_{method}_e40" / f"seed_{seed}"
        return last_checkpoint(run_dir)

    matrix_rows = []      # one entry per (seed, encoder_state, sample_id)
    metadata_rows = []
    template_config = dict(run_args)

    for seed in args.seeds:
        states = {}
        state_source_file = {}
        init_artifact = init_artifact_for(seed)
        init_state = torch.load(init_artifact, map_location="cpu", weights_only=False)
        states["animal_pretrained"] = init_state
        state_source_file["animal_pretrained"] = init_artifact
        s1_payload, s1_ckpt = run_payload("s1", seed)
        b1_payload, b1_ckpt = run_payload("b1", seed)
        states["s1_final"] = s1_payload["model_state"]
        state_source_file["s1_final"] = s1_ckpt
        states["b1_final"] = b1_payload["model_state"]
        state_source_file["b1_final"] = b1_ckpt

        state_checkpoint_sha256 = {
            name: sha256_file(path) for name, path in state_source_file.items()
        }

        embeddings = {}
        for state_name, state in states.items():
            model = None
            from scripts.d8_drift_recompute import load_model_from_state

            model = load_model_from_state(state, template_config, device)
            if state_name == "animal_pretrained":
                embeddings[state_name] = {
                    "animal": embed_state(model, animal_batch, probe_ids),
                    "human": embed_state(model, human_batch, human_ids),
                }
            else:
                embeddings[state_name] = {
                    "human": embed_state(model, human_batch, human_ids),
                }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # s1_final must equal animal_pretrained on the same molecules (§52)
        max_cos_diff = float(
            np.max(np.abs(
                1 - np.sum(
                    embeddings["s1_final"]["human"]
                    * embeddings["animal_pretrained"]["human"], axis=1
                )
                / (
                    np.linalg.norm(embeddings["s1_final"]["human"], axis=1)
                    * np.linalg.norm(embeddings["animal_pretrained"]["human"], axis=1)
                    + 1e-12
                )
            ))
        )
        print(f"seed {seed}: max |1 - cos(s1, pretrained)| = {max_cos_diff:.3e}")

        for state_name, per_group in embeddings.items():
            for group, ids in (("animal", probe_ids), ("human", human_ids)):
                if group not in per_group:
                    continue
                for row_index, sample_id in enumerate(ids):
                    tasks = sorted(animal_tasks_by_id.get(sample_id, set())) if group == "animal" \
                        else sorted(human_tasks_by_id.get(sample_id, set()))
                    label = ""
                    if group == "animal":
                        label = next(
                            (row["label"] for row in probe_rows if row["sample_id"] == sample_id),
                            "",
                        )
                    matrix_rows.append(per_group[group][row_index])
                    metadata_rows.append(
                        {
                            "seed": seed,
                            "encoder_state": state_name,
                            "species_group": group,
                            "task": "|".join(tasks),
                            "sample_id": sample_id,
                            "split": "validation",
                            "label": label,
                            # review P0-2: full encoder provenance per row
                            "checkpoint_sha256": state_checkpoint_sha256[state_name],
                            "datastore_sha256": datastore_sha256,
                            "manifest_sha256": manifest_sha256,
                            "pooling_name": POOLING_NAME,
                            "embedding_dim": int(per_group[group][row_index].shape[0]),
                        }
                    )

    matrix = np.stack(matrix_rows)
    np.save(output_dir / "D8_JOINT_EMBEDDING_MATRIX.npy", matrix)
    with (output_dir / "D8_JOINT_EMBEDDING_METADATA.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "encoder_state", "species_group", "task",
                        "sample_id", "split", "label",
                        "checkpoint_sha256", "datastore_sha256",
                        "manifest_sha256", "pooling_name", "embedding_dim"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"wrote joint embedding matrix {matrix.shape}")

    # ---- reference-projected PCA (§48): fit on animal_pretrained rows -----
    pre_mask = np.array(
        [row["encoder_state"] == "animal_pretrained" for row in metadata_rows]
    )
    reference_pca = PCA(n_components=10, svd_solver="full")
    reference_pca.fit(matrix[pre_mask])
    projected = reference_pca.transform(matrix)
    with (output_dir / "D8_REFERENCE_PCA_MODEL.pkl").open("wb") as handle:
        pickle.dump(reference_pca, handle)

    def coordinates_csv(path: Path, coords):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["seed", "encoder_state", "species_group", "task", "sample_id",
                 "PC1", "PC2", "PC3"]
            )
            for row, metadata, point in zip(matrix_rows, metadata_rows, coords):
                writer.writerow(
                    [metadata["seed"], metadata["encoder_state"],
                     metadata["species_group"], metadata["task"],
                     metadata["sample_id"],
                     f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"]
                )
        print(f"wrote {path}")

    coordinates_csv(output_dir / "D8_REFERENCE_PCA_COORDINATES.csv", projected)

    joint_pca = PCA(n_components=10, svd_solver="full").fit(matrix)
    coordinates_csv(
        output_dir / "D8_JOINT_PCA_COORDINATES.csv", joint_pca.transform(matrix)
    )

    # ---- per-seed movement + neighborhood retention ------------------------
    movement_rows = []
    retention_rows = []
    row_cursor = 0
    # rows are grouped by seed: [animal_pre: animal+human, s1: human, b1: human]
    rows_per_seed = len(probe_ids) + 3 * len(human_ids)
    for seed in args.seeds:
        block = matrix[row_cursor:row_cursor + rows_per_seed]
        row_cursor += rows_per_seed
        animal_block = block[: len(probe_ids)]
        human_pre = block[len(probe_ids): len(probe_ids) + len(human_ids)]
        human_s1 = block[len(probe_ids) + len(human_ids): len(probe_ids) + 2 * len(human_ids)]
        human_b1 = block[len(probe_ids) + 2 * len(human_ids):]

        pre_norm = np.linalg.norm(human_pre, axis=1, keepdims=True)
        for position, sample_id in enumerate(human_ids):
            move_b1 = human_b1[position] - human_pre[position]
            move_s1 = human_s1[position] - human_pre[position]
            cos_b1 = float(
                np.dot(human_b1[position], human_pre[position])
                / (
                    np.linalg.norm(human_b1[position]) * pre_norm[position, 0]
                    + 1e-12
                )
            )
            cos_s1 = float(
                np.dot(human_s1[position], human_pre[position])
                / (
                    np.linalg.norm(human_s1[position]) * pre_norm[position, 0]
                    + 1e-12
                )
            )
            movement_rows.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    "b1_l2_movement": f"{np.linalg.norm(move_b1):.6f}",
                    "s1_l2_movement": f"{np.linalg.norm(move_s1):.6f}",
                    "b1_cosine_to_pretrained": f"{cos_b1:.8f}",
                    "s1_cosine_to_pretrained": f"{cos_s1:.8f}",
                }
            )

        if len(animal_block) >= MIN_ANIMAL_POOL:
            pre_normed = animal_block / (
                np.linalg.norm(animal_block, axis=1, keepdims=True) + 1e-12
            )
            b1_normed = human_b1 / (
                np.linalg.norm(human_b1, axis=1, keepdims=True) + 1e-12
            )
            pre_human_normed = human_pre / (pre_norm + 1e-12)
            sim_pre = pre_human_normed @ pre_normed.T          # (H, A)
            sim_b1 = b1_normed @ pre_normed.T
            k = min(K_NEIGHBORS, len(animal_block))
            for position, sample_id in enumerate(human_ids):
                pre_neighbors = {
                    probe_ids[j] for j in np.argsort(-sim_pre[position])[:k]
                }
                b1_neighbors = {
                    probe_ids[j] for j in np.argsort(-sim_b1[position])[:k]
                }
                retention_rows.append(
                    {
                        "seed": seed,
                        "human_sample_id": sample_id,
                        "k": k,
                        "neighbor_jaccard": f"{len(pre_neighbors & b1_neighbors) / len(pre_neighbors | b1_neighbors):.6f}",
                        "mean_pretrained_distance": f"{1 - sim_pre[position].mean():.6f}",
                        "mean_b1_distance": f"{1 - sim_b1[position].mean():.6f}",
                    }
                )

    def _write_rows(path: Path, fieldnames, rows_list):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_list)
        print(f"wrote {path} ({len(rows_list)} rows)")

    _write_rows(
        output_dir / "D8_HUMAN_REPRESENTATION_MOVEMENT.csv",
        ["seed", "sample_id", "b1_l2_movement", "s1_l2_movement",
         "b1_cosine_to_pretrained", "s1_cosine_to_pretrained"],
        movement_rows,
    )
    _write_rows(
        output_dir / "D8_CROSS_SPECIES_NEIGHBOR_RETENTION.csv",
        ["seed", "human_sample_id", "k", "neighbor_jaccard",
         "mean_pretrained_distance", "mean_b1_distance"],
        retention_rows,
    )
    if retention_rows:
        jaccards = [float(r["neighbor_jaccard"]) for r in retention_rows]
        print(f"neighbor retention (B1 vs pretrained animal neighborhoods): "
              f"mean Jaccard={sum(jaccards) / len(jaccards):.4f}")
    print("TASK_E_OK (§54 cross-species label similarity skipped: no validated "
          "TDLo<->LD50 mapping; figures rendered offline)")


if __name__ == "__main__":
    main()
