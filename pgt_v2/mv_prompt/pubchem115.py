"""PubChem 115-endpoint cache/view builder, nested S21 downsampling views, and 59->36 zero-shot builder."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

from baselines.models.toxacol import task_adjacency
from molecular_features import PGT_FEATURE_SPEC
from ..artifacts import load_arrays, load_manifest, read_json, seal_manifest, write_json
from ..cache import file_sha256
from ..chemistry import UnsupportedChemistry, _atom_values, _bond_values, framed_digest, stereo_states
from ..data import sparse_visible_mask
from ..models import task_metadata
from ..splits import allocate_groups
from .features import CACHE_SCHEMA_VERSION, DESCRIPTOR_NAMES, extract_molecular_views

RDLogger.DisableLog("rdApp.*")

HUMAN11_PREFIXES = ("human_", "man_", "women_", "child_")


def build_nested_s21_view(
    src_full_view: Path,
    dst_view: Path,
    *,
    target_tasks: list[str] | tuple[str, ...],
    fraction: float,
    budget_id: str,
    subset_seed: int = 1042,
) -> dict:
    """Create a strictly nested S21 downsampled view from FULL using C06F's exact hash ordering."""
    src_full_view = Path(src_full_view)
    dst_view = Path(dst_view)
    if (dst_view / "manifest.json").is_file():
        return load_manifest(dst_view)

    src_manifest = load_manifest(src_full_view)
    tasks = list(src_manifest["task_names"])
    fold_id = str(src_manifest["fold"])

    train_arr = load_arrays(src_full_view, "train.npz", src_manifest)
    val_arr = load_arrays(src_full_view, "validation.npz", src_manifest)
    test_arr = load_arrays(src_full_view, "test.npz", src_manifest)

    full_train_labels = train_arr["labels"].astype(np.float64)
    n_train = full_train_labels.shape[0]
    observed_train = np.isfinite(full_train_labels)
    roles_train = np.array(["train"] * n_train)
    chem_ids_train = [str(c) for c in train_arr["chemical_id"].tolist()]

    vis_mask = sparse_visible_mask(
        observed=observed_train,
        task_names=tasks,
        roles=roles_train,
        chemical_ids=chem_ids_train,
        target_tasks=list(target_tasks),
        fraction=float(fraction),
        fold_id=fold_id,
        subset_seed=int(subset_seed),
    )

    masked_labels = np.full_like(full_train_labels, np.nan, dtype=np.float64)
    masked_labels[vis_mask] = full_train_labels[vis_mask]
    counts = np.isfinite(masked_labels).sum(axis=0).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError(f"Zero visible labels after downsampling {fold_id}/{budget_id}")

    means = np.nanmean(masked_labels, axis=0)
    raw_scale = np.nanstd(masked_labels, axis=0, ddof=0)
    scale_fallback = (counts < 2) | (raw_scale <= 1e-8)
    scale = np.where(scale_fallback, 1.0, raw_scale)
    train_grps = np.asarray(train_arr["group_id"])
    group_count = np.array(
        [len(set(train_grps[vis_mask[:, j]].tolist())) for j in range(len(tasks))],
        dtype=np.int64,
    )
    adj, _ = task_adjacency(masked_labels, min_shared=15, pcc_threshold=0.75)

    dst_view.mkdir(parents=True, exist_ok=True)
    common_keys = ("record_id", "cache_offset", "raw_row_index", "group_id", "chemical_id")
    files = []
    for role, arr, y_mat in (
        ("train", train_arr, masked_labels),
        ("validation", val_arr, val_arr["labels"].astype(np.float64)),
        ("test", test_arr, test_arr["labels"].astype(np.float64)),
    ):
        part = {k: arr[k] for k in common_keys}
        np.savez_compressed(dst_view / f"{role}.npz", **part, labels=y_mat)
        files.append(f"{role}.npz")
        np.savez_compressed(dst_view / f"{role}_inputs.npz", **part)
        files.append(f"{role}_inputs.npz")

    np.savez_compressed(
        dst_view / "statistics.npz",
        mean=means.astype(np.float64),
        scale=scale.astype(np.float64),
        count=counts,
        group_count=group_count,
        scale_fallback=scale_fallback,
        adjacency=adj,
    )
    files.append("statistics.npz")

    meta = {
        "format": src_manifest["format"],
        "view_id": f"{fold_id}_{budget_id}",
        "fold": fold_id,
        "task_names": tasks,
        "downsampled_tasks": list(target_tasks),
        "fraction": float(fraction),
        "subset_seed": int(subset_seed),
        "parent_view_sha256": src_manifest["manifest_sha256"],
        "source_sha256": src_manifest["source_sha256"],
        "feature_spec_sha256": src_manifest["feature_spec_sha256"],
        "cache_id": src_manifest["cache_id"],
        "active_train_rows": int(np.isfinite(masked_labels).any(axis=1).sum()),
        "visible_labels": int(counts.sum()),
    }
    return seal_manifest(dst_view, meta, files)


def _featurize_pubchem_chunk(rows: list[tuple[int, int, str]]) -> list[dict | None]:
    """Multiprocessing worker: RDKit parse + PGT_FEATURE_SPEC graph + Avalon/Morgan/2D desc."""
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    spec = PGT_FEATURE_SPEC
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    out: list[dict | None] = []
    for raw_idx, cid, smi in rows:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or mol.GetNumAtoms() == 0:
            out.append(None)
            continue
        if any(a.HasQuery() or a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            out.append(None)
            continue
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        try:
            mol_st, atom_st, bond_st = stereo_states(mol)
        except (UnsupportedChemistry, Exception):
            mol_st = Chem.Mol(mol)
            atom_st = ("UNASSIGNED",) * mol_st.GetNumAtoms()
            bond_st = ("UNASSIGNED",) * mol_st.GetNumBonds()

        av, mg, dv, dm = extract_molecular_views(mol_st, morgan_gen)
        n = mol_st.GetNumAtoms()
        x = np.asarray(
            [spec.encode("atom", _atom_values(a, s)) for a, s in zip(mol_st.GetAtoms(), atom_st)],
            dtype=np.int16,
        )
        in_deg = np.zeros(n, dtype=np.int16)
        srcs: list[int] = []
        dsts: list[int] = []
        eattrs: list[list[int]] = []
        for b, bs in zip(mol_st.GetBonds(), bond_st):
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            in_deg[i] += 1
            in_deg[j] += 1
            srcs.extend((i, j))
            dsts.extend((j, i))
            eattrs.append(spec.encode("bond", _bond_values(b, bs, False)))
            eattrs.append(spec.encode("bond", _bond_values(b, bs, True)))

        if srcs:
            edge_src = np.asarray(srcs, dtype=np.int16)
            edge_dst = np.asarray(dsts, dtype=np.int16)
            edge_attr = np.asarray(eattrs, dtype=np.int16)
        else:
            edge_src = np.zeros(0, dtype=np.int16)
            edge_dst = np.zeros(0, dtype=np.int16)
            edge_attr = np.zeros((0, len(spec.bond_fields)), dtype=np.int16)

        canon_iso = Chem.MolToSmiles(mol_st, canonical=True, isomericSmiles=True)
        canon_noniso = Chem.MolToSmiles(mol_st, canonical=True, isomericSmiles=False)
        mol_nostereo = Chem.Mol(mol_st)
        Chem.RemoveStereochemistry(mol_nostereo)
        scaf_mol = MurckoScaffold.GetScaffoldForMol(mol_nostereo)
        if scaf_mol is not None and scaf_mol.GetNumAtoms() > 0:
            scaf_key = "murcko:" + Chem.MolToSmiles(scaf_mol, canonical=True, isomericSmiles=False)
        else:
            scaf_key = "acyclic:" + canon_noniso

        out.append(
            {
                "raw_row_index": int(raw_idx),
                "cid": int(cid),
                "canon_iso": canon_iso,
                "canon_noniso": canon_noniso,
                "scaffold_key": scaf_key,
                "avalon": av,
                "morgan": mg,
                "desc_raw": dv,
                "desc_missing": dm,
                "num_nodes": n,
                "x": x,
                "in_deg": in_deg,
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "edge_attr": edge_attr,
            }
        )
    return out


def build_pubchem115_mv_cache(
    csv_path: str | Path,
    destination: str | Path,
    *,
    max_workers: int | None = None,
    chunk_size: int = 1000,
) -> Path:
    """Build a compact MVDataStore cache (.pt) + identity/scaffold table directly from PubChem 115 CSV."""
    csv_path = Path(csv_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta_path = destination.with_suffix(".meta.json")
    id_path = destination.with_suffix(".identities.json.gz")
    src_sha = file_sha256(csv_path)

    if destination.is_file() and meta_path.is_file() and id_path.is_file():
        meta = read_json(meta_path)
        if (
            meta.get("schema") == CACHE_SCHEMA_VERSION
            and meta.get("source_sha256") == src_sha
            and meta.get("feature_spec_sha256") == PGT_FEATURE_SPEC.sha256
        ):
            print(f"[pub115] Reusing verified cache: {destination}", flush=True)
            return destination

    started = time.perf_counter()
    df = pd.read_csv(csv_path, usecols=["cid", "smiles"])
    rows = [(int(i), int(c), str(s)) for i, (c, s) in enumerate(zip(df["cid"], df["smiles"]))]
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    workers = max_workers if max_workers is not None else min(16, max(1, (os.cpu_count() or 4)))

    parsed: list[dict] = []
    dropped = 0
    if workers <= 1 or len(chunks) <= 1:
        for ch in chunks:
            for item in _featurize_pubchem_chunk(ch):
                if item is None:
                    dropped += 1
                else:
                    parsed.append(item)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_featurize_pubchem_chunk, chunks):
                for item in res:
                    if item is None:
                        dropped += 1
                    else:
                        parsed.append(item)

    n_rec = len(parsed)
    if n_rec == 0:
        raise ValueError("No valid molecules parsed from PubChem 115 CSV")

    avalon_packed = np.zeros((n_rec, 128), dtype=np.uint8)
    morgan_packed = np.zeros((n_rec, 256), dtype=np.uint8)
    desc_raw = np.zeros((n_rec, len(DESCRIPTOR_NAMES)), dtype=np.float64)
    desc_missing = np.zeros((n_rec, len(DESCRIPTOR_NAMES)), dtype=np.bool_)
    sample_ids: list[str] = []
    raw_row_indices = np.zeros(n_rec, dtype=np.int64)
    chemical_ids: list[str] = []
    cids = np.zeros(n_rec, dtype=np.int64)
    canon_nonisos: list[str] = []
    scaffold_keys: list[str] = []

    num_nodes = np.zeros(n_rec, dtype=np.int16)
    node_offsets = np.zeros(n_rec + 1, dtype=np.int64)
    edge_offsets = np.zeros(n_rec + 1, dtype=np.int64)
    spatial_offsets = np.zeros(n_rec + 1, dtype=np.int64)

    xs: list[np.ndarray] = []
    in_degs: list[np.ndarray] = []
    e_srcs: list[np.ndarray] = []
    e_dsts: list[np.ndarray] = []
    e_attrs: list[np.ndarray] = []

    for off, item in enumerate(parsed):
        avalon_packed[off] = item["avalon"]
        morgan_packed[off] = item["morgan"]
        desc_raw[off] = item["desc_raw"]
        desc_missing[off] = item["desc_missing"]
        sample_ids.append(f"cid_{item['cid']}")
        raw_row_indices[off] = item["raw_row_index"]
        chemical_ids.append("chem_" + framed_digest([item["canon_iso"]])[:24])
        cids[off] = item["cid"]
        canon_nonisos.append(item["canon_noniso"])
        scaffold_keys.append(item["scaffold_key"])

        n = item["num_nodes"]
        e = int(item["edge_src"].shape[0])
        num_nodes[off] = n
        node_offsets[off + 1] = node_offsets[off] + n
        edge_offsets[off + 1] = edge_offsets[off] + e
        xs.append(item["x"])
        in_degs.append(item["in_deg"])
        e_srcs.append(item["edge_src"])
        e_dsts.append(item["edge_dst"])
        e_attrs.append(item["edge_attr"])

    cache_id = "pub115_cache_" + framed_digest([src_sha, PGT_FEATURE_SPEC.sha256, n_rec])[:32]
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "cache_id": cache_id,
        "source_sha256": src_sha,
        "feature_spec_sha256": PGT_FEATURE_SPEC.sha256,
        "record_count": n_rec,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "sample_ids": sample_ids,
        "raw_row_indices": torch.from_numpy(raw_row_indices),
        "chemical_ids": chemical_ids,
        "avalon_packed": torch.from_numpy(avalon_packed),
        "morgan_packed": torch.from_numpy(morgan_packed),
        "desc_raw": torch.from_numpy(desc_raw),
        "desc_missing": torch.from_numpy(desc_missing),
        "num_nodes": torch.from_numpy(num_nodes),
        "node_offsets": torch.from_numpy(node_offsets),
        "edge_offsets": torch.from_numpy(edge_offsets),
        "spatial_offsets": torch.from_numpy(spatial_offsets),
        "x_flat": torch.from_numpy(np.concatenate(xs, axis=0)),
        "in_deg_flat": torch.from_numpy(np.concatenate(in_degs, axis=0)),
        "edge_src_flat": torch.from_numpy(
            np.concatenate(e_srcs, axis=0) if any(len(a) for a in e_srcs) else np.zeros(0, dtype=np.int16)
        ),
        "edge_dst_flat": torch.from_numpy(
            np.concatenate(e_dsts, axis=0) if any(len(a) for a in e_dsts) else np.zeros(0, dtype=np.int16)
        ),
        "edge_attr_flat": torch.from_numpy(
            np.concatenate(e_attrs, axis=0)
            if any(len(a) for a in e_attrs)
            else np.zeros((0, len(PGT_FEATURE_SPEC.bond_fields)), dtype=np.int16)
        ),
        "spatial_flat": torch.zeros(0, dtype=torch.int8),
    }
    tmp_path = destination.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(destination)

    with gzip.open(id_path, "wt", encoding="utf-8") as fh:
        json.dump(
            {
                "cache_id": cache_id,
                "source_sha256": src_sha,
                "cids": cids.tolist(),
                "raw_row_indices": raw_row_indices.tolist(),
                "sample_ids": sample_ids,
                "chemical_ids": chemical_ids,
                "canon_nonisos": canon_nonisos,
                "scaffold_keys": scaffold_keys,
            },
            fh,
        )

    elapsed = time.perf_counter() - started
    meta = {
        "schema": CACHE_SCHEMA_VERSION,
        "cache_id": cache_id,
        "source_sha256": src_sha,
        "feature_spec_sha256": PGT_FEATURE_SPEC.sha256,
        "record_count": n_rec,
        "dropped_invalid": dropped,
        "file_bytes": destination.stat().st_size,
        "file_sha256": file_sha256(destination),
        "build_seconds": round(elapsed, 2),
    }
    write_json(meta_path, meta)
    print(
        f"[pub115] Built {destination} ({n_rec} valid, {dropped} dropped, "
        f"{meta['file_bytes']/1e6:.1f} MB) in {elapsed:.1f}s",
        flush=True,
    )
    return destination


def _save_pub115_view(
    dst_dir: Path,
    *,
    fold_id: str,
    tasks: list[str],
    roles: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
    raw_rows: np.ndarray,
    chem_ids: list[str],
    group_ids: list[str],
    cache_id: str,
    source_sha256: str,
) -> dict:
    if (dst_dir / "manifest.json").is_file():
        return load_manifest(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    n = len(sample_ids)
    offsets = np.arange(n, dtype=np.int64)
    rec_arr = np.asarray(sample_ids)
    chem_arr = np.asarray(chem_ids)
    grp_arr = np.asarray(group_ids)

    files = []
    train_mask = roles == "train"
    for role in ("train", "validation", "test"):
        idx = np.flatnonzero(roles == role)
        part = {
            "record_id": rec_arr[idx],
            "cache_offset": offsets[idx],
            "raw_row_index": raw_rows[idx],
            "group_id": grp_arr[idx],
            "chemical_id": chem_arr[idx],
        }
        np.savez_compressed(dst_dir / f"{role}.npz", **part, labels=labels[idx])
        files.append(f"{role}.npz")
        np.savez_compressed(dst_dir / f"{role}_inputs.npz", **part)
        files.append(f"{role}_inputs.npz")

    tr_y = labels[train_mask]
    counts = np.isfinite(tr_y).sum(axis=0).astype(np.int64)
    means = np.zeros(len(tasks), dtype=np.float64)
    scale = np.ones(len(tasks), dtype=np.float64)
    scale_fallback = np.zeros(len(tasks), dtype=bool)
    group_count = np.zeros(len(tasks), dtype=np.int64)
    tr_grps = grp_arr[train_mask]
    for j in range(len(tasks)):
        m = np.isfinite(tr_y[:, j])
        if not bool(m.any()):
            scale_fallback[j] = True
            continue
        vals = tr_y[m, j]
        means[j] = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        if counts[j] < 2 or sd <= 1e-8:
            scale[j] = 1.0
            scale_fallback[j] = True
        else:
            scale[j] = sd
        group_count[j] = int(len(set(tr_grps[m].tolist())))

    adj, _ = task_adjacency(tr_y, min_shared=15, pcc_threshold=0.75)
    np.savez_compressed(
        dst_dir / "statistics.npz",
        mean=means,
        scale=scale,
        count=counts,
        group_count=group_count,
        scale_fallback=scale_fallback,
        adjacency=adj,
    )
    files.append("statistics.npz")

    meta = {
        "format": "PGTV2_VIEW_1",
        "view_id": f"{fold_id}_FULL",
        "fold": fold_id,
        "task_names": tasks,
        "source_sha256": source_sha256,
        "feature_spec_sha256": PGT_FEATURE_SPEC.sha256,
        "cache_id": cache_id,
        "active_train_rows": int(np.isfinite(tr_y).any(axis=1).sum()),
        "visible_labels": int(counts.sum()),
    }
    return seal_manifest(dst_dir, meta, files)


def build_pubchem115_views(
    *,
    cache_path: str | Path,
    csv_path: str | Path,
    random_split_dir: str | Path,
    views_out_root: str | Path,
    groups_out_path: str | Path,
    seed: int = 20261127,
) -> dict:
    """Materialize official 5-fold random views (PUB115_R0..R4) and strict 5-fold scaffold views (PUB115_S0..S4)."""
    cache_path = Path(cache_path)
    csv_path = Path(csv_path)
    random_split_dir = Path(random_split_dir)
    views_out_root = Path(views_out_root)
    groups_out_path = Path(groups_out_path)

    with gzip.open(cache_path.with_suffix(".identities.json.gz"), "rt", encoding="utf-8") as fh:
        ident = json.load(fh)
    cache_id = ident["cache_id"]
    src_sha = ident["source_sha256"]
    cids = np.asarray(ident["cids"], dtype=np.int64)
    raw_rows = np.asarray(ident["raw_row_indices"], dtype=np.int64)
    sample_ids = list(ident["sample_ids"])
    chem_ids = list(ident["chemical_ids"])
    canon_nonisos = list(ident["canon_nonisos"])
    scaffold_keys = list(ident["scaffold_keys"])

    df = pd.read_csv(csv_path)
    tasks = [c for c in df.columns if c not in ("cid", "smiles") and not c.isdigit()]
    labels_all = df[tasks].to_numpy(dtype=np.float64)[raw_rows]
    observed = np.isfinite(labels_all)
    full_counts = observed.sum(axis=0)

    small68 = [t for t, n in zip(tasks, full_counts) if n < 200]
    med115 = [t for t, n in zip(tasks, full_counts) if 200 <= n < 1000]
    large115 = [t for t, n in zip(tasks, full_counts) if n >= 1000]
    human11 = [t for t in tasks if t.startswith(HUMAN11_PREFIXES)]
    groups = {
        "task_names": tasks,
        "ALL115": tasks,
        "Small68": small68,
        "Human11": human11,
        "Large115": large115,
        "Medium115": med115,
        # Aliases so training/eval loops work seamlessly across 59 and 115 datasets
        "S21": small68,
        "M27": med115,
        "L11": large115,
        "full_counts": full_counts.tolist(),
    }
    write_json(groups_out_path, groups)

    created = {"random": [], "scaffold": []}

    # 1. Official 5-fold random splits (ToxACoL Table 1 exact train_fold_k / test_fold_k CIDs)
    noniso_groups = ["noniso_" + framed_digest([s])[:20] for s in canon_nonisos]
    for k in range(5):
        tr_file = random_split_dir / f"train_fold_{k}.csv"
        te_file = random_split_dir / f"test_fold_{k}.csv"
        if not tr_file.is_file() or not te_file.is_file():
            continue
        te_cids = set(pd.read_csv(te_file)["cid"].astype(np.int64).tolist())
        is_test = np.asarray([int(c) in te_cids for c in cids], dtype=bool)
        remain = ~is_test
        rem_groups = [g for g, keep in zip(noniso_groups, remain) if keep]
        inner, _ = allocate_groups(
            rem_groups,
            observed[remain],
            fractions=[0.85, 0.15],
            minimum_groups=[1, 1],
            seed=seed + k,
            max_passes=8,
        )
        roles = np.asarray(
            [
                "test" if is_test[i] else ("train" if inner[noniso_groups[i]] == 0 else "validation")
                for i in range(len(cids))
            ]
        )
        fold_id = f"PUB115_R{k}"
        _save_pub115_view(
            views_out_root / fold_id / "FULL",
            fold_id=fold_id,
            tasks=tasks,
            roles=roles,
            labels=labels_all,
            sample_ids=sample_ids,
            raw_rows=raw_rows,
            chem_ids=chem_ids,
            group_ids=noniso_groups,
            cache_id=cache_id,
            source_sha256=src_sha,
        )
        created["random"].append(fold_id)

    # 2. Strict 5-fold non-overlapping Murcko scaffold split (zero scaffold leakage)
    scaf_groups = ["scaf_" + framed_digest([s])[:20] for s in scaffold_keys]
    outer, outer_rep = allocate_groups(
        scaf_groups,
        observed,
        fractions=[0.2] * 5,
        minimum_groups=[1] * 5,
        seed=seed,
        max_passes=10,
    )
    for k in range(5):
        remain = np.asarray([outer[g] != k for g in scaf_groups], dtype=bool)
        rem_groups = [g for g, keep in zip(scaf_groups, remain) if keep]
        inner, _ = allocate_groups(
            rem_groups,
            observed[remain],
            fractions=[0.85, 0.15],
            minimum_groups=[1, 1],
            seed=seed + 100 + k,
            max_passes=8,
        )
        roles = np.asarray(
            [
                "test" if outer[g] == k else ("train" if inner[g] == 0 else "validation")
                for g in scaf_groups
            ]
        )
        # Strict zero-scaffold-leakage assertion
        tr_g = {g for g, r in zip(scaf_groups, roles) if r == "train"}
        va_g = {g for g, r in zip(scaf_groups, roles) if r == "validation"}
        te_g = {g for g, r in zip(scaf_groups, roles) if r == "test"}
        if (tr_g & va_g) or (tr_g & te_g) or (va_g & te_g):
            raise AssertionError(f"Scaffold leakage detected in PUB115_S{k}")
        fold_id = f"PUB115_S{k}"
        _save_pub115_view(
            views_out_root / fold_id / "FULL",
            fold_id=fold_id,
            tasks=tasks,
            roles=roles,
            labels=labels_all,
            sample_ids=sample_ids,
            raw_rows=raw_rows,
            chem_ids=chem_ids,
            group_ids=scaf_groups,
            cache_id=cache_id,
            source_sha256=src_sha,
        )
        created["scaffold"].append(fold_id)

    summary = {
        "tasks": len(tasks),
        "small68": len(small68),
        "human11": len(human11),
        "records": len(cids),
        "scaffold_outer_coverage_pass": bool(outer_rep["coverage_pass"]),
        "created": created,
    }
    write_json(views_out_root / "pubchem115_views_summary.json", summary)
    return summary


def find_external_zeroshot_subset(
    *,
    pub115_cache_path: str | Path,
    pub115_csv_path: str | Path,
    tasks_59: list[str] | tuple[str, ...],
    seen_noniso_smiles: set[str],
) -> dict:
    """Identify external PubChem endpoints unseen in 59-endpoint training with strictly novel molecules."""
    meta59 = task_metadata(tuple(tasks_59))
    vocabs59 = {
        f: {meta59[t][f] for t in tasks_59}
        for f in ("organism", "route", "measurement", "population")
    }
    norm59 = {
        (meta59[t]["organism"], meta59[t]["route"], meta59[t]["measurement"], meta59[t]["population"])
        for t in tasks_59
    }

    with gzip.open(Path(pub115_cache_path).with_suffix(".identities.json.gz"), "rt", encoding="utf-8") as fh:
        ident = json.load(fh)
    raw_rows = np.asarray(ident["raw_row_indices"], dtype=np.int64)
    canon_nonisos = list(ident["canon_nonisos"])
    cids = np.asarray(ident["cids"], dtype=np.int64)

    df = pd.read_csv(pub115_csv_path)
    ep115 = [c for c in df.columns if c not in ("cid", "smiles") and not c.isdigit()]
    meta115 = task_metadata(tuple(ep115))

    unseen_eps = []
    for ep in ep115:
        m = meta115[ep]
        key = (m["organism"], m["route"], m["measurement"], m["population"])
        if key in norm59:
            continue
        if all(m[f] in vocabs59[f] for f in ("organism", "route", "measurement", "population")):
            unseen_eps.append(ep)

    labels_unseen = df[unseen_eps].to_numpy(dtype=np.float64)[raw_rows]
    has_label = np.isfinite(labels_unseen).any(axis=1)
    is_novel = np.asarray([s not in seen_noniso_smiles for s in canon_nonisos], dtype=bool)
    keep_idx = np.flatnonzero(has_label & is_novel).astype(np.int64)

    return {
        "unseen_tasks": unseen_eps,
        "cache_offsets": keep_idx,
        "cids": cids[keep_idx],
        "labels": labels_unseen[keep_idx],
        "n_total_with_label": int(has_label.sum()),
        "n_overlap_removed": int((has_label & ~is_novel).sum()),
        "n_novel_molecules": int(len(keep_idx)),
        "n_novel_measurements": int(np.isfinite(labels_unseen[keep_idx]).sum()),
    }
