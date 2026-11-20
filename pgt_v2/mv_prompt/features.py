"""Fast one-time feature & graph preloading from the accepted PGT GraphCache.

Eliminates per-batch SQLite reads and repeated RDKit fingerprint recomputation.
All label-dependent statistics (descriptor median/IQR scaling) are fitted
strictly on active training records of the given view and clipped to [-5, 5].
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import torch

from molecular_features import PGT_FEATURE_SPEC
from ..artifacts import read_json, write_json
from ..cache import GraphCache, file_sha256

CACHE_SCHEMA_VERSION = "mv_prompt_cache_v1"

DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeterocycles",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumValenceElectrons",
    "BertzCT",
    "BalabanJ",
    "Chi0v",
    "Chi1v",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "MolMR",
    "LabuteASA",
)


def compute_descriptors_for_mol(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    """Compute 26 2D physicochemical descriptors with finite/exception guards."""
    values = np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float64)
    missing = np.zeros(len(DESCRIPTOR_NAMES), dtype=np.bool_)
    for j, name in enumerate(DESCRIPTOR_NAMES):
        fn = getattr(Descriptors, name)
        try:
            val = float(fn(mol))
        except Exception:
            val = math.nan
        if not math.isfinite(val):
            values[j] = 0.0
            missing[j] = True
        else:
            values[j] = val
    return values, missing


def extract_molecular_views(
    mol: Chem.Mol,
    morgan_gen=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return packed Avalon (128B), packed Morgan (256B), raw desc (26), missing (26)."""
    if morgan_gen is None:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=True
        )
    avalon_arr = np.zeros(1024, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(pyAvalonTools.GetAvalonFP(mol, nBits=1024), avalon_arr)
    morgan_arr = np.asarray(morgan_gen.GetFingerprintAsNumPy(mol), dtype=np.uint8)
    desc_val, desc_miss = compute_descriptors_for_mol(mol)
    return (
        np.packbits(avalon_arr, bitorder="little"),
        np.packbits(morgan_arr, bitorder="little"),
        desc_val,
        desc_miss,
    )


def pack_graph_record(graph) -> dict[str, np.ndarray]:
    """Convert one GraphCache PyG Data item into compact arrays for GINE & Graphormer."""
    x = graph.x.cpu().numpy().astype(np.int16)
    n = int(x.shape[0])
    in_deg = graph.in_degree.cpu().numpy().astype(np.int16)
    spatial = np.clip(graph.spatial_pos.cpu().numpy(), -1, 20).astype(np.int8)
    edge_mat = graph.attn_edge_type.cpu().numpy()  # [n, n, 5]
    src, dst = np.nonzero(edge_mat[..., 0] > 0)
    edge_src = src.astype(np.int16)
    edge_dst = dst.astype(np.int16)
    edge_attr = edge_mat[src, dst].astype(np.int16)  # [E, 5]
    return {
        "num_nodes": n,
        "x": x,
        "in_degree": in_deg,
        "spatial_pos": spatial,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_attr": edge_attr,
    }


def build_mv_cache(
    cache_root: str | Path,
    destination: str | Path,
    *,
    source_sha256: str,
    verify_database: bool = False,
    log_every: int = 10000,
) -> Path:
    """Read GraphCache once and materialize compact multi-view + graph tensors to disk."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta_path = destination.with_suffix(".meta.json")
    graph_cache = GraphCache(
        cache_root, source_sha256=source_sha256, verify_database=verify_database
    )
    n_records = int(graph_cache.manifest["record_count"])
    if destination.is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        if (
            meta.get("schema") == CACHE_SCHEMA_VERSION
            and meta.get("cache_id") == graph_cache.manifest["cache_id"]
            and meta.get("source_sha256") == source_sha256
            and meta.get("feature_spec_sha256") == PGT_FEATURE_SPEC.sha256
            and meta.get("record_count") == n_records
        ):
            print(f"[mv_cache] Reusing existing verified cache: {destination}", flush=True)
            return destination

    started = time.perf_counter()
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )

    avalon_packed = np.zeros((n_records, 128), dtype=np.uint8)
    morgan_packed = np.zeros((n_records, 256), dtype=np.uint8)
    desc_raw = np.zeros((n_records, len(DESCRIPTOR_NAMES)), dtype=np.float64)
    desc_missing = np.zeros((n_records, len(DESCRIPTOR_NAMES)), dtype=np.bool_)

    sample_ids: list[str] = []
    raw_row_indices = np.zeros(n_records, dtype=np.int64)
    chemical_ids: list[str] = []

    num_nodes = np.zeros(n_records, dtype=np.int16)
    node_offsets = np.zeros(n_records + 1, dtype=np.int64)
    edge_offsets = np.zeros(n_records + 1, dtype=np.int64)
    spatial_offsets = np.zeros(n_records + 1, dtype=np.int64)

    xs: list[np.ndarray] = []
    in_degs: list[np.ndarray] = []
    e_srcs: list[np.ndarray] = []
    e_dsts: list[np.ndarray] = []
    e_attrs: list[np.ndarray] = []
    spatials: list[np.ndarray] = []

    for offset in range(n_records):
        graph, mol = graph_cache.read(offset)
        av, mg, dv, dm = extract_molecular_views(mol, morgan_gen)
        avalon_packed[offset] = av
        morgan_packed[offset] = mg
        desc_raw[offset] = dv
        desc_missing[offset] = dm

        sample_ids.append(str(graph.sample_id))
        raw_row_indices[offset] = int(graph.raw_row_index)
        chemical_ids.append(str(graph.model_input_id))

        packed = pack_graph_record(graph)
        n = packed["num_nodes"]
        e = int(packed["edge_src"].shape[0])
        num_nodes[offset] = n
        node_offsets[offset + 1] = node_offsets[offset] + n
        edge_offsets[offset + 1] = edge_offsets[offset] + e
        spatial_offsets[offset + 1] = spatial_offsets[offset] + n * n

        xs.append(packed["x"])
        in_degs.append(packed["in_degree"])
        e_srcs.append(packed["edge_src"])
        e_dsts.append(packed["edge_dst"])
        e_attrs.append(packed["edge_attr"])
        spatials.append(packed["spatial_pos"].reshape(-1))

        if log_every and (offset + 1) % log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[mv_cache] Processed {offset + 1}/{n_records} records ({elapsed:.1f}s)",
                flush=True,
            )

    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "cache_id": graph_cache.manifest["cache_id"],
        "source_sha256": source_sha256,
        "feature_spec_sha256": PGT_FEATURE_SPEC.sha256,
        "record_count": n_records,
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
        "spatial_flat": torch.from_numpy(np.concatenate(spatials, axis=0)),
    }

    tmp_path = destination.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(destination)
    elapsed = time.perf_counter() - started
    meta = {
        "schema": CACHE_SCHEMA_VERSION,
        "cache_id": graph_cache.manifest["cache_id"],
        "source_sha256": source_sha256,
        "feature_spec_sha256": PGT_FEATURE_SPEC.sha256,
        "record_count": n_records,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "file_bytes": destination.stat().st_size,
        "file_sha256": file_sha256(destination),
        "build_seconds": round(elapsed, 2),
    }
    write_json(meta_path, meta)
    print(
        f"[mv_cache] Built {destination} ({meta['file_bytes'] / 1e6:.1f} MB) in {elapsed:.1f}s",
        flush=True,
    )
    return destination


def fit_descriptor_scaler(
    desc_raw: np.ndarray,
    desc_missing: np.ndarray,
    train_offsets: np.ndarray,
    clip_value: float = 5.0,
) -> dict[str, np.ndarray | float]:
    """Fit robust median/IQR-fallback-std scaler STRICTLY on training offsets."""
    sub_val = np.asarray(desc_raw[train_offsets], dtype=np.float64)
    sub_miss = np.asarray(desc_missing[train_offsets], dtype=np.bool_)
    width = sub_val.shape[1]
    medians = np.zeros(width, dtype=np.float64)
    scales = np.ones(width, dtype=np.float64)
    for j in range(width):
        valid = sub_val[~sub_miss[:, j], j]
        if len(valid) == 0:
            medians[j] = 0.0
            scales[j] = 1.0
            continue
        med = float(np.median(valid))
        q75, q25 = np.percentile(valid, [75.0, 25.0])
        iqr = float(q75 - q25) / 1.349
        std = float(np.std(valid))
        scale = iqr if iqr > 1e-6 else (std if std > 1e-6 else 1.0)
        medians[j] = med
        scales[j] = scale
    return {"median": medians, "scale": scales, "clip": float(clip_value)}


def transform_descriptors(
    desc_raw: np.ndarray,
    desc_missing: np.ndarray,
    offsets: np.ndarray,
    scaler: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Apply train-fitted robust scaler + clip + append binary missing flags."""
    raw = np.asarray(desc_raw[offsets], dtype=np.float64)
    miss = np.asarray(desc_missing[offsets], dtype=np.bool_)
    med = np.asarray(scaler["median"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    clip = float(scaler["clip"])
    filled = np.where(miss, med[None, :], raw)
    scaled = np.clip((filled - med[None, :]) / scale[None, :], -clip, clip)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.concatenate([scaled.astype(np.float32), miss.astype(np.float32)], axis=1)


@dataclass
class BatchPayload:
    avalon: torch.Tensor | None = None
    morgan: torch.Tensor | None = None
    desc: torch.Tensor | None = None
    # Sparse graph for GINE
    x_flat: torch.Tensor | None = None
    edge_src: torch.Tensor | None = None
    edge_dst: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    graph_idx: torch.Tensor | None = None
    node_pos: torch.Tensor | None = None
    num_nodes: torch.Tensor | None = None
    max_nodes: int = 0
    batch_size: int = 0
    # Dense padded tensors for Graphormer
    graphormer_batch: object | None = None


class _SimpleGraphormerBatch:
    """Minimal attribute container matching MolecularGraphormerBackbone.encode_tokens."""

    def __init__(
        self,
        *,
        x,
        in_degree,
        out_degree,
        spatial_pos,
        attn_bias,
        attn_edge_type,
        padding_mask,
        spec=PGT_FEATURE_SPEC,
    ):
        self.x = x
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.spatial_pos = spatial_pos
        self.attn_bias = attn_bias
        self.attn_edge_type = attn_edge_type
        self.padding_mask = padding_mask
        self.feature_schema_version = spec.version
        self.feature_spec_sha256 = spec.sha256
        self.edge_bias_mode = "direct"

    def to(self, device):
        self.x = self.x.to(device, non_blocking=True)
        self.in_degree = self.in_degree.to(device, non_blocking=True)
        self.out_degree = self.out_degree.to(device, non_blocking=True)
        self.spatial_pos = self.spatial_pos.to(device, non_blocking=True)
        self.attn_bias = self.attn_bias.to(device, non_blocking=True)
        self.attn_edge_type = self.attn_edge_type.to(device, non_blocking=True)
        self.padding_mask = self.padding_mask.to(device, non_blocking=True)
        return self


class MVDataStore:
    """In-memory store loaded from mv_cache.pt, serving fast batches with identity checks."""

    def __init__(self, payload: dict):
        if payload.get("schema") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported mv_cache schema: {payload.get('schema')}")
        self.cache_id = payload["cache_id"]
        self.source_sha256 = payload["source_sha256"]
        self.feature_spec_sha256 = payload["feature_spec_sha256"]
        self.record_count = int(payload["record_count"])
        self.sample_ids = list(payload["sample_ids"])
        self.raw_row_indices = payload["raw_row_indices"].numpy()
        self.chemical_ids = list(payload["chemical_ids"])

        self.avalon_packed = payload["avalon_packed"].numpy()
        self.morgan_packed = payload["morgan_packed"].numpy()
        self.desc_raw = payload["desc_raw"].numpy()
        self.desc_missing = payload["desc_missing"].numpy()

        self.num_nodes = payload["num_nodes"].numpy()
        self.node_offsets = payload["node_offsets"].numpy()
        self.edge_offsets = payload["edge_offsets"].numpy()
        self.spatial_offsets = payload["spatial_offsets"].numpy()
        self.x_flat = payload["x_flat"].numpy()
        self.in_deg_flat = payload["in_deg_flat"].numpy()
        self.edge_src_flat = payload["edge_src_flat"].numpy()
        self.edge_dst_flat = payload["edge_dst_flat"].numpy()
        self.edge_attr_flat = payload["edge_attr_flat"].numpy()
        self.spatial_flat = payload["spatial_flat"].numpy()

    def verify_partition(self, records: dict, manifest: dict) -> None:
        if manifest["cache_id"] != self.cache_id:
            raise ValueError("View cache_id does not match MVDataStore")
        if manifest["source_sha256"] != self.source_sha256:
            raise ValueError("View source_sha256 does not match MVDataStore")
        offsets = np.asarray(records["cache_offset"], dtype=np.int64)
        raw_rows = np.asarray(records["raw_row_index"], dtype=np.int64)
        if not np.array_equal(self.raw_row_indices[offsets], raw_rows):
            raise ValueError("Partition raw_row_index mismatch with MVDataStore")
        for idx in (0, len(offsets) // 2, len(offsets) - 1) if len(offsets) else ():
            off = int(offsets[idx])
            if (
                str(records["record_id"][idx]) != self.sample_ids[off]
                or str(records["chemical_id"][idx]) != self.chemical_ids[off]
            ):
                raise ValueError("Partition record/chemical identity mismatch with MVDataStore")

    def unpack_avalon(self, offsets: np.ndarray) -> np.ndarray:
        return np.unpackbits(self.avalon_packed[offsets], axis=1, bitorder="little").astype(
            np.float32
        )

    def unpack_morgan(self, offsets: np.ndarray) -> np.ndarray:
        return np.unpackbits(self.morgan_packed[offsets], axis=1, bitorder="little").astype(
            np.float32
        )

    def build_batch(
        self,
        offsets: np.ndarray,
        *,
        desc_array: np.ndarray | None,
        need_avalon: bool,
        need_morgan: bool,
        need_desc: bool,
        graph_mode: str | None,
        device: str | torch.device,
    ) -> BatchPayload:
        offsets = np.asarray(offsets, dtype=np.int64)
        b = len(offsets)
        out = BatchPayload(batch_size=b)
        if need_avalon:
            out.avalon = torch.from_numpy(self.unpack_avalon(offsets)).to(device, non_blocking=True)
        if need_morgan:
            out.morgan = torch.from_numpy(self.unpack_morgan(offsets)).to(device, non_blocking=True)
        if need_desc:
            if desc_array is None:
                raise ValueError("desc_array required when need_desc=True")
            out.desc = torch.from_numpy(desc_array).to(device, non_blocking=True)

        if graph_mode == "gine":
            ns = self.num_nodes[offsets].astype(np.int64)
            max_n = int(ns.max())
            total_n = int(ns.sum())
            x_parts = []
            src_parts = []
            dst_parts = []
            eattr_parts = []
            graph_idx = np.empty(total_n, dtype=np.int64)
            node_pos = np.empty(total_n, dtype=np.int64)
            cursor = 0
            for i, off in enumerate(offsets):
                n = int(ns[i])
                n0, n1 = int(self.node_offsets[off]), int(self.node_offsets[off + 1])
                e0, e1 = int(self.edge_offsets[off]), int(self.edge_offsets[off + 1])
                x_parts.append(self.x_flat[n0:n1])
                if e1 > e0:
                    src_parts.append(self.edge_src_flat[e0:e1].astype(np.int64) + cursor)
                    dst_parts.append(self.edge_dst_flat[e0:e1].astype(np.int64) + cursor)
                    eattr_parts.append(self.edge_attr_flat[e0:e1])
                graph_idx[cursor : cursor + n] = i
                node_pos[cursor : cursor + n] = np.arange(n, dtype=np.int64)
                cursor += n
            out.x_flat = torch.from_numpy(np.concatenate(x_parts, axis=0).astype(np.int64)).to(
                device, non_blocking=True
            )
            if src_parts:
                out.edge_src = torch.from_numpy(np.concatenate(src_parts, axis=0)).to(
                    device, non_blocking=True
                )
                out.edge_dst = torch.from_numpy(np.concatenate(dst_parts, axis=0)).to(
                    device, non_blocking=True
                )
                out.edge_attr = torch.from_numpy(
                    np.concatenate(eattr_parts, axis=0).astype(np.int64)
                ).to(device, non_blocking=True)
            else:
                out.edge_src = torch.zeros(0, dtype=torch.long, device=device)
                out.edge_dst = torch.zeros(0, dtype=torch.long, device=device)
                out.edge_attr = torch.zeros(
                    (0, len(PGT_FEATURE_SPEC.bond_fields)), dtype=torch.long, device=device
                )
            out.graph_idx = torch.from_numpy(graph_idx).to(device, non_blocking=True)
            out.node_pos = torch.from_numpy(node_pos).to(device, non_blocking=True)
            out.num_nodes = torch.from_numpy(ns).to(device, non_blocking=True)
            out.max_nodes = max_n

        elif graph_mode == "graphormer":
            ns = self.num_nodes[offsets].astype(np.int64)
            max_n = int(ns.max())
            n_atom_fields = self.x_flat.shape[1]
            n_bond_fields = self.edge_attr_flat.shape[1]
            x_pad = np.zeros((b, max_n, n_atom_fields), dtype=np.int64)
            deg_pad = np.zeros((b, max_n), dtype=np.int64)
            sp_pad = np.zeros((b, max_n, max_n), dtype=np.int64)
            edge_pad = np.zeros((b, max_n, max_n, n_bond_fields), dtype=np.int64)
            pad_mask = np.ones((b, max_n), dtype=np.bool_)
            for i, off in enumerate(offsets):
                n = int(ns[i])
                n0, n1 = int(self.node_offsets[off]), int(self.node_offsets[off + 1])
                e0, e1 = int(self.edge_offsets[off]), int(self.edge_offsets[off + 1])
                s0, s1 = int(self.spatial_offsets[off]), int(self.spatial_offsets[off + 1])
                x_pad[i, :n] = self.x_flat[n0:n1]
                deg_pad[i, :n] = self.in_deg_flat[n0:n1].astype(np.int64) + 1
                sp_pad[i, :n, :n] = self.spatial_flat[s0:s1].reshape(n, n).astype(np.int64) + 1
                if e1 > e0:
                    es = self.edge_src_flat[e0:e1]
                    ed = self.edge_dst_flat[e0:e1]
                    edge_pad[i, es, ed] = self.edge_attr_flat[e0:e1]
                pad_mask[i, :n] = False
            attn_bias = torch.zeros((b, max_n + 1, max_n + 1), dtype=torch.float32)
            out.graphormer_batch = _SimpleGraphormerBatch(
                x=torch.from_numpy(x_pad),
                in_degree=torch.from_numpy(deg_pad),
                out_degree=torch.from_numpy(deg_pad.clone() if isinstance(deg_pad, torch.Tensor) else deg_pad.copy()),
                spatial_pos=torch.from_numpy(sp_pad),
                attn_bias=attn_bias,
                attn_edge_type=torch.from_numpy(edge_pad),
                padding_mask=torch.from_numpy(pad_mask),
            ).to(device)
            out.max_nodes = max_n

        return out


def load_mv_cache(path: str | Path) -> MVDataStore:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return MVDataStore(payload)
