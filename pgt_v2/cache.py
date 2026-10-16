"""Isolated, label-free PGT graph cache; never writes a legacy CURRENT pointer.

SQLite stores compressed numeric arrays and RDKit Mol binaries, not pickle.
Every build creates a new directory. An incomplete directory is not readable.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data

from molecular_features import PGT_FEATURE_SPEC
from .chemistry import MolecularInput, digest, graph_from_input, prepare_molecule
from .data import validate_graph

CACHE_FORMAT = "pgt_graph_sqlite_npz_v1"
ARRAY_FIELDS = ("x", "in_degree", "out_degree", "spatial_pos", "attn_edge_type", "edge_input")
META_FIELDS = ("sample_id", "raw_row_index", "num_nodes", "feature_schema_version", "feature_spec_sha256",
               "model_input_id", "nonstereo_id", "smiles", "canonical_smiles", "raw_to_model", "model_to_raw",
               "symmetry_classes", "edge_bias_mode")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    molecule: MolecularInput
    sample_id: str
    raw_row_index: int
    provenance: dict


def _verify_molecular_payload(graph, mol, spec):
    preserved = prepare_molecule(mol, spec=spec)
    if preserved.model_input_id != graph.model_input_id or preserved.nonstereo_id != graph.nonstereo_id:
        raise ValueError("Stored Mol and registered graph identity disagree")
    rebuilt = graph_from_input(preserved, sample_id=graph.sample_id, raw_row_index=graph.raw_row_index, spec=spec)
    if any(not torch.equal(getattr(graph, name), getattr(rebuilt, name)) for name in ARRAY_FIELDS):
        raise ValueError("Stored features disagree with the preserved molecular graph")


def build_cache(entries, destination, *, source_sha256, spec=PGT_FEATURE_SPEC):
    """Stream a complete new cache. Real-data invocation belongs to G01."""
    spec.validate_runtime()
    destination = Path(destination).resolve()
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("A source SHA256 is required")
    if any((p / "CURRENT").exists() for p in (destination, *destination.parents)):
        raise ValueError("The new cache cannot be placed inside a legacy CURRENT store")
    destination.mkdir(parents=True, exist_ok=False)
    database = destination / "graphs.sqlite3"
    records = []
    try:
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute("CREATE TABLE graphs (cache_offset INTEGER PRIMARY KEY, sample_id TEXT UNIQUE NOT NULL, "
                "raw_row_index INTEGER UNIQUE NOT NULL, input_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL, payload BLOB NOT NULL)")
            for offset, entry in enumerate(entries):
                graph = graph_from_input(entry.molecule, sample_id=entry.sample_id, raw_row_index=entry.raw_row_index, spec=spec)
                validate_graph(graph, spec)
                _verify_molecular_payload(graph, entry.molecule.mol, spec)
                metadata = {name: getattr(graph, name) for name in META_FIELDS}
                metadata.update(cache_offset=offset, provenance=entry.provenance, source_sha256=source_sha256)
                arrays = {name: getattr(graph, name).cpu().numpy() for name in ARRAY_FIELDS}
                arrays["metadata"] = np.frombuffer(json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8"), dtype=np.uint8)
                arrays["mol_binary"] = np.frombuffer(entry.molecule.mol.ToBinary(Chem.PropertyPickleOptions.AllProps), dtype=np.uint8)
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                payload = stream.getvalue()
                checksum = hashlib.sha256(payload).hexdigest()
                connection.execute("INSERT INTO graphs VALUES (?, ?, ?, ?, ?, ?)",
                    (offset, entry.sample_id, entry.raw_row_index, graph.model_input_id, checksum, payload))
                records.append([offset, entry.sample_id, entry.raw_row_index, graph.model_input_id, checksum])
        if not records:
            raise ValueError("Cannot publish an empty graph cache")
        identity = {"cache_format": CACHE_FORMAT, "source_sha256": source_sha256,
                    "feature_spec_sha256": spec.sha256, "records": records}
        manifest = {"status": "COMPLETE", "cache_format": CACHE_FORMAT, "cache_id": "cache_" + digest(identity),
                    "source_sha256": source_sha256, "feature_spec": spec.metadata(), "record_count": len(records),
                    "database_sha256": file_sha256(database), "records": records,
                    "labels": "not stored; explicit TrainingView supplies training labels",
                    "smiles_policy": "display only; all methods consume registered Mol or preserved source graph"}
        (destination / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception as exc:
        (destination / "INCOMPLETE.json").write_text(json.dumps({"status": "INCOMPLETE", "error_type": type(exc).__name__,
            "error": str(exc)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


class GraphCache:
    def __init__(self, root, *, source_sha256, spec=PGT_FEATURE_SPEC):
        self.root = Path(root).resolve()
        self.spec = spec
        spec.validate_runtime()
        marker = self.root / "cache_manifest.json"
        if not marker.is_file() or (self.root / "INCOMPLETE.json").exists():
            raise ValueError("Not a complete new PGT cache; legacy stores cannot be reused")
        self.manifest = json.loads(marker.read_text(encoding="utf-8"))
        meta = self.manifest
        if (meta.get("status") != "COMPLETE" or meta.get("cache_format") != CACHE_FORMAT or
                meta.get("feature_spec") != spec.metadata() or meta.get("source_sha256") != source_sha256):
            raise ValueError("Cache schema, source or FeatureSpec mismatch")
        identity = {"cache_format": CACHE_FORMAT, "source_sha256": source_sha256,
                    "feature_spec_sha256": spec.sha256, "records": meta["records"]}
        if meta["cache_id"] != "cache_" + digest(identity) or len(meta["records"]) != meta["record_count"]:
            raise ValueError("Cache record registry mismatch")
        if file_sha256(self.root / "graphs.sqlite3") != meta["database_sha256"]:
            raise ValueError("Cache database checksum mismatch")

    def __len__(self):
        return self.manifest["record_count"]

    def read(self, offset, *, verify_chemistry=False):
        if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset < len(self):
            raise IndexError(offset)
        # No inherited/shared LMDB handle; each short read owns its connection.
        uri = (self.root / "graphs.sqlite3").as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            row = connection.execute("SELECT sample_id, raw_row_index, input_id, payload_sha256, payload "
                                     "FROM graphs WHERE cache_offset=?", (offset,)).fetchone()
        if row is None or [offset, *row[:4]] != self.manifest["records"][offset]:
            raise ValueError("Cache offset/sample/source-row mapping mismatch")
        if hashlib.sha256(row[4]).hexdigest() != row[3]:
            raise ValueError("Graph payload checksum mismatch")
        with np.load(io.BytesIO(row[4]), allow_pickle=False) as arrays:
            meta = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
            graph = Data(**{name: torch.from_numpy(arrays[name].copy()) for name in ARRAY_FIELDS},
                         **{name: meta[name] for name in META_FIELDS}, y=torch.tensor([float("nan")]))
            mol = Chem.Mol(arrays["mol_binary"].tobytes())
        if (meta["cache_offset"] != offset or meta["sample_id"] != row[0] or meta["raw_row_index"] != row[1] or
                meta["model_input_id"] != row[2] or meta["source_sha256"] != self.manifest["source_sha256"]):
            raise ValueError("Graph payload record identity mismatch")
        validate_graph(graph, self.spec)
        # Full chemistry validation runs on every build record. Recheck on audit,
        # not on every epoch; payload/database hashes bind ordinary reads.
        if verify_chemistry:
            _verify_molecular_payload(graph, mol, self.spec)
        graph.cache_offset = offset
        graph.provenance = meta["provenance"]
        graph.source_sha256 = meta["source_sha256"]
        return graph, mol

    def __getitem__(self, offset):
        return self.read(offset)[0]
