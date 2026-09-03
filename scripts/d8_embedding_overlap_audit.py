"""D8 result-correction review P0-1: embedding source/target overlap audit.

Before any cross-species embedding claim may be made, the Animal probe set
and the Human validation pool must be checked for molecule-level overlap —
a shared molecule would make the "cross-species neighborhood retention"
partly a self-comparison (transductive leakage).

Checks exact canonical-SMILES overlap AND Bemis-Murcko scaffold overlap.
ANY exact overlap is a hard stop: STOP_EMBEDDING_ANALYSIS.  Scaffold
overlap is reported (same scaffold ≠ same molecule) and does not stop the
pipeline, but must be discussed if non-zero.

Outputs embedding/D8_EMBEDDING_OVERLAP_AUDIT.json:
    {
      "exact_smiles_overlap": 0,
      "scaffold_overlap": 0,
      "human_samples_checked": 39,
      "animal_samples_checked": 500,
      ...
      "test_accessed": false
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def canonical_smiles(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def murcko_scaffold_canonical(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def compute_overlap(human_ids, animal_ids, smiles_by_id: dict[str, str]) -> dict:
    """Pure overlap computation over canonical SMILES and scaffolds."""
    human_ids = [str(value) for value in human_ids]
    animal_ids = [str(value) for value in animal_ids]

    def canonical_map(ids):
        result, unparsed = {}, []
        for sample_id in ids:
            canon = canonical_smiles(smiles_by_id.get(sample_id, ""))
            if canon is None:
                unparsed.append(sample_id)
                continue
            result[sample_id] = canon
        return result, unparsed

    human_map, human_unparsed = canonical_map(human_ids)
    animal_map, animal_unparsed = canonical_map(animal_ids)

    human_scaffolds = {
        sample_id: murcko_scaffold_canonical(smiles)
        for sample_id, smiles in human_map.items()
    }
    animal_scaffolds = {
        sample_id: murcko_scaffold_canonical(smiles)
        for sample_id, smiles in animal_map.items()
    }

    human_canon_set = set(human_map.values())
    animal_canon_set = set(animal_map.values())
    exact_overlap = human_canon_set & animal_canon_set
    scaffold_overlap = (
        {value for value in human_scaffolds.values() if value}
        & {value for value in animal_scaffolds.values() if value}
    )
    exact_ids = sorted(
        sample_id for sample_id, smiles in human_map.items() if smiles in exact_overlap
    )
    return {
        "exact_smiles_overlap": len(exact_overlap),
        "scaffold_overlap": len(scaffold_overlap),
        "human_samples_checked": len(human_ids),
        "animal_samples_checked": len(animal_ids),
        "human_smiles_unparsed": len(human_unparsed),
        "animal_smiles_unparsed": len(animal_unparsed),
        "exact_overlap_sample_ids": exact_ids[:20],
        "scaffold_overlap_examples": sorted(scaffold_overlap)[:10],
        "scaffold_overlap_note": (
            "same scaffold does not imply same molecule; discuss if non-zero"
            if scaffold_overlap else ""
        ),
        "audit_sha256": hashlib.sha256(
            "\n".join(sorted(human_map.values()) + sorted(animal_map.values())).encode("utf-8")
        ).hexdigest(),
        "test_accessed": False,
    }


def run_overlap_audit(human_ids, animal_ids, smiles_by_id, output_path: Path) -> dict:
    """Compute, persist and enforce the audit; STOP on exact overlap."""
    audit = compute_overlap(human_ids, animal_ids, smiles_by_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_path}")
    print(
        f"overlap audit: exact={audit['exact_smiles_overlap']} "
        f"scaffold={audit['scaffold_overlap']} "
        f"(human n={audit['human_samples_checked']}, animal n={audit['animal_samples_checked']})"
    )
    if audit["exact_smiles_overlap"] > 0:
        print("STOP_EMBEDDING_ANALYSIS: exact molecule overlap between the "
              "Animal probe and the Human validation pool")
        raise SystemExit(2)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animal_probe_manifest",
        default="artifacts/results/d8_result_correction/embedding/D8_EMBEDDING_PROBE_MANIFEST.csv",
    )
    parser.add_argument(
        "--output_path",
        default="artifacts/results/d8_result_correction/embedding/D8_EMBEDDING_OVERLAP_AUDIT.json",
    )
    parser.add_argument("--data_store_dir", default=None)
    parser.add_argument("--raw_csv", default=None)
    args = parser.parse_args()

    import csv

    from scripts.d8_label_scaling_subset_audit import load_smiles_by_row

    probe_ids = []
    with Path(args.animal_probe_manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            probe_ids.append(row["sample_id"])

    if not args.raw_csv:
        from experiment_config import TOXACUTE_RAW_CSV

        raw_csv = TOXACUTE_RAW_CSV
    else:
        raw_csv = args.raw_csv
    smiles_by_row = load_smiles_by_row(Path(raw_csv))

    from toxacute_datastore import ToxAcuteDataStore

    store_dir = args.data_store_dir
    if not store_dir:
        from experiment_config import TOXACUTE_DATASTORE_DIR

        store_dir = TOXACUTE_DATASTORE_DIR
    store = ToxAcuteDataStore.resolve(store_dir)
    human_ids = []
    for task in ("man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"):
        for global_index in store.get_task_indices(task, split="validation"):
            sample_id = store.sample_ids[int(global_index)]
            sample_id = sample_id.decode() if isinstance(sample_id, bytes) else str(sample_id)
            if sample_id not in human_ids:
                human_ids.append(sample_id)

    smiles_by_id = {sid: smiles_by_row.get(sid, "") for sid in set(probe_ids) | set(human_ids)}
    run_overlap_audit(human_ids, probe_ids, smiles_by_id, Path(args.output_path))


if __name__ == "__main__":
    main()
