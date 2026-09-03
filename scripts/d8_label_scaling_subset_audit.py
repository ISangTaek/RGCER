"""D8 result-correction Task F: label-scaling subset nestedness audit.

Rebuilds, for every seed and fraction, the EXACT Human3 train subsets used
by the label-scaling runs — the same ``np.random.RandomState(split_seed)
.permutation(len(train_dataset))`` prefix rule as ``dataset.py`` — and
audits the nesting ladder 10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100% per (seed, endpoint).

Outputs (under --output_dir):
    figure4/D8_LABEL_SCALING_SUBSET_MANIFEST.csv   fraction,seed,endpoint,
                                                   sample_id,scaffold
    figure4/D8_LABEL_SCALING_NESTEDNESS_AUDIT.json nested_* flags +
                                                   per-ladder sizes

The scaffold is the Bemis-Murcko scaffold of the molecule's SMILES taken
from the raw CSV (sample_id ``row_N`` = N-th CSV data row).  No model is
loaded and no training is repeated; only the train split is read.
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

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
FRACTION_LADDER = [10, 25, 50, 75, 100]


def subset_sample_ids(dataset, fraction: float, split_seed: int) -> list[str]:
    """Reproduce dataset.py's deterministic subsample and return sample ids."""
    import numpy as np

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(dataset))
    n_keep = max(1, int(len(dataset) * fraction))
    keep = sorted(perm[:n_keep].tolist())
    return [str(dataset.get_sample_id(index)) for index in keep]


def task_train_dataset(store, task: str):
    from toxacute_datastore import ToxAcuteTaskDataset

    return ToxAcuteTaskDataset(store, task, split="train", max_nodes=None,
                               label_provider=None)


def murcko_scaffold(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def load_smiles_by_row(raw_csv: Path) -> dict[str, str]:
    mapping = {}
    with Path(raw_csv).open(newline="", encoding="utf-8") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            mapping[f"row_{position}"] = row["SMILES"]
    return mapping


def nesting_ladder(ids_by_fraction: dict[int, set[str]]) -> dict:
    rungs = {}
    ok = True
    for smaller, larger in zip(FRACTION_LADDER, FRACTION_LADDER[1:]):
        nested = ids_by_fraction[smaller] <= ids_by_fraction[larger]
        rungs[f"nested_{smaller}_{larger}"] = bool(nested)
        ok = ok and nested
    return {"rungs": rungs, "nested": bool(ok)}


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_store_dir", default=None)
    parser.add_argument("--raw_csv", default=None)
    parser.add_argument(
        "--args_json",
        default="artifacts/runs/d8_label_scaling/b1/d8_b1_f50_e40/seed_42/args.json",
        help="any scaling run's args.json; supplies data_store_dir/split_seed convention",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--output_dir", default="artifacts/results/d8_result_correction/figure4"
    )
    args = parser.parse_args()

    run_args = json.loads(Path(args.args_json).read_text(encoding="utf-8"))
    store_dir = args.data_store_dir or run_args.get("data_store_dir")
    raw_csv = args.raw_csv
    if not raw_csv:
        from experiment_config import TOXACUTE_RAW_CSV

        raw_csv = TOXACUTE_RAW_CSV

    from toxacute_datastore import ToxAcuteDataStore

    store = ToxAcuteDataStore.resolve(store_dir)
    smiles_by_row = load_smiles_by_row(Path(raw_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    audit_per_seed = {}
    for seed in args.seeds:
        split_seed = int(run_args.get("split_seed") or 42)
        audit_per_seed[seed] = {}
        for task in HUMAN_TASKS:
            dataset = task_train_dataset(store, task)
            ids_by_fraction = {}
            for fraction in FRACTION_LADDER:
                ids = subset_sample_ids(dataset, fraction / 100.0, split_seed)
                ids_by_fraction[fraction] = set(ids)
                for sample_id in ids:
                    smiles = smiles_by_row.get(sample_id, "")
                    scaffold = murcko_scaffold(smiles) if smiles else None
                    manifest_rows.append(
                        {
                            "fraction": fraction / 100.0,
                            "seed": seed,
                            "endpoint": task,
                            "sample_id": sample_id,
                            "scaffold": scaffold or "",
                        }
                    )
            audit_per_seed[seed][task] = nesting_ladder(ids_by_fraction)

    manifest_path = output_dir / "D8_LABEL_SCALING_SUBSET_MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["fraction", "seed", "endpoint", "sample_id", "scaffold"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {manifest_path} ({len(manifest_rows)} rows)")

    all_nested = all(
        entry["nested"]
        for per_task in audit_per_seed.values()
        for entry in per_task.values()
    )
    seed0 = args.seeds[0]
    sizes = {}
    for fraction in FRACTION_LADDER:
        sizes[str(fraction / 100.0)] = len(
            set(
                row["sample_id"]
                for row in manifest_rows
                if row["seed"] == seed0 and row["endpoint"] == "man_oral_TDLo"
                and row["fraction"] == fraction / 100.0
            )
        )
    audit = {
        "all_nested": bool(all_nested),
        "ladder": "10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ 100 (per seed per endpoint)",
        "per_seed_per_endpoint": {
            str(seed): {
                task: {"nested": entry["nested"], **entry["rungs"]}
                for task, entry in per_task.items()
            }
            for seed, per_task in audit_per_seed.items()
        },
        "man_subset_sizes_seed0": sizes,
        "manifest_sha256": manifest_sha256(manifest_path),
        "test_accessed": False,
    }
    audit_path = output_dir / "D8_LABEL_SCALING_NESTEDNESS_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {audit_path}")
    print(f"all_nested={all_nested}")
    if not all_nested:
        print("WARNING: nesting violated — Figure 4 explanation must pause (plan §60)")


if __name__ == "__main__":
    main()
