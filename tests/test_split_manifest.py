from pathlib import Path

from split_manifest import create_split_manifest, load_manifest, manifest_hash, write_manifest


def test_manifest_keeps_canonical_and_scaffold_groups_together(tmp_path: Path):
    records = [
        {"sample_id": "a1", "canonical_smiles": "CCO", "scaffold": "acyclic_a"},
        {"sample_id": "a2", "canonical_smiles": "CCO", "scaffold": "acyclic_a"},
        {"sample_id": "b1", "canonical_smiles": "c1ccccc1", "scaffold": "benzene"},
        {"sample_id": "c1", "canonical_smiles": "CCN", "scaffold": "acyclic_c"},
        {"sample_id": "d1", "canonical_smiles": "CCC", "scaffold": "acyclic_d"},
        {"sample_id": "e1", "canonical_smiles": "CCCl", "scaffold": "acyclic_e"},
        {"sample_id": "f1", "canonical_smiles": "CCBr", "scaffold": "acyclic_f"},
        {"sample_id": "g1", "canonical_smiles": "CCF", "scaffold": "acyclic_g"},
    ]
    manifest = create_split_manifest(records, ratios={
        "train": 0.5, "validation": 0.2, "calibration": 0.2, "test": 0.1
    })
    path = tmp_path / "split_manifest.json"
    write_manifest(manifest, path)
    loaded = load_manifest(path)
    assert manifest_hash(manifest) == manifest_hash(loaded)

    locations = {}
    for record in loaded["records"]:
        for field in ("canonical_smiles", "scaffold"):
            value = record[field]
            locations.setdefault((field, value), record["split"])
            assert locations[(field, value)] == record["split"]


def test_empty_subset_is_really_empty():
    from dataset import DataloaderWrapper

    loader = DataloaderWrapper._loader([], [], "validation", 4, 0, lambda x: x)
    assert len(loader) == 0
    assert len(loader.dataset) == 0
