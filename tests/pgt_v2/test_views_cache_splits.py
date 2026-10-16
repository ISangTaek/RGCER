import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
import torch

from molecular_features import PGT_FEATURE_SPEC
from pgt_v2.cache import CacheEntry, GraphCache, build_cache
from pgt_v2.chemistry import graph_from_input, prepare_molecule
from pgt_v2.data import (PGTCollator, align_task_columns, legacy_cache_row_mapping,
                         make_training_view, sparse_visible_mask)
from pgt_v2.splits import allocate_groups, plan_folds, structure_groups


def view_inputs():
    n = 26
    labels = np.stack([np.arange(n), 3 * np.arange(n) + 4, np.full(n, 9)], axis=1).astype(float)
    roles = ["train"] * 22 + ["validation"] * 2 + ["test"] * 2
    visible = np.ones((n, 3), dtype=bool)
    visible[22:] = False
    visible[1:22:4, 0] = False
    return dict(labels=labels, label_task_names=["target", "source", "constant"],
                task_names=["target", "source", "constant"], record_ids=[f"r{i}" for i in range(n)],
                raw_row_indices=list(range(n)),
                group_ids=[f"g{i}" for i in range(n)], chemical_ids=[f"c{i}" for i in range(n)],
                roles=roles, visible_mask=visible, source_sha256="f" * 64, fold_id=0)


def test_hidden_train_val_test_poison_cannot_change_statistics_or_adjacency():
    inputs = view_inputs()
    reference = make_training_view(**inputs)
    assert reference.adjacency[0, 1] > 0
    assert reference.scale_fallback.tolist() == [False, False, True]
    for poison in (1e8, -1e8, np.nan, np.inf):
        poisoned = inputs["labels"].copy()
        poisoned[~inputs["visible_mask"]] = poison
        changed = make_training_view(**{**inputs, "labels": poisoned})
        assert reference.view_id == changed.view_id
        for field in ("mean", "scale", "count", "group_count", "adjacency", "visible_mask", "visible_labels"):
            np.testing.assert_array_equal(getattr(reference, field), getattr(changed, field))
    altered = inputs["labels"].copy(); altered[0, 0] += 1
    assert make_training_view(**{**inputs, "labels": altered}).view_id != reference.view_id
    with pytest.raises(ValueError):
        reference.visible_labels.setflags(write=True)
    with pytest.raises(ValueError, match="validation/test"):
        reference.labels_for(["r25"])
    np.testing.assert_allclose(np.nanmean(reference.labels_for(reference.train_record_ids, standardized=True), axis=0), 0, atol=1e-12)


def test_task_reordering_applies_to_both_mask_and_labels():
    args = view_inputs()
    a = make_training_view(**args)
    order = [2, 0, 1]
    b = make_training_view(**{**args, "labels": args["labels"][:, order],
        "visible_mask": args["visible_mask"][:, order], "label_task_names": [args["label_task_names"][i] for i in order]})
    assert a.view_id == b.view_id
    np.testing.assert_array_equal(a.visible_labels, b.visible_labels)
    task_order = json.loads((Path(__file__).parent / "fixtures/task_orders.json").read_text(encoding="utf-8"))
    assert len(task_order["csv"]) == 59
    assert sum(a != b for a, b in zip(task_order["csv"], task_order["canonical"])) == 56
    table = np.arange(3 * 59).reshape(3, 59)
    canonical = align_task_columns(table, task_order["csv"], task_order["canonical"])
    restored = align_task_columns(canonical, task_order["canonical"], task_order["csv"])
    assert np.array_equal(table, restored)
    with pytest.raises(ValueError, match="same set"):
        align_task_columns(table, task_order["csv"], task_order["canonical"][:-1] + ["wrong"])


def test_zero_visible_task_cross_partition_and_visible_infinity_are_errors():
    args = view_inputs()
    invisible = args["visible_mask"].copy(); invisible[:, 0] = False
    with pytest.raises(ValueError, match="Zero visible"):
        make_training_view(**{**args, "visible_mask": invisible})
    leaking = args["visible_mask"].copy(); leaking[-1, 0] = True
    with pytest.raises(ValueError, match="Only training"):
        make_training_view(**{**args, "visible_mask": leaking})
    groups = args["group_ids"].copy(); groups[-1] = groups[0]
    with pytest.raises(ValueError, match="crosses"):
        make_training_view(**{**args, "group_ids": groups})
    chemicals = args["chemical_ids"].copy(); chemicals[1] = chemicals[0]
    with pytest.raises(ValueError, match="independent group"):
        make_training_view(**{**args, "chemical_ids": chemicals})
    labels = args["labels"].copy(); labels[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        make_training_view(**{**args, "labels": labels})


def test_training_view_attach_keeps_task_axis_and_rejects_mixed_views():
    a = graph_from_input(prepare_molecule("CC"), sample_id="r0", raw_row_index=0)
    b = graph_from_input(prepare_molecule("CCO"), sample_id="r1", raw_row_index=1)
    args = view_inputs()
    args["chemical_ids"][:2] = [a.model_input_id, b.model_input_id]
    a.source_sha256 = b.source_sha256 = args["source_sha256"]
    view = make_training_view(**args)
    wrong = a.clone(); wrong.model_input_id = b.model_input_id
    with pytest.raises(ValueError, match="identity mismatch"):
        view.attach(wrong)
    wrong = a.clone(); wrong.raw_row_index = 1
    with pytest.raises(ValueError, match="identity mismatch"):
        view.attach(wrong)
    wrong = a.clone(); wrong.source_sha256 = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        view.attach(wrong)
    a, b = view.attach(a), view.attach(b)
    batch = PGTCollator()([a, b])
    assert batch.y.shape == (2, 3)
    np.testing.assert_allclose(batch.y.numpy(), view.labels_for(["r0", "r1"], standardized=True), rtol=1e-6, atol=1e-6)
    assert batch.training_view_id == view.view_id
    b.training_view_id = "wrong"
    with pytest.raises(ValueError, match="Mixed"):
        PGTCollator()([a, b])


def test_sparse_views_are_nested_identity_level_and_leave_source_labels_fixed():
    chemicals = [f"c{i}" for i in range(20)] + ["c0", "val", "test"]
    roles = ["train"] * 21 + ["validation", "test"]
    observed = np.ones((23, 2), dtype=bool)
    kwargs = dict(observed=observed, task_names=["target", "source"], roles=roles, chemical_ids=chemicals,
                  target_tasks=["target"], fold_id=2, subset_seed=1042)
    masks = [sparse_visible_mask(**kwargs, fraction=p) for p in (0.1, 0.3, 1.0)]
    for left, right in zip(masks, masks[1:]):
        assert not (left & ~right).any()
        assert np.array_equal(left[:, 1], right[:, 1])
    for mask, expected_count in zip(masks, (2, 6, 20)):
        assert len({c for c, use in zip(chemicals, mask[:, 0]) if use}) == expected_count
        assert mask[0, 0] == mask[20, 0]
        assert not mask[-2:].any()
    p = np.random.default_rng(5).permutation(23)
    shuffled = sparse_visible_mask(**{**kwargs, "observed": observed[p], "chemical_ids": [chemicals[i] for i in p],
                                     "roles": [roles[i] for i in p]}, fraction=0.3)
    assert np.array_equal(shuffled[np.argsort(p)], masks[1])


def test_old_cache_offsets_are_not_original_rows():
    ids = ["sample0", "sample2", "sample5"]
    manifest = [{"sample_id": f"sample{i}", "row_index": i} for i in (5, 0, 2)]
    result = legacy_cache_row_mapping(ids, manifest)
    assert [r["cache_offset"] for r in result] == [0, 1, 2]
    assert [r["raw_row_index"] for r in result] == [0, 2, 5]
    with pytest.raises(ValueError, match="missing"):
        legacy_cache_row_mapping(["unknown"], manifest)


def test_cache_preserves_mol_binary_real_stereo_maps_and_coexists_with_legacy(tmp_path, c01_cases):
    old = tmp_path / "legacy"; old.mkdir()
    (old / "CURRENT").write_text("untouched-old-cache", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy CURRENT"):
        build_cache([], old / "new", source_sha256="f" * 64)
    with pytest.raises(ValueError, match="legacy stores"):
        GraphCache(old, source_sha256="f" * 64)
    cases = [prepare_molecule(c["raw_smiles_upper"]) for c in c01_cases]
    new = tmp_path / "pgt" / PGT_FEATURE_SPEC.sha256
    entries = [CacheEntry(mol, c["record_id"], c["raw_row_index"], {"reference_only": True}) for mol, c in zip(cases, c01_cases)]
    build_cache(entries, new, source_sha256="f" * 64)
    cached = GraphCache(new, source_sha256="f" * 64)
    assert len(cached) == 4
    for i, original in enumerate(cases):
        graph, mol = cached.read(i, verify_chemistry=True)
        assert graph.cache_offset == i
        assert graph.raw_row_index == c01_cases[i]["raw_row_index"]
        assert graph.model_input_id == original.model_input_id
        assert prepare_molecule(mol).model_input_id == original.model_input_id
        assert tuple(graph.raw_to_model) == original.raw_to_model
        assert tuple(graph.symmetry_classes) == original.symmetry_classes
        assert torch.isnan(graph.y).all()
    assert (old / "CURRENT").read_text(encoding="utf-8") == "untouched-old-cache"
    with pytest.raises(FileExistsError):
        build_cache(entries, new, source_sha256="f" * 64)
    with pytest.raises(ValueError, match="source"):
        GraphCache(new, source_sha256="0" * 64)
    metadata = json.loads((new / "cache_manifest.json").read_text(encoding="utf-8"))
    metadata["feature_spec"]["feature_schema_version"] = "old"
    (new / "cache_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        GraphCache(new, source_sha256="f" * 64)


def test_failed_or_corrupted_cache_cannot_be_read(tmp_path):
    entry = CacheEntry(prepare_molecule("CCO"), "row_1", 1, {})
    with pytest.raises(Exception, match="UNIQUE"):
        build_cache([entry, entry], tmp_path / "failed", source_sha256="f" * 64)
    with pytest.raises(ValueError, match="complete"):
        GraphCache(tmp_path / "failed", source_sha256="f" * 64)
    build_cache([entry], tmp_path / "ok", source_sha256="f" * 64)
    with (tmp_path / "ok/graphs.sqlite3").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        GraphCache(tmp_path / "ok", source_sha256="f" * 64)


def test_structure_groups_stereoisomers_scaffolds_acyclic_and_parent_policy():
    sources = {"r": "C[C@H](N)C(=O)O", "s": "C[C@@H](N)C(=O)O", "alcohol": "CCO", "same": "OCC",
               "benzene": "c1ccccc1", "phenol": "Oc1ccccc1", "salt": "CC(=O)[O-].[Na+]", "acid": "CC(=O)O"}
    inputs = {name: prepare_molecule(s) for name, s in sources.items()}
    groups = structure_groups(inputs)
    assert groups["r"] == groups["s"]
    assert groups["alcohol"] == groups["same"]
    assert groups["benzene"] == groups["phenol"]
    assert groups["benzene"] != groups["alcohol"]
    assert groups["salt"] != groups["acid"]
    assert groups == structure_groups(dict(reversed(list(inputs.items()))))
    links = [(inputs["salt"].model_input_id, inputs["acid"].model_input_id)]
    with pytest.raises(ValueError, match="written reviewed"):
        structure_groups(inputs, reviewed_parent_links=links)
    linked = structure_groups(inputs, reviewed_parent_links=links, parent_rule="fixture-only explicit reviewed pair")
    assert linked["salt"] == linked["acid"]


def test_presence_only_split_determinism_whole_groups_coverage_and_fallback():
    groups = [f"g{i}" for i in range(30) for _ in range(2)]
    observed = np.ones((len(groups), 3), dtype=bool)
    plan = plan_folds(groups, observed)
    assert plan["selected_outer_folds"] == 5
    for fold in plan["attempts"][0]["folds"]:
        for group in set(groups):
            assert len({role for role, g in zip(fold["roles"], groups) if g == group}) == 1
    kwargs = dict(fractions=[0.8, 0.2], minimum_groups=[5, 2])
    original, _ = allocate_groups(groups, observed, **kwargs)
    p = np.random.default_rng(33).permutation(len(groups))
    changed, _ = allocate_groups([groups[i] for i in p], observed[p], **kwargs)
    assert original == changed
    with pytest.raises(ValueError, match="boolean"):
        allocate_groups(groups, observed.astype(float), **kwargs)
    fallback = plan_folds([f"g{i}" for i in range(9)], np.ones((9, 1), dtype=bool))
    assert [a["outer_folds"] for a in fallback["attempts"]] == [5, 3]
    # Nine groups cannot meet 5 train + 2 validation + >=3 test in every 3-fold split.
    assert fallback["selected_outer_folds"] is None
    assert fallback["status"] == "STOP_COVERAGE_FAILED"
    big_groups = ["huge"] * 30 + [f"g{i}" for i in range(15)]
    assign, report = allocate_groups(big_groups, np.ones((45, 1), bool), **kwargs)
    assert "huge" in assign
    assert sum(report["record_counts"]) == 45
