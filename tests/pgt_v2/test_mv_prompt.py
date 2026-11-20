"""Unit and synthetic integration tests for pgt_v2.mv_prompt."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from molecular_features import PGT_FEATURE_SPEC
from pgt_v2.artifacts import save_view, write_json
from pgt_v2.cache import CacheEntry, build_cache
from pgt_v2.chemistry import graph_from_input, prepare_molecule
from pgt_v2.data import make_training_view
from pgt_v2.mv_prompt.features import (
    DESCRIPTOR_NAMES,
    build_mv_cache,
    fit_descriptor_scaler,
    load_mv_cache,
    transform_descriptors,
)
from pgt_v2.mv_prompt.models import build_mv_model, method_input_needs
from pgt_v2.mv_prompt.training import (
    endpoint_weights_for_gamma,
    tempered_mse_loss,
    train_and_evaluate,
)

TASKS = (
    "mouse_oral_LD50",
    "mouse_intravenous_LD50",
    "rat_oral_LD50",
    "rat_intravenous_LDLo",
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
    "mammal (species unspecified)_subcutaneous_LD50",
    "frog_subcutaneous_LDLo",
    "guinea pig_intravenous_LDLo",
)

SMILES_POOL = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "c1ccc(O)cc1",
    "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "O=C(O)c1ccccc1OC(=O)C",
    "CCN(CC)CC",
    "c1ccncc1",
    "Clc1ccccc1",
    "O=[N+]([O-])c1ccccc1",
    "CCCCO",
    "CC(C)(C)O",
    "c1ccc2ccccc2c1",
    "CC1=CC=C(C=C1)S(=O)(=O)N",
    "COC(=O)c1ccccc1",
    "c1ccc(N)cc1",
    "CC(=O)c1ccccc1",
    "FC(F)(F)c1ccccc1",
    "Brc1ccccc1",
    "CCC(=O)OCC",
    "C1CCCCC1",
    "C1CCNCC1",
]


def _make_synthetic_bundle(tmp_path: Path):
    source_sha = "a" * 64
    entries = []
    records = []
    for idx, smi in enumerate(SMILES_POOL):
        mol_in = prepare_molecule(smi)
        entries.append(
            CacheEntry(
                molecule=mol_in,
                sample_id=f"rec_{idx}",
                raw_row_index=idx,
                provenance={"source": "synthetic"},
            )
        )
        records.append(
            {
                "record_id": f"rec_{idx}",
                "cache_offset": idx,
                "raw_row_index": idx,
                "group_id": f"grp_{idx}",
                "chemical_id": mol_in.model_input_id,
            }
        )
    cache_dir = tmp_path / "graph_cache"
    cache_manifest = build_cache(entries, cache_dir, source_sha256=source_sha, spec=PGT_FEATURE_SPEC)

    rng = np.random.default_rng(123)
    n = len(SMILES_POOL)
    t = len(TASKS)
    labels = rng.normal(3.0, 1.0, size=(n, t))
    roles = np.array(["train"] * 16 + ["validation"] * 4 + ["test"] * 4)
    # Introduce realistic sparsity on train while ensuring >=2 labels per task
    vis = np.zeros((n, t), dtype=bool)
    vis[:16, :4] = True
    for col in range(4, t):
        vis[:6, col] = True

    view = make_training_view(
        labels=labels,
        label_task_names=TASKS,
        task_names=TASKS,
        record_ids=[r["record_id"] for r in records],
        raw_row_indices=[r["raw_row_index"] for r in records],
        group_ids=[r["group_id"] for r in records],
        chemical_ids=[r["chemical_id"] for r in records],
        roles=roles.tolist(),
        visible_mask=vis,
        source_sha256=source_sha,
        fold_id="S0",
    )
    view_dir = tmp_path / "view_S0"
    save_view(
        view_dir,
        view,
        cache_manifest=cache_manifest,
        records=records,
        roles=roles,
        labels=labels,
        fold="S0",
    )
    groups = {
        "task_names": list(TASKS),
        "L11": list(TASKS[:2]),
        "M27": list(TASKS[2:4]),
        "S21": list(TASKS[4:]),
    }
    mv_cache_path = build_mv_cache(
        cache_dir, tmp_path / "mv_cache.pt", source_sha256=source_sha, log_every=0
    )
    return mv_cache_path, view_dir, groups


def test_descriptor_scaler_clipping_and_train_only():
    raw = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [1e9, -1e9]], dtype=np.float64)
    miss = np.array([[False, False], [False, True], [False, False], [False, False]], dtype=bool)
    scaler = fit_descriptor_scaler(raw, miss, np.array([0, 1, 2]), clip_value=5.0)
    out = transform_descriptors(raw, miss, np.array([0, 1, 2, 3]), scaler)
    assert out.shape == (4, 4)
    assert np.isfinite(out).all()
    # Extreme test outlier must be clipped to [-5, 5], preventing Kappa3 explosion
    assert out[3, 0] == pytest.approx(5.0)
    assert out[3, 1] == pytest.approx(-5.0)
    # Missing indicator preserved
    assert out[1, 3] == 1.0
    assert out[0, 3] == 0.0


def test_tempered_loss_interpolates_observed_and_endpoint():
    counts = np.array([1000.0, 100.0, 10.0])
    w0 = endpoint_weights_for_gamma(counts, 0.0)
    w05 = endpoint_weights_for_gamma(counts, 0.5)
    w1 = endpoint_weights_for_gamma(counts, 1.0)
    assert np.allclose(w0, 1.0)
    assert w05[2] / w05[0] == pytest.approx(10.0, rel=1e-4)
    assert w1[2] / w1[0] == pytest.approx(100.0, rel=1e-4)

    pred = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.5, float("nan"), 2.0], [0.5, 1.0, float("nan")]])
    loss = tempered_mse_loss(pred, target, torch.from_numpy(w05))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_all_mv_models_train_and_predict_end_to_end(tmp_path):
    mv_cache_path, view_dir, groups = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    assert store.desc_raw.shape[1] == len(DESCRIPTOR_NAMES)

    methods = [
        "TOX_AUTHOR",
        "MV_MLP",
        "GINE_ONLY",
        "GRAPHORMER_ONLY",
        "MV_BIOPGT_NOGRAPH",
        "MV_BIOPGT_NOPROMPT_GINE",
        "MV_BIOPGT_GINE",
        "MV_BIOPGT_GRAPHORMER",
        "RF",
    ]
    for method in methods:
        cfg = (
            {"n_estimators": 10, "n_jobs": 1}
            if method == "RF"
            else {
                "max_epochs": 2,
                "min_epochs": 1,
                "patience": 2,
                "batch_size": 8,
                "eval_batch": 8,
                "hidden_dim": 32,
                "graph_layers": 2,
                "graph_heads": 4,
                "gamma": 0.25,
            }
        )
        if method == "TOX_AUTHOR":
            cfg = {"max_epochs": 2, "min_epochs": 2, "patience": 3, "batch_size": 8, "eval_batch": 8}
        out_dir = tmp_path / f"run_{method}"
        summary = train_and_evaluate(
            store=store,
            view_root=view_dir,
            endpoint_groups=groups,
            method=method,
            seed=42,
            config=cfg,
            output_dir=out_dir,
            device="cpu",
            save_checkpoint_file=(method != "RF"),
        )
        assert summary["status"] == "COMPLETE"
        assert np.isfinite(summary["test_groups"]["ALL59"]["macro_rmse"])
        assert (out_dir / "test_predictions.npz").is_file()
        assert (out_dir / "validation_metrics.json").is_file()


def test_prompt_intervention_changes_predictions(tmp_path):
    mv_cache_path, view_dir, _ = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    counts = np.array([100.0] * len(TASKS))
    model = build_mv_model("MV_BIOPGT_GINE", TASKS, counts, config={"hidden_dim": 32, "graph_layers": 2})
    model.eval()
    offsets = np.arange(4, dtype=np.int64)
    scaler = fit_descriptor_scaler(store.desc_raw, store.desc_missing, offsets)
    desc = transform_descriptors(store.desc_raw, store.desc_missing, offsets, scaler)
    needs = method_input_needs("MV_BIOPGT_GINE")
    payload = store.build_batch(offsets, desc_array=desc, device="cpu", **needs)
    with torch.no_grad():
        pred_normal, aux = model(payload, return_aux=True)
        perm = torch.roll(aux["prompts"], shifts=1, dims=0)
        pred_swapped = model(payload, prompt_override=perm)
    assert aux["atom_attention"].shape[0] == 4
    assert aux["atom_attention"].shape[1] == len(TASKS)
    assert not torch.allclose(pred_normal, pred_swapped, atol=1e-6)


def test_graphormer_preloaded_batch_matches_official_collator(tmp_path):
    from pgt_v2.cache import GraphCache
    from pgt_v2.data import PGTCollator, make_backbone

    mv_cache_path, view_dir, _ = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    graph_cache = GraphCache(tmp_path / "graph_cache", source_sha256="a" * 64, verify_database=False)
    offsets = np.array([0, 3, 5, 7], dtype=np.int64)
    official_graphs = [graph_cache.read(int(o))[0] for o in offsets]
    official_batch = PGTCollator()(official_graphs)
    fast_payload = store.build_batch(
        offsets,
        desc_array=None,
        need_avalon=False,
        need_morgan=False,
        need_desc=False,
        graph_mode="graphormer",
        device="cpu",
    )
    torch.manual_seed(7)
    backbone = make_backbone(hidden_dim=32, ffn_dim=64, num_heads=4, num_layers=2, dropout=0.0).eval()
    with torch.no_grad():
        t_official = backbone.encode_tokens(official_batch)
        t_fast = backbone.encode_tokens(fast_payload.graphormer_batch)
    assert torch.allclose(t_official.graph_token, t_fast.graph_token, atol=1e-6)
    assert torch.allclose(t_official.atom_tokens, t_fast.atom_tokens, atol=1e-6)
    assert torch.equal(t_official.atom_mask, t_fast.atom_mask)


def test_fast_toxacol_matches_official_toxacol(tmp_path):
    from pgt_v2.cache import GraphCache
    from pgt_v2.models import build_model, molecular_features

    mv_cache_path, view_dir, _ = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    graph_cache = GraphCache(tmp_path / "graph_cache", source_sha256="a" * 64, verify_database=False)
    adj = np.eye(len(TASKS), dtype=np.float32)
    torch.manual_seed(11)
    official = build_model("TOX_AUTHOR", TASKS, {"dropout": 0.1}, adjacency=adj, device="cpu").eval()
    fast = build_mv_model("TOX_AUTHOR", TASKS, np.ones(len(TASKS)), adjacency=adj, config={"dropout": 0.1}, device="cpu").eval()
    fast.net.load_state_dict(official.state_dict())

    offsets = np.array([0, 1, 4, 8], dtype=np.int64)
    mols = [graph_cache.read(int(o))[1] for o in offsets]
    fp_official = torch.tensor(molecular_features(mols, "avalon"))
    fast_payload = store.build_batch(
        offsets,
        desc_array=None,
        need_avalon=True,
        need_morgan=False,
        need_desc=False,
        graph_mode=None,
        device="cpu",
    )
    assert torch.equal(fp_official, fast_payload.avalon)
    with torch.no_grad():
        y_off = official(fp_official)
        y_fast = fast(fast_payload)
    assert torch.allclose(y_off, y_fast, atol=1e-6)


def test_gate_summarize_and_collect(tmp_path):
    from pgt_v2.mv_prompt.gate import collect_gate, summarize_gate

    mv_cache_path, view_dir, groups = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    views_root = tmp_path / "views_S0"
    (views_root / "FULL").mkdir(parents=True)
    (views_root / "S21_P10").mkdir(parents=True)
    for sub in ("FULL", "S21_P10"):
        for f in view_dir.iterdir():
            (views_root / sub / f.name).write_bytes(f.read_bytes())
    groups_path = tmp_path / "groups.json"
    write_json(groups_path, groups)

    work = tmp_path / "gate_work"
    runs_dir = work / "runs"
    cfg_small = {"max_epochs": 1, "min_epochs": 1, "patience": 2, "batch_size": 8, "eval_batch": 8, "hidden_dim": 32, "graph_layers": 2, "graph_heads": 4, "gamma": 0.25}
    cfg_tox = {"max_epochs": 1, "min_epochs": 1, "patience": 2, "batch_size": 8, "eval_batch": 8}
    for rid, method, budget, cfg in [
        ("FULL_MV_BIOPGT_GINE_g025_s42", "MV_BIOPGT_GINE", "FULL", cfg_small),
        ("FULL_TOX_AUTHOR_s42", "TOX_AUTHOR", "FULL", cfg_tox),
        ("FULL_MV_MLP_s42", "MV_MLP", "FULL", cfg_small),
        ("FULL_MV_BIOPGT_NOGRAPH_gbest_s42", "MV_BIOPGT_NOGRAPH", "FULL", cfg_small),
        ("FULL_MV_BIOPGT_NOPROMPT_GINE_gbest_s42", "MV_BIOPGT_NOPROMPT_GINE", "FULL", cfg_small),
        ("P10_MV_BIOPGT_BEST_s42", "MV_BIOPGT_GINE", "S21_P10", cfg_small),
        ("P10_TOX_AUTHOR_s42", "TOX_AUTHOR", "S21_P10", cfg_tox),
    ]:
        train_and_evaluate(
            store=store,
            view_root=views_root / budget,
            endpoint_groups=groups,
            method=method,
            seed=42,
            config=cfg,
            output_dir=runs_dir / rid,
            device="cpu",
            save_checkpoint_file=False,
        )
    write_json(
        work / "validation_selection.json",
        {
            "best_gamma": 0.25,
            "best_backbone": "GINE",
            "best_method": "MV_BIOPGT_GINE",
            "best_full_run_id": "FULL_MV_BIOPGT_GINE_g025_s42",
        },
    )
    decision = summarize_gate(work_dir=work, views_root=views_root, groups_path=groups_path)
    assert decision["verdict"] in ("PASS", "PARTIAL_PASS_NO_GRAPH_GAIN", "FAIL_FALLBACK_ROUTE1")
    assert (work / "gate_report.md").is_file()
    assert (work / "gate_summary.csv").is_file()
    archive = collect_gate(work)
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()

