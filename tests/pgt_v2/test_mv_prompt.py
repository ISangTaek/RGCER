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
    tempered_huber_loss,
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


def test_tempered_huber_matches_mse_near_zero_and_bounds_outliers():
    ep_w = torch.tensor([1.0, 2.0, 4.0])
    pred_small = torch.tensor([[2.1, 3.0, 4.2], [1.9, 2.5, 3.8]], requires_grad=True)
    target = torch.tensor([[2.0, float("nan"), 4.0], [2.2, 2.0, float("nan")]])
    l_mse = tempered_mse_loss(pred_small, target, ep_w)
    l_hub = tempered_huber_loss(pred_small, target, ep_w, beta=1.0)
    assert torch.allclose(l_mse, l_hub, atol=1e-6)

    pred_out = torch.tensor([[20.0, 3.0, 4.0]], requires_grad=True)
    pred_out_mse = torch.tensor([[20.0, 3.0, 4.0]], requires_grad=True)
    tgt_out = torch.tensor([[2.0, 3.0, 4.0]])
    tempered_huber_loss(pred_out, tgt_out, ep_w, beta=1.0).backward()
    tempered_mse_loss(pred_out_mse, tgt_out, ep_w).backward()
    assert torch.isfinite(pred_out.grad).all()
    assert abs(float(pred_out.grad[0, 0])) < abs(float(pred_out_mse.grad[0, 0])) * 0.2


def test_ordinal_ladder_monotonicity_and_eval_determinism(tmp_path):
    mv_cache_path, _, _ = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    counts = np.array([50.0] * len(TASKS))
    model = build_mv_model(
        "MV_BIOPGT_V2",
        TASKS,
        counts,
        config={"hidden_dim": 32, "fp_hidden_dim": 64, "graph_layers": 2},
    )
    with torch.no_grad():
        model.ord_step_ldlo.fill_(-5.0)
        model.ord_step_tdlo.fill_(-5.0)
    ladder = model.ordinal_threshold_ladder().detach()
    assert ladder.shape == (3,)
    assert float(ladder[0]) == pytest.approx(0.0)
    assert float(ladder[1]) >= float(ladder[0])
    assert float(ladder[2]) >= float(ladder[1])

    offsets = np.arange(6, dtype=np.int64)
    scaler = fit_descriptor_scaler(store.desc_raw, store.desc_missing, offsets)
    desc = transform_descriptors(store.desc_raw, store.desc_missing, offsets, scaler)
    needs = method_input_needs("MV_BIOPGT_V2")
    payload = store.build_batch(offsets, desc_array=desc, device="cpu", **needs)

    model.train()
    pred_tr, train_aux = model(payload, return_train_aux=True)
    assert pred_tr.shape == (6, len(TASKS))
    assert "graph_pred" in train_aux
    assert train_aux["graph_pred"].shape == (6, len(TASKS))

    model.eval()
    with torch.no_grad():
        y1 = model(payload)
        y2 = model(payload)
    assert torch.equal(y1, y2)


def test_v2_models_train_global_center_and_summarize_probe(tmp_path):
    from pgt_v2.mv_prompt.probe_v2 import collect_probe_v2, summarize_probe_v2

    mv_cache_path, view_dir, groups = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    groups_path = tmp_path / "groups.json"
    write_json(groups_path, groups)

    ref_runs = tmp_path / "ref_runs"
    train_and_evaluate(
        store=store,
        view_root=view_dir,
        endpoint_groups=groups,
        method="TOX_AUTHOR",
        seed=42,
        config={"max_epochs": 1, "min_epochs": 1, "patience": 2, "batch_size": 8, "eval_batch": 8},
        output_dir=ref_runs / "FULL_TOX_AUTHOR_s42",
        device="cpu",
        save_checkpoint_file=False,
    )

    work = tmp_path / "probe_work"
    cfg_v2 = {
        "max_epochs": 2,
        "min_epochs": 1,
        "patience": 2,
        "batch_size": 8,
        "eval_batch": 8,
        "hidden_dim": 32,
        "fp_hidden_dim": 64,
        "graph_layers": 2,
        "graph_heads": 4,
        "gamma": 0.50,
    }
    for rid, method in [
        ("FULL_MV_BIOPGT_V2_s42", "MV_BIOPGT_V2"),
        ("FULL_MV_BIOPGT_V2_NOMDROP_s42", "MV_BIOPGT_V2_NOMDROP"),
        ("FULL_MV_BIOPGT_V2_NOGRAPH_s42", "MV_BIOPGT_V2_NOGRAPH"),
        ("FULL_MV_BIOPGT_V2_NOPROMPT_s42", "MV_BIOPGT_V2_NOPROMPT"),
    ]:
        s = train_and_evaluate(
            store=store,
            view_root=view_dir,
            endpoint_groups=groups,
            method=method,
            seed=42,
            config=cfg_v2,
            output_dir=work / "runs" / rid,
            device="cpu",
            save_checkpoint_file=False,
        )
        assert s["status"] == "COMPLETE"
        assert s["label_scaling"] == "global_center"
        assert s["loss_type"] == "huber"
        if method != "MV_BIOPGT_V2_NOPROMPT":
            assert len(s["ordinal_ladder"]) == 3
            assert s["ordinal_ladder"][0] == 0.0
            assert s["ordinal_ladder"][1] >= 0.0
            assert s["ordinal_ladder"][2] >= s["ordinal_ladder"][1]

    report = summarize_probe_v2(work_dir=work, groups_path=groups_path, ref_runs_dir=ref_runs)
    assert "FULL_vs_ToxACoL_ALL59_pct" in report["comparisons"]
    assert (work / "probe_v2_report.md").is_file()
    assert (work / "probe_v2_summary.csv").is_file()
    archive = collect_probe_v2(work)
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()


def test_mv_biopgt_final_exact_parameter_parity_with_v2_zscore():
    from baselines.constants import TOXACUTE_TASKS
    from pgt_v2.mv_prompt.models import count_parameters

    adj = np.eye(len(TOXACUTE_TASKS), dtype=np.float32)
    counts = np.full(len(TOXACUTE_TASKS), 100.0, dtype=np.float32)
    m_final = build_mv_model("MV_BIOPGT_FINAL", tuple(TOXACUTE_TASKS), counts, adjacency=adj)
    m_zscore = build_mv_model(
        "MV_BIOPGT_V2",
        tuple(TOXACUTE_TASKS),
        counts,
        adjacency=adj,
        config={"ordinal_dose": False},
    )
    assert count_parameters(m_final)["total"] == 4505347
    assert count_parameters(m_final) == count_parameters(m_zscore)


def test_formal_b1_reuse_mechanistic_and_summary(tmp_path):
    from pgt_v2.mv_prompt.formal_b1 import (
        _try_import_s0_run,
        collect_formal_b1,
        export_mechanistic,
        summarize_formal_b1,
    )

    mv_cache_path, view_dir, groups = _make_synthetic_bundle(tmp_path)
    store = load_mv_cache(mv_cache_path)
    groups_path = tmp_path / "groups.json"
    write_json(groups_path, groups)

    views_root = tmp_path / "views"
    for fold in ("S0", "S4"):
        for sub in ("FULL", "S21_P10"):
            d = views_root / fold / sub
            d.mkdir(parents=True)
            for f in view_dir.iterdir():
                (d / f.name).write_bytes(f.read_bytes())

    cfg_small = {
        "max_epochs": 2,
        "min_epochs": 1,
        "patience": 2,
        "batch_size": 8,
        "eval_batch": 8,
        "hidden_dim": 32,
        "fp_hidden_dim": 64,
        "graph_layers": 2,
        "graph_heads": 4,
        "gamma": 0.50,
    }
    probe_prev = tmp_path / "probe_prev_runs"
    s0_prev = train_and_evaluate(
        store=store,
        view_root=views_root / "S0" / "FULL",
        endpoint_groups=groups,
        method="MV_BIOPGT_V2",
        seed=42,
        config={**cfg_small, "label_scaling": "per_task", "loss_type": "mse", "ordinal_dose": False},
        output_dir=probe_prev / "FULL_MV_BIOPGT_V2_ZSCORE_s42",
        device="cpu",
        save_checkpoint_file=True,
    )

    work = tmp_path / "b1_work"
    runs_dir = work / "runs"
    reused = _try_import_s0_run(
        src_dir=probe_prev / "FULL_MV_BIOPGT_V2_ZSCORE_s42",
        dst_dir=runs_dir / "S0_FULL_MV_BIOPGT_FINAL_s42",
        expected_view_sha=s0_prev["view_sha256"],
        canonical_method="MV_BIOPGT_FINAL",
    )
    assert reused is True

    s4_sum = train_and_evaluate(
        store=store,
        view_root=views_root / "S4" / "FULL",
        endpoint_groups=groups,
        method="MV_BIOPGT_FINAL",
        seed=42,
        config=cfg_small,
        output_dir=runs_dir / "S4_FULL_MV_BIOPGT_FINAL_s42",
        device="cpu",
        save_checkpoint_file=True,
    )
    assert s4_sum["label_scaling"] == "per_task"
    assert s4_sum["loss_type"] == "mse"
    assert s4_sum["gamma"] == pytest.approx(0.50)

    mech = export_mechanistic(
        mv_cache_path=mv_cache_path,
        views_root=views_root,
        groups_path=groups_path,
        work_dir=work,
        device="cpu",
        seed=42,
        folds=("S0", "S4"),
        probe_runs_dir=probe_prev,
    )
    assert len(mech["interventions"]) == 8  # 2 folds x 4 conditions
    assert (work / "mechanistic" / "prompts_S0.npz").is_file()
    assert (work / "mechanistic" / "prompts_S4.npz").is_file()

    rep = summarize_formal_b1(work_dir=work, groups_path=groups_path)
    assert rep["total_runs"] == 2
    assert (work / "formal_b1_report.md").is_file()
    assert (work / "formal_b1_5fold_summary.csv").is_file()
    archive = collect_formal_b1(work)
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()


def test_leo_zero_shot_strict_no_target_leakage(tmp_path):
    """Verify held-out LEO tasks leak zero training/validation target statistics or weights."""
    from pgt_v2.artifacts import load_arrays, load_manifest, seal_manifest
    from pgt_v2.mv_prompt.formal_b2 import build_leo_view

    mv_cache_path, view_a, groups = _make_synthetic_bundle(tmp_path / "bundle")
    store = load_mv_cache(mv_cache_path)
    held_out = ("rat_oral_LD50", "human_oral_TDLo")
    held_idx = [TASKS.index(t) for t in held_out]

    # Create perturbed view_b where held-out tasks' train & val labels are shifted by +500.0!
    view_b = tmp_path / "view_b_perturbed"
    view_b.mkdir(parents=True)
    m_a = load_manifest(view_a)
    files_b = []
    for fname in m_a["files"]:
        arr = load_arrays(view_a, fname, m_a)
        if fname in ("train.npz", "validation.npz"):
            arr["labels"] = arr["labels"].copy()
            arr["labels"][:, held_idx] += 500.0
        elif fname == "statistics.npz":
            arr["mean"] = arr["mean"].copy()
            arr["scale"] = arr["scale"].copy()
            arr["mean"][held_idx] += 500.0
            arr["scale"][held_idx] *= 50.0
        np.savez_compressed(view_b / fname, **arr)
        files_b.append(fname)
    meta_b = {k: v for k, v in m_a.items() if k not in ("files", "manifest_sha256")}
    seal_manifest(view_b, meta_b, files_b)

    leo_a = tmp_path / "leo_a"
    leo_b = tmp_path / "leo_b"
    ma_leo = build_leo_view(view_a, leo_a, held_out_tasks=held_out)
    mb_leo = build_leo_view(view_b, leo_b, held_out_tasks=held_out)

    sa = load_arrays(leo_a, "statistics.npz", ma_leo)
    sb = load_arrays(leo_b, "statistics.npz", mb_leo)
    assert np.all(sa["count"][held_idx] == 0)
    assert np.array_equal(sa["mean"], sb["mean"])
    assert np.array_equal(sa["scale"], sb["scale"])
    assert np.array_equal(sa["adjacency"], sb["adjacency"])

    cfg = {
        "max_epochs": 2,
        "min_epochs": 1,
        "patience": 2,
        "batch_size": 8,
        "eval_batch": 8,
        "hidden_dim": 32,
        "fp_hidden_dim": 64,
        "graph_layers": 2,
        "graph_heads": 4,
        "gamma": 0.50,
    }
    train_and_evaluate(
        store=store,
        view_root=leo_a,
        endpoint_groups=groups,
        method="MV_BIOPGT_FINAL",
        seed=42,
        config=cfg,
        output_dir=tmp_path / "run_leo_a",
        device="cpu",
        save_checkpoint_file=False,
    )
    train_and_evaluate(
        store=store,
        view_root=leo_b,
        endpoint_groups=groups,
        method="MV_BIOPGT_FINAL",
        seed=42,
        config=cfg,
        output_dir=tmp_path / "run_leo_b",
        device="cpu",
        save_checkpoint_file=False,
    )
    pa = np.load(tmp_path / "run_leo_a" / "test_predictions.npz")["prediction"]
    pb = np.load(tmp_path / "run_leo_b" / "test_predictions.npz")["prediction"]
    assert np.array_equal(pa, pb)


def test_formal_b2_end_to_end_pipeline(tmp_path):
    from pgt_v2.mv_prompt.formal_b2 import (
        collect_formal_b2,
        export_leo_interventions,
        prepare_unified_views_dir,
        summarize_formal_b2,
    )

    mv_cache_path, view_dir, groups = _make_synthetic_bundle(tmp_path / "syn")
    store = load_mv_cache(mv_cache_path)
    groups_path = tmp_path / "groups.json"
    write_json(groups_path, groups)

    views_root = tmp_path / "views"
    for fold in ("S0", "R0"):
        d = views_root / fold / "FULL"
        d.mkdir(parents=True)
        for f in view_dir.iterdir():
            (d / f.name).write_bytes(f.read_bytes())

    work = tmp_path / "b2_work"
    held = ("rat_oral_LD50", "human_oral_TDLo")
    from pgt_v2.mv_prompt import formal_b2 as fb2

    orig_leo = fb2.LEO6_TASKS
    orig_leo5 = fb2.LEO5_ANIMAL_TASKS
    try:
        fb2.LEO6_TASKS = held
        fb2.LEO5_ANIMAL_TASKS = ("rat_oral_LD50",)
        unified = prepare_unified_views_dir(views_root, work, ("S0",))
        cfg_small = {
            "max_epochs": 2,
            "min_epochs": 1,
            "patience": 2,
            "batch_size": 8,
            "eval_batch": 8,
            "hidden_dim": 32,
            "fp_hidden_dim": 64,
            "graph_layers": 2,
            "graph_heads": 4,
            "gamma": 0.50,
        }
        cfg_tox = {"max_epochs": 1, "min_epochs": 1, "patience": 2, "batch_size": 8, "eval_batch": 8}
        for seed in (42, 43):
            train_and_evaluate(
                store=store,
                view_root=unified / "S0" / "FULL",
                endpoint_groups=groups,
                method="MV_BIOPGT_FINAL",
                seed=seed,
                config=cfg_small,
                output_dir=work / "runs" / f"S0_FULL_MV_BIOPGT_FINAL_s{seed}",
                device="cpu",
                save_checkpoint_file=False,
            )
            train_and_evaluate(
                store=store,
                view_root=unified / "S0" / "FULL",
                endpoint_groups=groups,
                method="TOX_AUTHOR",
                seed=seed,
                config=cfg_tox,
                output_dir=work / "runs" / f"S0_FULL_TOX_AUTHOR_s{seed}",
                device="cpu",
                save_checkpoint_file=False,
            )
        for method, cfg in (
            ("MV_BIOPGT_FINAL", cfg_small),
            ("MV_BIOPGT_FINAL_NOPROMPT", cfg_small),
            ("TOX_AUTHOR", cfg_tox),
        ):
            train_and_evaluate(
                store=store,
                view_root=unified / "S0" / "LEO6",
                endpoint_groups=groups,
                method=method,
                seed=42,
                config=cfg,
                output_dir=work / "runs" / f"S0_LEO6_{method}_s42",
                device="cpu",
                save_checkpoint_file=(method == "MV_BIOPGT_FINAL"),
            )
        train_and_evaluate(
            store=store,
            view_root=unified / "R0" / "FULL",
            endpoint_groups=groups,
            method="MV_BIOPGT_FINAL",
            seed=42,
            config=cfg_small,
            output_dir=work / "runs" / "R0_FULL_MV_BIOPGT_FINAL_s42",
            device="cpu",
            save_checkpoint_file=False,
        )
        mech = export_leo_interventions(
            mv_cache_path=mv_cache_path,
            unified_views_root=unified,
            work_dir=work,
            device="cpu",
            seed=42,
            folds=("S0",),
            held_out_tasks=held,
        )
        assert len(mech["leo_interventions"]) == 4
        # Verify shrinkage=0 prompt intervention shifts zero-shot predictions!
        mads = {r["condition"]: r["LEO6_z_shift_mad"] for r in mech["leo_interventions"]}
        assert mads["intact"] == pytest.approx(0.0)
        assert mads["perm_all"] > 0.0

        rep = summarize_formal_b2(
            work_dir=work,
            unified_views_root=unified,
            groups_path=groups_path,
            held_out_tasks=held,
        )
        assert len(rep["multiseed_summary"]) >= 4
        assert len(rep["leo6_summary"]) >= 4
        assert (work / "formal_b2_report.md").is_file()
        archive = collect_formal_b2(work)
        assert archive.is_file()
        assert archive.with_suffix(archive.suffix + ".sha256").is_file()
    finally:
        fb2.LEO6_TASKS = orig_leo
        fb2.LEO5_ANIMAL_TASKS = orig_leo5


