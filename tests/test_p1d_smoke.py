"""Tiny CPU-only smoke contracts; no real engines, data, assets or training."""
import copy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from p1d_optimization import OptimizationControl, OptimizationSpec
from p1d_smoke import (SmokeAdapter, SmokeError, capture_rng, main, model_digest,
                       run_controlled_smoke, run_setting_smoke, verify_smoke, _equal)


def adapter_for(setting="ToxAcute", *, attached=True, fault=None):
    tasks = [f"task{i}" for i in range(3 if setting == "ToxAcute" else 5)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        template = nn.Module()
        template.encoder = nn.Module()
        template.encoder.backbone = nn.Sequential(nn.Linear(2, 3), nn.Dropout(.2))
        template.decoders = nn.ModuleDict({t: nn.Linear(3, 1) for t in tasks})
    identity = dict(setting=setting, task_names=tasks, source_sha256="synthetic-source",
                    train_sha256="synthetic-train", initial_model_sha256=model_digest(template.state_dict()))
    log = dict(arms=[], updates=0, batches=[])

    def factory(arm):
        log["arms"].append(arm)
        model = copy.deepcopy(template)
        trainer = SimpleNamespace(model=model, arm=arm, ticks=0,
                                  generator=torch.Generator().manual_seed(94), optimization=None)
        trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
        if arm and attached:
            trainer.optimization = OptimizationControl(model, OptimizationSpec(arm))
            trainer.optimizer = trainer.optimization.optimizer
        if fault == "wrong_init" and arm == "B1_high":
            with torch.no_grad():
                next(model.parameters()).add_(1)
        return trainer

    def step(trainer, task, batch):
        opt = trainer.optimizer
        opt.zero_grad(set_to_none=True)
        # Exercise global Python, NumPy, Torch and private generator restoration.
        x, target = batch
        noise = random.random() + float(np.random.random()) + float(torch.rand((), generator=trainer.generator))
        prediction = trainer.model.decoders[task](trainer.model.encoder.backbone(x))
        loss = (prediction - target - noise * .01).square().mean()
        if fault == "nan":
            loss = loss * float("nan")
        loss.backward()
        if fault == "no_update":
            return loss
        if fault == "reset_heads" and trainer.arm == "HF_low" and trainer.ticks == 3:
            opt.state.clear()
        if fault == "frozen_grad" and trainer.arm == "HF_low":
            next(trainer.model.encoder.backbone.parameters()).grad = torch.ones_like(next(trainer.model.parameters()))
        backbone_before = copy.deepcopy(trainer.model.encoder.backbone.state_dict())
        opt.step()
        log["updates"] += 1
        if fault == "no_unfreeze_change" and trainer.arm == "HF_low" and trainer.ticks == 5:
            trainer.model.encoder.backbone.load_state_dict(backbone_before)
        if fault == "double_update":
            opt.step()
            log["updates"] += 1
        # In-place mutation must not contaminate the cached paired batch.
        x.add_(2)
        trainer.ticks += 1
        return loss.detach()

    def batch_getter(task, index):
        log["batches"].append((task, index))
        return torch.tensor([[.2, .4], [.5, -.3]]) + index * .01, torch.ones(2, 1)

    def capture_extra(trainer):
        return dict(ticks=trainer.ticks, generator=trainer.generator.get_state())

    def restore_extra(trainer, state):
        trainer.ticks = state["ticks"]
        if fault != "bad_rng_restore":
            trainer.generator.set_state(state["generator"])

    return SmokeAdapter(factory, step, batch_getter, capture_extra, restore_extra), tasks, identity, log


def run_tiny(root, **kwargs):
    adapter, tasks, identity, log = adapter_for(**kwargs)
    receipt = run_setting_smoke(adapter, setting=identity["setting"], task_names=tasks,
                                expected_identity=identity, output_dir=root)
    return receipt, identity, log


@pytest.mark.parametrize("attached", [True, False])
def test_exact_eleven_and_independent_readonly_verifier(tmp_path, monkeypatch, attached):
    root = tmp_path / "smoke"
    rng = capture_rng()
    receipt, identity, log = run_tiny(root, attached=attached)
    _equal(rng, capture_rng(), "caller RNG")
    assert log["arms"] == [None, "B1_high", "HF_low", "HF_low"]
    assert log["updates"] == receipt["observed_optimizer_updates"] == 11
    assert len(log["batches"]) == 8
    before = {p.name: p.read_bytes() for p in root.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("verifier may not optimize or restore global RNG")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch, "set_rng_state", forbidden)
    result = verify_smoke(root, identity)
    assert result["observed_optimizer_updates"] == 11
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    _equal(rng, capture_rng(), "verifier RNG")
    checkpoint = torch.load(root / "epoch4_snapshot.pt", weights_only=True)["payload"]["state"]
    assert len(checkpoint["optimizer"]["state"]) == 6  # all three heads, zero backbone
    assert all(not value for name, value in checkpoint["requires_grad"].items()
               if name.startswith("encoder.backbone."))


def test_three_routes_exact_33(tmp_path):
    adapters, tasks, identities, logs = {}, {}, {}, {}
    for setting in ("ToxAcute", "A", "B"):
        adapters[setting], tasks[setting], identities[setting], logs[setting] = adapter_for(setting)
    receipt = run_controlled_smoke(adapters=adapters, task_names=tasks, expected_identity=identities,
                                   output_dir=tmp_path / "suite")
    assert receipt["observed_optimizer_updates"] == sum(x["updates"] for x in logs.values()) == 33
    assert verify_smoke(tmp_path / "suite", identities)["observed_optimizer_updates"] == 33


@pytest.mark.parametrize("fault,match", [
    ("no_update", "exactly one"), ("double_update", "more than one"),
    ("nan", "nonfinite"), ("reset_heads", "continuity"),
    ("wrong_init", "initialization"), ("frozen_grad", "frozen"),
    ("bad_rng_restore", "restored epoch4"),
    ("no_unfreeze_change", "did not change"),
])
def test_failing_adapters_no_receipt_or_extra_updates(tmp_path, fault, match):
    adapter, tasks, identity, log = adapter_for(fault=fault)
    root = tmp_path / "failed"
    rng = capture_rng()
    with pytest.raises(ValueError, match=match):
        run_setting_smoke(adapter, setting="ToxAcute", task_names=tasks,
                          expected_identity=identity, output_dir=root)
    assert not (root / "receipt.json").exists()
    assert log["updates"] <= 10
    if fault == "double_update":
        assert log["updates"] == 1
    _equal(rng, capture_rng(), "failure RNG")


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "evidence"
    _, identity, _ = run_tiny(root)
    return root, identity


def rewrite(root, filename, mutate):
    """Update SHA too: validator must check content beyond the hash receipt."""
    path = root / filename
    payload = torch.load(path, weights_only=True)
    mutate(payload["payload"])
    torch.save(payload, path)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["artifacts"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


@pytest.mark.parametrize("fault", ["count", "bool_count", "pass", "split", "missing_inventory", "unknown", "identity"])
def test_tampered_receipt_rejected(evidence, fault):
    root, identity = evidence
    path = root / "receipt.json"
    r = json.loads(path.read_text())
    if fault == "count": r["observed_optimizer_updates"] = 12
    if fault == "bool_count": r["updates_by_branch"]["resume"] = True
    if fault == "pass": r["acceptance_status"] = "PASS"
    if fault == "split": r["split"] = "test"
    if fault == "missing_inventory": del r["artifacts"]["HF_low_0.pt"]
    if fault == "unknown": r["PASS"] = True
    if fault == "identity": r["identity"]["source_sha256"] = "wrong"
    path.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(SmokeError): verify_smoke(root, identity)


@pytest.mark.parametrize("fault", ["frozen", "grad", "adam", "step", "resume_rng", "checkpoint", "source", "nan_loss"])
def test_self_consistent_hash_does_not_hide_bad_states(evidence, fault):
    root, identity = evidence
    filename = "HF_low_2.pt"
    def mutate(row):
        backbone = next(n for n in row["gradients"] if n.startswith("encoder.backbone."))
        if fault == "frozen": row["after"]["model"][backbone].add_(.1)
        if fault == "grad": row["gradients"][backbone] = torch.zeros_like(row["before"]["model"][backbone])
        if fault in ("adam", "step"):
            item = next(iter(row["after"]["optimizer"]["state"].values()))
            item["exp_avg" if fault == "adam" else "step"].add_(1)
        if fault == "resume_rng": row["after"]["rng"]["torch"][0] ^= 1
        if fault == "nan_loss": row["loss"] = float("nan")
    if fault == "resume_rng": filename = "resumed_epoch5.pt"
    if fault == "checkpoint":
        filename = "epoch4_snapshot.pt"
        def mutate(row): row["state"]["rng"]["torch"][0] ^= 1
    if fault == "source":
        filename = "initial.pt"
        def mutate(row): next(iter(row["model"].values())).add_(.5)
    rewrite(root, filename, mutate)
    with pytest.raises(SmokeError): verify_smoke(root, identity)


def test_missing_and_wrong_expected_identity(evidence):
    root, identity = evidence
    wrong = dict(identity, train_sha256="other")
    with pytest.raises(SmokeError, match="identity"): verify_smoke(root, wrong)
    (root / "epoch4_snapshot.pt").unlink()
    with pytest.raises(SmokeError, match="missing"): verify_smoke(root, identity)


def test_cli_is_verification_only(evidence, tmp_path, monkeypatch, capsys):
    root, identity = evidence
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity), encoding="utf-8")
    def forbidden(*args, **kwargs): raise AssertionError("unexpected optimizer update")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    assert main(["verify", str(root), "--expected-identity", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["observed_optimizer_updates"] == 11


def test_invalid_schedule_stops_before_factory(tmp_path):
    adapter, tasks, identity, log = adapter_for()
    with pytest.raises(SmokeError, match="schedule"):
        run_setting_smoke(adapter, setting="ToxAcute", task_names=tasks, expected_identity=identity,
                          output_dir=tmp_path / "bad", hf_tasks=[tasks[0]] * 6)
    assert not log["arms"] and log["updates"] == 0


def test_output_directory_never_overwritten(evidence):
    root, identity = evidence
    adapter, tasks, _, log = adapter_for()
    with pytest.raises(FileExistsError):
        run_setting_smoke(adapter, setting="ToxAcute", task_names=tasks, expected_identity=identity, output_dir=root)
    assert not log["arms"]


def test_wrong_trusted_source_fails_before_update(tmp_path):
    adapter, tasks, identity, log = adapter_for()
    identity["initial_model_sha256"] = "0" * 64
    with pytest.raises(SmokeError, match="source tensor identity"):
        run_setting_smoke(adapter, setting="ToxAcute", task_names=tasks, expected_identity=identity,
                          output_dir=tmp_path / "bad-source")
    assert log["updates"] == 0


def test_corrupt_weights_only_artifact_is_closed(evidence):
    root, identity = evidence
    path = root / "epoch4_snapshot.pt"
    path.write_bytes(b"not a torch checkpoint")
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SmokeError): verify_smoke(root, identity)


def test_invalid_rng_shape_rejected_even_with_new_sha(evidence):
    root, identity = evidence
    def mutate(row): row["after"]["rng"]["torch"] = torch.zeros(2, dtype=torch.uint8)
    rewrite(root, "HF_low_2.pt", mutate)
    with pytest.raises(SmokeError): verify_smoke(root, identity)
