"""Bounded, adapter-driven P1D smoke; never a formal epoch or scientific PASS.

Factories are closures over the parent's verified initialization/assets/config.
``factory(None)`` returns the original trainer; other arms return fresh trainers
with an optional ``.optimization`` OptimizationControl. ``original_step`` must
execute the original numerical path using trainer.model/trainer.optimizer,
including backward and exactly one optimizer.step, and return its scalar loss.
``batch_getter(task, index)`` must return a small TRAIN batch (indices 0..7).
It must not access test/calibration or change the formal engine's sampler.

Only global Python/NumPy/Torch RNG is handled automatically. Private generators,
AMP scalers, trainer counters etc. belong in capture_extra/restore_extra. Those
callbacks must be paired, RNG-neutral, and return weights_only-safe state.
Factories and getters must not train. No engines, datasets or CUDA are started
on import. The parent is responsible for authentic asset/data identity; these
numerical receipts do not prove provenance or grant training authorization.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
from typing import Callable

import numpy as np
import torch

from p1d_optimization import OptimizationControl, OptimizationSpec


SETTINGS = ("ToxAcute", "A", "B")


class SmokeError(ValueError):
    """A smoke invariant failed; dependent work must stop."""


def _require(ok, message):
    if not ok:
        raise SmokeError(message)


@dataclass(frozen=True)
class SmokeAdapter:
    factory: Callable
    original_step: Callable
    batch_getter: Callable
    capture_extra: Callable | None = None
    restore_extra: Callable | None = None

    def __post_init__(self):
        _require(all(callable(x) for x in
                     (self.factory, self.original_step, self.batch_getter)), "adapter callbacks")
        _require((self.capture_extra is None and self.restore_extra is None) or
                 (callable(self.capture_extra) and callable(self.restore_extra)),
                 "extra callbacks must be paired")


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(v) for v in value)
    return copy.deepcopy(value)


def _finite(value, label):
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), f"nonfinite {label}")
    elif isinstance(value, dict):
        for v in value.values():
            _finite(v, label)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _finite(v, label)
    elif isinstance(value, float):
        _require(math.isfinite(value), f"nonfinite {label}")


def _equal(a, b, label):
    if isinstance(a, torch.Tensor):
        ok = (isinstance(b, torch.Tensor) and a.dtype == b.dtype and
              a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
        _require(ok, f"mismatch {label}")
    elif isinstance(a, dict):
        _require(isinstance(b, dict) and a.keys() == b.keys(), f"keys {label}")
        for k in a:
            _equal(a[k], b[k], f"{label}/{k}")
    elif isinstance(a, (list, tuple)):
        _require(type(a) is type(b) and len(a) == len(b), f"length/type {label}")
        for i, (x, y) in enumerate(zip(a, b)):
            _equal(x, y, f"{label}/{i}")
    else:
        _require(type(a) is type(b) and a == b, f"mismatch {label}")


def capture_rng():
    """Serializable global RNG state; do not initialize CUDA just to inspect it."""
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=(n[0], n[1].tolist(), *n[2:]),
                torch=torch.get_rng_state().clone(),
                cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        _require(torch.cuda.is_initialized(), "CUDA RNG restore requires initialized runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def _snapshot(trainer, adapter):
    state = dict(model=_clone(trainer.model.state_dict()),
                 optimizer=_clone(trainer.optimizer.state_dict()),
                 requires_grad={n: p.requires_grad for n, p in trainer.model.named_parameters()},
                 modes={n: m.training for n, m in trainer.model.named_modules()},
                 extra=_clone(adapter.capture_extra(trainer)) if adapter.capture_extra else None,
                 rng=capture_rng())
    _finite(state, "snapshot")
    return state


def _fresh(adapter, arm):
    trainer = adapter.factory(arm)
    _require(hasattr(trainer, "model") and hasattr(trainer, "optimizer"), "trainer model/optimizer")
    control = getattr(trainer, "optimization", None)
    if arm is None:
        _require(control is None, "original trainer already controlled")
    else:
        if control is None:
            control = OptimizationControl(trainer.model, OptimizationSpec(arm))
            trainer.optimization = control
            trainer.optimizer = control.optimizer
        _require(isinstance(control, OptimizationControl) and control.model is trainer.model
                 and control.spec.arm == arm and control.epoch is None
                 and control.optimizer is trainer.optimizer, "fresh control binding")
    opt = trainer.optimizer
    _require(type(opt) is torch.optim.AdamW, "original/controlled AdamW required")
    _require(not opt.state, "factory optimizer must be fresh")
    named = dict(trainer.model.named_parameters())
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    _require(len(ids) == len(set(ids)) == len(named) and
             set(ids) == {id(p) for p in named.values()}, "optimizer must retain every model parameter")
    _require(all(p.requires_grad for p in named.values()), "factory model must be unfrozen")
    _finite(trainer.model.state_dict(), "initial model")
    return trainer


def _named_adam(snapshot, names):
    opt = snapshot["optimizer"]
    ids = [i for g in opt["param_groups"] for i in g["params"]]
    return {n: opt["state"].get(i, {}) for n, i in zip(names, ids)}


def _step(trainer, adapter, task, batch, counter):
    """Observe actual pre/post optimizer calls; forbid a second call before it runs."""
    opt = trainer.optimizer
    names_by_id = {id(p): n for n, p in trainer.model.named_parameters()}
    order = [names_by_id[id(p)] for g in opt.param_groups for p in g["params"]]
    before = _snapshot(trainer, adapter)
    captured = {}
    calls = 0
    start_count = counter[0]

    def pre(optimizer, args, kwargs):
        nonlocal calls
        _require(calls == 0, "more than one optimizer update in callback")
        _require(trainer.optimizer is opt, "optimizer replaced")
        _equal(before["optimizer"], _clone(opt.state_dict()), "Adam continuity before step")
        captured.update({n: _clone(p.grad) for n, p in trainer.model.named_parameters()})
        _finite(captured, "gradient")
        _require(any(g is not None for g in captured.values()), "no gradients at optimizer step")
        calls += 1

    def post(optimizer, args, kwargs):
        counter[0] += 1

    handles = [opt.register_step_pre_hook(pre), opt.register_step_post_hook(post)]
    try:
        loss = adapter.original_step(trainer, task, copy.deepcopy(batch))
    finally:
        for h in handles:
            h.remove()
    _require(calls == 1 and counter[0] == start_count + 1 and trainer.optimizer is opt,
             "exactly one bound optimizer update required")
    if isinstance(loss, torch.Tensor):
        _require(loss.numel() == 1 and loss.is_floating_point(), "scalar loss required")
        loss = float(loss.detach().cpu())
    _require(type(loss) in (float, int) and math.isfinite(loss), "finite scalar loss required")
    after = _snapshot(trainer, adapter)
    old, new = _named_adam(before, order), _named_adam(after, order)
    for n in order:
        if captured[n] is None:
            _equal(old[n], new[n], f"inactive Adam/{n}")
        else:
            _require(set(new[n]) == {"step", "exp_avg", "exp_avg_sq"}, f"Adam fields/{n}")
            previous = float(old[n]["step"]) if old[n] else 0
            _require(float(new[n]["step"]) == previous + 1, f"Adam step continuity/{n}")
    control = getattr(trainer, "optimization", None)
    if control:
        control.verify_frozen()
        control.validate_optimizer_state(opt.state_dict(), control.epoch)
    return dict(task=task, loss=float(loss), gradients=captured, before=before,
                after=after, optimizer_parameter_order=order, observed_updates=counter[0] - start_count,
                optimizer_kind=type(opt).__name__)


def _save(root, name, payload, artifacts, identity):
    path = root / name
    with path.open("xb") as stream:
        torch.save(dict(identity=identity, payload=payload), stream)
    artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()


def _load(root, name):
    return torch.load(root / name, map_location="cpu", weights_only=True)["payload"]


def _compare_step(a, b, label):
    for k in ("task", "loss", "gradients", "before", "after", "optimizer_parameter_order", "optimizer_kind"):
        _equal(a[k], b[k], f"{label}/{k}")


def run_setting_smoke(adapter, *, setting, task_names, output_dir, expected_identity,
                      pair_tasks=None, hf_tasks=None):
    """Run exactly 2 original + 2 B1_high + 6 HF_low + 1 resume updates.

    Task sequences are explicit strings; the HF sequence must exercise every
    endpoint. Batch getter is called only eight times, before any updates.
    Evidence directory must not exist. Failure leaves diagnostic artifacts but
    never a completion receipt. Caller global RNG is restored even on failure.
    """
    _identity(expected_identity)
    expected_identity = copy.deepcopy(expected_identity)
    _require(setting in SETTINGS, "setting")
    _require(type(task_names) in (list, tuple) and len(task_names) ==
             (3 if setting == "ToxAcute" else 5) and
             all(type(t) is str and t for t in task_names) and
             len(set(task_names)) == len(task_names), "task names")
    for key, actual in (("setting", setting), ("task_names", list(task_names))):
        if key in expected_identity:
            _equal(actual, expected_identity[key], "trusted " + key)
    pair = list(pair_tasks) if pair_tasks is not None else [task_names[i % len(task_names)] for i in range(2)]
    hf = list(hf_tasks) if hf_tasks is not None else [task_names[i % len(task_names)] for i in range(6)]
    _require(len(pair) == 2 and all(t in task_names for t in pair) and
             len(hf) == 6 and set(hf) == set(task_names), "step task schedule")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    caller_rng = capture_rng()
    artifacts, counter = {}, [0]
    branch_counts = dict(original=0, B1_high=0, HF_low=0, resume=0)
    try:
        batches = [copy.deepcopy(adapter.batch_getter(t, i)) for i, t in enumerate(pair + hf)]
        original = _fresh(adapter, None)
        initial = _snapshot(original, adapter)
        _check_source(initial, expected_identity)
        _save(root, "initial.pt", initial, artifacts, expected_identity)
        for i, task in enumerate(pair):
            row = _step(original, adapter, task, batches[i], counter)
            branch_counts["original"] += row["observed_updates"]
            _save(root, f"original_{i}.pt", row, artifacts, expected_identity)
        del original, row

        controlled = _fresh(adapter, "B1_high")
        restore_rng(initial["rng"])
        _equal(initial, _snapshot(controlled, adapter), "original/controlled initialization")
        controlled.optimization.begin_epoch(0)
        for i, task in enumerate(pair):
            row = _step(controlled, adapter, task, batches[i], counter)
            branch_counts["B1_high"] += row["observed_updates"]
            _compare_step(_load(root, f"original_{i}.pt"), row, f"B1 pair {i}")
            _save(root, f"B1_high_{i}.pt", row, artifacts, expected_identity)
        del controlled, row

        trainer = _fresh(adapter, "HF_low")
        _equal(initial["model"], _clone(trainer.model.state_dict()), "HF initialization")
        restore_rng(initial["rng"])
        control = trainer.optimization
        for epoch, task in enumerate(hf):
            control.begin_epoch(epoch)
            row = _step(trainer, adapter, task, batches[2 + epoch], counter)
            branch_counts["HF_low"] += row["observed_updates"]
            if epoch == 5:
                _require(any(not torch.equal(row["before"]["model"][n], row["after"]["model"][n])
                             for n, _ in control.backbone), "backbone did not change at epoch5")
            _save(root, f"HF_low_{epoch}.pt", row, artifacts, expected_identity)
            if epoch == 4:
                checkpoint = dict(epoch=4, arm="HF_low", setting=setting,
                                  task_names=list(task_names), state=row["after"])
                _save(root, "epoch4_snapshot.pt", checkpoint, artifacts, expected_identity)
        # Require that all heads were exercised, including inactive heads whose
        # existing Adam moments must survive updates to other tasks.
        _require(all(control.optimizer.state.get(p) for _, p in control.heads), "head coverage")
        del trainer, control, row

        resumed = _fresh(adapter, "HF_low")
        checkpoint = _load(root, "epoch4_snapshot.pt")
        state = checkpoint["state"]
        control = resumed.optimization
        control.validate_optimizer_state(state["optimizer"], 4, state["model"])
        resumed.model.load_state_dict(state["model"], strict=True)
        resumed.optimizer.load_state_dict(state["optimizer"])
        for n, p in resumed.model.named_parameters():
            p.requires_grad_(state["requires_grad"][n])
            p.grad = None
        for n, m in resumed.model.named_modules():
            m.training = state["modes"][n]
        if adapter.restore_extra:
            adapter.restore_extra(resumed, copy.deepcopy(state["extra"]))
        control.epoch = 4
        restore_rng(state["rng"])
        control.verify_frozen()
        _equal(state, _snapshot(resumed, adapter), "restored epoch4")
        control.begin_epoch(5)
        row = _step(resumed, adapter, hf[5], batches[7], counter)
        branch_counts["resume"] += row["observed_updates"]
        _compare_step(_load(root, "HF_low_5.pt"), row, "resume epoch5")
        _save(root, "resumed_epoch5.pt", row, artifacts, expected_identity)
        _require(counter[0] == 11, "11 observed updates required")
        receipt = dict(schema="p1d_smoke_v1", identity=expected_identity, setting=setting, task_names=list(task_names),
                       pair_tasks=pair, hf_tasks=hf, split="train", simulated_epochs=True,
                       observed_optimizer_updates=counter[0],
                       updates_by_branch=branch_counts,
                       comparison="exact", artifacts=artifacts,
                       scope="SMOKE_ONLY_NOT_FORMAL_TRAINING", acceptance_status="PENDING_REVIEW")
        with (root / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return receipt
    finally:
        restore_rng(caller_rng)


def run_controlled_smoke(*, adapters, task_names, output_dir, expected_identity):
    """Fixed three-setting wrapper, 33 updates; no retries or extra branches."""
    _require(set(adapters) == set(task_names) == set(expected_identity) == set(SETTINGS), "three settings required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    results = {s: run_setting_smoke(adapters[s], setting=s, task_names=task_names[s],
                                   output_dir=root / s, expected_identity=expected_identity[s]) for s in SETTINGS}
    total = sum(r["observed_optimizer_updates"] for r in results.values())
    _require(total == 33, "33 observed updates required")
    receipt = dict(schema="p1d_smoke_suite_v1", observed_optimizer_updates=total, settings=results)
    with (root / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


def _identity(value):
    _require(type(value) is dict and bool(value), "nonempty expected_identity required")
    def visit(v):
        _require(type(v) in (dict, list, str, int, float, bool, type(None)), "JSON identity types")
        if type(v) is dict:
            _require(all(type(k) is str for k in v), "identity keys")
            for x in v.values():
                visit(x)
        elif type(v) is list:
            for x in v:
                visit(x)
    visit(value)
    _finite(value, "identity")


def model_digest(model_state):
    """Canonical tensor digest the parent may bind as initial_model_sha256."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model_state.items()):
        _require(isinstance(tensor, torch.Tensor), "tensor-only model state required")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(header.encode("utf-8") + b"\0" + len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def _check_source(state, identity):
    if "initial_model_sha256" in identity:
        _require(model_digest(state["model"]) == identity["initial_model_sha256"], "source tensor identity")


def _keys(value, keys, label):
    _require(type(value) is dict and set(value) == set(keys.split()), f"schema {label}")


def _verify_snapshot(state, order):
    _keys(state, "model optimizer requires_grad modes extra rng", "snapshot")
    _finite(state, "saved snapshot")
    _require(type(order) is list and len(order) == len(set(order)) and
             set(order) == set(state["requires_grad"]), "parameter order")
    _require(all(n.startswith(("encoder.backbone.", "decoders.")) for n in order)
             and any(n.startswith("encoder.backbone.") for n in order)
             and any(n.startswith("decoders.") for n in order), "parameter scopes")
    _require(all(type(v) is bool for v in state["requires_grad"].values()) and
             state["modes"] and all(type(v) is bool for v in state["modes"].values()), "model flags")
    _require(all(isinstance(v, torch.Tensor) for v in state["model"].values()) and
             set(order).issubset(state["model"]), "model tensors")
    opt = state["optimizer"]
    _keys(opt, "state param_groups", "optimizer")
    ids = [i for g in opt["param_groups"] for i in g["params"]]
    _require(all(type(i) is int for i in ids) and len(ids) == len(set(ids)) == len(order)
             and set(opt["state"]).issubset(ids), "optimizer parameter IDs")
    for n, item in _named_adam(state, order).items():
        if not item:
            continue
        _keys(item, "step exp_avg exp_avg_sq", "Adam")
        step = item["step"]
        _require(isinstance(step, torch.Tensor) and step.numel() == 1 and
                 float(step) > 0 and float(step).is_integer(), "Adam step")
        for key in ("exp_avg", "exp_avg_sq"):
            t, p = item[key], state["model"][n]
            _require(isinstance(t, torch.Tensor) and t.shape == p.shape and t.dtype == p.dtype,
                     "Adam moment shape/dtype")
        _require(bool((item["exp_avg_sq"] >= 0).all()), "negative Adam variance")
    rng = state["rng"]
    _keys(rng, "python numpy torch cuda", "RNG")
    _require(type(rng["python"]) is tuple and len(rng["python"]) == 3 and
             type(rng["numpy"]) is tuple and len(rng["numpy"]) == 5 and
             rng["numpy"][0] == "MT19937" and len(rng["numpy"][1]) == 624 and
             isinstance(rng["torch"], torch.Tensor) and rng["torch"].dtype == torch.uint8
             and rng["torch"].ndim == 1 and type(rng["cuda"]) is list and
             all(isinstance(t, torch.Tensor) and t.dtype == torch.uint8 and t.ndim == 1
                 for t in rng["cuda"]), "RNG state schema")
    _require(all(type(x) is int and 0 <= x <= 2**32 - 1 for x in rng["numpy"][1]) and
             type(rng["numpy"][2]) is int and 0 <= rng["numpy"][2] <= 624 and
             type(rng["numpy"][3]) is int and rng["numpy"][3] in (0, 1), "NumPy RNG values")
    # Validate syntax with private, never-sampled generators; no global RNG write.
    random.Random(0).setstate(rng["python"])
    n = rng["numpy"]
    np.random.RandomState(0).set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.Generator(device="cpu").set_state(rng["torch"])


def _near(actual, expected, label):
    # Only independent Adam algebra uses a tolerance (operation ordering/fused
    # kernels). All paired trajectories, frozen tensors and resume use equality.
    _require(actual.dtype == expected.dtype and actual.shape == expected.shape and
             torch.allclose(actual, expected, rtol=2e-5, atol=1e-8), f"Adam arithmetic {label}")


def _verify_row(row, *, task, arm, epoch):
    _keys(row, "task loss gradients before after optimizer_parameter_order observed_updates optimizer_kind", "step")
    _require(row["task"] == task and type(row["loss"]) is float and math.isfinite(row["loss"]), "task/loss")
    _require(type(row["observed_updates"]) is int and row["observed_updates"] == 1,
             "observed update count")
    _require(row["optimizer_kind"] == "AdamW", "original/controlled AdamW required")
    order = row["optimizer_parameter_order"]
    a, b, gradients = row["before"], row["after"], row["gradients"]
    _verify_snapshot(a, order)
    _verify_snapshot(b, order)
    _equal(a["requires_grad"], b["requires_grad"], "step requires_grad")
    _equal(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"], "group continuity")
    groups = a["optimizer"]["param_groups"]
    _require(len(groups) == (2 if arm == "HF_low" else 1), "group count")
    _require(set(gradients) == set(order) and any(v is not None for v in gradients.values()), "gradient coverage")
    _finite(gradients, "saved gradient")
    old, new = _named_adam(a, order), _named_adam(b, order)
    cursor = 0
    for gi, group in enumerate(groups):
        _require(set(group).issubset({"params", "lr", "betas", "eps", "weight_decay", "amsgrad",
                                     "maximize", "foreach", "capturable", "differentiable", "fused",
                                     "decoupled_weight_decay"}), "unknown Adam group field")
        lr = .0001 if arm == "HF_low" and gi == 0 else .001
        _require(group["lr"] == lr and group["weight_decay"] == 1e-5 and
                 group["betas"] == (.9, .999) and group["eps"] == 1e-8 and
                 not group.get("amsgrad") and not group.get("maximize"), "Adam group contract")
        for n in order[cursor:cursor + len(group["params"])]:
            backbone = n.startswith("encoder.backbone.")
            if arm == "HF_low":
                _require(backbone == (gi == 0), "backbone/head group order")
            frozen = arm == "HF_low" and epoch < 5 and backbone
            _require(a["requires_grad"][n] is (not frozen), "freeze flags")
            grad = gradients[n]
            if frozen:
                _require(grad is None and not old[n] and not new[n], "frozen gradient/Adam state")
            if grad is None:
                _equal(old[n], new[n], f"inactive Adam/{n}")
                _equal(a["model"][n], b["model"][n], f"inactive tensor/{n}")
                continue
            p = a["model"][n]
            _require(isinstance(grad, torch.Tensor) and grad.shape == p.shape and grad.dtype == p.dtype,
                     "gradient shape/dtype")
            _require(bool(new[n]), "missing active Adam state")
            step = float(old[n]["step"]) if old[n] else 0
            _require(float(new[n]["step"]) == step + 1, "Adam update counter")
            m0 = old[n]["exp_avg"] if old[n] else torch.zeros_like(p)
            v0 = old[n]["exp_avg_sq"] if old[n] else torch.zeros_like(p)
            beta1, beta2 = group["betas"]
            m = m0 * beta1 + grad * (1 - beta1)
            v = v0 * beta2 + grad.square() * (1 - beta2)
            _near(new[n]["exp_avg"], m, n + "/moment")
            _near(new[n]["exp_avg_sq"], v, n + "/variance")
            expected = p * (1 - lr * group["weight_decay"]) - (lr / (1 - beta1 ** (step + 1))) * m / (
                v.sqrt() / math.sqrt(1 - beta2 ** (step + 1)) + group["eps"])
            _near(b["model"][n], expected, n + "/parameter")
        cursor += len(group["params"])
    if arm == "HF_low" and epoch == 5:
        _require(any(not torch.equal(a["model"][n], b["model"][n]) for n in order
                     if n.startswith("encoder.backbone.")), "no epoch5 backbone change")


def _read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            _require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=pairs)
    _finite(value, "JSON")
    return value


def verify_smoke(output, expected_identity):
    """Read-only verification, including Adam algebra; never performs updates.

    For a suite, expected_identity is a mapping of the three settings to their
    identities. Hashes detect file changes, not malicious coordinated rewriting.
    Bind initial_model_sha256 in the trusted identity to independently anchor
    source tensors. Actual batch provenance/loss recomputation needs the parent's
    trusted real-data adapter; a saved scalar alone cannot establish that fact.
    """
    try:
        return _verify_smoke(Path(output), expected_identity)
    except SmokeError:
        raise
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, EOFError, IndexError,
            AttributeError, OverflowError, pickle.UnpicklingError) as exc:
        raise SmokeError(f"invalid smoke evidence: {exc}") from exc


def _verify_smoke(root, expected_identity):
    _identity(expected_identity)
    receipt = _read_json(root / "receipt.json")
    if receipt.get("schema") == "p1d_smoke_suite_v1":
        _keys(receipt, "schema observed_optimizer_updates settings", "suite receipt")
        _require(set(receipt["settings"]) == set(expected_identity) == set(SETTINGS), "suite settings")
        for setting in SETTINGS:
            verify_smoke(root / setting, expected_identity[setting])
            _equal(receipt["settings"][setting], _read_json(root / setting / "receipt.json"), "suite child")
        _require(type(receipt["observed_optimizer_updates"]) is int and
                 receipt["observed_optimizer_updates"] == 33, "suite count")
        return dict(verified=True, observed_optimizer_updates=33, settings=list(SETTINGS))
    _keys(receipt, "schema identity setting task_names pair_tasks hf_tasks split simulated_epochs "
          "observed_optimizer_updates updates_by_branch comparison artifacts scope acceptance_status", "receipt")
    _equal(receipt["identity"], expected_identity, "trusted identity")
    _require(receipt["schema"] == "p1d_smoke_v1" and receipt["setting"] in SETTINGS and
             receipt["split"] == "train" and receipt["simulated_epochs"] is True and
             receipt["comparison"] == "exact" and receipt["scope"] == "SMOKE_ONLY_NOT_FORMAL_TRAINING"
             and receipt["acceptance_status"] == "PENDING_REVIEW", "receipt contract")
    tasks, pair, hf = (receipt[k] for k in ("task_names", "pair_tasks", "hf_tasks"))
    _require(type(tasks) is list and len(tasks) == (3 if receipt["setting"] == "ToxAcute" else 5)
             and all(type(t) is str and t for t in tasks) and len(set(tasks)) == len(tasks)
             and type(pair) is list and len(pair) == 2 and all(t in tasks for t in pair)
             and type(hf) is list and len(hf) == 6 and set(hf) == set(tasks), "saved task schedule")
    for key in ("setting", "task_names"):
        if key in expected_identity:
            _equal(receipt[key], expected_identity[key], "trusted " + key)
    names = (["initial.pt", "epoch4_snapshot.pt", "resumed_epoch5.pt"] +
             [f"{branch}_{i}.pt" for branch, count in (("original", 2), ("B1_high", 2), ("HF_low", 6))
              for i in range(count)])
    _require(set(receipt["artifacts"]) == set(names), "artifact inventory")
    def load(name):
        path = root / name
        _require(not path.is_symlink() and path.is_file(), "missing/linked artifact")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == receipt["artifacts"][name], "artifact SHA")
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        _keys(envelope, "identity payload", "artifact")
        _equal(envelope["identity"], expected_identity, "artifact identity")
        return envelope["payload"]
    initial = load("initial.pt")
    _check_source(initial, expected_identity)
    _require(not initial["optimizer"]["state"] and
             all(v is True for v in initial["requires_grad"].values()), "fresh initial state")
    counts = {}
    for branch, schedule in (("original", pair), ("B1_high", pair), ("HF_low", hf)):
        previous = None
        counts[branch] = 0
        for i, task in enumerate(schedule):
            row = load(f"{branch}_{i}.pt")
            _verify_row(row, task=task, arm=branch, epoch=i if branch == "HF_low" else 0)
            if i == 0:
                if branch != "HF_low":
                    _equal(initial, row["before"], "initial B1 state")
                else:
                    _equal(initial["model"], row["before"]["model"], "HF source")
                    _equal(initial["rng"], row["before"]["rng"], "HF initial RNG")
                    _require(not row["before"]["optimizer"]["state"], "HF initial optimizer")
            else:
                expected = _clone(previous)
                if branch == "HF_low" and i == 5:
                    expected["requires_grad"] = {n: True for n in expected["requires_grad"]}
                _equal(expected, row["before"], "trajectory continuity")
            if branch == "B1_high":
                _compare_step(load(f"original_{i}.pt"), row, f"saved pair {i}")
            if branch == "HF_low" and i < 5:
                for n in row["optimizer_parameter_order"]:
                    if n.startswith("encoder.backbone."):
                        _equal(initial["model"][n], row["after"]["model"][n], "frozen source")
            previous = row["after"]
            counts[branch] += row["observed_updates"]
        if branch == "HF_low":
            heads = _named_adam(previous, row["optimizer_parameter_order"])
            _require(all(v for n, v in heads.items() if n.startswith("decoders.")), "saved head coverage")
    checkpoint = load("epoch4_snapshot.pt")
    _keys(checkpoint, "epoch arm setting task_names state", "epoch4 checkpoint")
    _require(type(checkpoint["epoch"]) is int and checkpoint["epoch"] == 4 and
             checkpoint["arm"] == "HF_low" and checkpoint["setting"] == receipt["setting"], "checkpoint identity")
    _equal(checkpoint["task_names"], tasks, "checkpoint tasks")
    _equal(checkpoint["state"], load("HF_low_4.pt")["after"], "epoch4 saved state")
    resumed = load("resumed_epoch5.pt")
    _verify_row(resumed, task=hf[5], arm="HF_low", epoch=5)
    _compare_step(load("HF_low_5.pt"), resumed, "saved resume")
    counts["resume"] = resumed["observed_updates"]
    _equal(receipt["updates_by_branch"], counts, "branch counters")
    _require(type(receipt["observed_optimizer_updates"]) is int and
             receipt["observed_optimizer_updates"] == sum(counts.values()) == 11, "total counter")
    return dict(verified=True, setting=receipt["setting"], observed_optimizer_updates=11,
                identity=copy.deepcopy(expected_identity), acceptance_status="PENDING_REVIEW")


def main(argv=None):
    """Verification-only CLI. Execution requires explicit parent adapters."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("output", type=Path)
    verify.add_argument("--expected-identity", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_smoke(args.output, _read_json(args.expected_identity))
    except SmokeError as exc:
        parser.exit(1, f"smoke verification failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
