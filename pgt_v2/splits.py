"""C01 registered structure grouping and presence-only deterministic allocation.

C02 exercises small fixtures only. G01 materializes the real structures/folds;
C04 chooses eligible endpoints and locks the protocol without model outcomes.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from .chemistry import _Union, framed_digest, prepare_molecule


def structure_groups(inputs, *, reviewed_parent_links=(), parent_rule=None):
    """inputs maps record_id to MolecularInput; source IDs are not accepted."""
    records = sorted(inputs)
    if not records:
        raise ValueError("No molecular inputs")
    identities = sorted({inputs[r].model_input_id for r in records})
    positions = {identity: i for i, identity in enumerate(identities)}
    by_identity = {inputs[r].model_input_id: inputs[r] for r in records}
    union = _Union(len(identities))
    link_keys = {}
    acyclic = []
    for identity in identities:
        item = by_identity[identity]
        keys = ["constitutional:" + item.nonstereo_id]
        mol = Chem.Mol(item.mol)
        Chem.RemoveStereochemistry(mol)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms():
            keys.append("murcko:" + prepare_molecule(scaffold).nonstereo_id)
        else:
            acyclic.append(identity)
        for key in keys:
            if key in link_keys:
                union.join(positions[identity], positions[link_keys[key]])
            else:
                link_keys[key] = identity
    if acyclic:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
        fingerprints = [generator.GetFingerprint(by_identity[c].mol) for c in acyclic]
        distances = []
        for i in range(1, len(acyclic)):
            distances.extend(1.0 - x for x in DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i]))
        clusters = Butina.ClusterData(distances, len(acyclic), 0.2, isDistData=True, reordering=False)
        for cluster in clusters:
            for index in cluster[1:]:
                union.join(positions[acyclic[cluster[0]]], positions[acyclic[index]])
    if reviewed_parent_links and not parent_rule:
        raise ValueError("Parent-family links require a written reviewed chemical rule")
    for left, right in reviewed_parent_links:
        if left not in positions or right not in positions:
            raise ValueError("Unknown verified identity in parent link")
        union.join(positions[left], positions[right])
    members = {}
    for identity in identities:
        members.setdefault(union.find(positions[identity]), []).append(identity)
    group_ids = {root: "group_" + framed_digest(sorted(values)) for root, values in members.items()}
    lookup = {c: group_ids[union.find(positions[c])] for c in identities}
    return {r: lookup[inputs[r].model_input_id] for r in records}


def allocate_groups(group_ids, observed, *, fractions, minimum_groups, seed=20261016, max_passes=20):
    """Greedy + bounded whole-group repair of the exact registered objectives."""
    observed = np.asarray(observed)
    if observed.dtype != np.bool_ or observed.ndim != 2 or observed.shape[0] != len(group_ids):
        raise ValueError("Allocation accepts a boolean presence matrix, never label values")
    if not group_ids or not all(group_ids) or observed.shape[1] == 0:
        raise ValueError("Empty group or task registry")
    fractions = np.asarray(fractions, dtype=np.float64)
    minimums = np.asarray(minimum_groups)
    if (fractions.ndim != 1 or len(fractions) < 2 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1) or
            minimums.shape != fractions.shape or not np.issubdtype(minimums.dtype, np.integer) or np.any(minimums < 0)):
        raise ValueError("Invalid destination fractions/minimums")
    if not 0 <= max_passes <= 20:
        raise ValueError("Repair is bounded to at most 20 passes")
    unique = sorted(set(group_ids))
    # Aggregate once. No quadratic group-by-row scans on the real data.
    index = {g: i for i, g in enumerate(unique)}
    sizes = np.zeros(len(unique), dtype=np.int64)
    presence = np.zeros((len(unique), observed.shape[1]), dtype=np.bool_)
    for group, row in zip(group_ids, observed):
        i = index[group]; sizes[i] += 1; presence[i] |= row
    ordered = sorted(range(len(unique)), key=lambda i: (-int(sizes[i]), framed_digest([seed, unique[i]]), unique[i]))
    destinations = len(fractions)
    counts = np.zeros(destinations, dtype=np.int64)
    positives = np.zeros((destinations, observed.shape[1]), dtype=np.int64)
    total_groups = presence.sum(axis=0)
    normalizer = np.maximum(total_groups, 1)
    total_records = sizes.sum()
    def objective():
        deficit = int(np.maximum(0, minimums[:, None] - positives).sum())
        record_error = np.mean((counts / total_records - fractions) ** 2)
        task_error = np.mean((positives / normalizer[None, :] - fractions[:, None]) ** 2)
        return deficit, float(record_error + task_error)
    def change(i, target, sign):
        counts[target] += sign * sizes[i]
        positives[target] += sign * presence[i]
    assignments = {}
    for i in ordered:
        candidates = []
        for target in range(destinations):
            change(i, target, 1); candidates.append((objective(), target)); change(i, target, -1)
        _, best = min(candidates)
        assignments[i] = best
        change(i, best, 1)
    passes = 0
    moves = 0
    for _ in range(max_passes):
        passes += 1
        moved = False
        for i in ordered:
            old = assignments[i]
            before = objective()
            change(i, old, -1)
            candidates = []
            for target in range(destinations):
                change(i, target, 1); candidates.append((objective(), target)); change(i, target, -1)
            score, best = min(candidates)
            if score < before:
                assignments[i] = best; moves += 1; moved = True
            change(i, assignments[i], 1)
        if not moved:
            break
    return {unique[i]: destination for i, destination in assignments.items()}, {
        "objective": list(objective()), "record_counts": counts.tolist(),
        "positive_group_counts": positives.tolist(), "minimum_groups": minimums.tolist(),
        "target_fractions": fractions.tolist(), "realized_fractions": (counts / total_records).tolist(),
        "repair_passes": passes, "moves": moves, "seed": seed,
        "coverage_pass": bool(np.all(positives >= minimums[:, None]))}


def plan_folds(group_ids, observed, *, seed=20261016):
    """Try K=5 once, then K=3 once on coverage failure; retain both reports."""
    observed = np.asarray(observed)
    attempts = []
    for k in (5, 3):
        outer, outer_report = allocate_groups(group_ids, observed, fractions=[1 / k] * k, minimum_groups=[2] * k, seed=seed)
        folds = []
        for test_fold in range(k):
            remain = np.asarray([outer[g] != test_fold for g in group_ids])
            if not remain.any():
                folds.append({"fold_id": test_fold, "roles": None, "coverage_pass": False, "reason": "no remaining groups"})
                continue
            inner, inner_report = allocate_groups([g for g, keep in zip(group_ids, remain) if keep], observed[remain],
                fractions=[7 / 8, 1 / 8], minimum_groups=[5, 2], seed=seed)
            roles = ["test" if outer[g] == test_fold else ("train" if inner[g] == 0 else "validation") for g in group_ids]
            folds.append({"fold_id": test_fold, "roles": roles, "inner_report": inner_report,
                          "coverage_pass": inner_report["coverage_pass"] and outer_report["coverage_pass"]})
        passed = all(f["coverage_pass"] for f in folds)
        attempt = {"outer_folds": k, "outer_report": outer_report, "folds": folds, "coverage_pass": passed}
        attempts.append(attempt)
        if passed:
            return {"status": "COVERAGE_PASS_CANDIDATE", "selected_outer_folds": k, "attempts": attempts}
    return {"status": "STOP_COVERAGE_FAILED", "selected_outer_folds": None, "attempts": attempts}
