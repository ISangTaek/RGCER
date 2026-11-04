"""Inference only. I01 compares six existing weights on the registered S0 panel."""
import argparse
from collections import Counter
from pathlib import Path
import platform

import numpy as np
import torch
from rdkit import Chem

from pgt_v2.artifacts import load_arrays, load_manifest, read_json, write_json
from pgt_v2.cache import GraphCache
from pgt_v2.chemistry import digest, graph_from_input
from pgt_v2.config import git_identity
from pgt_v2.data import PGTCollator
from pgt_v2.evaluation import load_model_checkpoint
from pgt_v2.explain import integrated_gradients, training_embedding_mean
from pgt_v2.explanation_pilot import check_reference_predictions, predict_molecules
from pgt_v2.fragment_explanations import build_library, evaluate_fragments, make_panel, matched_pairs


METHODS = ("M2", "N1S")
SEEDS = (42, 43, 44)


def check_registry(reg):
    if git_identity() != reg["code_commit"]:
        raise ValueError("Use the published runtime commit")
    cells = [(r["method"], r["seed"]) for r in reg["runs"]]
    if sorted(cells) != sorted((m,s) for m in METHODS for s in SEEDS):
        raise ValueError("Expected the six registered model/seed pairs")
    counts = Counter(o["panel"] for p in reg["panel"] for o in p["observations"])
    if len(reg["panel"]) != 29 or counts != {"general_explanation":18, "species_explanation":20, "analogue_pairs":2}:
        raise ValueError("Wrong fixed explanation panel")
    return counts


def load_runs(reg, args):
    runs = []
    for spec in reg["runs"]:
        work = args.a00 if spec["method"] == "M2" else args.h02
        path = work / "results" / spec["unit"] / "training/best.pt"
        if read_json(str(path)+".sha256.json")["sha256"] != spec["checkpoint_sha256"]:
            raise ValueError("The checkpoint is not the archived best weight: "+spec["unit"])
        model, ck = load_model_checkpoint(path, device=args.device)
        ident = ck["identity"]
        if (ident["method_id"], ident["init_seed"], ident["run_sha256"], ident["code_commit"],
                ident["view_sha256"], ident["task_names"]) != (spec["method"], spec["seed"], spec["run_sha256"],
                spec["training_commit"], reg["view_sha256"], reg["tasks"]):
            raise ValueError("Checkpoint identity differs: "+spec["unit"])
        model.requires_grad_(False)
        runs.append((spec,model,ck["stats"]))
    return runs


def variable_atom_sets(mol, constant, variable):
    """Return every exact single-cut match, using the cached model's atom ordering."""
    matches = set()
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        cut = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(0,0)])
        for atom in cut.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(1)
        mapping = []
        parts = Chem.GetMolFrags(cut, asMols=True, fragsMolAtomMapping=mapping)
        smiles = [Chem.MolToSmiles(p, isomericSmiles=True) for p in parts]
        if len(parts) == 2 and constant in smiles and variable in smiles:
            matches.add(tuple(sorted(i for i in mapping[smiles.index(variable)] if i < mol.GetNumAtoms())))
    return [list(s) for s in sorted(matches)]


def make_perturbations(point, graph, mol, library, seed, analogue):
    if (graph.sample_id, graph.model_input_id, graph.raw_row_index) != (
            point["record_id"], point["chemical_id"], point["raw_row_index"]):
        raise ValueError("Frozen observation and cached chemical differ")
    panel = make_panel(mol, library, seed=seed)
    seen = {graph.model_input_id}
    unique = []
    for c in panel["candidates"]:
        key = c["molecule"].model_input_id
        if key in seen:
            panel["rejected"]["repeated_model_input"] = panel["rejected"].get("repeated_model_input",0)+1
            continue
        seen.add(key)
        c["candidate_id"] = len(unique)
        unique.append(c)
    panel["candidates"] = unique
    counts = Counter((c["fragment_id"],c["set"]) for c in unique)
    for f in panel["fragments"]:
        f["a_count"],f["b_count"] = counts[f["fragment_id"],"A"],counts[f["fragment_id"],"B"]
    graphs, mols = [graph], [mol]
    candidates = []
    for i,c in enumerate(panel["candidates"]):
        prepared = c["molecule"]
        kept = c["retained_original_to_model"]
        removed = panel["fragments"][c["fragment_id"]]["atoms"]
        if set(map(int,kept)) != set(range(mol.GetNumAtoms()))-set(removed) or len(set(kept.values())) != len(kept):
            raise ValueError("Retained atom mapping differs from original fragment complement")
        graphs.append(graph_from_input(prepared,sample_id=f"{graph.sample_id}_edit_{i}",raw_row_index=0))
        mols.append(prepared.mol)
        candidates.append({**{k:v for k,v in c.items() if k!="molecule"}, "chemical_id":prepared.model_input_id})
    meta = dict(record_id=graph.sample_id,chemical_id=graph.model_input_id,group_id=point["group_id"],
        canonical_smiles=graph.canonical_smiles,model_to_raw=graph.model_to_raw,raw_to_model=graph.raw_to_model,
        symmetry_classes=graph.symmetry_classes, fragments=panel["fragments"], candidates=candidates,
        rejected=panel["rejected"], coverage_reason=panel["coverage_reason"], matched_pairs=matched_pairs(panel["fragments"]))
    for side in ("a","b"):
        if graph.sample_id == analogue["record_"+side]:
            meta["analogue_variable_atom_sets"] = variable_atom_sets(mol,analogue["constant_smiles"],analogue["variable_"+side])
    return panel, graphs, mols, meta


def evaluate_run(point, run, mean_embedding, prepared, device):
    spec, model, stats = run
    panel, graphs, mols, meta = prepared
    pred = predict_molecules(model,graphs,mols,spec["method"],stats,device)
    max_error = check_reference_predictions(pred[0],spec["original_predictions"][point["record_id"]])
    batch = PGTCollator()([graphs[0]]).to(device)
    observations = []
    for obs in point["observations"]:
        t = obs["task_index"]
        ig = {}
        for label,baseline in [("training_mean",mean_embedding),("zero",None)]:
            result = integrated_gradients(model,batch,task=t,mean=stats["mean"],scale=stats["scale"],
                                          baseline=baseline,points=64,path_power=2)
            data = {k:v.detach().cpu()[0].tolist() for k,v in result.items() if isinstance(v,torch.Tensor)}
            data["path_power"] = result["path_power"]
            check_reference_predictions(np.asarray([data["prediction"]]),pred[0,t:t+1])
            if len(data["attributions"]) != len(meta["model_to_raw"]):
                raise ValueError("Attribution does not cover every model atom")
            ig[label] = data
        effects = evaluate_fragments(panel,pred,task=t,scale=stats["scale"][t],
            atom_explanations={"ig_"+k:v["attributions"] for k,v in ig.items()})
        observations.append(dict(**obs, prediction=float(pred[0,t]), training_mean=float(stats["mean"][t]),
            ig=ig, fragment_evaluation=effects, perturbation_predictions=pred[1:,t].tolist()))
    return dict(unit=spec["unit"],method=spec["method"],seed=spec["seed"],replay_max_abs=max_error,
                observations=observations)


def execute(args, reg):
    view = load_manifest(args.view)
    if view["manifest_sha256"] != reg["view_sha256"]:
        raise ValueError("I01 requires the S0/S21_P10 training view")
    train = load_arrays(args.view,"train_inputs.npz",view)
    test = load_arrays(args.view,"test_inputs.npz",view)
    offsets = {str(r):i for i,r in enumerate(test["record_id"])}
    for point in reg["panel"]:
        i = offsets[point["record_id"]]
        if any(test[key][i] != point[key] for key in ("cache_offset","raw_row_index","chemical_id","group_id")):
            raise ValueError("S0 test record has changed")
    cache = GraphCache(args.cache,source_sha256=reg["source_sha256"],verify_database=False)
    if cache.manifest["cache_id"] != reg["cache_id"]:
        raise ValueError("Unexpected graph cache")
    runs = load_runs(reg,args)
    root = args.out
    root.mkdir(parents=True,exist_ok=True)
    marker = dict(registry=digest(reg),commit=reg["code_commit"],new_fits=0)
    if any(root.iterdir()) and (not (root/"identity.json").exists() or read_json(root/"identity.json") != marker):
        raise ValueError("Use a new directory, or the matching partial I01 export")
    write_json(root/"identity.json",marker)
    first = reg["panel"][0]
    g,m = cache.read(first["cache_offset"])
    replay = {}
    for spec,model,stats in runs:
        y = predict_molecules(model,[g],[m],spec["method"],stats,args.device)[0]
        replay[spec["unit"]] = check_reference_predictions(y,spec["original_predictions"][g.sample_id])
    write_json(root/"server_preflight.json",dict(host=platform.node(),device=str(args.device),record=g.sample_id,
                                               original_prediction_max_abs=replay))
    print("S0 inputs and all six original checkpoint predictions verified",flush=True)
    refs = set(reg["reference_record_ids"])
    reference_graphs = {}
    def donors():
        for record,offset in zip(train["record_id"],train["cache_offset"]):
            graph,mol = cache.read(int(offset))
            if graph.sample_id != record:
                raise ValueError("Training donor cache record differs")
            if record in refs:
                reference_graphs[str(record)] = graph
            yield str(record),mol
    library = build_library(donors())
    ordered = [reference_graphs[r] for r in reg["reference_record_ids"]]
    means = {}
    for spec,model,_ in runs:
        batches = (PGTCollator()(ordered[i:i+8]).to(args.device) for i in range(0,len(ordered),8))
        means[spec["unit"]] = training_embedding_mean(model,batches)
    write_json(root/"reference.json",dict(record_ids=reg["reference_record_ids"],training_donor_rows=len(train["record_id"]),
        environments=len(library), library_size=sum(map(len,library.values())),
        embedding_means={k:v.cpu().tolist() for k,v in means.items()}))
    for i,point in enumerate(reg["panel"]):
        path = root/"records"/(point["record_id"]+".json")
        if not path.exists():
            graph,mol = cache.read(point["cache_offset"])
            prepared = make_perturbations(point,graph,mol,library,reg["panel_seed"]+i,reg["analogue"])
            cells = [evaluate_run(point,run,means[run[0]["unit"]],prepared,args.device) for run in runs]
            write_json(path,dict(structure=prepared[3],runs=cells))
        print(f"{i+1}/{len(reg['panel'])} {point['record_id']}",flush=True)
    from .i01_scores import summarize
    summary = summarize(reg,root)
    write_json(root/"summary.json",summary)
    write_json(root/"execution.json",dict(status="COMPLETE",new_fits=0,host=platform.node(),
        fixed_checkpoint_count=len(runs),records=len(reg["panel"]),task_model_observations=summary["observations"],
        evaluated_role=reg["role"],scientific_review="PENDING"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",type=Path,required=True,help="frozen I01 registry from the handoff")
    commands = parser.add_subparsers(dest="command",required=True)
    commands.add_parser("check", help="Verify frozen panel and runtime at the current checkout")
    run = commands.add_parser("run")
    for name in ("view","cache","a00","h02","out"):
        run.add_argument("--"+name,type=Path,required=True)
    run.add_argument("--device",default="cuda:0")
    options = parser.parse_args()
    registry = read_json(options.registry)
    print(dict(check_registry(registry)),flush=True)
    if options.command == "run":
        torch.set_num_threads(1)
        execute(options,registry)
