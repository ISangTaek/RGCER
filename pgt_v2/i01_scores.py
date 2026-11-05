"""Paired descriptive scores and case readouts for the fixed I01 panel."""
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from pgt_v2.artifacts import read_json, write_json
from pgt_v2.explanation_pilot import endpoint_differences
from pgt_v2.fragment_explanations import attribution_scores, correlation


EXPLAINERS = ("common_perturbation", "ig_training_mean", "ig_zero")


def top_overlap(left, right):
    """Tie-inclusive top 20%. A constant ranking has no identifiable top set."""
    a,b = np.asarray(left),np.asarray(right)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:
        return None
    n = int(np.ceil(.2*len(a)))
    x,y = (set(np.flatnonzero(v>=np.sort(v)[-n])) for v in (a,b))
    return len(x&y)/len(x|y)


def task_macro(rows, kind, value, *, overall_panel="all40", task_groups=None):
    """Average observations within task, then give each task equal weight."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"],row["explanation"],row["panel"]].append(row)
        groups[row["method"],row["explanation"],overall_panel].append(row)
        for name, tasks in (task_groups or {}).items():
            if row["task"] in tasks:
                groups[row["method"],row["explanation"],name].append(row)
    results = []
    for (method,explanation,panel),members in sorted(groups.items()):
        by_task = defaultdict(list)
        tasks = sorted({r["task"] for r in members})
        for r in members:
            if r[value] is not None:
                by_task[r["task"]].append(r[value])
        means = {t:float(np.mean(by_task[t])) if by_task[t] else None for t in tasks}
        covered = [v for v in means.values() if v is not None]
        score = float(np.mean(covered)) if covered else None
        usable = [r for r in members if r[value] is not None]
        results.append(dict(metric=kind,statistic=value,method=method,explanation=explanation,panel=panel,
            registered_observations=len(members),shared_observations=len(usable),
            shared_records=len({r["record_id"] for r in usable}),shared_structure_groups=len({r["group_id"] for r in usable}),
            covered_tasks=len(covered),registered_tasks=len(tasks),per_task=means,covered_macro=score,
            full_macro=score if len(covered)==len(tasks) else None))
    return results


def summarize(reg, root, *, record_loader=None):
    observations,rank_rows,stability_rows,species = [],[],[],[]
    analogue_rows = defaultdict(dict)
    flags = Counter()
    candidates = 0
    sample_count = 0
    for p in reg["panel"]:
        saved = (record_loader(p) if record_loader is not None else
                 read_json(root/"records"/(p["record_id"]+".json")))
        structure,runs = saved["structure"],saved["runs"]
        fragments = structure["fragments"]
        candidates += len(structure["candidates"])
        if {(u["method"],u["seed"]) for u in runs}!={(r["method"],r["seed"]) for r in reg["runs"]}:
            raise ValueError("Incomplete set of paired weights")
        for u in runs:
            if [(v["panel"],v["task"]) for v in u["observations"]] != [(v["panel"],v["task"]) for v in p["observations"]]:
                raise ValueError("Incomplete fixed observation set")
            sample_count += len(u["observations"])
            tagged = {v["task"]:v for v in u["observations"] if v["panel"]=="species_explanation"}
            if tagged:
                left,right=tagged["mouse_oral_LD50"],tagged["rat_oral_LD50"]
                delta = endpoint_differences([left,right])[0]
                control = left["training_mean"] - right["training_mean"]
                species.append(dict(record_id=p["record_id"],method=u["method"],seed=u["seed"],
                    training_mean_difference=control, absolute_error=abs(delta["prediction"]-delta["truth"]),
                    training_mean_absolute_error=abs(control-delta["truth"]), **delta))
            for v in u["observations"]:
                for name,ig in v["ig"].items():
                    flags[(u["method"],name,"total")]+=1
                    flags[(u["method"],name,"flagged")]+=int(ig["unreliable"])
                if v["panel"]=="analogue_pairs":
                    choices=structure["analogue_variable_atom_sets"]
                    analogue_rows[u["method"],u["seed"]][p["record_id"]]=dict(observation=v, choices=choices)
        for pos,obs in enumerate(p["observations"]):
            key=dict(record_id=p["record_id"],group_id=p["group_id"],task=obs["task"],panel=obs["panel"])
            cells={(u["method"],u["seed"]):u["observations"][pos] for u in runs}
            groups={m:sorted(seed for method,seed in cells if method==m) for m in ("M2","N1S")}
            pairs=structure["matched_pairs"]
            eligible=[i for i,f in enumerate(fragments) if f["a_count"] and f["b_count"]]
            reason=None if pairs else (structure["coverage_reason"] if not fragments else
                "insufficient_A_or_B" if len(eligible)<2 else "no_matched_size_aromaticity_pair")
            observations.append(dict(**key,n_fragments=len(fragments),n_with_A_and_B=len(eligible),
                                     matched_pairs=len(pairs),reason=reason))
            for name in EXPLAINERS:
                scores={k:v["fragment_evaluation"]["metrics"][name]["pair_concordance"] for k,v in cells.items()}
                common=all(x is not None for x in scores.values())
                stability={}
                for method,seeds in groups.items():
                    ranks={}
                    for s in seeds:
                        v=cells[method,s]
                        a=(v["fragment_evaluation"]["a_mean_absolute_response"] if name=="common_perturbation"
                           else attribution_scores(v["ig"][name[3:]]["attributions"],fragments))
                        ranks[s]=[a[i] for i in eligible]
                    score=float(np.mean([scores[method,s] for s in seeds])) if common else None
                    qualified=(name=="common_perturbation" or
                               all(not cell["ig"][name[3:]]["unreliable"] for cell in cells.values()))
                    rank_rows.append(dict(**key,method=method,explanation=name,
                        per_seed={str(s):scores[method,s] for s in seeds},
                        score=score, qualified_score=score if qualified else None,
                        prediction_sd=float(np.std([cells[method,s]["prediction"] for s in seeds],ddof=1))))
                    stability[method]=[dict(seeds=[a,b],spearman=correlation(ranks[a],ranks[b]),
                        top20_jaccard=top_overlap(ranks[a],ranks[b])) for a,b in combinations(seeds,2)]
                for method,seed_pairs in stability.items():
                    entry=dict(**key,method=method,explanation=name,per_seed_pair=seed_pairs)
                    for m in ("spearman","top20_jaccard"):
                        shared=all(v[m] is not None for items in stability.values() for v in items)
                        entry[m]=float(np.mean([v[m] for v in seed_pairs])) if shared else None
                    stability_rows.append(entry)
    analogues=[]
    pair=reg["analogue"]
    for (method,seed),cases in sorted(analogue_rows.items()):
        a,b=(cases[pair["record_"+side]] for side in ("a","b"))
        left,right=a["observation"],b["observation"]
        observed = left["truth"] - right["truth"]
        predicted = left["prediction"] - right["prediction"]
        analogues.append(dict(method=method,seed=seed,record_a=pair["record_a"],record_b=pair["record_b"],
            truth_A_minus_B=observed,prediction_A_minus_B=predicted,absolute_error=abs(predicted-observed),
            variable_atom_sets_A=a["choices"],variable_atom_sets_B=b["choices"],
            attribution={k:dict(baseline_difference=left["ig"][k]["baseline_prediction"]-right["ig"][k]["baseline_prediction"],
                variable_signed_sums_A=[float(np.sum(np.asarray(left["ig"][k]["attributions"])[idx])) for idx in a["choices"]],
                variable_signed_sums_B=[float(np.sum(np.asarray(right["ig"][k]["attributions"])[idx])) for idx in b["choices"]])
                for k in left["ig"]}))
    write_json(root/"paired_scores.json",rank_rows)
    write_json(root/"stability.json",stability_rows)
    write_json(root/"cases.json",dict(species=species,analogue_definition=pair,analogue=analogues))
    options = dict(overall_panel=reg.get("overall_panel", "all40"),
                   task_groups=reg.get("explanation_task_groups"))
    paired = {(r["record_id"],r["task"]) for r in observations if r["matched_pairs"]}
    same_scores = [r for r in rank_rows if (r["record_id"],r["task"]) in paired]
    same_stability = [r for r in stability_rows if (r["record_id"],r["task"]) in paired]
    write_json(root/"same_coverage_summary.json",
        task_macro(same_scores,"faithfulness","score",**options)+
        task_macro(same_stability,"stability","spearman",**options)+
        task_macro(same_stability,"stability","top20_jaccard",**options))
    return dict(observations=sample_count,unique_molecules=len(reg["panel"]),coverage=observations,
        replacement_molecules_across_parents=candidates,ig_flags={"|".join(k):v for k,v in flags.items()},
        metrics=task_macro(rank_rows,"faithfulness","score",**options)+
                task_macro([r for r in rank_rows if r["explanation"].startswith("ig_")],
                           "faithfulness_common_unflagged","qualified_score",**options)+
                task_macro(stability_rows,"stability","spearman",**options)+
                task_macro(stability_rows,"stability","top20_jaccard",**options))


def organic_sensitivity(reg, root, *, record_loader=None):
    """Re-score existing responses after the registered chemistry filter; no inference."""
    from rdkit import Chem
    from pgt_v2.fragment_explanations import evaluate_fragments, matched_pairs

    allowed = set("H B C N O F Si P S Cl Se Br I".split())
    excluded = []

    def filtered(point):
        saved = (record_loader(point) if record_loader is not None else
                 read_json(root/"records"/(point["record_id"]+".json")))
        structure = saved["structure"]
        keep = []
        for i, candidate in enumerate(structure["candidates"]):
            mol = Chem.MolFromSmiles(candidate["smiles"])
            uncommon = sorted({a.GetSymbol() for a in mol.GetAtoms()}-allowed)
            radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            if uncommon or radicals:
                excluded.append(dict(record_id=point["record_id"],candidate_id=candidate["candidate_id"],
                    smiles=candidate["smiles"],uncommon_elements=uncommon,radical_electrons=radicals))
            else:
                keep.append(i)
        structure["candidates"] = [structure["candidates"][i] for i in keep]
        counts = Counter((c["fragment_id"],c["set"]) for c in structure["candidates"])
        for fragment in structure["fragments"]:
            fragment["a_count"] = counts[fragment["fragment_id"],"A"]
            fragment["b_count"] = counts[fragment["fragment_id"],"B"]
        structure["matched_pairs"] = matched_pairs(structure["fragments"])
        for run in saved["runs"]:
            for obs in run["observations"]:
                obs["perturbation_predictions"] = [obs["perturbation_predictions"][i] for i in keep]
                predictions = np.asarray([obs["prediction"]]+obs["perturbation_predictions"])[:,None]
                obs["fragment_evaluation"] = evaluate_fragments(structure,predictions,task=0,
                    scale=reg["training_scales"][obs["task_index"]],
                    atom_explanations={"ig_"+k:v["attributions"] for k,v in obs["ig"].items()})
        return saved

    destination = root/"organic_sensitivity"
    summary = summarize(reg,destination,record_loader=filtered)
    write_json(destination/"summary.json",summary)
    write_json(destination/"exclusions.json",dict(policy="organic_closed_shell",allowed_elements=sorted(allowed),
        excluded=excluded,extra_model_inference=0))
    return summary
