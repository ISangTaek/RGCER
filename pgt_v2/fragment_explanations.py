"""Training-fragment substitutions with disjoint explanation/evaluation samples.

These are local model perturbations, not measured or causal toxicity effects.
No label or model prediction enters fragment generation.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator
from scipy.stats import rankdata

from .chemistry import UnsupportedChemistry, prepare_molecule

ORIGIN = '_pgt_explanation_origin'
MAX_FRAGMENTS = 8
MAX_ATTEMPTS_PER_FRAGMENT = 128
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def size_bin(size):
    return 1 if size == 1 else 3 if size <= 3 else 6 if size <= 6 else 12


def _key_atom(atom):
    return (atom.GetAtomicNum(), str(atom.GetHybridization()),
            atom.GetIsAromatic(), atom.GetFormalCharge())


def cut_fragments(mol):
    """One attachment, non-ring single cuts; retain parent atom indices.

    Prefer BRICS environments. Other single cuts use the observed retained-root
    environment from training, explicitly labelled as a separate cut family.
    Assigned stereo at the cut is excluded, rather than silently discarded.
    """
    if len(Chem.GetMolFrags(mol)) != 1:
        return []
    marked = Chem.Mol(mol)
    for atom in marked.GetAtoms():
        atom.SetIntProp(ORIGIN, atom.GetIdx())
    brics = {}
    for (left, right), labels in BRICS.FindBRICSBonds(marked):
        brics[(left, right)] = str(labels[0])
        brics[(right, left)] = str(labels[1])
    result = []
    for bond in marked.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ends = [marked.GetAtomWithIdx(i) for i in (left, right)]
        if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED or
               any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in a.GetBonds()) for a in ends):
            continue
        broken = Chem.FragmentOnBonds(marked, [bond.GetIdx()], dummyLabels=[(0, 0)])
        parts = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
        if len(parts) != 2:
            continue
        for side, fragment in enumerate(parts):
            atoms = [a.GetIntProp(ORIGIN) for a in fragment.GetAtoms() if a.GetAtomicNum()]
            if not atoms or len(atoms) > 12 or len(atoms) * 2 > marked.GetNumAtoms():
                continue
            dummy = next(a for a in fragment.GetAtoms() if not a.GetAtomicNum())
            root = dummy.GetNeighbors()[0].GetIntProp(ORIGIN)
            kept_root = right if root == left else left
            family = 'BRICS' if (root, kept_root) in brics else 'observed_single_bond'
            environment = (family, brics[(root, kept_root)]) if family == 'BRICS' else (family, *_key_atom(marked.GetAtomWithIdx(kept_root)))
            aromatic = any(a.GetIsAromatic() for a in fragment.GetAtoms())
            result.append(dict(atoms=sorted(atoms), fragment=fragment, context=parts[1-side],
                environment=environment, size=len(atoms), aromatic=aromatic, family=family,
                fragment_smiles=Chem.MolToSmiles(fragment, isomericSmiles=True)))
    # BRICS first, then small disjoint terminal pieces during panel selection.
    return sorted(result, key=lambda x: (x['family'] != 'BRICS', x['size'], x['atoms']))


def build_library(records):
    """records: iterable of (training record ID, registered Mol), no labels."""
    library = defaultdict(dict)
    for record_id, mol in records:
        for cut in cut_fragments(mol):
            key = (cut['environment'], size_bin(cut['size']), cut['aromatic'])
            smiles = cut['fragment_smiles']
            if smiles not in library[key] and len(library[key]) < 128:
                donor = Chem.Mol(cut['fragment'])
                for atom in donor.GetAtoms():
                    if atom.HasProp(ORIGIN):
                        atom.ClearProp(ORIGIN)
                library[key][smiles] = dict(mol=donor, smiles=smiles, source_record_id=record_id)
    return {key: [values[s] for s in sorted(values)] for key, values in library.items()}


def substitute(context, donor):
    joined = Chem.RWMol(Chem.CombineMols(context, donor))
    dummies = [a.GetIdx() for a in joined.GetAtoms() if not a.GetAtomicNum()]
    if len(dummies) != 2:
        raise ValueError('Substitution requires one attachment on each fragment')
    roots = [joined.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in dummies]
    joined.AddBond(*roots, Chem.BondType.SINGLE)
    for index in sorted(dummies, reverse=True):
        joined.RemoveAtom(index)
    product = joined.GetMol()
    Chem.SanitizeMol(product)
    Chem.AssignStereochemistry(product, cleanIt=True, force=True)
    return product


def make_panel(mol, library, *, seed):
    """<=8 fragments, <=8 A and <=4 B substitutions each, <=96 total.

    Candidate identity is unique over the entire molecular panel, so A and B
    never share a replacement even when two cuts could generate it.
    """
    rng = np.random.default_rng(seed)
    selected, occupied = [], set()
    for cut in cut_fragments(mol):
        if occupied.intersection(cut['atoms']):
            continue
        selected.append(cut)
        occupied.update(cut['atoms'])
        if len(selected) == MAX_FRAGMENTS:
            break
    original_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    seen = {original_smiles}
    original_fp = MORGAN.GetFingerprint(mol)
    candidates, fragments = [], []
    failures = defaultdict(int)
    for cut in selected:
        fragment_id = len(fragments)
        key = (cut['environment'], size_bin(cut['size']), cut['aromatic'])
        donors = library.get(key, [])
        valid = []
        for index in rng.permutation(len(donors))[:MAX_ATTEMPTS_PER_FRAGMENT]:
            donor = donors[index]
            if donor['smiles'] == cut['fragment_smiles']:
                continue
            try:
                product = substitute(cut['context'], donor['mol'])
                smiles = Chem.MolToSmiles(product, isomericSmiles=True)
                if smiles in seen:
                    continue
                prepared = prepare_molecule(product)
            except (ValueError, RuntimeError, UnsupportedChemistry) as exc:
                failures[type(exc).__name__] += 1
                continue
            seen.add(smiles)
            mapping = {str(a.GetIntProp(ORIGIN)): a.GetIdx() for a in prepared.mol.GetAtoms() if a.HasProp(ORIGIN)}
            valid.append(dict(molecule=prepared, fragment_id=fragment_id, smiles=smiles,
                donor_record_id=donor['source_record_id'], donor_fragment=donor['smiles'],
                retained_original_to_model=mapping,
                parent_similarity=float(DataStructs.TanimotoSimilarity(original_fp, MORGAN.GetFingerprint(product)))))
            if len(valid) == 12:
                break
        # Both sets receive samples, even when a small donor pool has <12.
        b_count = min(4, len(valid)//3) if len(valid) >= 3 else 1 if len(valid) == 2 else 0
        a_count = min(8, len(valid)-b_count)
        valid = valid[:a_count+b_count]
        for index, item in enumerate(valid):
            item['set'] = 'A' if index < a_count else 'B'
            item['candidate_id'] = len(candidates)
            candidates.append(item)
        fragments.append(dict(fragment_id=fragment_id, atoms=cut['atoms'], size=cut['size'],
            size_bin=size_bin(cut['size']), aromatic=cut['aromatic'], family=cut['family'],
            environment=list(cut['environment']), original_fragment=cut['fragment_smiles'],
            a_count=a_count, b_count=b_count))
    return dict(fragments=fragments, candidates=candidates, rejected=dict(failures),
        coverage_reason='multi_component' if len(Chem.GetMolFrags(mol)) != 1 else
        'no_eligible_local_cut' if not fragments else 'evaluated_or_limited_donor_pool')


def correlation(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def attribution_scores(atom_values, fragments):
    return [float(np.abs(np.asarray(atom_values)[f['atoms']]).sum()) for f in fragments]


def compare_ranking(ranking_scores, responses, fragments):
    """Expected size/aromaticity-matched random control, with ties averaged."""
    scores = np.asarray(ranking_scores, dtype=float)
    effects = np.asarray(responses, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(effects)
    ids = np.flatnonzero(valid)
    result = dict(evaluable_fragments=len(ids), capture_gain=None, capture_fraction=None,
                  random_capture_fraction=None, rank_correlation=None)
    if len(ids) < 2 or effects[ids].sum() <= 1e-10:
        return result
    count = max(1, math.ceil(.2 * len(ids)))
    cutoff = np.sort(scores[ids])[-count]
    inclusion = (scores[ids] > cutoff).astype(float)
    tied = scores[ids] == cutoff
    inclusion[tied] = (count-inclusion.sum())/tied.sum()
    groups = defaultdict(list)
    for position, i in enumerate(ids):
        f = fragments[i]
        groups[(f['size_bin'], f['aromatic'])].append(position)
    random_effect = 0.0
    exchangeable = False
    for positions in groups.values():
        mass = inclusion[positions].sum()
        random_effect += mass * effects[ids[positions]].mean()
        exchangeable |= len(positions) > 1 and 0 < mass < len(positions)
    total = float(effects[ids].sum())
    capture = float(inclusion @ effects[ids] / total)
    random_capture = float(random_effect / total)
    result.update(capture_fraction=capture, random_capture_fraction=random_capture,
        capture_gain=capture-random_capture if exchangeable else None,
        matched_random_has_alternatives=bool(exchangeable),
        rank_correlation=correlation(scores[ids], effects[ids]))
    return result


def evaluate_fragments(panel, predictions, *, task, scale, atom_explanations=None):
    """predictions[0] is original; remaining rows follow panel candidates."""
    delta = predictions[1:, task]-predictions[0, task]
    responses, scores, signed = [], [], []
    for fragment in panel['fragments']:
        a = [i for i, c in enumerate(panel['candidates']) if c['fragment_id'] == fragment['fragment_id'] and c['set'] == 'A']
        b = [i for i, c in enumerate(panel['candidates']) if c['fragment_id'] == fragment['fragment_id'] and c['set'] == 'B']
        scores.append(float(np.abs(delta[a]).mean()) if a else np.nan)
        responses.append(float(np.abs(delta[b]).mean()) if b else np.nan)
        signed.append(float((-delta[a]).mean()) if a else np.nan)
    rankings = {'common_perturbation':scores}
    for name, values in (atom_explanations or {}).items():
        rankings[name] = attribution_scores(values, panel['fragments'])
    result = {name:compare_ranking(values, responses, panel['fragments']) for name, values in rankings.items()}
    clean = lambda values: [float(v) if np.isfinite(v) else None for v in values]
    return dict(metrics=result, a_mean_absolute_response=clean(scores),
        b_mean_absolute_response=clean(responses), a_signed_removal_response=clean(signed),
        b_response_in_training_scale=clean(np.asarray(responses)/scale))
