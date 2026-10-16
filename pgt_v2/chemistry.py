"""Graph-preserving chemical identities and explicit PGT features.

SMILES serialization is for display only. A bounded exact graph-labeling search
supplies identities and atom ordering, including the C01 serialization cycles.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler
import torch
from torch_geometric.data import Data

from molecular_features import FeatureSpec, PGT_FEATURE_SPEC


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def framed_digest(parts):
    """C01 identity/ranking serialization: UTF-8 parts with uint64 length prefixes."""
    result = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode('utf-8')
        result.update(len(encoded).to_bytes(8, 'big'))
        result.update(encoded)
    return result.hexdigest()


class UnsupportedChemistry(ValueError):
    pass


class _Union:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def stereo_states(mol):
    """CIP assignment on a copy, never inference of an unspecified stereocentre."""
    mol = Chem.Mol(mol)
    if mol.GetStereoGroups():
        raise UnsupportedChemistry("Enhanced relative/AND/OR stereochemistry needs an explicit policy")
    potential = list(Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=True))
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'): atom.ClearProp('_CIPCode')
    for bond in mol.GetBonds():
        if bond.HasProp('_CIPCode'): bond.ClearProp('_CIPCode')
    rdCIPLabeler.AssignCIPLabels(mol, maxRecursiveIterations=1250000)
    atoms = ["NONE"] * mol.GetNumAtoms()
    bonds = ["NONE"] * mol.GetNumBonds()
    for info in potential:
        kind = str(info.type)
        target = atoms if kind.startswith('Atom_') else bonds
        if kind not in {'Atom_Tetrahedral', 'Bond_Double'}:
            target[info.centeredOn] = 'UNSUPPORTED'
        else:
            target[info.centeredOn] = 'UNASSIGNED'
    for a in mol.GetAtoms():
        tag = str(a.GetChiralTag())
        if tag not in {'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'}:
            atoms[a.GetIdx()] = 'UNSUPPORTED'
        elif tag != 'CHI_UNSPECIFIED':
            code = a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else ''
            atoms[a.GetIdx()] = code if code in {'R','S','r','s'} else 'UNSUPPORTED'
    for b in mol.GetBonds():
        code = b.GetProp('_CIPCode') if b.HasProp('_CIPCode') else ''
        if code in {'E','Z'}:
            bonds[b.GetIdx()] = code
        elif str(b.GetStereo()) == 'STEREOANY':
            bonds[b.GetIdx()] = 'UNASSIGNED'
        elif str(b.GetStereo()) != 'STEREONONE':
            bonds[b.GetIdx()] = 'UNSUPPORTED'
    if 'UNSUPPORTED' in atoms or 'UNSUPPORTED' in bonds:
        raise UnsupportedChemistry('Unsupported assigned/potential stereochemistry; do not drop it or encode it as NONE')
    return mol, tuple(atoms), tuple(bonds)


def _atom_values(atom, stereo):
    return {
        'atomic_number': atom.GetAtomicNum(), 'absolute_stereochemistry': stereo,
        'isotope': atom.GetIsotope(), 'total_degree': atom.GetTotalDegree(),
        'formal_charge': atom.GetFormalCharge(), 'total_hydrogen_count': atom.GetTotalNumHs(),
        'radical_electron_count': atom.GetNumRadicalElectrons(), 'hybridization': str(atom.GetHybridization()),
        'aromaticity': int(atom.GetIsAromatic()), 'ring_membership': int(atom.IsInRing()),
    }


def _bond_values(bond, stereo, reverse=False):
    direction = 'NONE'
    if str(bond.GetBondType()) == 'DATIVE':
        direction = 'ACCEPTOR_TO_DONOR' if reverse else 'DONOR_TO_ACCEPTOR'
    return {'bond_type': str(bond.GetBondType()), 'absolute_stereochemistry': stereo,
            'conjugation': int(bond.GetIsConjugated()), 'ring_membership': int(bond.IsInRing()),
            'dative_direction': direction}


def _graph_labels(mol, atom_stereo, bond_stereo, *, include_stereo=True):
    # Full chemical values, not vocabulary bins: UNK cannot merge identities.
    nodes = []
    for a, state in zip(mol.GetAtoms(), atom_stereo):
        values = _atom_values(a, state if include_stereo else 'IGNORED_FOR_GROUPING')
        nodes.append(tuple((k, str(v)) for k,v in sorted(values.items())))
    edges = {}
    for b, state in zip(mol.GetBonds(), bond_stereo):
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        for u,v,reverse in [(i,j,False),(j,i,True)]:
            values = _bond_values(b, state if include_stereo else 'IGNORED_FOR_GROUPING', reverse)
            edges[u,v] = tuple((k,str(x)) for k,x in sorted(values.items()))
    return nodes, edges


def _canonical_component(nodes, edges, members, budget):
    """Individualization/refinement; enumerate unresolved cells, no hash-only proof.

    The minimum complete labeled adjacency is canonical. Equal minima give
    automorphism orbits. A search-limit failure never returns an approximate ID.
    """
    labels = [nodes[i] for i in members]
    local = {v:i for i,v in enumerate(members)}
    adj = [[] for _ in members]
    for (a,b),label in edges.items():
        if a in local and b in local: adj[local[a]].append((local[b],label))
    def colors_for(signatures):
        vocab = {s:i for i,s in enumerate(sorted(set(signatures)))}
        return tuple(vocab[s] for s in signatures)
    def refine(colors):
        while True:
            signatures = [(colors[i],tuple(sorted((edge,colors[j]) for j,edge in adj[i]))) for i in range(len(labels))]
            new = colors_for(signatures)
            if len(set(new)) == len(set(colors)): return new
            colors = new
    best = None
    best_order = None
    orbit = _Union(len(members))
    visited = 0
    def visit(colors):
        nonlocal best, best_order, orbit, visited
        visited += 1
        if visited > budget:
            raise UnsupportedChemistry(f'Exact chemical canonicalization exceeded {budget} search nodes')
        colors = refine(colors)
        cells = {}
        for i,c in enumerate(colors): cells.setdefault(c,[]).append(i)
        unresolved = [(len(v),c,v) for c,v in cells.items() if len(v)>1]
        if unresolved:
            _,_,cell = min(unresolved)
            for vertex in cell:
                child = list(colors); child[vertex] = max(colors)+1
                visit(tuple(child))
            return
        order = sorted(range(len(members)), key=colors.__getitem__)
        inverse = {v:i for i,v in enumerate(order)}
        code = (tuple(labels[i] for i in order),
                tuple(sorted((inverse[i],inverse[j],label) for i in range(len(members)) for j,label in adj[i])))
        if best is None or code < best:
            best,best_order,orbit = code,order,_Union(len(members))
        elif code == best:
            for a,b in zip(best_order,order): orbit.join(a,b)
    visit(colors_for(labels))
    assert best is not None and best_order is not None
    return best, [members[i] for i in best_order], [orbit.find(i) for i in best_order], visited


def canonical_graph(mol, atom_stereo, bond_stereo, *, include_stereo=True, search_limit=50000):
    nodes,edges = _graph_labels(mol,atom_stereo,bond_stereo,include_stereo=include_stereo)
    components=[]
    for members in Chem.GetMolFrags(mol):
        components.append(_canonical_component(nodes,edges,members,search_limit))
    components.sort(key=lambda x:x[0])
    order=[]; classes=[]; seen_component_classes={}; steps=0
    for code,part,orbits,visited in components:
        steps += visited
        # Identical disconnected components have interchangeable matching atoms.
        by_orbit={}
        canonical_orbits=[]
        for i,orbit in enumerate(orbits):
            by_orbit.setdefault(orbit,i);canonical_orbits.append(by_orbit[orbit])
        base = seen_component_classes.setdefault(code,len(order))
        classes.extend(base+i for i in canonical_orbits)
        order.extend(part)
    payload={'algorithm':'colored_graph_ir_v1','components':[x[0] for x in components]}
    return digest(payload),tuple(order),tuple(classes),steps


@dataclass
class MolecularInput:
    mol: Chem.Mol
    input_smiles: str
    display_smiles: str
    model_input_id: str
    nonstereo_id: str
    raw_to_model: tuple[int,...]
    model_to_raw: tuple[int,...]
    symmetry_classes: tuple[int,...]
    atom_stereo: tuple[str,...]
    bond_stereo: tuple[str,...]
    search_nodes: int


def prepare_molecule(source, *, spec: FeatureSpec=PGT_FEATURE_SPEC, search_limit=50000):
    spec.validate_runtime()
    raw = Chem.MolFromSmiles(source) if isinstance(source,str) else Chem.Mol(source)
    if raw is None or raw.GetNumAtoms()==0:
        raise UnsupportedChemistry('Invalid or empty molecular input')
    if any(a.HasQuery() or a.GetAtomicNum()==0 for a in raw.GetAtoms()):
        raise UnsupportedChemistry('Query/dummy atoms are not concrete molecular inputs')
    for atom in raw.GetAtoms(): atom.SetAtomMapNum(0)
    raw,atoms,bonds = stereo_states(raw)
    identity,order,symmetry,steps = canonical_graph(raw,atoms,bonds,search_limit=search_limit)
    nonstereo,_,_,extra = canonical_graph(raw,atoms,bonds,include_stereo=False,search_limit=search_limit)
    model = Chem.RenumberAtoms(raw,list(order))
    model,model_atoms,model_bonds = stereo_states(model)
    inverse=[0]*len(order)
    for model_idx,raw_idx in enumerate(order): inverse[raw_idx]=model_idx
    original_nodes,original_edges = _graph_labels(raw,atoms,bonds)
    model_nodes,model_edges = _graph_labels(model,model_atoms,model_bonds)
    if any(original_nodes[r]!=model_nodes[inverse[r]] for r in range(len(order))):
        raise UnsupportedChemistry('Atom properties changed during graph renumbering')
    if {(inverse[a],inverse[b]):label for (a,b),label in original_edges.items()} != model_edges:
        raise UnsupportedChemistry('Bond/stereo properties changed during graph renumbering')
    return MolecularInput(model,source if isinstance(source,str) else '<parsed Mol>',
        Chem.MolToSmiles(model,canonical=True,isomericSmiles=True), 'graph_'+identity,'nonstereo_'+nonstereo,
        tuple(inverse),order,symmetry,model_atoms,model_bonds,steps+extra)


def graph_from_input(molecule: MolecularInput, *, sample_id, raw_row_index, spec=PGT_FEATURE_SPEC):
    import algos  # preserved Cython distance kernel; never gen_edge_input in direct mode
    spec.validate_runtime()
    mol=molecule.mol;n=mol.GetNumAtoms();width=len(spec.bond_fields)
    x=torch.tensor([spec.encode('atom',_atom_values(a,s)) for a,s in zip(mol.GetAtoms(),molecule.atom_stereo)],dtype=torch.long)
    edge_type=torch.zeros((n,n,width),dtype=torch.long)
    adjacency=np.zeros((n,n),dtype=np.int64)
    for b,s in zip(mol.GetBonds(),molecule.bond_stereo):
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        edge_type[i,j]=torch.tensor(spec.encode('bond',_bond_values(b,s)))
        edge_type[j,i]=torch.tensor(spec.encode('bond',_bond_values(b,s,True)))
        adjacency[i,j]=adjacency[j,i]=1
    distance,_=algos.floyd_warshall(np.ascontiguousarray(adjacency))
    return Data(x=x,in_degree=torch.from_numpy(adjacency.sum(1)),out_degree=torch.from_numpy(adjacency.sum(1)),
        spatial_pos=torch.from_numpy(np.asarray(distance,dtype=np.int64)),attn_edge_type=edge_type,
        # Compatibility shape only, not a selected shortest-path sequence.
        edge_input=torch.full((n,n,1,width),-1,dtype=torch.long),y=torch.tensor([float('nan')]),
        sample_id=str(sample_id),raw_row_index=int(raw_row_index),num_nodes=n,
        feature_schema_version=spec.version,feature_spec_sha256=spec.sha256,
        model_input_id=molecule.model_input_id,nonstereo_id=molecule.nonstereo_id,
        smiles=molecule.input_smiles,canonical_smiles=molecule.display_smiles,
        raw_to_model=list(molecule.raw_to_model),model_to_raw=list(molecule.model_to_raw),
        symmetry_classes=list(molecule.symmetry_classes),edge_bias_mode='direct')


def resolve_input_record(record: Mapping, *, source_sha256: str, overrides: Mapping):
    """Only reviewed, content-matching overrides may resolve two-field conflicts."""
    sample_id=record['record_id'];lower=record['selected_input_smiles'];upper=record['raw_smiles_upper']
    override=overrides.get(sample_id)
    if override:
        expected=(override['source_sha256'],override['TAID'],override['raw_smiles_upper'],override['selected_input_smiles'])
        actual=(source_sha256,record['TAID'],upper,lower)
        if actual!=expected: raise ValueError('Reviewed structure override identity mismatch')
        chosen=upper
        provenance={'selection':'reviewed_author_raw_graph','uncertainty':override['uncertainty']}
    else:
        chosen=lower
        if upper!=lower and prepare_molecule(upper).model_input_id!=prepare_molecule(lower).model_input_id:
            raise UnsupportedChemistry(f'Unreviewed structure conflict for {sample_id}')
        provenance={'selection':'provided_lowercase_graph','uncertainty':None}
    # Source InChIKey/CID are retained as provenance, never used to merge inputs.
    provenance.update(source_inchikey=record.get('source_inchikey',''),source_cid=record.get('pubchem_cid',''))
    return prepare_molecule(chosen),provenance
