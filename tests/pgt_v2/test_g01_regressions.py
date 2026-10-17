"""Returned G01 chemistry failures and platform-independent source identities."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
import torch

from molecular_features import PGT_FEATURE_SPEC
from pgt_v2.cache import CacheEntry, GraphCache, build_cache
from pgt_v2.chemistry import (
    UnsupportedChemistry, _canonical_component, graph_from_input,
    prepare_molecule, resolve_input_record,
)
from pgt_v2.verification import text_source_identity

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/g01_structures.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['record_id'])
def test_all_returned_failures_preserve_graph_and_permutation(case, record_property):
    original, _ = resolve_input_record(case, source_sha256=FIXTURE['source_sha256'], overrides={})
    assert original.mol.GetNumAtoms() == case['num_atoms']
    assert original.search_nodes <= 100000  # two exact searches; unchanged per-component budget
    record_property('record_id', case['record_id'])
    record_property('canonical_search_nodes', original.search_nodes)
    graph = graph_from_input(original, sample_id=case['record_id'], raw_row_index=int(case['record_id'][4:]))
    for seed in (20261017, 20261018):
        perm = np.random.default_rng(seed).permutation(original.mol.GetNumAtoms()).tolist()
        changed = prepare_molecule(Chem.RenumberAtoms(original.mol, perm))
        assert (changed.model_input_id, changed.nonstereo_id, changed.symmetry_classes) == (
            original.model_input_id, original.nonstereo_id, original.symmetry_classes)
        rebuilt = graph_from_input(changed, sample_id=case['record_id'], raw_row_index=int(case['record_id'][4:]))
        for field in ('x', 'attn_edge_type', 'spatial_pos', 'in_degree', 'out_degree'):
            assert torch.equal(getattr(graph, field), getattr(rebuilt, field))
        assert all(changed.model_to_raw[changed.raw_to_model[i]] == i for i in range(case['num_atoms']))
    for kind in case['potential_coordination']:
        state = {'Atom_SquarePlanar':'UNASSIGNED_SQUARE_PLANAR',
                 'Atom_TrigonalBipyramidal':'UNASSIGNED_TRIGONAL_BIPYRAMIDAL',
                 'Atom_Octahedral':'UNASSIGNED_OCTAHEDRAL'}[kind]
        assert state in original.atom_stereo
        vocab = PGT_FEATURE_SPEC.metadata()['atom_vocabs']['absolute_stereochemistry']
        assert vocab[state] not in (vocab['NONE'], vocab['UNK'], vocab['UNASSIGNED'])


@pytest.mark.parametrize('source', ['Cl[Pt@SP1](Cl)(N)N', 'S[As@TB1](F)(Cl)(Br)N', 'S[Co@OH1](F)(Cl)(Br)(I)N'])
def test_assigned_coordination_is_still_rejected(source):
    with pytest.raises(UnsupportedChemistry, match='Unsupported'):
        prepare_molecule(source)


def test_non_tetrahedral_perception_cannot_be_disabled():
    before = Chem.GetAllowNontetrahedralChirality()
    try:
        Chem.SetAllowNontetrahedralChirality(False)
        with pytest.raises(ValueError, match='perception enabled'):
            prepare_molecule('Cl[Pt@SP1](Cl)(N)N')
    finally:
        Chem.SetAllowNontetrahedralChirality(before)


def test_new_coordination_states_survive_preserved_mol_cache(tmp_path):
    cases = [next(c for c in FIXTURE['cases'] if 'Atom_TrigonalBipyramidal' in c['potential_coordination']),
             next(c for c in FIXTURE['cases'] if 'Atom_Octahedral' in c['potential_coordination'])]
    entries = []
    for case in cases:
        molecule, provenance = resolve_input_record(case, source_sha256=FIXTURE['source_sha256'], overrides={})
        entries.append(CacheEntry(molecule, case['record_id'], int(case['record_id'][4:]), provenance))
    build_cache(entries, tmp_path/'cache', source_sha256=FIXTURE['source_sha256'])
    cache = GraphCache(tmp_path/'cache', source_sha256=FIXTURE['source_sha256'])
    for offset, entry in enumerate(entries):
        graph, mol = cache.read(offset, verify_chemistry=True)
        assert prepare_molecule(mol).atom_stereo == entry.molecule.atom_stereo
        assert graph.feature_spec_sha256 == PGT_FEATURE_SPEC.sha256


def _assert_orbits_from_brute_automorphisms(nodes, edges):
    """Independent exhaustive permutations, with no refinement or pruning."""
    n = len(nodes)
    code, order, orbits, _ = _canonical_component(nodes, edges, tuple(range(n)), 50000)
    expected = [{i} for i in range(n)]
    for permutation in itertools.permutations(range(n)):
        if any(nodes[i] != nodes[permutation[i]] for i in range(n)):
            continue
        transformed = {(permutation[a],permutation[b]):label for (a,b),label in edges.items()}
        if transformed == edges:
            for i,j in enumerate(permutation): expected[i].add(j)
    for i,a in enumerate(order):
        for j,b in enumerate(order):
            assert (orbits[i] == orbits[j]) == (b in expected[a])
    permutation = list(reversed(range(n)))
    renamed_nodes = [nodes[permutation[i]] for i in range(n)]
    renamed_edges = {(permutation[a],permutation[b]):label for (a,b),label in edges.items()}
    renamed, _, _, _ = _canonical_component(renamed_nodes, renamed_edges, tuple(range(n)), 50000)
    assert renamed == code
    return code


def test_exact_pruning_all_1024_five_node_graphs_and_directed_labels():
    pairs = list(itertools.combinations(range(5), 2))
    for bits in range(1 << len(pairs)):
        edges = {}
        for k,(a,b) in enumerate(pairs):
            if bits & (1 << k): edges[a,b] = edges[b,a] = 'single'
        _assert_orbits_from_brute_automorphisms(['same']*5, edges)
    rng = np.random.default_rng(17)
    for _ in range(32):
        nodes = rng.integers(0,2,5).tolist()
        edges = {(a,b):int(rng.integers(1,3)) for a in range(5) for b in range(5) if a!=b and rng.random()<.4}
        _assert_orbits_from_brute_automorphisms(nodes, edges)


def test_equal_refinement_colors_do_not_merge_nonisomorphic_graphs():
    k33 = {(a,b):'single' for a in range(6) for b in range(6) if (a<3)!=(b<3)}
    prism_pairs = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]
    prism = {(a,b):'single' for i,j in prism_pairs for a,b in ((i,j),(j,i))}
    assert _assert_orbits_from_brute_automorphisms(['same']*6,k33) != _assert_orbits_from_brute_automorphisms(['same']*6,prism)


def test_text_checksum_accepts_only_line_ending_conversion(tmp_path):
    lf = b'class Reference:\n    value = 1\n'
    expected = hashlib.sha256(lf).hexdigest()
    file = tmp_path/'reference.py'
    for payload in (lf, lf.replace(b'\n',b'\r\n')):
        file.write_bytes(payload)
        identity = text_source_identity(file, expected)
        assert identity['lf_sha256'] == expected
        assert identity['raw_sha256'] == hashlib.sha256(payload).hexdigest()
    for payload in (lf.replace(b'1',b'2'), lf.replace(b'1',b'1 '), lf.replace(b'\n',b'\r')):
        file.write_bytes(payload)
        with pytest.raises(ValueError, match='checksum mismatch'):
            text_source_identity(file, expected)
