from rdkit import Chem

from molecular_features import atom_features, bond_features, canonicalize_smiles
from preprocess_data import get_graph_data_from_smiles


def _bond_feature(smiles):
    mol, _ = canonicalize_smiles(smiles)
    return bond_features(next(iter(mol.GetBonds())))


def _double_bond_feature(smiles):
    mol, _ = canonicalize_smiles(smiles)
    return bond_features(next(bond for bond in mol.GetBonds() if str(bond.GetBondType()) == "DOUBLE"))


def test_compact_atom_and_bond_schema():
    graph = get_graph_data_from_smiles("C[C@H](O)F", 1.0)
    assert graph.x.shape[1] == 9
    assert str(graph.x.dtype).endswith("int64")
    assert graph.edge_attr.shape[1] == 4
    assert graph.attn_edge_type.shape[-1] == 4
    assert graph.feature_schema_version == "atom_v2_bond_v1"


def test_bond_order_and_stereo_are_encoded():
    single = _bond_feature("CC")
    double = _bond_feature("C=C")
    assert single[0] != double[0]

    ez = _double_bond_feature("C/C=C/C")
    ze = _double_bond_feature("C/C=C\\C")
    assert ez[1] != ze[1]


def test_chirality_and_formal_charge_are_encoded():
    r_like = Chem.MolFromSmiles("N[C@H](C)C(=O)O")
    s_like = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    r_chiral = [atom_features(atom)[1] for atom in r_like.GetAtoms()]
    s_chiral = [atom_features(atom)[1] for atom in s_like.GetAtoms()]
    assert r_chiral != s_chiral

    neutral = Chem.MolFromSmiles("N")
    charged = Chem.MolFromSmiles("[NH4+]")
    assert atom_features(neutral.GetAtomWithIdx(0))[3] != atom_features(charged.GetAtomWithIdx(0))[3]
