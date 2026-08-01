"""Versioned categorical molecular atom and bond features."""

from __future__ import annotations

from typing import Any, Iterable

from rdkit import Chem


FEATURE_SCHEMA_VERSION = "atom_v2_bond_v1"

ATOM_FEATURE_NAMES = (
    "atomic_number",
    "chirality",
    "total_degree",
    "formal_charge",
    "total_hydrogen_count",
    "radical_electron_count",
    "hybridization",
    "aromaticity",
    "ring_membership",
)

BOND_FEATURE_NAMES = (
    "bond_type",
    "bond_stereo",
    "conjugation",
    "ring_membership",
)


def _enum_token(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    token = str(value).split(".")[-1]
    return token.upper()


def _make_vocab(values: Iterable[Any]) -> dict[str, int]:
    """Return a vocabulary with 0 reserved for padding and 1 for UNK."""
    vocab = {"UNK": 1}
    for value in values:
        token = str(value).upper()
        if token not in vocab:
            vocab[token] = len(vocab) + 1
    return vocab


ATOM_VOCABS = {
    "atomic_number": _make_vocab(range(1, 119)),
    "chirality": _make_vocab(
        (
            "CHI_UNSPECIFIED",
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
            "CHI_OTHER",
        )
    ),
    "total_degree": _make_vocab(range(0, 13)),
    "formal_charge": _make_vocab(range(-5, 6)),
    "total_hydrogen_count": _make_vocab(range(0, 10)),
    "radical_electron_count": _make_vocab(range(0, 7)),
    "hybridization": _make_vocab(
        (
            "UNSPECIFIED",
            "S",
            "SP",
            "SP2",
            "SP3",
            "SP3D",
            "SP3D2",
            "OTHER",
        )
    ),
    "aromaticity": _make_vocab((0, 1)),
    "ring_membership": _make_vocab((0, 1)),
}

BOND_VOCABS = {
    "bond_type": _make_vocab(("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE", "OTHER")),
    "bond_stereo": _make_vocab(
        (
            "STEREONONE",
            "STEREOANY",
            "STEREOZ",
            "STEREOE",
            "STEREOCIS",
            "STEREOTRANS",
            "STEREOATROPOISOMER",
            "OTHER",
        )
    ),
    "conjugation": _make_vocab((0, 1)),
    "ring_membership": _make_vocab((0, 1)),
}

ATOM_CARDINALITIES = tuple(max(ATOM_VOCABS[name].values()) for name in ATOM_FEATURE_NAMES)
BOND_CARDINALITIES = tuple(max(BOND_VOCABS[name].values()) for name in BOND_FEATURE_NAMES)


def _encode(vocab: dict[str, int], value: Any) -> int:
    token = str(value).upper()
    return vocab.get(token, vocab["UNK"])


def atom_features(atom: Chem.Atom) -> list[int]:
    """Encode one atom using the fixed 9-field schema."""
    return [
        _encode(ATOM_VOCABS["atomic_number"], atom.GetAtomicNum()),
        _encode(ATOM_VOCABS["chirality"], _enum_token(atom.GetChiralTag())),
        _encode(ATOM_VOCABS["total_degree"], atom.GetTotalDegree()),
        _encode(ATOM_VOCABS["formal_charge"], atom.GetFormalCharge()),
        _encode(ATOM_VOCABS["total_hydrogen_count"], atom.GetTotalNumHs()),
        _encode(ATOM_VOCABS["radical_electron_count"], atom.GetNumRadicalElectrons()),
        _encode(ATOM_VOCABS["hybridization"], _enum_token(atom.GetHybridization())),
        _encode(ATOM_VOCABS["aromaticity"], int(atom.GetIsAromatic())),
        _encode(ATOM_VOCABS["ring_membership"], int(atom.IsInRing())),
    ]


def bond_features(bond: Chem.Bond) -> list[int]:
    """Encode one chemical bond; the same values are used in both directions."""
    return [
        _encode(BOND_VOCABS["bond_type"], _enum_token(bond.GetBondType())),
        _encode(BOND_VOCABS["bond_stereo"], _enum_token(bond.GetStereo())),
        _encode(BOND_VOCABS["conjugation"], int(bond.GetIsConjugated())),
        _encode(BOND_VOCABS["ring_membership"], int(bond.IsInRing())),
    ]


def canonicalize_smiles(smiles: str) -> tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return mol, canonical


def scaffold_from_mol(mol: Chem.Mol) -> str:
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    return MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "__ACYCLIC__"


def schema_metadata() -> dict[str, object]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "atom_feature_names": list(ATOM_FEATURE_NAMES),
        "bond_feature_names": list(BOND_FEATURE_NAMES),
        "atom_cardinalities": list(ATOM_CARDINALITIES),
        "bond_cardinalities": list(BOND_CARDINALITIES),
    }
