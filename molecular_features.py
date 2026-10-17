"""Versioned categorical molecular atom and bond features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from rdkit import Chem


FEATURE_SCHEMA_VERSION = "atom_v2_bond_v1_pathavg_v1"

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


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable, content-addressed feature contract, separate from legacy V2.

    Tokens are case-sensitive: pseudoasymmetric r/s must not become R/S.
    Each tuple lists real tokens in index order; 0=padding and 1=unknown.
    """

    version: str
    rdkit_version: str
    atom_fields: tuple[tuple[str, tuple[str, ...]], ...]
    bond_fields: tuple[tuple[str, tuple[str, ...]], ...]
    identity_algorithm: str = "colored_graph_ir_v1"

    def __post_init__(self):
        if not self.version or not self.rdkit_version or self.identity_algorithm != "colored_graph_ir_v1":
            raise ValueError("FeatureSpec requires a version, RDKit pin and supported identity algorithm")
        for fields in (self.atom_fields, self.bond_fields):
            if not isinstance(fields, tuple) or any(not isinstance(f, tuple) or not isinstance(f[1], tuple) for f in fields):
                raise TypeError("FeatureSpec fields/vocabularies must be immutable tuples")
            names = [name for name, _ in fields]
            if not names or len(set(names)) != len(names):
                raise ValueError("Feature names must be nonempty and unique")
            for name, tokens in fields:
                if not tokens or len(set(tokens)) != len(tokens) or {"PAD", "UNK"}.intersection(tokens):
                    raise ValueError(f"Invalid vocabulary for {name}")

    @property
    def atom_names(self):
        return tuple(name for name, _ in self.atom_fields)

    @property
    def bond_names(self):
        return tuple(name for name, _ in self.bond_fields)

    @property
    def atom_cardinalities(self):
        return tuple(len(tokens) + 1 for _, tokens in self.atom_fields)

    @property
    def bond_cardinalities(self):
        return tuple(len(tokens) + 1 for _, tokens in self.bond_fields)

    def _payload(self):
        return {
            "feature_schema_version": self.version,
            "rdkit_version": self.rdkit_version,
            "identity_algorithm": self.identity_algorithm,
            "atom_feature_names": list(self.atom_names),
            "bond_feature_names": list(self.bond_names),
            "atom_cardinalities": list(self.atom_cardinalities),
            "bond_cardinalities": list(self.bond_cardinalities),
            "atom_vocabs": self.vocabularies(self.atom_fields),
            "bond_vocabs": self.vocabularies(self.bond_fields),
            "edge_bias_mode": "direct",
            "unknown_stereo_policy": "explicit unassigned; unsupported assigned stereo raises",
        }

    @staticmethod
    def vocabularies(fields):
        return {name: {"PAD": 0, "UNK": 1, **{t: i + 2 for i, t in enumerate(tokens)}}
                for name, tokens in fields}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def metadata(self):
        return {**self._payload(), "feature_spec_sha256": self.sha256}

    def validate_runtime(self):
        if Chem.rdBase.rdkitVersion != self.rdkit_version:
            raise ValueError(f"RDKit mismatch: spec={self.rdkit_version}, runtime={Chem.rdBase.rdkitVersion}")
        if not Chem.GetAllowNontetrahedralChirality():
            raise ValueError("FeatureSpec requires RDKit non-tetrahedral stereo perception enabled")

    def encode(self, kind, values):
        fields = self.atom_fields if kind == "atom" else self.bond_fields if kind == "bond" else None
        if fields is None or set(values) != {n for n, _ in fields}:
            raise ValueError("Feature values do not match the explicit spec")
        result = []
        for name, tokens in fields:
            token = str(values[name])
            result.append(tokens.index(token) + 2 if token in tokens else 1)
        return result


def _tokens(values):
    return tuple(str(v) for v in values)


# Legacy constants/functions above retain their original values and behaviour.
# New callers must opt in, carry this hash in batches, and initialize new weights.
PGT_FEATURE_SPEC = FeatureSpec(
    version="atom_v4_cip_coord_unknown_bond_v2_direct_v1",
    rdkit_version="2025.09.6",
    atom_fields=(
        ("atomic_number", _tokens(range(1, 119))),
        ("absolute_stereochemistry", ("NONE", "R", "S", "r", "s", "UNASSIGNED", "UNSUPPORTED",
            "UNASSIGNED_SQUARE_PLANAR", "UNASSIGNED_TRIGONAL_BIPYRAMIDAL", "UNASSIGNED_OCTAHEDRAL")),
        ("isotope", _tokens(range(0, 301))),
        ("total_degree", _tokens(range(0, 13))),
        ("formal_charge", _tokens(range(-5, 6))),
        ("total_hydrogen_count", _tokens(range(0, 10))),
        ("radical_electron_count", _tokens(range(0, 7))),
        ("hybridization", ("UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER")),
        ("aromaticity", ("0", "1")),
        ("ring_membership", ("0", "1")),
    ),
    bond_fields=(
        ("bond_type", ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE", "ZERO", "OTHER")),
        ("absolute_stereochemistry", ("NONE", "E", "Z", "UNASSIGNED", "UNSUPPORTED")),
        ("conjugation", ("0", "1")),
        ("ring_membership", ("0", "1")),
        ("dative_direction", ("NONE", "DONOR_TO_ACCEPTOR", "ACCEPTOR_TO_DONOR")),
    ),
)
