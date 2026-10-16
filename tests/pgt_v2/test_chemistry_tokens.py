from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
import torch

from architecture.graphormer_backbone import MolecularGraphormerBackbone
from molecular_features import PGT_FEATURE_SPEC
from pgt_v2.chemistry import UnsupportedChemistry, graph_from_input, prepare_molecule, resolve_input_record
from pgt_v2.data import PGTCollator, make_backbone

ROOT = Path(__file__).resolve().parents[2]
PARAMS = dict(hidden_dim=24, num_heads=4, num_layers=2, ffn_dim=32, dropout=0.0)


def graph(value, index=0):
    item = prepare_molecule(value)
    return graph_from_input(item, sample_id=f"row_{index}", raw_row_index=index)


def close(left, right, record_property, name):
    record_property(name, float((left - right).abs().max()))
    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("position", range(4))
def test_c01_graph_preservation_under_20_atom_permutations(c01_cases, position):
    case = c01_cases[position]
    for field in ("raw_smiles_upper", "selected_smiles_lower"):
        source = Chem.MolFromSmiles(case[field])
        reference = prepare_molecule(source)
        features = graph_from_input(reference, sample_id=case["record_id"], raw_row_index=case["raw_row_index"])
        for seed in range(20):
            permutation = np.random.default_rng(seed).permutation(source.GetNumAtoms()).tolist()
            changed = prepare_molecule(Chem.RenumberAtoms(source, permutation))
            assert changed.model_input_id == reference.model_input_id
            assert changed.nonstereo_id == reference.nonstereo_id
            result = graph_from_input(changed, sample_id=case["record_id"], raw_row_index=case["raw_row_index"])
            for name in ("x", "attn_edge_type", "spatial_pos", "in_degree", "out_degree"):
                assert torch.equal(getattr(features, name), getattr(result, name))
            assert changed.symmetry_classes == reference.symmetry_classes
            assert all(changed.model_to_raw[changed.raw_to_model[r]] == r for r in range(source.GetNumAtoms()))


def test_real_stereo_override_and_source_identifier_collision(c01_cases):
    a, b, c, d = c01_cases
    assert prepare_molecule(a["raw_smiles_upper"]).model_input_id != prepare_molecule(a["selected_smiles_lower"]).model_input_id
    assert prepare_molecule(a["raw_smiles_upper"]).nonstereo_id == prepare_molecule(a["selected_smiles_lower"]).nonstereo_id
    assert prepare_molecule(b["raw_smiles_upper"]).model_input_id == prepare_molecule(b["selected_smiles_lower"]).model_input_id
    assert c["source_inchikey"] == d["source_inchikey"]
    assert prepare_molecule(c["raw_smiles_upper"]).nonstereo_id != prepare_molecule(d["raw_smiles_upper"]).nonstereo_id
    overrides = json.loads((ROOT / "configs/pgt_v2/input_overrides.json").read_text(encoding="utf-8"))
    sha = overrides["row_12241"]["source_sha256"]
    record = {**a, "selected_input_smiles": a["selected_smiles_lower"]}
    with pytest.raises(UnsupportedChemistry, match="Unreviewed"):
        resolve_input_record(record, source_sha256=sha, overrides={})
    mol, provenance = resolve_input_record(record, source_sha256=sha, overrides=overrides)
    assert mol.model_input_id == prepare_molecule(a["raw_smiles_upper"]).model_input_id
    assert "does not recover" in provenance["uncertainty"]
    with pytest.raises(ValueError, match="override identity"):
        resolve_input_record(record, source_sha256="0" * 64, overrides=overrides)
    with pytest.raises(ValueError, match="override identity"):
        resolve_input_record({**record, "TAID": "wrong"}, source_sha256=sha, overrides=overrides)


def test_absolute_cip_equivalent_smiles_and_enantiomers(record_property):
    a, b = graph("N[C@@H](C)C(=O)O"), graph("C[C@H](N)C(=O)O")
    opposite = graph("C[C@@H](N)C(=O)O")
    assert a.model_input_id == b.model_input_id
    assert torch.equal(a.x, b.x)
    assert a.model_input_id != opposite.model_input_id
    assert a.nonstereo_id == opposite.nonstereo_id
    stereo_col = PGT_FEATURE_SPEC.atom_names.index("absolute_stereochemistry")
    assert not torch.equal(a.x[:, stereo_col], opposite.x[:, stereo_col])
    torch.manual_seed(51)
    model = make_backbone(**PARAMS).eval()
    with torch.no_grad():
        left = model.encode_tokens(PGTCollator()([a]))
        right = model.encode_tokens(PGTCollator()([b]))
    close(left.graph_token, right.graph_token, record_property, "equivalent_smiles_graph_max_error")
    close(left.atom_tokens, right.atom_tokens, record_property, "equivalent_smiles_atoms_max_error")


def test_unknown_stereo_is_not_none_and_unsupported_is_not_silently_encoded():
    unassigned = prepare_molecule("CC(N)C(=O)O")
    assert "UNASSIGNED" in unassigned.atom_stereo
    assert set(prepare_molecule("CC").atom_stereo) == {"NONE"}
    assert "UNASSIGNED" in prepare_molecule("FC=CF").bond_stereo
    assert "E" in prepare_molecule("F/C=C/F").bond_stereo
    with pytest.raises(UnsupportedChemistry, match="Unsupported"):
        prepare_molecule("Cl[Pt@SP1](Cl)(N)N")
    vocab = PGT_FEATURE_SPEC.metadata()["atom_vocabs"]["absolute_stereochemistry"]
    assert len({vocab[x] for x in ("R", "S", "r", "s", "NONE", "UNASSIGNED", "UNSUPPORTED")}) == 7
    assert prepare_molecule("[13CH3]C").model_input_id != prepare_molecule("CC").model_input_id


@pytest.mark.parametrize("smiles", ["C1CCCCC1", "c1ccccc1", "c1ccc2ccccc2c1", "CC(=O)[O-].[Na+]"])
def test_full_pipeline_and_tensor_permutation_equivariance(smiles, record_property):
    original = Chem.MolFromSmiles(smiles)
    permutation = list(reversed(range(original.GetNumAtoms())))
    alternate_smiles = Chem.MolToSmiles(Chem.RenumberAtoms(original, permutation), canonical=False, isomericSmiles=True)
    a, b = graph(smiles), graph(alternate_smiles)
    assert a.model_input_id == b.model_input_id
    torch.manual_seed(42)
    model = make_backbone(**PARAMS).eval()
    batch = PGTCollator()([a])
    permuted = batch.clone()
    p = torch.tensor(permutation)
    for name in ("x", "in_degree", "out_degree", "padding_mask", "node_mask"):
        setattr(permuted, name, getattr(batch, name)[:, p])
    for name in ("spatial_pos", "attn_edge_type", "edge_input"):
        setattr(permuted, name, getattr(batch, name)[:, p][:, :, p])
    token_order = torch.cat([torch.zeros(1, dtype=torch.long), p + 1])
    permuted.attn_bias = batch.attn_bias[:, token_order][:, :, token_order]
    with torch.no_grad():
        left = model.encode_tokens(batch, return_attention=True)
        reordered = model.encode_tokens(permuted, return_attention=True)
        rebuilt = model.encode_tokens(PGTCollator()([b]))
        original_bias = model._encode_attention_bias(batch, len(p))
        changed_bias = model._encode_attention_bias(permuted, len(p))
    assert torch.equal(original_bias[:, :, token_order][:, :, :, token_order], changed_bias)
    close(left.graph_token, reordered.graph_token, record_property, "permutation_graph_max_error")
    close(left.atom_tokens[:, p], reordered.atom_tokens, record_property, "permutation_atoms_max_error")
    close(left.graph_token, rebuilt.graph_token, record_property, "rebuilt_smiles_graph_max_error")
    for index, (before, after) in enumerate(zip(left.attentions, reordered.attentions)):
        close(before[:, :, token_order][:, :, :, token_order], after, record_property, f"attention_layer_{index}_max_error")


def test_padding_after_final_norm_and_mixed_batches_full_139_atoms(record_property):
    torch.manual_seed(44)
    model = make_backbone(**PARAMS).eval()
    with torch.no_grad():
        model.final_norm.bias.fill_(0.731)  # catches final LayerNorm reintroducing padding
    small, large = graph("CCO", 1), graph("C" * 139, 2)
    assert large.x.shape[0] == 139
    with torch.no_grad():
        alone = model.encode_tokens(PGTCollator()([small]), return_attention=True)
        together = model.encode_tokens(PGTCollator()([small, large]), return_attention=True)
        old_interface = model(PGTCollator()([small, large]))
    assert together.atom_mask.sum(dim=1).tolist() == [3, 139]
    assert torch.count_nonzero(together.atom_tokens[0, 3:]) == 0
    assert torch.equal(together.graph_token, old_interface)
    for attn in together.attentions:
        assert torch.count_nonzero(attn[0, :, 4:, :]) == 0
        assert torch.count_nonzero(attn[0, :, :, 4:]) == 0
    close(alone.graph_token[0], together.graph_token[0], record_property, "mixed_batch_graph_max_error")
    close(alone.atom_tokens[0], together.atom_tokens[0, :3], record_property, "mixed_batch_atoms_max_error")


@pytest.mark.parametrize("position", range(14))
def test_all_c01_records_above_128_atoms_remain_complete(position, record_property):
    cases = json.loads((Path(__file__).parent / "fixtures/large_structures.json").read_text(encoding="utf-8"))["cases"]
    case = cases[position]
    molecule = prepare_molecule(case["selected_input_smiles"])
    source = Chem.MolFromSmiles(case["selected_input_smiles"])
    reverse = prepare_molecule(Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms())))))
    assert reverse.model_input_id == molecule.model_input_id
    record_property("real_record_id", case["record_id"])
    record_property("canonical_search_nodes", molecule.search_nodes)
    actual = graph_from_input(molecule, sample_id=case["record_id"], raw_row_index=int(case["raw_row_index"]))
    assert actual.num_nodes == int(case["num_atoms"]) > 128
    batch = PGTCollator()([actual, graph("CC", 1)])
    assert batch.node_mask.sum(dim=1).tolist() == [int(case["num_atoms"]), 2]
    torch.manual_seed(88)
    with torch.no_grad():
        result = make_backbone(**PARAMS).eval().encode_tokens(batch)
    assert torch.isfinite(result.atom_tokens).all()
    assert result.atom_mask.sum(dim=1).tolist() == [int(case["num_atoms"]), 2]


def test_symmetry_orbits_and_identical_disconnected_components():
    benzene = prepare_molecule("c1ccccc1")
    assert len(set(benzene.symmetry_classes)) == 1
    ethanol = prepare_molecule("CCO")
    assert len(set(ethanol.symmetry_classes)) == 3
    duplicate = prepare_molecule("CC.CC")
    assert len(set(duplicate.symmetry_classes)) == 1
    with pytest.raises(UnsupportedChemistry, match="exceeded"):
        prepare_molecule("c1ccccc1", search_limit=1)


def test_feature_spec_checkpoint_and_cache_boundaries():
    spec_file = json.loads((ROOT / "configs/pgt_v2/feature_spec.json").read_text(encoding="utf-8"))
    assert spec_file == PGT_FEATURE_SPEC.metadata()
    model = make_backbone(**PARAMS)
    same = make_backbone(**PARAMS)
    same.load_state_dict(model.state_dict())
    legacy = MolecularGraphormerBackbone(**PARAMS)
    with pytest.raises(RuntimeError, match="FeatureSpec"):
        model.load_state_dict(legacy.state_dict(), strict=False)
    with pytest.raises(RuntimeError, match="FeatureSpec"):
        legacy.load_state_dict(same.state_dict(), strict=False)
    fields = list(PGT_FEATURE_SPEC.atom_fields)
    name, tokens = fields[1]
    fields[1] = (name, tuple(reversed(tokens)))  # same tensor shapes, wrong meanings
    different = replace(PGT_FEATURE_SPEC, atom_fields=tuple(fields))
    with pytest.raises(RuntimeError, match="FeatureSpec"):
        make_backbone(feature_spec=different, **PARAMS).load_state_dict(same.state_dict(), strict=False)
    wrong = graph("CC")
    wrong.feature_schema_version = "atom_v2_bond_v1_pathavg_v1"
    with pytest.raises(ValueError, match="FeatureSpec"):
        PGTCollator()([wrong])
    with pytest.raises(ValueError, match="path modes"):
        make_backbone(edge_bias_mode="path", **PARAMS)
    with pytest.raises(ValueError, match="does not silently"):
        PGTCollator()([None])
    assert len(same.path_bond_embeddings) == 0
