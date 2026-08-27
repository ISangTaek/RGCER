"""Bounded gen_edge_input: fixed [N, N, max_dist, F] shape for long chains."""

import numpy as np
import pytest

import algos
from preprocess_data import (
    _BOUNDED_GEN_EDGE_INPUT,
    _gen_edge_input_bounded,
    get_graph_features_from_smiles,
)
from molecular_features import BOND_FEATURE_NAMES


def test_long_chain_edge_input_shape_is_bounded_to_max_path_distance():
    """§11.4: an alkane whose graph diameter far exceeds max_path_distance."""

    record = get_graph_features_from_smiles("C" * 20, sample_id="probe", max_path_distance=8)
    n = int(record["num_nodes"])
    edge_input = record["edge_input"]

    assert tuple(edge_input.shape) == (n, n, 8, len(BOND_FEATURE_NAMES))
    # Adjacent atoms are one hop away and every (hop, feature) slot for that
    # hop carries the filled zero-based single-bond features...
    adjacent_hop = edge_input[0, 1, 0]
    assert bool((adjacent_hop >= 0).all())
    assert bool((edge_input[0, 1, 1:] < 0).all())
    # ...while the terminal pair at distance 19 fills exactly the eight kept
    # hops (counted per-hop via the first feature column).
    filled_hops = edge_input[n - 1, 0, :, 0]
    assert int((filled_hops >= 0).sum()) == 8
    assert bool((filled_hops[1:] >= 1).all())  # interior nodes of the path


def test_bounded_call_matches_unbounded_diameter_reference():
    """Truncation must equal slicing a full-diameter allocation elementwise."""

    smiles = "C" * 18
    record = get_graph_features_from_smiles(smiles, sample_id="probe", max_path_distance=6)
    n = int(record["num_nodes"])
    spatial = record["spatial_pos"].numpy().astype(np.int64)

    adjacency = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    spatial_fw, path_np = algos.floyd_warshall(np.ascontiguousarray(adjacency))
    np.testing.assert_array_equal(spatial_fw, spatial)

    _, path_full = algos.floyd_warshall(np.ascontiguousarray(adjacency))
    finite_max = int(spatial_fw[spatial_fw < 510].max()) if (spatial_fw < 510).any() else 0
    from preprocess_data import _build_edge_tensors
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    _, _, _, _, edge_matrix = _build_edge_tensors(mol)

    reference = np.asarray(
        algos.gen_edge_input(
            max(6, finite_max),
            np.ascontiguousarray(path_full),
            np.ascontiguousarray(edge_matrix),
        )
    )[:, :, :6, :]
    compact = np.asarray(_gen_edge_input_bounded(6, path_np, edge_matrix, spatial_fw))

    np.testing.assert_array_equal(compact, reference)


@pytest.mark.skipif(
    not _BOUNDED_GEN_EDGE_INPUT,
    reason="algos extension predates BOUNDED_EDGE_INPUT; rebuild required",
)
def test_extension_reports_bounded_gen_edge_input_support():
    assert getattr(algos, "BOUNDED_EDGE_INPUT") == 1
