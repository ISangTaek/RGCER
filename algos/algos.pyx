# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cython
# from cython.parallel cimport prange, parallel
cimport numpy as cnp
import numpy

# Sentinel for callers: builds compiled from this source bound their
# per-pair write loop at max_dist (see gen_edge_input), so consumers may
# request exactly max_path_distance instead of over-allocating to the
# molecule diameter.  Binaries compiled before the fix lack this attribute.
BOUNDED_EDGE_INPUT = 1

def floyd_warshall(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(numpy.int64, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] M = adj_mat_copy
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] path = -1 * numpy.ones([n, n],dtype = numpy.int64)

    cdef unsigned int i, j, k
    cdef cnp.int64_t M_ij, M_ik, cost_ikkj
    cdef cnp.int64_t* M_ptr = &M[0,0]
    cdef cnp.int64_t* M_i_ptr
    cdef cnp.int64_t* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path


def get_all_edges(path, i, j):
    cdef int k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(numpy.int64, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(numpy.int64, order='C', casting='safe', copy=True)
    # edge_feat_copy = edge_feat
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef cnp.ndarray[cnp.int64_t, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, num_path, path_limit

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            # Only the first max_dist hops fit into edge_fea_all; writing the
            # remaining hops of a longer shortest path used to overflow the
            # buffer (crash/corruption on long chains).  Truncate to max_dist.
            path_limit = num_path
            if path_limit > max_dist_copy:
                path_limit = max_dist_copy
            for k in range(path_limit):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all