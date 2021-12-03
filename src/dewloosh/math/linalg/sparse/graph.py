# -*- coding: utf-8 -*-
import numpy as np
import networkx as ntx
from numba import jit
from numba.types import int32, int64, Array
from numba.typed import Dict
from spdata.csr import csr_matrix


class Graph(ntx.Graph):

    def adjacency_matrix(self, sparse=False):
        if sparse:
            return csr_matrix(ntx.adjacency_matrix(self))
        else:
            return ntx.adjacency_matrix(self)

    def rooted_level_structure(self, root=0):
        return rooted_level_structure(csr_matrix(ntx.adjacency_matrix(self)), root)

    def pseudo_peripheral_nodes(self):
        return pseudo_peripheral_nodes(csr_matrix(ntx.adjacency_matrix(self)))


int32A = Array(int32, 1, 'C')
int64A = Array(int64, 1, 'C')


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def rooted_level_structure(adj: csr_matrix, root: int = 0) -> Dict:
    """
    Turns a sparse adjacency matrix into a rooted level structure.
    """
    nN = len(adj.indptr) - 1
    rls = Dict.empty(
        key_type=int64,
        value_type=int64A,
    )
    level = 0
    rls[level] = np.array([root], dtype=np.int64)
    nodes = np.zeros(nN, dtype=np.int64)
    nodes[root] = 1
    levelset = np.zeros(nN, dtype=np.int64)
    nE = 1
    while nE < nN:
        levelset[:] = 0
        for node in rls[level]:
            _, neighbours = adj.row(node)
            levelset[neighbours] = 1
        for iN in range(nN):
            if nodes[iN] == 1:
                levelset[iN] = 0
        level += 1
        rls[level] = np.where(levelset == 1)[0]
        nE += len(rls[level])
        for iN in range(nN):
            if levelset[iN] == 1:
                nodes[iN] = 1
    return rls


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def pseudo_peripheral_nodes(adj: csr_matrix):

    def length_width(RLS):
        length = len(RLS)
        width = 0
        for i in range(length):
            width = max(width, len(RLS[i]))
        return length, width

    RLS = rooted_level_structure(adj, root=0)
    length, width = length_width(RLS)
    while True:
        nodes = RLS[len(RLS) - 1]
        found = False
        for iN, node in enumerate(nodes):
            iRLS = rooted_level_structure(adj, root=node)
            iL, iW = length_width(iRLS)
            if (iL > length) or (iL == length and iW < width):
                RLS = iRLS
                length = iL
                width = iW
                found = True
        if not found:
            nR = len(RLS[len(RLS) - 1]) + 1
            res = np.zeros(nR, dtype=np.int32)
            res[:-1] = RLS[len(RLS) - 1]
            res[-1] = RLS[0][0]
            return res

