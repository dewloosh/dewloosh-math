# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, prange

__cache = True


@njit(nogil=True, cache=__cache)
def show_vector(dcm: np.ndarray, arr: np.ndarray):
    return dcm @ arr


@njit(nogil=True, parallel=True, cache=__cache)
def show_vectors(dcm: np.ndarray, arr: np.ndarray):
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm @ arr[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def show_vectors(dcm: np.ndarray, arr: np.ndarray):
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm @ arr[i, :]
    return res
