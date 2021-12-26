# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray as nparray
from numba import njit, prange
from typing import Union
from collections import Iterable
__cache = True


ArrayOrFloat = Union[float, np.ndarray, list]


def itype_of_ftype(dtype):
    name = np.dtype(dtype).name
    if '32' in name:
        return np.int32
    elif '64' in name:
        return np.int64
    else:
        raise TypeError
    

def i32array(*args, **kwargs) -> np.ndarray:
    """NumPy array contructor with built in argument `dtype=np.int32`."""
    return np.array(*args, dtype=np.int32, **kwargs)


@njit(nogil=True, cache=__cache)
def minmax(a: nparray):
    return a.min(), a.max()


def ascont(array, *args, dtype=None, **kwargs):
    if dtype is None:
        dtype = array.dtype
    return np.ascontiguousarray(array, dtype=dtype)


@njit(nogil=True, cache=__cache)
def clip1d(a, a_min, a_max):
    a[a < a_min] = a_min
    a[a > a_max] = a_max
    return a


def atleastnd(a: nparray, n=2):
    shp = a.shape
    nD = len(shp)
    if nD >= n:
        return a
    else:
        newshape = (n - nD) * (1,) + shp
        return np.reshape(a, newshape)


def atleast1d(a: ArrayOrFloat):
    if not isinstance(a, Iterable):
        a = [a,]
    return np.array(a)


def atleast2d(a: nparray):
    return atleastnd(a, 2)


def matrixform(f: np.ndarray):
    size = len(f.shape)
    assert size <= 2, "Input array must be at most 2 dimensional."
    if size == 1:
        nV, nC = len(f), 1
        return f.reshape(nV, nC)
    return f


def atleast3d(a: nparray):
    return atleastnd(a, 3)


def atleast4d(a: nparray):
    return atleastnd(a, 4)


@njit(nogil=True, cache=__cache)
def flatten2dC(a: np.ndarray):
    I, J = a.shape
    res = np.zeros(I * J, dtype=a.dtype)
    ind = 0
    for i in range(I):
        for j in range(J):
            res[ind] = a[i, j]
            ind += 1
    return res


@njit(nogil=True, cache=__cache)
def flatten2dF(a: np.ndarray):
    I, J = a.shape
    res = np.zeros(I * J, dtype=a.dtype)
    ind = 0
    for j in range(J):
        for i in range(I):
            res[ind] = a[i, j]
            ind += 1
    return res


def flatten2d(a: np.ndarray, order: str = 'C'):
    if order == 'C':
        return flatten2dC(a)
    elif order == 'F':
        return flatten2dF(a)


def isfloatarray(a: np.ndarray):
    return np.issubdtype(a.dtype, np.float)


def isintegerarray(a: np.ndarray):
    return np.issubdtype(a.dtype, np.integer)


def isintarray(a: np.ndarray):
    return isintegerarray(a)


def isboolarray(a: np.ndarray):
    return np.issubdtype(a.dtype, np.bool)


def is1dfloatarray(a: np.ndarray):
    return isfloatarray(a) and len(a.shape) == 1


def is1dintarray(a: np.ndarray):
    return isintarray(a) and len(a.shape) == 1


def isposdef(a: np.ndarray):
    return np.all(np.linalg.eigvals(a) > 0)


def issymmetric(a: np.ndarray, tol=1e-8):
    return np.linalg.norm(a-a.T) < tol


def bool_to_float(a: np.ndarray, true=1.0, false=0.0):
    res = np.full(a.shape, false, dtype=float)
    res[a] = true
    return res


def choice(choices, size, probs=None):
    """
    Returns a numpy array, whose elements are selected from
    'choices' under probabilities provided with 'probs' (optionally).

    Example
    -------
    >>> N, p = 10, 0.2
    >>> randomarray([False, True], (N, N), [p, 1-p])
    """
    if probs is None:
        probs = np.full((len(choices),), 1/len(choices))
    return np.random.choice(a=choices, size=size, p=probs)


@njit(nogil=True, parallel=True, cache=__cache)
def repeat(a: np.ndarray, N=1):
    res = np.zeros((N, a.shape[0], a.shape[1]), dtype=a.dtype)
    for i in prange(N):
        res[i, :, :] = a
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat1d(a: np.ndarray, N=1):
    M = a.shape[0]
    res = np.zeros(N * M, dtype=a.dtype)
    for i in prange(N):
        res[i*M : (i+1)*M] = a
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tile(a: np.ndarray, da: np.ndarray, N=1):
    res = np.zeros((N, a.shape[0], a.shape[1]), dtype=a.dtype)
    for i in prange(N):
        res[i, :, :] = a + i*da
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tile1d(a: np.ndarray, da: np.ndarray, N=1):
    M = a.shape[0]
    res = np.zeros(N * M, dtype=a.dtype)
    for i in prange(N):
        res[i*M : (i+1)*M] = a + i*da
    return res


def indices_of_equal_rows(x: np.ndarray, y: np.ndarray, tol=1e-12):
    nX, dX = x.shape
    nY, dY = y.shape
    assert dX == dY, "Input arrays must have identical second dimensions."
    square = nX == nY
    integer = isintegerarray(x) and isintegerarray(y)
    if integer:
        if square:
            inds = np.flatnonzero((x == y).all(1))
            return inds, np.copy(inds)
        else:
            return indices_of_equal_rows_njit(x, y, 0)
    else:
        if square:
            return indices_of_equal_rows_square_njit(x, y, tol)
        else:
            return indices_of_equal_rows_njit(x, y, tol)


@njit(nogil=True, parallel=True, cache=__cache)
def indices_of_equal_rows_njit(x: np.ndarray, y: np.ndarray, tol=1e-12):
    R = np.zeros((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i in prange(R.shape[0]):
        for j in prange(R.shape[1]):
            R[i, j] = np.sum(np.abs(x[i] - y[j]))
    return np.where(R <= tol)


@njit(nogil=True, parallel=True, cache=__cache)
def indices_of_equal_rows_square_njit(x: np.ndarray, y: np.ndarray,
                                      tol=1e-12):
    n = np.min([x.shape[0], y.shape[0]])
    R = np.zeros((n, n), dtype=x.dtype)
    for i in prange(n):
        for j in prange(n):
            R[i, j] = np.sum(np.abs(x[i] - y[j]))
    return np.where(R <= tol)


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def count_cols(arr: np.ndarray):
    n = len(arr)
    res = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        res[i] = len(arr[i])
    return res


# !FIXME : assumes a unique input array
@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def find1d(arr, space):
    res = np.zeros(arr.shape, dtype=np.uint64)
    for i in prange(arr.shape[0]):
        res[i] = np.where(arr[i] == space)[0][0]
    return res


# !TODO generalize this up to n dimensions
# !FIXME : assumes a unique input array
@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def find2d(arr, space):
    res = np.zeros(arr.shape, dtype=np.uint64)
    for i in prange(arr.shape[0]):
        res[i] = find1d(arr[i], space)
    return res