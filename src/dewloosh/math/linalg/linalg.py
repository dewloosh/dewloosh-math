# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, guvectorize
from numpy import ndarray
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def linspace1d(start, stop, N):
    res = np.zeros(N)
    di = (stop - start) / (N - 1)
    for i in prange(N):
        res[i] = start + i * di
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def vpath(p1: np.ndarray, p2: np.ndarray, n: int):
    nD = len(p1)
    dist = p2 - p1
    length = np.linalg.norm(dist)
    s = np.linspace(0, length, n)
    res = np.zeros((n, nD), dtype=p1.dtype)
    d = dist / length
    for i in prange(n):
        res[i] = p1 + s[i] * d
    return res


@njit(nogil=True, cache=__cache)
def solve(A, b):
    return np.linalg.solve(A, b)


@njit(nogil=True, cache=__cache)
def inv(A: np.ndarray):
    return np.linalg.inv(A)


@njit(nogil=True, parallel=True, cache=__cache)
def _matmul(A: np.ndarray, B: np.ndarray):
    I, K = A.shape
    _, J = B.shape
    res = np.zeros((I, J), dtype=A.dtype)
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                res[i, j] += A[i, k] * B[k, j]
    return res


@njit(nogil=True, cache=__cache)
def matmul(A: np.ndarray, B: np.ndarray):
    return A @ B


@njit(nogil=True, parallel=True, cache=__cache)
def _matvec(A: np.ndarray, b: np.ndarray):
    I, J = A.shape
    res = np.zeros((I), dtype=A.dtype)
    for i in prange(I):
        for j in prange(J):
            res[i] += A[i, j] * b[j]
    return res


@njit(nogil=True, cache=__cache)
def ATB(A: np.ndarray, B: np.ndarray):
    return A.T @ B


@njit(nogil=True, parallel=True, cache=__cache)
def _ATB(A: np.ndarray, B: np.ndarray):
    kk, ii = A.shape
    _, jj = B.shape
    res = np.zeros((ii, jj), dtype=A.dtype)
    for i in prange(ii):
        for j in prange(jj):
            for k in prange(kk):
                res[i, j] += A[k, i] * B[k, j]
    return res


@njit(nogil=True, cache=__cache)
def matmulw(A: np.ndarray, B: np.ndarray, w: float = 1.0):
    return w * (A @ B)


@njit(nogil=True, parallel=True, cache=__cache)
def _matmulw(A: np.ndarray, B: np.ndarray, w: float = 1.0):
    I, K = A.shape
    _, J = B.shape
    res = np.zeros((I, J), A.dtype)
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                res[i, j] += A[i, k] * B[k, j] * w
    return res


@njit(nogil=True, cache=__cache)
def ATBA(A: np.ndarray, B: np.ndarray):
    return A.T @ B @ A


@njit(nogil=True, cache=__cache)
def ATBAw(A: np.ndarray, B: np.ndarray, w: float = 1.0):
    return w * (A.T @ B @ A)


@guvectorize(['(f8[:, :], f8)'], '(n, n) -> ()',
             nopython=True, cache=__cache)
def det3x3(A, res):
    res = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]


@guvectorize(['(f8[:, :], f8)'], '(n, n) -> ()',
             nopython=True, cache=__cache)
def det2x2(A, res):
    res = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@guvectorize(['(f8[:, :], f8[:, :])'], '(n, n) -> (n, n)',
             nopython=True, cache=__cache)
def adj3x3(A, res):
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[0, 0] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@guvectorize(['(f8[:, :], f8[:, :])'], '(n, n) -> (n, n)',
             nopython=True, cache=__cache)
def inv3x3u(A, res):
    det = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[0, 0] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res /= det


@njit(nogil=True, cache=__cache)
def inv3x3(A):
    res = np.zeros_like(A)
    det = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[0, 0] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        det = A[i, 0, 0] * A[i, 1, 1] * A[i, 2, 2] - A[i, 0, 0] * A[i, 1, 2] \
            * A[i, 2, 1] - A[i, 0, 1] * A[i, 1, 0] * A[i, 2, 2] \
            + A[i, 0, 1] * A[i, 1, 2] * A[i, 2, 0] + A[i, 0, 2] \
            * A[i, 1, 0] * A[i, 2, 1] - A[i, 0, 2] * A[i, 1, 1] * A[i, 2, 0]
        res[i, 0, 0] = A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1]
        res[i, 0, 1] = -A[i, 0, 1] * A[i, 2, 2] + A[i, 0, 2] * A[i, 2, 1]
        res[i, 0, 2] = A[i, 0, 1] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 1]
        res[i, 1, 0] = -A[i, 1, 0] * A[i, 2, 2] + A[i, 1, 2] * A[i, 2, 0]
        res[i, 1, 1] = A[i, 0, 0] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 0]
        res[i, 1, 2] = -A[i, 0, 0] * A[i, 1, 2] + A[i, 0, 2] * A[i, 1, 0]
        res[i, 2, 0] = A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0]
        res[i, 2, 1] = -A[i, 0, 0] * A[i, 2, 1] + A[i, 0, 1] * A[i, 2, 0]
        res[i, 0, 0] = A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0]
        res[i] /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk2(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        res[i] = inv3x3(A[i])
    return res


@njit(nogil=True, cache=__cache)
def normalize(A):
    return A/np.linalg.norm(A)


@njit(nogil=True, parallel=True, cache=__cache)
def _to_range(vals: ndarray, source: ndarray, target: ndarray):
    res = np.zeros_like(vals)
    s0, s1 = source
    t0, t1 = target
    b = (t1- t0) / (s1 - s0)
    a = (t0 + t1) / 2 - b * (s0 + s1) / 2
    for i in prange(res.shape[0]):
        res[i] = a + b * vals[i]
    return res


def to_range(vals: ndarray, *args, source: ndarray, target: ndarray=None, 
             squeeze=False, **kwargs):
    if not isinstance(vals, ndarray):
        vals = np.array([vals,])
    source = np.array([0., 1.]) if source is None else np.array(source)
    target = np.array([-1., 1.]) if target is None else np.array(target)
    if squeeze:
        return np.squeeze(_to_range(vals, source, target))
    else:
        return _to_range(vals, source, target)


if __name__ == '__main__':
    from dewloosh.math.array import repeat
    from time import time

    p1 = np.array([0., 0., 0.])
    p2 = np.array([10., 10., 10.])
    path = vpath(p1, p2, 10)

    matrix1 = np.random.rand(20, 20)
    matrix2 = np.random.rand(20, 20)
    _matmul(matrix1, matrix2)
    ATB(matrix1, matrix2)
    ATBA(matrix1, matrix2)

    v1 = np.random.rand(20)
    matmul(matrix1, v1)

    matmul(ATB(matrix1, matrix2), matrix1)
    _matmul(ATB(matrix1, matrix2), matrix1)

    matrix1 = np.random.rand(20, 40)
    matrix2 = np.random.rand(40, 20)
    _matmul(matrix1, matrix2)

    _matmul(_matmul(np.random.rand(20, 40), np.random.rand(40, 20)),
            np.random.rand(20, 20))

    a = repeat(np.eye(3), 10000)
    inv3x3u(a)
    inv3x3_bulk(a)
    inv3x3_bulk2(a)

    t_ = time()
    for i in range(100):
        inv3x3u(a)
    print(time()-t_)

    t_ = time()
    for i in range(100):
        inv3x3_bulk(a)
    print(time()-t_)

    t_ = time()
    for i in range(100):
        inv3x3_bulk2(a)
    print(time()-t_)
    """
    t_ = time()
    for i in range(100):
        np.linalg.inv(a)
    print(time()-t_)
    """