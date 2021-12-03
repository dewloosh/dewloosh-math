# -*- coding: utf-8 -*-
import numpy as np
from numba.core import types as nbtypes, cgutils
from numba.extending import typeof_impl, models, make_attribute_wrapper, \
    register_model, box, unbox, NativeValue, overload_method
from scipy.sparse import issparse
from scipy.sparse import csr_matrix as csr_scipy
from spdata.utils import get_shape_sp


__all__ = ['csr_matrix']


class csr_matrix(object):
    """
    Numba-jittable python class for a sparse matrix in csr format. 

    Parameters
    ----------
    data    
        Contains the non-zero values of the matrix, in the order in which
        they would be encountered if we walked along the rows left to
        right and top to bottom. If this is a csc matrix, the walk
        happens along the columns.

    indices
        The indices of the columns (rows) during the walk.

    indptr
        Stores row (column) boundaries.

    """

    def __init__(self, data: None, indices: np.array = None,
                 indptr: np.array = None, shape: tuple = None):
        self.sptype = csr_scipy
        if issparse(data):
            data = data.tocsr()
            self.data = data.data.astype(np.float64)
            self.indices = data.indices.astype(np.int32)
            self.indptr = data.indptr.astype(np.int32)
            self.shape = data.shape
        elif isinstance(data, np.ndarray) and indices is None:
            sp = self.csr_scipy(data)
            self.data = sp.data.astype(np.float64)
            self.indices = sp.indices.astype(np.int32)
            self.indptr = sp.indptr.astype(np.int32)
            self.shape = sp.shape
        else:
            self.data = np.array(data).astype(np.float64)
            self.indices = np.array(indices).astype(np.int32)
            self.indptr = np.array(indptr).astype(np.int32)
            if shape is None:
                shape = get_shape_sp(indptr)
            self.shape = shape

    def to_scipy(self):
        return csr_scipy((self.data, self.indices, self.indptr), shape=self.shape)
    
    @staticmethod
    def eye(N: int, *args, **kwargs):
        indices = np.arange(N)
        indptr = np.arange(N+1)
        data = np.ones(N, dtype=float)
        return csr_matrix(data=data, indices=indices, indptr=indptr, shape=(N, N))

    def first_nonzero_col(self):
        return self.indices[self.indptr[:-1]]

    def new_rows_per_col(self):
        return np.bincount(self.first_nonzero_col(), minlength=self.shape[1])

    def row(self, i: int = 0):
        """Returns the values and colum indices of the i-th row."""
        return self.data[self.indptr[i]: self.indptr[i+1]], \
            self.indices[self.indptr[i]: self.indptr[i+1]]


class csr_matrix_nb(nbtypes.Type):
    """Numba type for a sparse matrix."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.data = nbtypes.Array(self.dtype, 1, 'C')
        self.indices = nbtypes.Array(nbtypes.int32, 1, 'C')
        self.indptr = nbtypes.Array(nbtypes.int32, 1, 'C')
        self.shape = nbtypes.UniTuple(nbtypes.int64, 2)
        super(csr_matrix_nb, self).__init__('csr_matrix')


@overload_method(csr_matrix_nb, 'row')
def row(csr, i: int):
    if isinstance(csr, csr_matrix_nb):
        def row_impl(csr, i: int):
            return csr.data[csr.indptr[i]:csr.indptr[i+1]], \
                csr.indices[csr.indptr[i]:csr.indptr[i+1]]
        return row_impl


@typeof_impl.register(csr_matrix)
def typeof_csr(val, c):
    data = typeof_impl(val.data, c)
    return csr_matrix_nb(data.dtype)


make_attribute_wrapper(csr_matrix_nb, 'data', 'data')
make_attribute_wrapper(csr_matrix_nb, 'indices', 'indices')
make_attribute_wrapper(csr_matrix_nb, 'indptr', 'indptr')
make_attribute_wrapper(csr_matrix_nb, 'shape', 'shape')


@register_model(csr_matrix_nb)
class csr_model(models.StructModel):
    """Data model for nopython mode."""

    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data),
            ('indices', fe_type.indices),
            ('indptr', fe_type.indptr),
            ('shape', fe_type.shape)
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@unbox(csr_matrix_nb)
def unbox_csr(typ, obj, c):
    """Convert a python object to a numba-native structure."""
    data = c.pyapi.object_getattr_string(obj, "data")
    indices = c.pyapi.object_getattr_string(obj, "indices")
    indptr = c.pyapi.object_getattr_string(obj, "indptr")
    shape = c.pyapi.object_getattr_string(obj, "shape")
    matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    matrix.data = c.unbox(typ.data, data).value
    matrix.indices = c.unbox(typ.indices, indices).value
    matrix.indptr = c.unbox(typ.indptr, indptr).value
    matrix.shape = c.unbox(typ.shape, shape).value
    for att in [data, indices, indptr, shape]:
        c.pyapi.decref(att)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(matrix._getvalue(), is_error=is_error)


@box(csr_matrix_nb)
def box_csr(typ, val, c):
    """Convert a numba-native structure to a python object."""
    matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder,
                                              value=val)
    classobj = c.pyapi.unserialize(
        c.pyapi.serialize_object(csr_matrix))
    data_obj = c.box(typ.data, matrix.data)
    indices_obj = c.box(typ.indices, matrix.indices)
    indptr_obj = c.box(typ.indptr, matrix.indptr)
    shape_obj = c.box(typ.shape, matrix.shape)
    matrix_obj = c.pyapi.call_function_objargs(classobj,
                                               (data_obj, indices_obj,
                                                indptr_obj, shape_obj))
    return matrix_obj


if __name__ == '__main__':
    from numba import njit

    @njit
    def csr_row(csr: csr_matrix, i: int):
        return csr.row(i)

    @njit
    def csr_m(dtype=np.float64):
        return csr_matrix_nb(dtype)

    np.random.seed = 0
    mat = csr_scipy(np.random.rand(10, 12) > 0.8, dtype=np.int64)
    print(mat.A)

    csr = csr_matrix(mat)
    sdata, scols = csr_row(csr, 0)

    e = csr_matrix.eye(3)

    csrm = csr_m(np.int64)
