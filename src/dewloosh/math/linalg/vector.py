# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.math.linalg import ReferenceFrame as Frame
from dewloosh.core.abc.array import ArrayBase, Array
from numba import njit, prange
from dewloosh.math.linalg import ReferenceFrame
__cache = True


__all__ = ['Vector']


@njit(nogil=True, cache=__cache)
def show_vector(dcm: np.ndarray, arr: np.ndarray):
    return dcm @ arr


@njit(nogil=True, parallel=True, cache=__cache)
def show_vectors(dcm: np.ndarray, coords: np.ndarray):
    res = np.zeros_like(coords)
    for i in prange(coords.shape[0]):
        res[i] = dcm @ coords[i, :]
    return res


class VectorBase(ArrayBase):
    
    def __new__(subtype, shape=None, dtype=float, buffer=None, 
                offset=0, strides=None, order=None, frame=None):
        obj = super().__new__(subtype, shape, dtype, buffer, 
                              offset, strides, order)
        obj._frame = frame
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self._frame = getattr(obj, '_frame', None)
    
    @property    
    def frame(self):
        return self._frame
    
    @frame.setter    
    def frame(self, value):
        if isinstance(value, Frame):
            self._frame = value
        else:
            raise TypeError('Value must be a {} instance'.format(Frame))
    
    
class Vector(Array):
    
    _array_cls_ = VectorBase
    _frame_cls_ = Frame

    def __init__(self, *args, frame=None, **kwargs):
        cls_params=kwargs.get('cls_params', dict())
        if frame is not None:
            cls_params['frame'] = frame            
        kwargs['cls_params'] = cls_params
        super().__init__(*args, **kwargs)
        if self._array._frame is None:
            self._array._frame = self._frame_cls_(dim=self._array.shape[0])
    
    @property
    def array(self) -> VectorBase:
        return self._array
    
    @array.setter
    def array(self, value):
        buf = np.array(value)
        self._array = self._array_cls_(shape=buf.shape, buffer=buf,
                                       dtype=buf.dtype)

    def show(self, target: Frame=None, *args, **kwargs):
        target = target if target is not None else \
            ReferenceFrame(dim=self._array.shape[-1])
        if len(self.array.shape) == 1:
            return show_vector(self.frame.dcm(target=target), self.array)
        else:
            return show_vectors(self.frame.dcm(target=target), self.array)
    
    def orient(self, *args, **kwargs):
        dcm = Frame.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
        self.array = dcm.T @ self._array
        return self
    
    def orient_new(self, *args, keep_frame=True, **kwargs):
        dcm = Frame.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
        if keep_frame:
            array = dcm.T @ self._array
            return Vector(array, frame=self.frame)
        else:
            raise NotImplementedError
            
    def __repr__(self):
        return np.ndarray.__repr__(self._array)

    def __str__(self):
        return np.ndarray.__str__(self._array)


if __name__ == '__main__':

    A = Frame()
    B = A.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')

    vA = Vector([1.0, 1.0, 0.0], frame=A)
    vB = vA.orient_new('Body', [0, 0, -30*np.pi/180], 'XYZ')
