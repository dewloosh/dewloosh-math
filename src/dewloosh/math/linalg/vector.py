# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.math.linalg import ReferenceFrame as Frame
from dewloosh.core.typing.array import ArrayBase, Array


__all__ = ['Vector']


class VectorBase(ArrayBase):
    
    def __new__(subtype, shape=None, dtype=float, buffer=None, 
                offset=0, strides=None, order=None, frame=None):
        obj = super().__new__(subtype, shape, dtype, buffer, 
                              offset, strides, order)
        obj.frame = frame
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.frame = getattr(obj, 'frame', None)
    
    
class Vector(Array):
    
    _array_cls_ = VectorBase

    def __init__(self, *args, frame=None, **kwargs):
        cls_params=kwargs.get('cls_params', dict())
        if frame is not None:
            cls_params['frame'] = frame
        kwargs['cls_params'] = cls_params
        super().__init__(*args, **kwargs)
    
    @property
    def array(self) -> VectorBase:
        return self._array
    
    @array.setter
    def array(self, value):
        buf = np.array(value)
        self._array = self._array_cls_(shape=buf.shape, buffer=buf,
                                       dtype=buf.dtype)

    def view(self, frame: Frame=None, *args, **kwargs):
        return self.frame.dcm(target=frame) @ self.array
    
    def orient(self, *args, **kwargs):
        dcm = Frame.ambient().orient_new(*args, **kwargs).dcm()
        self.array = dcm.T @ self._array
        return self
    
    def orient_new(self, *args, keep_frame=True, **kwargs):
        dcm = Frame.ambient().orient_new(*args, **kwargs).dcm()
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
    B = A.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')

    vA = Vector([1.0, 1.0, 0.0], frame=A)
    vB = vA.orient_new('Body', [0, 0, -30*np.pi/180],  'XYZ')
