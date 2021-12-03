import numpy as np
from polydata.math.linalg.frame import ReferenceFrame


class Vector(object):

    def __init__(self, array=None, frame=None, *args, **kwargs):
        super().__init__()
        try:
            if not isinstance(array, np.ndarray):
                array = np.array(array)
        except Exception:
            array = None
        self._array = array
        self.frame = frame

    @property
    def array(self):
        return self._array

    @property
    def dim(self):
        return len(self._array.shape)

    @property
    def shape(self):
        return self._array.shape

    @property
    def size(self):
        return self._array.size

    def _transform(self, dcm: np.ndarray = None):
        return dcm.T @ self._array

    def orient(self, frame: ReferenceFrame = None):
        if self.frame is not None:
            dcm = self.frame.dcm(target=frame)
        else:
            if frame is not None:
                dcm = frame.dcm()
            else:
                dcm = np.eye(self.dim)
        self._array = self._transform(dcm)
        self.frame = frame
        return self

    def orient_new(self, *args, **kwargs):
        dcm = ReferenceFrame.rotation_matrix(*args, **kwargs)
        cls = type(self)
        return cls(array=self._transform(dcm.T), frame=self.frame)

    def transform_to_frame(self, frame: ReferenceFrame = None):
        return self.orient(frame)

    def rotate(self, *args, **kwargs):
        return self.orient_new(*args, **kwargs)

    def in_frame(self, frame: ReferenceFrame = None):
        """
        Returns components of current vector in the specified frame.
        """
        if self.frame is not None:
            dcm = self.frame.dcm(target=frame)
        else:
            dcm = frame.dcm()
        return self._transform(dcm.T)

    def dot(self, other):
        return np.dot(self._array, other._array)

    def __repr__(self):
        return np.ndarray.__repr__(self._array)

    def __str__(self):
        return np.ndarray.__str__(self._array)


if __name__ == '__main__':

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')

    vA = Vector([1.0, 1.0, 0.0], frame=A)
    vA.transform_to_frame(B)
    vB = vA.rotate('Body', [0, 0, -30*np.pi/180],  'XYZ')
    vB.in_frame(A)
