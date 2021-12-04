# -*- coding: utf-8 -*-
import numpy as np
from sympy.physics.vector import ReferenceFrame as RFrame


class ReferenceFrame:

    def __init__(self, axes=None, parent=None, *args, order='col',
                 name=None, dim=3, **kwargs):
        super().__init__()
        self.name = name
        self.parent = parent
        self.order = order
        if axes is None:
            axes = np.eye(dim)
        else:
            if order == 'row':
                axes = axes.T
        self._axes = axes
        self.dim = dim

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    @property
    def array(self):
        """
        Returns a matrix, where each column is the component array
        of a basis vector with respect to the parent frame, or ambient
        space if there is none.
        """
        return self._axes

    @property
    def dtype(self):
        return self._axes.dtype

    @staticmethod
    def rotation_matrix(*args, **kwargs):
        r"""
        Returns the matrix that rotates the base vectors (dcm matrix).
        Columns of the matrix are the components of the new
        basis vectors in the current frame. To calculate the
        new components of vectors known in the current frame,
        multiply with the transpose of this matrix.

        Parameters
        ----------

        newname : str
            Name for the new reference frame.
        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts :
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3,3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.
        """
        source = RFrame('s')
        target = source.orientnew('t', *args, **kwargs)
        return np.array(source.dcm(target), dtype=float)

    @staticmethod
    def transformation_matrix(*args, **kwargs):
        """
        Returns the matrix that transforms vectors defined in the current
        frame to the same vector with components in the new frame.
        Rows of the matrix are the components of the new
        basis vectors in the current frame, which is the transpose
        of the 'dcm' matrix.
        """
        return ReferenceFrame.rotation_matrix(*args, **kwargs).T

    def dcm(self, *args, target: 'ReferenceFrame' = None,
            source: 'ReferenceFrame' = None, **kwargs):
        """
        Returns the matrix that rotates the base vectors.
        Columns of the matrix are the components of the new
        basis vectors in the current frame. To calculate the
        new components of vectors known in the current frame,
        use the transpose of this matrix.
        """
        if len(args) != 0:
            return ReferenceFrame.rotation_matrix(*args, **kwargs)
        if (source is not None) or (target is not None):
            if target is not None:
                return self.dcm().T @ target.dcm()
            else:
                return source.dcm().T @ self.dcm()
        if self.parent is None:
            return self._axes
        else:
            return self.parent.dcm() @ self._axes

    def rtMatrix(self, *args, **kwargs):
        return self.dcm(*args, **kwargs)

    def trMatrix(self, *args, **kwargs):
        """
        Returns the matrix that transforms global basis vectors into
        the basis vectors of the current frame.
        Therefore, each column is a basis vector of the
        current frame, expressed with coordinates in the global frame.
        Mutiply from the left to get the new components of a vector being
        known in the current frame.
        """
        return self.dcm(*args, **kwargs).T

    def orient(self, frame: 'ReferenceFrame' = None):
        self._axes = self.in_frame(frame=frame)
        self.parent = frame
        return self

    def orient_new(self, *args, name='', **kwargs):
        dcm = self.dcm(*args, **kwargs)
        return ReferenceFrame(axes=dcm, parent=self, name=name)

    def rotate(self, *args, **kwargs):
        return self.orient_new(*args, **kwargs)

    def in_frame(self, frame: 'ReferenceFrame' = None):
        """
        Returns components of current frame in the specified frame.
        """
        return self.dcm(source=frame)

    def __repr__(self):
        return np.ndarray.__repr__(self._axes)

    def __str__(self):
        return np.ndarray.__str__(self._axes)


if __name__ == '__main__':

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
