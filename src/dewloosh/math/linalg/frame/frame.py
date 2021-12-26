# -*- coding: utf-8 -*-
import numpy as np
from sympy.physics.vector import ReferenceFrame as Frame
from dewloosh.core.typing.array import Array
    

class ReferenceFrame(Array):
    
    def __init__(self, axes=None, parent=None, *args, 
                 order='row', name=None, origo=None, **kwargs):
        order = 'C' if order in ['row', 'C'] else 'F'
        super().__init__(axes, *args, order=order, **kwargs)
        self.name = name
        self.parent = parent
        self.origo = origo
        self._order = 0 if order == 'C' else 1
    
    @classmethod
    def ambient(cls, *args, dim=3, **kwargs):
        return ReferenceFrame(np.eye(dim), *args, **kwargs)
        
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()
        
    @property
    def order(self):
        return 'row' if self._order == 0 else 'col'

    @property
    def axes(self):
        """
        Returns a matrix, where each row (or column) is the component array
        of a basis vector with respect to the parent frame, or ambient
        space if there is none.
        """
        return self._array
    
    def dcm(self, *args, target: 'ReferenceFrame'=None,
            source: 'ReferenceFrame'=None, **kwargs):
        """
        Returns the direction cosine matrix (DCM) of a transformation
        between this frame and another. The other frame can be the target,
        or the source as well, depending on the arguments.
        
        If the function is called without arguments, it returns the DCM
        matrix of the current object relative to ambient space.
        
        If `source` is not `None`, the function returns the DCM of the
        current frame, relative to the source frame.
        
        If `target` is not `None`, the function returns the DCM of the
        target frame, relative to the current frame.
                
        Parameters
        ----------
        args : tuple, Optional
            A tuple of arguments to pass to the `orientnew` 
            function in `sympy`. 
            
        args : dict, Optional
            A dictionary of keyword arguments to pass to the 
            `orientnew` function in `sympy`. 
        
        source : 'ReferenceFrame', Optional
            Source frame. Default is None.

        target : 'ReferenceFrame', Optional
            Target frame. Default is None.

        Returns:
        --------        
        numpy.ndarray
            DCM matrix.
                    
        """
        if source is not None:
            return self.dcm() @ source.dcm().T
        elif target is not None:
            return target.dcm() @ self.dcm().T        
        # We only get here if the function is called without arguments.
        # The dcm of the current frame relative to the ambient frame 
        # is returned.
        if self.parent is None:
            return self.axes
        else:
            return self.axes @ self.parent.dcm() 

    def orient(self, *args, name='', **kwargs) -> 'ReferenceFrame':
        """
        Orients the current frame inplace. 
        See `Referenceframe.orient_new` for the possible arguments.
                    
        """
        source = Frame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        self._array = dcm @ self.axes
        return self

    def orient_new(self, *args, name='', **kwargs) -> 'ReferenceFrame':
        """
        Returns a new frame, oriented relative to the ccalled object. 
        The orientation can be provided by all ways supported in 
        `sympy.orientnew`.
        
        Parameters
        ==========

        name : str
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
            
        Returns:
        ========      
        ReferenceFrame
            A new ReferenceFrame object.
                   
        """
        source = Frame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        return ReferenceFrame(axes=dcm, parent=self, name=name)


if __name__ == '__main__':

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
