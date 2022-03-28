# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from sympy.physics.vector import ReferenceFrame as SymPyFrame

from ..array import Array
    

class ReferenceFrame(Array):
    """
    A base reference-frame for orthogonal vector spaces. 
    It facilitates tramsformation of tensor-like quantities across 
    different coordinate frames.
    
    The class is basically an interface on the `ReferenceFrame` class 
    in `sympy.physics.vector`, with a similarly working `orient_new` function.
    
    Examples
    --------
    Define a standard Cartesian frame and rotate it around axis 'Z'
    with an amount of 180 degrees:
    
    >>> A = ReferenceFrame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, np.pi], 'XYZ')
    
    To create a third frame that rotates from B the way B rotates from A, we
    can do
    
    >>> A = ReferenceFrame(dim=3)
    >>> C = A.orient_new('Body', [0, 0, 2*np.pi], 'XYZ')
    
    or we can define it relative to B (this literally makes C to looke 
    in B like B looks in A)
    
    >>> C = ReferenceFrame(B.axes, parent=B)
    
    Notes
    -----
    The `dewloosh.geom.CartesianFrame` class takes the idea of the reference 
    frame a step further by introducing the idea of the 'origo'. 
    
    """
    
    def __init__(self, axes:ndarray=None, parent=None, *args, 
                 order:str='row', name:str=None, dim:int=None, **kwargs):
        order = 'C' if order in ['row', 'C'] else 'F'
        try:
            axes = axes if axes is not None else np.eye(dim)
        except Exception as e:
            if not isinstance(dim, int):
                raise TypeError('If `axes` is `None`, `dim` must be provided as `int`.')
            else:
                raise e        
        super().__init__(axes, *args, order=order, **kwargs)
        self.name = name
        self.parent = parent
        self._order = 0 if order == 'C' else 1
                
    @classmethod
    def eye(cls, *args, dim=3, **kwargs):
        if len(args) > 0 and isinstance(args[0], int):
            dim = args[0]
        return cls(np.eye(dim), *args, **kwargs)
        
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
    
    @axes.setter
    def axes(self, value):
        if isinstance(value, np.ndarray):
            if value.shape == self._array.shape:
                self._array = value
            else:
                raise RuntimeError("Mismatch in data dimensinons!")
        else:
            raise TypeError("Only numpy arras are supported here!")
    
    def show(self, target: 'ReferenceFrame'=None):
        """
        Returns the components of the current frame in a target frame.
        If the target is None, the componants are returned in the ambient frame.
        """
        return self.dcm(target=target)
    
    def dcm(self, *args, target: 'ReferenceFrame'=None,
            source: 'ReferenceFrame'=None, **kwargs):
        """
        Returns the direction cosine matrix (DCM) of a transformation
        from a source (S) to a target (T) frame. The current frame can be the 
        source or the target, depending on the arguments. 
        
        If called without arguments, it returns the DCM matrix from the 
        root frame to the current frame (S=root, T=self).
                
        If `source` is not `None`, then T=self.
        
        If `target` is not `None`, then S=self.
                
        Parameters
        ----------
        args : tuple, Optional
            A tuple of arguments to pass to the `orientnew` 
            function in `sympy`. 
            
        kwargs : dict, Optional
            A dictionary of keyword arguments to pass to the 
            `orientnew` function in `sympy`. 
        
        source : 'ReferenceFrame', Optional
            Source frame. Default is None.

        target : 'ReferenceFrame', Optional
            Target frame. Default is None.

        Returns:
        --------        
        numpy.ndarray
            DCM matrix from S to T.
                    
        """
        if source is not None:
            return self.dcm() @ source.dcm().T
        elif target is not None:
            return target.dcm() @ self.dcm().T        
        # We only get here if the function is called without arguments.
        # The dcm from the ambient frame to the current frame is returned.
        if self.parent is None:
            return self.axes
        else:
            return self.axes @ self.parent.dcm() 

    def orient(self, *args, name='', **kwargs) -> 'ReferenceFrame':
        """
        Orients the current frame inplace. 
        See `Referenceframe.orient_new` for the possible arguments.
                    
        """
        source = SymPyFrame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        self._array = dcm @ self.axes
        return self

    def orient_new(self, *args, name='', **kwargs) -> 'ReferenceFrame':
        """
        Returns a new frame, oriented relative to the called object. 
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
        source = SymPyFrame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        return self.__class__(axes=dcm, parent=self, name=name)


if __name__ == '__main__':

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
