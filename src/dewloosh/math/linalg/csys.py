# -*- coding: utf-8 -*-
from dewloosh.math.linalg.frame import ReferenceFrame
import numpy as np


class CoordinateSystem(ReferenceFrame):

    def __init__(self, *args, origo=None, **kwargs):
        super().__init__(*args, **kwargs)
        if origo is None:
            self.origo = np.zeros(self.dim)
        else:
            assert isinstance(origo, np.ndarray)
            assert len(origo.shape) == 1
            assert len(origo) == self.dim
            self.origo = origo
