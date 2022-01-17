# -*- coding: utf-8 -*-
from dewloosh.math.array import random_pos_semidef, ispossemidef
from dewloosh.math.linalg import ReferenceFrame as Frame
from dewloosh.math.linalg.vector import Vector
import numpy as np
from hypothesis import given, strategies as st
import unittest


class TestEncoding(unittest.TestCase):
        
    @given(st.integers(min_value=2, max_value=10))
    def test_random_semipos(self, N):
        assert ispossemidef(random_pos_semidef(N, N))
            
    @given(st.integers(min_value=0, max_value=2), st.floats(min_value=0., max_value=360.))
    def test_tr_vector_1(self, i, a):
        A = Frame(dim=3)
        vA = Vector([1.0, 0., 0.0], frame=A)
        amounts = [0, 0, 0]
        amounts[i] = a*np.pi/180
        B = A.orient_new('Body', amounts,  'XYZ')
        vB = Vector(vA.view(B), frame=B)
        assert np.all(np.isclose(vB.view(), vA))
        
    def test_tr_vector_2(self):
        def test_tr_vector_2(a):
            A = Frame(dim=3)
            vA = Vector([1.0, 0., 0.0], frame=A)
            B = A.orient_new('Body', [0., 0., 0.],  'XYZ')
            N = 3
            dtheta = a*np.pi/180/N
            theta = 0.
            for _ in range(N):
                B = B.orient_new('Body', [0., 0., dtheta],  'XYZ')
                vB_rel_1 = Vector(vA.array, frame=B)
                theta += dtheta
                vB_tot = vA.orient_new('Body', [0., 0., theta],  'XYZ')
                assert np.all(np.isclose(vB_rel_1.view(), vB_tot.view()))
        
        for i in range(5):
            test_tr_vector_2(30*i)

    
if __name__ == "__main__":
    
     
    
    unittest.main()