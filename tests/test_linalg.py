# -*- coding: utf-8 -*-
from hypothesis import given, settings, strategies as st, HealthCheck
import unittest
import numpy as np

from dewloosh.math.array import random_pos_semidef, ispossemidef
from dewloosh.math.linalg import ReferenceFrame, Vector


settings.register_profile(
    "linalg_test",
    max_examples=100,
    deadline=None,  
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)


class TestLinalg(unittest.TestCase):
        
    @given(st.integers(min_value=2, max_value=10))
    def test_random_semipos(self, N):
        """
        Tests the creation of random, positive semi-dfinite matrices.
        """
        assert ispossemidef(random_pos_semidef(N, N))
            
    @given(st.integers(min_value=0, max_value=2), 
           st.floats(min_value=0., max_value=360.))
    @settings(settings.load_profile("linalg_test"))
    def test_tr_vector_1(self, i, a):
        """
        Applies a random rotation of a frame around a random axis
        and tests the transformation of components.
        """
        # the original frame    
        A = ReferenceFrame(dim=3)
        
        # the original vector
        vA = Vector([1., 0., 0.], frame=A)
        
        # random rotation
        amounts = [0., 0., 0.]
        amounts[i] = a * np.pi / 180
        B = A.orient_new('Body', amounts, 'XYZ')
        
        # Image of vA in B
        vB = Vector(vA.show(B), frame=B)
        
        # test if the image of vB in A is the same as vA
        assert np.all(np.isclose(vB.show(A), vA.array))
    
    @given(st.floats(min_value=0., max_value=360.))
    def test_tr_vector_2(self, angle):
        """
        Tests the equivalence of a series of relative transformations
        against an absolute transformation.
        """
        A = ReferenceFrame(dim=3)
        vA = Vector([1.0, 0., 0.0], frame=A)
        B = A.orient_new('Body', [0., 0., 0], 'XYZ')
        N = 3
        dtheta = angle * np.pi / 180 / N
        theta = 0.
        for _ in range(N):
            B.orient('Body', [0., 0., dtheta], 'XYZ')
            vB_rel = Vector(vA.array, frame=B)
            theta += dtheta
            vB_tot = vA.orient_new('Body', [0., 0., theta], 'XYZ')
            assert np.all(np.isclose(vB_rel.show(), vB_tot.show()))

    
if __name__ == "__main__":
    
    unittest.main()