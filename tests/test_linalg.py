# -*- coding: utf-8 -*-
from hypothesis import given, settings, strategies as st, HealthCheck
import unittest
import numpy as np

from dewloosh.math.array import random_pos_semidef_matrix, random_posdef_matrix, \
    ispossemidef, isposdef
from dewloosh.math.linalg import ReferenceFrame, Vector, inv3x3, det3x3, inv3x3u


settings.register_profile(
    "linalg_test",
    max_examples=100,
    deadline=None,  
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)


class TestTransform(unittest.TestCase):
        
    @given(st.integers(min_value=2, max_value=10))
    def test_random_pos_semidef(self, N):
        """
        Tests the creation of random, positive semidefinite matrices.
        """
        assert ispossemidef(random_pos_semidef_matrix(N))
        
    @given(st.integers(min_value=2, max_value=10))
    def test_random_posdef(self, N):
        """
        Tests the creation of random, positive definite matrices.
        """
        assert isposdef(random_posdef_matrix(N))
            
    @given(st.integers(min_value=0, max_value=2), 
           st.floats(min_value=0., max_value=360.))
    @settings(settings.load_profile("linalg_test"))
    def test_tr_vector_1(self, i, a):
        """
        Applies a random rotation of a frame around a random axis
        and tests the transformation of components.
        """
        pass
        """# the original frame    
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
        assert np.all(np.isclose(vB.show(A), vA.array))"""
    
    @given(st.floats(min_value=0., max_value=360.))
    def test_tr_vector_2(self, angle):
        """
        Tests the equivalence of a series of relative transformations
        against an absolute transformation.
        """
        pass
        """A = ReferenceFrame(dim=3)
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
            assert np.all(np.isclose(vB_rel.show(), vB_tot.show()))"""
            
            
class Test3x3(unittest.TestCase):
        
    @given(st.integers(min_value=1, max_value=10))
    def test_linsolve_3x3(self, shift):
        A = random_posdef_matrix(3) + shift
        b = np.random.rand(3) + shift
        x = np.linalg.solve(A, b)
        x1 = inv3x3(A) @ b
        x2 = inv3x3u(A) @ b
        diff1 = np.abs(x - x1)
        diff2 = np.abs(x - x2)
        err1 = np.dot(diff1, diff1)
        err2 = np.dot(diff2, diff2)
        assert err1 < 1e-12 and err2 < 1e-12
        
    @given(st.integers(min_value=1, max_value=10))
    def test_det_3x3(self, shift):
        A = random_posdef_matrix(3) + shift
        det = np.linalg.det(A)
        det1 = det3x3(A)
        diff1 = np.abs(det - det1)
        assert diff1 < 1e-12

    
if __name__ == "__main__":
        
    unittest.main()