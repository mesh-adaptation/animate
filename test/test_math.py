from test_setup import *
import numpy as np
from parameterized import parameterized
import ufl
import unittest


class TestOrthogonalisation(unittest.TestCase):
    """
    Unit tests for orthogonalisation.
    """

    def setUp(self):
        np.random.seed(0)

    def test_gram_schmidt_type_error_numpy(self):
        with self.assertRaises(TypeError) as cm:
            gram_schmidt(np.ones(2), [1, 2])
        msg = (
            "Inconsistent vector types:"
            " '<class 'numpy.ndarray'>' vs. '<class 'list'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_gram_schmidt_type_error_ufl(self):
        x = ufl.SpatialCoordinate(UnitTriangleMesh())
        with self.assertRaises(TypeError) as cm:
            gram_schmidt(x, [1, 2])
        msg = (
            "Inconsistent vector types:"
            " '<class 'ufl.core.expr.Expr'>' vs. '<class 'list'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[2], [3]])
    def test_gram_schmidt_orthonormal_numpy(self, dim):
        v = np.random.rand(dim, dim)
        u = np.array(gram_schmidt(*v, normalise=True))
        self.assertTrue(np.allclose(u.transpose() @ u, np.eye(dim)))

    @parameterized.expand([[2], [3]])
    def test_gram_schmidt_nonorthonormal_numpy(self, dim):
        v = np.random.rand(dim, dim)
        u = gram_schmidt(*v, normalise=False)
        for i, ui in enumerate(u):
            for j, uj in enumerate(u):
                if i != j:
                    self.assertAlmostEqual(np.dot(ui, uj), 0)

    def test_basis_shape_error_numpy(self):
        with self.assertRaises(ValueError) as cm:
            construct_basis(np.ones((1, 2)))
        msg = "Expected a vector, got an array of shape (1, 2)."
        self.assertEqual(str(cm.exception), msg)

    def test_basis_dim_error_numpy(self):
        with self.assertRaises(ValueError) as cm:
            construct_basis(np.ones(0))
        msg = "Dimension 0 not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_basis_ufl_type_error(self):
        with self.assertRaises(TypeError) as cm:
            construct_basis(UnitTriangleMesh())
        msg = "Expected UFL Expr, not '<class 'firedrake.mesh.MeshGeometry'>'."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[2], [3]])
    def test_basis_orthonormal_numpy(self, dim):
        u = np.array(construct_basis(np.random.rand(dim), normalise=True))
        self.assertTrue(np.allclose(u.transpose() @ u, np.eye(dim)))

    @parameterized.expand([[2], [3]])
    def test_basis_orthogonal_numpy(self, dim):
        u = construct_basis(np.random.rand(dim), normalise=False)
        for i, ui in enumerate(u):
            for j, uj in enumerate(u):
                if i != j:
                    self.assertAlmostEqual(np.dot(ui, uj), 0)
