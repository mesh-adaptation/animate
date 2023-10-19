"""
Test interpolation schemes.
"""
from test_setup import *
import numpy as np
from parameterized import parameterized
import unittest


class TestClement(unittest.TestCase):
    """
    Unit tests for Clement interpolant.
    """

    def setUp(self):
        n = 5
        self.mesh = UnitSquareMesh(n, n)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)

        h = 1 / n
        self.x, self.y = SpatialCoordinate(self.mesh)
        self.interior = conditional(
            And(And(self.x > h, self.x < 1 - h), And(self.y > h, self.y < 1 - h)), 1, 0
        )
        self.boundary = 1 - self.interior
        self.corner = conditional(
            And(Or(self.x < h, self.x > 1 - h), Or(self.y < h, self.y > 1 - h)), 1, 0
        )

    def get_space(self, rank, family, degree):
        if rank == 0:
            return FunctionSpace(self.mesh, family, degree)
        elif rank == 1:
            return VectorFunctionSpace(self.mesh, family, degree)
        else:
            shape = tuple(rank * [self.mesh.topological_dimension()])
            return TensorFunctionSpace(self.mesh, family, degree, shape=shape)

    def analytic(self, rank):
        if rank == 0:
            return self.x
        elif rank == 1:
            return as_vector((self.x, self.y))
        else:
            return as_matrix([[self.x, self.y], [-self.y, -self.x]])

    def test_source_type_error(self):
        with self.assertRaises(TypeError) as cm:
            clement_interpolant(Constant(0.0))
        msg = (
            "Expected Cofunction or Function, got"
            " '<class 'firedrake.constant.Constant'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_rank_error(self):
        with self.assertRaises(ValueError) as cm:
            clement_interpolant(Function(self.get_space(3, "DG", 0)))
        msg = "Rank-4 tensors are not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_source_space_error(self):
        with self.assertRaises(ValueError) as cm:
            clement_interpolant(Function(self.get_space(0, "CG", 1)))
        msg = "Source function provided must be from a P0 space."
        self.assertEqual(str(cm.exception), msg)

    def test_target_space_error(self):
        with self.assertRaises(ValueError) as cm:
            clement_interpolant(Function(self.P0), target_space=self.P0)
        msg = "Target space provided must be P1."
        self.assertEqual(str(cm.exception), msg)

    def test_cofunction_dual_target_space(self):
        P0 = self.get_space(0, "DG", 0)
        P1 = self.get_space(0, "CG", 1)
        source = Cofunction(P0.dual())
        source.dat.data_with_halos[:] = 1.0
        target = clement_interpolant(source, target_space=P1.dual())
        self.assertTrue(isinstance(target, Cofunction))
        self.assertTrue(np.allclose(target.dat.data_with_halos, 1.0))

    def test_cofunction_primal_target_space(self):
        P0 = self.get_space(0, "DG", 0)
        P1 = self.get_space(0, "CG", 1)
        source = Cofunction(P0.dual())
        source.dat.data_with_halos[:] = 1.0
        target = clement_interpolant(source, target_space=P1)
        self.assertTrue(isinstance(target, Function))
        self.assertTrue(np.allclose(target.dat.data_with_halos, 1.0))

    @parameterized.expand([[0], [1], [2]])
    def test_volume_average_2d(self, rank):
        exact = self.analytic(rank)
        P0 = self.get_space(rank, "DG", 0)
        P1 = self.get_space(rank, "CG", 1)
        source = Function(P0).project(exact)
        target = clement_interpolant(source)
        expected = Function(P1).interpolate(exact)
        err = assemble(self.interior * (target - expected) ** 2 * dx)
        self.assertAlmostEqual(err, 0)

    @parameterized.expand([[0], [1], [2]])
    def test_facet_average_2d(self, rank):
        exact = self.analytic(rank)
        P0 = self.get_space(rank, "DG", 0)
        source = Function(P0).project(exact)
        target = clement_interpolant(source, boundary=True)
        expected = source
        integrand = (1 - self.corner) * (target - expected) ** 2

        # Check approximate recovery
        for tag in [1, 2, 3, 4]:
            self.assertLess(assemble(integrand * ds(tag)), 5e-3)
