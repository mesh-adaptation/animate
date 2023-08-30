"""
Test derivative recovery techniques.
"""
from utility import *
from sensors import bowl, mesh_for_sensors
from parameterized import parameterized
import unittest


# ---------------------------
# standard tests for pytest
# ---------------------------


class TestRecoverySetup(unittest.TestCase):
    """
    Unit tests for derivative recovery.
    """

    def setUp(self):
        self.mesh = mesh_for_sensors(2, 4)
        P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.metric = RiemannianMetric(P1_ten)
        self.expr = bowl(*self.mesh.coordinates)

    def get_func_ones(self, family, degree):
        V = FunctionSpace(self.mesh, family, degree)
        return Function(V).assign(1.0)

    def test_clement_function_error(self):
        f = bowl(*SpatialCoordinate(self.mesh))
        with self.assertRaises(ValueError) as cm:
            self.metric.compute_hessian(f, method="Clement")
        msg = (
            "Clement interpolation can only be used to compute gradients of"
            " Lagrange Functions of degree > 0."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_clement_space_error(self):
        f = self.get_func_ones("RT", 1)
        with self.assertRaises(ValueError) as cm:
            self.metric.compute_hessian(f, method="Clement")
        msg = (
            "Clement interpolation can only be used to compute gradients of"
            " Lagrange Functions of degree > 0."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_clement_degree_error(self):
        f = self.get_func_ones("DG", 0)
        with self.assertRaises(ValueError) as cm:
            self.metric.compute_hessian(f, method="Clement")
        msg = (
            "Clement interpolation can only be used to compute gradients of"
            " Lagrange Functions of degree > 0."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_l2_projection_function_error(self):
        f = bowl(*SpatialCoordinate(self.mesh))
        with self.assertRaises(ValueError) as cm:
            self.metric.compute_hessian(f, method="L2")
        msg = "If a target space is not provided then the input must be a Function."
        self.assertEqual(str(cm.exception), msg)

    def test_l2_projection_rank_error(self):
        f = Function(TensorFunctionSpace(self.mesh, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            self.metric.compute_hessian(f, method="L2")
        msg = (
            "L2 projection can only be used to compute gradients of scalar or vector"
            " Functions, not Functions of rank 2."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_zz_notimplemented_error(self):
        f = self.get_func_ones("CG", 1)
        with self.assertRaises(NotImplementedError) as cm:
            self.metric.compute_hessian(f, method="ZZ")
        msg = "Zienkiewicz-Zhu recovery not yet implemented."
        self.assertEqual(str(cm.exception), msg)

    def test_unrecognised_interior_method_error(self):
        f = self.get_func_ones("CG", 1)
        with self.assertRaises(ValueError) as cm:
            self.metric.compute_hessian(f, method="some_method")
        msg = "Recovery method 'some_method' not recognised."
        self.assertEqual(str(cm.exception), msg)

    def test_unrecognised_boundary_method_error(self):
        f = self.get_func_ones("CG", 1)
        with self.assertRaises(ValueError) as cm:
            self.metric.compute_boundary_hessian(f, method="some_method")
        msg = (
            "Recovery method 'some_method' not supported for Hessians on the boundary."
        )
        self.assertEqual(str(cm.exception), msg)


class TestRecoveryBowl(unittest.TestCase):
    """
    Unit tests for recovery applied to a quadratic 'bowl' sensor.
    """

    @staticmethod
    def metric(mesh):
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        return RiemannianMetric(P1_ten)

    @staticmethod
    def relative_error(approx, ignore_boundary=False):
        mesh = approx.function_space().mesh()
        dim = mesh.topological_dimension()
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        I = interpolate(Identity(dim), P1_ten)

        # Check that they agree
        cond = Constant(1.0)
        if ignore_boundary:
            x = SpatialCoordinate(mesh)
            cond = And(x[0] > -0.8, x[0] < 0.8)
            for i in range(1, dim):
                cond = And(cond, And(x[i] > -0.8, x[i] < 0.8))
            cond = conditional(cond, 1, 0)
        err = errornorm(approx, I, norm_type="L2", condition=cond)
        err /= norm(I, norm_type="L2", condition=cond)
        return err

    @parameterized.expand(
        [
            (2, "L2"),
            (2, "mixed_L2"),
            (3, "L2"),
            (3, "mixed_L2"),
        ]
    )
    def test_interior_L2_quadratic(self, dim, method):
        # TODO: parallel version
        mesh = mesh_for_sensors(dim, 4)
        f = bowl(*mesh.coordinates)
        if method == "L2":
            f = interpolate(f, FunctionSpace(mesh, "CG", 2))
        metric = self.metric(mesh)
        metric.compute_hessian(f, method=method)
        self.assertLess(self.relative_error(metric), 1.0e-07)

    @parameterized.expand([[2], [3]])
    def test_interior_Clement_linear(self, dim):
        mesh = mesh_for_sensors(dim, 20)
        f = interpolate(bowl(*mesh.coordinates), FunctionSpace(mesh, "CG", 1))
        metric = self.metric(mesh)
        metric.compute_hessian(f, method="Clement")
        self.assertLess(self.relative_error(metric, ignore_boundary=True), 1.0e-05)

    @parameterized.expand([[2], [3]])
    def test_interior_Clement_quadratic(self, dim):
        mesh = mesh_for_sensors(dim, 20)
        f = interpolate(bowl(*mesh.coordinates), FunctionSpace(mesh, "CG", 2))
        metric = self.metric(mesh)
        metric.compute_hessian(f, method="Clement")
        self.assertLess(self.relative_error(metric, ignore_boundary=True), 1.0e-08)

    @parameterized.expand([[2]])
    def test_boundary_mixed_L2(self, dim):
        # FIXME: 3D case for test_boundary_mixed_L2
        mesh = mesh_for_sensors(dim, 4)
        f = bowl(*mesh.coordinates)
        metric = self.metric(mesh)
        metric.compute_boundary_hessian(f, method="mixed_L2")

        # Check its directional derivatives in boundaries are zero
        for s in construct_basis(FacetNormal(mesh))[1:]:
            dHds = abs(assemble(dot(div(metric), s) * ds))
            self.assertLess(dHds, 5.0e-08)

    @parameterized.expand([[2]])
    def test_boundary_Clement(self, dim):
        # FIXME: 3D case for test_boundary_Clement
        mesh = mesh_for_sensors(dim, 20)
        f = bowl(*mesh.coordinates)
        metric = self.metric(mesh)
        metric.compute_boundary_hessian(f, method="Clement")

        # Check its directional derivatives in boundaries are zero
        for s in construct_basis(FacetNormal(mesh))[1:]:
            dHds = abs(assemble(dot(div(metric), s) * ds))
            self.assertLess(dHds, 2.0e-08)


# TODO: Implement a difficult recovery test that force use of an LU solver.
