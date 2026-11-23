"""
Test derivative recovery techniques.
"""

import unittest

import ufl
from firedrake.assemble import assemble
from firedrake.functionspace import TensorFunctionSpace
from parameterized import parameterized
from sensors import bowl, mesh_for_sensors

from animate.math import construct_basis
from animate.metric import RiemannianMetric

# ---------------------------
# standard tests for pytest
# ---------------------------


class TestRecoveryBoundary(unittest.TestCase):
    """
    Unit tests for boundary recovery applied to a quadratic 'bowl' sensor.
    """

    def test_unrecognised_boundary_method_error(self):
        mesh = mesh_for_sensors(2, 4)
        f = bowl(*mesh.coordinates)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            metric.compute_boundary_hessian(f, method="some_method")
        msg = (
            "Recovery method 'some_method' not supported for Hessians on the boundary."
        )
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[2]])
    def test_boundary_mixed_L2(self, dim):
        # FIXME: 3D case for test_boundary_mixed_L2
        mesh = mesh_for_sensors(dim, 4)
        f = bowl(*mesh.coordinates)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        metric.compute_boundary_hessian(f, method="mixed_L2")

        # Check its directional derivatives in boundaries are zero
        for s in construct_basis(ufl.FacetNormal(mesh))[1:]:
            dHds = abs(assemble(ufl.dot(ufl.div(metric), s) * ufl.ds))
            self.assertLess(dHds, 5.0e-08)

    @parameterized.expand([[2]])
    def test_boundary_Clement(self, dim):
        mesh = mesh_for_sensors(dim, 20)
        f = bowl(*mesh.coordinates)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        metric.compute_boundary_hessian(f, method="Clement")

        # Check its directional derivatives in boundaries are zero
        for s in construct_basis(ufl.FacetNormal(mesh))[1:]:
            dHds = abs(assemble(ufl.dot(ufl.div(metric), s) * ufl.ds))
            self.assertLess(dHds, 2.0e-08)
