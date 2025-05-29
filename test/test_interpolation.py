"""
Test interpolation schemes.
"""

import unittest

import numpy as np
import pytest
import ufl
from firedrake.assemble import assemble
from firedrake.cofunction import Cofunction
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.functionspace import (
    FunctionSpace,
    TensorFunctionSpace,
    VectorFunctionSpace,
)
from firedrake.norms import errornorm
from firedrake.utility_meshes import UnitSquareMesh
from parameterized import parameterized

from animate.interpolation import (
    _supermesh_project,
    clement_interpolant,
    project,
    transfer,
)
from animate.utility import function2cofunction


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
        self.x, self.y = ufl.SpatialCoordinate(self.mesh)
        self.interior = ufl.conditional(
            ufl.And(
                ufl.And(self.x > h, self.x < 1 - h), ufl.And(self.y > h, self.y < 1 - h)
            ),
            1,
            0,
        )
        self.boundary = 1 - self.interior
        self.corner = ufl.conditional(
            ufl.And(
                ufl.Or(self.x < h, self.x > 1 - h), ufl.Or(self.y < h, self.y > 1 - h)
            ),
            1,
            0,
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
            return ufl.as_vector((self.x, self.y))
        else:
            return ufl.as_matrix([[self.x, self.y], [-self.y, -self.x]])

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
        err = assemble(self.interior * (target - expected) ** 2 * ufl.dx)
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
            self.assertLess(assemble(integrand * ufl.ds(tag)), 5e-3)


class TestTransfer(unittest.TestCase):
    """
    Unit tests for mesh-to-mesh projection.
    """

    def setUp(self):
        self.source_mesh = UnitSquareMesh(4, 4, diagonal="left")
        self.target_mesh = UnitSquareMesh(4, 5, diagonal="right")

    def sinusoid(self, source=True):
        x, y = ufl.SpatialCoordinate(self.source_mesh if source else self.target_mesh)
        return ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)

    def test_method_typo_error(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        with self.assertRaises(ValueError) as cm:
            transfer(Function(Vs), Function(Vt), transfer_method="proj")
        msg = "Invalid transfer method: proj. Options are 'interpolate' and 'project'."
        self.assertEqual(str(cm.exception), msg)

    def test_method_typo_error(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        with self.assertRaises(ValueError) as cm:
            transfer(Function(Vs), Function(Vt), transfer_method="proj")
        msg = "Invalid transfer method: proj. Options are 'interpolate' and 'project'."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["interpolate", "project"])
    def test_notimplemented_error(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        with self.assertRaises(NotImplementedError) as cm:
            transfer(2 * Function(Vs), Vt, transfer_method)
        msg = f"Can only currently {transfer_method} Functions and Cofunctions."
        self.assertEqual(str(cm.exception), msg)

    def test_primal_dual_inconsistency_error(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1).dual()
        with self.assertRaises(ValueError) as cm:
            transfer(Function(Vs), Function(Vt))
        self.assertEqual(str(cm.exception), "Spaces must be both primal or both dual.")

    @parameterized.expand(["interpolate", "project"])
    def test_no_sub_source_space(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vt = Vt * Vt
        with self.assertRaises(ValueError) as cm:
            transfer(Function(Vs), Function(Vt), transfer_method)
        msg = "Target space has multiple components but source space does not."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["interpolate", "project"])
    def test_no_sub_source_space_adjoint(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vt = Vt * Vt
        with self.assertRaises(ValueError) as cm:
            transfer(Cofunction(Vs.dual()), Cofunction(Vt.dual()), transfer_method)
        msg = "Target space has multiple components but source space does not."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["interpolate", "project"])
    def test_no_sub_target_space(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vs = Vs * Vs
        with self.assertRaises(ValueError) as cm:
            transfer(Function(Vs), Function(Vt), transfer_method)
        msg = "Source space has multiple components but target space does not."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["interpolate", "project"])
    def test_no_sub_target_space_adjoint(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vs = Vs * Vs
        with self.assertRaises(ValueError) as cm:
            transfer(Cofunction(Vs.dual()), Cofunction(Vt.dual()), transfer_method)
        msg = "Source space has multiple components but target space does not."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["interpolate", "project"])
    def test_wrong_number_sub_spaces(self, transfer_method):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        P0 = FunctionSpace(self.target_mesh, "DG", 0)
        Vs = P1 * P1 * P1
        Vt = P0 * P0
        with self.assertRaises(ValueError) as cm:
            transfer(Function(Vs), Function(Vt), transfer_method)
        msg = "Inconsistent numbers of components in source and target spaces: 3 vs. 2."
        self.assertEqual(str(cm.exception), msg)

    def test_lumping_space_error(self):
        P0 = FunctionSpace(self.source_mesh, "DG", 0)
        source = Function(P0)
        target = Function(P0)
        with self.assertRaises(ValueError) as cm:
            _supermesh_project(source, target, bounded=True)
        msg = "Mass lumping is not recommended for spaces other than P1."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_space(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        source = Function(Vs).interpolate(self.sinusoid())
        target = Function(Vs)
        transfer(source, target, transfer_method)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_space_adjoint(self, transfer_method):
        pytest.skip()  # TODO: (#114)
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        source = Function(Vs).interpolate(self.sinusoid())
        source = function2cofunction(source)
        target = Cofunction(Vs.dual())
        transfer(source, target, transfer_method)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["project", "interpolate"])
    def test_transfer_same_space_mixed(self, transfer_method):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        Vs = P1 * P1
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Function(Vs)
        transfer(source, target, transfer_method)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_space_mixed_adjoint(self, transfer_method):
        pytest.skip()  # TODO: (#114)
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        Vs = P1 * P1
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        source = function2cofunction(source)
        target = Cofunction(Vs.dual())
        transfer(source, target, transfer_method)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_mesh(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.source_mesh, "DG", 0)
        source = Function(Vs).interpolate(self.sinusoid())
        target = Function(Vt)
        transfer(source, target)
        if transfer_method == "interpolate":
            expected = Function(Vt).interpolate(source)
        else:
            expected = Function(Vt).project(source)
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_mesh_adjoint(self, transfer_method):
        pytest.skip()  # TODO: (#114)
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.source_mesh, "DG", 0)
        source = Function(Vs).interpolate(self.sinusoid())
        target = Cofunction(Vt.dual())
        transfer(function2cofunction(source), target)
        if transfer_method == "interpolate":
            expected = function2cofunction(Function(Vt).interpolate(source))
        else:
            expected = function2cofunction(Function(Vt).project(source))
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_mesh_mixed(self, transfer_method):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        P0 = FunctionSpace(self.source_mesh, "DG", 0)
        Vs = P1 * P1
        Vt = P0 * P0
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Function(Vt)
        transfer(source, target)
        expected = Function(Vt)
        e1, e2 = expected.subfunctions
        if transfer_method == "interpolate":
            e1.interpolate(s1)
            e2.interpolate(s2)
        else:
            e1.project(s1)
            e2.project(s2)
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_mesh_mixed_adjoint(self, transfer_method):
        pytest.skip()  # TODO: (#114)
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        P0 = FunctionSpace(self.source_mesh, "DG", 0)
        Vs = P1 * P1
        Vt = P0 * P0
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Cofunction(Vt.dual())
        transfer(function2cofunction(source), target, transfer_method)
        expected = Function(Vt)
        e1, e2 = expected.subfunctions
        e1.project(s1)
        e2.project(s2)
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @staticmethod
    def check_conservation(source, target, tol=1.0e-08):
        return np.isclose(
            assemble(source * ufl.dx), assemble(target * ufl.dx), atol=tol
        )

    @staticmethod
    def check_no_new_extrema(source, target, tol=1.0e-08):
        return (target.dat.data.max() <= source.dat.data.max() + tol) and (
            target.dat.data.min() >= source.dat.data.min() - tol
        )

    @parameterized.expand([(True, True), (True, False), (False, True), (False, False)])
    def test_supermesh_project(self, same_mesh, same_degree, tol=1.0e-07):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        x, y = ufl.SpatialCoordinate(self.source_mesh)
        source = Function(Vs).interpolate(self.sinusoid())
        target_mesh = self.source_mesh if same_mesh else self.target_mesh
        target_degree = 1 if same_degree else 0
        Vt = FunctionSpace(target_mesh, "DG", target_degree)
        target = Function(Vt)
        _supermesh_project(source, target, bounded=False)
        expected = Function(Vt).project(source)
        self.assertLess(errornorm(target, expected), tol)
        # TODO: The above check should be met at small tolerance; requires the same
        #       implementation as in SupermeshProjector (#123)
        self.assertTrue(self.check_conservation(source, target, tol=tol))

    @parameterized.expand([(True,), (False,)])
    def test_mass_lumping(self, same_mesh, tol=1.0e-08):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        x, y = ufl.SpatialCoordinate(self.source_mesh)
        source = Function(Vs).interpolate(self.sinusoid())
        target_mesh = self.source_mesh if same_mesh else self.target_mesh
        Vt = FunctionSpace(target_mesh, "CG", 1)
        target = Function(Vt)
        project(source, target, bounded=True)
        self.assertTrue(self.check_conservation(source, target, tol=tol))
        self.assertTrue(self.check_no_new_extrema(source, target, tol=tol))
