"""
Test interpolation schemes.
"""

from firedrake import *
from animate.utility import errornorm
from goalie import *
from goalie.interpolation import _transfer_forward, _transfer_adjoint
from animate.utility import function2cofunction
import unittest
from parameterized import parameterized


class TestTransfer(unittest.TestCase):
    """
    Unit tests for mesh-to-mesh projection.
    """

    def setUp(self):
        self.source_mesh = UnitSquareMesh(1, 1, diagonal="left")
        self.target_mesh = UnitSquareMesh(1, 1, diagonal="right")

    def sinusoid(self, source=True):
        x, y = SpatialCoordinate(self.source_mesh if source else self.target_mesh)
        return sin(pi * x) * sin(pi * y)

    def test_method_typo_error(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        with self.assertRaises(ValueError) as cm:
            transfer(Function(Vs), Function(Vt), transfer_method="proj")
        msg = "Invalid transfer method: proj. Options are 'interpolate' and 'project'."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([_transfer_forward, _transfer_adjoint])
    def test_method_typo_error_private(self, private_transfer_func):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        with self.assertRaises(ValueError) as cm:
            private_transfer_func(Function(Vs), Function(Vt), transfer_method="proj")
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
        msg = "Source space has multiple components but target space does not."
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
        msg = "Target space has multiple components but source space does not."
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

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_space(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        source = assemble(interpolate(self.sinusoid(), Vs))
        target = Function(Vs)
        transfer(source, target, transfer_method)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_space_adjoint(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        source = assemble(interpolate(self.sinusoid(), Vs))
        source = function2cofunction(source)
        target = Cofunction(Vs.dual())
        transfer(source, target, transfer_method)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
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
        source = assemble(interpolate(self.sinusoid(), Vs))
        target = Function(Vt)
        transfer(source, target)
        if transfer_method == "interpolate":
            expected = Function(Vt).interpolate(source)
        else:
            expected = Function(Vt).project(source)
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand(["interpolate", "project"])
    def test_transfer_same_mesh_adjoint(self, transfer_method):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.source_mesh, "DG", 0)
        source = assemble(interpolate(self.sinusoid(), Vs))
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
