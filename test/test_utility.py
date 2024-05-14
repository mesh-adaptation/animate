import os
import unittest

from firedrake.norms import errornorm as ferrnorm
from firedrake.norms import norm as fnorm
from parameterized import parameterized
from test_setup import *
from test_setup import uniform_mesh

from animate.utility import assemble_mass_matrix

pointwise_norm_types = [["l1"], ["l2"], ["linf"]]
integral_scalar_norm_types = [["L1"], ["L2"], ["L4"], ["H1"], ["HCurl"]]
scalar_norm_types = pointwise_norm_types + integral_scalar_norm_types

# ---------------------------
# standard tests for pytest
# ---------------------------


class TestVTK(unittest.TestCase):
    """
    Test the subclass of Firedrake's :class:`VTKFile`.
    """

    def setUp(self):
        mesh = UnitSquareMesh(1, 1)
        self.fs = FunctionSpace(mesh, "CG", 1)
        pwd = os.path.dirname(__file__)
        self.fname = os.path.join(pwd, "tmp.pvd")

    def tearDown(self):
        fname = os.path.splitext(self.fname)[0]
        for ext in (".pvd", "_0.vtu", "_1.vtu"):
            if os.path.exists(fname + ext):
                os.remove(fname + ext)

    def test_adaptive(self):
        file = VTKFile(self.fname)
        self.assertTrue(os.path.exists(self.fname))
        self.assertTrue(file._adaptive)

    def test_different_fnames(self):
        f = Function(self.fs, name="f")
        g = Function(self.fs, name="g")
        file = VTKFile(self.fname)
        file.write(f)
        file.write(g)
        self.assertEqual("f", g.name())

    def test_different_lengths(self):
        f = Function(self.fs, name="f")
        g = Function(self.fs, name="g")
        file = VTKFile(self.fname)
        file.write(f)
        with self.assertRaises(ValueError) as cm:
            file.write(f, g)
        msg = "Writing different number of functions: expected 1, got 2."
        self.assertEqual(str(cm.exception), msg)


class TestMassMatrix(unittest.TestCase):
    """
    Unit tests for :func:`~.assemble_mass_matrix`.
    """

    @parameterized.expand([("L2",), ("H1",)])
    def test_tiny(self, norm_type):
        mesh = uniform_mesh(2, 1)
        V = FunctionSpace(mesh, "DG", 0)
        matrix = assemble_mass_matrix(V, norm_type=norm_type)
        expected = np.array([[0.5, 0], [0, 0.5]])
        got = matrix.convert("dense").getDenseArray()
        self.assertTrue(np.allclose(expected, got))

    def test_norm_type_error(self):
        mesh = uniform_mesh(2, 1)
        V = FunctionSpace(mesh, "DG", 0)
        with self.assertRaises(ValueError) as cm:
            assemble_mass_matrix(V, norm_type="HDiv")
        self.assertEqual(str(cm.exception), "Norm type 'HDiv' not recognised.")

    def test_lumping(self):
        mesh = UnitTriangleMesh()
        fs = FunctionSpace(mesh, "CG", 1)
        matrix = assemble_mass_matrix(fs, norm_type="L2", lumped=True)
        self.assertEqual(matrix.type, "diagonal")
        expected = np.eye(3) / 6
        got = matrix.convert("dense").getDenseArray()
        self.assertTrue(np.allclose(expected, got))


class TestNorm(unittest.TestCase):
    """
    Unit tests for :func:`norm`.
    """

    def setUp(self):
        self.mesh = uniform_mesh(2, 4)
        self.x, self.y = SpatialCoordinate(self.mesh)
        V = FunctionSpace(self.mesh, "CG", 1)
        self.f = Function(V).interpolate(self.x**2 + self.y)

    def test_boundary_error(self):
        with self.assertRaises(NotImplementedError) as cm:
            norm(self.f, norm_type="l1", boundary=True)
        msg = "lp errors on the boundary not yet implemented."
        self.assertEqual(str(cm.exception), msg)

    def test_l1(self):
        expected = np.sum(np.abs(self.f.dat.data))
        got = norm(self.f, norm_type="l1")
        self.assertAlmostEqual(expected, got)

    def test_l2(self):
        expected = np.sqrt(np.sum(self.f.dat.data**2))
        got = norm(self.f, norm_type="l2")
        self.assertAlmostEqual(expected, got)

    def test_linf(self):
        expected = np.max(self.f.dat.data)
        got = norm(self.f, norm_type="linf")
        self.assertAlmostEqual(expected, got)

    def test_Linf(self):
        expected = np.max(self.f.dat.data)
        got = norm(self.f, norm_type="Linf")
        self.assertAlmostEqual(expected, got)

    def test_notimplemented_lp_error(self):
        with self.assertRaises(NotImplementedError) as cm:
            norm(self.f, norm_type="lp")
        msg = "lp norm of order p not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_L0_error(self):
        with self.assertRaises(ValueError) as cm:
            norm(self.f, norm_type="L0")
        msg = "'L0' norm does not make sense."
        self.assertEqual(str(cm.exception), msg)

    def test_notimplemented_Lp_error(self):
        with self.assertRaises(ValueError) as cm:
            norm(self.f, norm_type="Lp")
        msg = "Unable to interpret 'Lp' norm."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(integral_scalar_norm_types)
    def test_consistency_firedrake(self, norm_type):
        expected = fnorm(self.f, norm_type=norm_type)
        got = norm(self.f, norm_type=norm_type)
        self.assertAlmostEqual(expected, got)

    def test_consistency_hdiv(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        x, y = SpatialCoordinate(self.mesh)
        f = Function(V).interpolate(as_vector([y * y, -x * x]))
        expected = fnorm(f, norm_type="HDiv")
        got = norm(f, norm_type="HDiv")
        self.assertAlmostEqual(expected, got)

    def test_invalid_norm_type_error(self):
        with self.assertRaises(ValueError) as cm:
            norm(self.f, norm_type="X")
        msg = "Unknown norm type 'X'."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([("L1", 0.25), ("L2", 0.5)])
    def test_condition_integral(self, norm_type, expected):
        self.f.assign(1.0)
        condition = conditional(And(self.x < 0.5, self.y < 0.5), 1.0, 0.0)
        val = norm(self.f, norm_type=norm_type, condition=condition)
        self.assertAlmostEqual(val, expected)


class TestErrorNorm(unittest.TestCase):
    """
    Unit tests for :func:`errornorm`.
    """

    def setUp(self):
        self.mesh = uniform_mesh(2, 4)
        self.x, self.y = SpatialCoordinate(self.mesh)
        V = FunctionSpace(self.mesh, "CG", 1)
        self.f = Function(V).interpolate(self.x**2 + self.y)
        self.g = Function(V).interpolate(self.x + self.y**2)

    def test_shape_error(self):
        with self.assertRaises(RuntimeError) as cm:
            errornorm(self.f, self.mesh.coordinates)
        msg = "Mismatching rank between u and uh."
        self.assertEqual(str(cm.exception), msg)

    def test_not_function_error(self):
        with self.assertRaises(TypeError) as cm:
            errornorm(self.f, 1.0)
        msg = "uh should be a Function, is a '<class 'float'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_not_function_error_lp(self):
        with self.assertRaises(TypeError) as cm:
            errornorm(1.0, self.f, norm_type="l1")
        msg = "u should be a Function, is a '<class 'float'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_mixed_space_invalid_norm_error(self):
        V = self.f.function_space() * self.f.function_space()
        with self.assertRaises(NotImplementedError) as cm:
            errornorm(Function(V), Function(V), norm_type="L1")
        msg = "Norm type 'L1' not supported for mixed spaces."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(scalar_norm_types)
    def test_zero_scalar(self, norm_type):
        err = errornorm(self.f, self.f, norm_type=norm_type)
        self.assertAlmostEqual(err, 0.0)

    def test_zero_hdiv(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        x, y = SpatialCoordinate(self.mesh)
        f = Function(V).interpolate(as_vector([y * y, -x * x]))
        err = errornorm(f, f, norm_type="HDiv")
        self.assertAlmostEqual(err, 0.0)

    @parameterized.expand(integral_scalar_norm_types)
    def test_consistency_firedrake(self, norm_type):
        expected = ferrnorm(self.f, self.g, norm_type=norm_type)
        got = errornorm(self.f, self.g, norm_type=norm_type)
        self.assertAlmostEqual(expected, got)

    def test_consistency_hdiv(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        x, y = SpatialCoordinate(self.mesh)
        f = Function(V).interpolate(as_vector([y * y, -x * x]))
        g = Function(V).interpolate(as_vector([x * y, 1.0]))
        expected = ferrnorm(f, g, norm_type="HDiv")
        got = errornorm(f, g, norm_type="HDiv")
        self.assertAlmostEqual(expected, got)

    @parameterized.expand([("L1", 0.25), ("L2", 0.5)])
    def test_condition_integral(self, norm_type, expected):
        self.f.assign(1.0)
        self.g.assign(0.0)
        condition = conditional(And(self.x < 0.5, self.y < 0.5), 1.0, 0.0)
        val = errornorm(self.f, self.g, norm_type=norm_type, condition=condition)
        self.assertAlmostEqual(val, expected)


if __name__ == "__main__":
    unittest.main()
