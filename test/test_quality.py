import unittest

import numpy as np
import pytest
import ufl
from adapt_common.reduction import function_data_sum
from firedrake.function import Function
from firedrake.functionspace import TensorFunctionSpace
from firedrake.norms import errornorm
from parameterized import parameterized
from test_setup import uniform_mesh

from animate.quality import QualityMeasure


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_facet_areas(dim):
    """
    Check that the computation of facet areas
    sums to the expected value for a uniform
    (isotropic) triangular or tetrahedral mesh.
    """
    mesh = uniform_mesh(dim, 1)
    qm = QualityMeasure(mesh)
    f = qm("facet_area")
    expected = 5.414214 if dim == 2 else 10.242641
    assert np.isclose(sum(f.dat.data), expected)


class TestQuality(unittest.TestCase):
    """
    Unit tests for quality metrics.
    """

    def quality(self, name, mesh, **kwargs):
        dim = mesh.topological_dimension
        if name == "metric":
            P1_ten = TensorFunctionSpace(mesh, "CG", 1)
            M = Function(P1_ten).interpolate(ufl.Identity(dim))
            kwargs["metric"] = M
        return QualityMeasure(mesh, **kwargs)(name)

    @parameterized.expand(
        [
            ("min_angle", np.pi / 4),
            ("area", 0.005),
            ("eskew", 1.070796),
            ("aspect_ratio", 1.207107),
            ("scaled_jacobian", 0.707107),
            ("skewness", 0.463648),
            ("metric", 6.928203),
        ]
    )
    def test_uniform_quality_2d(self, measure, expected):
        mesh = uniform_mesh(2, 10)
        q = self.quality(measure, mesh)
        truth = Function(q.function_space()).assign(expected)
        self.assertAlmostEqual(errornorm(truth, q), 0.0, places=6)
        if measure == "area":
            s = function_data_sum(q)
            self.assertAlmostEqual(s, 1.0)

    @parameterized.expand(
        [
            ("min_angle", 0.61547971),
            ("volume", 0.00260417),
            ("eskew", 0.41226017),
            ("aspect_ratio", 1.39384685),
            ("scaled_jacobian", 0.40824829),
            ("metric", 1.25),
        ]
    )
    def test_uniform_quality_3d(self, measure, expected):
        mesh = uniform_mesh(3, 4)
        q = self.quality(measure, mesh)
        truth = Function(q.function_space()).assign(expected)
        self.assertAlmostEqual(errornorm(truth, q), 0.0)
        if measure == "volume":
            s = function_data_sum(q)
            self.assertAlmostEqual(s, 1.0)

    @parameterized.expand(
        [
            ("min_angle", 2),
            ("min_angle", 3),
            ("aspect_ratio", 3),
            ("eskew", 2),
            ("eskew", 3),
            ("skewness", 2),
            ("skewness", 3),
            ("scaled_jacobian", 3),
            ("metric", 2),
            ("metric", 3),
        ]
    )
    def test_notimplemented(self, measure, dim):
        mesh = uniform_mesh(dim, 1)
        with self.assertRaises(NotImplementedError) as cm:
            self.quality(measure, mesh)
        msg = f"Quality measure '{measure}' not implemented in the {dim}D case."
        self.assertEqual(str(cm.exception), msg)

    def test_unrecognised_error(self):
        mesh = uniform_mesh(2, 1)
        with self.assertRaises(ValueError) as cm:
            self.quality("invalid", mesh)
        msg = "Quality measure 'invalid' not recognised."
        self.assertEqual(str(cm.exception), msg)
