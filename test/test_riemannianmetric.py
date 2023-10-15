from test_setup import *
from animate.metric import P0Metric
from parameterized import parameterized
import numpy as np
import unittest


class TestMetricSetup(unittest.TestCase):
    r"""
    Unit tests for constructing :class:`RiemannianMetric`\s.
    """

    def test_dimension_error(self):
        with self.assertRaises(ValueError) as cm:
            RiemannianMetric(uniform_mesh(1, 1))
        msg = "Riemannian metric should be 2D or 3D, not 1D."
        self.assertEqual(str(cm.exception), msg)

    def test_mixed_error(self):
        mesh = uniform_mesh(2, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        with self.assertRaises(ValueError) as cm:
            RiemannianMetric(P1_ten * P1_ten)
        msg = "Riemannian metric cannot be built in a mixed space."
        self.assertEqual(str(cm.exception), msg)

    def test_rank_error(self):
        mesh = uniform_mesh(2, 1)
        with self.assertRaises(ValueError) as cm:
            RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2, 2)))
        msg = "Riemannian metric should be matrix-valued, not rank-3 tensor-valued."
        self.assertEqual(str(cm.exception), msg)

    def test_family_error(self):
        mesh = uniform_mesh(2, 1)
        with self.assertRaises(ValueError) as cm:
            RiemannianMetric(TensorFunctionSpace(mesh, "DG", 1))
        msg = (
            "Riemannian metric should be in P1 space, not"
            " '<tensor element with shape (2, 2) of <DG1 on a triangle>>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_degree_error(self):
        mesh = uniform_mesh(2, 1)
        with self.assertRaises(ValueError) as cm:
            RiemannianMetric(TensorFunctionSpace(mesh, "CG", 2))
        msg = (
            "Riemannian metric should be in P1 space, not"
            " '<tensor element with shape (2, 2) of <CG2 on a triangle>>'."
        )
        self.assertEqual(str(cm.exception), msg)


class TestSetParameters(unittest.TestCase):
    """
    Unit tests for the :meth:`set_parameters` method of :class:`RiemannianMetric`.
    """

    def setUp(self):
        self.metric = uniform_metric(uniform_mesh(2))

    def test_defaults(self):
        plex = self.metric._plex
        self.assertAlmostEqual(plex.metricGetMinimumMagnitude(), 1e-30)
        self.assertAlmostEqual(plex.metricGetMaximumMagnitude(), 1e30)

    def test_set_h_max(self):
        hmax = 1.0
        metric = self.metric
        metric.set_parameters({"dm_plex_metric_h_max": hmax})
        self.assertAlmostEqual(metric._plex.metricGetMaximumMagnitude(), hmax)
        self.assertAlmostEqual(metric.metric_parameters["dm_plex_metric_h_max"], hmax)

    def test_set_h_min(self):
        hmin = 0.1
        metric = self.metric
        metric.set_parameters({"dm_plex_metric_h_min": hmin})
        self.assertAlmostEqual(metric._plex.metricGetMinimumMagnitude(), hmin)
        self.assertAlmostEqual(metric.metric_parameters["dm_plex_metric_h_min"], hmin)

    def test_no_reset(self):
        hmin = 0.1
        hmax = 1.0
        metric = self.metric
        metric.set_parameters({"dm_plex_metric_h_max": hmax})
        metric.set_parameters({"dm_plex_metric_h_min": hmin})
        self.assertAlmostEqual(metric._plex.metricGetMaximumMagnitude(), hmax)
        self.assertAlmostEqual(metric.metric_parameters["dm_plex_metric_h_max"], hmax)


class TestMetricCombination(unittest.TestCase):
    """
    Unit tests for :class:`RiemannianMetric` combination methods.
    """

    @parameterized.expand([(2, True), (2, False), (3, True), (3, False)])
    def test_combine(self, dim, average):
        mesh = uniform_mesh(dim, 1)
        P1_ten = firedrake.TensorFunctionSpace(mesh, "CG", 1)

        metric1 = uniform_metric(P1_ten, 100.0)
        metric2 = uniform_metric(P1_ten, 20.0)
        metric = RiemannianMetric(P1_ten)
        expected = uniform_metric(P1_ten, 60.0 if average else 100.0)

        metric.assign(metric1)
        metric.combine(metric2, average=average)
        self.assertAlmostEqual(errornorm(metric, expected), 0)

        metric.assign(metric2)
        metric.combine(metric1, average=average)
        self.assertAlmostEqual(errornorm(metric, expected), 0)


class TestMetricDrivers(unittest.TestCase):
    """
    Unit tests for :class:`RiemannianMetric` drivers.
    """

    @staticmethod
    def uniform_indicator(mesh):
        return Function(FunctionSpace(mesh, "DG", 0)).assign(1.0)

    def test_riemannianmetric_space_error(self):
        mesh = uniform_mesh(2, 1)
        with self.assertRaises(ValueError) as cm:
            RiemannianMetric(TensorFunctionSpace(mesh, "DG", 0))
        msg = (
            "Riemannian metric should be in P1 space, not"
            " '<tensor element with shape (2, 2) of <DG0 on a triangle>>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_p0metric_space_error(self):
        mesh = uniform_mesh(2, 1)
        with self.assertRaises(ValueError) as cm:
            P0Metric(TensorFunctionSpace(mesh, "CG", 1))
        msg = (
            "P0 metric should be in P0 space, not"
            " '<tensor element with shape (2, 2) of <CG1 on a triangle>>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_isotropic_metric_mesh_error(self):
        mesh1 = uniform_mesh(2, 1, diagonal="left")
        mesh2 = uniform_mesh(2, 1, diagonal="right")
        metric = RiemannianMetric(TensorFunctionSpace(mesh1, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            metric.compute_isotropic_metric(self.uniform_indicator(mesh2))
        msg = "Cannot use an error indicator from a different mesh."
        self.assertEqual(str(cm.exception), msg)

    def test_isotropic_metric_interpolant_error(self):
        mesh = uniform_mesh(2, 1)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        indicator = self.uniform_indicator(mesh)
        with self.assertRaises(ValueError) as cm:
            metric.compute_isotropic_metric(indicator, interpolant="interpolant")
        msg = "Interpolant 'interpolant' not recognised."
        self.assertEqual(str(cm.exception), msg)

    def test_anisotropic_dwr_metric_target_complexity_error(self):
        mesh = uniform_mesh(2, 1)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            metric.compute_anisotropic_dwr_metric(self.uniform_indicator(mesh))
        msg = "Target complexity must be set."
        self.assertEqual(str(cm.exception), msg)

    def test_anisotropic_dwr_metric_mesh_error(self):
        mesh1 = uniform_mesh(2, 1, diagonal="left")
        mesh2 = uniform_mesh(2, 1, diagonal="right")
        metric = RiemannianMetric(TensorFunctionSpace(mesh1, "CG", 1))
        metric.set_parameters({"dm_plex_metric_target_complexity": 1000.0})
        with self.assertRaises(ValueError) as cm:
            metric.compute_anisotropic_dwr_metric(self.uniform_indicator(mesh2))
        msg = "Cannot use an error indicator from a different mesh."
        self.assertEqual(str(cm.exception), msg)

    def test_anisotropic_dwr_metric_convergence_rate_error(self):
        mesh = uniform_mesh(2, 1)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        metric.set_parameters({"dm_plex_metric_target_complexity": 1000.0})
        indicator = Function(FunctionSpace(mesh, "DG", 0)).assign(1.0)
        with self.assertRaises(ValueError) as cm:
            metric.compute_anisotropic_dwr_metric(indicator, convergence_rate=0.999)
        msg = "Convergence rate must be at least one, not 0.999."
        self.assertEqual(str(cm.exception), msg)

    def test_anisotropic_dwr_metric_min_eigenvalue_error(self):
        mesh = uniform_mesh(2, 1)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        metric.set_parameters({"dm_plex_metric_target_complexity": 1000.0})
        indicator = Function(FunctionSpace(mesh, "DG", 0)).assign(1.0)
        with self.assertRaises(ValueError) as cm:
            metric.compute_anisotropic_dwr_metric(indicator, min_eigenvalue=0.0)
        msg = "Minimum eigenvalue must be positive, not 0.0."
        self.assertEqual(str(cm.exception), msg)

    def test_anisotropic_dwr_metric_interpolant_error(self):
        mesh = uniform_mesh(2, 1)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        metric.set_parameters({"dm_plex_metric_target_complexity": 1000.0})
        indicator = self.uniform_indicator(mesh)
        with self.assertRaises(ValueError) as cm:
            metric.compute_anisotropic_dwr_metric(indicator, interpolant="interpolant")
        msg = "Interpolant 'interpolant' not recognised."
        self.assertEqual(str(cm.exception), msg)

    def test_anisotropic_dwr_metric_nan_error(self):
        mesh = uniform_mesh(2, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        metric.set_parameters({"dm_plex_metric_target_complexity": 1.0})
        indicator = self.uniform_indicator(mesh)
        indicator.dat.data[0] = np.nan
        hessian = uniform_metric(P1_ten, 1.0)
        with self.assertRaises(ValueError) as cm:
            metric.compute_anisotropic_dwr_metric(indicator, hessian)
        msg = "K_ratio contains non-finite values."
        self.assertEqual(str(cm.exception), msg)

    def test_weighted_hessian_metric_mesh_error1(self):
        mesh1 = uniform_mesh(2, 1, diagonal="left")
        P1_ten = TensorFunctionSpace(mesh1, "CG", 1)
        mesh2 = uniform_mesh(2, 1, diagonal="right")
        metric = RiemannianMetric(P1_ten)
        indicator = self.uniform_indicator(mesh2)
        hessian = uniform_metric(P1_ten, 1.0)
        with self.assertRaises(ValueError) as cm:
            metric.compute_weighted_hessian_metric(indicator, hessian)
        msg = "Cannot use an error indicator from a different mesh."
        self.assertEqual(str(cm.exception), msg)

    def test_weighted_hessian_metric_mesh_error2(self):
        mesh1 = uniform_mesh(2, 1, diagonal="left")
        mesh2 = uniform_mesh(2, 1, diagonal="right")
        metric = RiemannianMetric(TensorFunctionSpace(mesh1, "CG", 1))
        indicator = self.uniform_indicator(mesh1)
        hessian = uniform_metric(TensorFunctionSpace(mesh2, "CG", 1), 1.0)
        with self.assertRaises(ValueError) as cm:
            metric.compute_weighted_hessian_metric(indicator, hessian)
        msg = "Cannot use a Hessian from a different mesh."
        self.assertEqual(str(cm.exception), msg)

    def test_weighted_hessian_type_error(self):
        mesh = uniform_mesh(2, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        indicator = self.uniform_indicator(mesh)
        hessian = interpolate(Identity(2), P1_ten)
        with self.assertRaises(TypeError) as cm:
            metric.compute_weighted_hessian_metric(indicator, hessian)
        msg = (
            "Expected Hessian to be a RiemannianMetric, not"
            " <class 'firedrake.function.Function'>."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_weighted_hessian_interpolant_error(self):
        mesh = uniform_mesh(2, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        indicator = self.uniform_indicator(mesh)
        hessian = uniform_metric(P1_ten, 1.0)
        with self.assertRaises(ValueError) as cm:
            metric.compute_weighted_hessian_metric(
                indicator, hessian, interpolant="interpolant"
            )
        msg = "Interpolant 'interpolant' not recognised."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([(2, "Clement"), (2, "L2"), (3, "Clement"), (3, "L2")])
    def test_uniform_isotropic_metric(self, dim, interpolant):
        mesh = uniform_mesh(dim, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        indicator = self.uniform_indicator(mesh)
        metric.compute_isotropic_metric(indicator, interpolant=interpolant)
        expected = uniform_metric(P1_ten, 1.0)
        self.assertAlmostEqual(errornorm(metric, expected), 0)

    @parameterized.expand([(2, "Clement"), (2, "L2"), (3, "Clement"), (3, "L2")])
    def test_uniform_isotropic_dwr_metric(self, dim, interpolant):
        mesh = uniform_mesh(dim, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        metric.set_parameters({"dm_plex_metric_target_complexity": 1.0})
        indicator = self.uniform_indicator(mesh)
        metric.compute_isotropic_dwr_metric(indicator, interpolant=interpolant)
        expected = uniform_metric(P1_ten, 1.0)
        self.assertAlmostEqual(errornorm(metric, expected), 0)

    @parameterized.expand([(2, "Clement"), (2, "L2"), (3, "Clement"), (3, "L2")])
    def test_uniform_anisotropic_dwr_metric(self, dim, interpolant):
        mesh = uniform_mesh(dim, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        metric.set_parameters({"dm_plex_metric_target_complexity": 1.0})
        indicator = self.uniform_indicator(mesh)
        hessian = uniform_metric(P1_ten, 1.0)
        metric.compute_anisotropic_dwr_metric(
            indicator, hessian, interpolant=interpolant
        )
        expected = uniform_metric(P1_ten, 1.0)
        self.assertAlmostEqual(errornorm(metric, expected), 0)

    @parameterized.expand([(2, "Clement"), (2, "L2"), (3, "Clement"), (3, "L2")])
    def test_uniform_weighted_hessian_metric(self, dim, interpolant):
        mesh = uniform_mesh(dim, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        indicators = [self.uniform_indicator(mesh)]
        hessians = [uniform_metric(P1_ten, 1.0)]
        expected = uniform_metric(P1_ten, 1.0)
        metric.compute_weighted_hessian_metric(
            indicators, hessians, interpolant=interpolant
        )
        self.assertAlmostEqual(errornorm(metric, expected), 0)


class TestMetricDecompositions(unittest.TestCase):
    """
    Unit tests for metric decompositions.
    """

    @staticmethod
    def mesh(dim):
        return uniform_mesh(dim, 1)

    @staticmethod
    def metric(mesh):
        return RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))

    def test_assemble_eigendecomposition_evectors_rank2_error(self):
        mesh = self.mesh(2)
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)
        evalues = Function(P1_vec)
        evectors = Function(P1_vec)
        metric = self.metric(mesh)
        with self.assertRaises(ValueError) as cm:
            metric.assemble_eigendecomposition(evectors, evalues)
        msg = "Eigenvector Function should be rank-2, not rank-1."
        self.assertEqual(str(cm.exception), msg)

    def test_assemble_eigendecomposition_evalues_rank1_error(self):
        mesh = self.mesh(2)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        evalues = Function(P1_ten)
        evectors = Function(P1_ten)
        metric = self.metric(mesh)
        with self.assertRaises(ValueError) as cm:
            metric.assemble_eigendecomposition(evectors, evalues)
        msg = "Eigenvalue Function should be rank-1, not rank-2."
        self.assertEqual(str(cm.exception), msg)

    def test_assemble_eigendecomposition_family_error(self):
        mesh = self.mesh(2)
        evalues = Function(VectorFunctionSpace(mesh, "DG", 1))
        evectors = Function(TensorFunctionSpace(mesh, "CG", 1))
        metric = self.metric(mesh)
        with self.assertRaises(ValueError) as cm:
            metric.assemble_eigendecomposition(evectors, evalues)
        msg = (
            "Mismatching finite element families:"
            " 'Lagrange' vs. 'Discontinuous Lagrange'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_assemble_eigendecomposition_degree_error(self):
        mesh = self.mesh(2)
        evalues = Function(VectorFunctionSpace(mesh, "CG", 2))
        evectors = Function(TensorFunctionSpace(mesh, "CG", 1))
        metric = self.metric(mesh)
        with self.assertRaises(ValueError) as cm:
            metric.assemble_eigendecomposition(evectors, evalues)
        msg = "Mismatching finite element space degrees: 1 vs. 2."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([(2, True), (2, False), (3, True), (3, False)])
    def test_eigendecomposition(self, dim, reorder):
        """
        Check decomposition of a metric into its eigenvectors
        and eigenvalues.

          * The eigenvectors should be orthonormal.
          * Applying `compute_eigendecomposition` followed by
            `set_eigendecomposition` should get back the metric.
        """
        mesh = self.mesh(dim)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)

        # Create a simple metric
        metric = RiemannianMetric(P1_ten)
        mat = [[1, 0], [0, 2]] if dim == 2 else [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        metric.interpolate(as_matrix(mat))

        # Extract the eigendecomposition
        evectors, evalues = metric.compute_eigendecomposition(reorder=reorder)

        # Check eigenvectors are orthonormal
        err = Function(P1_ten)
        err.interpolate(dot(evectors, transpose(evectors)) - Identity(dim))
        if not np.isclose(norm(err), 0.0):
            raise ValueError(f"Eigenvectors are not orthonormal: {evectors.dat.data}")

        # Check eigenvalues are in descending order
        if reorder:
            P1 = FunctionSpace(mesh, "CG", 1)
            for i in range(dim - 1):
                f = interpolate(evalues[i], P1)
                f -= interpolate(evalues[i + 1], P1)
                if f.vector().gather().min() < 0.0:
                    raise ValueError(
                        f"Eigenvalues are not in descending order: {evalues.dat.data}"
                    )

        # Reassemble
        metric.assemble_eigendecomposition(evectors, evalues)

        # Check against the expected result
        expected = RiemannianMetric(P1_ten)
        expected.interpolate(as_matrix(mat))
        if not np.isclose(errornorm(metric, expected), 0.0):
            raise ValueError("Reassembled metric does not match.")

    @parameterized.expand([(2, True), (2, False), (3, True), (3, False)])
    def test_density_quotients_decomposition(self, dim, reorder):
        """
        Check decomposition of a metric into its density
        and anisotropy quotients.

        Reassembling should get back the metric.
        """
        mesh = self.mesh(dim)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)

        # Create a simple metric
        metric = RiemannianMetric(P1_ten)
        mat = [[1, 0], [0, 2]] if dim == 2 else [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        metric.interpolate(as_matrix(mat))

        # Extract the eigendecomposition
        density, quotients, evectors = metric.density_and_quotients(reorder=reorder)

        # Check eigenvectors are orthonormal
        err = Function(P1_ten)
        err.interpolate(dot(evectors, transpose(evectors)) - Identity(dim))
        if not np.isclose(norm(err), 0.0):
            raise ValueError(f"Eigenvectors are not orthonormal: {evectors.dat.data}")

        # Reassemble
        rho = pow(density, 2 / dim)
        Qd = [pow(quotients[i], -2 / dim) for i in range(dim)]
        if dim == 2:
            Q = as_matrix([[Qd[0], 0], [0, Qd[1]]])
        else:
            Q = as_matrix([[Qd[0], 0, 0], [0, Qd[1], 0], [0, 0, Qd[2]]])
        metric.interpolate(rho * dot(evectors, dot(Q, transpose(evectors))))

        # Check against the expected result
        expected = RiemannianMetric(P1_ten)
        expected.interpolate(as_matrix(mat))
        if not np.isclose(errornorm(metric, expected), 0.0):
            raise ValueError("Reassembled metric does not match.")
