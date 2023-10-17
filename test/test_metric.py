from test_setup import *
import pytest
import numpy as np


def bowl(*coords):
    """
    Quadratic bowl sensor function in arbitrary dimensions.
    """
    return 0.5 * sum([xi**2 for xi in coords])


def hyperbolic(x, y):
    """
    Hyperbolic sensor function in 2D.
    """
    sn = sin(50 * x * y)
    return conditional(abs(x * y) < 2 * pi / 50, 0.01 * sn, sn)


def multiscale(x, y):
    """
    Multi-scale sensor function in 2D.
    """
    return 0.1 * sin(50 * x) + atan(0.1 / (sin(5 * y) - 2 * x))


def interweaved(x, y):
    """
    Interweaved sensor function in 2D.
    """
    return atan(0.1 / (sin(5 * y) - 2 * x)) + atan(0.5 / (sin(3 * y) - 7 * x))


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=[1, 2, np.inf])
def degree(request):
    return request.param


def test_hessian_bowl(dim):
    """
    Test that the Hessian recovery technique is able to recover the analytical
    Hessian of a quadratic sensor function.
    """
    mesh = uniform_mesh(dim, 4, recentre=True)
    f = bowl(*mesh.coordinates)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.compute_hessian(f)
    expected = interpolate(Identity(dim), P1_ten)
    err = errornorm(metric, expected) / norm(expected)
    assert err < 1.0e-07


@pytest.mark.parallel(nprocs=2)
def test_hessian_bowl_np2():
    """
    Test that the Hessian recovery technique works in parallel.
    """
    test_hessian_bowl(3)


def test_symmetric():
    """
    Test that a metric is indeed symmetric.
    """
    mesh = uniform_mesh(2, 4, recentre=True)
    f = hyperbolic(*SpatialCoordinate(mesh))
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.compute_hessian(f)
    metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
    err = assemble(abs(det(metric - transpose(metric))) * dx) / assemble(
        abs(det(metric)) * dx
    )
    assert err < 1.0e-08


@pytest.mark.parallel(nprocs=2)
def test_normalise(dim):
    """
    Test that normalising a metric w.r.t.
    a given metric complexity and the
    normalisation order :math:`p=1` DTRT.
    """
    mesh = uniform_mesh(dim)
    target = 200.0 if dim == 2 else 2500.0
    mp = {
        "dm_plex_metric": {
            "target_complexity": target,
            "normalization_order": 1.0,
        }
    }
    metric = uniform_metric(mesh, metric_parameters=mp)
    metric.normalise()
    expected = uniform_metric(mesh, a=pow(target, 2.0 / dim))
    assert np.isclose(errornorm(metric, expected), 0.0)


def test_hessian_normalise(sensor, degree):
    """
    Test that normalising a metric enables the attainment of the target metric
    complexity.

    Note that we should only expect this to be true if the underlying mesh is
    unit w.r.t. the metric.
    """
    dim = 2
    target = 1000.0
    metric_parameters = {
        "dm_plex_metric": {
            "target_complexity": target,
            "normalization_order": degree,
        }
    }

    # Construct a normalised Hessian metric
    mesh = uniform_mesh(dim, 100, recentre=True)
    f = sensor(*mesh.coordinates)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(metric_parameters)
    metric.compute_hessian(f)
    metric.normalise(restrict_sizes=False, restrict_anisotropy=False)

    # Check that the target metric complexity is (approximately) attained
    assert abs(metric.complexity() - target) < 0.1 * target


@pytest.mark.parallel(nprocs=2)
def test_intersection(dim):
    """
    Test that intersecting two metrics results
    in a metric with the minimal ellipsoid.
    """
    mesh = uniform_mesh(dim)
    metric1 = uniform_metric(mesh, a=100.0)
    metric2 = uniform_metric(mesh, a=25.0)
    expected = uniform_metric(mesh, a=100.0)
    metric_int = RiemannianMetric(metric1.function_space())

    metric_int.assign(metric1)
    metric_int.intersect(metric2)
    assert np.isclose(errornorm(metric_int, expected), 0.0)

    metric_int.assign(metric2)
    metric_int.intersect(metric1)
    assert np.isclose(errornorm(metric_int, expected), 0.0)


def test_average(dim):
    mesh = uniform_mesh(dim, 1)
    x = SpatialCoordinate(mesh)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric1 = RiemannianMetric(P1_ten)
    metric2 = RiemannianMetric(P1_ten)
    if dim == 2:
        mat1 = [[2 + x[0], 0], [0, 2 + x[1]]]
        mat2 = [[2 - x[0], 0], [0, 2 - x[1]]]
    else:
        mat1 = [[2 + x[0], 0, 0], [0, 2 + x[1], 0], [0, 0, 2 + x[2]]]
        mat2 = [[2 - x[0], 0, 0], [0, 2 - x[1], 0], [0, 0, 2 - x[2]]]
    metric1.interpolate(as_matrix(mat1))
    metric2.interpolate(as_matrix(mat2))

    metric_avg = metric1.copy(deepcopy=True)
    metric_avg.average(metric1, metric1)
    assert np.isclose(errornorm(metric_avg, metric1), 0.0)

    metric_avg.average(metric2)
    expected = uniform_metric(mesh, a=2.0)
    assert np.isclose(errornorm(metric_avg, expected), 0.0)


def test_complexity(dim):
    mesh = uniform_mesh(dim, 1)
    x = SpatialCoordinate(mesh)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    if dim == 2:
        mat = [[1 + x[0], 0], [0, 1 + x[1]]]
        expected = 4 - 16 * np.sqrt(2) / 9
    else:
        mat = [[1 + x[0], 0, 0], [0, 1 + x[1], 0], [0, 0, 1 + x[2]]]
        expected = 8 / 27 * (22 * np.sqrt(2) - 25)
    metric.interpolate(as_matrix(mat))
    assert np.isclose(metric.complexity(), expected)


def test_enforce_spd_a_max(dim):
    """
    Tests that the :meth:`enforce_spd` method applies maximum anisotropy as expected.
    """
    mesh = uniform_mesh(dim)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    M = np.eye(dim)
    M[0][0] = 10.0
    metric.interpolate(as_matrix(M))
    mp = {"dm_plex_metric_a_max": 1.0}
    metric.set_parameters(mp)
    metric.enforce_spd(restrict_anisotropy=True)

    expected = Function(P1_ten)
    expected.interpolate(10 * Identity(dim))
    assert np.isclose(errornorm(metric, expected), 0)
