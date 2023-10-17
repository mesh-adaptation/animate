from test_setup import *
import pytest
import numpy as np


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=[1, 2, np.inf])
def degree(request):
    return request.param


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
