"""
Tests under MPI parallelism.
"""
from firedrake import COMM_WORLD
import test_metric
import pytest


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.mark.parallel(nprocs=2)
def test_intersect_uniform_np2(dim):
    assert COMM_WORLD.size == 2
    test_metric.TestMetricCombination().test_uniform_combine(dim, False)


@pytest.mark.parallel(nprocs=2)
def test_average_uniform_np2(dim):
    assert COMM_WORLD.size == 2
    test_metric.TestMetricCombination().test_uniform_combine(dim, True)


@pytest.mark.parallel(nprocs=2)
def test_average_variable_np2(dim):
    assert COMM_WORLD.size == 2
    test_metric.TestMetricCombination().test_variable_average(dim)


@pytest.mark.parallel(nprocs=2)
def test_hessian_bowl_np2(dim):
    assert COMM_WORLD.size == 2
    test_metric.TestHessianMetric().test_bowl(dim, places=6)


@pytest.mark.parallel(nprocs=2)
def test_normalise_uniform_np2(dim):
    assert COMM_WORLD.size == 2
    test_metric.TestNormalisation().test_uniform(dim)
