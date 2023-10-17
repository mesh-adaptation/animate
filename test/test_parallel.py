"""
Tests under MPI parallelism.
"""
import test_metric
import pytest


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.mark.parallel(nprocs=2)
def test_hessian_bowl_np2(dim):
    test_metric.TestHessianMetric()._test_bowl(dim, places=6)


@pytest.mark.parallel(nprocs=2)
def test_normalise_uniform_np2(dim):
    test_metric.TestNormalisation()._test_uniform(dim)
