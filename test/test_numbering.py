"""Module containing unit tests for the cython.numbering module."""

import firedrake as fd
import numpy as np
import pytest
from animate.cython.numbering import to_petsc_local_numbering


@pytest.fixture
def mesh():
    """Fixture to create a uniform unit square mesh."""
    return fd.UnitSquareMesh(4, 4)


@pytest.fixture(params=[0, 1, 2])
def rank(request):
    """Fixture specifying function space rank."""
    return request.param


@pytest.fixture
def function_space(mesh, rank):
    """Fixture to create a function space on a given mesh with a given rank."""
    try:
        return {
            0: fd.FunctionSpace,
            1: fd.VectorFunctionSpace,
            2: fd.TensorFunctionSpace,
        }[rank](mesh, "CG", 1)
    except KeyError as key_err:
        raise ValueError(f"Rank {rank} not considered.") from key_err


def test_is_permutation(function_space):
    """Test that to_petsc_local_numbering is indeed a permutation."""
    f = fd.Function(function_space)

    # Fill the vector with arbitrary values
    with f.dat.vec as vec:
        vec.array[:] = np.arange(vec.size)

    # Verify that reordering according to the PETSc numbering is a permutation
    with f.dat.vec_ro as vec:
        reordered_vec = to_petsc_local_numbering(vec, function_space)
        assert reordered_vec is not None
        assert reordered_vec.size == vec.size
        assert sorted(reordered_vec.array) == sorted(vec.array)
