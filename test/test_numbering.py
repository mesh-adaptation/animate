"""Module containing unit tests for the cython.numbering module."""

import firedrake as fd
import numpy as np
import pytest
from animate.cython.numbering import to_petsc_local_numbering


@pytest.fixture
def setup_function_space():
    """Fixture to create a Firedrake FunctionSpace."""
    return fd.FunctionSpace(fd.UnitSquareMesh(4, 4), "CG", 1)


def test_is_permutation(setup_function_space):
    """Test that to_petsc_local_numbering is indeed a permutation."""
    V = setup_function_space
    f = fd.Function(V)

    # Fill the vector with arbitrary values
    with f.dat.vec as vec:
        vec.array[:] = np.arange(vec.size)

    # Verify that reordering according to the PETSc numbering is a permutation
    with f.dat.vec_ro as vec:
        reordered_vec = to_petsc_local_numbering(vec, V)
        assert reordered_vec is not None
        assert reordered_vec.size == vec.size
        assert sorted(reordered_vec.array) == sorted(vec.array)
