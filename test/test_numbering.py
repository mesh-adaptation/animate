"""Module containing unit tests for the cython.numbering module."""

import firedrake as fd
import numpy as np
import pytest
from animate.cython.numbering import to_petsc_local_numbering
from firedrake.petsc import PETSc


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

    comm = function_space.comm
    # unique float 0<=x<1 for each rank:
    rank_fraction = comm.rank / (comm.size + 1)

    # Initialise owned and in particular halo DoFs, so we can check they
    # have been updated later on
    f.assign(-1)

    # Fill the vector with arbitrary values
    # f.dat.vec below is a global Vec, so owned entries only
    # halos values should be updated automatically coming out of the context
    with f.dat.vec as vec:
        vec.array[:] = np.arange(vec.sizes[0]) + rank_fraction

    # Check that this is the case:
    owned = f.dat.data_ro.flatten()
    halos = f.dat.data_ro_with_halos[owned.size:].flatten()
    np.testing.assert_equal(owned, np.arange(owned.size) + rank_fraction)
    if halos.size > 0:
        # all halo enties should be set and come from different rank
        assert all(halos >= 0.0)
        assert all(np.modf(halos)[0] != rank_fraction)

    # Verify that reordering according to the PETSc numbering is a permutation
    data = f.dat.data_ro_with_halos
    lvec = PETSc.Vec().createWithArray(
        data, size=data.size, bsize=f.dat.cdim, comm=fd.COMM_SELF
    )
    reordered_vec = to_petsc_local_numbering(lvec, function_space)
    assert reordered_vec is not None
    assert reordered_vec.size == lvec.size
    assert sorted(reordered_vec.array) == sorted(lvec.array)
    lvec.destroy()
    reordered_vec.destroy()


@pytest.mark.parallel(nprocs=2)
def test_is_permutation_np2(function_space):
    test_is_permutation(function_space)


@pytest.mark.parallel(nprocs=3)
def test_is_permutation_np3(function_space):
    test_is_permutation(function_space)
