"""
Pytest configuration for the Animate test suite.
"""

import gc

from firedrake import COMM_WORLD
from firedrake.petsc import PETSc

# Workaround for a deadlock at exit when running the test suite in parallel.
#
# PETSc's garbage collector only destroys objects that are in the garbage on all
# ranks. Firedrake's solvers call ``PETSc.garbage_cleanup`` without first calling
# ``gc.collect()``, so what is in the garbage depends on when Python's automatic
# cyclic garbage collector last ran on each rank, which differs between ranks
# (e.g. when only rank 0 compiles kernels that miss the disk cache). An object that
# is destroyed directly (through its parent) on one rank, but is put in the garbage
# later on another rank, is then never cleaned up on the latter. That leaves the
# ranks disagreeing on whether ``COMM_WORLD`` still has a PETSc inner communicator,
# so the collective garbage cleanup at exit deadlocks (one rank waiting in
# ``MPI_Comm_dup``, the other in ``MPI_Allreduce``).
#
# Collecting cyclic garbage on all ranks immediately before every PETSc garbage
# cleanup keeps the garbage consistent across ranks. This can be removed once
# Firedrake does this itself.
# See: https://github.com/mesh-adaptation/animate/issues/254
if COMM_WORLD.size > 1:
    _petsc_garbage_cleanup = PETSc.garbage_cleanup

    def _synchronised_garbage_cleanup(*args, **kwargs):
        gc.collect()
        return _petsc_garbage_cleanup(*args, **kwargs)

    PETSc.garbage_cleanup = _synchronised_garbage_cleanup
