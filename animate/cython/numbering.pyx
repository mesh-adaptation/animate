import numpy as np
from firedrake.petsc import PETSc
from firedrake.utils import IntType, ScalarType

cimport numpy as np
cimport petsc4py.PETSc as PETSc

from petsc4py.PETSc cimport CHKERR


cdef extern from "petsc.h":
    ctypedef long PetscInt
    ctypedef enum PetscErrorCode:
        PETSC_SUCCESS
        PETSC_ERR_LIB


cdef extern from "petscis.h" nogil:
    PetscErrorCode PetscSectionGetDof(PETSc.PetscSection,PetscInt,PetscInt*)
    PetscErrorCode PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)


def to_petsc_local_numbering(PETSc.Vec vec, V):
    """
    Reorder a PETSc Vec corresponding to a Firedrake Function w.r.t.
    the PETSc natural numbering.

    :arg vec: the PETSc Vec to reorder; must be a global vector
    :arg V: the FunctionSpace of the Function which the Vec comes from
    :ret out: a copy of the Vec, ordered with the PETSc natural numbering
    """
    cdef int dim, idx, start, end, p, d, k
    cdef PetscInt dof, off
    cdef PETSc.Vec out
    cdef PETSc.Section section
    cdef np.ndarray varray, oarray

    section = V.dm.getGlobalSection()
    out = vec.duplicate()
    varray = vec.array_r
    oarray = out.array
    dim = V.value_size
    idx = 0
    start, end = vec.getOwnershipRange()
    for p in range(*section.getChart()):
        CHKERR(PetscSectionGetDof(section.sec, p, &dof))
        if dof > 0:
            CHKERR(PetscSectionGetOffset(section.sec, p, &off))
            assert off >= 0
            off *= dim
            for d in range(dof):
                for k in range(dim):
                    oarray[idx] = varray[off + dim * d + k - start]
                    idx += 1
    assert idx == (end - start)
    return out
