"""
Cython module for handling numberings of PETSc Vecs corresponding to Firedrake Functions.
"""
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
    the PETSc natural numbering, i.e. the numbering consistent with
    that of the DMPlex topological points.

    :arg vec: the PETSc Vec to reorder; must be a local vector, i.e.
              a sequential vector that includes all (owned and halo) DoFs
    :arg V: the FunctionSpace of the Function which the Vec comes from
    :ret out: a copy of the Vec, ordered with the PETSc natural numbering
    """
    cdef int dim, idx, lsize, p, d, k
    cdef PetscInt dof, off
    cdef PETSc.Vec out
    cdef PETSc.Section section
    cdef np.ndarray varray, oarray

    section = V.dm.getLocalSection()
    out = vec.duplicate()
    varray = vec.array_r
    oarray = out.array
    dim = V.value_size
    idx = 0
    lsize = vec.getSize()
    for p in range(*section.getChart()):
        CHKERR(PetscSectionGetDof(section.sec, p, &dof))
        if dof > 0:
            CHKERR(PetscSectionGetOffset(section.sec, p, &off))
            assert off >= 0
            off *= dim
            for d in range(dof):
                for k in range(dim):
                    oarray[idx] = varray[off + dim * d + k]
                    idx += 1
    if idx != lsize:
        raise ValueError(
           f"Number of local section entries not the same as vector size"
           f"({idx} vs. {lsize}). Need to provide local vector including halo DoFs."
        )
        
    return out
