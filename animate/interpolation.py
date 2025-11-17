"""
Driver functions for mesh-to-mesh data transfer.
"""

import firedrake
import numpy as np
import ufl
from firedrake.functionspaceimpl import FiredrakeDualSpace, WithGeometry
from firedrake.petsc import PETSc
from firedrake.supermeshing import assemble_mixed_mass_matrix
from petsc4py import PETSc as petsc4py
from pyop2 import op2

from animate.quality import QualityMeasure
from animate.utility import (
    assemble_mass_matrix,
    cofunction2function,
    function2cofunction,
)

__all__ = ["transfer", "interpolate", "project", "clement_interpolant"]


@PETSc.Log.EventDecorator()
def transfer(source, target_space, transfer_method="project", **kwargs):
    r"""
    Overload functions :func:`firedrake.interpolation.interpolate` and
    :func:`firedrake.projection.project` to account for the case of two mixed
    function spaces defined on different meshes and for the adjoint interpolation
    operator when applied to :class:`firedrake.cofunction.Cofunction`\s.

    Note that the "interpolate" option works straightforwardly with MPI parallelism,
    whereas the "project" option can be difficult to set up to make use of this.

    :arg source: the function to be transferred
    :type source: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :arg target_space: the function space which we seek to transfer onto, or the
        function or cofunction to use as the target
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`,
        :class:`firedrake.function.Function` or :class:`firedrake.cofunction.Cofunction`
    :kwarg transfer_method: the method to use for the transfer. Options are
        "interpolate" (default) and "project"
    :type transfer_method: str
    :returns: the transferred function
    :rtype: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`

    Extra keyword arguments are passed to :func:`firedrake.interpolation.interpolate` or
        :func:`firedrake.projection.project`.
    """
    if transfer_method not in ("interpolate", "project"):
        raise ValueError(
            f"Invalid transfer method: {transfer_method}."
            " Options are 'interpolate' and 'project'."
        )
    if not isinstance(source, (firedrake.Function, firedrake.Cofunction)):
        raise NotImplementedError(
            f"Can only currently {transfer_method} Functions and Cofunctions."
        )
    if isinstance(target_space, WithGeometry):
        target = firedrake.Function(target_space)
    elif isinstance(target_space, (firedrake.Cofunction, firedrake.Function)):
        target = target_space
    else:
        raise TypeError(
            "Second argument must be a FunctionSpace, Function, or Cofunction."
        )
    if transfer_method == "interpolate":
        return interpolate(source, target, **kwargs)
    else:
        return project(source, target, **kwargs)


def _validate_consistent_spaces(Vs, Vt):
    if Vs._dual != Vt._dual:
        raise ValueError("Spaces must be both primal or both dual.")
    if hasattr(Vs, "num_sub_spaces"):
        if not hasattr(Vt, "num_sub_spaces"):
            raise ValueError(
                "Source space has multiple components but target space does not."
            )
        if Vs.num_sub_spaces() != Vt.num_sub_spaces():
            raise ValueError(
                "Inconsistent numbers of components in source and target spaces:"
                f" {Vs.num_sub_spaces()} vs. {Vt.num_sub_spaces()}."
            )
    elif hasattr(Vt, "num_sub_spaces"):
        raise ValueError(
            "Target space has multiple components but source space does not."
        )


@PETSc.Log.EventDecorator()
def interpolate(source, target, **kwargs):
    r"""
    Overload :func:`firedrake.interpolation.interpolate` to account for the case of
    mixed function spaces.

    :arg source: the function or cofunction to be transferred
    :type source: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :arg target: the function or cofunction to use as the target, which is modified in
        place
    :type target: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`

    Extra keyword arguments are passed to :func:`firedrake.interpolation.interpolate`
    """
    _validate_consistent_spaces(source.function_space(), target.function_space())
    if hasattr(target.function_space(), "num_sub_spaces"):
        for s, t in zip(source.subfunctions, target.subfunctions, strict=True):
            t.interpolate(s, **kwargs)
    else:
        target.interpolate(source, **kwargs)


# TODO: Reimplement by introducing a LumpedSupermeshProjector subclass of
#       firedrake.projection.SupermeshProjector (#123)
# TODO: Implement minimal diffusion correction (#124)
def _supermesh_project(source, target, bounded=False):
    Vs = source.function_space()
    Vt = target.function_space()
    element_t = Vt.ufl_element()
    if bounded and (element_t.family(), element_t.degree()) != ("Lagrange", 1):
        raise ValueError("Mass lumping is not recommended for spaces other than P1.")

    # Create a linear system using the lumped mass matrix for the target space
    mixed_mass = assemble_mixed_mass_matrix(Vs, Vt)
    ksp = petsc4py.KSP().create()
    ksp.setOperators(assemble_mass_matrix(Vt, lumped=bounded))

    # Solve the linear system
    with source.dat.vec_ro as s, target.dat.vec_wo as t:
        rhs = t.copy()
        mixed_mass.mult(s, rhs)
        ksp.solve(rhs, t)


@PETSc.Log.EventDecorator()
def project(source, target, bounded=False, **kwargs):
    r"""
    Overload :func:`firedrake.projection.project` to account for the case of mixed
    function spaces.

    For details on the approach for achieving boundedness through mass lumping and
    post-processing, see :cite:`Farrell:2009`.

    :arg source: the function or cofunction to be transferred
    :type source: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :arg target: the function or cofunction to transfer onto, which is modified in
        place
    :type target: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :kwarg bounded: apply mass lumping to the mass matrix to ensure boundedness
    :type bounded: :class:`bool`

    Extra keyword arguments are passed to :func:`firedrake.projection.project`.
    """
    _validate_consistent_spaces(source.function_space(), target.function_space())
    if not source.function_space()._dual:
        for s, t in zip(source.subfunctions, target.subfunctions, strict=True):
            if bounded:
                _supermesh_project(s, t, bounded=True)
            else:
                t.project(s, **kwargs)
    else:
        for s, t in zip(source.subfunctions, target.subfunctions, strict=True):
            sf = cofunction2function(s)
            tf = cofunction2function(t)
            Vs = sf.function_space()
            ksp = petsc4py.KSP().create()
            ksp.setOperators(assemble_mass_matrix(Vs, lumped=bounded))
            mixed_mass = assemble_mixed_mass_matrix(Vs, tf.function_space())
            with sf.dat.vec_ro as vs, tf.dat.vec_wo as vt:
                residual = vs.copy()
                ksp.solveTranspose(vs, residual)
                mixed_mass.mult(residual, vt)  # NOTE: already transposed above
            function2cofunction(tf, cofunc=t)


@PETSc.Log.EventDecorator()
def clement_interpolant(source, target_space=None, boundary=False):
    r"""
    Compute the Clement interpolant of a :math:`\mathbb P0` source field, i.e. take the
    volume average over neighbouring cells at each vertex. See :cite:`Clement:1975`.

    :arg source: the :math:`\mathbb P0` source field
    :type source: :class:`firedrake.function.Function`
    :kwarg target_space: the :math:`\mathbb P1` space to interpolate into
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :kwarg boundary: interpolate over boundary facets or cells?
    :type boundary: :class:`bool`
    """
    if not isinstance(source, (firedrake.Cofunction, firedrake.Function)):
        raise TypeError(f"Expected Cofunction or Function, got '{type(source)}'.")

    # Map Cofunctions to Functions for the interpolation
    is_cofunction = isinstance(source, firedrake.Cofunction)
    if is_cofunction:
        data = source.dat.data_with_halos
        source = firedrake.Function(source.function_space().dual())
        source.dat.data_with_halos[:] = data

    # Process source space
    Vs = source.function_space()
    Vs_e = Vs.ufl_element()
    if not (Vs_e.family() == "Discontinuous Lagrange" and Vs_e.degree() == 0):
        raise ValueError("Source function provided must be from a P0 space.")
    rank = len(Vs.value_shape)
    if rank not in (0, 1, 2):
        raise ValueError(f"Rank-{rank + 1} tensors are not supported.")
    mesh = Vs.mesh()
    dim = mesh.topological_dimension

    # Process target space
    Vt = target_space
    if Vt is None:
        Vt = {
            0: firedrake.FunctionSpace,
            1: firedrake.VectorFunctionSpace,
            2: firedrake.TensorFunctionSpace,
        }[rank](mesh, "CG", 1)
    elif isinstance(Vt, FiredrakeDualSpace):
        Vt = Vt.dual()
    else:
        is_cofunction = False
    Vt_e = Vt.ufl_element()
    if not (Vt_e.family() == "Lagrange" and Vt_e.degree() == 1):
        raise ValueError("Target space provided must be P1.")
    target = firedrake.Function(Vt)

    # Scalar P0 and P1 spaces to hold volumes, etc.
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    P1 = firedrake.FunctionSpace(mesh, "CG", 1)

    # Determine target domain
    tdomain = {
        0: "{[i]: 0 <= i < t.dofs}",
        1: f"{{[i, j]: 0 <= i < t.dofs and 0 <= j < {dim}}}",
        2: f"{{[i, j, k]: 0 <= i < t.dofs and 0 <= j < {dim} and 0 <= k < {dim}}}",
    }[rank]

    # Compute the patch volume at each vertex
    if not boundary:
        dX = ufl.dx(domain=mesh)
        volume = QualityMeasure(mesh, python=True)("volume")

        # Compute patch volume
        patch_volume = firedrake.Function(P1)
        domain = "{[i]: 0 <= i < patch.dofs}"
        instructions = "patch[i] = patch[i] + vol[0]"
        keys = {"vol": (volume, op2.READ), "patch": (patch_volume, op2.RW)}
        firedrake.par_loop((domain, instructions), dX, keys)

        # Take weighted average
        instructions = {
            0: "t[i] = t[i] + v[0] * s[0]",
            1: "t[i, j] = t[i, j] + v[0] * s[0, j]",
            2: (
                f"t[i, {dim} * j + k] ="
                f" t[i, {dim} * j + k] + v[0] * s[0, {dim} * j + k]"
            ),
        }[rank]
        keys = {
            "s": (source, op2.READ),
            "v": (volume, op2.READ),
            "t": (target, op2.RW),
        }
        firedrake.par_loop((tdomain, instructions), dX, keys)
    else:
        dX = ufl.ds(domain=mesh)

        # Indicate appropriate boundary
        bnd_indicator = firedrake.Function(P1)
        firedrake.DirichletBC(P1, 1, "on_boundary").apply(bnd_indicator)

        # Determine facet area for boundary edges
        v = firedrake.TestFunction(P0)
        u = firedrake.TrialFunction(P0)
        bnd_volume = firedrake.Function(P0)
        mass_term = v * u * dX
        rhs = v * ufl.FacetArea(mesh) * dX
        sp = {"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": "jacobi"}
        firedrake.solve(mass_term == rhs, bnd_volume, solver_parameters=sp)

        # Compute patch volume
        patch_volume = firedrake.Function(P1)
        domain = "{[i]: 0 <= i < patch.dofs}"
        instructions = "patch[i] = patch[i] + indicator[i] * bnd_vol[0]"
        keys = {
            "bnd_vol": (bnd_volume, op2.READ),
            "indicator": (bnd_indicator, op2.READ),
            "patch": (patch_volume, op2.RW),
        }
        firedrake.par_loop((domain, instructions), dX, keys)

        # Take weighted average
        instructions = {
            0: "t[i] = t[i] + v[0] * b[i] * s[0]",
            1: "t[i, j] = t[i, j] + v[0] * b[i] * s[0, j]",
            2: (
                f"t[i, {dim} * j + k] = "
                f"t[i, {dim} * j + k] + v[0] * b[i] * s[0, {dim} * j + k]"
            ),
        }[rank]
        keys = {
            "s": (source, op2.READ),
            "v": (bnd_volume, op2.READ),
            "b": (bnd_indicator, op2.READ),
            "t": (target, op2.RW),
        }
        firedrake.par_loop((tdomain, instructions), dX, keys)

    # Divide by patch volume and ensure finite
    target.interpolate(target / patch_volume)
    target.dat.data_with_halos[:] = np.nan_to_num(
        target.dat.data_with_halos, posinf=0, neginf=0
    )

    # Map back to Cofunction, if one was passed originally
    if is_cofunction:
        data = target.dat.data_with_halos
        target = firedrake.Cofunction(target.function_space().dual())
        target.dat.data_with_halos[:] = data
    return target
