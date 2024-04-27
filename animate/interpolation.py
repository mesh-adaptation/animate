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
    Overload functions :func:`firedrake.__future__.interpolate` and
    :func:`firedrake.projection.project` to account for the case of two mixed
    function spaces defined on different meshes and for the adjoint interpolation
    operator when applied to :class:`firedrake.cofunction.Cofunction`\s.

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

    Extra keyword arguments are passed to :func:`firedrake.__future__.interpolate` or
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
    if isinstance(source, firedrake.Cofunction):
        return _transfer_adjoint(source, target, transfer_method, **kwargs)
    elif source.function_space() == target.function_space():
        return target.assign(source)
    else:
        return _transfer_forward(source, target, transfer_method, **kwargs)


@PETSc.Log.EventDecorator()
def interpolate(source, target_space, **kwargs):
    """
    A wrapper for :func:`transfer` with ``transfer_method="interpolate"``.
    """
    return transfer(source, target_space, transfer_method="interpolate", **kwargs)


@PETSc.Log.EventDecorator()
def project(source, target_space, **kwargs):
    """
    A wrapper for :func:`transfer` with ``transfer_method="interpolate"``.

    :kwarg lumped: if `True`, mass lumping is applied to the mass matrix
    :type lumped: :class:`bool`
    """
    return transfer(source, target_space, transfer_method="project", **kwargs)


def _supermesh_project(source, target, lumped=False):
    Vs = source.function_space()
    Vt = target.function_space()
    element_t = Vt.ufl_element()
    if lumped and (element_t.family(), element_t.degree()) != ("Lagrange", 1):
        raise ValueError("Mass lumping is not recommended for spaces other than P1.")
    mixed_mass = assemble_mixed_mass_matrix(Vs, Vt)
    ksp = petsc4py.KSP().create()
    ksp.setOperators(assemble_mass_matrix(Vt, lumped=lumped))
    with source.dat.vec_ro as s, target.dat.vec_wo as t:
        rhs = t.copy()
        mixed_mass.mult(s, rhs)
        ksp.solve(rhs, t)


@PETSc.Log.EventDecorator()
def _transfer_forward(source, target, transfer_method, **kwargs):
    """
    Apply mesh-to-mesh transfer operator to a Function.

    This function extends the functionality of :func:`firedrake.__future__.interpolate`
    and :func:`firedrake.projection.project` to account for mixed spaces.

    :arg source: the Function to be transferred
    :type source: :class:`firedrake.function.Function`
    :arg target: the Function which we seek to transfer onto
    :type target: :class:`firedrake.function.Function`
    :kwarg transfer_method: the method to use for the transfer. Options are
        "interpolate" (default) and "project"
    :type transfer_method: str
    :kwarg lumped: if `True`, mass lumping is applied to the mass matrix (project only)
    :type lumped: :class:`bool`
    :returns: the transferred Function
    :rtype: :class:`firedrake.function.Function`

    Extra keyword arguments are passed to :func:`firedrake.__future__.interpolate` or
        :func:`firedrake.projection.project`.
    """
    lumped = transfer_method == "project" and kwargs.pop("lumped", False)
    Vs = source.function_space()
    Vt = target.function_space()
    _validate_matching_spaces(Vs, Vt)
    assert isinstance(target, firedrake.Function)
    if hasattr(Vt, "num_sub_spaces"):
        for s, t in zip(source.subfunctions, target.subfunctions):
            if transfer_method == "interpolate":
                t.interpolate(s, **kwargs)
            elif transfer_method == "project":
                if lumped:
                    _supermesh_project(s, t, lumped=True)
                else:
                    t.project(s, **kwargs)
            else:
                raise ValueError(
                    f"Invalid transfer method: {transfer_method}."
                    " Options are 'interpolate' and 'project'."
                )
    else:
        if transfer_method == "interpolate":
            target.interpolate(source, **kwargs)
        elif transfer_method == "project":
            if lumped:
                _supermesh_project(source, target, lumped=True)
            else:
                target.project(source, **kwargs)
        else:
            raise ValueError(
                f"Invalid transfer method: {transfer_method}."
                " Options are 'interpolate' and 'project'."
            )
    return target


@PETSc.Log.EventDecorator()
def _transfer_adjoint(target_b, source_b, transfer_method, **kwargs):
    """
    Apply an adjoint mesh-to-mesh transfer operator to a Cofunction.

    :arg target_b: seed Cofunction from the target space of the forward projection
    :type target_b: :class:`firedrake.cofunction.Cofunction`
    :arg source_b: output Cofunction from the source space of the forward projection
    :type source_b: :class:`firedrake.cofunction.Cofunction`
    :kwarg transfer_method: the method to use for the transfer. Options are
        "interpolate" (default) and "project"
    :type transfer_method: str
    :kwarg lumped: if `True`, mass lumping is applied to the mass matrix (project only)
    :type lumped: :class:`bool`
    :returns: the transferred Cofunction
    :rtype: :class:`firedrake.cofunction.Cofunction`

    Extra keyword arguments are passed to :func:`firedrake.__future__.interpolate` or
        :func:`firedrake.projection.project`.
    """
    lumped = transfer_method == "project" and kwargs.pop("lumped", False)

    # Map to Functions to apply the adjoint transfer
    if not isinstance(target_b, firedrake.Function):
        target_b = cofunction2function(target_b)
    if not isinstance(source_b, firedrake.Function):
        source_b = cofunction2function(source_b)

    Vt = target_b.function_space()
    Vs = source_b.function_space()
    if Vs == Vt:
        source_b.assign(target_b)
        return function2cofunction(source_b)

    _validate_matching_spaces(Vs, Vt)
    if hasattr(Vs, "num_sub_spaces"):
        target_b_split = target_b.subfunctions
        source_b_split = source_b.subfunctions
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint transfer operator to each component
    for i, (t_b, s_b) in enumerate(zip(target_b_split, source_b_split)):
        if transfer_method == "interpolate":
            raise NotImplementedError(
                "Adjoint of interpolation operator not implemented."
            )  # TODO (#113)
        elif transfer_method == "project":
            ksp = petsc4py.KSP().create()
            ksp.setOperators(assemble_mass_matrix(t_b.function_space(), lumped=lumped))
            mixed_mass = assemble_mixed_mass_matrix(Vt[i], Vs[i])
            with t_b.dat.vec_ro as tb, s_b.dat.vec_wo as sb:
                residual = tb.copy()
                ksp.solveTranspose(tb, residual)
                mixed_mass.mult(residual, sb)  # NOTE: already transposed above
        else:
            raise ValueError(
                f"Invalid transfer method: {transfer_method}."
                " Options are 'interpolate' and 'project'."
            )

    # Map back to a Cofunction
    return function2cofunction(source_b)


def _validate_matching_spaces(Vs, Vt):
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
def clement_interpolant(source, target_space=None, boundary=False):
    r"""
    Compute the Clement interpolant of a :math:`\mathbb P0` source field, i.e. take the
    volume average over neighbouring cells at each vertex. See :cite:`Cle:75`.

    :arg source: the :math:`\mathbb P0` source field
    :kwarg target_space: the :math:`\mathbb P1` space to interpolate into
    :kwarg boundary: interpolate over boundary facets or cells?
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
    rank = len(Vs_e.value_shape)
    if rank not in (0, 1, 2):
        raise ValueError(f"Rank-{rank + 1} tensors are not supported.")
    mesh = Vs.mesh()
    dim = mesh.topological_dimension()

    # Process target space
    Vt = target_space
    if Vt is None:
        if rank == 0:
            Vt = firedrake.FunctionSpace(mesh, "CG", 1)
        elif rank == 1:
            Vt = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        else:
            Vt = firedrake.TensorFunctionSpace(mesh, "CG", 1)
    elif isinstance(Vt, FiredrakeDualSpace):
        Vt = Vt.dual()
    elif not isinstance(Vt, FiredrakeDualSpace):
        is_cofunction = False
    Vt_e = Vt.ufl_element()
    if not (Vt_e.family() == "Lagrange" and Vt_e.degree() == 1):
        raise ValueError("Target space provided must be P1.")
    target = firedrake.Function(Vt)

    # Scalar P0 and P1 spaces to hold volumes, etc.
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    P1 = firedrake.FunctionSpace(mesh, "CG", 1)

    # Determine target domain
    if rank == 0:
        tdomain = "{[i]: 0 <= i < t.dofs}"
    elif rank == 1:
        tdomain = f"{{[i, j]: 0 <= i < t.dofs and 0 <= j < {dim}}}"
    else:
        tdomain = (
            f"{{[i, j, k]: 0 <= i < t.dofs and 0 <= j < {dim} and 0 <= k < {dim}}}"
        )

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
        if rank == 0:
            instructions = "t[i] = t[i] + v[0] * s[0]"
        elif rank == 1:
            instructions = "t[i, j] = t[i, j] + v[0] * s[0, j]"
        else:
            instructions = f"t[i, {dim} * j + k] = t[i, {dim} * j + k] + v[0] * s[0, {dim} * j + k]"
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
        if rank == 0:
            instructions = "t[i] = t[i] + v[0] * b[i] * s[0]"
        elif rank == 1:
            instructions = "t[i, j] = t[i, j] + v[0] * b[i] * s[0, j]"
        else:
            instructions = f"t[i, {dim} * j + k] = t[i, {dim} * j + k] + v[0] * b[i] * s[0, {dim} * j + k]"
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
