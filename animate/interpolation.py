"""
Driver functions for mesh-to-mesh data transfer.
"""

import firedrake
from adapt_common.utility import assemble_mass_matrix, cofunction2function
from firedrake.functionspaceimpl import WithGeometry
from firedrake.petsc import PETSc
from firedrake.supermeshing import assemble_mixed_mass_matrix
from petsc4py import PETSc as petsc4py

__all__ = ["transfer", "interpolate", "project"]


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
    Overload function :func:`firedrake.interpolation.interpolate` to account for the
    case of two mixed function spaces defined on different meshes and for the adjoint
    interpolation operator when applied to :class:`firedrake.cofunction.Cofunction`\s.

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
    function spaces and handle dual Functions, with a bounded option.

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
            t.dat.data_with_halos[:] = tf.dat.data_with_halos
