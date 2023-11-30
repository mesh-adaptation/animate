"""
Driver functions for mesh-to-mesh data transfer.
"""
from animate.quality import QualityMeasure
import firedrake
from firedrake.functionspaceimpl import FiredrakeDualSpace
from firedrake.petsc import PETSc
from pyop2 import op2
import numpy as np
import ufl


__all__ = ["clement_interpolant"]


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
