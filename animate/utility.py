"""
Utility functions and classes for metric-based mesh adaptation.
"""

from collections import OrderedDict

import firedrake
import firedrake.function as ffunc
import firedrake.mesh as fmesh
import ufl
from adapt_common.utility import cofunction2function
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc

__all__ = ["Mesh", "norm", "errornorm"]


@PETSc.Log.EventDecorator()
def Mesh(arg, **kwargs):
    """
    Overload :func:`firedrake.mesh.Mesh` to endow the output mesh with useful
    quantities.

    The following quantities are computed by default:
        * cell size;
        * facet area.

    The argument and keyword arguments are passed to Firedrake's
    :func:`firedrake.mesh.Mesh` constructor, modified so that the argument could also be
    a mesh.

    Arguments and keyword arguments are the same as for :func:`firedrake.mesh.Mesh`.

    :returns: the constructed mesh
    :rtype: :class:`firedrake.mesh.MeshGeometry`
    """
    try:
        mesh = firedrake.Mesh(arg, **kwargs)
    except TypeError:
        mesh = firedrake.Mesh(arg.coordinates, **kwargs)
    if isinstance(mesh._topology, fmesh.VertexOnlyMeshTopology):
        return mesh
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    P1 = firedrake.FunctionSpace(mesh, "CG", 1)
    dim = mesh.topological_dimension

    # Facet area
    boundary_markers = sorted(mesh.exterior_facets.unique_markers)
    one = ffunc.Function(P1).assign(1.0)
    bnd_len = OrderedDict(
        {i: firedrake.assemble(one * ufl.ds(int(i))) for i in boundary_markers}
    )
    if dim == 2:
        mesh.boundary_len = bnd_len
    else:
        mesh.boundary_area = bnd_len

    # Cell size
    if dim == 2 and mesh.coordinates.ufl_element().cell == ufl.triangle:
        mesh.delta_x = firedrake.assemble(interpolate(ufl.CellDiameter(mesh), P0))

    return mesh


@PETSc.Log.EventDecorator()
def norm(v, norm_type="L2", condition=None, boundary=False):
    r"""
    Overload :func:`firedrake.norms.norm` to allow for :math:`\ell^p` norms.

    Currently supported ``norm_type`` options:
    * ``'l1'``
    * ``'l2'``
    * ``'linf'``
    * ``'L2'``
    * ``'Linf'``
    * ``'H1'``
    * ``'Hdiv'``
    * ``'Hcurl'``
    * or any ``'Lp'`` with :math:`p >= 1`.

    Note that this version is case sensitive, i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg v: the function to take the norm of
    :type v: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :kwarg norm_type: the type of norm to use
    :type norm_type: :class:`str`
    :kwarg condition: a UFL condition for specifying a subdomain to compute the norm
        over
    :kwarg boundary: if ``True``, the norm is computed over the domain boundary
    :type boundary: :class:`bool`
    :returns: the norm value
    :rtype: :class:`float`
    """
    if isinstance(v, firedrake.Cofunction):
        v = cofunction2function(v)
    condition = condition or firedrake.Constant(1.0)
    norm_codes = {"l1": 0, "l2": 2, "linf": 3}
    p = 2
    if norm_type in norm_codes or norm_type == "Linf":
        if boundary:
            raise NotImplementedError("lp errors on the boundary not yet implemented.")
        v.interpolate(condition * v)
        if norm_type == "Linf":
            with v.dat.vec_ro as vv:
                return vv.max()[1]
        else:
            with v.dat.vec_ro as vv:
                return vv.norm(norm_codes[norm_type])
    elif norm_type[0] == "l":
        raise NotImplementedError(
            "lp norm of order {:s} not supported.".format(norm_type[1:])
        )
    else:
        dX = ufl.ds if boundary else ufl.dx
        if norm_type.startswith("L"):
            try:
                p = int(norm_type[1:])
            except Exception as exc:
                raise ValueError(f"Unable to interpret '{norm_type}' norm.") from exc
            if p < 1:
                raise ValueError(f"'{norm_type}' norm does not make sense.")
            integrand = ufl.inner(v, v)
        elif norm_type.lower() in ("h1", "hdiv", "hcurl"):
            integrand = {
                "h1": lambda w: ufl.inner(w, w) + ufl.inner(ufl.grad(w), ufl.grad(w)),
                "hdiv": lambda w: ufl.inner(w, w) + ufl.div(w) * ufl.div(w),
                "hcurl": lambda w: ufl.inner(w, w)
                + ufl.inner(ufl.curl(w), ufl.curl(w)),
            }[norm_type.lower()](v)
        else:
            raise ValueError(f"Unknown norm type '{norm_type}'.")
        return firedrake.assemble(condition * integrand ** (p / 2) * dX) ** (1 / p)


@PETSc.Log.EventDecorator()
def errornorm(u, uh, norm_type="L2", boundary=False, **kwargs):  # noqa: C901
    r"""
    Overload :func:`firedrake.norms.errornorm` to allow for :math:`\ell^p` norms.

    Currently supported ``norm_type`` options:
    * ``'l1'``
    * ``'l2'``
    * ``'linf'``
    * ``'L2'``
    * ``'Linf'``
    * ``'H1'``
    * ``'Hdiv'``
    * ``'Hcurl'``
    * or any ``'Lp'`` with :math:`p >= 1`.

    Note that this version is case sensitive, i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg u: the 'true' value
    :type u: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :arg uh: the approximation of the 'truth'
    :type uh: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :kwarg norm_type: the type of norm to use
    :type norm_type: :class:`str`
    :kwarg boundary: if ``True``, the norm is computed over the domain boundary
    :type boundary: :class:`bool`
    :returns: the error norm value
    :rtype: :class:`float`

    Any other keyword arguments are passed to :func:`firedrake.norms.errornorm`.
    """
    if isinstance(u, firedrake.Cofunction):
        u = cofunction2function(u)
    if isinstance(uh, firedrake.Cofunction):
        uh = cofunction2function(uh)
    if not isinstance(uh, ffunc.Function):
        raise TypeError(f"uh should be a Function, is a '{type(uh)}'.")
    if norm_type[0] == "l":
        if not isinstance(u, ffunc.Function):
            raise TypeError(f"u should be a Function, is a '{type(u)}'.")

    if len(u.ufl_shape) != len(uh.ufl_shape):
        raise RuntimeError("Mismatching rank between u and uh.")

    if isinstance(u, ffunc.Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            firedrake.logging.warning(
                "Degree of exact solution less than approximation degree"
            )

    # Case 1: point-wise norms
    if norm_type[0] == "l":
        v = u
        v -= uh

    # Case 2: UFL norms for mixed function spaces
    elif hasattr(uh.function_space(), "num_sub_spaces"):
        if norm_type == "L2":
            vv = [
                uu - uuh
                for uu, uuh in zip(u.subfunctions, uh.subfunctions, strict=False)
            ]
            dX = ufl.ds if boundary else ufl.dx
            return ufl.sqrt(firedrake.assemble(sum([ufl.inner(v, v) for v in vv]) * dX))
        else:
            raise NotImplementedError(
                f"Norm type '{norm_type}' not supported for mixed spaces."
            )

    # Case 3: UFL norms for non-mixed spaces
    else:
        v = u - uh

    return norm(v, norm_type=norm_type, **kwargs)
