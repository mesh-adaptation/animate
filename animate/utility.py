"""
Utility functions and classes for metric-based mesh adaptation.
"""

from collections import OrderedDict

import firedrake
import firedrake.function as ffunc
import firedrake.mesh as fmesh
import ufl
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc

__all__ = ["Mesh"]


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
