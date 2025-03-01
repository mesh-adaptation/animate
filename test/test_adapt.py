"""
Unit tests for invoking mesh adaptation tools Mmg2d, Mmg3d, and ParMmg.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import ufl
from firedrake.assemble import assemble
from firedrake.constant import Constant
from firedrake.mesh import Mesh
from petsc4py import PETSc
from pyop2.mpi import COMM_WORLD
from test_setup import uniform_mesh, uniform_metric

from animate.adapt import adapt


def load_mesh(fname):
    """
    Load a mesh in gmsh format.

    :arg fname: file name, without the .msh extension
    :type fname: :class:`str`
    :return: the mesh
    :rtype: :class:`firedrake.mesh.MeshGeometry`
    """
    firedrake_spec = importlib.util.find_spec("firedrake")
    firedrake_basedir = Path(firedrake_spec.origin).parent.parent
    mesh_dir = firedrake_basedir / "tests" / "firedrake" / "meshes"
    return Mesh(str(mesh_dir / f"{fname}.msh"))


def try_adapt(mesh, metric, serialise=None):
    """
    Attempt to invoke PETSc's mesh adaptation functionality and xfail if it is not
    installed.

    :arg mesh: mesh to be adapted
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :arg metric: Riemannian metric to adapt with respect to
    :type metric: :class:`animate.metric.RiemannianMetric`
    :return: mesh adapted according to the metric
    :rtype: :class:`firedrake.mesh.MeshGeometry`
    """
    try:
        return adapt(mesh, metric, serialise=serialise)
    except PETSc.Error as exc:
        if exc.ierr == 63:
            pytest.xfail("No mesh adaptation tools are installed")
        else:
            raise Exception(f"PETSc error code {exc.ierr}") from exc


@pytest.mark.parallel(nprocs=2)
def test_adapt_2dparallel_error():
    """
    Ensure adapting a 2D mesh in parallel raises a ``ValueError`` with the message
    "Parallel adaptation is only supported in 3D."
    """
    mesh = uniform_mesh(2)
    with pytest.raises(ValueError) as e_info:
        try_adapt(mesh, uniform_metric(mesh), serialise=False)
    assert str(e_info.value) == "Parallel adaptation is only supported in 3D."


@pytest.mark.parametrize(
    "dim,serialise",
    [(2, True), (3, True), (3, False)],
    ids=["mmg2d", "mmg3d", "ParMmg"],
)
def test_no_adapt(dim, serialise):
    """
    Ensure mesh adaptation operations can be turned off.
    """
    mesh = uniform_mesh(dim)
    dofs = mesh.coordinates.vector().gather().shape
    mp = {
        "dm_plex_metric": {
            "no_insert": None,
            "no_move": None,
            "no_swap": None,
            "no_surf": None,
        }
    }
    metric = uniform_metric(mesh, metric_parameters=mp)
    newmesh = try_adapt(mesh, metric, serialise=serialise)
    assert newmesh.coordinates.vector().gather().shape == dofs


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize(
    # "dim,serialise", [(3, True), (3, False)], ids=["mmg3d", "ParMmg"]  # Hangs (#136)
    "dim,serialise", [(3, True),], ids=["mmg3d",]
)
def test_no_adapt_parallel(dim, serialise):
    """
    Ensure mesh adaptation operations can be turned off when running in parallel.
    """
    assert COMM_WORLD.size == 2
    test_no_adapt(dim, serialise=serialise)


@pytest.mark.parametrize(
    "meshname",
    ["annulus", "cell-sets", "square_with_embedded_line"],
)
def test_preserve_cell_tags_2d(meshname):
    """
    Ensure cell tags are preserved after mesh adaptation.
    """
    mesh = load_mesh(meshname)
    metric = uniform_metric(mesh)
    newmesh = try_adapt(mesh, metric)

    tags = set(mesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    newtags = set(newmesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    assert tags == newtags, "Cell tags do not match"

    one = Constant(1.0)
    for tag in tags:
        bnd = assemble(one * ufl.dx(tag, domain=mesh))
        newbnd = assemble(one * ufl.dx(tag, domain=newmesh))
        assert np.isclose(bnd, newbnd), f"Area of region {tag} not preserved"


@pytest.mark.parametrize(
    "meshname",
    ["annulus", "circle_in_square"],
)
def test_preserve_facet_tags_2d(meshname):
    """
    Ensure facet tags are preserved after mesh adaptation.
    """
    mesh = load_mesh(meshname)
    metric = uniform_metric(mesh)
    newmesh = try_adapt(mesh, metric)

    newmesh.init()
    tags = set(mesh.exterior_facets.unique_markers)
    newtags = set(newmesh.exterior_facets.unique_markers)
    assert tags == newtags, "Facet tags do not match"

    one = Constant(1.0)
    for tag in tags:
        bnd = assemble(one * ufl.ds(tag, domain=mesh))
        newbnd = assemble(one * ufl.ds(tag, domain=newmesh))
        assert np.isclose(bnd, newbnd), f"Length of arc {tag} not preserved"


@pytest.mark.parametrize(
    "dim,serialise",
    [(2, True)],  # [(2, True), (3, True), (3, False)], # FIXME: hang (#136)
    ids=["mmg2d"],  # ids=["mmg2d", "mmg3d, ParMmg"],
)
def test_adapt(dim, serialise):
    """
    Test that we can successfully invoke Mmg and that it changes the DoF count.
    """
    mesh = uniform_mesh(dim)
    dofs = mesh.coordinates.vector().gather().shape
    mp = {
        "dm_plex_metric": {
            "target_complexity": 100.0,
            "p": 1.0,
        }
    }
    metric = uniform_metric(mesh, metric_parameters=mp)
    newmesh = try_adapt(mesh, metric, serialise=serialise)
    assert newmesh.coordinates.vector().gather().shape != dofs


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize(
    "dim,serialise",
    [(2, True)],  # [(2, True), (3, True), (3, False)], # FIXME: hang (#136)
    ids=["mmg2d"],  # ["mmg2d", "mmg3d, ParMmg"],
)
def test_adapt_parallel_np2(dim, serialise):
    """
    Test that we can successfully invoke [Par]Mmg with 2 MPI processes and that it
    changes the DoF count.
    """
    assert COMM_WORLD.size == 2
    test_adapt(dim, serialise=serialise)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize(
    "dim,serialise",
    [(2, True)],  # [(2, True), (3, True), (3, False)], # FIXME: hang (#136)
    ids=["mmg2d"],  # ["mmg2d", "mmg3d, ParMmg"],
)
def test_adapt_parallel_np3(dim, serialise):
    """
    Test that we can successfully invoke [Par]Mmg with 3 MPI processes and that it
    changes the DoF count.
    """
    assert COMM_WORLD.size == 3
    test_adapt(dim, serialise=serialise)


# @pytest.mark.parametrize("dim", [2, 3], ids=["mmg2d", "mmg3d"])  # FIXME: hang (#136)
@pytest.mark.parametrize("dim", [2], ids=["mmg2d"])
def test_enforce_spd_h_min(dim):
    """
    Tests that :meth:`animate.metric.RiemannianMetric.enforce_spd` applies minimum
    magnitudes as expected.
    """
    mesh = uniform_mesh(dim)
    h = 0.1
    metric = uniform_metric(mesh, a=1 / h**2)
    newmesh = try_adapt(mesh, metric)
    num_vertices = newmesh.coordinates.vector().gather().shape[0]
    metric.set_parameters({"dm_plex_metric_h_min": 0.2})  # h_min > h => h := h_min
    metric.enforce_spd(restrict_sizes=True)
    newmesh = try_adapt(mesh, metric)
    assert newmesh.coordinates.vector().gather().shape[0] < num_vertices


# @pytest.mark.parametrize("dim", [2, 3], ids=["mmg2d", "mmg3d"])  # FIXME: hang (#136)
@pytest.mark.parametrize("dim", [2], ids=["mmg2d"])
def test_enforce_spd_h_max(dim):
    """
    Tests that :meth:`animate.metric.RiemannianMetric.enforce_spd` applies maximum
    magnitudes as expected.
    """
    mesh = uniform_mesh(dim)
    h = 0.1
    metric = uniform_metric(mesh, a=1 / h**2)
    newmesh = try_adapt(mesh, metric)
    num_vertices = newmesh.coordinates.vector().gather().shape[0]
    metric.set_parameters({"dm_plex_metric_h_max": 0.05})  # h_max < h => h := h_max
    metric.enforce_spd(restrict_sizes=True)
    newmesh = try_adapt(mesh, metric)
    assert newmesh.coordinates.vector().gather().shape[0] > num_vertices


# Debugging
if __name__ == "__main__":
    test_adapt(3)
