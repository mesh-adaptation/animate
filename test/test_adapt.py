from test_setup import *
from petsc4py import PETSc
import pytest
import numpy as np
import os


def load_mesh(fname):
    """
    Load a mesh in gmsh format.

    :param fname: file name, without the .msh extension
    """
    venv = os.environ.get("VIRTUAL_ENV")
    mesh_dir = os.path.join(venv, "src", "firedrake", "tests", "meshes")
    return Mesh(os.path.join(mesh_dir, fname + ".msh"))


def try_adapt(mesh, metric):
    """
    Attempt to invoke PETSc's mesh adaptation functionality
    and xfail if it is not installed.

    :param mesh: the current mesh
    :param metric: the :class:`RiemannianMetric` instance
    :return: the adapted mesh w.r.t. the metric
    """
    try:
        return adapt(mesh, metric)
    except PETSc.Error as exc:
        if exc.ierr == 63:
            pytest.xfail("No mesh adaptation tools are installed")
        else:
            raise Exception(f"PETSc error code {exc.ierr}")


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_no_adapt(dim):
    """
    Test that we can turn off all of Mmg's
    mesh adaptation operations.
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
    newmesh = try_adapt(mesh, metric)
    assert newmesh.coordinates.vector().gather().shape == dofs


@pytest.mark.parallel(nprocs=2)
def test_no_adapt_parallel():
    """
    Test that we can turn off all of ParMmg's
    mesh adaptation operations.
    """
    test_no_adapt(3)


@pytest.mark.parametrize(
    "meshname",
    [
        "annulus",
        "cell-sets",
        "square_with_embedded_line",
    ],
)
def test_preserve_cell_tags_2d(meshname):
    """
    Test that cell tags are preserved
    after mesh adaptation.
    """
    mesh = load_mesh(meshname)
    metric = uniform_metric(mesh)
    newmesh = try_adapt(mesh, metric)

    tags = set(mesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    newtags = set(newmesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    assert tags == newtags, "Cell tags do not match"

    one = Constant(1.0)
    for tag in tags:
        bnd = assemble(one * dx(tag, domain=mesh))
        newbnd = assemble(one * dx(tag, domain=newmesh))
        assert np.isclose(bnd, newbnd), f"Area of region {tag} not preserved"


@pytest.mark.parametrize(
    "meshname",
    [
        "annulus",
        "circle_in_square",
    ],
)
def test_preserve_facet_tags_2d(meshname):
    """
    Test that facet tags are preserved
    after mesh adaptation.
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
        bnd = assemble(one * ds(tag, domain=mesh))
        newbnd = assemble(one * ds(tag, domain=newmesh))
        assert np.isclose(bnd, newbnd), f"Length of arc {tag} not preserved"


def test_adapt_3d():
    """
    Test that we can successfully invoke
    Mmg3d and that it changes the DoF count.
    """
    mesh = uniform_mesh(3)
    dofs = mesh.coordinates.vector().gather().shape
    mp = {
        "dm_plex_metric": {
            "target_complexity": 100.0,
            "p": 1.0,
        }
    }
    metric = uniform_metric(mesh, metric_parameters=mp)
    newmesh = try_adapt(mesh, metric)
    assert newmesh.coordinates.vector().gather().shape != dofs


@pytest.mark.parallel(nprocs=2)
def test_adapt_parallel_3d_np2():
    """
    Test that we can successfully invoke ParMmg with 2 MPI processes and that
    it changes the DoF count.
    """
    test_adapt_3d()


@pytest.mark.parallel(nprocs=3)
def test_adapt_parallel_3d_np3():
    """
    Test that we can successfully invoke ParMmg with 3 MPI processes and that
    it changes the DoF count.
    """
    test_adapt_3d()


def test_enforce_spd_h_min(dim):
    """
    Tests that the :meth:`enforce_spd` method applies minimum magnitudes as expected.
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


def test_enforce_spd_h_max(dim):
    """
    Tests that the :meth:`enforce_spd` method applies maximum magnitudes as expected.
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
