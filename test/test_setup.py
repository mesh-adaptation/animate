import ufl
from firedrake.function import Function
from firedrake.functionspace import TensorFunctionSpace
from firedrake.mesh import Mesh, MeshGeometry
from firedrake.utility_meshes import CubeMesh, IntervalMesh, SquareMesh

from animate.metric import RiemannianMetric


def uniform_mesh(dim, n=5, length=1, recentre=False, **kwargs):
    """
    Create a uniform mesh of a specified dimension and size.

    :arg dim: the topological dimension
    :kwarg n: the number of subdivisions in each coordinate direction
    :kwarg l: extent in each direction
    :kwarg recentre: if ``True``, the mesh is re-centred on the origin

    All other keyword arguments are passed to the :func:`SquareMesh` or :func:`CubeMesh`
    constructor.
    """
    if dim == 1:
        mesh = IntervalMesh(n, length, **kwargs)
    elif dim == 2:
        mesh = SquareMesh(n, n, length, **kwargs)
    elif dim == 3:
        mesh = CubeMesh(n, n, n, length, **kwargs)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D")
    if recentre:
        coords = Function(mesh.coordinates)
        coords.interpolate(2 * (coords - ufl.as_vector([0.5 * length] * dim)))
        return Mesh(coords)
    return mesh


def uniform_metric(mesh, a=100.0, metric_parameters=None):
    """
    Create a metric which is just the identity
    matrix scaled by `a` at each vertex.

    :param mesh: the mesh or function space to define the metric upon
    :param a: the scale factor for the identity
    :param: parameters to pass to PETSc's Riemannian metric
    """
    metric_parameters = metric_parameters or {}
    if isinstance(mesh, MeshGeometry):
        function_space = TensorFunctionSpace(mesh, "CG", 1)
    else:
        function_space = mesh
        mesh = function_space.mesh()
    dim = mesh.topological_dimension
    metric = RiemannianMetric(function_space)
    metric.interpolate(a * ufl.Identity(dim))
    metric.set_parameters(metric_parameters)
    return metric
