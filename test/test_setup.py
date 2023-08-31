from firedrake import *
from animate import *


def uniform_mesh(dim, n=5, l=1, recentre=False, **kwargs):
    """
    Create a uniform mesh of a specified dimension and size.

    :arg dim: the topological dimension
    :kwarg n: the number of subdivisions in each coordinate direction
    :kwarg l: extent in each direction
    :kwarg recentre: if ``True``, the mesh is re-centred on the origin

    All other keyword arguments are passed to the :func:`SquareMesh` or :func:`CubeMesh`
    constructor.
    """
    if dim == 2:
        mesh = SquareMesh(n, n, l, **kwargs)
    elif dim == 3:
        mesh = CubeMesh(n, n, n, l, **kwargs)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D")
    if recentre:
        coords = Function(mesh.coordinates)
        coords.interpolate(2 * (coords - as_vector([0.5 * l] * dim)))
        return Mesh(coords)
    return mesh


def uniform_metric(mesh, a=100.0, metric_parameters={}):
    """
    Create a metric which is just the identity
    matrix scaled by `a` at each vertex.

    :param mesh: the mesh or function space to define the metric upon
    :param a: the scale factor for the identity
    :param: parameters to pass to PETSc's Riemannian metric
    """
    if isinstance(mesh, firedrake.mesh.MeshGeometry):
        function_space = TensorFunctionSpace(mesh, "CG", 1)
    else:
        function_space = mesh
        mesh = function_space.mesh()
    dim = mesh.topological_dimension()
    metric = RiemannianMetric(function_space)
    metric.interpolate(a * Identity(dim))
    metric.set_parameters(metric_parameters)
    return metric


def norm(v, norm_type="L2", condition=Constant(1.0), boundary=False):
    r"""
    Overload :func:`firedrake.norms.norm` to allow for :math:`\ell^p` norms.

    Note that this version is case sensitive, i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg v: the :class:`firedrake.function.Function` to take the norm of
    :kwarg norm_type: choose from ``'l1'``, ``'l2'``, ``'linf'``, ``'L2'``, ``'Linf'``,
        ``'H1'``, ``'Hdiv'``, ``'Hcurl'``, or any ``'Lp'`` with :math:`p >= 1`.
    :kwarg condition: a UFL condition for specifying a subdomain to compute the norm
        over
    :kwarg boundary: should the norm be computed over the domain boundary?
    """
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
        raise NotImplementedError(f"lp norm of order {norm_type[1:]} not supported.")
    else:
        dX = ds if boundary else dx
        if norm_type.startswith("L"):
            try:
                p = int(norm_type[1:])
            except Exception:
                raise ValueError(f"Don't know how to interpret '{norm_type}' norm.")
            if p < 1:
                raise ValueError(f"'{norm_type}' norm does not make sense.")
            integrand = inner(v, v)
        elif norm_type.lower() == "h1":
            integrand = inner(v, v) + inner(grad(v), grad(v))
        elif norm_type.lower() == "hdiv":
            integrand = inner(v, v) + div(v) * div(v)
        elif norm_type.lower() == "hcurl":
            integrand = inner(v, v) + inner(curl(v), curl(v))
        else:
            raise ValueError(f"Unknown norm type '{norm_type}'.")
        return firedrake.assemble(condition * integrand ** (p / 2) * dX) ** (1 / p)


def errornorm(u, uh, norm_type="L2", **kwargs):
    r"""
    Overload :func:`firedrake.norms.errornorm` to allow for :math:`\ell^p` norms.

    Note that this version is case sensitive, i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg u: the 'true' value
    :arg uh: the approximation of the 'truth'
    :kwarg norm_type: choose from ``'l1'``, ``'l2'``, ``'linf'``, ``'L2'``, ``'Linf'``,
        ``'H1'``, ``'Hdiv'``, ``'Hcurl'``, or any ``'Lp'`` with :math:`p >= 1`.
    :kwarg condition: a UFL condition for specifying a subdomain to compute the norm
        over
    :kwarg boundary: should the norm be computed over the domain boundary?
    """
    if len(u.ufl_shape) != len(uh.ufl_shape):
        raise RuntimeError("Mismatching rank between u and uh.")

    if not isinstance(uh, Function):
        raise TypeError(f"uh should be a Function, is a {type(uh).__name__}.")
    if norm_type[0] == "l":
        if not isinstance(u, Function):
            raise TypeError(f"u should be a Function, is a {type(u).__name__}.")

    if isinstance(u, Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            logging.warning("Degree of exact solution less than approximation degree")

    # Case 1: point-wise norms
    if norm_type[0] == "l":
        v = u
        v -= uh

    # Case 2: UFL norms for mixed function spaces
    elif hasattr(uh.function_space(), "num_sub_spaces"):
        if norm_type == "L2":
            vv = [uu - uuh for uu, uuh in zip(u.subfunctions, uh.subfunctions)]
            dX = ds if kwargs.get("boundary", False) else dx
            return sqrt(assemble(sum([inner(v, v) for v in vv]) * dX))
        else:
            raise NotImplementedError(
                f"Norm type '{norm_type}' not supported for mixed spaces."
            )

    # Case 3: UFL norms for non-mixed spaces
    else:
        v = u - uh

    return norm(v, norm_type=norm_type, **kwargs)
