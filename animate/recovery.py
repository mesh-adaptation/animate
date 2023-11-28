"""
Driver functions for derivative recovery.
"""
from .interpolation import clement_interpolant
from .quality import QualityMeasure, include_dir
from .math import construct_basis
import firedrake
from firedrake.petsc import PETSc
import os
from pyop2 import op2
import ufl

__all__ = ["recover_gradient_l2", "recover_hessian_clement", "recover_boundary_hessian"]


def get_metric_kernel(func: str, dim: int) -> op2.Kernel:
    """
    Helper function to easily pass Eigen kernels
    for metric utilities to Firedrake via PyOP2.

    :arg func: function name
    :arg dim: spatial dimension
    """
    pwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "cxx"))
    with open(os.path.join(pwd, f"metric{dim}d.cxx"), "r") as code:
        return op2.Kernel(code.read(), func, cpp=True, include_dirs=include_dir)


@PETSc.Log.EventDecorator()
def recover_gradient_l2(f, target_space=None):
    r"""
    Recover the gradient of a scalar or vector field using :math:`L^2` projection.

    :arg f: the scalar field whose derivatives we seek to recover
    :kwarg mesh: the underlying mesh
    :kwarg target_space: the :func:`firedrake.functionspace.FunctionSpace`
        recovered gradient should live in
    """
    if target_space is None:
        if not isinstance(f, firedrake.Function):
            raise ValueError(
                "If a target space is not provided then the input must be a Function."
            )
        degree = max(1, f.ufl_element().degree() - 1)
        mesh = f.function_space().mesh()
        rank = len(f.ufl_element().value_shape)
        if rank == 0:
            target_space = firedrake.VectorFunctionSpace(mesh, "CG", degree)
        elif rank == 1:
            target_space = firedrake.TensorFunctionSpace(mesh, "CG", degree)
        else:
            raise ValueError(
                "L2 projection can only be used to compute gradients of scalar or"
                f" vector Functions, not Functions of rank {rank}."
            )
    return firedrake.project(ufl.grad(f), target_space)


@PETSc.Log.EventDecorator()
def recover_hessian_clement(f):
    r"""
    Recover the gradient and Hessian of a scalar field using two applications of
    Clement interpolation.

    Note that if the field is of degree 2 then projection will be used to obtain the
    gradient. If the field is of degree 3 or greater then projection will be used
    for the Hessian recovery, too.

    :arg f: the scalar field whose derivatives we seek to recover
    """
    if not isinstance(f, firedrake.Function):
        raise ValueError(
            "Clement interpolation can only be used to compute gradients of"
            " Lagrange Functions of degree > 0."
        )
    family = f.ufl_element().family()
    degree = f.ufl_element().degree()
    if family not in ("Lagrange", "Discontinuous Lagrange") or degree == 0:
        raise ValueError(
            "Clement interpolation can only be used to compute gradients of"
            " Lagrange Functions of degree > 0."
        )
    mesh = f.function_space().mesh()

    # Recover gradient
    if degree <= 1:
        V = firedrake.VectorFunctionSpace(mesh, "DG", 0)
        g = clement_interpolant(firedrake.project(ufl.grad(f), V))
    else:
        V = firedrake.VectorFunctionSpace(mesh, "DG", degree - 1)
        g = recover_gradient_l2(f, target_space=V)

    # Recover Hessian
    if degree <= 2:
        W = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        H = clement_interpolant(firedrake.project(ufl.grad(g), W))
    else:
        W = firedrake.TensorFunctionSpace(mesh, "DG", degree - 2)
        H = recover_gradient_l2(g, target_space=W)
    return g, H


@PETSc.Log.EventDecorator()
def recover_boundary_hessian(f, method="Clement", target_space=None, **kwargs):
    """
    Recover the Hessian of a scalar field on the domain boundary.

    :arg f: field to recover over the domain boundary
    :kwarg method: choose from 'mixed_L2' and 'Clement'
    :kwarg target_space: the :func:`firedrake.functionspace.TensorFunctionSpace`
        in which the metric will exist
    """

    mesh = ufl.domain.extract_unique_domain(f)
    d = mesh.topological_dimension()
    assert d in (2, 3)

    # Apply Gram-Schmidt to get tangent vectors
    n = ufl.FacetNormal(mesh)
    ns = construct_basis(n)
    s = ns[1:]
    ns = ufl.as_vector(ns)

    # Setup
    P1 = firedrake.FunctionSpace(mesh, "CG", 1)
    P1_ten = target_space or firedrake.TensorFunctionSpace(mesh, "CG", 1)
    assert P1_ten.ufl_element().family() == "Lagrange"
    assert P1_ten.ufl_element().degree() == 1
    boundary_tag = kwargs.get("boundary_tag", "on_boundary")
    Hs = firedrake.TrialFunction(P1)
    v = firedrake.TestFunction(P1)
    l2_proj = [[firedrake.Function(P1) for i in range(d - 1)] for j in range(d - 1)]
    h = firedrake.interpolate(
        ufl.CellDiameter(mesh), firedrake.FunctionSpace(mesh, "DG", 0)
    )
    h = firedrake.Constant(1 / h.vector().gather().max() ** 2)
    sp = {
        "ksp_type": "gmres",
        "ksp_gmres_restart": 20,
        "pc_type": "ilu",
    }

    if method == "mixed_L2":
        # Arbitrary value on domain interior
        a = v * Hs * ufl.dx
        L = v * h * ufl.dx

        # Hessian on boundary
        nullspace = firedrake.VectorSpaceBasis(constant=True)
        for j, s1 in enumerate(s):
            for i, s0 in enumerate(s):
                bcs = []
                for tag in mesh.exterior_facets.unique_markers:
                    a_bc = v * Hs * ufl.ds(tag)
                    L_bc = (
                        -ufl.dot(s0, ufl.grad(v))
                        * ufl.dot(s1, ufl.grad(f))
                        * ufl.ds(tag)
                    )
                    bcs.append(firedrake.EquationBC(a_bc == L_bc, l2_proj[i][j], tag))
                firedrake.solve(
                    a == L,
                    l2_proj[i][j],
                    bcs=bcs,
                    nullspace=nullspace,
                    solver_parameters=sp,
                )

    elif method == "Clement":
        P0_vec = firedrake.VectorFunctionSpace(mesh, "DG", 0)
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        P1_vec = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        H = firedrake.Function(P1_ten)
        p0test = firedrake.TestFunction(P0_vec)
        p1test = firedrake.TestFunction(P1)
        fa = QualityMeasure(mesh, python=True)("facet_area")

        source = firedrake.assemble(ufl.inner(p0test, ufl.grad(f)) / fa * ufl.ds)

        # Recover gradient
        c = clement_interpolant(source, boundary=True, target_space=P1_vec)

        # Recover Hessian
        H += clement_interpolant(
            firedrake.interpolate(ufl.grad(c), P0_ten),
            boundary=True,
            target_space=P1_ten,
        )

        # Compute tangential components
        for j, s1 in enumerate(s):
            for i, s0 in enumerate(s):
                l2_proj[i][j] = firedrake.Function(P1)
                l2_proj[i][j].dat.data_with_halos[:] = firedrake.assemble(
                    p1test * ufl.dot(ufl.dot(s0, H), s1) / fa * ufl.ds
                ).dat.data_with_halos
                # TODO: Avoid accessing .dat.data_with_halos
    else:
        raise ValueError(
            f"Recovery method '{method}' not supported for Hessians on the boundary."
        )

    # Construct tensor field
    Hbar = firedrake.Function(P1_ten)
    if d == 2:
        Hsub = firedrake.interpolate(abs(l2_proj[0][0]), P1)
        H = ufl.as_matrix([[h, 0], [0, Hsub]])
    else:
        fs = firedrake.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2))
        Hsub = firedrake.Function(fs)
        Hsub.interpolate(
            ufl.as_matrix(
                [[l2_proj[0][0], l2_proj[0][1]], [l2_proj[1][0], l2_proj[1][1]]]
            )
        )

        # Enforce SPD
        metric = firedrake.Function(fs)
        op2.par_loop(
            get_metric_kernel("metric_from_hessian", 2),
            fs.node_set,
            metric.dat(op2.RW),
            Hsub.dat(op2.READ),
        )
        Hsub.assign(metric)
        # TODO: Could this be supported using RiemannianMetric.enforce_spd?

        # Construct Hessian
        H = ufl.as_matrix(
            [[h, 0, 0], [0, Hsub[0, 0], Hsub[0, 1]], [0, Hsub[1, 0], Hsub[1, 1]]]
        )

    # Arbitrary value on domain interior
    sigma = firedrake.TrialFunction(P1_ten)
    tau = firedrake.TestFunction(P1_ten)
    a = ufl.inner(tau, sigma) * ufl.dx
    L = ufl.inner(tau, h * ufl.Identity(d)) * ufl.dx

    # Boundary values imposed as in [Loseille et al. 2011]
    a_bc = ufl.inner(tau, sigma) * ufl.ds
    L_bc = ufl.inner(tau, ufl.dot(ufl.transpose(ns), ufl.dot(H, ns))) * ufl.ds
    bcs = firedrake.EquationBC(a_bc == L_bc, Hbar, boundary_tag)
    firedrake.solve(a == L, Hbar, bcs=bcs, solver_parameters=sp)
    return Hbar
