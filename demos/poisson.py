# Steady-state adaptation (Poisson equation)
# ==========================================
#
# In this demo we introduce fundamental ideas of anisotropic mesh adaptation and
# demonstrate how Animate allows us to control the mesh adaptation process.
# We consider the steady-state Poisson equation described in :cite:`Dundovic:2024`.
# The Poisson equation is solved on a unit square domain with homogeneous Dirichlet
# boundary conditions. Using the method of manufactured solutions, the function
# :math:`u(x,y) = (1-e^{-x/\epsilon})(x-1)\sin(\pi y)$, where :math:`\epsilon=0.01`,
# is selected as the exact solution, with the corresponding Poisson equation given by
#
# .. math::
#   \nabla^2 u = \left(\left(2/\epsilon - 1/\epsilon^2\right) e^{-x/\epsilon} - \pi^2 (x-1)\left(1 - e^{-x/\epsilon}\right)\right) \sin(\pi y).
#
# Let us plot the exact solution. ::

from firedrake import *
from firedrake.pyplot import *
import matplotlib.pyplot as plt

epsilon = Constant(0.01)

uniform_mesh = UnitSquareMesh(64, 64)
x, y = SpatialCoordinate(uniform_mesh)
V = FunctionSpace(uniform_mesh, "CG", 2)
u_exact = Function(V).interpolate((1-exp(-x/epsilon))*(x-1)*sin(pi*y))

fig, ax = plt.subplots()
# tripcolor(u_exact, axes=ax, shading="flat")
im = tricontourf(u_exact, axes=ax, cmap="coolwarm", levels=20)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
fig.colorbar(im, ax=ax)
fig.savefig("poisson_exact-solution.jpg", bbox_inches="tight")

# .. figure:: poisson_exact-solution.jpg
#    :figwidth: 80%
#    :align: center
#
# We observe that the solution exhibits a sharp gradient in the :math:`x`-direction near
# the :math:`x=0` boundary since the term :math:`1-e^{-x/\epsilon}` decays rapidly.
# Since the solution exhibits less rapid variation in the :math:`y`-direction compared
# to the :math:`x`-direction, we expect anisotropic mesh adaptation to be beneficial.
# Let us test this.
#
# First, let us define a function that solves the Poisson equation on a given mesh. ::

def solve_poisson(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    f = Function(V)

    eps = 0.01
    f.interpolate(
    - sin(pi * y) * ((2 / eps) * exp(-x / eps) - (1 / eps**2) * (x - 1) * exp(-x / eps)) 
    + pi**2 * sin(pi * y) * ((x - 1) - (x - 1) * exp(-x / eps)))

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    L = f*v*dx
    
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

    u_numerical = Function(V)
    solve(a == L, u_numerical, bcs=bcs)

    return u_numerical

# Now we can solve the Poisson equation on the above-defined uniform mesh. We will also
# compute the error between the numerical solution and the exact solution.

u_numerical = solve_poisson(uniform_mesh)
error = errornorm(u_numerical, u_exact)
print(f"Error on uniform mesh = {error:.2e}")

# In previous demos we have seen how to adapt a mesh using a
# :class:`~animate.metric.RiemannianMetric` object, where we defined the metric from
# analytical expressions. Such approaches may be suitable for problems where we know
# beforehand where we require fine and coarse resolution. A more general approach is to
# adapt the mesh based on the features of the solution field; i.e., based on its
# Hessian, as described in TODO. Animate provides several Hessian recovery methods
# through the :meth:`~animate.metric.RiemannianMetric.compute_hessian` method of the
# :class:`~animate.metric.RiemannianMetric` class.
#
# For example, we compute the Hessian of the above numerical solution as follows. ::

from animate.metric import RiemannianMetric

P1_ten = TensorFunctionSpace(uniform_mesh, "CG", 1)
isotropic_metric = RiemannianMetric(P1_ten)
isotropic_metric.compute_hessian(u_numerical)

# Before adapting the mesh from the metric above, let us further ... We can do this
# using the :meth:`~animate.metric.RiemannianMetric.set_parameters` method. We must
# always specify the target complexity, which will influence the number of elements in
# the adapted mesh (i.e., the higher the target complexity, the more elements in the
# adapted mesh). We can specify many other parameters, such as the maximum tolerated
# anisotropy. Since we want to demonstrate isotropic adaptation, we set the maximum
# tolerated anisotropy to 1. ::

isotropic_metric_parameters = {
    "dm_plex_metric_target_complexity": 200,
    "dm_plex_metric_a_max": 1,
}
isotropic_metric.set_parameters(isotropic_metric_parameters)
isotropic_metric.normalise()

# Note that these parameters are not guaranteed to be satisfied exactly. They certainly
# will not be in this case, since it is impossible to discretise a rectangular domain
# with non-overlapping equilateral triangles.
#
# To analyse the metric, it is useful to visualise its density and anisotropy quotient
# components (see
# `the documentation <https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#geometric-interpretation>`). ::

def plot_metric(metric, figname):
    density, quotients, _ = metric.density_and_quotients()

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    im_density = tripcolor(density, axes=axes[0], norm="log", cmap="coolwarm")
    im_quotients = tripcolor(quotients, axes=axes[1], cmap="coolwarm")
    axes[0].set_title("Metric density")
    axes[1].set_title("Metric anisotropy quotient")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.colorbar(im_density, ax=axes[0], orientation="horizontal")
    fig.colorbar(im_quotients, ax=axes[1], orientation="horizontal")
    fig.savefig(figname, bbox_inches="tight")

plot_metric(isotropic_metric, "poisson_isotropic-metric.jpg")

# .. figure:: poisson_isotropic-metric.jpg
#    :figwidth: 80%
#    :align: center
#
# ...
#
# Finally, let us adapt the mesh and plot it. We will also zoom in on the :math:`x=0`
# boundary.

from animate.adapt import adapt

isotropic_mesh = adapt(uniform_mesh, isotropic_metric)

fig, axes = plt.subplots(1, 2)
triplot(isotropic_mesh, axes=axes[0])
triplot(isotropic_mesh, axes=axes[1])
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[1].set_xlim(0, 0.15)
axes[1].set_ylim(0.3, 0.7)
for ax in axes:
    ax.set_aspect("equal")
fig.suptitle("Isotropic mesh")
fig.savefig("poisson_isotropic-mesh.jpg", bbox_inches="tight")

# .. figure:: poisson_isotropic-mesh.jpg
#    :figwidth: 80%
#    :align: center
#
# ... ::

u_numerical_isotropic = solve_poisson(isotropic_mesh)
V_isotropic = FunctionSpace(isotropic_mesh, "CG", 2)
u_exact_isotropic = Function(V_isotropic).interpolate(u_exact)
error = errornorm(u_numerical_isotropic, u_exact_isotropic)
print(f"Error on isotropic mesh = {error:.2e}")

# ... ::

anisotropic_metric = RiemannianMetric(P1_ten)
anisotropic_metric_parameters = {
    "dm_plex_metric_target_complexity": 200,
    "dm_plex_metric_a_max": 16,
}
anisotropic_metric.set_parameters(anisotropic_metric_parameters)
anisotropic_metric.compute_hessian(u_numerical)
anisotropic_metric.normalise()

plot_metric(anisotropic_metric, "poisson_anisotropic-metric.jpg")

anisotropic_mesh = adapt(uniform_mesh, anisotropic_metric)

fig, axes = plt.subplots(1, 2)
triplot(anisotropic_mesh, axes=axes[0])
triplot(anisotropic_mesh, axes=axes[1])
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[1].set_xlim(0, 0.15)
axes[1].set_ylim(0.3, 0.7)
for ax in axes:
    ax.set_aspect("equal")
fig.suptitle("Anisotropic mesh")
fig.savefig("poisson_anisotropic-mesh.jpg", bbox_inches="tight")

u_numerical_anisotropic = solve_poisson(anisotropic_mesh)
u_exact_anisotropic = Function(FunctionSpace(anisotropic_mesh, "CG", 2)).interpolate(u_exact)
error = errornorm(u_numerical_anisotropic, u_exact_anisotropic)
# error = errornorm(u_numerical_anisotropic, u_exact)
print(f"error after anisotropic adaptation = {error:.2e}")
