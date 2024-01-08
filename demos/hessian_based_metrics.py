# Hessian based metrics
# ===========================================
# In this demo we will demonstrate how to adapt
# the mesh to optimally represent a given function
# guided by a metric derived from its Hessian.
#
# We will consider the function:
#
# .. math:: u(x,y) = \exp\left(-\frac{\left(\sqrt{x^2+y^2}-r_0\right)^2}{d^2}\right)
#

import matplotlib.pyplot as plt
import numpy as np
from firedrake import *
from animate import *

mesh = RectangleMesh(100, 100, 1, 1, -1, -1)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
r0 = Constant(0.5)
d = Constant(0.05)
u = interpolate(exp(-((sqrt(x**2 + y**2) - r0) ** 2) / d**2), V)

fig, axes = plt.subplots(figsize=(16, 16))
tricontourf(u, axes=axes)
axes.set_aspect("equal")
axes.set_title("Mesh based on metric1")
fig.show()
fig.savefig("hessian_based_function.jpg")

# To analyse the error from interpolating the analytical expression onto the
# mesh we consider its Taylor expansion around a point :math:`{\mathbf
# x}=(x,y)` over a small distance (vector) :math:`\mathbf h`
#
# .. math::
#
#         u({\mathbf x}+{\mathbf h}) = u({\mathbf x}) + {\mathbf h}^T \nabla
#         u({\mathbf x}) + \tfrac 12 {\mathbf h}^T H({\mathbf x}) {\mathbf h} +
#         \mathcal{O}\left(\|{\mathbf h}\|^3\right)
#
# where :math:`H({\mathbf x})` is the Hessian of :math:`u` in :math:`\mathbf x`
# Within each element, the first two terms can be exactly represented by a
# discrete function in the `"CG", 1` function space `V`.  So it is to be
# expected that the leading order interpolation error is given by the Hessian
# term. If we want to restrtict this approximation of the interpolation error
# to some maximum value :math:`\epsilon`, i.e.
#
# .. math:: {\mathbf h}^T H(\mathbf x) {\mathbf h} \leq \epsilon
#
# (incorporating the :math:`\tfrac 12` into :math:`\epsilon` for convenience).
# Imposing this restriction to the interpolation error within an element with
# edges :math:`\mathbf e` we want
#
# .. math:: {\mathbf e}^T \left[\frac{1}{\epsilon} H({\mathbf x})\right]
#           {\mathbf e} \leq 1
#           :label: metric_edge_bound
#
# This is exactly what we can achieve through metric-based adaptivity by
# choosing a metric
#
# .. math:: \mathcal{M}({\mathbf x}) = \frac{1}{\epsilon} H({\mathbf x})
#
# Thus we see the Hessian provides a natural way to construct an *anisotropic*
# metric that will give *small* edge lenghts in the direction of *large*
# curvature, and vice versa.
#
# There are a number of ways to approximate the Hessian of a given discrete
# function. `animate` provides the `mixed_L2` (default), `L2` and `Clement`
# methods through the :meth:`~.RiemannianMetric.compute_hessian()` method of a
# :class:`~.RiemannianMetric`. After computing the Hessian we scale it by the
# factor :math:`1/\epsilon`.
#
# The result is not yet a valid metric, as a metric is required to be Symmetric
# Positive Definite (SPD).  A Hessian is always symmetric, but not necessarily
# positive. We can turn it into a positive definite tensor however by computing
# its eigendecomposition and replacing any negative eigenvalues, associated
# with negative curvature, by their absolute value. This is done in the
# :meth:`~.RiemannianMetric.enforce_spd()` method. For a positive _definite_
# tensor, the eigenvalues needs to be strictly positive. If, for instance, the
# function is completely flat in some region, its Hessian becomes zero. This
# would allow arbitrarily large edges in :eq:`metric_edge_bound`. Imposing a
# minimum eigenvalue of :math:`1/h_{\text{max}}^2` corresponds to imposing a
# maximum edge length :math:`h_{\text{max}}`.
#
# In the code below this maximum is set by the :code:`"dm_plex_metric_h_max"`
# parameter. In addition we can also impose a _minimum_ edge length through
# :code:`"dm_plex_metric_h_min"`. This is in particular important when trying
# to approximate discontinuities in the solution: the more resolution we start
# off with, the steeper our approximation of the jump becomes, and thus
# derivates can become arbitrarily large, which would correspond to asking for
# arbitrarily small edges. The meaning of :code:`"dm_plex_metric_p": np.inf"`
# will become clear in the next section.  We use the
# :meth:`~.RiemannianMetric.normalise()` method to rescale the metric which in
# turn calls :meth:`~.RiemannianMetric.enforce_spd()` and through
# :code:`restrict_sizes=True` enforces the maximum (and/or minimum) edge length set
# in the metric parameters dictionary.

TP1 = TensorFunctionSpace(mesh, "CG", 1)
metric = RiemannianMetric(TP1)
metric.compute_hessian(u)
eps = 0.05  # upper bound on interpolation error
metric.set_parameters({"dm_plex_metric_h_max": 0.1, "dm_plex_metric_p": np.inf})
metric.normalise(global_factor=1 / eps, restrict_sizes=True)
mesh2 = adapt(mesh, metric)

# plot the adapted mesh
fig, axes = plt.subplots(figsize=(16, 16))
triplot(mesh2, axes=axes)
axes.set_aspect("equal")
axes.set_title("Adapted mesh")
fig.show()
fig.savefig("hessian_based_mesh.jpg")
print(mesh2.num_cells())

#
# .. figure:: hessian_based_mesh.jpg
#    :figwidth: 90%
#    :align: center

# Target complexity and different norms
# -------------------------------------
