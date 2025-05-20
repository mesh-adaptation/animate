# Metric normalisation
# ====================
#
# In this example, we demonstrate metric normalisation in Animate. In particular, we
# explain the meaning of the normalisation parameter :math:`p`.
#
# Consider a Riemannian metric :math:`\mathcal{M}=\{M(x)\}_{x\in\Omega}` defined over a
# domain :math:`\Omega`. We have no guarantee that the scaling of this metric is
# appropriate for use in mesh adaptation for any particular problem. The primary purpose
# of metric normalisation is to rescale appropriately. There are two main ways to do
# this: to rescale such that a target metric complexity is achieved, or to rescale such
# that interpolation error is below a given threshold (assuming that the metric is
# Hessian-based). The former case is more often used in Animate and is used throughout
# this demo.
#
# A naive approach is to rescale as
#
# ..math::
#     :label:`l_infty`
#
#     \widetilde{\mathcal{M}}=
#     \frac{\mathcal{C}_T}{\mathcal{C}(\mathcal{M})}\:\mathcal{M},
#
# where
#
# ..math::
#
#     \mathcal{C}(\mathcal{M})=\int_\Omega\det(M(x))\,\mathrm{d}x
#
# is the complexity of :math:`\mathcal{M}` and :math:`\mathcal{C}_T` is the target
# complexity. This is actually a special case of the more general :math:`L^p`
# normalisation approach (the infinite limit :math:`p\rightarrow\infty`. Before digging
# further into :math:`L^p` normalisation, let's test out the so-called :math:`L^\infty`
# approach described above.
#
# For testing purposes, consider a hyperbolic sensor function defined on the square
# domain :math:`[-1,1]^2`. The sensor functions used was defined in
# :cite:`Olivier:2011` and is interesting because it contains background oscillations,
# in addition to larger-scale features. ::

from firedrake import *
from firedrake.pyplot import *
from animate import *
import matplotlib.pyplot as plt
import numpy as np


def hyperbolic(x, y):
    sn = sin(50 * x * y)
    return conditional(abs(x * y) < 2 * pi / 50, 0.01 * sn, sn)


# Define a square mesh on :math:`[0,2]^2` and subtract 1 from each coordinate to get a
# mesh of the desired domain. ::

n = 100
mesh = SquareMesh(n, n, 2, 2)
coords = Function(mesh.coordinates.function_space())
coords.interpolate(mesh.coordinates - as_vector([1, 1]))
mesh = Mesh(coords)

# Interpolate the hyberbolic sensor function in a :math:`\mathbb{P}1` space defined on
# the initial mesh and plot it. ::

x, y = SpatialCoordinate(mesh)
P1 = FunctionSpace(mesh, "CG", 1)
sensor = Function(P1).interpolate(hyperbolic(x, y))

fig, axes = plt.subplots()
fig.colorbar(tricontourf(sensor, axes=axes, cmap="coolwarm"))
plt.savefig("metric_normalisation-sensor.jpg", bbox_inches="tight")

# .. figure:: metric_normalisation-sensor.jpg
#    :figwidth: 90%
#    :align: center
#
# Now construct a metric based off the Hessian of the sensor function, setting the
# normalisation order :math:`p` to infinity. ::

P1_ten = TensorFunctionSpace(mesh, "CG", 1)
linf_metric = RiemannianMetric(P1_ten)
linf_metric.set_parameters(
    {
        "dm_plex_metric_p": np.inf,
        "dm_plex_metric_target_complexity": 1000.0,
    }
)
linf_metric.compute_hessian(sensor)
linf_metric.normalise()

# Adapt the mesh with respect to the :math:`L^\infty` normalised metric and visualise
# it. ::

fig, axes = plt.subplots()
triplot(adapt(mesh, linf_metric), axes=axes)
axes.set_aspect("equal")
axes.axis(False)
plt.savefig("metric_normalisation-linf_mesh.jpg", bbox_inches="tight")

# .. figure:: metric_normalisation-linf_mesh.jpg
#    :figwidth: 90%
#    :align: center

# TODO: Describe L^p normalisation
# TODO: Adapt with L^p normalisation for p=1,2,4
# TODO: Add Olivier citation
