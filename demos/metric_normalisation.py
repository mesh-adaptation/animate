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
#     \mathcal{M}_{L^\infty}=
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
# For testing purposes, consider a multi-scale sensor function defined on the square
# domain :math:`[-1,1]^2`. This sensor function was defined in :cite:`Olivier:2011` and
# is interesting because it contains background oscillations, in addition to
# larger-scale features. The idea of multi-scale mesh adaptation is to vary the spatial
# resolution such that we are able to capture multiple such scales at the same time.::

from firedrake import *
from firedrake.pyplot import *
from animate import *
import matplotlib.pyplot as plt
import numpy as np


def sensor_fn(x, y):
    return 0.1 * ufl.sin(50 * x) + ufl.atan(0.1 / (ufl.sin(5 * y) - 2 * x))


# To quote :cite:`Wallwork:2021` p.116, the sensor function contains a 'long wavelength,
# low frequency oscillation in the :math:`y`-direction, as well as a short wavelength,
# high frequency oscillation in the :math:`x`-direction. The long wavelength profile is
# more prominent, as it is bounded by :math:`\pm\frac{\pi}{2}\approx\pm1.5708`, whereas
# the short wavelength profile is bounded by :math:`\pm0.1`.'
#
# Define a square mesh on :math:`[0,2]^2` and subtract 1 from each coordinate to get a
# mesh of the desired domain. ::

n = 50
base_mesh = SquareMesh(n, n, 2, 2)
coords = Function(base_mesh.coordinates.function_space())
coords.interpolate(base_mesh.coordinates - as_vector([1, 1]))
base_mesh = Mesh(coords)

# Interpolate the sensor function in a :math:`\mathbb{P}1` space defined on the initial
# mesh and plot it. ::

x, y = SpatialCoordinate(base_mesh)
P1 = FunctionSpace(base_mesh, "CG", 1)
sensor = Function(P1).interpolate(sensor_fn(x, y))

fig, axes = plt.subplots()
fig.colorbar(tricontourf(sensor, axes=axes, cmap="coolwarm"))
plt.savefig("metric_normalisation-sensor.jpg", bbox_inches="tight")

# .. figure:: metric_normalisation-sensor.jpg
#    :figwidth: 90%
#    :align: center
#
# Define a function for constructing a metric based off the Hessian of the sensor
# function, given some normalisation order value :math:`p`. ::

target_complexity = 10000.0


def lp_metric(mesh, p):
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(
        {
            "dm_plex_metric_p": p,
            "dm_plex_metric_target_complexity": target_complexity,
        }
    )
    x, y = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    metric.compute_hessian(Function(P1).interpolate(sensor_fn(x, y)))
    metric.normalise()
    return metric


# Adapt the mesh several times with respect to the :math:`L^\infty` normalised metric
# and visualise it. ::

num_adapt = 4
linf_mesh = base_mesh
for _ in range(num_adapt):
    linf_mesh = adapt(linf_mesh, lp_metric(linf_mesh, np.inf))

fig, axes = plt.subplots()
kwargs = {
    "interior_kw": {"linewidth": 0.2},
    "boundary_kw": {"linewidth": 0.2, "color": "k"},
}
triplot(linf_mesh, axes=axes, **kwargs)
axes.set_aspect("equal")
axes.axis(False)
plt.savefig("metric_normalisation-linf_mesh.jpg", bbox_inches="tight")

# .. figure:: metric_normalisation-linf_mesh.jpg
#    :figwidth: 90%
#    :align: center
#
# The general formula for :math:`L^p` normalisation is given by
#
# .. math::
#     :label: lp_metric
#
#     \mathcal M_{L^p}:=
#     \mathcal C_T^{\frac2n}
#     \:\left(\int_{\Omega}\mathrm{det}(\underline{\mathbf M})^{\frac p{2p+n}}\;\mathrm dx\right)^{-\frac2n}
#     \:\mathrm{det}(\mathcal M)^{-\frac1{2p+n}}
#     \:\mathcal M,
#
# where :math:`p\in[1,\infty)`.
#
# ..rubric:: Exercise
#
#     Convince yourself that
#     :math:`\lim_{p\rightarrow\infty}\mathcal{M}_{L^p}=\mathcal{M}_{L^\infty}`.
#
# We've tried :math:`L^\infty` normalisation - the upper limit of the range of
# acceptable values for :math:`p`. Now let's try the other end of the scale:
# :math:`L^1` normalisation. ::

l1_mesh = base_mesh
for _ in range(num_adapt):
    l1_mesh = adapt(l1_mesh, lp_metric(l1_mesh, 1.0))

fig, axes = plt.subplots()
triplot(l1_mesh, axes=axes, **kwargs)
axes.set_aspect("equal")
axes.axis(False)
plt.savefig("metric_normalisation-l1_mesh.jpg", bbox_inches="tight")

# .. figure:: metric_normalisation-l1_mesh.jpg
#    :figwidth: 90%
#    :align: center
#
# TODO: Observations
#
# Another common normalisation order is :math:`p=2`.

l2_mesh = base_mesh
for _ in range(num_adapt):
    l2_mesh = adapt(l2_mesh, lp_metric(l2_mesh, 2.0))

fig, axes = plt.subplots()
triplot(l2_mesh, axes=axes, **kwargs)
axes.set_aspect("equal")
axes.axis(False)
plt.savefig("metric_normalisation-l2_mesh.jpg", bbox_inches="tight")

# .. figure:: metric_normalisation-l2_mesh.jpg
#    :figwidth: 90%
#    :align: center
#
# TODO: Observations
#
# ..rubric:: Exercise
#
#    Experiment with other intermediate normalisation orders such as :math:`p=4` or
#    :math:`p=10` and consider the different ways in which the sensor's features
#    manifest in the adapted meshes.
#
# This demo can also be accessed as a `Python script <metric_normalisation.py>`__.
