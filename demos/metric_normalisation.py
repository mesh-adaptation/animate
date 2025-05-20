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
from animate import *


def hyperbolic(x, y):
    sn = sin(50 * x * y)
    return conditional(abs(x * y) < 2 * pi / 50, 0.01 * sn, sn)


# Define a square mesh on :math:`[0,2]^2` and subtract 1 from each coordinate to get a
# mesh of the desired domain. ::

mesh = SquareMesh(20, 20, 2, 2)
mesh.coordinates[:] -= 1
mesh = Mesh(mesh.coordinates)

# TODO: Plot sensor function
# TODO: Adapt with L^infty normalisation
# TODO: Describe L^p normalisation
# TODO: Adapt with L^p normalisation for p=1,2,4
# TODO: Add Olivier citation
