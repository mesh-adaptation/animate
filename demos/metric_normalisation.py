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
# complexity.
