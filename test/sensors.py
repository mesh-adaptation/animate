"""
Sensor functions defined in :cite`Olivier:2011`.
"""

import ufl
from test_setup import uniform_mesh

__all__ = ["bowl", "hyperbolic", "multiscale", "interweaved", "mesh_for_sensors"]


def bowl(*coords):
    """Quadratic bowl sensor function in arbitrary dimensions."""
    return 0.5 * sum([xi**2 for xi in coords])


def hyperbolic(x, y):
    """Hyperbolic sensor function in 2D."""
    sn = ufl.sin(50 * x * y)
    return ufl.conditional(abs(x * y) < 2 * ufl.pi / 50, 0.01 * sn, sn)


def multiscale(x, y):
    """Multi-scale sensor function in 2D."""
    return 0.1 * ufl.sin(50 * x) + ufl.atan(0.1 / (ufl.sin(5 * y) - 2 * x))


def interweaved(x, y):
    """Interweaved sensor function in 2D."""
    return ufl.atan(0.1 / (ufl.sin(5 * y) - 2 * x)) + ufl.atan(
        0.5 / (ufl.sin(3 * y) - 7 * x)
    )


def mesh_for_sensors(dim, n):
    r"""
    Uniform mesh of :math:`[-1, 1]^{\mathrm{dim}}`.

    :arg dim: spatial dimension
    :type dim: :class:`int`
    :arg n: number of elements in each direction
    :type n: :class:`int`
    """
    return uniform_mesh(dim, n, 2, recentre=True)
