"""
Sensor functions defined in [Olivier 2011].

Olivier, Géraldine. Anisotropic metric-based mesh
adaptation for unsteady CFD simulations involving
moving geometries. Diss. 2011.
"""
from ufl import *
from test_setup import uniform_mesh


__all__ = ["bowl", "hyperbolic", "multiscale", "interweaved", "mesh_for_sensors"]


def bowl(*coords):
    return 0.5 * sum([xi**2 for xi in coords])


def hyperbolic(x, y):
    return conditional(
        abs(x * y) < 2 * pi / 50, 0.01 * sin(50 * x * y), sin(50 * x * y)
    )


def multiscale(x, y):
    return 0.1 * sin(50 * x) + atan(0.1 / (sin(5 * y) - 2 * x))


def interweaved(x, y):
    return atan(0.1 / (sin(5 * y) - 2 * x)) + atan(0.5 / (sin(3 * y) - 7 * x))


def mesh_for_sensors(dim, n):
    return uniform_mesh(dim, n, l=2, recentre=True)
