# 'Ping pong' interpolation experiment
# ====================================
#
# In this demo, we perform a 'ping poing test', which amounts to interpolating a given
# sensor function repeatedly between the two meshes using a particular interpolation
# method. The purpose of this experiment is to assess the properties of different
# interpolation methods. ::

import os

import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tricontourf, triplot

from animate import *

# Consider two different meshes of the unit square: mesh A, which is laid out
# :math:`20\times25` with diagonals from top left to bottom right and mesh B
# :math:`20\times20` with diagonals from bottom left to top right. ::

mesh_A = UnitSquareMesh(20, 25, diagonal="left")
mesh_B = UnitSquareMesh(20, 20, diagonal="right")

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
for i, mesh in enumerate((mesh_A, mesh_B)):
    triplot(mesh, axes=axes[i])
    axes[i].set_title(f"Mesh {'B' if i else 'A'}")
    axes[i].axis(False)
    axes[i].set_aspect(1)
plt.savefig("ping_pong-meshes.jpg")

# Define the sensor function
#
# ..math::
#   f(x,y) = \sin(\pi x) \sin(\pi y).
#
# Let's plot the sensor function as represented in :math:`\mathbb{P}1` spaces on mesh A.
# ::

V_A = FunctionSpace(mesh_A, "CG", 1)
V_B = FunctionSpace(mesh_B, "CG", 1)
x, y = SpatialCoordinate(mesh_A)
sensor = Function(V_A).interpolate(sin(pi * x) * sin(pi * y))

fig, axes = plt.subplots()
tricontourf(sensor, axes=axes)
axes.set_title("Source function")
axes.axis(False)
axes.set_aspect(1)
plt.savefig("ping_pong-source_function.jpg")

# To start with, let's consider the `interpolate` approach, which point evaluates the
# input field at the locations where degrees of freedom of the target function space.
# We run the experiment for 100 iterations and track three quantities as the iterations
# progress: the integral of the field, its global minimum, and its global maximum. ::

niter = 1 if os.environ.get("ANIMATE_REGRESSION_TEST") else 100
initial_integral = assemble(sensor * dx)
initial_min = sensor.vector().gather().min()
initial_max = sensor.vector().gather().max()
quantities = {
    "integral": {"interpolate": [initial_integral]},
    "minimum": {"interpolate": [initial_min]},
    "maximum": {"interpolate": [initial_max]},
}
f_interp = Function(V_A).assign(sensor)
for _ in range(niter):
    f_interp = interpolate(interpolate(f_interp, V_B), V_A)
    quantities["integral"]["interpolate"].append(assemble(f_interp * dx))
    quantities["minimum"]["interpolate"].append(f_interp.vector().gather().min())
    quantities["maximum"]["interpolate"].append(f_interp.vector().gather().max())

fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
for i, (key, subdict) in enumerate(quantities.items()):
    axes[i].plot(subdict["interpolate"])
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(key.capitalize())
    axes[i].grid(True)
plt.savefig("ping_pong-quantities_interpolate.jpg")

# .. figure:: ping_pong-quantities_interpolate.jpg
#    :figwidth: 90%
#    :align: center
#
# The first plot shows the integral of the field (the 'mass') as the interpolation
# iterations progress. Note that the mass drops steeply, before leveling out after
# around 50 iterations. Minimum values are perfectly maintained for this example,
# staying at zero for all iterations. Maximum values, however, decrease for several
# iterations before leveling out after around 30 iterations.
#
# In this example, we observe two general properties of the point interpolation method:
#
# 1. It is not conservative.
# 2. It does not introduce new extrema.
#
# To elaborate on the second point, whilst the maximum value does decrease, new maxima
# are not introduced. Similarly, no new minima are introduced.
#
# Next, we move on to the `project` approach, which uses the concept of a 'supermesh'
# to set up a conservative projection operator. The clue is in the name: we expect this
# approach to conserve 'mass', i.e., the integral of the field. In fact, the approach
# is designed to satisfy this property. Let :math:`f\in V_A` denote the source field.
# Then the projection operator, :math:`\Pi`, is chosen such that
#
# ..math::
#
#   \int_\Omega f\;\mathrm{d}x = \int_\Omega \Pi(f)\;\mathrm{d}x.
#
# This is achieved by solving the equation
#
# ..math::
#
#   \underline{\mathbf{M}_B} \boldsymbol{\pi} = \underline{\mathbf{M}_{BA}} \mathbf{f},
#
# for :math:`\boldsymbol{\pi}` - the vector of data underlying :math:`\Pi(f)` - where
# :math:`\mathbf{f}` is the vector of data underlying :math:`f`,
# :math:`\underline{\mathbf{M}_B}` is the mass matrix for the target space, :math:`V_B`,
# and :math:`\underline{\mathbf{M}_{BA}}` is the mixed mass matrix mapping from the
# source space :math:`V_A` to the target space. ::

quantities["integral"]["project"] = [initial_integral]
quantities["minimum"]["project"] = [initial_min]
quantities["maximum"]["project"] = [initial_max]
f_proj = Function(V_A).assign(sensor)
for _ in range(niter):
    f_proj = project(project(f_proj, V_B), V_A)
    quantities["integral"]["project"].append(assemble(f_proj * dx))
    quantities["minimum"]["project"].append(f_proj.vector().gather().min())
    quantities["maximum"]["project"].append(f_proj.vector().gather().max())

fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
for i, (key, subdict) in enumerate(quantities.items()):
    axes[i].plot(subdict["interpolate"], label="Interpolate")
    axes[i].plot(subdict["project"], label="Project")
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(key.capitalize())
    axes[i].grid(True)
axes[1].legend()
plt.savefig("ping_pong-quantities_project.jpg")

# .. figure:: ping_pong-quantities_project.jpg
#    :figwidth: 90%
#    :align: center
#
# The first subfigure shows that mass is indeed conserved by the `project` approach,
# unlike the `interpolate` approach. However, contrary to the `interpolate` approach
# not introducing new extrema, the `project` approach can be seen to introduce new
# minimum values. No new maximum values are introduced, although the maximum values do
# decrease slightly.
#
# To summarise, the `project` approach:
#
# 1. It is conservative.
# 2. It may introduce new extrema.
#
# Note that neither of the approaches considered thus far provide an interpolation
# method that is both conservative and does not introduce new extrema. In the case of
# :math:`\mathbb P1` fields specifically, it is possible to achieve this using 'mass
# lumping'. Recall the linear system above. Lumping amounts to replacing the mass matrix
# :math:`\underline{\mathbf{M}_B}` with a diagonal matrix, whose entries correspond to
# the sums over the corresponding mass matrix rows.
#
# Whilst lumping allows us to satisfy the two desired properties, it tends to add a
# significant amount of artifical diffusion. To remedy this, we apply a post-processing
# algorithm, which limits the amount of diffusion. (See :cite:`FPP+:2009` for details.)
#
# Lumping and bounding can be used in Animate by passing passing `bounded=True` to the
# `project` function. ::

quantities["integral"]["bounded"] = [initial_integral]
quantities["minimum"]["bounded"] = [initial_min]
quantities["maximum"]["bounded"] = [initial_max]
f_bounded = Function(V_A).assign(sensor)
for _ in range(niter):
    f_bounded = project(project(f_bounded, V_B, bounded=True), V_A, bounded=True)
    quantities["integral"]["bounded"].append(assemble(f_bounded * dx))
    quantities["minimum"]["bounded"].append(f_bounded.vector().gather().min())
    quantities["maximum"]["bounded"].append(f_bounded.vector().gather().max())

fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
for i, (key, subdict) in enumerate(quantities.items()):
    axes[i].plot(subdict["interpolate"], label="Interpolate")
    axes[i].plot(subdict["project"], label="Project")
    axes[i].plot(subdict["bounded"], label="Minimally diffusive project")
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(key.capitalize())
    axes[i].grid(True)
axes[1].legend()
plt.savefig("ping_pong-quantities_bounded.jpg")


# To check that the interpolants still resemble the sensor function after 100
# iterations, we plot the four final fields. ::

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
colourbars = []
colourbars.append(fig.colorbar(tricontourf(sensor, axes=axes[0][0])))
axes[0][0].set_title("Source function")
colourbars.append(fig.colorbar(tricontourf(f_interp, axes=axes[0][1])))
axes[0][1].set_title("Interpolate")
colourbars.append(fig.colorbar(tricontourf(f_proj, axes=axes[1][0])))
axes[1][0].set_title("Project")
colourbars.append(fig.colorbar(tricontourf(f_bounded, axes=axes[1][1])))
axes[1][1].set_title("Minimally diffusive")
for i in range(2):
    for j in range(2):
        axes[i][j].axis(False)
        axes[i][j].set_aspect(1)
for cbar in colourbars:
    cbar.set_ticks([-0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05])
plt.savefig("ping_pong-final.jpg")

# .. figure:: ping_pong-final.jpg
#    :figwidth: 90%
#    :align: center
#
# Whilst the first two approaches clearly resemble the input field, the bounded version
# gives a poorer representation. As such, we find that, whilst the bounded conservative
# projection allows for an interpolation operator with attractive properties, we
# shouldn't expect it to give smaller errors. To demonstrate this, we print the
# :math:`L^2` errors for each method. ::

print(f"Interpolate:     {errornorm(sensor, f_interp):.4e}")
print(f"Project:         {errornorm(sensor, f_proj):.4e}")
print(f"Bounded project: {errornorm(sensor, f_bounded):.4e}")

# ..code-block:: console
#
#   Interpolate:     1.7232e-02
#   Project:         4.2817e-03
#   Bounded project: 1.2731e-01
#
# This demo can also be accessed as a `Python script <ping_pong.py>`__.
