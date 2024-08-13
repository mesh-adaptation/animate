# On-the-fly time-dependent mesh adaptation
###########################################

# In this demo we consider the 2-dimensional version of the mesh adaptation experiment
# presented in :cite:<Barral:2016>. The problem comprises a bubble of tracer
# concentration field advected by a time-varying flow.
# We will consider two different mesh adaptation strategies: the classical mesh
# adaptation algorithm, which adapts the mesh several times throughout the simulation,
# before solving the advection equation, and the metric advection algorithm, which
# advects the initial metric tensor along the flow in order to predict where to
# prescribe fine resolution in the future.
#
# We begin by defining the advection problem. We consider the advection equation
#
# .. math::
#   \begin{array}{rl}
#       \frac{\partial c}{\partial t} + \nabla\cdot(\mathbf{u}c)=0& \text{in}\:\Omega,\\
#       c=0 & \text{on}\:\partial\Omega,\\
#       c=c_0(x,y) & \text{at}\:t=0,
#   \end{array},
#
# where :math:`c=c(x,y,t)` is the sought tracer concentration,
# :math:`\mathbf{u}=\mathbf{u}(x,y,t)` is the background velocity field, and
# :math:`\Omega=[0, 1]^2` is the spatial domain of interest.
#
# The background velocity field :math:`\mathbf{u}(x, y, t)` is chosen to be periodic in
# time, and is given by
#
# .. math::
#   \mathbf{u}(x, y, t) := \left(2\sin^2(\pi x)\sin(2\pi y), -\sin(2\pi x)\sin^2(\pi y) \right) \cos(2\pi t/T),
#
# where :math:`T` is the period. Note that the velocity field is solenoidal. At each
# timestep of the simulation we will update this field so we define a function that will
# return its vector expression. ::

from firedrake import *

T = 6.0


def velocity_expression(mesh, t):
    x, y = SpatialCoordinate(mesh)
    u_expr = as_vector(
        [
            2 * sin(pi * x) ** 2 * sin(2 * pi * y) * cos(2 * pi * t / T),
            -sin(2 * pi * x) * sin(pi * y) ** 2 * cos(2 * pi * t / T),
        ]
    )
    return u_expr


# We proceed similarly with prescribing initial conditions. At :math:`t=0`, we
# initialise the tracer concentration :math:`c_0 = c(x, y, 0)` to be :math:`1` inside
# a circular region of radius :math:`r_0=0.15` centred at :math:`(x_0, y_0)=(0.5, 0.65)`
# and :math:`0` elsewhere in the domain. Note that such a discontinuous function will
# not be represented well on a coarse uniform mesh. ::


def get_initial_condition(mesh):
    x, y = SpatialCoordinate(mesh)
    ball_r, ball_x0, ball_y0 = 0.15, 0.5, 0.65
    r = sqrt(pow(x - ball_x0, 2) + pow(y - ball_y0, 2))
    c0 = Function(FunctionSpace(mesh, "CG", 1))
    c0.interpolate(conditional(r < ball_r, 1.0, 0.0))
    # return conditional(r < ball_r0, 1.0, 0.0)
    return c0


# Now we are ready to solve the advection problem. Since we will solve the problem
# several times, we will wrap up the code in a function that we can easily call
# for each subinterval. The function takes the mesh over which to solve the problem,
# time interval :math:`[t_{\text{start}}, t_{\text{end}}]` over which to solve it, and
# the initial condition :math:`c_0 = c(x, y, t_{\text{start}})`. The function returns
# the solution :math:`c(x, y, t_{\text{end}})`. ::


def run_simulation(mesh, t_start, t_end, c0):
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    c_ = Function(Q).project(c0)  # project initial condition onto the current mesh
    c = Function(Q)  # solution at current timestep

    u_ = Function(V).interpolate(velocity_expression(mesh, t_start))  # vel. at t_start
    u = Function(V)  # velocity field at current timestep

    # SUPG stabilisation parameters
    D = Function(R).assign(0.1)  # diffusivity coefficient
    h = CellSize(mesh)  # mesh cell size
    U = sqrt(dot(u, u))  # velocity magnitude
    tau = 0.5 * h / U
    tau = min_value(tau, U * h / (6 * D))

    # Apply SUPG stabilisation
    phi = TestFunction(Q)
    phi += tau * dot(u, grad(phi))

    # Time-stepping parameters
    dt = Function(R).assign(0.01)  # timestep size
    theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

    # Variational form of the advection equation
    trial = TrialFunction(Q)
    a = inner(trial, phi) * dx + dt * theta * inner(dot(u, grad(trial)), phi) * dx
    L = inner(c_, phi) * dx - dt * (1 - theta) * inner(dot(u_, grad(c_)), phi) * dx

    # Define variational problem
    lvp = LinearVariationalProblem(a, L, c, bcs=DirichletBC(Q, 0.0, "on_boundary"))
    lvs = LinearVariationalSolver(lvp)

    # Integrate from t_start to t_end
    t = t_start + float(dt)
    while t < t_end + 0.5 * float(dt):
        # Update the background velocity field at the current timestep
        u.interpolate(velocity_expression(mesh, t))

        # Solve the advection equation
        lvs.solve()

        # Update the solution at the previous timestep
        c_.assign(c)
        u_.assign(u)
        t += float(dt)

    return c


# Finally, we are ready to run the simulation. We will first solve the entire
# problem over two uniform meshes: one with 32 elements in each direction and another
# with 128 elements in each direction. Since the flow reverts to its initial state at
# time :math:`t=T/2`, we run the simulations over the interval :math:`[0, T/2]`. In
# order to compare the efficacy of mesh adaptation methods, we will also keep track
# of the time it takes to run the simulation. ::

import time

simulation_end_time = T / 2.0

# mesh_coarse = UnitSquareMesh(32, 32)
mesh_coarse = UnitSquareMesh(256, 256)
c0_coarse = get_initial_condition(mesh_coarse)
cpu_time_coarse = time.time()
c_coarse_final = run_simulation(mesh_coarse, 0.0, simulation_end_time, c0_coarse)
cpu_time_coarse = time.time() - cpu_time_coarse

mesh_fine = UnitSquareMesh(128, 128)
c0_fine = get_initial_condition(mesh_fine)
cpu_time_fine = time.time()
c_fine_final = run_simulation(mesh_fine, 0.0, simulation_end_time, c0_fine)
cpu_time_fine = time.time() - cpu_time_fine

print(f"CPU time on the coarse mesh: {cpu_time_coarse:.2f} s")
print(f"CPU time on the fine mesh: {cpu_time_fine:.2f} s")

# We can now visualise the final concentration fields on the two meshes. ::

import matplotlib.pyplot as plt
from firedrake.pyplot import *

fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
for ax, c, mesh in zip(axes, [c_coarse_final, c_fine_final], [mesh_coarse, mesh_fine]):
    tripcolor(c, axes=ax)
    ax.set_title(f"{mesh.num_cells()} mesh elements")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.tight_layout()
fig.savefig("bubble_shear-uniform.jpg", dpi=300, bbox_inches="tight")

# .. figure:: bubble_shear-uniform.jpg
#    :figwidth: 80%
#    :align: center
#
# We observe that the concentration fields have not returned to their initial value
# at the end of the simulation, as we would have expected. This is particularly obvious
# in the coarse resolution simulation. This is due to the addition of the SUPG
# stabilisation to the weak form of the advection equation, which adds numerical
# diffusion. Numerical diffusion is necessary for numerical stability and for preventing
# oscillations, but it also makes the solution irreversible. The amount of difussion
# added is related to the grid Péclet number :math:`Pe = U\,h/2D`: the coarser the mesh
# is, the more diffusion is added. We encourage the reader to verify this by running the
# simulation on a sequence of finer uniform meshes.
#
# In ... ::


def compute_rel_error(c_init, c_final):
    init_l2_norm = norm(c_init, norm_type="L2")
    abs_l2_error = errornorm(c_init, c_final, norm_type="L2")
    return 100 * abs_l2_error / init_l2_norm


coarse_error = compute_rel_error(c0_coarse, c_coarse_final)
print(f"Relative L2 error on the coarse mesh: {coarse_error:.2f}%")
fine_error = compute_rel_error(c0_fine, c_fine_final)
print(f"Relative L2 error on the fine mesh: {fine_error:.2f}%")

# .. code-block:: console
#
#    Relative L2 error on the coarse mesh: 58.85%
#    Relative L2 error on the fine mesh: 34.60%
#
# TODO Explain the error
#
# Instead of running the simulation on a very fine mesh, we can use mesh adaptation
# techniques to refine the mesh only in regions and at times where that is necessary.
# For the purposes of this demo, we are going to adapt the mesh 20 times throughout the
# simulation, at equal time intervals (i.e., every 0.15s of simulation time). ::

num_adaptations = 20
interval_length = simulation_end_time / num_adaptations

# As mentioned in the introduction, we shall demonstrate two different mesh adaptation
# strategies. Let us begin with the classical mesh adaptation algorithm, which adapts
# each mesh before solving the advection equation over the corresponding subinterval.
# Here we will use the :class:`RiemannianMetric` class to define the metric, which we
# will compute based on the Hessian of the concentration field. Let us therefore define
# a function which takes the original mesh and the concentration field as arguments. We
# will also define parameters for computing the metric. ::

from animate.adapt import adapt
from animate.metric import RiemannianMetric

metric_params = {
    "dm_plex_metric": {
        "target_complexity": 2000.0,
        "p": 2.0,
        "h_min": 1e-04,  # minimum edge length
        "h_max": 1.0,  # maximum edge length
    }
}


def adapt_classical(mesh, c):
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(metric_params)
    metric.compute_hessian(c)
    metric.normalise()
    adapted_mesh = adapt(mesh, metric)
    return adapted_mesh


# We can now run the simulation over each subinterval. ::

mesh = mesh_coarse
c = get_initial_condition(mesh_coarse)
cpu_time_classical = time.time()
for i in range(num_adaptations):
    t0 = i * interval_length  # subinterval start time
    t1 = (i + 1) * interval_length  # subinterval end time

    # Adapt the mesh based on the concentration field at t0
    mesh = adapt_classical(mesh, c)

    # Solve the advection equation over the subinterval (t0, t1]
    c = run_simulation(mesh, t0, t1, c)
cpu_time_classical = time.time() - cpu_time_classical
print(f"CPU time with classical adaptation algorithm: {cpu_time_classical:.2f} s")

# Now let us plot the final adapted mesh and final concentration field computed on it.
# We will also compute the relative L2 error. ::

fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
triplot(mesh, axes=axes[0])
tripcolor(c, axes=axes[1])

for ax in axes.flat:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.tight_layout()
fig.savefig("bubble_shear-classical.jpg", dpi=300, bbox_inches="tight")

# Redefine the initial condition on the final adapted mesh
c0 = get_initial_condition(mesh)
classical_c0 = Function(FunctionSpace(mesh, "CG", 1)).interpolate(c0)
classical_error = compute_rel_error(classical_c0, c)

print(f"Relative L2 error with classical adaptation algorithm: {classical_error:.2f}%")

# .. code-block:: console
#
#    Relative L2 error with classical adaptation algorithm: 26.54%
#
# .. figure:: bubble_shear-classical.jpg
#    :figwidth: 80%
#    :align: center
#
# TODO talk about this ::


def adapt_metric_advection(mesh, t_start, t_end, c):
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric_ = RiemannianMetric(P1_ten)
    metric_intersect = RiemannianMetric(P1_ten)

    for mtrc in [metric, metric_, metric_intersect]:
        mtrc.set_parameters(metric_params)
    # metric_.set_parameters(mp)
    metric_.compute_hessian(c)
    metric_.normalise()

    Q = FunctionSpace(mesh, "CG", 1)
    m_ = Function(Q)
    m = Function(Q)
    h = Function(Q)

    V = VectorFunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u_ = Function(V).interpolate(velocity_expression(mesh, t_start))

    R = FunctionSpace(mesh, "R", 0)
    dt = Function(R).assign(0.01)  # timestep size
    theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

    # Metric advection by component
    trial = TrialFunction(Q)
    phi = TestFunction(Q)
    a = inner(trial, phi) * dx + dt * theta * inner(dot(u, grad(trial)), phi) * dx
    L = inner(m_, phi) * dx - dt * (1 - theta) * inner(dot(u_, grad(m_)), phi) * dx
    lvp = LinearVariationalProblem(a, L, m, bcs=DirichletBC(Q, h, "on_boundary"))
    lvs = LinearVariationalSolver(lvp)

    t = t_start + float(dt)
    while t < t_end + 0.5 * float(dt):
        u.interpolate(velocity_expression(mesh, t))

        # Advect each metric component
        for i in range(2):
            for j in range(2):
                h_max = metric_params["dm_plex_metric"]["h_max"]
                h.assign(1.0 / h_max**2 if i == j else 0.0)
                m_.assign(metric_.sub(2 * i + j))
                lvs.solve()
                metric.sub(2 * i + j).assign(m)

        # Ensure symmetry
        m_.assign(0.5 * (metric.sub(1) + metric.sub(2)))
        metric.sub(1).assign(m_)
        metric.sub(2).assign(m_)

        # Intersect metrics at each timestep
        metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
        metric_intersect.intersect(metric)

        metric_.assign(metric)
        u_.assign(u)
        t += float(dt)

    metric_intersect.normalise()
    amesh = adapt(mesh, metric_intersect)
    return amesh


mesh = UnitSquareMesh(32, 32)
c = get_initial_condition(mesh)
cpu_time_metric_advection = time.time()
for i in range(num_adaptations):
    t0 = i * interval_length  # subinterval start time
    t1 = (i + 1) * interval_length  # subinterval end time

    # Adapt the mesh based on the concentration field at t0
    mesh = adapt_metric_advection(mesh, t0, t1, c)

    # Solve the advection equation over the subinterval (t0, t1]
    c = run_simulation(mesh, t0, t1, c)
cpu_time_metric_advection = time.time() - cpu_time_metric_advection

print(f"CPU time with metric advection: {cpu_time_metric_advection:.2f} s")

fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
triplot(mesh, axes=axes[0])
tripcolor(c, axes=axes[1])

for ax in axes.flat:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.tight_layout()
fig.savefig("bubble_shear-metric_advection.jpg", dpi=300, bbox_inches="tight")

# Redefine the initial condition on the final adapted mesh
c0 = get_initial_condition(mesh)
metric_adv_c0 = Function(FunctionSpace(mesh, "CG", 1)).interpolate(c0)
metric_adv_error = compute_rel_error(metric_adv_c0, c)

print(f"Relative L2 error with metric advection: {metric_adv_error:.2f}%")

# .. code-block:: console
#
#    Relative L2 error with metric advection: 29.31%
#
# .. figure:: bubble_shear-metric_advection.jpg
#    :figwidth: 80%
#    :align: center

fig, ax = plt.subplots()
errors = [coarse_error, fine_error, classical_error, metric_adv_error]
cpu_times = [
    cpu_time_coarse,
    cpu_time_fine,
    cpu_time_classical,
    cpu_time_metric_advection,
]
ax.plot(cpu_times, errors, "o")
# label each point
for i, txt in enumerate(["Coarse mesh", "Fine mesh", "Classical", "Metric advection"]):
    ax.annotate(
        txt,
        (cpu_times[i], errors[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )
ax.set_xlabel("CPU time (s)")
ax.set_ylabel("Relative L2 error (%)")
fig.savefig("bubble_shear-time_error.jpg", dpi=300, bbox_inches="tight")

# .. figure:: bubble_shear-time_error.jpg
#    :figwidth: 80%
#    :align: center
