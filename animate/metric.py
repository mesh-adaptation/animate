from collections.abc import Iterable

import firedrake
import firedrake.function as ffunc
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
import numpy as np
import sympy
import ufl
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc
from pyop2 import op2

try:
    # only works in firedrake release <=2025.4.x
    from firedrake.petsc import OptionsManager
except ImportError:
    from petsctools import OptionsManager

from .interpolation import clement_interpolant
from .recovery import (
    get_metric_kernel,
    recover_boundary_hessian,
    recover_gradient_l2,
    recover_hessian_clement,
)

__all__ = ["RiemannianMetric", "determine_metric_complexity", "intersect_on_boundary"]


class RiemannianMetric(ffunc.Function):
    r"""
    Class for defining a Riemannian metric over a given mesh.

    A metric is a symmetric positive-definite field, which conveys how the mesh is to
    be adapted. If the mesh is of dimension :math:`d` then the metric takes the value
    of a square :math:`d\times d` matrix at each point.

    The implementation of metric-based mesh adaptation used in PETSc assumes that the
    metric is piece-wise linear and continuous, with its degrees of freedom at the
    mesh vertices.

    For details, see the PETSc manual entry:
      https://petsc.org/release/docs/manual/dmplex/#metric-based-mesh-adaptation
    """

    _supported_parameters = (
        "dm_plex_metric_target_complexity",
        "dm_plex_metric_h_min",
        "dm_plex_metric_h_max",
        "dm_plex_metric_a_max",
        "dm_plex_metric_p",
        "dm_plex_metric_gradation_factor",
        "dm_plex_metric_hausdorff_number",
        "dm_plex_metric_boundary_tag",
        "dm_plex_metric_no_insert",
        "dm_plex_metric_no_swap",
        "dm_plex_metric_no_move",
        "dm_plex_metric_no_surf",
        "dm_plex_metric_num_iterations",
        "dm_plex_metric_verbosity",
        "dm_plex_metric_isotropic",
        "dm_plex_metric_uniform",
        "dm_plex_metric_restrict_anisotropy_first",
    )

    @PETSc.Log.EventDecorator()
    def __init__(self, function_space, *args, **kwargs):
        r"""
        :arg function_space: the tensor function space, on which to build the metric.
            Alternatively, another Function may be passed, in which case its function
            space is used to build the metric and its values are copied. Yet another
            alternative is to pass a mesh, whereby a tensor :math:`\mathbb P1` space is
            built on top of it.
        :type function_space: :class:`firedrake.functionspaceimpl.FunctionSpace`,
            :class:`firedrake.function.Function`, or
            :class:`firedrake.mesh.MeshGeometry`
        :kwarg metric_parameters: see :func:`set_parameters <set_parameters>`.
        :type metric_parameters: :class:`dict`
        """
        if isinstance(function_space, fmesh.MeshGeometry):
            function_space = ffs.TensorFunctionSpace(function_space, "CG", 1)
        self._metric_parameters = {}
        metric_parameters = kwargs.pop("metric_parameters", {})
        super().__init__(function_space, *args, **kwargs)

        # Check that we have an appropriate tensor P1 function
        fs = self.function_space()
        mesh = fs.mesh()
        tdim = mesh.topological_dimension()
        if tdim not in (2, 3):
            raise ValueError(f"Riemannian metric should be 2D or 3D, not {tdim}D.")
        if isinstance(fs.dof_count, Iterable):
            raise ValueError("Riemannian metric cannot be built in a mixed space.")
        self._check_space()
        rank = len(fs.dof_dset.dim)
        if rank != 2:
            raise ValueError(
                "Riemannian metric should be matrix-valued,"
                f" not rank-{rank} tensor-valued."
            )

        # Stash mesh data
        plex = mesh.topology_dm.clone()
        self._mesh = mesh
        self._plex = plex
        self._tdim = tdim

        # Ensure DMPlex coordinates are consistent
        self._set_plex_coordinates()

        # Adjust the section
        entity_dofs = np.zeros(tdim + 1, dtype=np.int32)
        entity_dofs[0] = tdim**2
        plex.setSection(mesh.create_section(entity_dofs)[0])

        # Process spatially variable metric parameters
        self._variable_parameters = {
            "dm_plex_metric_h_min": firedrake.Constant(1.0e-30),
            "dm_plex_metric_h_max": firedrake.Constant(1.0e30),
            "dm_plex_metric_a_max": firedrake.Constant(1.0e5),
            "dm_plex_metric_boundary_tag": None,
        }
        self._variable_parameters_set = False
        if metric_parameters:
            self.set_parameters(metric_parameters)

    def _check_space(self):
        el = self.function_space().ufl_element()
        if (el.family(), el.degree()) != ("Lagrange", 1):
            raise ValueError(f"Riemannian metric should be in P1 space, not '{el}'.")

    @staticmethod
    def _collapse_parameters(metric_parameters):
        """
        Account for concise nested dictionary formatting.

        See :func:`set_parameters <set_parameters>`.

        :kwarg metric_parameters: nested parameter dictionary
        :type metric_parameters: :class:`dict`
        :returns: collapsed parameter dictionary
        :rtype: :class:`dict`
        """
        if "dm_plex_metric" in metric_parameters:
            for key, value in metric_parameters["dm_plex_metric"].items():
                metric_parameters["_".join(["dm_plex_metric", key])] = value
            metric_parameters.pop("dm_plex_metric")
        return metric_parameters

    def _process_parameters(self, metric_parameters):
        mp = self._collapse_parameters(metric_parameters.copy())

        # Check all parameters
        for key in mp:
            if not key.startswith("dm_plex_metric_"):
                raise ValueError(
                    f"Unsupported metric parameter '{key}'."
                    " Metric parameters must start with the prefix 'dm_plex_metric_'."
                )
            if key not in self._supported_parameters:
                raise ValueError(f"Unsupported metric parameter '{key}'.")

        # Spatially varying parameters need to be treated differently
        vp = {}
        for key in ("h_min", "h_max", "a_max"):
            key = f"dm_plex_metric_{key}"
            value = mp.get(key)
            if not value:
                continue
            if isinstance(value, (firedrake.Constant, ffunc.Function)):
                vp[key] = value
                mp.pop(key)
                self._variable_parameters_set = True
            else:
                vp[key] = firedrake.Constant(value)

        # The boundary_tag parameter does not currently exist in PETSc
        if "dm_plex_metric_boundary_tag" in mp:
            self._variable_parameters_set = True
            vp["dm_plex_metric_boundary_tag"] = mp.pop("dm_plex_metric_boundary_tag")

        return mp, vp

    def set_parameters(self, metric_parameters=None):
        r"""
        Set metric parameter values internally. All options have the prefix
        `dm_plex_metric_` and are listed below (with prefix dropped for brevity). Note
        that any parameter which supports :class:`~.Function` values must be in a
        :math:`\mathbb{P}1` space defined on the same mesh as the metric, i.e., from
        ``FunctionSpace(mesh, "CG", 1)``.

        * `target_complexity`: Strictly positive target metric complexity value. No
            default - **must be set**.
        * `h_min`: Minimum tolerated metric magnitude, which allows approximate control
            of minimum element size in the adapted mesh. Supports :class:`~.Constant`
            and :class:`~.Function` input, as well as :class:`float`. Default: 1.0e-30.
        * `h_max`: Maximum tolerated metric magnitude, which allows approximate control
            of maximum element size in the adapted mesh. Supports :class:`~.Constant`
            and :class:`~.Function` input, as well as :class:`float`. Default: 1.0e+30.
        * `a_max`: Maximum tolerated metric anisotropy, which allows approximate control
            of maximum element anisotropy in the adapted mesh. Supports
            :class:`~.Constant` and :class:`~.Function` input, as well as
            :class:`float`. Default: 1.0e+05.
        * `p`: :math:`L^p` normalisation order. Supports ``np.inf`` as well as
            :class:`float` values from :math:`[0,\infty)`. Default: 1.0.
        * `gradation_factor`: Maximum ratio by which adjacent edges in the adapted mesh
            may differ. Default: 1.3. For more detail, see
            https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-hgrad.
        * `hausdorff_number`: Spatial scale factor for the problem. The default value
            0.01 corresponds to an :math:`\mathcal{O}(1)` length scale. A rule of thumb
            is to scale this value appropriately to the length scale of your problem.
            For more detail, see
            https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-hausd.
        * `boundary_tag`: Mesh boundary tag to restrict attention to during
            boundary-specific metric manipulations. Unset by default, which implies all
            boundaries are considered. (Note that this parameter does not currently
            exist in the underlying PETSc implementation.)
        * `no_insert`: Boolean flag for turning off node insertion and deletion during
            adaptation. Default: False.
        * `no_swap`: Boolean flag for turning off edge and face swapping during
            adaptation. Default: False.
        * `no_move`: Boolean flag for turning off node movement during adaptation.
            Default: False.
        * `no_surf`: Boolean flag for turning off surface modification during
            adaptation. Default: False.
        * `num_iterations`: Number of adaptation-repartitioning iterations in the
            parallel case. Default: 3. For details on the parallel algorithm, see
            https://inria.hal.science/hal-02386837.
        * `verbosity`: Verbosity of the mesh adaptation package (-1 = silent,
            10 = maximum). Default: -1. For more detail, see
            https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-v.
        * `isotropic`: Optimisation for isotropic metrics. (Currently unsupported.)
        * `uniform`: Optimisation for uniform metrics. (Currently unsupported.)
        * `restrict_anisotropy_first`: Boolean flag specifying that anisotropy should be
            restricted before normalisation when True. Default: True.

        :kwarg metric_parameters: parameters as above
        :type metric_parameters: :class:`dict` with :class:`str` keys and value which
            may take various types
        """
        metric_parameters = metric_parameters or {}
        mp, vp = self._process_parameters(metric_parameters)

        # Metric normalisation is implemented in the RiemannianMetric class so we
        # handle the `dm_plex_metric_restrict_anisotropy_first` option here rather than
        # delegating it to PETSc (which has its own normalisation implementation).
        if "dm_plex_metric_restrict_anisotropy_first" in mp:
            self._restrict_anisotropy_first = mp.pop(
                "dm_plex_metric_restrict_anisotropy_first"
            )
        else:
            self._restrict_anisotropy_first = True

        # Stash the metric parameters on the RiemannianMetric
        self._metric_parameters.update(mp)
        self._variable_parameters.update(vp)

        # Prepare a copy of the metric parameters to pass to PETSc
        mp = self._metric_parameters.copy()
        if mp.get("dm_plex_metric_p") == np.inf:
            mp["dm_plex_metric_p"] = 1.79769e308

        # Pass parameters to PETSc
        with OptionsManager(mp, "").inserted_options():
            self._plex.metricSetFromOptions()
        if self._plex.metricIsUniform():
            raise NotImplementedError(
                "Uniform metric optimisations are not supported in Animate."
            )
        if self._plex.metricIsIsotropic():
            raise NotImplementedError(
                "Isotropic metric optimisations are not supported in Animate."
            )

    @property
    def metric_parameters(self):
        """
        Parameters for configuring the metric.

        See :func:`set_parameters <set_parameters>`.

        :returns: parameter dictionary
        :rtype: :class:`dict`
        """
        mp = self._metric_parameters.copy()
        if self._variable_parameters_set:
            mp.update(self._variable_parameters)
        return mp

    def _create_from_array(self, array):
        bsize = self.dat.cdim
        size = [self.dat.dataset.total_size * bsize] * 2
        comm = PETSc.COMM_SELF
        return PETSc.Vec().createWithArray(array, size=size, bsize=bsize, comm=comm)

    @PETSc.Log.EventDecorator()
    def _set_plex_coordinates(self):
        """
        Ensure that the coordinates of the Firedrake mesh and the underlying DMPlex are
        consistent.
        """
        entity_dofs = np.zeros(self._tdim + 1, dtype=np.int32)
        entity_dofs[0] = self._mesh.geometric_dimension()
        coord_section = self._mesh.create_section(entity_dofs)[0]
        # NOTE: section doesn't have any fields, but PETSc assumes it to have one
        coord_dm = self._plex.getCoordinateDM()
        coord_dm.setSection(coord_section)
        coords_local = coord_dm.createLocalVec()
        coords_local.array[:] = np.reshape(
            self._mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape
        )
        self._plex.setCoordinatesLocal(coords_local)

    # --- Methods for creating metrics

    def copy(self, deepcopy=False):
        """
        Copy the metric and any associated parameters.

        :kwarg deepcopy: If ``True``, the new metric will allocate new space and copy
            values. If ``False`` (default) then the new metric will share the DoF
            values.
        :type deepcopy: :class:`bool`
        :return: a copy of the metric with the same parameters set
        :rtype: :class:`~.RiemannianMetric`
        """
        metric = type(self)(super().copy(deepcopy=deepcopy))
        metric.set_parameters(self.metric_parameters)
        return metric

    @PETSc.Log.EventDecorator()
    def compute_hessian(self, field, method="mixed_L2", **kwargs):
        """
        Recover the Hessian of a scalar field.

        :arg field: the scalar field whose Hessian we seek to recover
        :type field: :class:`firedrake.function.Function`
        :kwarg method: recovery method
        :type method: :class:`str`
        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`

        All other keyword arguments are passed to the chosen recovery routine.

        In the case of the `'L2'` method, the `target_space` keyword argument is used
        for the gradient recovery. The target space for the Hessian recovery is
        inherited from the metric itself.
        """
        if method == "L2":
            gradient = recover_gradient_l2(
                field, target_space=kwargs.get("target_space")
            )
            return self.assign(recover_gradient_l2(gradient))
        elif method == "mixed_L2":
            return self.interpolate(
                self._compute_gradient_and_hessian(field, **kwargs)[1]
            )
        elif method == "Clement":
            return self.assign(recover_hessian_clement(field, **kwargs)[1])
        elif method == "ZZ":
            raise NotImplementedError(
                "Zienkiewicz-Zhu recovery not yet implemented."
            )  # TODO (#130)
        else:
            raise ValueError(f"Recovery method '{method}' not recognised.")

    @PETSc.Log.EventDecorator()
    def compute_boundary_hessian(self, field, method="mixed_L2", **kwargs):
        """
        Recover the Hessian of a scalar field on the domain boundary.

        :arg field: field to recover over the domain boundary
        :type field: :class:`firedrake.function.Function`
        :kwarg method: recovery method, chosen from 'mixed_L2' and 'Clement'
        :type method: :class:`str`
        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        return self.assign(recover_boundary_hessian(field, method=method, **kwargs))

    def _compute_gradient_and_hessian(self, field, solver_parameters=None):
        mesh = self.function_space().mesh()
        V = ffs.VectorFunctionSpace(mesh, "CG", 1)
        W = V * self.function_space()
        g, H = firedrake.TrialFunctions(W)
        phi, tau = firedrake.TestFunctions(W)
        sol = ffunc.Function(W)
        n = ufl.FacetNormal(mesh)

        a = (
            ufl.inner(tau, H) * ufl.dx
            + ufl.inner(ufl.div(tau), g) * ufl.dx
            - ufl.dot(g, ufl.dot(tau, n)) * ufl.ds
            - ufl.dot(ufl.avg(g), ufl.jump(tau, n)) * ufl.dS
            + ufl.inner(phi, g) * ufl.dx
        )
        L = (
            field * ufl.dot(phi, n) * ufl.ds
            + ufl.avg(field) * ufl.jump(phi, n) * ufl.dS
            - field * ufl.div(phi) * ufl.dx
        )
        if solver_parameters is None:
            solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "gmres",
                "ksp_max_it": 20,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_0_fields": "1",
                "pc_fieldsplit_1_fields": "0",
                "pc_fieldsplit_schur_precondition": "selfp",
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "gamg",
                "fieldsplit_1_mg_levels_ksp_max_it": 5,
            }
            if firedrake.COMM_WORLD.size == 1:
                solver_parameters["fieldsplit_0_pc_type"] = "ilu"
                solver_parameters["fieldsplit_1_mg_levels_pc_type"] = "ilu"
            else:
                solver_parameters["fieldsplit_0_pc_type"] = "bjacobi"
                solver_parameters["fieldsplit_0_sub_ksp_type"] = "preonly"
                solver_parameters["fieldsplit_0_sub_pc_type"] = "ilu"
                solver_parameters["fieldsplit_1_mg_levels_pc_type"] = "bjacobi"
                solver_parameters["fieldsplit_1_mg_levels_sub_ksp_type"] = "preonly"
                solver_parameters["fieldsplit_1_mg_levels_sub_pc_type"] = "ilu"
        firedrake.solve(a == L, sol, solver_parameters=solver_parameters)
        return sol.subfunctions

    # --- Methods for processing metrics

    @PETSc.Log.EventDecorator()
    def enforce_spd(self, restrict_sizes=False, restrict_anisotropy=False):
        """
        Enforce that the metric is symmetric positive-definite.

        :kwarg restrict_sizes: should minimum and maximum metric magnitudes be enforced?
        :type restrict_sizes: :class:`bool`
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        :type restrict_anisotropy: :class:`bool`
        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        kw = {
            "restrictSizes": restrict_sizes,
            "restrictAnisotropy": restrict_anisotropy,
        }
        if self._variable_parameters_set:
            kw["restrictSizes"] = False
            kw["restrictAnisotropy"] = False
        v = self._create_from_array(self.dat.data_with_halos)
        det, _ = self._plex.metricDeterminantCreate()
        self._plex.metricEnforceSPD(v, v, det, **kw)
        size = np.shape(self.dat.data_with_halos)
        self.dat.data_with_halos[:] = np.reshape(v.array, size)
        v.destroy()
        if self._variable_parameters_set:
            if restrict_sizes and restrict_anisotropy:
                return self._enforce_variable_constraints()
            elif restrict_sizes or restrict_anisotropy:
                raise NotImplementedError(
                    "Can only currently restrict both sizes and anisotropy."
                )
        return self

    # TODO: Implement this on the PETSc side
    #       See https://gitlab.com/petsc/petsc/-/issues/1450
    @PETSc.Log.EventDecorator()
    def _enforce_variable_constraints(self):
        """
        Post-process a metric to enforce minimum and maximum metric magnitudes
        and maximum anisotropy, any of which may vary spatially.

        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        mesh = self.function_space().mesh()
        P1 = firedrake.FunctionSpace(mesh, "CG", 1)

        def interp(f):
            r"""
            Try to apply a Clement interpolant.
              * `TypeError` indicates something other than a :class:`Function` passed.
              * `ValueError` indicates the :class:`Function` is not :math:`\mathbb{P}0`.
            """
            try:
                return clement_interpolant(f, target_space=P1)
            except TypeError:
                return ffunc.Function(P1).assign(f)
            except ValueError:
                return ffunc.Function(P1).interpolate(f)

        h_min = interp(self._variable_parameters["dm_plex_metric_h_min"])
        h_max = interp(self._variable_parameters["dm_plex_metric_h_max"])
        a_max = interp(self._variable_parameters["dm_plex_metric_a_max"])

        # Check minimal h_min value is positive and smaller than minimal h_max value
        _hmin = h_min.dat.global_data.min()
        if _hmin <= 0.0:
            raise ValueError(f"Encountered non-positive h_min value: {_hmin}.")
        if h_max.dat.global_data.min() < _hmin:
            raise ValueError(
                "Minimum h_max value is smaller than minimum h_min value:"
                f"{h_max.dat.global_data.min()} < {_hmin}."
            )

        # Check h_max is always at least h_min
        dx = ufl.dx(domain=mesh)
        integral = firedrake.assemble(ufl.conditional(h_max < h_min, 1, 0) * dx)
        if not np.isclose(integral, 0.0):
            raise ValueError("Encountered regions where h_max < h_min.")

        # Check minimal a_max value is close to unity or larger
        _a_max = a_max.dat.global_data.min()
        if not np.isclose(_a_max, 1.0) and _a_max < 1.0:
            raise ValueError(f"Encountered a_max value smaller than unity: {_a_max}.")

        dim = mesh.topological_dimension()
        boundary_tag = self._variable_parameters.get("dm_plex_metric_boundary_tag")
        if boundary_tag is None:
            node_set = self.function_space().node_set
        else:
            bc = firedrake.DirichletBC(self.function_space(), 0, boundary_tag)
            node_set = bc.node_set
        op2.par_loop(
            get_metric_kernel("postproc_metric", dim),
            node_set,
            self.dat(op2.RW),
            h_min.dat(op2.READ),
            h_max.dat(op2.READ),
            a_max.dat(op2.READ),
        )
        return self

    @PETSc.Log.EventDecorator()
    def normalise(
        self,
        global_factor=None,
        boundary=False,
        restrict_sizes=True,
        restrict_anisotropy=True,
    ):
        """
        Apply :math:`L^p` normalisation to the metric.

        See
        https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#metric-normalisation
        for more details.

        :kwarg global_factor: pre-computed global normalisation factor
        :type global_factor: :class:`float`
        :kwarg boundary: is the normalisation to be done over the boundary?
        :type boundary: :class:`bool`
        :kwarg restrict_sizes: should minimum and maximum metric magnitudes be enforced?
        :type restrict_sizes: :class:`bool`
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        :type restrict_anisotropy: :class:`bool`
        :return: the normalised metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        d = self._tdim - 1 if boundary else self._tdim
        p = self.metric_parameters.get("dm_plex_metric_p", 1.0)
        target = self.metric_parameters.get("dm_plex_metric_target_complexity")
        if target is None:
            raise ValueError("dm_plex_metric_target_complexity must be set.")

        # Enforce that the metric is SPD
        self.enforce_spd(
            restrict_sizes=False, restrict_anisotropy=self._restrict_anisotropy_first
        )

        # Compute global normalisation factor
        detM = ufl.det(self)
        if global_factor is None:
            dX = (ufl.ds if boundary else ufl.dx)(domain=self._mesh)
            exponent = 0.5 if np.isinf(p) else (p / (2 * p + d))
            integral = firedrake.assemble(pow(detM, exponent) * dX)
            global_factor = firedrake.Constant(pow(target / integral, 2 / d))

        # Normalise the metric
        if boundary:
            raise NotImplementedError(
                "Normalisation on the boundary not yet implemented."
            )
        determinant = 1 if np.isinf(p) else pow(detM, -1 / (2 * p + d))
        self.interpolate(global_factor * determinant * self)

        # Enforce element constraints
        return self.enforce_spd(
            restrict_sizes=restrict_sizes,
            restrict_anisotropy=not self._restrict_anisotropy_first,
        )

    # --- Methods for combining metrics

    @PETSc.Log.EventDecorator()
    def intersect(self, *metrics):
        r"""
        Intersect the metric with other metrics.

        Metric intersection means taking the minimal ellipsoid in the direction of each
        eigenvector at each point in the domain.

        See
        https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#combining-metrics
        for more details.

        :arg metrics: the metrics to be intersected with
        :type metrics: :class:`tuple` of :class:`~.RiemannianMetric`\s
        :return: the intersected metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        fs = self.function_space()
        for metric in metrics:
            assert isinstance(metric, RiemannianMetric)
            if fs != metric.function_space():
                raise ValueError(
                    "Cannot intersect metrics with different function spaces."
                )

        # Intersect the metrics recursively one at a time
        if len(metrics) == 0:
            pass
        elif len(metrics) == 1:
            v1 = self._create_from_array(self.dat.data_with_halos)
            v2 = self._create_from_array(metrics[0].dat.data_ro_with_halos)
            vout = self._create_from_array(np.zeros_like(self.dat.data_with_halos))

            # Compute the intersection on the PETSc level
            self._plex.metricIntersection2(v1, v2, vout)

            # Assign to the output of the intersection
            size = np.shape(self.dat.data_with_halos)
            self.dat.data_with_halos[:] = np.reshape(vout.array, size)
            v2.destroy()
            v1.destroy()
            vout.destroy()
        else:
            self.intersect(*metrics[1:])
        return self

    @PETSc.Log.EventDecorator()
    def average(self, *metrics, weights=None):
        r"""
        Average the metric with other metrics.

        :arg metrics: the metrics to be averaged with
        :type metrics: :class:`tuple` of :class:`~.RiemannianMetric`\s
        :kwarg weights: list of weights to apply to each metric
        :type weights: :class:`tuple` of :class:`float`\s
        :return: the averaged metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        num_metrics = len(metrics) + 1
        if num_metrics == 1:
            return self
        if weights is None:
            weights = np.ones(num_metrics) / num_metrics
        if len(weights) != num_metrics:
            raise ValueError(
                f"Number of weights ({len(weights)}) does not match number of metrics"
                f" ({num_metrics})."
            )
        self *= weights[0]
        fs = self.function_space()
        for i, metric in enumerate(metrics):
            assert isinstance(metric, RiemannianMetric)
            if fs != metric.function_space():
                raise ValueError(
                    "Cannot average metrics with different function spaces."
                )
            self += weights[i + 1] * metric
        return self

    def combine(self, *metrics, average=True, **kwargs):
        r"""
        Combine metrics using either averaging or intersection.

        See
        https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#combining-metrics
        for more details.

        :arg metrics: the list of metrics to combine with
        :type metrics: :class:`tuple` of :class:`~.RiemannianMetric`\s
        :kwarg average: toggle between averaging and intersection
        :type average: :class:`bool`

        All other keyword arguments are passed to the relevant method.
        """
        return (self.average if average else self.intersect)(*metrics, **kwargs)

    # --- Metric diagnostics

    @PETSc.Log.EventDecorator()
    def complexity(self, boundary=False):
        """
        Compute the metric complexity - the continuous analogue
        of the (inherently discrete) mesh vertex count.

        See
        https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#continuous-mesh-analogy
        for more details.

        :kwarg boundary: should the complexity be computed over the domain boundary?
        :type boundary: :class:`bool`
        :return: the complexity of the metric
        :rtype: :class:`float`
        """
        dX = ufl.ds if boundary else ufl.dx
        return firedrake.assemble(ufl.sqrt(ufl.det(self)) * dX)

    # --- Metric factorisations

    @PETSc.Log.EventDecorator()
    def compute_eigendecomposition(self, reorder=False):
        r"""
        Compute the eigenvectors and eigenvalues of a matrix-valued function.

        See
        https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#geometric-interpretation
        for more information.

        :kwarg reorder: should the eigendecomposition be reordered in order of
            *descending* eigenvalue magnitude?
        :type reorder: :class:`bool`
        :return: eigenvectors and eigenvalues represented as matrix-valued and
            vector-valued functions, respectively
        :rtype: :class:`tuple` of :class:`firedrake.function.Function`\s
        """
        V_ten = self.function_space()
        mesh = V_ten.mesh()
        fe = (V_ten.ufl_element().family(), V_ten.ufl_element().degree())
        V_vec = firedrake.VectorFunctionSpace(mesh, *fe)
        dim = mesh.topological_dimension()
        evectors, evalues = firedrake.Function(V_ten), firedrake.Function(V_vec)
        if reorder:
            name = "get_reordered_eigendecomposition"
        else:
            name = "get_eigendecomposition"
        kernel = get_metric_kernel(name, dim)
        op2.par_loop(
            kernel,
            V_ten.node_set,
            evectors.dat(op2.RW),
            evalues.dat(op2.RW),
            self.dat(op2.READ),
        )
        return evectors, evalues

    @PETSc.Log.EventDecorator()
    def assemble_eigendecomposition(self, evectors, evalues):
        """
        Assemble a matrix from its eigenvectors and eigenvalues.

        See
        https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#geometric-interpretation
        for more information.

        :arg evectors: eigenvectors represented as a matrix-valued function
        :type evectors: :class:`firedrake.function.Function`
        :arg evalues: eigenvalues represented as a vector-valued function
        :type evalues: :class:`firedrake.function.Function`
        """
        V_ten = evectors.function_space()
        fe_ten = V_ten.ufl_element()
        if len(V_ten.value_shape) != 2:
            raise ValueError(
                "Eigenvector Function should be rank-2,"
                f" not rank-{len(V_ten.value_shape)}."
            )
        V_vec = evalues.function_space()
        fe_vec = V_vec.ufl_element()
        if len(V_vec.value_shape) != 1:
            raise ValueError(
                "Eigenvalue Function should be rank-1,"
                f" not rank-{len(V_vec.value_shape)}."
            )
        if fe_ten.family() != fe_vec.family():
            raise ValueError(
                "Mismatching finite element families:"
                f" '{fe_ten.family()}' vs. '{fe_vec.family()}'."
            )
        if fe_ten.degree() != fe_vec.degree():
            raise ValueError(
                "Mismatching finite element space degrees:"
                f" {fe_ten.degree()} vs. {fe_vec.degree()}."
            )
        dim = V_ten.mesh().topological_dimension()
        op2.par_loop(
            get_metric_kernel("set_eigendecomposition", dim),
            V_ten.node_set,
            self.dat(op2.RW),
            evectors.dat(op2.READ),
            evalues.dat(op2.READ),
        )
        return self

    @PETSc.Log.EventDecorator()
    def density_and_quotients(self, reorder=False):
        r"""
        Extract the density and anisotropy quotients from a metric.

        By symmetry, Riemannian metrics admit an orthogonal eigendecomposition,

        .. math::
            \underline{\mathbf M}(\mathbf x)
            = \underline{\mathbf V}(\mathbf x)\:
            \underline{\boldsymbol\Lambda}(\mathbf x)\:
            \underline{\mathbf V}(\mathbf x)^T,

        at each point :math:`\mathbf x\in\Omega`, where
        :math:`\underline{\mathbf V}` and :math:`\underline{\boldsymbol\Sigma}` are
        matrices holding the eigenvectors and eigenvalues, respectively. By
        positive-definiteness, entries of :math:`\underline{\boldsymbol\Lambda}` are all
        positive.

        An alternative decomposition,

        .. math::
            \underline{\mathbf M}(\mathbf x)
            = d(\mathbf x)^\frac2n
            \underline{\mathbf V}(\mathbf x)\:
            \underline{\mathbf R}(\mathbf x)^{-\frac2n}
            \underline{\mathbf V}(\mathbf x)^T

        can also be deduced, in terms of the `metric density` and
        `anisotropy quotients`,

        .. math::
            d = \prod_{i=1}^n h_i,\qquad
            r_i = h_i^n d,\qquad \forall i=1:n,

        where :math:`h_i := \frac1{\sqrt{\lambda_i}}`.

        See
        https://mesh-adaptation.github.io/docs/animate/1-metric-based.html#geometric-interpretation
        for more details.

        :kwarg reorder: should the eigendecomposition be reordered?
        :type reorder: :class:`bool`
        :return: metric density, anisotropy quotients and eigenvector matrix represented
            as scalar-valued, vector-valued, and matrix-valued functions, respectively
        :rtype: :class:`tuple` of :class:`firedrake.function.Function`\s
        """
        fs_ten = self.function_space()
        mesh = fs_ten.mesh()
        fe = (fs_ten.ufl_element().family(), fs_ten.ufl_element().degree())
        dim = mesh.topological_dimension()
        evectors, evalues = self.compute_eigendecomposition(reorder=reorder)

        # Extract density and quotients
        density = firedrake.Function(
            firedrake.FunctionSpace(mesh, *fe), name="Metric density"
        )
        density.interpolate(np.prod([ufl.sqrt(e) for e in evalues]))
        quotients = firedrake.Function(
            firedrake.VectorFunctionSpace(mesh, *fe), name="Anisotropic quotients"
        )
        quotients.interpolate(
            ufl.as_vector([density / ufl.sqrt(e) ** dim for e in evalues])
        )
        return density, quotients, evectors

    # --- Goal-oriented metric drivers

    @PETSc.Log.EventDecorator()
    def compute_isotropic_metric(
        self, error_indicator, interpolant="Clement", **kwargs
    ):
        r"""
        Compute an isotropic metric from some error indicator.

        The result is a :math:`\mathbb P1` diagonal tensor field whose entries are
        projections of the error indicator in modulus.

        :arg error_indicator: the error indicator
        :type error_indicator: :class:`firedrake.function.Function`
        :kwarg interpolant: interpolation method, chosen from 'Clement' or 'L2'
        :type interpolant: :class:`str`
        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        mesh = ufl.domain.extract_unique_domain(error_indicator)
        if mesh != self.function_space().mesh():
            raise ValueError("Cannot use an error indicator from a different mesh.")
        dim = mesh.topological_dimension()

        # Interpolate P0 indicators into P1 space
        if interpolant == "Clement":
            P1_indicator = clement_interpolant(error_indicator)
        elif interpolant == "L2":
            P1_indicator = firedrake.project(
                error_indicator, firedrake.FunctionSpace(mesh, "CG", 1)
            )
        else:
            raise ValueError(f"Interpolant '{interpolant}' not recognised.")
        return self.interpolate(abs(P1_indicator) * ufl.Identity(dim))

    def compute_isotropic_dwr_metric(
        self,
        error_indicator,
        convergence_rate=1.0,
        min_eigenvalue=1.0e-05,
        interpolant="Clement",
    ):
        r"""
        Compute an isotropic metric from some error indicator using an element-based
        formulation.

        The formulation is based on that presented in :cite:`Carpio:2013`. Note that
        normalisation is implicit in the metric construction and involves the
        `convergence_rate` parameter, named :math:`alpha` in :cite:`Carpio:2013`.

        Whilst an element-based formulation is used to derive the metric, the result is
        projected into :math:`\mathbb P1` space, by default.

        :arg error_indicator: the error indicator
        :type error_indicator: :class:`firedrake.function.Function`
        :kwarg convergence_rate: normalisation parameter
        :type convergence_rate: :class:`float`
        :kwarg min_eigenvalue: minimum tolerated eigenvalue
        :type min_eigenvalue: :class:`float`
        :kwarg interpolant: interpolation method, chosen from 'Clement' or 'L2'
        :type interpolant: :class:`str`
        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        return self.compute_anisotropic_dwr_metric(
            error_indicator=error_indicator,
            convergence_rate=convergence_rate,
            min_eigenvalue=min_eigenvalue,
            interpolant=interpolant,
        )

    def _any_inf(self, f):
        arr = f.dat.global_data
        return np.isinf(arr).any() or np.isnan(arr).any()

    @PETSc.Log.EventDecorator()
    def compute_anisotropic_dwr_metric(
        self,
        error_indicator,
        hessian=None,
        convergence_rate=1.0,
        min_eigenvalue=1.0e-05,
        interpolant="Clement",
    ):
        r"""
        Compute an anisotropic metric from some error indicator, given a Hessian field.

        The formulation used is based on that presented in :cite:`Carpio:2013`. Note
        that normalisation is implicit in the metric construction and involves the
        `convergence_rate` parameter, named :math:`alpha` in :cite:`Carpio:2013`.

        If a Hessian is not provided then an isotropic formulation is used.

        Whilst an element-based formulation is used to derive the metric, the result is
        projected into :math:`\mathbb P1` space, by default.

        :arg error_indicator: the error indicator
        :type error_indicator: :class:`firedrake.function.Function`
        :kwarg hessian: the Hessian
        :type hessian: :class:`~.RiemannianMetric`
        :kwarg convergence_rate: normalisation parameter
        :type convergence_rate: :class:`float`
        :kwarg min_eigenvalue: minimum tolerated eigenvalue
        :type min_eigenvalue: :class:`float`
        :kwarg interpolant: interpolation method, chosen from 'Clement' or 'L2'
        :type interpolant: :class:`str`
        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        mp = self.metric_parameters.copy()
        target_complexity = mp.get("dm_plex_metric_target_complexity")
        if target_complexity is None:
            raise ValueError("Target complexity must be set.")
        mesh = ufl.domain.extract_unique_domain(error_indicator)
        if mesh != self.function_space().mesh():
            raise ValueError("Cannot use an error indicator from a different mesh.")
        dim = mesh.topological_dimension()
        if convergence_rate < 1.0:
            raise ValueError(
                f"Convergence rate must be at least one, not {convergence_rate}."
            )
        if min_eigenvalue <= 0.0:
            raise ValueError(
                f"Minimum eigenvalue must be positive, not {min_eigenvalue}."
            )
        if interpolant not in ("Clement", "L2"):
            raise ValueError(f"Interpolant '{interpolant}' not recognised.")
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        P0_metric = P0Metric(P0_ten)

        # Get reference element volume
        K_hat = 1 / 2 if dim == 2 else 1 / 6

        # Get current element volume
        K = K_hat * abs(ufl.JacobianDeterminant(mesh))

        # Get optimal element volume
        P0 = firedrake.FunctionSpace(mesh, "DG", 0)
        K_opt = pow(error_indicator, 1 / (convergence_rate + 1))
        K_opt_av = (
            K_opt / firedrake.assemble(interpolate(K_opt, P0)).dat.global_data.sum()
        )
        K_ratio = target_complexity * pow(abs(K_opt_av * K_hat / K), 2 / dim)

        if self._any_inf(firedrake.assemble(interpolate(K_ratio, P0))):
            raise ValueError("K_ratio contains non-finite values.")

        # Interpolate from P1 to P0
        #   Note that this shouldn't affect symmetric positive-definiteness.
        if hessian is not None:
            hessian.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
        P0_metric.project(hessian or ufl.Identity(dim))

        # Compute stretching factors (in ascending order)
        evectors, evalues = P0_metric.compute_eigendecomposition(reorder=True)
        divisor = pow(np.prod(evalues), 1 / dim)
        modified_evalues = [
            abs(ufl.max_value(e, min_eigenvalue) / divisor) for e in evalues
        ]

        # Assemble metric with modified eigenvalues
        evalues.interpolate(K_ratio * ufl.as_vector(modified_evalues))
        if self._any_inf(evalues):
            raise ValueError(
                "At least one modified stretching factor contains non-finite values."
            )
        P0_metric.assemble_eigendecomposition(evectors, evalues)

        # Interpolate the metric into the target space
        fs = self.function_space()
        metric = RiemannianMetric(fs)
        if interpolant == "Clement":
            metric.assign(clement_interpolant(P0_metric, target_space=fs))
        else:
            metric.project(P0_metric)

        # Rescale to enforce that the target complexity is met
        #   NOTE: We use the L-infinity norm so that the metric is just scaled to the
        #         target metric complexity, as opposed to being redistributed spatially.
        mp["dm_plex_metric_p"] = np.inf
        metric.set_parameters(mp)
        metric.normalise()
        return self.assign(metric)

    @PETSc.Log.EventDecorator()
    def compute_weighted_hessian_metric(
        self,
        error_indicators,
        hessians,
        average=False,
        interpolant="Clement",
    ):
        r"""
        Compute a vertex-wise anisotropic metric from a list of error indicators, given
        a list of corresponding Hessian fields.

        The formulation used is based on that presented in :cite:`Power:2006`. It is
        assumed that the error indicators have been constructed in the appropriate way.

        :arg error_indicators: list of error indicators
        :type error_indicators: :class:`list` of :class:`firedrake.function.Function`\s
        :arg hessians: list of Hessians
        :type hessians: :class:`list` of :class:`~.RiemannianMetric`\s
        :kwarg average: should metric components be averaged or intersected?
        :type average: :class:`bool`
        :kwarg interpolant: interpolation method, chosen from 'Clement' or 'L2'
        :type interpolant: :class:`str`
        :return: the metric, modified in-place
        :rtype: :class:`~.RiemannianMetric`
        """
        if isinstance(error_indicators, firedrake.Function):
            error_indicators = [error_indicators]
        if isinstance(hessians, firedrake.Function):
            hessians = [hessians]
        mesh = self.function_space().mesh()
        P1 = firedrake.FunctionSpace(mesh, "CG", 1)
        for error_indicator, hessian in zip(error_indicators, hessians):
            if mesh != error_indicator.function_space().mesh():
                raise ValueError("Cannot use an error indicator from a different mesh.")
            if mesh != hessian.function_space().mesh():
                raise ValueError("Cannot use a Hessian from a different mesh.")
            if not isinstance(hessian, RiemannianMetric):
                raise TypeError(
                    f"Expected Hessian to be a RiemannianMetric, not {type(hessian)}."
                )
            if interpolant == "Clement":
                error_indicator = clement_interpolant(error_indicator, target_space=P1)
            elif interpolant == "L2":
                error_indicator = firedrake.project(error_indicator, P1)
            else:
                raise ValueError(f"Interpolant '{interpolant}' not recognised.")
            hessian.interpolate(abs(error_indicator) * hessian)
        return self.combine(*hessians, average=average)


class P0Metric(RiemannianMetric):
    r"""
    Subclass of :class:`~.RiemannianMetric` which allows use of :math:`\mathbb P0`
    space.
    """

    def _check_space(self):
        el = self.function_space().ufl_element()
        if (el.family(), el.degree()) != ("Discontinuous Lagrange", 0):
            raise ValueError(f"P0 metric should be in P0 space, not '{el}'.")


@PETSc.Log.EventDecorator()
def determine_metric_complexity(H_interior, H_boundary, target, p, **kwargs):
    """
    Solve an algebraic problem to obtain coefficients for the interior and boundary
    metrics to obtain a given metric complexity.

    See :cite:`Loseille:2010` for details. Note that we use a slightly different
    formulation here.

    :arg H_interior: Hessian component from domain interior
    :type H_interior: :class:`~.RiemannianMetric`
    :arg H_boundary: Hessian component from domain boundary
    :type H_boundary: :class:`~.RiemannianMetric`
    :arg target: target metric complexity
    :type target: :class:`float`
    :arg p: normalisation order
    :type p: :class:`float`
    :kwarg H_interior_scaling: optional scaling for interior component
    :type H_interior_scaling: :class:`float`
    :kwarg H_boundary_scaling: optional scaling for boundary component
    :type H_boundary_scaling: :class:`float`
    :returns: unique solution of algebraic problem
    :rtype: :class:`float`
    """
    d = H_interior.function_space().mesh().topological_dimension()
    if d not in (2, 3):
        raise ValueError(f"Spatial dimension {d} not supported.")
    if np.isinf(p):
        raise NotImplementedError(
            "Metric complexity cannot be determined in the L-infinity case."
        )
    g = kwargs.get("H_interior_scaling", firedrake.Constant(1.0))
    gbar = kwargs.get("H_boundary_scaling", firedrake.Constant(1.0))
    g = pow(g, d / (2 * p + d))
    gbar = pow(gbar, d / (2 * p + d - 1))

    # Compute coefficients for the algebraic problem
    a = firedrake.assemble(g * pow(ufl.det(H_interior), p / (2 * p + d)) * ufl.dx)
    b = firedrake.assemble(
        gbar * pow(ufl.det(H_boundary), p / (2 * p + d - 1)) * ufl.ds
    )

    # Solve algebraic problem
    c = sympy.Symbol("c")
    c = sympy.solve(a * pow(c, d / 2) + b * pow(c, (d - 1) / 2) - target, c)
    eq = f"{a}*c^{d / 2} + {b}*c^{(d - 1) / 2} = {target}"
    if len(c) == 0:
        raise ValueError(f"Could not find any solutions for equation {eq}.")
    elif len(c) > 1:
        raise ValueError(f"Could not find a unique solution for equation {eq}.")
    elif not np.isclose(float(sympy.im(c[0])), 0.0):
        raise ValueError(f"Could not find any real solutions for equation {eq}.")
    else:
        return float(sympy.re(c[0]))


# TODO: Use the intersection functionality in PETSc
#       See https://gitlab.com/petsc/petsc/-/issues/1452
@PETSc.Log.EventDecorator()
def intersect_on_boundary(*metrics, boundary_tag="on_boundary"):
    r"""
    Combine a list of metrics by intersection.

    :arg metrics: the metrics to be combined
    :type metrics: :class:`tuple` of :class:`~.RiemannianMetric`\s
    :kwarg boundary_tag: optional boundary segment physical ID for boundary
        intersection. Otherwise, the intersection is over the whole boundary.
    :type boundary_tag: :class:`str` or :class:`int`
    :returns: the intersected metric
    :rtype: :class:`~.RiemannianMetric`
    """
    n = len(metrics)
    assert n > 0, "Nothing to combine"
    fs = metrics[0].function_space()
    dim = fs.mesh().topological_dimension()
    if dim not in (2, 3):
        raise ValueError(
            f"Spatial dimension {dim} not supported. Must be either 2 or 3."
        )
    for i, metric in enumerate(metrics):
        if not isinstance(metric, RiemannianMetric):
            raise ValueError(
                f"Metric {i} should be of type 'RiemannianMetric',"
                f" but is of type '{type(metric)}'."
            )
        fsi = metric.function_space()
        if fs != fsi:
            raise ValueError(
                f"Function space of metric {i} does not match that"
                f" of metric 0: {fsi} vs. {fs}."
            )

    # Create the metric to be returned
    intersected_metric = RiemannianMetric(fs)
    intersected_metric.assign(metrics[0])

    # Establish the boundary node set
    if isinstance(boundary_tag, (list, tuple)) and len(boundary_tag) == 0:
        raise ValueError(
            "It is unclear what to do with an empty"
            f" {type(boundary_tag)} of boundary tags."
        )
    node_set = firedrake.DirichletBC(fs, 0, boundary_tag).node_set

    # Compute the intersection
    Mtmp = RiemannianMetric(fs)
    for metric in metrics[1:]:
        Mtmp.assign(intersected_metric)
        op2.par_loop(
            get_metric_kernel("intersect", dim),
            node_set,
            intersected_metric.dat(op2.RW),
            Mtmp.dat(op2.READ),
            metric.dat(op2.READ),
        )
    return intersected_metric
