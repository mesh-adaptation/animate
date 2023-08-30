from .recovery import (
    recover_gradient_l2,
    recover_hessian_clement,
    recover_boundary_hessian,
)
from collections.abc import Iterable
import firedrake
import firedrake.function as ffunc
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc, OptionsManager
import numpy as np
import ufl

__all__ = ["RiemannianMetric"]


class RiemannianMetric(ffunc.Function):
    r"""
    Class for defining a Riemannian metric over a
    given mesh.

    A metric is a symmetric positive-definite field,
    which conveys how the mesh is to be adapted. If
    the mesh is of dimension :math:`d` then the metric
    takes the value of a square :math:`d\times d`
    matrix at each point.

    The implementation of metric-based mesh adaptation
    used in PETSc assumes that the metric is piece-wise
    linear and continuous, with its degrees of freedom
    at the mesh vertices.

    For details, see the PETSc manual entry:
      https://petsc.org/release/docs/manual/dmplex/#metric-based-mesh-adaptation
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, function_space, *args, **kwargs):
        r"""
        :param function_space: the tensor :class:`~.FunctionSpace`, on which to build
            this :class:`~.RiemannianMetric`. Alternatively, another :class:`~.Function`
            may be passed here and its function space will be used to build this
            :class:`~.Function`. In this case, the function values are copied. If a
            :class:`~firedrake.mesh.MeshGeometry` is passed here then a tensor :math:`\mathbb P1`
            space is built on top of it.
        """
        if isinstance(function_space, fmesh.MeshGeometry):
            function_space = ffs.TensorFunctionSpace(function_space, "CG", 1)
        super().__init__(function_space, *args, **kwargs)
        self.metric_parameters = {}

        # Check that we have an appropriate tensor P1 function
        fs = self.function_space()
        mesh = fs.mesh()
        tdim = mesh.topological_dimension()
        if tdim not in (2, 3):
            raise ValueError(f"Riemannian metric should be 2D or 3D, not {tdim}D")
        self._check_space()
        if isinstance(fs.dof_count, Iterable):
            raise ValueError("Riemannian metric cannot be built in a mixed space")
        rank = len(fs.dof_dset.dim)
        if rank != 2:
            raise ValueError(
                "Riemannian metric should be matrix-valued,"
                f" not rank-{rank} tensor-valued"
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
        plex.setSection(mesh.create_section(entity_dofs))

    def _check_space(self):
        el = self.function_space().ufl_element()
        if (el.family(), el.degree()) != ("Lagrange", 1):
            raise ValueError(f"Riemannian metric should be in P1 space, not '{el}'.")

    @staticmethod
    def _process_parameters(metric_parameters):
        mp = metric_parameters.copy()
        if "dm_plex_metric" in mp:
            for key, value in mp["dm_plex_metric"].items():
                mp["_".join(["dm_plex_metric", key])] = value
            mp.pop("dm_plex_metric")
        return mp

    def set_parameters(self, metric_parameters={}):
        """
        Set metric parameter values internally.

        :param metric_parameters: a dictionary of parameters to be passed to PETSc's
            Riemannian metric implementation. All such options have the prefix
            `dm_plex_metric_`.
        """
        mp = self._process_parameters(metric_parameters)
        self.metric_parameters.update(mp)
        opts = OptionsManager(self.metric_parameters, "")
        with opts.inserted_options():
            self._plex.metricSetFromOptions()
        if self._plex.metricIsIsotropic():
            raise NotImplementedError(
                "Isotropic metric optimisations are not supported in Firedrake"
            )
        if self._plex.metricIsUniform():
            raise NotImplementedError(
                "Uniform optimisations are not supported in Firedrake"
            )

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
        coord_section = self._mesh.create_section(entity_dofs)
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

        :param deepcopy: If ``True``, the new
            :class:`~.RiemannianMetric` will allocate new space
            and copy values.  If ``False``, the default, then the new
            :class:`~.RiemannianMetric` will share the dof values.
        :return: a copy of the metric with the same parameters set
        """
        metric = type(self)(super().copy(deepcopy=deepcopy))
        metric.set_parameters(self.metric_parameters.copy())
        return metric

    @PETSc.Log.EventDecorator()
    def compute_hessian(self, field, method="mixed_L2", **kwargs):
        """
        Recover the Hessian of a scalar field.

        :arg f: the scalar field whose Hessian we seek to recover
        :kwarg method: recovery method

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
            )  # TODO
        else:
            raise ValueError(f"Recovery method '{method}' not recognised.")

    @PETSc.Log.EventDecorator()
    def compute_boundary_hessian(self, f, method="mixed_L2", **kwargs):
        """
        Recover the Hessian of a scalar field on the domain boundary.

        :arg f: field to recover over the domain boundary
        :kwarg method: choose from 'mixed_L2' and 'Clement'
        """
        return self.assign(recover_boundary_hessian(f, method=method, **kwargs))

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

        :param restrict_sizes: should minimum and maximum metric magnitudes be
            enforced?
        :param restrict_anisotropy: should maximum anisotropy be enforced?
        :return: the :class:`~.RiemannianMetric`, modified in-place.
        """
        kw = {
            "restrictSizes": restrict_sizes,
            "restrictAnisotropy": restrict_anisotropy,
        }
        v = self._create_from_array(self.dat.data_with_halos)
        det = self._plex.metricDeterminantCreate()
        self._plex.metricEnforceSPD(v, v, det, **kw)
        size = np.shape(self.dat.data_with_halos)
        self.dat.data_with_halos[:] = np.reshape(v.array, size)
        v.destroy()
        return self

    @PETSc.Log.EventDecorator()
    def normalise(self, global_factor=None, boundary=False, **kwargs):
        """
        Apply :math:`L^p` normalisation to the metric.

        :param global_factor: pre-computed global normalisation factor
        :param boundary: is the normalisation to be done over the boundary?
        :param restrict_sizes: should minimum and maximum metric magnitudes be
            enforced?
        :param restrict_anisotropy: should maximum anisotropy be enforced?
        :return: the normalised :class:`~.RiemannianMetric`, modified in-place
        """
        kwargs.setdefault("restrict_sizes", True)
        kwargs.setdefault("restrict_anisotropy", True)
        d = self._tdim
        if kwargs.get("boundary", False):
            d -= 1
        p = self.metric_parameters.get("dm_plex_metric_p", 1.0)
        if not np.isinf(p) and p < 1.0:
            raise ValueError(f"Metric normalisation order must be at least 1, not {p}.")
        target = self.metric_parameters.get("dm_plex_metric_target_complexity")
        if target is None:
            raise ValueError("dm_plex_metric_target_complexity must be set.")

        # Enforce that the metric is SPD
        self.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)

        # Compute global normalisation factor
        detM = ufl.det(self)
        if global_factor is None:
            dX = (ufl.ds if boundary else ufl.dx)(domain=self._mesh)
            exponent = 0.5 if np.isinf(p) else (p / (2 * p + d))
            integral = firedrake.assemble(pow(detM, exponent) * dX)
            global_factor = firedrake.Constant(pow(target / integral, 2 / d))

        # Normalise the metric
        determinant = 1 if np.isinf(p) else pow(detM, -1 / (2 * p + d))
        self.interpolate(global_factor * determinant * self)

        # Enforce element constraints
        return self.enforce_spd(**kwargs)

    # --- Methods for combining metrics

    @PETSc.Log.EventDecorator()
    def intersect(self, *metrics):
        """
        Intersect the metric with other metrics.

        Metric intersection means taking the minimal ellipsoid in the direction of each
        eigenvector at each point in the domain.

        :param metrics: the metrics to be intersected with
        :return: the intersected :class:`~.RiemannianMetric`, modified in-place
        """
        fs = self.function_space()
        for metric in metrics:
            assert isinstance(metric, RiemannianMetric)
            fsi = metric.function_space()
            if fs != fsi:
                raise ValueError(
                    "Cannot combine metrics from different function spaces:"
                    f" {fs} vs. {fsi}."
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
        """
        Average the metric with other metrics.

        :param metrics: the metrics to be averaged with
        :param weights: list of weights to apply to each metric
        :return: the averaged :class:`~.RiemannianMetric`, modified in-place
        """
        num_metrics = len(metrics) + 1
        if weights is None:
            weights = np.ones(num_metrics) / num_metrics
        if len(weights) != num_metrics:
            raise ValueError(
                f"Number of weights ({len(weights)}) does not match"
                f" number of metrics ({num_metrics})."
            )
        self *= weights[0]
        fs = self.function_space()
        for i, metric in enumerate(metrics):
            assert isinstance(metric, RiemannianMetric)
            fsi = metric.function_space()
            if fs != fsi:
                raise ValueError(
                    "Cannot combine metrics from different function spaces:"
                    f" {fs} vs. {fsi}."
                )
            self += weights[i + 1] * metric
        return self

    # --- Metric diagnostics

    @PETSc.Log.EventDecorator()
    def complexity(self, boundary=False):
        """
        Compute the metric complexity - the continuous analogue
        of the (inherently discrete) mesh vertex count.

        :param boundary: should the complexity be computed over the
            domain boundary?
        :return: the complexity of the :class:`~.RiemannianMetric`
        """
        dX = ufl.ds if boundary else ufl.dx
        return firedrake.assemble(ufl.sqrt(ufl.det(self)) * dX)
