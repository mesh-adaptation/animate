import abc
import gc
import os
from functools import cached_property
from shutil import rmtree

import firedrake.checkpointing as fchk
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc
from firedrake.projection import Projector

from .checkpointing import get_checkpoint_dir, load_checkpoint, save_checkpoint
from .cython.numbering import to_petsc_local_numbering
from .metric import RiemannianMetric

__all__ = ["MetricBasedAdaptor", "adapt"]


class AdaptorBase(abc.ABC):
    """
    Abstract base class that defines the API for all mesh adaptors.
    """

    def __init__(self, mesh, name=None, comm=None):
        """
        :arg mesh: mesh to be adapted
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :kwarg name: name for the adapted mesh
        :type name: :class:`str`
        :kwarg comm: MPI communicator to use for the adapted mesh
        :type comm: :class:`mpi4py.MPI.Intracom`
        """
        self.mesh = mesh
        self.name = name or mesh.name
        self.comm = comm or mesh.comm

    @abc.abstractmethod
    def adapted_mesh(self):
        """
        Adapt the mesh.

        :returns: the adapted mesh
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        pass

    @abc.abstractmethod
    def interpolate(self, f):
        """
        Interpolate a field from the initial mesh to the adapted mesh.

        :arg f: a Function on the initial mesh
        :type f: :class:`firedrake.function.Function`
        :returns: its interpolation onto the adapted mesh
        :rtype: :class:`firedrake.function.Function`
        """
        pass


class MetricBasedAdaptor(AdaptorBase):
    """
    Class for driving metric-based mesh adaptation.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, metric, name=None, comm=None):
        """
        :arg mesh: mesh to be adapted
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg metric: metric to use for the adaptation
        :type metric: :class:`animate.metric.RiemannianMetric`
        :kwarg name: name for the adapted mesh
        :type name: :class:`str`
        :kwarg comm: MPI communicator to use for the adapted mesh
        :type comm: :class:`mpi4py.MPI.Intracom`
        """
        if metric._mesh is not mesh:
            raise ValueError("The mesh associated with the metric is inconsistent")
        if isinstance(mesh.topology, fmesh.ExtrudedMeshTopology):
            raise NotImplementedError("Cannot adapt extruded meshes")
        coord_fe = mesh.coordinates.ufl_element()
        if (coord_fe.family(), coord_fe.degree()) != ("Lagrange", 1):
            raise NotImplementedError(f"Mesh coordinates must be P1, not {coord_fe}")
        assert isinstance(metric, RiemannianMetric)
        super().__init__(mesh, name=name, comm=comm)
        self.metric = metric
        self.projectors = []
        self.fixed_boundary_id_map = {}

    @staticmethod
    def _facets_in_stratum(plex, label_name, value):
        """
        Get the points assigned to a given value of a label on a plex, restricted to
        those points which are actually facets (i.e., height-1 points).

        :arg plex: the DMPlex to query
        :type plex: :class:`petsc4py.PETSc.DMPlex`
        :arg label_name: the name of the label to query
        :type label_name: :class:`str`
        :arg value: the label value (stratum) to look up
        :type value: :class:`int`
        :returns: the facet point indices assigned to ``value`` in the label
        :rtype: :class:`numpy.ndarray`
        """
        fStart, fEnd = plex.getHeightStratum(1)
        points = plex.getStratumIS(label_name, value).indices
        return points[(points >= fStart) & (points < fEnd)]

    @PETSc.Log.EventDecorator()
    def _color_fixed_boundary_facets(self, fix_boundary):
        """
        Create a "Colored Face Sets" label on ``self.metric._plex``, in which facets
        that are *not* to be fixed retain their original "Face Sets" boundary id,
        whilst facets that *are* to be fixed are each assigned a new value, unique to
        that facet, that does not clash with any existing boundary id.

        :arg fix_boundary: list of boundary ids whose facets should be fixed, or the
            string "on_boundary", meaning that all facets with a boundary id should be
            fixed
        :type fix_boundary: :class:`list` of :class:`int`\\s, or :class:`str`
        :returns: a mapping from each original boundary id to the list of new,
            unique ids assigned to its facets
        :rtype: :class:`dict`
        """
        plex = self.metric._plex
        plex.createLabel("Colored Face Sets")
        colored_label = plex.getLabel("Colored Face Sets")

        fix_all = fix_boundary == "on_boundary"
        boundary_id_map = {} if fix_all else {bid: [] for bid in fix_boundary}
        if not plex.hasLabel("Face Sets"):
            return boundary_id_map

        boundary_ids = [int(b) for b in plex.getLabelIdIS("Face Sets").indices]
        next_id = max(boundary_ids, default=0) + 1

        for bid in boundary_ids:
            facets = self._facets_in_stratum(plex, "Face Sets", bid)
            if fix_all or bid in fix_boundary:
                new_ids = list(range(next_id, next_id + len(facets)))
                next_id += len(facets)
                for facet, new_id in zip(facets, new_ids, strict=True):
                    colored_label.setValue(int(facet), new_id)
                boundary_id_map.setdefault(bid, []).extend(new_ids)
            else:
                for facet in facets:
                    colored_label.setValue(int(facet), bid)
        return boundary_id_map

    @PETSc.Log.EventDecorator()
    def _restore_fixed_boundary_face_sets(self, newplex, boundary_id_map):
        """
        Reconstruct a "Face Sets" label on the adapted ``newplex`` from its
        "Colored Face Sets" label, converting the unique ids introduced by
        :meth:`_color_fixed_boundary_facets` back into their original boundary ids,
        before removing the "Colored Face Sets" label.

        :arg newplex: the adapted DMPlex, which has a "Colored Face Sets" label but no
            "Face Sets" label
        :type newplex: :class:`petsc4py.PETSc.DMPlex`
        :arg boundary_id_map: mapping from original boundary id to the list of new,
            unique ids assigned to its facets prior to adaptation, as returned by
            :meth:`_color_fixed_boundary_facets`
        :type boundary_id_map: :class:`dict`
        """
        new_id_to_original = {
            new_id: original_id
            for original_id, new_ids in boundary_id_map.items()
            for new_id in new_ids
        }
        newplex.createLabel("Face Sets")
        face_sets_label = newplex.getLabel("Face Sets")
        if newplex.hasLabel("Colored Face Sets"):
            colored_ids = [
                int(v) for v in newplex.getLabelIdIS("Colored Face Sets").indices
            ]
            for value in colored_ids:
                original_id = new_id_to_original.get(value, value)
                facets = self._facets_in_stratum(newplex, "Colored Face Sets", value)
                for facet in facets:
                    face_sets_label.setValue(int(facet), original_id)
            newplex.removeLabel("Colored Face Sets")

    @cached_property
    @PETSc.Log.EventDecorator()
    def adapted_mesh(self):
        """
        Adapt the mesh with respect to the provided metric.

        :returns: the adapted mesh
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        self.metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
        data = self.metric.dat.data_ro_with_halos
        v = PETSc.Vec().createWithArray(
            data, size=data.size, bsize=self.metric.dat.cdim, comm=COMM_SELF
        )
        reordered = to_petsc_local_numbering(v, self.metric.function_space())
        v.destroy()

        fix_boundary = getattr(self.metric, "_fix_boundary", None)
        boundary_label_name = "Face Sets"
        if fix_boundary:
            boundary_label_name = "Colored Face Sets"
            self.fixed_boundary_id_map = self._color_fixed_boundary_facets(
                fix_boundary
            )

        newplex = self.metric._plex.adaptMetric(
            reordered, boundary_label_name, "Cell Sets"
        )
        newplex.setName(fmesh._generate_default_mesh_topology_name(self.name))
        reordered.destroy()

        if fix_boundary:
            self._restore_fixed_boundary_face_sets(
                newplex, self.fixed_boundary_id_map
            )

        return fmesh.Mesh(
            newplex,
            distribution_parameters={"partition": False},
            name=self.name,
            comm=self.comm,
        )

    @PETSc.Log.EventDecorator()
    def project(self, f):
        """
        Project a Function into the corresponding FunctionSpace defined on the adapted
        mesh using conservative projection.

        :arg f: a Function on the initial mesh
        :type f: :class:`firedrake.function.Function`
        :returns: its projection onto the adapted mesh
        :rtype: :class:`firedrake.function.Function`
        """
        fs = f.function_space()
        for projector in self.projectors:
            if fs == projector.source.function_space():
                projector.source = f
                return projector.project().copy(deepcopy=True)
        else:
            new_fs = ffs.FunctionSpace(self.adapted_mesh, f.ufl_element())
            projector = Projector(f, new_fs)
            self.projectors.append(projector)
            return projector.project().copy(deepcopy=True)

    @PETSc.Log.EventDecorator()
    def interpolate(self, f):
        """
        Interpolate a :class:`.Function` into the corresponding :class:`.FunctionSpace`
        defined on the adapted mesh.

        :arg f: a Function on the initial mesh
        :type f: :class:`firedrake.function.Function`
        :returns: its interpolation onto the adapted mesh
        :rtype: :class:`firedrake.function.Function`
        """
        raise NotImplementedError(
            "Consistent interpolation has not yet been implemented in parallel"
        )  # TODO (#132)


def adapt(mesh, *metrics, name=None, serialise=None, remove_checkpoints=True):
    r"""
    Adapt a mesh with respect to a metric and some adaptor parameters.

    If multiple metrics are provided, then they are intersected.

    :arg mesh: mesh to be adapted.
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :arg metrics: metrics to guide the mesh adaptation
    :type metrics: :class:`list` of :class:`.RiemannianMetric`\s
    :kwarg name: name for the adapted mesh
    :type name: :class:`str`
    :kwarg serialise: if ``True``, adaptation is done in serial using
        :class:`firedrake.checkpointing.CheckpointFile`s. Defaults to ``True`` if
        the mesh is 2D, and to ``False`` if the mesh is 3D or if the code is already
        run in serial. This is because parallel adaptation is only supported in 3D.
    :type serialise: :class:`bool`
    :kwarg remove_checkpoints: if ``True``, checkpoint files are deleted after use
    :type remove_checkpoints: :class:`bool`
    :returns: the adapted mesh
    :rtype: :class:`~firedrake.mesh.MeshGeometry`
    """
    nprocs = COMM_WORLD.size

    dim = mesh.topological_dimension
    if serialise is None:
        serialise = nprocs > 1 and dim != 3
    elif not serialise and dim != 3:
        raise ValueError("Parallel adaptation is only supported in 3D.")

    # Combine metrics by intersection, if multiple are passed
    metric = metrics[0]
    if len(metrics) > 1:
        metric.intersect(*metrics[1:])

    if serialise:
        # In parallel, save input mesh and metric to a temporary checkpoint directory
        chk_dir = get_checkpoint_dir()
        chk_fpath = os.path.join(chk_dir, "adapted_mesh_checkpoint.h5")
        metric_name = "tmp_metric"
        save_checkpoint(chk_fpath, metric, metric_name)
        # Ensure all processes are finished writing
        COMM_WORLD.barrier()

        if COMM_WORLD.rank == 0:
            metric0 = load_checkpoint(chk_fpath, mesh.name, metric_name, comm=COMM_SELF)
            adaptor0 = MetricBasedAdaptor(metric0._mesh, metric0, name=name)
            with fchk.CheckpointFile(chk_fpath, "w", comm=COMM_SELF) as chk:
                chk.save_mesh(adaptor0.adapted_mesh)
        # Ensure rank 0 is finished writing
        COMM_WORLD.barrier()
        # Garbage collection might be called at different times on different ranks due
        # to diverging paths, which appears to stall in final cleanup on Python
        # system exit. Ensure everything is in-sync again at this point
        gc.collect()

        # In parallel, load from the checkpoint
        if not os.path.exists(chk_fpath):
            raise Exception(f"Adapted mesh file does not exist! Path: {chk_fpath}.")
        with fchk.CheckpointFile(chk_fpath, "r") as chk:
            newmesh = chk.load_mesh(name or fmesh.DEFAULT_MESH_NAME)

        # Delete temporary checkpoint directory
        if remove_checkpoints and COMM_WORLD.rank == 0:
            rmtree(chk_dir)
    else:
        newmesh = MetricBasedAdaptor(mesh, metric, name=name).adapted_mesh
    return newmesh
