import abc
import os
from shutil import rmtree

import firedrake.checkpointing as fchk
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
import firedrake.utils as futils
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.cython.dmcommon import to_petsc_local_numbering
from firedrake.petsc import PETSc
from firedrake.projection import Projector

from .checkpointing import get_checkpoint_dir, load_checkpoint, save_checkpoint
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

    @futils.cached_property
    @PETSc.Log.EventDecorator()
    def adapted_mesh(self):
        """
        Adapt the mesh with respect to the provided metric.

        :returns: the adapted mesh
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        self.metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
        size = self.metric.dat.dataset.layout_vec.getSizes()
        data = self.metric.dat._data[: size[0]]
        v = PETSc.Vec().createWithArray(
            data, size=size, bsize=self.metric.dat.cdim, comm=self.mesh.comm
        )
        reordered = to_petsc_local_numbering(v, self.metric.function_space())
        v.destroy()
        newplex = self.metric._plex.adaptMetric(reordered, "Face Sets", "Cell Sets")
        newplex.setName(fmesh._generate_default_mesh_topology_name(self.name))
        reordered.destroy()
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
        )  # TODO


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

    dim = mesh.topological_dimension()
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

        if COMM_WORLD.rank == 0:
            metric0 = load_checkpoint(chk_fpath, mesh.name, metric_name, comm=COMM_SELF)
            adaptor0 = MetricBasedAdaptor(metric0._mesh, metric0, name=name)
            with fchk.CheckpointFile(chk_fpath, "w", comm=COMM_SELF) as chk:
                chk.save_mesh(adaptor0.adapted_mesh)
        COMM_WORLD.barrier()

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
