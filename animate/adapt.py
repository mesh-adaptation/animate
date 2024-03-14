import abc
from firedrake.cython.dmcommon import to_petsc_local_numbering
import firedrake.checkpointing as fchk
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc
from firedrake.projection import Projector
import firedrake.utils as futils
from firedrake import COMM_SELF, COMM_WORLD
from .checkpointing import load_checkpoint, save_checkpoint
from .metric import RiemannianMetric
from .utility import get_checkpoint_dir
import os

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
        :kwarg comm: MPI communicator for handling the checkpoint file
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
        :kwarg comm: MPI communicator for handling the checkpoint file
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


def adapt(mesh, *metrics, name=None, serialise=False, remove_checkpoints=True):
    r"""
    Adapt a mesh with respect to a metric and some adaptor parameters.

    If multiple metrics are provided, then they are intersected.

    :arg mesh: mesh to be adapted.
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :arg metrics: metrics to guide the mesh adaptation
    :type metrics: :class:`list` of :class:`.RiemannianMetric`\s
    :kwarg name: name for the adapted mesh
    :type name: :class:`str`
    :kwarg serialise: if ``True``, adaptation is done in serial
    :type serialise: :class:`bool`
    :kwarg remove_checkpoints: if ``True``, checkpoint files are deleted after use
    :type remove_checkpoints: :class:`bool`
    :returns: the adapted mesh
    :rtype: :class:`~firedrake.mesh.MeshGeometry`
    """
    nprocs = COMM_WORLD.size

    # Parallel adaptation is currently only supported in 3D
    if mesh.topological_dimension() != 3:
        serialise = nprocs > 1

    # If already running in serial then no need to use checkpointing
    if nprocs == 1:
        serialise = False

    # Combine metrics by intersection, if multiple are passed
    metric = metrics[0]
    if len(metrics) > 1:
        metric.intersect(*metrics[1:])

    if serialise:
        checkpoint_dir = get_checkpoint_dir()
        if not os.path.exists(checkpoint_dir) and COMM_WORLD.rank == 0:
            os.makedirs(checkpoint_dir)
        COMM_WORLD.barrier()
        metric_name = "tmp_metric"
        metric_fname = "metric_checkpoint"
        input_fname = os.path.join(checkpoint_dir, metric_fname + ".h5")
        output_fname = os.path.join(checkpoint_dir, "adapted_mesh_checkpoint.h5")

        # In parallel, save input mesh and metric to a checkpoint file
        save_checkpoint(metric_fname, metric, metric_name)

        # In serial, load the checkpoint, adapt and write out the result
        PETSc.Sys.Print("Processor %d" % COMM_WORLD.rank, comm=COMM_SELF)
        saved = [False] * COMM_WORLD.size
        if COMM_WORLD.rank == 0:
            PETSc.Sys.Print("Processor %d: DEBUG 1" % COMM_WORLD.rank, comm=COMM_SELF)
            PETSc.Sys.Print("Processor %d: DEBUG 2" % COMM_WORLD.rank, comm=COMM_SELF)
            metric0 = load_checkpoint(
                metric_fname, mesh.name, metric_name, comm=COMM_SELF
            )
            PETSc.Sys.Print("Processor %d: DEBUG 3" % COMM_WORLD.rank, comm=COMM_SELF)
            adaptor0 = MetricBasedAdaptor(metric0._mesh, metric0, name=name)
            PETSc.Sys.Print("Processor %d: DEBUG 4" % COMM_WORLD.rank, comm=COMM_SELF)
            with fchk.CheckpointFile(output_fname, "w", comm=COMM_SELF) as chk:
                PETSc.Sys.Print(
                    "Processor %d: DEBUG 5" % COMM_WORLD.rank, comm=COMM_SELF
                )
                chk.save_mesh(adaptor0.adapted_mesh)
            PETSc.Sys.Print("Processor %d: DEBUG 6" % COMM_WORLD.rank, comm=COMM_SELF)
            saved[0] = True
            PETSc.Sys.Print("Processor %d: DEBUG 7" % COMM_WORLD.rank, comm=COMM_SELF)
        # COMM_WORLD.barrier()
        if not COMM_WORLD.scatter(saved, root=0):
            raise Exception

        # In parallel, load from the checkpoint
        if not os.path.exists(output_fname):
            raise Exception(f"Adapted mesh file does not exist! Path: {output_fname}.")
        with fchk.CheckpointFile(output_fname, "r") as chk:
            newmesh = chk.load_mesh(name or fmesh.DEFAULT_MESH_NAME)

        # Delete temporary checkpoint files
        if remove_checkpoints and COMM_WORLD.rank == 0:
            os.remove(input_fname)
            os.remove(output_fname)
    else:
        newmesh = MetricBasedAdaptor(mesh, metric, name=name).adapted_mesh
    return newmesh
