import abc
import firedrake.checkpointing as fchk
from firedrake.cython.dmcommon import to_petsc_local_numbering
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc
from firedrake.projection import Projector
import firedrake.utils as futils
from firedrake import CheckpointFile, COMM_WORLD
from animate.metric import RiemannianMetric
from animate.utility import get_animate_dir, get_checkpoint_dir
import os
import subprocess

__all__ = ["MetricBasedAdaptor", "adapt"]


class AdaptorBase(abc.ABC):
    """
    Abstract base class that defines the API for all mesh adaptors.
    """

    def __init__(self, mesh):
        """
        :param mesh: mesh to be adapted
        """
        self.mesh = mesh

    @abc.abstractmethod
    def adapted_mesh(self):
        pass

    @abc.abstractmethod
    def interpolate(self, f):
        """
        Interpolate a field from the initial mesh to the adapted mesh.

        :param f: the field to be interpolated
        """
        pass


class MetricBasedAdaptor(AdaptorBase):
    """
    Class for driving metric-based mesh adaptation.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, metric, name=None):
        """
        :param mesh: :class:`~firedrake.mesh.MeshGeometry` to be adapted
        :param metric: :class:`.RiemannianMetric` to use for the adaptation
        :param name: name for the adapted mesh
        """
        if metric._mesh is not mesh:
            raise ValueError("The mesh associated with the metric is inconsistent")
        if isinstance(mesh.topology, fmesh.ExtrudedMeshTopology):
            raise NotImplementedError("Cannot adapt extruded meshes")
        coord_fe = mesh.coordinates.ufl_element()
        if (coord_fe.family(), coord_fe.degree()) != ("Lagrange", 1):
            raise NotImplementedError(f"Mesh coordinates must be P1, not {coord_fe}")
        assert isinstance(metric, RiemannianMetric)
        super().__init__(mesh)
        self.metric = metric
        self.projectors = []
        if name is None:
            name = mesh.name
        self.name = name

    @futils.cached_property
    @PETSc.Log.EventDecorator()
    def adapted_mesh(self):
        """
        Adapt the mesh with respect to the provided metric.

        :return: a new :class:`~firedrake.mesh.MeshGeometry`.
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
            newplex, distribution_parameters={"partition": False}, name=self.name
        )

    @PETSc.Log.EventDecorator()
    def project(self, f):
        """
        Project a :class:`.Function` into the corresponding :class:`.FunctionSpace`
        defined on the adapted mesh using supermeshing.

        :param: the scalar :class:`.Function` on the initial mesh
        :return: its projection onto the adapted mesh
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

        :param: the scalar :class:`.Function` on the initial mesh
        :return: its interpolation onto the adapted mesh
        """
        raise NotImplementedError(
            "Consistent interpolation has not yet been implemented in parallel"
        )  # TODO

    # --- Checkpointing

    def _fix_checkpoint_filename(self, filename):
        checkpoint_dir = get_checkpoint_dir()
        if "/" in filename:
            raise ValueError(
                "Provide a filename, not a filepath. Checkpoints will be stored in"
                f" '{checkpoint_dir}'."
            )
        name, ext = os.path.splitext(filename)
        ext = ext or ".h5"
        if ext != ".h5":
            raise ValueError(f"File extension '{ext}' not recognised. Use '.h5'.")
        return os.path.join(checkpoint_dir, name + ext)

    def save_checkpoint(self, filename):
        """
        Write the metric and underlying mesh to a :class:`~.CheckpointFile`.

        Note that the checkpoint will be stored within Animate's ``.checkpoints``
        subdirectory.

        :arg filename: the filename to use for the checkpoint
        """
        with fchk.CheckpointFile(self._fix_checkpoint_filename(filename), "w") as chk:
            chk.save_mesh(self.mesh)
            chk.save_function(self.metric)

    def load_checkpoint(self, filename):
        """
        Load a mesh from a :class:`~.CheckpointFile`.

        Note that the checkpoint will have to be stored within Animate's ``.checkpoints``
        subdirectory.

        :arg filename: the filename of the checkpoint
        """
        with fchk.CheckpointFile(self._fix_checkpoint_filename(filename), "w") as chk:
            return chk.load_mesh()


def adapt(mesh, *metrics, name=None, serialise=False, remove_checkpoints=True):
    r"""
    Adapt a mesh with respect to a metric and some adaptor parameters.

    If multiple metrics are provided, then they are intersected.

    :param mesh: :class:`~firedrake.mesh.MeshGeometry` to be adapted.
    :param metrics: list of :class:`.RiemannianMetric`\s
    :param name: name for the adapted mesh
    :param serialise: should the parallel adaptation be done in serial?
    :param remove_checkpoints: should checkpoint files be deleted after use?
    :return: a new :class:`~firedrake.mesh.MeshGeometry`.
    """

    # Parallel adaptation is currently only supported in 3D
    if mesh.topological_dimension() != 3:
        serialise = False

    # If already running in serial then no need to use checkpointing
    if COMM_WORLD.size == 1:
        serialise = False

    # Combine metrics by intersection, if multiple are passed
    metric = metrics[0]
    if len(metrics) > 1:
        metric.intersect(*metrics[1:])

    if serialise:
        checkpoint_dir = get_checkpoint_dir()
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # In parallel, save input mesh and metric to a checkpoint file
        input_fname = os.path.join(checkpoint_dir, "tmp_metric.h5")
        with CheckpointFile(input_fname, "w") as chk:
            chk.save_mesh(mesh)
            chk.save_function(metric, name="tmp_metric")

        # In serial, load the checkpoint, adapt and write out the result
        if COMM_WORLD.rank == 0:
            adapt_script = os.path.join(get_animate_dir(), "animate", "adapt.py")
            subprocess.run(["mpiexec", "-n", "1", "python3", adapt_script])
        COMM_WORLD.barrier()

        # In parallel, load from the checkpoint
        output_fname = os.path.join(checkpoint_dir, "tmp_mesh.h5")
        if not os.path.exists(output_fname):
            raise Exception(f"Adapted mesh file does not exist! Path: {output_fname}.")
        with CheckpointFile(output_fname, "r") as chk:
            newmesh = chk.load_mesh("tmp_adapted_mesh")

        # Delete temporary checkpoint files
        if remove_checkpoints and COMM_WORLD.rank == 0:
            os.remove(input_fname)
            os.remove(output_fname)
        COMM_WORLD.barrier()
    else:
        adaptor = MetricBasedAdaptor(mesh, metric, name=name)
        newmesh = adaptor.adapted_mesh
    return newmesh


if __name__ == "__main__":
    # This section of code is called on a subprocess by adapt
    assert COMM_WORLD.size == 1

    # Load input mesh and metric from checkpoint
    checkpoint_dir = get_checkpoint_dir()
    assert os.path.exists(checkpoint_dir)
    input_fname = os.path.join(checkpoint_dir, "tmp_metric.h5")
    if not os.path.exists(input_fname):
        raise Exception(f"Metric file does not exist! Path: {input_fname}.")
    with CheckpointFile(input_fname, "r") as chk:
        mesh = chk.load_mesh()
        metric = chk.load_function(mesh, "tmp_metric")

    # Convert metric from Function to RiemannianMetric, then adapt
    metric = RiemannianMetric(metric.function_space()).assign(metric)
    adaptor = MetricBasedAdaptor(mesh, metric, name="tmp_adapted_mesh")

    # Write adapted mesh to another checkpoint
    output_fname = os.path.join(checkpoint_dir, "tmp_mesh.h5")
    with CheckpointFile(output_fname, "w") as chk:
        chk.save_mesh(adaptor.adapted_mesh)
