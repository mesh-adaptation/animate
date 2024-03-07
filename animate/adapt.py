import abc
import firedrake.checkpointing as fchk
from firedrake.cython.dmcommon import to_petsc_local_numbering
import firedrake.function as ffunc
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc
from firedrake.projection import Projector
import firedrake.utils as futils
from firedrake import COMM_WORLD
from animate.metric import RiemannianMetric
from animate.utility import get_animate_dir, get_checkpoint_dir
import os
import subprocess

__all__ = ["load_checkpoint", "MetricBasedAdaptor", "adapt"]


# TODO: This should live somewhere else
def _fix_checkpoint_filename(filename):
    """
    Convert a checkpoint filename to absolute form.

    :arg filename: the filename without its path
    :type filename: :class:`str`
    :returns: the absolute filename
    :rtype: :class:`str`
    """
    if "/" in filename:
        raise ValueError(
            "Provide a filename, not a filepath. Checkpoints will be stored in"
            f" '{get_checkpoint_dir()}'."
        )
    name, ext = os.path.splitext(filename)
    ext = ext or ".h5"
    if ext != ".h5":
        raise ValueError(f"File extension '{ext}' not recognised. Use '.h5'.")
    return os.path.join(get_checkpoint_dir(), name + ext)


# TODO: This should live somewhere else
def load_checkpoint(filename, metric_name):
    """
    Load a metric from a :class:`~.CheckpointFile`.

    Note that the checkpoint will have to be stored within Animate's ``.checkpoints``
    subdirectory.

    :arg filename: the filename of the checkpoint
    :type filename: :class:`str`
    :arg metric_name: the name used to store the metric
    :type metric_name: :class:`str`
    :returns: the mesh loaded from the checkpoint
    :rtype: :class:`firedrake.mesh.MeshGeometry`
    """
    fname = _fix_checkpoint_filename(filename)
    if not os.path.exists(fname):
        raise Exception(f"Metric file does not exist! Path: {fname}.")
    with fchk.CheckpointFile(fname, "r") as chk:
        mesh = chk.load_mesh()
        metric = chk.load_function(mesh, metric_name)

        # Load stashed metric parameters
        mp = chk._read_pickled_dict("metric_parameters", "mp_dict")
        for key, value in mp.items():
            if value == "Function":
                mp[key] = chk.load_function(mesh, key)

    metric = RiemannianMetric(metric.function_space()).assign(metric)
    metric.set_parameters(mp)
    return metric


class AdaptorBase(abc.ABC):
    """
    Abstract base class that defines the API for all mesh adaptors.
    """

    def __init__(self, mesh):
        """
        :arg mesh: mesh to be adapted
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        """
        self.mesh = mesh

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
    def __init__(self, mesh, metric, name=None):
        """
        :arg mesh: mesh to be adapted
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg metric: metric to use for the adaptation
        :type metric: :class:`animate.metric.RiemannianMetric`
        :kwarg name: name for the adapted mesh
        :type name: :class:`str`
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
            newplex, distribution_parameters={"partition": False}, name=self.name
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

    # --- Checkpointing

    def save_checkpoint(self, filename, metric_name=None):
        """
        Write the metric and underlying mesh to a :class:`~.CheckpointFile`.

        Note that the checkpoint will be stored within Animate's ``.checkpoints``
        subdirectory.

        :arg filename: the filename to use for the checkpoint
        :type filename: :class:`str`
        :kwarg metric_name: the name to save the metric under
        :type metric_name: :class:`str`
        """
        mp = self.metric.metric_parameters.copy()
        with fchk.CheckpointFile(_fix_checkpoint_filename(filename), "w") as chk:
            chk.save_mesh(self.mesh)
            chk.save_function(self.metric, name=metric_name or self.name)

            # Stash metric parameters
            for key, value in mp.items():
                if isinstance(value, ffunc.Function):
                    chk.save_function(value, name=key)
                    mp[key] = "Function"
            chk._write_pickled_dict("metric_parameters", "mp_dict", mp)


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

    adaptor = MetricBasedAdaptor(mesh, metric, name=name)
    if serialise:
        checkpoint_dir = get_checkpoint_dir()
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # In parallel, save input mesh and metric to a checkpoint file
        input_fname = os.path.join(checkpoint_dir, "metric_checkpoint.h5")
        adaptor.save_checkpoint("metric_checkpoint", metric_name="tmp_metric")
        # In serial, load the checkpoint, adapt and write out the result
        if COMM_WORLD.rank == 0:
            adapt_script = os.path.join(get_animate_dir(), "animate", "adapt.py")
            subprocess.run(["mpiexec", "-n", "1", "python3", adapt_script])
        COMM_WORLD.barrier()

        # In parallel, load from the checkpoint
        output_fname = os.path.join(checkpoint_dir, "adapted_mesh_checkpoint.h5")
        if not os.path.exists(output_fname):
            raise Exception(f"Adapted mesh file does not exist! Path: {output_fname}.")
        with fchk.CheckpointFile(output_fname, "r") as chk:
            newmesh = chk.load_mesh("tmp_adapted_mesh")

        # Delete temporary checkpoint files
        if remove_checkpoints and COMM_WORLD.rank == 0:
            os.remove(input_fname)
            os.remove(output_fname)
        COMM_WORLD.barrier()
    else:
        newmesh = adaptor.adapted_mesh
    return newmesh


if __name__ == "__main__":
    # This section of code is called on a subprocess by adapt
    assert COMM_WORLD.size == 1

    # Load metric from checkpoint then adapt
    metric = load_checkpoint("metric_checkpoint", "tmp_metric")
    adaptor = MetricBasedAdaptor(metric._mesh, metric, name="tmp_adapted_mesh")

    # Write adapted mesh to another checkpoint
    output_fname = os.path.join(get_checkpoint_dir(), "adapted_mesh_checkpoint.h5")
    with fchk.CheckpointFile(output_fname, "w") as chk:
        chk.save_mesh(adaptor.adapted_mesh)
