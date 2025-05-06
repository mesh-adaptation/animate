import os
from tempfile import mkdtemp

import firedrake
import firedrake.checkpointing as fchk
import firedrake.function as ffunc

from .metric import RiemannianMetric

__all__ = ["get_checkpoint_dir", "load_checkpoint", "save_checkpoint"]


def get_checkpoint_dir():
    """
    Make a temporary directory for checkpointing and return its path.

    :returns: path to the temporary checkpoint directory
    :rtype: :class:`str`
    """
    if os.environ.get("ANIMATE_CHECKPOINT_DIR"):
        checkpoint_dir = os.environ["ANIMATE_CHECKPOINT_DIR"]
    else:
        animate_base_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoint_dir = os.path.join(animate_base_dir, ".checkpoints")
    comm = firedrake.COMM_WORLD
    if comm.rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tmpdir = mkdtemp(prefix="animate-checkpoint", dir=checkpoint_dir)
        comm.bcast(tmpdir, root=0)
    else:
        tmpdir = comm.bcast(None, root=0)
    comm.barrier()
    return tmpdir


def load_checkpoint(filepath, mesh_name, metric_name, comm=firedrake.COMM_WORLD):
    """
    Load a metric from a :class:`~.CheckpointFile`.

    Note that the checkpoint will have to be stored within Animate's ``.checkpoints``
    subdirectory.

    :arg filepath: the path to the checkpoint file
    :type filepath: :class:`str`
    :arg mesh_name: the name under which the mesh is saved in the checkpoint file
    :type mesh_name: :class:`str`
    :arg metric_name: the name under which the metric is saved in the checkpoint file
    :type metric_name: :class:`str`
    :kwarg comm: MPI communicator for handling the checkpoint file
    :type comm: :class:`mpi4py.MPI.Intracom`
    :returns: the metric loaded from the checkpoint
    :rtype: :class:`animate.metric.RiemannianMetric`
    """
    if not os.path.exists(filepath):
        raise Exception(f"Metric file does not exist! Path: {filepath}.")
    with fchk.CheckpointFile(filepath, "r", comm=comm) as chk:
        mesh = chk.load_mesh(mesh_name)
        metric = chk.load_function(mesh, metric_name)

        # Load stashed metric parameters
        mp = chk._read_pickled_dict("metric_parameters", "mp_dict")
        for key, value in mp.items():
            if value == "Function":
                mp[key] = chk.load_function(mesh, key)

    metric = RiemannianMetric(metric.function_space()).assign(metric)
    metric.set_parameters(mp)
    return metric


def save_checkpoint(filepath, metric, metric_name=None, comm=firedrake.COMM_WORLD):
    """
    Write the metric and underlying mesh to a :class:`~.CheckpointFile`.

    Note that the checkpoint will be stored within Animate's ``.checkpoints``
    subdirectory.

    :arg filepath: the path of the checkpoint file
    :type filepath: :class:`str`
    :arg metric: the metric to save to the checkpoint
    :type metric: :class:`animate.metric.RiemannianMetric`
    :kwarg metric_name: the name under which to save the metric in the checkpoint file
    :type metric_name: :class:`str`
    :kwarg comm: MPI communicator for handling the checkpoint file
    :type comm: :class:`mpi4py.MPI.Intracom`
    """
    mp = metric.metric_parameters.copy()
    with fchk.CheckpointFile(filepath, "w", comm=comm) as chk:
        chk.save_mesh(metric._mesh)
        chk.save_function(metric, name=metric_name or metric.name())

        # Stash metric parameters
        for key, value in metric._variable_parameters.items():
            if isinstance(value, ffunc.Function):
                chk.save_function(value, name=key)
                mp[key] = "Function"
            elif isinstance(value, firedrake.Constant):
                mp[key] = float(value)
            else:
                mp[key] = value
        chk._write_pickled_dict("metric_parameters", "mp_dict", mp)
