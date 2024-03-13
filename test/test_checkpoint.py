from test_setup import *
from animate.checkpointing import _fix_checkpoint_filename
import h5py
import numpy as np
import unittest


class TestCheckpointing(unittest.TestCase):
    """
    Unit tests for methods of :class:`~.MetricBasedAdaptor` related to checkpointing.
    """

    def setUp(self):
        self.mesh = uniform_mesh(2, 1)
        self.metric = RiemannianMetric(TensorFunctionSpace(self.mesh, "CG", 1))

    def test_filepath_error(self):
        with self.assertRaises(ValueError) as cm:
            _fix_checkpoint_filename("path/to/file")
        error_message = str(cm.exception)
        msg = "Provide a filename, not a filepath. Checkpoints will be stored in '"
        self.assertTrue(error_message.startswith(msg))
        self.assertTrue(get_checkpoint_dir() in error_message)

    def test_extension_error(self):
        with self.assertRaises(ValueError) as cm:
            _fix_checkpoint_filename("checkpoint.wrong")
        msg = "File extension '.wrong' not recognised. Use '.h5'."
        self.assertEqual(str(cm.exception), msg)

    def test_file_created(self):
        filename = "test_file_created"
        fname = _fix_checkpoint_filename(filename)
        self.assertTrue(fname.endswith(".h5"))
        self.assertFalse(os.path.exists(fname))
        save_checkpoint(filename, self.metric)
        self.assertTrue(os.path.exists(fname))
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))

    def test_save(self, filename="test_save", metric=None):
        fname = _fix_checkpoint_filename(filename)
        self.assertFalse(os.path.exists(fname))
        metric = metric or self.metric
        save_checkpoint(filename, metric)
        self.assertTrue(os.path.exists(fname))
        mesh_name = metric._mesh.name
        topology_name = firedrake.mesh._generate_default_mesh_topology_name(mesh_name)
        with h5py.File(fname, "r") as h5:
            self.assertTrue("topologies" in h5)
            self.assertTrue(topology_name in h5["topologies"].keys())
        if filename == "test_save":
            os.remove(fname)
            self.assertFalse(os.path.exists(fname))

    def test_load(self, filename="test_load", metric=None):
        fname = _fix_checkpoint_filename(filename)
        self.test_save(filename=filename, metric=metric)
        metric = metric or self.metric
        self.loaded_metric = load_checkpoint(filename, metric._mesh.name, metric.name())
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))
        self.assertTrue(np.allclose(self.metric.dat.data, self.loaded_metric.dat.data))

    def test_load_custom_mesh_name(self):
        custom_mesh_name = "custom_mesh_name"
        custom_mesh = uniform_mesh(2, 1, name=custom_mesh_name)
        custom_metric = RiemannianMetric(TensorFunctionSpace(custom_mesh, "CG", 1))
        self.test_load(filename="test_load_custom_mesh_name", metric=custom_metric)
        self.assertEqual(self.loaded_metric._mesh.name, custom_mesh_name)

    def get_loaded_metric_parameters(self):
        self.assertTrue(hasattr(self, "loaded_metric"))
        mp = self.loaded_metric.metric_parameters
        mp.update(self.loaded_metric._variable_parameters)
        return mp

    def test_load_nonfunction_parameters(self):
        mp = {
            "dm_plex_metric_h_max": 1,  # integer
            "dm_plex_metric_h_min": 1.0e-08,  # float
            "dm_plex_metric_no_surf": True,  # bool
            "dm_plex_metric_boundary_tag": "on_boundary",  # str
        }
        self.metric.set_parameters(mp)
        self.test_load(filename="test_load_nonfunction_parameters")
        loaded_mp = self.get_loaded_metric_parameters()
        for key, value in mp.items():
            self.assertTrue(key in loaded_mp)
            self.assertAlmostEqual(loaded_mp[key], value)

    def test_load_function_parameters(self):
        P1 = FunctionSpace(self.mesh, "CG", 1)
        x, y = SpatialCoordinate(self.mesh)
        a_max = Function(P1).interpolate(1 + x * y)
        self.metric.set_parameters({"dm_plex_metric_a_max": a_max})
        self.test_load(filename="test_load_function_parameters")
        a_max_loaded = self.get_loaded_metric_parameters().get("dm_plex_metric_a_max")
        self.assertIsNotNone(a_max_loaded)
        self.assertTrue(np.allclose(a_max.dat.data, a_max_loaded.dat.data))
