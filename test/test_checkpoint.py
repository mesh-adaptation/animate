from test_setup import *
from animate.adapt import _fix_checkpoint_filename
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
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric, name=filename)
        fname = _fix_checkpoint_filename(filename)
        self.assertTrue(fname.endswith(".h5"))
        self.assertFalse(os.path.exists(fname))
        self.adaptor.save_checkpoint(filename)
        self.assertTrue(os.path.exists(fname))
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))

    def test_save(self, filename="test_save"):
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric, name=filename)
        fname = _fix_checkpoint_filename(filename)
        self.assertFalse(os.path.exists(fname))
        self.adaptor.save_checkpoint(filename)
        self.assertTrue(os.path.exists(fname))
        with h5py.File(fname, "r") as h5:
            self.assertTrue("topologies" in h5)
            self.assertTrue("firedrake_default_topology" in h5["topologies"].keys())
        if filename == "test_save":
            os.remove(fname)
            self.assertFalse(os.path.exists(fname))

    def test_load(self):
        filename = "test_load"
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric, name=filename)
        fname = _fix_checkpoint_filename(filename)
        self.test_save(filename=filename)
        metric = load_checkpoint(filename, filename)
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))
        self.assertTrue(np.allclose(self.metric.dat.data, metric.dat.data))

    # TODO: Check saving and loading metric parameters
