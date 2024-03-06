from test_setup import *
import h5py
import unittest


class TestCheckpointing(unittest.TestCase):
    """
    Unit tests for methods of :class:`~.MetricBasedAdaptor` related to checkpointing.
    """

    def setUp(self):
        self.mesh = uniform_mesh(2, 1)
        self.metric = RiemannianMetric(TensorFunctionSpace(self.mesh, "CG", 1))

    def test_filepath_error(self):
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric)
        with self.assertRaises(ValueError) as cm:
            self.adaptor._fix_checkpoint_filename("path/to/file")
        error_message = str(cm.exception)
        msg = "Provide a filename, not a filepath. Checkpoints will be stored in '"
        self.assertTrue(error_message.startswith(msg))
        self.assertTrue(get_checkpoint_dir() in error_message)

    def test_extension_error(self):
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric)
        with self.assertRaises(ValueError) as cm:
            self.adaptor._fix_checkpoint_filename("checkpoint.wrong")
        msg = "File extension '.wrong' not recognised. Use '.h5'."
        self.assertEqual(str(cm.exception), msg)

    def test_file_created(self):
        filename = "test_file_created"
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric, name=filename)
        fname = self.adaptor._fix_checkpoint_filename(filename)
        self.assertTrue(fname.endswith(".h5"))
        self.assertFalse(os.path.exists(fname))
        self.adaptor.save_checkpoint(filename)
        self.assertTrue(os.path.exists(fname))
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))

    def test_save(self, filename="test_save"):
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric, name=filename)
        fname = self.adaptor._fix_checkpoint_filename(filename)
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
        # FIXME
        filename = "test_load"
        self.adaptor = MetricBasedAdaptor(self.mesh, self.metric, name=filename)
        fname = self.adaptor._fix_checkpoint_filename(filename)
        self.test_save(filename=filename)
        metric = self.adaptor.load_checkpoint(filename)
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))
        self.assertAlmostEqual(errornorm(self.metric, metric), 0)

    # TODO: Check saving and loading metric parameters
