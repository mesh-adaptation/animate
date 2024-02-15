from test_setup import *
import unittest


class TestSaveCheckpoint(unittest.TestCase):
    """
    Unit tests for methods of :class:`~.MetricBasedAdaptor` related to checkpointing.
    """

    def setUp(self):
        mesh = uniform_mesh(2, 1)
        metric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        self.adaptor = MetricBasedAdaptor(mesh, metric)

    def test_filepath_error(self):
        with self.assertRaises(ValueError) as cm:
            self.adaptor._fix_checkpoint_filename("path/to/file")
        error_message = str(cm.exception)
        msg = "Provide a filename, not a filepath. Checkpoints will be stored in '"
        self.assertTrue(error_message.startswith(msg))
        self.assertTrue(self.adaptor._checkpoint_dir in error_message)

    def test_extension_error(self):
        with self.assertRaises(ValueError) as cm:
            self.adaptor._fix_checkpoint_filename("checkpoint.wrong")
        msg = "File extension '.wrong' not recognised. Use '.h5'."
        self.assertEqual(str(cm.exception), msg)

    def test_file_created(self):
        filename = "test_file_created"
        fname = self.adaptor._fix_checkpoint_filename(filename)
        self.assertFalse(os.path.exists(fname))
        self.adaptor.save_checkpoint(filename)
        self.assertTrue(fname.endswith(".h5"))
        self.assertTrue(os.path.exists(fname))
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))
