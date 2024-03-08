from animate.utility import get_checkpoint_dir
import os
import unittest


class TestGetCheckpointDir(unittest.TestCase):
    """
    Unit tests for :func:`get_checkpoint_dir`.
    """

    def test_exists(self):
        checkpoint_dir = get_checkpoint_dir()
        self.assertTrue(os.path.exists(checkpoint_dir))
