from animate import *
import os
import unittest


class TestGetVenvPath(unittest.TestCase):
    """
    Unit tests for :func:`get_venv_path`.
    """

    def test_exists(self):
        self.assertTrue(os.path.exists(get_venv_path()))

    def test_env(self):
        venv = os.environ.get("VIRTUAL_ENV")
        self.assertTrue(venv in get_venv_path())

    def test_prompt(self):
        prompt = os.environ.get("VIRTUAL_ENV_PROMPT")[1:-2]
        self.assertTrue(prompt in get_venv_path())

    def test_animate(self):
        fpath = os.path.join(get_venv_path(), "src", "animate")
        self.assertTrue(os.path.exists(fpath))
