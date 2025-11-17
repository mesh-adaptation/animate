"""
Checks that all demo scripts run.
"""

import glob
import os
import re
import shutil
from os.path import splitext

import pytest

cwd = os.path.abspath(os.path.dirname(__file__))
demo_dir = os.path.abspath(os.path.join(cwd, "..", "demos"))
all_demos = glob.glob(os.path.join(demo_dir, "*.py"))

# Modifications dictionary to cut down run time of demos:
# - Each key is the name of a demo script to be modified
# - Each value is a dictionary where:
# -- The key is the original string or regex pattern that we wish to replace
# -- The value is the replacement string (use "" to remove the original code)
modifications = {
    "ping_pong.py": {"niter = 50": "niter = 1"},
    "bubble_shear.py": {
        "simulation_end_time = T / 2.0": "simulation_end_time = T / 10.0",
        "num_adaptations = 15": "num_adaptations = 2",
    },
    "metric_normalisation.py": {
        "num_adapt = 4": "num_adapt = 1",
        "target_complexity = 10000.0": "target_complexity = 100.0",
    },
}


@pytest.fixture(params=all_demos, ids=lambda x: splitext(x.split("demos/")[-1])[0])
def demo_file(request):
    return os.path.abspath(request.param)


def test_modifications_demo_exists():
    """
    Check that all demos named in the modifications dictionary exist in the 'demos' dir.
    """
    for demo_name in modifications.keys():
        demo_fpath = os.path.join(demo_dir, demo_name)
        assert demo_fpath in all_demos, f"Error: Demo '{demo_name}' not found."


def test_modifications_original_exists():
    """
    Check that all 'original' code snippets in the modifications dictionary exist in the
    corresponding demo scripts.
    """
    for demo_name, changes in modifications.items():
        demo_path = os.path.join(demo_dir, demo_name)
        with open(demo_path, "r") as file:
            demo_content = file.read()
            for original in changes.keys():
                assert re.search(original, demo_content, re.DOTALL), (
                    f"Error: '{original}' not found in '{demo_name}'."
                )


def test_demos(demo_file, tmpdir, monkeypatch):
    assert os.path.isfile(demo_file), f"Demo file '{demo_file}' not found."

    # Copy mesh files
    source = os.path.dirname(demo_file)
    for f in glob.glob(os.path.join(source, "*.msh")):
        shutil.copy(f, str(tmpdir))

    # Change working directory to temporary directory
    monkeypatch.chdir(tmpdir)

    # Read the original demo script
    with open(demo_file, "r") as f:
        demo_content = f.read()

    # Modify the demo script to shorten run time
    demo_name = os.path.basename(demo_file)
    if demo_name in modifications:
        for original, replacement in modifications[demo_name].items():
            demo_content = re.sub(original, replacement, demo_content, flags=re.DOTALL)

    # Execute the modified demo as a standalone script in a clean namespace
    exec_namespace = {
        "__file__": demo_file,
        "__name__": "__main__",
        "__package__": None,
    }
    exec(demo_content, exec_namespace)

    # Clean up plots
    for ext in ("jpg", "pvd", "vtu"):
        for f in glob.glob(f"*.{ext}"):
            os.remove(f)
