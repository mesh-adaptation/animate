import os


__all__ = ["get_animate_dir", "get_checkpoint_dir"]


def get_venv_path():
    """
    Retrieve the path to the current virtual environment.
    """
    fpath = os.environ.get("VIRTUAL_ENV")
    if fpath is None:
        raise Exception("Virtual environment is not active!")
    return fpath


def get_animate_dir():
    """
    Retrieve the path to the Animate installation.
    """
    if os.environ.get("ANIMATE_DIR"):
        return os.environ["ANIMATE_DIR"]
    else:
        return os.path.join(get_venv_path(), "src", "animate")


def get_checkpoint_dir():
    """
    Retrieve the path to Animate's checkpoint directory.
    """
    if os.environ.get("ANIMATE_CHECKPOINT_DIR"):
        checkpoint_dir = os.environ["ANIMATE_CHECKPOINT_DIR"]
    else:
        checkpoint_dir = os.path.join(get_animate_dir(), ".checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir
