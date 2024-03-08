import os


__all__ = ["get_checkpoint_dir"]


def get_venv_path():
    """
    Retrieve the path to the current virtual environment.
    """
    fpath = os.environ.get("VIRTUAL_ENV")
    if fpath is None:
        raise Exception("Virtual environment is not active!")
    return fpath


def get_checkpoint_dir():
    """
    Retrieve the path to Animate's checkpoint directory.
    """
    if os.environ.get("ANIMATE_CHECKPOINT_DIR"):
        checkpoint_dir = os.environ["ANIMATE_CHECKPOINT_DIR"]
    else:
        import animate

        checkpoint_dir = os.path.join(animate.__file__[:-20], ".checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir
