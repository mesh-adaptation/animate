import os


__all__ = ["get_venv_path"]


def get_venv_path():
    """
    Retrieve the path to the current virtual environment.
    """
    fpath = os.environ.get("VIRTUAL_ENV")
    if fpath is None:
        raise Exception("Virtual environment is not active!")
    return fpath
