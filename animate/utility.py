import os


__all__ = ["get_checkpoint_dir"]


def get_checkpoint_dir():
    """
    Retrieve the path to Animate's checkpoint directory.
    """
    if os.environ.get("ANIMATE_CHECKPOINT_DIR"):
        checkpoint_dir = os.environ["ANIMATE_CHECKPOINT_DIR"]
    else:
        import animate

        assert os.path.basename(animate.__file__) == "__init__.py"
        animate_base_dir = os.path.dirname(animate.__file__)
        checkpoint_dir = os.path.join(animate_base_dir, ".checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
