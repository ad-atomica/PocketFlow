import os


def ensure_parent_dir_exists(filepath: str) -> None:
    """Ensure the parent directory of filepath exists, creating it if necessary."""
    parent = os.path.dirname(filepath)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
