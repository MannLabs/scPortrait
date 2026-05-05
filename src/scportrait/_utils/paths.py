import os
from pathlib import Path


def normalize_path(path: str | os.PathLike[str]) -> Path:
    """Return a concrete ``Path`` for internal path operations."""
    return Path(path)


def path_suffix(path: str | os.PathLike[str]) -> str:
    """Return a normalized suffix without the leading dot."""
    return normalize_path(path).suffix.lstrip(".")
