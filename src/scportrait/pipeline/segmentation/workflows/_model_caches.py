import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

from cellpose import models, utils

ZENODO_RECORD_ID = "17564109"
_LEGACY_CHANNEL_MODELS = {"cyto", "cyto2", "cyto3", "nuclei"}


def _make_zenodo_download_link(record_id: str, filename: str) -> str:
    """
    Construct a direct download URL for a file stored in a Zenodo record.

    Args:
        record_id : The Zenodo record identifier (e.g., "1234567").
        filename : The exact filename stored in the Zenodo record (case sensitive).

    Returns
    -------
    str
        A direct HTTPS download URL suitable for urllib / requests / wget.
    """
    return f"https://zenodo.org/records/{record_id}/files/{quote(filename)}?download=1"


def _get_model_dir() -> Path:
    model_dir = getattr(models, "MODEL_DIR", None)
    if model_dir is None:
        return Path.home().joinpath(".cellpose", "models")
    return Path(model_dir)


def _scportrait_cache_model_path(basename: str) -> str:
    """Download a model from a public Zenodo share into Cellpose's model cache if missing."""
    model_dir = _get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    url = _make_zenodo_download_link(
        record_id=ZENODO_RECORD_ID,
        filename=basename,
    )
    cached_file = model_dir / basename

    if not cached_file.exists():
        print(f'Downloading: "{url}" -> {cached_file}')
        utils.download_url_to_file(url, os.fspath(cached_file), progress=True)

    return os.fspath(cached_file)


def _model_path(model_type: str, model_index: int = 0) -> str:
    """Return local path to a legacy channel-aware Cellpose model (downloading if needed)."""
    torch_str = "torch"
    if model_type in ("cyto", "cyto2", "nuclei"):
        basename = f"{model_type}{torch_str}_{model_index}"
    else:
        basename = model_type
    return _scportrait_cache_model_path(basename)


def _size_model_path(model_type: str) -> str | None:
    """Return local path to the size model (downloading if needed)."""
    torch_str = "torch"

    if model_type in ("cyto", "nuclei", "cyto2", "cyto3"):
        if model_type == "cyto3":
            basename = f"size_{model_type}.npy"
        else:
            basename = f"size_{model_type}{torch_str}_0.npy"
        return _scportrait_cache_model_path(basename)
    return None


def _download_model(name: str) -> str:
    """
    Resolve a model reference to a local file path in the Cellpose cache.

    Cellpose 4 removed `models.Cellpose` and defaults to `cpsam`; for scPortrait we still
    need explicit legacy channel-aware models for workflows that pass channel pairs.
    """
    model_path_fn = getattr(models, "model_path", None)

    if name in _LEGACY_CHANNEL_MODELS:
        model_file = _model_path(name)
        _size_model_path(name)
        return model_file

    if callable(model_path_fn):
        try:
            return os.fspath(model_path_fn(name))
        except HTTPError:
            print("Cellpose model server appears to be down. Trying scPortrait backup cache...")
        except (FileNotFoundError, OSError, TypeError, ValueError):
            # Fall through to backup cache handling.
            pass

    try:
        model_file = _model_path(name)
        _size_model_path(name)
        print("Cellpose model and size file downloaded from scPortrait cache.")
        return model_file
    except HTTPError as e:
        raise FileNotFoundError(f"Could not resolve Cellpose model '{name}' via Cellpose or scPortrait cache.") from e
