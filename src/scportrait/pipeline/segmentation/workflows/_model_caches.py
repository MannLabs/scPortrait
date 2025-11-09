import os
from urllib.error import HTTPError
from urllib.parse import quote

from cellpose import utils
from cellpose.models import MODEL_DIR, model_path

ZENODO_RECORD_ID = "17564109"


def _make_zenodo_download_link(record_id: str, filename: str) -> str:
    """
    Construct a direct download URL for a file stored in a Zenodo record.

    Parameters
    ----------
    record_id : str
        The Zenodo record identifier (e.g., "1234567").
    filename : str
        The exact filename stored in the Zenodo record (case sensitive).

    Returns
    -------
    str
        A direct HTTPS download URL suitable for urllib / requests / wget.
    """
    return f"https://zenodo.org/records/{record_id}/files/{quote(filename)}?download=1"


def _scportrait_cache_model_path(basename: str) -> None:
    """Download a model from a public Nextcloud share into Cellpose's model cache if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    url = _make_zenodo_download_link(
        record_id=ZENODO_RECORD_ID,
        filename=basename,
    )
    cached_file = MODEL_DIR / basename

    if not cached_file.exists():
        print(f'Downloading: "{url}" → {cached_file}')
        utils.download_url_to_file(url, os.fspath(cached_file), progress=True)

    return None


def _model_path(model_type: str, model_index: int = 0) -> None:
    """Return local path to a Cellpose model (downloading if needed)."""
    torch_str = "torch"
    if model_type in ("cyto", "cyto2", "nuclei"):
        basename = f"{model_type}{torch_str}_{model_index}"
    else:
        basename = model_type
    return _scportrait_cache_model_path(basename)


def _size_model_path(model_type: str) -> None:
    """Return local path to the size model (downloading if needed)."""
    torch_str = "torch"

    if model_type in ("cyto", "nuclei", "cyto2", "cyto3"):
        if model_type == "cyto3":
            basename = f"size_{model_type}.npy"
        else:
            basename = f"size_{model_type}{torch_str}_0.npy"
        return _scportrait_cache_model_path(basename)
    else:
        # nothing to do
        return None


def _download_model(name: str):
    try:
        # Try default cellpose download
        model_path(name)
    except HTTPError:
        print("Cellpose model server appears to be down. Trying scPortrait backup cache...")

        # Try scPortrait backup cache
        _model_path(name)
        _size_model_path(name)
        print("Cellpose model and size file downloaded from scPortrait cache.")
