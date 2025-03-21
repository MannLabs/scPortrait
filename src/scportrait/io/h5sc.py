from pathlib import Path
from typing import Literal

import h5py
from anndata import AnnData
from anndata._io.h5ad import _clean_uns, _read_raw, read_dataframe, read_elem

from scportrait.pipeline._utils.constants import DEFAULT_NAME_SINGLE_CELL_IMAGES, IMAGE_DATACONTAINER_NAME


def read_h5sc(filename: str | Path) -> AnnData:
    """Read scportrait's single-cell image dataset format.

    Args:
        filename: Path to the file to read.
        mode: Mode in which to open the file.

    Returns:
        An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
    """
    mode = "r"  # hard code as there is no benefit from letting the user set this manually as writing back and changing values is currently not supported
    d = {}

    # connect to h5py file
    f = h5py.File(filename, mode)

    attributes = ["varm", "obsp", "varp", "uns", "layers"]
    df_attributes = ["obs", "var"]

    if "encoding-type" in f.attrs:
        attributes.extend(df_attributes)
    else:
        for k in df_attributes:
            if k in f:  # Backwards compat
                d[k] = read_dataframe(f[k])

    d.update({k: read_elem(f[k]) for k in attributes if k in f})

    d["raw"] = _read_raw(f, attrs={"var", "varm"})
    adata = AnnData(**d)

    # Backwards compat to <0.7
    if isinstance(f["obs"], h5py.Dataset):
        _clean_uns(adata)

    adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES] = f.get(IMAGE_DATACONTAINER_NAME)
    return adata
