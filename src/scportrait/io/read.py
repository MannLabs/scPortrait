import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def read_ome_zarr(path, magnification="0", array=None):
    """Reads an OME-Zarr file from a given path.

    Parameters
    ----------
    path : str
        Path to the OME-Zarr file.
    magnification : str
        Magnification level to be read.
    array : None or np.array
        If None, the image data is read into memory and returned. If not None, the image data is read into the supplied numpy array

    Returns
    -------
    np.array or None
        If array is None, the image data is read into memory and returned. Otherwise the provided variable is updated.
    """

    # read the image data
    loc = parse_url(path, mode="r")
    zarr_reader = Reader(loc).zarr

    # read entire data into memory
    if array is None:
        image = np.array(zarr_reader.load(magnification).compute())
        return image
    else:
        array = np.array(zarr_reader.load(magnification).compute())
        return None
