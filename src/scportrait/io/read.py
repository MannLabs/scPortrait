import numpy as np
from numpy.typing import NDArray
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def read_ome_zarr(path: str, magnification: str = "0", array: NDArray | None = None) -> NDArray | None:
    """Reads an OME-Zarr file from a given path.

    Args:
        path: Path to the OME-Zarr file
        magnification: Magnification level to be read
        array: Optional numpy array to store the image data. If None, returns a new array

    Returns:
        The image data as a numpy array if array is None, otherwise None after updating
        the provided array

    Example:
        >>> image = read_ome_zarr("path/to/file.zarr")
        >>> # Or with existing array:
        >>> existing_array = np.zeros((100, 100))
        >>> read_ome_zarr("path/to/file.zarr", array=existing_array)
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
