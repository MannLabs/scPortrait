"""
h5sc
=======

Functions to work with scPortrait's standardized single-cell data format.
"""

import numpy as np

from scportrait.pipeline._utils.constants import DEFAULT_CELL_ID_NAME, DEFAULT_NAME_SINGLE_CELL_IMAGES


def get_image_with_cellid(adata, cell_id: list[int] | int, select_channel: int | list[int] | None = None) -> np.ndarray:
    """Get single cell images from the cells with the provided cell IDs. Images are returned in the order of the cell IDs.

    Args:
        adata: An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
        cell_id: The cell ID of the cell to retrieve the image for.
        select_channel: The channel to select from the image. If `None`, all channels are returned.

    Returns:
        The image(s) of the cell with the passed Cell IDs.
    """
    lookup = dict(zip(adata.obs[DEFAULT_CELL_ID_NAME], adata.obs.index.astype(int), strict=True))
    image_container = adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES]

    if isinstance(cell_id, int):
        cell_id = [cell_id]

    for x in cell_id:
        assert x in lookup.keys(), f"CellID {x} is not present in the AnnData object."

    images = []
    for _id in cell_id:
        idx = lookup[_id]
        if select_channel is None:
            image = image_container[idx][:]
        else:
            image = image_container[idx][select_channel]
        images.append(image)

    array = np.array(images)
    if array.shape[0] == 1:  # Check if the first dimension is 1
        return array.squeeze(axis=0)  # Remove the first dimension
    else:
        return array
