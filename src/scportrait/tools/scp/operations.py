import numpy as np

from scportrait.pipeline._utils.constants import DEFAULT_CELL_ID_NAME, DEFAULT_NAME_SINGLE_CELL_IMAGES


def get_scp_images(adata, cell_id: list[int]) -> np.ndarray:
    """Get the image of a single cell by its cell ID.

    Args:
        adata: An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
        cell_id: The cell ID of the cell to retrieve the image for.

    Returns:
        The image of the cell with the given cell ID.
    """
    lookup = dict(zip(adata.obs[DEFAULT_CELL_ID_NAME], adata.obs.index.astype(int), strict=True))
    image_container = adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES]

    for x in cell_id:
        assert x in lookup.keys(), f"CellID {x} is not present in the AnnData object."

    images = []
    for _id in cell_id:
        idx = lookup[_id]
        image = image_container[idx][:]
        images.append(image)

    array = np.array(images)
    if array.shape[0] == 1:  # Check if the first dimension is 1
        return array.squeeze(axis=0)  # Remove the first dimension
    else:
        return array
