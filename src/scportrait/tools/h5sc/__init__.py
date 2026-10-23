"""
h5sc
=======

Functions to work with scPortrait's standardized single-cell data format.
"""

from .operations import (
    add_spatial_coordinates,
    get_cell_id_index,
    get_image_with_cellid,
    subset_cells_region,
    subset_h5sc,
    update_obs_on_disk,
)

__all__ = [
    "update_obs_on_disk",
    "get_image_with_cellid",
    "get_cell_id_index",
    "add_spatial_coordinates",
    "subset_h5sc",
    "subset_cells_region",
]
