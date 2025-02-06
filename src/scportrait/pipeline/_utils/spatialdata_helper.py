"""Utility functions for handling SpatialData objects and operations."""

from typing import TypeAlias, Union

import numpy as np
import pandas as pd
import psutil
import xarray
from dask.array import unique as DaskUnique
from spatialdata import SpatialData, get_centroids
from spatialdata._core.operations.transform import transform
from spatialdata.models import PointsModel, TableModel
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation

from scportrait.pipeline._utils.segmentation import numba_mask_centroid

# Type aliases
DataElement: TypeAlias = xarray.DataTree | xarray.DataArray
ChunkSize: TypeAlias = tuple[int, ...]
ChunkSizes: TypeAlias = list[tuple[int, ...]]


def check_memory(item: xarray.DataArray) -> bool:
    """Check if item can fit in available memory.

    Args:
        item: Array to check memory requirements for

    Returns:
        Whether item can fit in memory
    """
    array_size = item.nbytes
    available_memory = psutil.virtual_memory().available
    return array_size < available_memory


def generate_region_annotation_lookuptable(sdata: SpatialData) -> dict[str, list[tuple[str, TableModel]]]:
    """Generate a lookup table for the region annotation tables contained in a SpatialData object ordered according to the region they annotate.

    Args:
        sdata: SpatialData object to process

    Returns:
        Mapping of region names to list of (table_name, table) tuples
    """
    table_names = list(sdata.tables.keys())
    region_lookup: dict[str, list[tuple[str, TableModel]]] = {}

    for table_name in table_names:
        table = sdata.tables[table_name]
        region = table.uns["spatialdata_attrs"]["region"]

        if region not in region_lookup:
            region_lookup[region] = [(table_name, table)]
        else:
            region_lookup[region].append((table_name, table))

    return region_lookup


def remap_region_annotation_table(table: TableModel, region_name: str) -> TableModel:
    """Remap region annotation table to new region name.

    Args:
        table: Region annotation table to remap
        region_name: New region name

    Returns:
        Remapped table
    """
    table = table.copy()
    table.obs["region"] = region_name
    table.obs["region"] = table.obs["region"].astype("category")

    if "spatialdata_attrs" in table.uns:
        del table.uns["spatialdata_attrs"]  # remove the spatialdata attributes so that the table can be re-written

    return TableModel.parse(table, region_key="region", region=region_name, instance_key="cell_id")


def get_chunk_size(element: DataElement) -> ChunkSize | ChunkSizes:
    """Get chunk size of image data.

    Args:
        element: Element to get chunk size from

    Returns:
        Chunk size(s) of the image data

    Raises:
        ValueError: If element type is not supported
    """
    if isinstance(element, xarray.DataArray):
        if len(element.shape) == 2:
            y, x = element.chunksizes.values()
            y = y[0] if isinstance(y, tuple | list) or len(y) > 1 else y
            x = x[0] if isinstance(x, tuple | list) or len(x) > 1 else x
            return (int(y), int(x))
        elif len(element.shape) == 3:
            c, y, x = element.chunksizes.values()
            c = c[0] if isinstance(c, tuple | list) or len(c) > 1 else c
            y = y[0] if isinstance(y, tuple | list) or len(y) > 1 else y
            x = x[0] if isinstance(x, tuple | list) or len(x) > 1 else x
            return (int(c), int(y), int(x))

    elif isinstance(element, xarray.DataTree):
        chunk_sizes: ChunkSizes = []
        for scale in element:
            if len(element[scale]["image"].shape) == 2:
                y, x = element[scale].chunksizes.values()
                y = y[0] if len(y) > 1 else y
                x = x[0] if len(x) > 1 else x
                chunk_sizes.append((int(y), int(x)))
            elif len(element[scale]["image"].shape) == 3:
                c, y, x = element[scale].chunksizes.values()
                c = c[0] if len(c) > 1 else c
                y = y[0] if len(y) > 1 else y
                x = x[0] if len(x) > 1 else x
                chunk_sizes.append((int(c), int(y), int(x)))
        return chunk_sizes

    raise ValueError(f"Element must be DataTree or DataArray, found {type(element)}")


def rechunk_image(element: DataElement, chunk_size: ChunkSize) -> DataElement:
    """Rechunk image data to desired chunk size.

    Args:
        element: Element to rechunk
        chunk_size: Desired chunk dimensions

    Returns:
        Rechunked element

    Raises:
        ValueError: If element type is not supported
    """
    if isinstance(element, xarray.DataArray):
        element.data = element.data.rechunk(chunk_size)
        return element
    elif isinstance(element, xarray.DataTree):
        for scale in element:
            element[scale]["image"].data = element[scale]["image"].data.rechunk(chunk_size)
        return element
    raise ValueError(f"Element must be DataTree or DataArray, found {type(element)}")


def make_centers_object(
    centers: np.ndarray, ids: list[int], transformation: BaseTransformation, coordinate_system: str = "global"
) -> PointsModel:
    """Create PointsModel from centers and IDs.

    Args:
        centers: Array of center coordinates
        ids: List of point IDs
        transformation: Transformation to apply
        coordinate_system: Coordinate system name

    Returns:
        Points model containing the centers
    """
    coordinates = pd.DataFrame(centers, columns=["y", "x"], index=ids)
    centroids = PointsModel.parse(coordinates, transformations={coordinate_system: transformation})
    return transform(centroids, to_coordinate_system=coordinate_system)


def calculate_centroids(mask: xarray.DataArray, coordinate_system: str = "global") -> PointsModel:
    """Calculate centroids of labeled regions.

    Args:
        mask: Labeled mask
        coordinate_system: Coordinate system name

    Returns:
        Points model containing centroids
    """
    transform = get_transformation(mask, coordinate_system)

    if check_memory(mask):
        centers, _, _ids = numba_mask_centroid(mask.values)
        return make_centers_object(centers, _ids, transform, coordinate_system)

    print("Array larger than memory, using dask-delayed calculation.")
    return get_centroids(mask, coordinate_system)


def get_unique_cell_ids(data: xarray.DataArray, remove_background: bool = True) -> set[int]:
    """Get unique cell IDs from segmentation mask.

    Args:
        data: Segmentation mask
        remove_background: Whether to remove background (0) label

    Returns:
        Set of unique cell IDs
    """
    cell_ids = set(DaskUnique(data.data).compute())
    if remove_background:
        cell_ids = cell_ids - {0}
    return cell_ids
