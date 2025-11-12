"""
h5sc
=======

Functions to work with scPortrait's standardized single-cell data format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dask.dataframe.core import DataFrame as da_DataFrame
    from spatialdata import SpatialData

import os
import shutil
from pathlib import Path
from warnings import warn

import dask.array as da
import geopandas as gpd
import h5py
import numpy as np
from anndata import AnnData
from shapely.geometry import Point

from scportrait.io.h5sc import read_h5sc
from scportrait.pipeline._utils.constants import (
    DEFAULT_CELL_ID_NAME,
    DEFAULT_IDENTIFIER_FILENAME,
    DEFAULT_NAME_SINGLE_CELL_IMAGES,
    IMAGE_DATACONTAINER_NAME,
)


def update_obs_on_disk(adata: AnnData) -> None:
    """
    Temporarily close the HDF5 handle from a read-only AnnData,
    overwrite .obs on disk, then reopen it and restore the image dataset.

    Args:
        adata: AnnData object whose .obs will replace the existing one.
    """

    # 1. Get the open HDF5 file handle
    file_handle = adata.uns.get("_h5sc_file_handle", None)

    # 2. Close file to release read-only lock
    if file_handle:
        file_handle.close()
        adata.uns["_h5sc_file_handle"] = None

    # 3. Write updated obs
    obs_df = adata.obs.copy()
    obs_df.index = obs_df.index.astype(str)

    with h5py.File(adata.uns[DEFAULT_IDENTIFIER_FILENAME], "r+") as f:
        if "obs" in f:
            del f["obs"]
        grp = f.create_group("obs")
        for col in obs_df.columns:
            grp.create_dataset(col, data=obs_df[col].to_numpy())

    # 4. Reopen file handle and restore image dataset
    f = h5py.File(adata.uns[DEFAULT_IDENTIFIER_FILENAME], "r")
    adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES] = f.get(IMAGE_DATACONTAINER_NAME)
    adata.uns["_h5sc_file_handle"] = f


def get_cell_id_index(adata: AnnData, cell_id: int | list[int]) -> int | list[int]:
    """
    Retrieve the index (row index) of a specific cell id in a H5SC object.

    Args:
        adata: An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
        cell_id: A single cell ID or a list of cell IDs to retrieve indices for.

    Returns:
        The corresponding index or list of indices from `adata.obs.index`.

    """
    lookup = dict(zip(adata.obs[DEFAULT_CELL_ID_NAME], adata.obs.index.astype(int), strict=True))
    if isinstance(cell_id, int):
        assert cell_id in lookup, f"CellID {cell_id} not present in the AnnData object."
        return lookup[cell_id]

    missing = [x for x in cell_id if x not in lookup]
    assert not missing, f"CellIDs not present in the AnnData object: {missing}"
    return [lookup[_id] for _id in cell_id]


def get_image_with_cellid(
    adata: AnnData, cell_id: list[int] | int, select_channel: int | list[int] | None = None
) -> np.ndarray:
    """Get single cell images from the cells with the provided cell IDs. Images are returned in the order of the cell IDs.

    Args:
        adata: An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
        cell_id: The cell ID of the cell to retrieve the image for.
        select_channel: The channel to select from the image. If `None`, all channels are returned.

    Returns:
        The image(s) of the cell with the passed Cell IDs.
    """
    idxs = get_cell_id_index(adata, cell_id)
    if isinstance(idxs, int):
        idxs = [idxs]  # Ensure idxs is always a list

    # get the image container from the AnnData object
    image_container = adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES]

    images = []
    for idx in idxs:
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


def add_spatial_coordinates(
    adata: AnnData,
    centers_object: da_DataFrame,
    cell_id_identifier: str = "scportrait_cell_id",
    update_on_disk: bool = False,
) -> None:
    """Add spatial coordinates to the AnnData object from scPortrait's standardized centers object.
    Args:
        adata: AnnData object to add spatial coordinates to.
        centers_object: Dask DataFrame containing the spatial coordinates with columns "x" and "y" and the scportrait cell id as index.
        cell_id_identifier: The column name in `adata.obs` that contains the cell IDs.
        update_on_disk: boolean value indicating if the updated obs containing the spatial coordinates should be written to disk. This will overwrite the existing obs.

    Returns:
        Updates the obs object of the passed h5sc object.
    """

    assert cell_id_identifier in adata.obs.columns, f"{cell_id_identifier} must be a column in h5sc.obs"
    assert (
        ["x", "y"] == list(centers_object.columns)
    ), "centers_object must be scportrait's standardized centers object containing columns 'x' and 'y' and the scportrait cell id as index, but detected columns are {centers_object.columns}"

    if ("x" in adata.obs.columns) or ("y" in adata.obs.columns):
        adata.obs.drop(columns=["x", "y"], inplace=True, errors="ignore")
        warn(
            "Removed existing 'x' and 'y' columns from adata.obs. If this is not intended, please check the input data.",
            stacklevel=2,
        )

    adata.obs = adata.obs.merge(centers_object.compute(), left_on=cell_id_identifier, right_index=True)

    if update_on_disk:
        update_obs_on_disk(adata)


def subset_cells_region(
    adata: AnnData,
    sdata: SpatialData,
    region_name: str,
    outpath: str | Path = None,
    within_region: bool = True,
    to_disk: bool = True,
    return_anndata: bool = True,
) -> AnnData | None:
    """
    Subset cells in the specified region.

    Args:
        adata: AnnData object containing the cell data.
        sdata: SpatialData object containing the region geometry.
        region_name: Name of the region to subset cells from.
        outpath: Path to save the subsetted AnnData object. If None, the subsetted file is saved in the same directory as the original h5sc file with a prefix "subset_{select_region}".
        within_region: If True, select cells within the region. If False, select cells outside the region.
        to_disk: If True, save the subsetted AnnData object to disk. If False, return the subsetted AnnData object in memory.
        return_anndata: If True, return a memory mapped version of the subsetted AnnData object.

    Returns:
        If `to_disk` is False, returns the subsetted AnnData object. If `to_disk` is True, saves the subsetted AnnData object to disk and returns None.
    """
    if outpath is not None:
        if not isinstance(outpath, (str | Path)):
            raise ValueError("outpath must be a string or Path object.")
        assert to_disk, "outpath is only used if to_disk is True."

    if region_name not in sdata:
        raise ValueError(f"Region '{region_name}' not found in spatialdata object.")

    xs, ys = adata.obs.get(["x", "y"]).values.T
    points = gpd.GeoSeries([Point(xi, yi) for xi, yi in zip(xs, ys, strict=True)])
    is_inside = points.apply(lambda p: sdata[region_name].geometry.contains(p).any()).values

    if not within_region:
        selection = ~is_inside
        key = "outside"
    else:
        selection = is_inside
        key = "within"

    if not to_disk:
        return adata[selection]
    else:
        cell_ids = adata.obs.loc[selection, DEFAULT_CELL_ID_NAME].values

        if outpath is None:
            outpath = adata.uns["h5sc_source_path"].replace("single_cells.h5sc", f"subset_{key}_{region_name}.h5sc")
        subset_h5sc(adata, cell_ids, outpath=outpath)

        if return_anndata:
            return read_h5sc(outpath)
        else:
            return None


def subset_h5sc(adata: AnnData, cell_id: int | list[int], outpath: str | Path) -> None:
    """
    Write a subset of the AnnData object to disk based on the provided cell IDs.

    Args:
        adata: AnnData object containing the single-cell data.
        cell_id: A single cell ID or a list of cell IDs to subset the AnnData
        outpath: Path to save the subsetted AnnData object.

    Returns:
        None. The AnnData object is written to disk at the specified outpath.
    """
    idx = get_cell_id_index(adata, cell_id)

    if isinstance(idx, int):
        idx = [idx]  # Ensure idx is always a list

    obs = adata.obs.iloc[idx, :].copy()
    obs.reset_index(drop=True, inplace=True)
    obs.index = obs.index.astype(str)  # Ensure index is string type for consistency
    var = adata.var.copy()
    uns = {DEFAULT_NAME_SINGLE_CELL_IMAGES: adata.uns[DEFAULT_NAME_SINGLE_CELL_IMAGES]}

    adata_subset = AnnData(obs=obs, var=var, uns=uns)

    if os.path.exists(outpath):
        shutil.rmtree(outpath, ignore_errors=True)
    adata_subset.write_h5ad(outpath)

    # initialize the obsm with the single cell images
    orig = adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES]
    single_cell_data_shape = (len(idx),) + orig.shape[1:]
    with h5py.File(outpath, "a") as hf:
        hf.create_dataset(
            IMAGE_DATACONTAINER_NAME,
            shape=single_cell_data_shape,
            chunks=orig.chunks,
            compression=orig.compression,
            dtype=orig.dtype,
        )
        for key, value in orig.attrs.items():
            hf[IMAGE_DATACONTAINER_NAME].attrs[key] = value

        # transfer the images
        for i, ix in enumerate(idx):
            hf[IMAGE_DATACONTAINER_NAME][i] = orig[ix]
        hf.close()

    print(f"Subsetted AnnData object saved to {outpath}.")
    return None
