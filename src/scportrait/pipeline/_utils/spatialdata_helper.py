from spatialdata.models import TableModel, PointsModel
from spatialdata import SpatialData, get_centroids
from spatialdata._core.operations.transform import transform
from spatialdata.transformations.operations import get_transformation
from scportrait.pipeline._utils.segmentation import numba_mask_centroid
import datatree
import xarray

from dask.array import unique as DaskUnique

from typing import List, Tuple, Dict, Any, Set, Union

import pandas as pd
import numpy as np
import psutil

def check_memory(item):
    """
    Check the memory usage of the given if it were completely loaded into memory using .compute().
    """
    array_size = item.nbytes
    available_memory = psutil.virtual_memory().available

    return array_size < available_memory

def generate_region_annotation_lookuptable(sdata: SpatialData) -> Dict:
    """Generate a lookup table for the region annotation tables contained in a SpatialData object ordered according to the region they annotate.
    
    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to generate the lookup table from.
    
    Returns
    -------
    dict
        A dictionary where the keys are the region names and the values are lists of tuples where each tuple contains the table name and the TableModel object.
    """

    table_names = list(sdata.tables.keys())

    region_lookup = {}
    for table_name in table_names:
        table = sdata.tables[table_name]
        region = table.uns["spatialdata_attrs"]["region"]

        if region not in region_lookup:
            region_lookup[region] = [(table_name, table)]
        else:
            region_lookup[region].append((table_name, table))
    
    return region_lookup

def remap_region_annotation_table(table: TableModel, 
                                  region_name: str) -> TableModel:

    """Produce an identical region annotation table that is mapped to a new region name.

    Parameters
    ----------
    table : TableModel
        The region annotation table to remap.
    region_name : str
        The new region name to remap the table to.
    
    Returns
    -------
    TableModel
        The region annotation table mapped to the new region name.
    """
    
    table = table.copy()
    table.obs["region"] = region_name
    table.obs["region"] = table.obs["region"].astype("category")

    if "spatialdata_attrs" in table.uns:
        del table.uns["spatialdata_attrs"] #remove the spatialdata attributes so that the table can be re-written

    table = TableModel.parse(table, region_key="region", region=region_name, instance_key="cell_id")
    return(table)   

def get_chunk_size(element: Union[datatree.DataTree, xarray.DataArray]) ->  Union[Tuple, List[Tuple]]:

    """Get the chunk size of the image data.
    
    Parameters
    ----------
    element : datatree.DataTree or xarray.DataArray
        The element to get the chunk size from. If a DataTree, then the chunk size of the first scale will be returned. If a DataArray, then the chunk size of the image data will be returned.
    
    Returns
    -------
    tuple or [tuple]
        The chunk size of the image data. If a multiscale image was provided then a list of chunk sizes will be returned.
    """

    if isinstance(element, xarray.DataArray):
        if len(element.shape) == 2:
            y, x = element.chunksizes.values()
            if len(y) > 1 or isinstance(y, tuple):
                    y = y[0]
            if len(x) > 1 or isinstance(x, tuple):
                x = x[0]
            y, x = np.array([y, x]).flatten()
            chunksize = (y, x)

        elif len(element.shape) == 3:
            c, y, x = element.chunksizes.values()
            if len(y) > 1 or isinstance(y, tuple):
                y = y[0]
            if len(x) > 1 or isinstance(x, tuple):
                x = x[0]
            if len(c) > 1 or isinstance(c, tuple):
                c = c[0]
            c, y, x = np.array([c, y, x]).flatten()
            chunksize = (c, y, x)
        return chunksize
    
    elif isinstance(element, datatree.DataTree):
        scales = list(element.keys())
        chunk_sizes = []
        
        for scale in scales:

            if len(element[scale]["image"].shape) == 2:
                y, x = element[scale].chunksizes.values()
                if len(y) > 1:
                    y = y[0]
                if len(x) > 1:
                    x = x[0]
                y, x = np.array([y, x]).flatten()
                chunksize = (y, x)
            
            elif len(element[scale]["image"].shape) == 3:
                c, y, x = element[scale].chunksizes.values()
                if len(y) > 1:
                    y = y[0]
                if len(x) > 1:
                    x = x[0]
                if len(c) > 1:
                    c = c[0]
                c, y, x = np.array([c, y, x]).flatten()
                chunksize = (c, y, x)
                chunk_sizes.append(chunksize)
        
        return chunk_sizes
    else:
        raise ValueError(f"element must be a datatree.DataTree or xarray.DataArray  but found {type(element)} instead")

def rechunk_image(element: Union[datatree.DataTree, xarray.DataArray],
                 chunk_size: Tuple) -> Union[datatree.DataTree, xarray.DataArray]:
    """ 
    Rechunk the image data to the desired chunksize. This is useful for ensuring that the data is chunked in a regular manner.

    Parameters
    ----------
    element : datatree.DataTree or xarray.DataArray
        The element to rechunk. If a DataTree, then all the scales will be rechunked. If a DataArray, then only the image data will be rechunked.
    chunk_size : tuple
        The desired chunk size. The chunk size should be a tuple of integers.
    
    Returns
    -------
    datatree.DataTree or xarray.DataArray
        The rechunked element.
    """

    if isinstance(element, xarray.DataArray):
        element["image"].data = element["image"].data.rechunk(chunk_size)
        return element
    
    elif  isinstance(element, datatree.DataTree):
        scales = list(element.keys())
        
        for scale in scales:
            element[scale]["image"].data = element[scale]["image"].data.rechunk(chunk_size)
        
        return element
    else:
        raise ValueError(f"element must be a datatree.DataTree or xarray.DataArray  but found {type(element)} instead")

def make_centers_object(
        centers: np.ndarray,
        ids: List,
        transformation: str,
        coordinate_system="global",
    ):
        """ 
        Create a spatialdata PointsModel object from the provided centers and ids.
        """
        coordinates = pd.DataFrame(centers, columns=["y", "x"], index=ids)
        centroids = PointsModel.parse(
            coordinates, transformations={coordinate_system: transformation}
        )
        centroids = transform(centroids, to_coordinate_system=coordinate_system)

        return centroids

def calculate_centroids(mask, coordinate_system = "global"):

    transform = get_transformation(mask, coordinate_system)
    
    if check_memory(mask):
        centers, _, _ids = numba_mask_centroid(mask.values)
        centroids = make_centers_object(centers, _ids, transform, coordinate_system=coordinate_system)
    else:
        print("Array larger than available memory, using dask-delayed calculation of centers.")
        centroids = get_centroids(mask, coordinate_system)
    
    return(centroids)

def get_unique_cell_ids(data, remove_background = True):
    """Get the unique cell ids from the segmentation mask.
    
    Parameters
    ----------
    data : xarray.DataArray
        The segmentation mask data.
    remove_background : bool, optional
        Whether to remove the background class from the unique cell ids (default: True).
    
    Returns
    -------
    set
        The unique cell ids.
    """

    cell_ids = set(DaskUnique(data.data).compute())
    
    if remove_background:
        cell_ids = cell_ids - {0} #remove background class 
    
    return cell_ids
