import os
from typing import Any, Literal, TypeAlias

import numpy as np
from spatialdata import SpatialData, read_zarr
from spatialdata.models import get_model
from xarray import DataArray, DataTree

ObjectType: TypeAlias = Literal["images", "labels", "points", "tables", "shapes"]


def _get_image(elem: DataArray | DataTree) -> DataArray:
    """Get the image from the element.

    Args:
        elem: Element to get the image from

    Returns:
        Image of the element or None if the element is not an image
    """
    if isinstance(elem, DataArray):
        image = elem
    elif isinstance(elem, DataTree):
        image = elem.scale0.image
    else:
        image = None
    return image


def _get_shape(elem: DataArray | DataTree) -> tuple[int, int, int]:
    """Get the shape of the element.

    Args:
        elem: Element to get the shape of

    Returns:
        Tuple of the shape of the element with c, x, y dimensions. If the element is 2D, the first dimension is set to np.nan.
    """
    if isinstance(elem, DataArray):
        shape = elem.shape
    elif isinstance(elem, DataTree):
        shape = elem.scale0.image.shape
    else:
        raise ValueError(f"Element type {type(elem)} not supported.")

    if len(shape) == 2:
        shape = (np.nan, shape[0], shape[1])
    elif len(shape) == 3:
        shape = (shape[0], shape[1], shape[2])
    return shape


def _make_key_lookup(sdata: SpatialData) -> dict:
    """Make a lookup dictionary for the keys in the SpatialData object.

    Args:
        sdata: SpatialData object

    Returns:
        Dictionary of the keys in the SpatialData object
    """
    dict_lookup: dict[str, list[str]] = {}
    for elem in sdata.elements_paths_in_memory():
        key, name = elem.split("/")
        if key not in dict_lookup:
            dict_lookup[key] = []
        dict_lookup[key].append(name)
    return dict_lookup


def _force_delete_object(sdata: SpatialData, name: str) -> None:
    """Force delete an object from the SpatialData object and directory.

    Args:
        sdata: SpatialData object
        name: Name of object to delete
        type: Type of object ("images", "labels", "points", "tables")

    Returns:
        None the SpatialData object is updated on file
    """
    if name in sdata:
        del sdata[name]

    in_memory_only, _ = sdata._symmetric_difference_with_zarr_store()
    if name not in in_memory_only:
        sdata.delete_element_from_disk(name)


def add_element_sdata(sdata: SpatialData, element: Any, element_name: str, overwrite: bool = True):
    """Add an element to the SpatialData object.

    Args:
        sdata: SpatialData object
        element: Element to add
        element_name: Name of the element to be added
        overwrite: Whether to overwrite the element if it already exists

    Returns:
        None: the SpatialData object is updated on file
    """

    if element_name in sdata:
        if not overwrite:
            raise ValueError(
                f"Object with name '{element_name}' already exists in SpatialData." f"Set overwrite=True to replace it."
            )
        _force_delete_object(sdata, element_name)

    # the element needs to validate with exactly one of the models
    get_model(element)

    # Add the element to the SpatialData object
    sdata[element_name] = element
    sdata.write_element(element_name)


def rename_image_element(sdata: SpatialData, image_element: str, new_element_name: str) -> SpatialData:
    """Rename an image element in the sdata object.

    Args:
        sdata: SpatialData object
        image_element: Image element to rename
        new_element_name: New name for the image element

    Returns:
        sdata: Updated SpatialData object with the renamed image element.
    """
    assert image_element in sdata, f"Image element '{image_element}' not found in SpatialData."

    path_sdata = sdata.path
    path_elem = path_sdata / "images" / image_element
    new_path_elem = sdata.path / "images" / new_element_name

    # Check if the new name already exists
    if new_path_elem.exists():
        raise ValueError(f"Image element with name '{new_element_name}' already exists.")

    short_path_elem = f"images/{image_element}"
    short_path_new_elem = f"images/{new_element_name}"

    assert (
        str(short_path_elem) in sdata.elements_paths_on_disk()
    ), f"Element {image_element} needs to be on disk to rename it."

    # rename metadata
    zattrs_path = sdata.path / "images" / image_element / ".zattrs"
    with open(zattrs_path) as file:
        content = file.read()
        content = content.replace(f"/{short_path_elem}", f"/{short_path_new_elem}")

    with open(zattrs_path, "w") as file:
        file.write(content)

    # rename image files
    os.rename(path_elem, new_path_elem)

    # read and return update spatialdata object
    return read_zarr(path_sdata)
