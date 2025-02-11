from typing import Any, Literal, TypeAlias

from spatialdata import SpatialData
from spatialdata.models import get_model

ObjectType: TypeAlias = Literal["images", "labels", "points", "tables", "shapes"]


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
