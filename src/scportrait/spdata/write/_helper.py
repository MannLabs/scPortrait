from typing import Literal, TypeAlias

from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from xarray_schema.base import SchemaError

SPATIALDATA_MODELS = [Labels2DModel, Labels3DModel, Image2DModel, Image3DModel, PointsModel, TableModel, ShapesModel]

ObjectType: TypeAlias = Literal["images", "labels", "points", "tables", "shapes"]
spObject: TypeAlias = (
    Labels2DModel | Labels3DModel | Image2DModel | Image3DModel | PointsModel | TableModel | ShapesModel
)


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


def add_element_sdata(sdata: SpatialData, element: spObject, element_name: str, overwrite: bool = True):
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
    valid_model = False
    for model in SPATIALDATA_MODELS:
        try:
            model().validate(element)
        except SchemaError:
            continue
        except ValueError:
            continue
        except AttributeError:
            continue
        except KeyError:
            continue
        valid_model = True
        break  # Exit loop early once validation is performed

    if not valid_model:
        raise ValueError(f"Element does not validate with any of the models: {SPATIALDATA_MODELS}")

    # Add the element to the SpatialData object
    sdata[element_name] = element
    sdata.write_element(element_name)
