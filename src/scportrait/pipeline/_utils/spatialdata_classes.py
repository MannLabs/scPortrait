"""Extended Labels2D Model with cell ID tracking."""

from functools import singledispatchmethod
from typing import Any

from dask.array import unique as DaskUnique
from datatree import DataTree
from spatialdata.models import C, Labels2DModel, X, Y, Z, get_axes_names
from spatialdata.transformations.transformations import BaseTransformation
from xarray import DataArray
from xarray_schema.components import (
    AttrSchema,
    AttrsSchema,
)

Transform_s = AttrSchema(BaseTransformation, None)


class spLabels2DModel(Labels2DModel):
    """Extended Labels2DModel that maintains cell IDs in attributes."""

    # Add attribute that always contains unique classes in labels image
    attrs = AttrsSchema(
        {"transform": Transform_s},
        {"cell_ids": set[int]},  # More specific type hint for set contents
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the extended Labels2D model."""
        super().__init__(*args, **kwargs)

    @classmethod
    def parse(cls, *args: Any, **kwargs: Any) -> DataArray:
        """Parse data and extract cell IDs.

        Returns:
            DataArray with cell IDs in attributes
        """
        data = super().parse(*args, **kwargs)
        data = cls._get_cell_ids(data)
        return data

    @staticmethod
    def _get_cell_ids(data: DataArray, remove_background: bool = True) -> DataArray:
        """Get unique values from labels image.

        Args:
            data: Input label array
            remove_background: Whether to remove background (0) label

        Returns:
            DataArray with cell IDs added to attributes
        """
        cell_ids = set(DaskUnique(data.data).compute())
        if remove_background:
            cell_ids = cell_ids - {0}  # Remove background class
        data.attrs["cell_ids"] = cell_ids
        return data

    @singledispatchmethod
    def convert(self, data: DataTree | DataArray, classes: set[int] | None = None) -> DataTree | DataArray:
        """Convert data to include cell IDs.

        Args:
            data: Input data to convert
            classes: Optional set of class IDs to use

        Returns:
            Converted data with cell IDs

        Raises:
            ValueError: If data type is not supported
        """
        raise ValueError(f"Unsupported data type: {type(data)}. " "Please use .convert() from Labels2DModel instead.")

    @convert.register(DataArray)
    def _(self, data: DataArray, classes: set[int] | None = None) -> DataArray:
        """Convert DataArray to include cell IDs.

        Args:
            data: Input DataArray
            classes: Optional set of class IDs to use

        Returns:
            DataArray with cell IDs in attributes
        """
        if classes is not None:
            data.attrs["cell_ids"] = classes
        else:
            data = self._get_cell_ids(data)
        return data

    @convert.register(DataTree)
    def _(self, data: DataTree, classes: set[int] | None = None) -> DataTree:
        """Convert DataTree to include cell IDs.

        Args:
            data: Input DataTree
            classes: Optional set of class IDs to use

        Returns:
            DataTree with cell IDs in attributes
        """
        if classes is not None:
            for d in data:
                data[d].attrs["cell_ids"] = classes
        for d in data:
            data[d] = self._get_cell_ids(data[d])
        return data
