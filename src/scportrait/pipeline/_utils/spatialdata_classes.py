from functools import singledispatchmethod

from spatialdata.models import Labels2DModel
from spatialdata.models import C, X, Y, Z, get_axes_names
from spatialdata.transformations.transformations import BaseTransformation, Identity

from dask.array import Array as DaskArray
from datatree import DataTree
from xarray import DataArray

from dask.array import unique as DaskUnique
from xarray_schema.components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    DimsSchema,
)

from typing import List, Tuple, Dict, Any, Set, Union

Transform_s = AttrSchema(BaseTransformation, None)

class spLabels2DModel(Labels2DModel):
    
    #add an additional attribute that always contains the unique classes in a labels image
    attrs = AttrsSchema({"transform": Transform_s}, 
                        {"cell_ids": Set})
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
    
    @classmethod
    def parse(cls, *args, **kwargs):
        # Call the original __init__ method from the Parent class with dynamic arguments
        data = super().parse(*args, **kwargs)
        data = cls._get_cell_ids(data)
        return(data)

    @staticmethod
    def _get_cell_ids(data, remove_background = True):
        """get unique values contained in the labels image and save them as an additional attribute"""
        cell_ids = set(DaskUnique(data.data).compute())
        
        if remove_background:
            cell_ids = cell_ids - {0} #remove background class 
        
        #save cell ids as an additional attribute
        data.attrs["cell_ids"] = cell_ids
        return(data)
    
    @singledispatchmethod
    def convert(self, data: Union[DataTree, DataArray], classes: set = None) -> Union[DataTree, DataArray]:
        """
        """
        raise ValueError(
            f"Unsupported data type: {type(data)}. Please use .convert() from Labels2DModel instead."
        )

    @convert.register(DataArray)
    def _(self, data: DataArray, classes: set) -> DataArray:
        if classes is not None:
            data.attrs["cell_ids"] = classes
        else:
            data = self._get_cell_ids(data)
        return(data)

    @convert.register(DataTree)
    def _(self, data: DataTree, classes: set) -> DataTree:
        if classes is not None:
            for d in data:
                data[d].attrs["cell_ids"] = classes
        for d in data:
                data[d] = self._get_cell_ids(data[d])
        return(data)



        
      

