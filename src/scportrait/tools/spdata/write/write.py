from typing import Any, Literal, TypeAlias

import numpy as np
from dask.array.core import Array as daArray
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity
from xarray import DataArray, DataTree

from scportrait.tools.spdata.write._helper import add_element_sdata

ChunkSize2D: TypeAlias = tuple[int, int]
ChunkSize3D: TypeAlias = tuple[int, int, int]
ObjectType: TypeAlias = Literal["images", "labels", "points", "tables", "shapes"]


def image(
    sdata: SpatialData,
    image: np.ndarray | DataArray | DataTree | daArray,
    image_name: str,
    channel_names: list[str] = None,
    scale_factors: list[int] = None,
    chunks: ChunkSize3D = (1, 1000, 1000),
    overwrite=False,
    transform: None = None,
    rgb: bool = False,
):
    """
    Add the image to the spatialdata object

    Args:
        image : Image to be written to the spatialdata object. Can be an array (numpy or dask) or an already validated image model.
        image_name: Name of the image to be written to the spatialdata object.
        channel_names: List of channel names for the image. Default is None.
        scale_factors: List of scale factors for the image. Default is [2, 4, 8]. This will load the image at 4 different resolutions to allow for fluid visualization.
        chunks: Chunk size for the image. Default is (1, 1000, 1000).
        overwrite: Whether to overwrite existing data. Default is False.
    """
    # check if the image is already a multi-scale image
    if isinstance(image, DataTree):
        # if so only validate the model since this means we are getting the image from a spatialdata object already
        # fix until #https://github.com/scverse/spatialdata/issues/528 is resolved
        Image2DModel().validate(image)
        if scale_factors is not None:
            Warning("Scale factors are ignored when passing a multi-scale image.")
    else:
        if scale_factors is None:
            scale_factors = [2, 4, 8]

        if rgb:
            dimensions = ["y", "x", "c"]
        else:
            dimensions = ["c", "y", "x"]

        if isinstance(image, DataArray):
            # if so first validate the model since this means we are getting the image from a spatialdata object already
            # fix until #https://github.com/scverse/spatialdata/issues/528 is resolved
            Image2DModel().validate(image)

            if channel_names is not None:
                Warning(
                    "Channel names are ignored when passing a single scale image in the DataArray format. Channel names are read directly from the DataArray."
                )

            image = Image2DModel.parse(
                image,
                scale_factors=scale_factors,
                rgb=rgb,
            )

        else:
            if channel_names is None:
                if rgb:
                    channel_names = ["r", "g", "b"]
                else:
                    channel_names = [f"channel_{i}" for i in range(image.shape[0])]

            # transform to spatialdata image model
            if transform is None:
                transform_original = Identity()
            else:
                transform_original = transform

            if isinstance(image, daArray):
                image = image.rechunk(chunks)

            image = Image2DModel.parse(
                image,
                dims=dimensions,
                chunks=chunks,
                c_coords=channel_names,
                scale_factors=scale_factors,
                transformations={"global": transform_original},
                rgb=rgb,
            )

    add_element_sdata(sdata, image, image_name, overwrite=overwrite)
