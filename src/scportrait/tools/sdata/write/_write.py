import warnings
from typing import Any, Literal, TypeAlias

import numpy as np
from dask.array.core import Array as daArray
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity
from xarray import DataArray, DataTree

from scportrait.tools.sdata.write._helper import add_element_sdata

ChunkSize2D: TypeAlias = tuple[int, int]
ChunkSize3D: TypeAlias = tuple[int, int, int]
ObjectType: TypeAlias = Literal["images", "labels", "points", "tables", "shapes"]

from scportrait.pipeline._utils.constants import DEFAULT_CHUNK_SIZE_2D, DEFAULT_CHUNK_SIZE_3D, DEFAULT_SCALE_FACTORS


def image(
    sdata: SpatialData,
    image: np.ndarray | DataArray | DataTree | daArray,
    image_name: str,
    channel_names: list[str] = None,
    scale_factors: list[int] = None,
    chunks: ChunkSize3D | None = None,
    transform: None = None,
    rgb: bool = False,
    overwrite=False,
) -> None:
    """
    Add the image to the spatialdata object

    Args:
        sdata : SpatialData object to which the image will be added.
        image : Image to be written to the spatialdata object. Can be an array (numpy or dask) or an already validated image model.
        image_name: Name of the image to be written to the spatialdata object.
        channel_names: List of channel names for the image. Default is None.
        scale_factors: List of scale factors for the image. Default is [2, 4, 8]. This will load the image at 4 different resolutions to allow for fluid visualization.
        chunks: Chunk size for the image. Default is (1, 1000, 1000).
        transform: Transformation to be applied to the image. Uses the Identity transformation by default.
        rgb: Whether the image is RGB. Default is False.
        overwrite: Whether to overwrite existing data. Default is False.

    Returns:
        None The spatialdata object is updated on disk.
    """
    # check if the image is already a multi-scale image
    if isinstance(image, DataTree):
        # if so only validate the model since this means we are getting the image from a spatialdata object already
        # fix until #https://github.com/scverse/spatialdata/issues/528 is resolved
        Image2DModel().validate(image)
        if scale_factors is not None:
            warnings.warn("Scale factors are ignored when passing a multi-scale image.", stacklevel=2)
    else:
        if scale_factors is None:
            scale_factors = DEFAULT_SCALE_FACTORS

        if chunks is None:
            chunks = DEFAULT_CHUNK_SIZE_3D

        if rgb:
            dimensions = ["y", "x", "c"]
            channel_names = ["r", "g", "b"]
        else:
            dimensions = ["c", "y", "x"]

        if isinstance(image, DataArray):
            # if so first validate the model since this means we are getting the image from a spatialdata object already
            # fix until #https://github.com/scverse/spatialdata/issues/528 is resolved
            Image2DModel().validate(image)

            if channel_names is not None:
                warnings.warn(
                    "Channel names are ignored when passing a single scale image in the DataArray format. Channel names are read directly from the DataArray.",
                    stacklevel=2,
                )

            if chunks is not None:
                warnings.warn(
                    "Chunks are ignored when passing a single scale image in the DataArray format. Chunks are read directly from the DataArray.",
                    stacklevel=2,
                )

            image = Image2DModel.parse(
                image,
                scale_factors=scale_factors,
                rgb=rgb,
            )

        else:
            if channel_names is None:
                channel_names = [f"channel_{i}" for i in range(image.shape[0])]

            # transform to spatialdata image model
            if transform is None:
                transform_original = Identity()
            else:
                transform_original = transform

            if isinstance(image, daArray):
                # rechunk dask array to match the desired chunk size
                image = image.rechunk(chunks)
                image = Image2DModel.parse(
                    image,
                    dims=dimensions,
                    c_coords=channel_names,
                    scale_factors=scale_factors,
                    transformations={"global": transform_original},
                    rgb=rgb,
                )
            else:
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


def labels(
    sdata: SpatialData,
    labels: np.ndarray | DataArray | DataTree | daArray,
    labels_name: str,
    scale_factors: list[int] = None,
    chunks: ChunkSize2D | None = None,
    overwrite=False,
    transform: None = None,
) -> None:
    """Add the labels to the spatialdata object

    Args:
        sdata : SpatialData object to which the labels will be added.
        labels : Labels to be written to the spatialdata object. Can be an array (numpy or dask) or an already validated labels model.
        labels_name: Name of the labels to be written to the spatialdata object.
        scale_factors: List of scale factors for the labels. Default is [2, 4, 8]. This will load the labels at 4 different resolutions to allow for fluid visualization.
        chunks: Chunk size for the labels.Default is (1000, 1000).
        overwrite: Whether to overwrite existing data.
        transform: Transformation to be applied to the labels. Uses the Identity transformation by default.

    Returns:
        None The spatialdata object is updated on disk.
    """

    if isinstance(labels, DataTree):
        # if so only validate the model since this means we are getting the image from a spatialdata object already
        # fix until #https://github.com/scverse/spatialdata/issues/528 is resolved
        Labels2DModel().validate(labels)
        if scale_factors is not None:
            warnings.warn("Scale factors are ignored when passing a multi-scale label layer.", stacklevel=2)
    else:
        if scale_factors is None:
            scale_factors = DEFAULT_SCALE_FACTORS

        if chunks is None:
            chunks = DEFAULT_CHUNK_SIZE_2D

        if isinstance(labels, DataArray):
            # if so first validate the model since this means we are getting the labels from a spatialdata object already
            # fix until #https://github.com/scverse/spatialdata/issues/528 is resolved
            Labels2DModel().validate(labels)

            if chunks is not None:
                warnings.warn(
                    "Chunks are ignored when passing a single scale image in the DataArray format. Chunks are read directly from the DataArray.",
                    stacklevel=2,
                )

            labels = Labels2DModel.parse(
                labels,
                scale_factors=scale_factors,
            )

        else:
            # transform to spatialdata labels model
            if transform is None:
                transform_original = Identity()
            else:
                transform_original = transform

            dimensions = ["y", "x"]

            if isinstance(labels, daArray):
                # rechunk dask array to match the desired chunk size
                labels = labels.rechunk(chunks)
                labels = Labels2DModel.parse(
                    labels,
                    dims=dimensions,
                    scale_factors=scale_factors,
                    transformations={"global": transform_original},
                )
            else:
                labels = Labels2DModel.parse(
                    labels,
                    dims=dimensions,
                    chunks=chunks,
                    scale_factors=scale_factors,
                )

    add_element_sdata(sdata, labels, labels_name, overwrite=overwrite)
