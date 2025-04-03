from typing import TypeAlias

import dask.array as da
import numpy as np
import xarray
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from scportrait.tools.sdata.write._helper import _force_delete_object, rename_image_elem
from scportrait.tools.sdata.write._write import image as write_image
from functools import partial
import spatialdata

ChunkSize2D: TypeAlias = tuple[int, int]
ChunkSize3D: TypeAlias = tuple[int, int, int]


def _rescale_image(
    img: da.Array, lower_quantiles: np.ndarray, IPRs: np.ndarray, dtype: np.dtype = np.uint16
) -> da.Array:
    """Helper function to apply rescaling to image

    Args:
        img: Image to be rescaled
        lower_quantiles: Lower quantiles for each scale
        IPRs: Inter-percentile ranges for each scale
        dtype: Data type of image

    Returns:
        Rescaled image with the same dtype as the input image.
    """
    return (((img - lower_quantiles[:, None, None]) / IPRs[:, None, None]) * np.iinfo(dtype).max).astype(dtype)


def percentile_normalize_image(
    sdata: SpatialData,
    image_name: str,
    lower_percentile: float | list[float] = 0.1,
    upper_percentile: float | list[float] = 99.9,
    rescaled_image_name: str | None = None,
    scale_factors: list[int] | None = None,
    overwrite: bool = True,
    chunks: ChunkSize3D | None = None,
) -> SpatialData:
    """Percentile Normalize an image in a spatialdata object.

    Args:
        sdata: SpatialData object containing the image to be percentile normalized.
        image_name: Name of the image to be percentile normalized.
        lower_percentile: Lower percentile for normalization. Default is 0.1.
        upper_percentile: Upper percentile for normalization. Default is 99.9.
        rescaled_image_name: Name of the rescaled image. Default is None.
        overwrite: Whether to overwrite existing data. Default is True.
        chunks: Chunk size for the image. Default is None.

    Returns:
        the SpatialData object is updated on file with a new element containing the rescaled image
        with the name "rescaled_image_name". If this is not provided, the name will be "{image_name}_rescaled".
        If the name is identical to the input image the input image will be overwritten.

    """
    if image_name not in sdata:
        raise ValueError(f"Image {image_name} not found in sdata")

    #define default values for generated rescaled image
    if rescaled_image_name is None:
        rescaled_image_name = f"{image_name}_rescaled"

    # get image and check for proper scaling
    image = sdata[image_name]

    if isinstance(image, xarray.DataTree):
        image = image.get("scale0").image
        # placeholder needs to be implemented
        # if a multiscale image is provided the scale_factors should be kept the same
        # scale_factors = .... code here ...

    elif isinstance(image, xarray.DataArray):
        image = image

    # get dtype
    image_dtype = image.data.dtype.type
        # get channel names

    # get channel names
    channel_names = image.c.values.tolist()

    # convert percentiles to quantiles
    if isinstance(lower_percentile, list) and isinstance(upper_percentile, list):
        assert len(lower_percentile) == len(upper_percentile), "Lower and upper percentiles must have the same length"
        for lower, upper in zip(lower_percentile, upper_percentile, strict=True):
            assert lower < upper, "Lower percentile must be less than upper percentile"
            assert lower >= 0 and upper <= 100, "Percentiles must be between 0 and 100"

        # convert to quantiles
        lower_quantile = np.array(lower_percentile) / 100
        upper_quantile = np.array(upper_percentile) / 100

        # calculate quantiles for a specific scale for each channel
        assert len(image.c) == len(lower_quantile), "Number of channels must match number of quantiles"

        lower_quantiles = np.zeros((len(image.c),), dtype=image_dtype)
        upper_quantiles = np.zeros((len(image.c),), dtype=image_dtype)
        IPRs = np.zeros((len(image.c),), dtype=image_dtype)

        for i, x in enumerate(zip(image.c, lower_quantile, upper_quantile, strict = True)):
            c, lower, upper = x
            channel = image.sel(c=c)
            lower_quantiles[i] = channel.quantile(lower, dim=["x", "y"]).compute().values
            upper_quantiles[i] = channel.quantile(upper, dim=["x", "y"]).compute().values
            IPRs[i] = upper_quantiles[i] - lower_quantiles[i]

    elif isinstance(lower_percentile, float|int) and isinstance(upper_percentile, float|int):
        assert lower_percentile < upper_percentile, "Lower percentile must be less than upper percentile"
        assert lower_percentile >= 0 and upper_percentile <= 100, "Percentiles must be between 0 and 100"

        lower_quantile = lower_percentile / 100
        upper_quantile = upper_percentile / 100

        # calculate quantiles for specific scale
        lower_quantiles = image.quantile(lower_quantile, dim=["x", "y"]).compute().values
        upper_quantiles = image.quantile(upper_quantile, dim=["x", "y"]).compute().values
        IPRs = upper_quantiles - lower_quantiles
    else:
        raise ValueError(f"Lower and upper percentiles are an unsupported dtype {type(lower_percentile)} and {type(upper_percentile)}")

    # apply rescaling to image
    rescale_fn = partial(_rescale_image, lower_quantiles=lower_quantiles, IPRs=IPRs, dtype=image_dtype)

    data_rescaled = da.map_blocks(
        rescale_fn,
        image.data,
        dtype=image_dtype
    )

    # get local transform
    local_transform = get_transformation(image)

    if rescaled_image_name == image_name:
        # to  overwrite the image in place we currently need to use a workaround that
        # involves writing the object multiple times as we can first delete the old object
        # after the new one has been created (since it reads lazily from this object)

        rescaled_image_name = f"{image_name}_rescaled"
        write_image(
            sdata,
            image=data_rescaled,
            image_name=rescaled_image_name,
            channel_names=channel_names,
            transform=local_transform,
            overwrite=overwrite,
            scale_factors=scale_factors,
            chunks=chunks,
        )
        # to ensure that the object is updated and is backed by the new written file otherwise the original input image can not be deleted
        sdata= spatialdata.read_zarr(sdata.path)
        _force_delete_object(sdata, image_name)
        sdata = rename_image_elem(sdata, image_element = rescaled_image_name, new_element_name = image_name) #rename directory on disk instead of rewriting to improve performance

    else:
        # write rescaled image back to sdata object
        write_image(
            sdata,
            image=data_rescaled,
            image_name=rescaled_image_name,
            channel_names=channel_names,
            transform=local_transform,
            overwrite=overwrite,
            scale_factors=scale_factors,
            chunks=chunks,
        )
        sdata = spatialdata.read_zarr(sdata.path) # to ensure that the object is updated and is backed by the new written file

    return(sdata)
