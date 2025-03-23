from typing import TypeAlias

import dask.array as da
import numpy as np
import xarray
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from scportrait.tools.spdata.write._helper import _force_delete_object
from scportrait.tools.spdata.write._write import image as write_image

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
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
    rescaled_image_name: str | None = None,
    scale_factors: list[int] | None = None,
    overwrite: bool = True,
    chunks: ChunkSize3D | None = None,
) -> None:
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

    image = sdata[image_name]
    if rescaled_image_name is None:
        rescaled_image_name = f"{image_name}_rescaled"

    if isinstance(image, xarray.DataTree):
        image = image.get("scale0").image
        # placeholder needs to be implemented
        # if a multiscale image is provided the scale_factors should be kept the same
        # scale_factors = .... code here ...

    elif isinstance(image, xarray.DataArray):
        image = image

    # get dtype
    image_dtype = image.data.dtype.type

    # convert percentiles to quantiles
    assert lower_percentile < upper_percentile, "Lower percentile must be less than upper percentile"
    assert lower_percentile >= 0 and upper_percentile <= 100, "Percentiles must be between 0 and 100"

    lower_quantile = lower_percentile / 100
    upper_quantile = upper_percentile / 100

    # calculate quantiles for specific scale
    lower_quantiles = image.quantile(lower_quantile, dim=["x", "y"]).compute().values
    upper_quantiles = image.quantile(upper_quantile, dim=["x", "y"]).compute().values
    IPRs = upper_quantiles - lower_quantiles

    # apply rescaling to image
    data_rescaled = da.map_blocks(
        lambda x: _rescale_image(x, lower_quantiles, IPRs, dtype=image_dtype), image.data, dtype=image_dtype
    )

    # get local transform
    local_transform = get_transformation(image)

    # get channel names
    channel_names = image.c.values.tolist()

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

        _force_delete_object(sdata, image_name)

        # currently we need to write the rescaled image again because spatialdata
        # does not yet support renaming of objects
        # once https://github.com/scverse/spatialdata/issues/906 has been implemented this
        # can be changes to use that syntax
        image = sdata[rescaled_image_name]
        if isinstance(image, xarray.DataTree):
            image = image.get("scale0").image
        write_image(
            sdata,
            image=image,
            image_name=image_name,
            channel_names=channel_names,
            transform=local_transform,
            overwrite=overwrite,
            scale_factors=scale_factors,
            chunks=chunks,
        )
        _force_delete_object(sdata, rescaled_image_name)

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
