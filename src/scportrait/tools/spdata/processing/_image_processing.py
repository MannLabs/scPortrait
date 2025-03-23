import dask.array as da
import numpy as np
import xarray
from spatialdata.transformations import get_transformation

from scportrait.tools.spdata.write import image as write_image


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
    sdata,
    image_name: str,
    lower_percentile: int = 1,
    upper_percentile: int = 99,
    scale_factors: list[int] = None,
    rescaled_image_name: str | None = None,
    overwrite: bool = True,
):
    if image_name not in sdata:
        raise ValueError(f"Image {image_name} not found in sdata")

    image = sdata[image_name]

    if isinstance(image, xarray.DataTree):
        image = image.get("scale0").image
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
    data = image.data
    data_rescaled = da.map_blocks(
        lambda x: _rescale_image(x, lower_quantiles, IPRs, dtype=image_dtype), data, dtype=image_dtype
    )

    # get local transform
    local_transform = get_transformation(image)

    # write rescaled image back to sdata object
    write_image(
        sdata, image=data_rescaled, image_name=f"{image_name}_rescaled", transform=local_transform, overwrite=overwrite
    )
