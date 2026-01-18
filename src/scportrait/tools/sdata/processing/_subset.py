import warnings

import dask.array as da
import numpy as np
import xarray as xr
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import mapping
from spatialdata import SpatialData


def get_bounding_box_sdata(
    sdata: SpatialData, max_width: int, center_x: int, center_y: int, drop_points: bool = True
) -> SpatialData:
    """apply bounding box to sdata object

    Args:
        sdata: spatialdata object
        max_width: maximum width of the bounding box
        center_x: x coordinate of the center of the bounding box
        center_y: y coordinate of the center of the bounding box

    Returns:
        spatialdata object with bounding box applied
    """
    _sdata = sdata
    # remove points object to improve subsetting
    if drop_points:
        points_keys = list(_sdata.points.keys())
        if len(points_keys) > 0:
            # add check to make sure we aren't deleting a points object that is only in memory
            in_memory_only, _ = _sdata._symmetric_difference_with_zarr_store()
            in_memory_only = [x.split("/")[-1] for x in in_memory_only]

            for x in points_keys:
                if x not in in_memory_only:
                    del _sdata.points[x]
                else:
                    warnings.warn(
                        f"Points object {x} is in memory only and will not be deleted despite the drop_points flag being set to True.",
                        stacklevel=2,
                    )

    width = max_width // 2

    # ensure that the image is large enough
    if center_x - width < 0:
        center_x = width
    if center_y - width < 0:
        center_y = width

    # subset spatialdata object if its too large
    _sdata = _sdata.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[center_x - width, center_y - width],
        max_coordinate=[center_x + width, center_y + width],
        target_coordinate_system="global",
    )

    if drop_points:
        # re-add points object
        __sdata = SpatialData.read(sdata.path, selection=["points"])
        for x in points_keys:
            sdata[x] = __sdata[x]
        del __sdata

    return _sdata


def mask_region(
    sdata: SpatialData,
    image_name: str = "input_image",
    shape_name: str = "select_region",
    mask: bool = True,
    crop: bool = False,
) -> xr.DataArray:
    """Mask and/or crop the input image to the selected region.

    Args:
        sdata: SpatialData object containing the image and shape.
        image_name: Name of the image to be masked/cropped.
        shape_name: Name of the shape to mask/crop the image with.
        mask: Whether to apply the mask to the image. Default is True.
        crop: Whether to crop the image to the outer bounding box of the shape. Default is False.
    Returns:
        masked/cropped image as a DataArray. If crop is False, the image has the same dimensions as the input image, otherwise it has the dimensions of the outer bounding box of the shape.
    """
    assert mask or crop, "Either mask or crop must be True"

    # get image and check for proper scaling
    if image_name not in sdata:
        raise ValueError(f"Image {image_name} not found in sdata")
    image = sdata[image_name]

    if isinstance(image, xr.DataTree):
        image = image.get("scale0").image

    elif isinstance(image, xr.DataArray):
        image = image

    print(image.dtype)

    # get shape and check for single-shape selection
    shape = sdata[shape_name].geometry
    if len(shape) == 1:
        shape = shape[0]
    elif len(shape) > 1:
        raise ValueError("Expected a single shape, but found multiple shapes. Please select only one region.")
    else:
        raise ValueError("No shapes found in the specified region.")

    # initialize empty array
    H, W = image.sizes["y"], image.sizes["x"]
    chunks_yx = (image.data.chunks[image.get_axis_num("y")], image.data.chunks[image.get_axis_num("x")])
    template = da.zeros((H, W), chunks=chunks_yx, dtype=np.uint16)

    def _mask_block(block, block_info=None):
        info = block_info[None]
        (y0, y1), (x0, x1) = info["array-location"][:2]
        h, w = (y1 - y0), (x1 - x0)

        # shift transform to this block’s window
        window_transform = Affine.translation(x0, y0)

        m = rasterize(
            [(geom, 1)],
            out_shape=(h, w),
            transform=window_transform,
            fill=0,
            dtype=image.dtype,
            all_touched=True,  # set True if you want any touched pixel included
        )
        return m.astype(bool)

    geom = mapping(shape)
    mask_dask = da.map_blocks(_mask_block, template, dtype=bool)
    mask_da = xr.DataArray(mask_dask, dims=("y", "x"), coords={"y": image.coords["y"], "x": image.coords["x"]})

    other = np.array(0, dtype=image.dtype)
    if mask:
        if "c" in image.dims:
            m = mask_da.broadcast_like(image.isel(c=0))
            masked = image.where(m, other=other)
        else:
            masked = image.where(mask_da, other=other)
    else:
        masked = image

    if crop:
        minx, miny, maxx, maxy = shape.bounds
        minx, miny, maxx, maxy = int(np.floor(minx)), int(np.floor(miny)), int(np.ceil(maxx)), int(np.ceil(maxy))
        return masked.isel(x=slice(minx, maxx), y=slice(miny, maxy))
    else:
        return masked
