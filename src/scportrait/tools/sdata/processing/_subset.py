import warnings

import dask.array as da
import numpy as np
import xarray as xr
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import mapping
from spatialdata import SpatialData
import spatialdata


def _infer_spatial_extent_xy(sdata: spatialdata) -> tuple[int, int] | None:
    """Infer approximate x/y extent from the first available element."""
    for element in sdata.images.values():
        if hasattr(element, "scale0"):
            shape = element.scale0.image.shape
        elif hasattr(element, "data"):
            shape = element.data.shape
        else:
            continue
        if len(shape) >= 2:
            return int(shape[-1]), int(shape[-2])

    for element in sdata.labels.values():
        if hasattr(element, "scale0"):
            shape = element.scale0.image.shape
        elif hasattr(element, "data"):
            shape = element.data.shape
        else:
            continue
        if len(shape) >= 2:
            return int(shape[-1]), int(shape[-2])

    for element in sdata.shapes.values():
        if hasattr(element, "total_bounds"):
            min_x, min_y, max_x, max_y = element.total_bounds
            return max(1, int(round(max_x - min_x))), max(1, int(round(max_y - min_y)))

    return None


def get_bounding_box_sdata(
    sdata: SpatialData, max_width: int, center_x: int, center_y: int, drop_points: bool = True
) -> SpatialData:
    """apply bounding box to sdata object

    Args:
        sdata: SpatialData object
        max_width: maximum width of the bounding box
        center_x: x coordinate of the center of the bounding box
        center_y: y coordinate of the center of the bounding box

    Returns:
        SpatialData object with bounding box applied
    """
    _sdata = sdata
    points_keys = list(_sdata.points.keys()) if drop_points else []
    points_backup: dict[str, object] = {}
    if drop_points and points_keys:
        for key in points_keys:
            points_backup[key] = _sdata.points[key]
            del _sdata.points[key]

    width = max_width // 2

    # ensure that the image is large enough
    if center_x - width < 0:
        center_x = width
    if center_y - width < 0:
        center_y = width

    extent_xy = _infer_spatial_extent_xy(_sdata)
    if extent_xy is not None:
        extent_x, extent_y = extent_xy
        max_center_x = max(width, extent_x - width)
        max_center_y = max(width, extent_y - width)
        center_x = min(max(center_x, width), max_center_x)
        center_y = min(max(center_y, width), max_center_y)

    try:
        # subset spatialdata object if its too large
        _sdata = _sdata.query.bounding_box(
            axes=["x", "y"],
            min_coordinate=[center_x - width, center_y - width],
            max_coordinate=[center_x + width, center_y + width],
            target_coordinate_system="global",
        )
    finally:
        # Re-attach points in the original object to avoid side effects.
        if drop_points and points_backup:
            for key, element in points_backup.items():
                sdata[key] = element

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
