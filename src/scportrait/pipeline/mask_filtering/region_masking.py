import spatialdata as sd
from napari_spatialdata import Interactive
from shapely.geometry import mapping
from rasterio.features import geometry_mask
import rasterio
import dask
from spatialdata.models import Image2DModel
import numpy as np
from scipy import ndimage
from skimage.measure import find_contours
from shapely.geometry import Polygon
from shapely import unary_union
from skimage.segmentation import watershed
from skimage.draw import disk

def mask_image(sdata, image, mask, invert, automatic_masking, threshold, overwrite, masked_image_name)
    """
    Given an image and mask, either masks or crops the image.

    Parameters
    ----------
    sdata : sd.SpatialData
        spatialdata object containing the image and mask.
    image : str
        Name of the image in sdata.images to mask.
    mask : str | shapely.geometry.Polygon
        Mask, either str of the name of the shape in sdata.shapes or a shapely polygon.
    invert : bool
        If True, inverts the mask, such that only pixels within mask remain, while the rest gets cropped.
    automatic_masking : bool
        If True, uses threshold + watershed to automatically create a mask based on shapes. Threshold needs to be adjusted manually.
    threshold : float
            Threshold for pixel intensity values at which to segment image into foreground and background.
    overwrite : bool
        Whether to overwrite the image in sdata.images.
    masked_image_name : None | str
        Name of the masked image in sdata.images if overwrite==True. Defaults to f"{image}_masked".
    Returns
    -------
    sd.SpatialData
        spatialdata object with masked image
    """
    channels, height, width = sdata.images[image].data.shape

    if automatic_masking:
        polygon = _draw_polygons(sdata.images[image].data, threshold)
    elif isinstance(mask, str):
        polygon = sdata.shapes[mask].iloc[0].geometry
    else:
        polygon = mask

    polygon_geom = [mapping(polygon)]

    transform = rasterio.transform.Affine(1, 0, 0, 0, 1, 0) # identity transform

    image_mask = geometry_mask(
        polygon_geom,
        invert=invert,
        out_shape=(height, width),
        transform=transform
    )

    if channels > 1:
        image_mask = dask.array.broadcast_to(image_mask, (channels, height, width))

    masked_image = sdata.images[image].data * image_mask
    images = {}
    images["masked_image"] = Image2DModel.parse(masked_image)

    if overwrite:
        sdata.images[image] = images["masked_image"]
    else:
        if masked_image_name is None:
            masked_image_name = f"{image}_masked"
        sdata.images[masked_image_name] = images["masked_image"]

def _draw_polygons(image, threshold):
    """
    Given an image, detect regions to turn into polygon shapes, which are then used as a mask.

    Parameters
    ----------
    image : np.ndarray
        Image to find regions in.
    threshold : float
        Threshold for pixel intensity values at which to segment image into foreground and background.
    Returns
    -------
    shapely.geometry.Polygon
        Polygon containing the detected regions.
    """
    if image.shape[0] == 1:
        image = image[0]
    binary_image = image > np.percentile(image.flatten(), threshold)

    distance = ndimage.distance_transform_edt(binary_image)
    markers, _ = ndimage.label(distance)

    segmented = watershed(-distance, markers, mask=binary_image)

    contours = find_contours(segmented, level=0.5)

    polygons = [Polygon(contour) for contour in contours if len(contour) > 2]
    polygon = unary_union(polygons) 

    return polygon