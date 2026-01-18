import copy
import warnings

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
