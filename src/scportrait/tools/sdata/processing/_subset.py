import spatialdata


def get_bounding_box_sdata(
    sdata: spatialdata, max_width: int, center_x: int, center_y: int, drop_points: bool = True
) -> spatialdata:
    """apply bounding box to sdata object

    Args:
        sdata: spatialdata object
        max_width: maximum width of the bounding box
        center_x: x coordinate of the center of the bounding box
        center_y: y coordinate of the center of the bounding box

    Returns:
        spatialdata: spatialdata object with bounding box applied
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
