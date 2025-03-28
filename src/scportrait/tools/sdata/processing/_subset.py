import spatialdata


def get_bounding_box_sdata(sdata: spatialdata, max_width: int, center_x: int, center_y: int) -> spatialdata:
    """apply bounding box to sdata object

    Args:
        sdata: spatialdata object
        max_width: maximum width of the bounding box
        center_x: x coordinate of the center of the bounding box
        center_y: y coordinate of the center of the bounding box

    Returns:
        spatialdata: spatialdata object with bounding box applied
    """

    # remove points object to improve subsetting
    points_keys = list(sdata.points.keys())
    if len(points_keys) > 0:
        for x in points_keys:
            del sdata.points[x]

    width = max_width // 2

    # ensure that the image is large enough
    if center_x - width < 0:
        center_x = width
    if center_y - width < 0:
        center_y = width

    # subset spatialdata object if its too large
    sdata = sdata.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[center_x - width, center_y - width],
        max_coordinate=[center_x + width, center_y + width],
        target_coordinate_system="global",
    )
    return sdata
