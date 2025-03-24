from collections.abc import Iterable

import matplotlib.pyplot as plt
import spatialdata
from spatialdata import to_polygons

PALETTE = [
    "blue",
    "green",
    "red",
    "yellow",
    "purple",
    "orange",
    "pink",
    "cyan",
    "magenta",
    "lime",
    "teal",
    "lavender",
    "brown",
    "beige",
    "maroon",
    "mint",
    "olive",
    "apricot",
    "navy",
    "grey",
    "white",
    "black",
]


def _bounding_box_sdata(sdata: spatialdata, max_width: int, center_x: int, center_y: int) -> spatialdata:
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


def plot_segmentation_mask(
    sdata: spatialdata.SpatialData,
    masks: list[str],
    max_width: int = 1000,
    title: str | None = None,
    selected_channels: int | list[int] | None = None,
    select_region: tuple[int, int] | None = None,
    axs: plt.Axes | None = None,
    font_size: int = 20,
    return_fig: bool = False,
    show_fig: bool = True,
    linewidth: int = 1,
) -> plt.Figure | None:
    """Plot the segmentation mask on the input image.

    Requires an installed spatialdata_plot package.

    Args:
        sdata: SpatialData object containing the input image and segmentation mask.
        masks: List of keys identifying the segmentation masks to plot.
        max_width: Maximum width of the plot. Defaults to 1000.
        title: Title of the plot. Defaults to None.
        select_region: coordinates on which the region to plot should be centered. If not supplied then the middle of the entire image will be used.
        axs: Matplotlib axis object to plot on. Defaults to None.
        return_fig: Whether to return the figure. Defaults to False.
        show_fig: Whether to show the figure. Defaults to True.
    """
    # check for spatialdata_plot
    try:
        import spatialdata_plot
    except ImportError:
        raise ImportError(
            "spatialdata_plot must be installed to use the plotting capabilites. please install with `pip install spatialdata-plot`."
        ) from None

    # get relevant information from spatialdata object
    c, x, y = sdata["input_image"].scale0.image.shape
    channel_names = sdata["input_image"].scale0.c.values

    # get center coordinates
    if select_region is None:
        center_x = x // 2
        center_y = y // 2
    else:
        center_x, center_y = select_region

    # subset spatialdata object if its too large
    if x > max_width or y > max_width:
        sdata = _bounding_box_sdata(sdata, max_width, center_x, center_y)

    # do not plot more than 4 channels
    if selected_channels is not None:
        if not isinstance(selected_channels, Iterable):
            selected_channels = [selected_channels]
        channel_names = [channel_names[i] for i in selected_channels]
        c = len(channel_names)
        palette = [PALETTE[x] for x in selected_channels]
    else:
        if c > 4:
            c = 4
        palette = PALETTE[:c]
        channel_names = list(channel_names[:c])

    # create figure and axis if it does not exist
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    # plot background image
    sdata.pl.render_images("input_image", channel=channel_names, palette=palette).pl.show(
        ax=axs, title="", colorbar=False
    )

    # plot selected segmentation masks
    for mask in masks:
        assert mask in sdata, f"Mask {mask} not found in sdata object."
        if f"{mask}_vectorized" not in sdata:
            sdata[f"{mask}_vectorized"] = to_polygons(sdata[mask])
        sdata.pl.render_shapes(
            f"{mask}_vectorized", fill_alpha=0, outline_alpha=1, outline_width=linewidth, outline_color="white"
        ).pl.show(ax=axs, title=mask)

    # turn off axis
    axs.axis("off")
    axs.set_title(title, fontsize=font_size)

    # return elements
    if return_fig:
        return fig
    elif show_fig:
        plt.show()
        return None
    else:
        return None
