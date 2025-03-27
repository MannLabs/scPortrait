from collections.abc import Iterable

import matplotlib.pyplot as plt
import spatialdata
import xarray
from matplotlib.axes import Axes

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


def _check_for_spatialdata_plot():
    # check for spatialdata_plot
    try:
        import spatialdata_plot
    except ImportError:
        raise ImportError(
            "Extended plotting capabilities required. Please install with `pip install 'scportrait[plotting]'`."
        ) from None


def plot_image(
    sdata: spatialdata.SpatialData,
    image_name: str,
    channel_names: list[str] | list[int] | None = None,
    palette: list[str] | None = None,
    title: str | None = None,
    title_fontsize: int = 20,
    dpi: int | None = None,
    ax: Axes = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> plt.Figure | None:
    """Plot the image with the given name from the spatialdata object.

    Args:
        sdata: SpatialData object containing the image.
        image_name: Name of the image to plot.
        channel_names: List of channel names to plot. If None then all channels will be plotted. Defaults to None.
        palette: List of colors to use for the channels. If None then the default palette will be used. Defaults to None.
        title: Title of the plot. Defaults to None.
        dpi: Dots per inch of the plot. Defaults to None.
        ax: Matplotlib axis object to plot on. Defaults to None.
        return_fig: Whether to return the figure. Defaults to False.
        show_fig: Whether to show the figure. Defaults to True.

    Returns:
        Matplotlib figure object if return_fig is True.
    """
    _check_for_spatialdata_plot()

    if ax is not None:
        if dpi is not None:
            raise Warning("DPI is ignored when an axis is provided.")
    else:
        if dpi is None:
            dpi = 300
        shape = sdata[image_name].scale0.image.shape
        if len(shape) == 3:
            axs_length = shape[1] / dpi
            axs_width = shape[2] / dpi
        elif len(shape) == 2:
            axs_length = shape[0] / dpi
            axs_width = shape[1] / dpi

        fig = plt.figure(figsize=(axs_width, axs_length), frameon=False, dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    if palette is None:
        if channel_names is None:
            palette = None
        else:
            palette = PALETTE[: len(channel_names)]

    # turn off axis
    ax.set_axis_off()

    sdata.pl.render_images(image_name, channel=channel_names, palette=palette).pl.show(ax=ax, colorbar=False)
    ax.set_title(title, fontsize=title_fontsize)

    if return_fig:
        return fig
    elif show_fig:
        plt.show()
        return None
    else:
        return None


def plot_segmentation_mask(
    sdata: spatialdata.SpatialData,
    masks: list[str],
    background_image: str | None = "input_image",
    selected_channels: int | list[int] | None = None,
    max_channels_to_plot: int = 4,
    title: str | None = None,
    title_fontsize: int = 20,
    line_width: int = 1,
    line_color: str = "white",
    line_alpha: float = 1,
    fill_alpha: float = 0,
    fill_color: str | None = None,
    dpi: int | None = None,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> plt.Figure | None:
    """Visualize segmentation masks over selected background image.
    Will transform the provided labels layers to polygons and plot them over the background image.
    Requires an installed spatialdata_plot package.

    Args:
        sdata: SpatialData object containing the input image and segmentation mask.
        masks: List of keys identifying the segmentation masks to plot.
        background_image: Key identifying the background image to plot the segmentation masks on. Defaults to "input_image". If set to None then only the segmentation masks will be plotted as outlines.
        selected_channels: List of indices of the channels in the background image to plot. If None then the first `max_channels_to_plot` channels will be plotted. Defaults to None.
        max_width: Maximum width of the region to plot. Defaults to 1000.
        title: Title of the plot. Defaults to None.
        select_region: coordinates on which the region to plot should be centered. If not supplied then the middle of the entire image will be used.
        axs: Matplotlib axis object to plot on. Defaults to None.
        return_fig: Whether to return the figure. Defaults to False.
        show_fig: Whether to show the figure. Defaults to True.
    """
    # check for spatialdata_plot
    _check_for_spatialdata_plot()

    # get relevant information from spatialdata object
    mask = sdata[masks[0]]
    if isinstance(mask, xarray.DataTree):
        shape = mask.scale0.image.shape
    else:
        shape = mask.data.shape
    if len(shape) == 3:
        _, x, y = shape
    elif len(shape) == 2:
        x, y = shape

    # create figure and axis if it does not exist
    if ax is None:
        if dpi is None:
            dpi = 300
        axs_length = x / dpi
        axs_width = y / dpi

        fig = plt.figure(figsize=(axs_width, axs_length), frameon=False, dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    # plot background image if desired
    if background_image is not None:
        # get channel names
        channel_names = sdata[background_image].scale0.c.values
        c = len(channel_names)

        # do not plot more than `max_channels_to_plot` overlapping channels
        if selected_channels is not None:
            if not isinstance(selected_channels, Iterable):
                selected_channels = [selected_channels]
            channel_names = [channel_names[i] for i in selected_channels]
            c = len(channel_names)
            palette = [PALETTE[x] for x in selected_channels]
        else:
            if c > max_channels_to_plot:
                c = 4
            palette = PALETTE[:c]
            channel_names = list(channel_names[:c])

        sdata.pl.render_images(background_image, channel=channel_names, palette=palette).pl.show(ax=ax, colorbar=False)

    # plot selected segmentation masks
    for mask in masks:
        assert mask in sdata, f"Mask {mask} not found in sdata object."
        if f"{mask}_vectorized" not in sdata:
            sdata[f"{mask}_vectorized"] = spatialdata.to_polygons(sdata[mask])
        sdata.pl.render_shapes(
            f"{mask}_vectorized",
            color=fill_color,
            fill_alpha=fill_alpha,
            outline_alpha=line_alpha,
            outline_width=line_width,
            outline_color=line_color,
        ).pl.show(ax=ax)

    # configure axes
    ax.axis("off")
    ax.set_title(title, fontsize=title_fontsize)

    # return elements
    if return_fig:
        return fig
    elif show_fig:
        plt.show()
        return None
    else:
        return None


def plot_labels(
    sdata: spatialdata.SpatialData,
    label_layer: str,
    title: str | None = None,
    font_size: int = 20,
    color="grey",
    fill_alpha: float = 1,
    cmap: str = None,
    vectorized: bool = False,
    dpi: int | None = None,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    show_fig: bool = True,
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
    _check_for_spatialdata_plot()

    # get relevant information from spatialdata object
    element = sdata[label_layer]
    if isinstance(element, xarray.DataTree):
        shape = element.scale0.image.shape
    else:
        shape = element.data.shape

    if len(shape) == 3:
        _, x, y = shape
    elif len(shape) == 2:
        x, y = shape

    # create figure and axis if it does not exist
    if ax is None:
        if dpi is None:
            dpi = 300
        axs_length = x / dpi
        axs_width = y / dpi

        fig = plt.figure(figsize=(axs_width, axs_length), frameon=False, dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    # plot selected segmentation masks
    if vectorized:
        if f"{label_layer}_vectorized" not in sdata:
            sdata[f"{label_layer}_vectorized"] = spatialdata.to_polygons(sdata[label_layer])
        found_annotation = None
        for table in sdata.tables:
            annotated_regions = sdata.get_annotated_regions(sdata[table])
            for region in annotated_regions:
                if region == label_layer:
                    found_annotation = region
                    annotating_table = sdata[table].copy()
                    annotating_table.uns["spatialdata_attrs"]["region"] = f"{label_layer}_vectorized"
                    annotating_table.obs["region"] = f"{label_layer}_vectorized"
                    annotating_table.obs["region"] = annotating_table.obs["region"].astype("category")

                    # check for annotating column
                    if color in annotating_table.obs:
                        annotating_table.obs[color] = annotating_table.obs[color].astype(
                            "str"
                        )  # this resets the categories to only contain those present in the the datasubset
                        annotating_table.obs[color] = annotating_table.obs[color].astype("category")
                        # check for NaN values
                        if annotating_table.obs[color].isna().sum() > 0:
                            # NaN values need to be filled as otherwise the plotting will throw an error
                            if "NaN" not in annotating_table.obs[color].cat.categories:
                                annotating_table.obs[color] = annotating_table.obs[color].cat.add_categories("NaN")
                            annotating_table.obs[color] = annotating_table.obs[color].fillna("NaN")
                    break
        if found_annotation is not None:
            sdata["_annotation"] = annotating_table
            sdata.pl.render_shapes(
                f"{label_layer}_vectorized",
                color=color,
                fill_alpha=fill_alpha,
                outline_alpha=0,
                cmap=cmap,
            ).pl.show(ax=ax)
            del sdata["_annotation"]  # delete element again after plotting
        else:
            try:
                sdata.pl.render_labels(
                    f"{label_layer}", color=color, fill_alpha=fill_alpha, outline_alpha=1, cmap=cmap
                ).pl.show(ax=ax)
            except Exception as err:
                raise Exception from err

    else:
        sdata.pl.render_labels(f"{label_layer}", color=color, fill_alpha=fill_alpha, cmap=cmap).pl.show(ax=ax)

    # configure axes
    ax.axis("off")
    ax.set_title(title, fontsize=font_size)

    # return elements
    if return_fig:
        return fig
    elif show_fig:
        plt.show()
        return None
    else:
        return None
