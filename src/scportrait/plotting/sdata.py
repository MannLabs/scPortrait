import warnings
from collections.abc import Iterable
from numbers import Integral

import matplotlib as mpl
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

from scportrait.pipeline._utils.helper import _check_for_spatialdata_plot


def _get_shape_element(sdata, element_name) -> tuple[int, int]:
    """Get the x, y shape of the element in the spatialdata object.

    Args:
        sdata: SpatialData object containing the element.
        element_name: Name of the element to get the shape of.

    Returns:
        Tuple containing the shape of the x, y coordinates of the element.
    """
    if isinstance(sdata[element_name], xarray.DataTree):
        shape = sdata[element_name].scale0.image.shape
    else:
        shape = sdata[element_name].data.shape

    if len(shape) == 3:
        _, x, y = shape
    elif len(shape) == 2:
        x, y = shape
    return x, y


def _create_figure_dpi(x: float, y: float, dpi: int | None = 300) -> tuple[plt.Figure, Axes]:
    """Helper function to create a figure to plot a given image with x, y resolution at the specified DPI.
    Args:
        x: x resolution of the image.
        y: y resolution of the image.
        dpi: Dots per inch of the image. Defaults to 300.
    Returns:
        Tuple containing the figure and axis objects.
    """
    # create figure and axis if it does not exist
    if dpi is None:
        dpi = 300

    axs_length = x / dpi
    axs_width = y / dpi

    fig = plt.figure(figsize=(axs_width, axs_length), frameon=False, dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    return fig, ax


def plot_image(
    sdata: spatialdata.SpatialData,
    image_name: str,
    channel_names: list[str] | list[int] | None = None,
    palette: list[str] | None = None,
    cmap: mpl.colors.ListedColormap | None = None,
    title: str | None = None,
    title_fontsize: int = 20,
    dpi: int | None = None,
    norm=None,
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
        title_fontsize: Font size of the title in points. Defaults to 20.
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
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
    else:
        # get size of spatialdata object to plot (required for calculating figure size if DPI is set)
        x, y = _get_shape_element(sdata, image_name)

        # create figure and axis if it does not exist
        fig, ax = _create_figure_dpi(x=x, y=y, dpi=dpi)

    if palette is None and cmap is None:
        if channel_names is None:
            palette = None
        else:
            palette = PALETTE[: len(channel_names)]

    # plot figure
    sdata.pl.render_images(image_name, channel=channel_names, palette=palette, cmap=cmap, norm=norm).pl.show(
        ax=ax, colorbar=False
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=title_fontsize)

    if return_fig:
        return fig
    elif show_fig:
        plt.show()
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
        title: Title of the plot. Defaults to None.
        title_fontsize: Font size of the title in points.
        line_width: Width of the outline of the segmentation masks.
        line_color: Color of the outline of the segmentation masks.
        line_alpha: Alpha value of the outline of the segmentation masks.
        fill_alpha: Alpha value of the fill of the segmentation masks.
        fill_color: Color of the fill of the segmentation masks. If None then no fill will be used. Defaults to None.
        dpi: Dots per inch of the plot. Defaults to None.
        axs: Matplotlib axis object to plot on. Defaults to None.
        return_fig: Whether to return the figure. Defaults to False.
        show_fig: Whether to show the figure. Defaults to True.
    """
    # check for spatialdata_plot
    _check_for_spatialdata_plot()

    # get relevant information from spatialdata object

    if ax is not None:
        if dpi is not None:
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
    else:
        # get size of spatialdata object to plot (required for calculating figure size if DPI is set)
        x, y = _get_shape_element(sdata, masks[0])

        # create figure and axis if it does not exist
        fig, ax = _create_figure_dpi(x=x, y=y, dpi=dpi)

    # plot background image if desired
    if background_image is not None:
        # get channel names
        if background_image not in sdata:
            raise ValueError(f"Background image {background_image} not found in sdata object.")

        if isinstance(sdata[background_image], xarray.DataTree):
            channel_names = sdata[background_image].scale0.c.values
        else:
            channel_names = sdata[background_image].c.values

        # required work around because spatialdata plot only supports int and not things like int64
        if all(isinstance(x, Integral) for x in channel_names):
            channel_names = [int(x) for x in channel_names]

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


def plot_shapes(
    sdata: spatialdata.SpatialData,
    shapes_layer: str,
    title: str | None = None,
    title_fontsize: int = 20,
    fill_color: str = "grey",
    fill_alpha: float = 1,
    outline_color: str = "black",
    outline_alpha: float = 0,
    outline_width: float = 1,
    cmap: str = None,
    dpi: int | None = None,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> plt.Figure | None:
    """Visualize shapes layer.

    Args:
        sdata: SpatialData object containing the shapes layer.
        shapes_layer: Key identifying the shapes layer to plot.
        title: Title of the plot.
        title_fontsize: Font size of the title in points.
        fill_color: Color of the fill of the shapes.
        fill_alpha: Alpha value of the fill of the shapes.
        outline_color: Color of the outline of the shapes.
        outline_alpha: Alpha value of the outline of the shapes.
        outline_width: Width of the outline of the shapes.
        cmap: Colormap to use for the shapes.
        dpi: Dots per inch of the plot.
        ax: Matplotlib axis object to plot on.
        return_fig: Whether to return the figure.
        show_fig: Whether to show the figure.

    Returns:
        Matplotlib figure object if return_fig is True.
    """

    # check for spatialdata_plot
    _check_for_spatialdata_plot()

    if ax is not None:
        if dpi is not None:
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
    else:
        # get size of spatialdata object to plot (required for calculating figure size if DPI is set)
        x, y = _get_shape_element(sdata, shapes_layer)

        # create figure and axis if it does not exist
        fig, ax = _create_figure_dpi(x=x, y=y, dpi=dpi)

    # plot selected shapes layer
    assert shapes_layer in sdata, f"Shapes layer {shapes_layer} not found in sdata object."

    sdata.pl.render_shapes(
        f"{shapes_layer}",
        fill_alpha=fill_alpha,
        color=fill_color,
        outline_alpha=outline_alpha,
        outline_color=outline_color,
        outline_width=outline_width,
        cmap=cmap,
    ).pl.show(ax=ax)

    ax.axis("off")
    ax.set_title(title, fontsize=title_fontsize)

    # return elements
    if return_fig:
        return fig
    elif show_fig:
        plt.show()
    return None


def plot_labels(
    sdata: spatialdata.SpatialData,
    label_layer: str,
    title: str | None = None,
    title_fontsize: int = 20,
    color: str = "grey",
    fill_alpha: float = 1,
    cmap: str = None,
    palette: dict | list = None,
    groups: list = None,
    norm: mpl.colors.Normalize = None,
    vectorized: bool = False,
    dpi: int | None = None,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> plt.Figure | None:
    """Plot the segmentation mask on the input image.

    Args:
        sdata: SpatialData object containing the input image and segmentation mask.
        labels_layer: Key identifying the label layer to plot.
        title: Title of the plot.
        title_fontsize: Font size of the title in points.
        color: color to plot the label layer in. Can be a string specifying a specific color or linking to a column in an annotating table.
        fill_alpha: Alpha value of the fill of the segmentation masks.
        cmap: Colormap to use for the labels.
        palette: Palette for discrete annotations. List of valid color names that should be used for the categories. Must match the number of groups. The list can contain multiple palettes (one per group) to be visualized. If groups is provided but not palette, palette is set to default “lightgray”.
        groups: When using color and the key represents discrete labels, groups can be used to show only a subset of them. Other values are set to NA.
        norm: Colormap normalization for continuous annotations
        vectorized: Whether to plot a vectorized version of the labels.
        dpi: Dots per inch of the plot.
        axs: Matplotlib axis object to plot on.
        return_fig: Whether to return the figure.
        show_fig: Whether to show the figure.

    Returns:
        Matplotlib figure object if return_fig is True.
    """
    # check for spatialdata_plot
    _check_for_spatialdata_plot()

    if ax is not None:
        if dpi is not None:
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
    else:
        # get size of spatialdata object to plot (required for calculating figure size if DPI is set)
        x, y = _get_shape_element(sdata, label_layer)

        # create figure and axis if it does not exist
        fig, ax = _create_figure_dpi(x=x, y=y, dpi=dpi)

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
                        annotating_table.obs[color] = (
                            annotating_table.obs[color].astype("category").cat.remove_unused_categories()
                        )
                        # check for NaN values
                        if annotating_table.obs[color].isna().sum() > 0:
                            # NaN values need to be filled as otherwise the plotting will throw an error
                            if "NaN" not in annotating_table.obs[color].cat.categories:
                                annotating_table.obs[color] = annotating_table.obs[color].cat.add_categories("NaN")
                            annotating_table.obs[color] = annotating_table.obs[color].fillna("NaN")
                    annotating_table = spatialdata.models.TableModel.parse(annotating_table)
                    break
        if found_annotation is not None:
            sdata["_annotation"] = annotating_table
            sdata.pl.render_shapes(
                f"{label_layer}_vectorized",
                color=color,
                fill_alpha=fill_alpha,
                outline_alpha=0,
                cmap=cmap,
                palette=palette,
                groups=groups,
                norm=norm,
            ).pl.show(ax=ax)
            del sdata["_annotation"]  # delete element again after plotting
        else:
            try:
                sdata.pl.render_labels(
                    f"{label_layer}",
                    color=color,
                    fill_alpha=fill_alpha,
                    outline_alpha=1,
                    cmap=cmap,
                    palette=palette,
                    groups=groups,
                    norm=norm,
                ).pl.show(ax=ax)
            except Exception as err:
                raise Exception from err

    else:
        sdata.pl.render_labels(
            f"{label_layer}",
            color=color,
            fill_alpha=fill_alpha,
            cmap=cmap,
            palette=palette,
            groups=groups,
            norm=norm,
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
