import warnings
from collections.abc import Iterable
from math import ceil
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


def _is_valid_matplotlib_color(color: str | None) -> bool:
    """Return True if the input is a valid Matplotlib color specification."""
    if color is None:
        return False
    try:
        mpl.colors.to_rgba(color)
        return True
    except (TypeError, ValueError):
        return False


def _annotation_columns_for_layer(sdata: spatialdata.SpatialData, label_layer: str) -> set[str]:
    """Collect annotation columns from tables annotating a specific labels layer."""
    columns: set[str] = set()
    for table in sdata.tables:
        annotated_regions = sdata.get_annotated_regions(sdata[table])
        if label_layer in annotated_regions:
            columns.update(sdata[table].obs.columns)
    return columns


def _get_vectorized_annotating_table(
    sdata: spatialdata.SpatialData,
    label_layer: str,
) -> spatialdata.models.TableModel | None:
    """Return an annotation table remapped to the vectorized labels layer."""
    vectorized_layer = f"{label_layer}_vectorized"
    for table in sdata.tables:
        annotated_regions = sdata.get_annotated_regions(sdata[table])
        if label_layer in annotated_regions:
            annotating_table = sdata[table].copy()
            annotating_table.uns["spatialdata_attrs"]["region"] = vectorized_layer
            annotating_table.obs["region"] = vectorized_layer
            annotating_table.obs["region"] = annotating_table.obs["region"].astype("category")
            return spatialdata.models.TableModel.parse(annotating_table)
    return None


def _render_labels_as_fixed_color_shapes(
    sdata: spatialdata.SpatialData,
    label_layer: str,
    color: str,
    fill_alpha: float,
    ax: Axes,
    coordinate_systems: str | list[str] | None = None,
    method: str | None = None,
) -> None:
    """Render a labels layer as polygons with a fixed color."""
    vectorized_layer = f"{label_layer}_vectorized"
    if vectorized_layer not in sdata:
        sdata[vectorized_layer] = spatialdata.to_polygons(sdata[label_layer])
    sdata.pl.render_shapes(
        vectorized_layer,
        color=color,
        fill_alpha=fill_alpha,
        outline_alpha=0,
        method=method,
    ).pl.show(coordinate_systems=coordinate_systems, ax=ax)


def _normalize_groups(groups: list | None) -> list[str] | None:
    """Normalize groups to a list of strings for plotting."""
    if groups is None:
        return None
    if isinstance(groups, str):
        return [groups]
    return [str(group) for group in groups]


def _get_shape_element(sdata, element_name) -> tuple[int, int]:
    """Get the x, y shape of the element in the spatialdata object.

    Args:
        sdata: SpatialData object containing the element.
        element_name: Name of the element to get the shape of.

    Returns:
        Tuple containing the shape of the x, y coordinates of the element.
    """
    element = sdata[element_name]
    if isinstance(element, xarray.DataTree):
        shape = element.scale0.image.shape
    elif hasattr(element, "data") and hasattr(element.data, "shape"):
        shape = element.data.shape
    elif hasattr(element, "total_bounds"):
        min_x, min_y, max_x, max_y = element.total_bounds
        x = max(1, ceil(max_y - min_y))
        y = max(1, ceil(max_x - min_x))
        return x, y
    else:
        raise ValueError(f"Unsupported element type for '{element_name}': {type(element)}.")

    if len(shape) == 3:
        _, x, y = shape
    elif len(shape) == 2:
        x, y = shape
    else:
        raise ValueError(f"Unsupported shape for element '{element_name}': expected 2D or 3D array, got {shape}.")
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
    coordinate_systems: str | list[str] | None = None,
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

    .. note::
        Requires a working installation of `spatialdata_plot`. Can be installed via
        `pip install spatialdata_plot` or is shipped with scPortrait when installing
        with the `plotting` extra (`pip install scportrait[plotting]`).

    Args:
        sdata: SpatialData object containing the image.
        image_name: Name of the image to plot.
        coordinate_systems: Coordinate system(s) to plot. If None, all coordinate systems are plotted.
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

    Examples:
        >>> from spatialdata.datasets import blobs
        >>> from scportrait.plotting import sdata as splot
        >>> sdata = blobs()
        >>> fig = splot.plot_image(
        ...     sdata=sdata,
        ...     image_name="blobs_image",
        ...     channel_names=[0],
        ...     palette=["red"],
        ...     return_fig=True,
        ...     show_fig=False,
        ... )
    """
    _check_for_spatialdata_plot()

    if ax is not None:
        if dpi is not None:
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
        fig = ax.figure
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
    render_kwargs = {
        "channel": channel_names,
        "palette": palette,
        "cmap": cmap,
        "norm": norm,
    }

    sdata.pl.render_images(
        image_name,
        **render_kwargs,
    ).pl.show(coordinate_systems=coordinate_systems, ax=ax, colorbar=False)
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
    ax: Axes | None = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> plt.Figure | None:
    """Visualize segmentation masks over selected background image.
    Will transform the provided labels layers to polygons and plot them over the background image.

    .. note::
        Requires a working installation of `spatialdata_plot`. Can be installed via
        `pip install spatialdata_plot` or is shipped with scPortrait when installing
        with the `plotting` extra (`pip install scportrait[plotting]`).

    Args:
        sdata: SpatialData object containing the input image and segmentation mask.
        masks: List of keys identifying the segmentation masks to plot.
        background_image: Key identifying the background image to plot the segmentation masks on. Defaults to "input_image". If set to None then only the segmentation masks will be plotted as outlines.
        selected_channels: Index or list of indices of background-image channels to plot. If None then the first
            `max_channels_to_plot` channels will be plotted. Defaults to None.
        title: Title of the plot. Defaults to None.
        title_fontsize: Font size of the title in points.
        line_width: Width of the outline of the segmentation masks.
        line_color: Color of the outline of the segmentation masks.
        line_alpha: Alpha value of the outline of the segmentation masks.
        fill_alpha: Alpha value of the fill of the segmentation masks.
        fill_color: Color of the fill of the segmentation masks. If None then no fill will be used. Defaults to None.
        dpi: Dots per inch of the plot. Defaults to None.
        ax: Matplotlib axis object to plot on. Defaults to None.
        return_fig: Whether to return the figure. Defaults to False.
        show_fig: Whether to show the figure. Defaults to True.

    Examples:
        >>> from spatialdata.datasets import blobs
        >>> from scportrait.plotting import sdata as splot
        >>> sdata = blobs()
        >>> fig = splot.plot_segmentation_mask(
        ...     sdata=sdata,
        ...     masks=["blobs_labels"],
        ...     background_image="blobs_image",
        ...     selected_channels=0,
        ...     return_fig=True,
        ...     show_fig=False,
        ... )
    """
    # check for spatialdata_plot
    _check_for_spatialdata_plot()

    # get relevant information from spatialdata object

    if ax is not None:
        if dpi is not None:
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
        fig = ax.figure
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
            if any(i < 0 or i >= len(channel_names) for i in selected_channels):
                raise ValueError(
                    f"selected_channels contains out-of-range indices for background image '{background_image}'."
                )
            if len(selected_channels) > len(PALETTE):
                raise ValueError("selected_channels has more entries than the available palette length.")
            channel_names = [channel_names[i] for i in selected_channels]
            c = len(channel_names)
            palette = PALETTE[:c]
        else:
            if c > max_channels_to_plot:
                c = min(c, max_channels_to_plot)
            palette = PALETTE[:c]
            channel_names = list(channel_names[:c])

        sdata.pl.render_images(background_image, channel=channel_names, palette=palette).pl.show(ax=ax, colorbar=False)

    # plot selected segmentation masks
    for mask in masks:
        if mask not in sdata:
            raise KeyError(f"Mask {mask} not found in sdata object.")
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
    shapes_layer: str | None = None,
    label_layer: str | None = None,
    coordinate_systems: str | list[str] | None = None,
    method: str | None = None,
    title: str | None = None,
    title_fontsize: int = 20,
    fill_color: str = "grey",
    fill_alpha: float = 1,
    outline_color: str = "black",
    outline_alpha: float = 0,
    outline_width: float = 1,
    cmap: str = None,
    palette: dict | list | None = None,
    groups: list | None = None,
    dpi: int | None = None,
    ax: Axes | None = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> plt.Figure | None:
    """Visualize a shapes layer.

    .. note::
        Requires a working installation of `spatialdata_plot`. Can be installed via
        `pip install spatialdata_plot` or is shipped with scPortrait when installing
        with the `plotting` extra (`pip install scportrait[plotting]`).

    Args:
        sdata: SpatialData object containing the shapes or labels layer.
        shapes_layer: Key identifying the shapes layer to plot.
        label_layer: Key identifying a labels layer to convert to shapes and plot.
        coordinate_systems: Coordinate system(s) to plot. If None, all coordinate systems are plotted.
        method: Plotting backend passed to spatialdata_plot (`None`, "matplotlib", or "datashader").
        title: Title of the plot.
        title_fontsize: Font size of the title in points.
        fill_color: Color of the fill of the shapes.
        fill_alpha: Alpha value of the fill of the shapes.
        outline_color: Color of the outline of the shapes.
        outline_alpha: Alpha value of the outline of the shapes.
        outline_width: Width of the outline of the shapes.
        cmap: Colormap to use for the shapes.
        palette: Palette for discrete annotations.
        groups: Groups to plot when the color key refers to discrete annotations.
        dpi: Dots per inch of the plot.
        ax: Matplotlib axis object to plot on.
        return_fig: Whether to return the figure.
        show_fig: Whether to show the figure.

    Returns:
        Matplotlib figure object if return_fig is True.

    Examples:
        >>> from spatialdata.datasets import blobs
        >>> from scportrait.plotting import sdata as splot
        >>> sdata = blobs()
        >>> fig = splot.plot_shapes(
        ...     sdata=sdata,
        ...     shapes_layer="blobs_polygons",
        ...     return_fig=True,
        ...     show_fig=False,
        ... )
    """
    if (shapes_layer is None and label_layer is None) or (shapes_layer is not None and label_layer is not None):
        raise ValueError("Provide exactly one of 'shapes_layer' or 'label_layer'.")

    normalized_groups = _normalize_groups(groups)

    if label_layer is not None:
        if label_layer not in sdata:
            raise KeyError(f"Label layer {label_layer} not found in sdata object.")
        shapes_layer = f"{label_layer}_vectorized"
        if shapes_layer not in sdata:
            sdata[shapes_layer] = spatialdata.to_polygons(sdata[label_layer])

    if shapes_layer is None:
        raise ValueError("Unable to resolve shapes layer for plotting.")

    # check for spatialdata_plot
    _check_for_spatialdata_plot()

    if ax is not None:
        if dpi is not None:
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
        fig = ax.figure
    else:
        # get size of spatialdata object to plot (required for calculating figure size if DPI is set)
        x, y = _get_shape_element(sdata, shapes_layer)

        # create figure and axis if it does not exist
        fig, ax = _create_figure_dpi(x=x, y=y, dpi=dpi)

    # plot selected shapes layer
    if shapes_layer not in sdata:
        raise KeyError(f"Shapes layer {shapes_layer} not found in sdata object.")

    annotating_table = _get_vectorized_annotating_table(sdata, label_layer) if label_layer is not None else None
    if annotating_table is not None:
        if normalized_groups is not None and isinstance(fill_color, str) and fill_color in annotating_table.obs:
            annotating_table.obs[fill_color] = annotating_table.obs[fill_color].astype("string")
            annotating_table.obs[fill_color] = annotating_table.obs[fill_color].astype("category")
        had_annotation = "_annotation" in sdata
        prev_annotation = sdata["_annotation"] if had_annotation else None
        sdata["_annotation"] = annotating_table
        try:
            sdata.pl.render_shapes(
                f"{shapes_layer}",
                fill_alpha=fill_alpha,
                color=fill_color,
                outline_alpha=outline_alpha,
                outline_color=outline_color,
                outline_width=outline_width,
                cmap=cmap,
                palette=palette,
                groups=normalized_groups,
                method=method,
            ).pl.show(coordinate_systems=coordinate_systems, ax=ax)
        finally:
            if had_annotation:
                sdata["_annotation"] = prev_annotation
            else:
                del sdata["_annotation"]
    else:
        sdata.pl.render_shapes(
            f"{shapes_layer}",
            fill_alpha=fill_alpha,
            color=fill_color,
            outline_alpha=outline_alpha,
            outline_color=outline_color,
            outline_width=outline_width,
            cmap=cmap,
            palette=palette,
            groups=normalized_groups,
            method=method,
        ).pl.show(coordinate_systems=coordinate_systems, ax=ax)

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
    coordinate_systems: str | list[str] | None = None,
    method: str | None = None,
    title: str | None = None,
    title_fontsize: int = 20,
    color: str = "grey",
    fill_alpha: float = 1,
    cmap: str = None,
    palette: dict | list | None = None,
    groups: list | None = None,
    norm: mpl.colors.Normalize = None,
    vectorized: bool = False,
    dpi: int | None = None,
    ax: Axes | None = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> plt.Figure | None:
    """Plot a labels layer.

    .. note::
        Requires a working installation of `spatialdata_plot`. Can be installed via
        `pip install spatialdata_plot` or is shipped with scPortrait when installing
        with the `plotting` extra (`pip install scportrait[plotting]`).

    Args:
        sdata: SpatialData object containing the input image and segmentation mask.
        label_layer: Key identifying the label layer to plot.
        coordinate_systems: Coordinate system(s) to plot. If None, all coordinate systems are plotted.
        method: Plotting backend passed to spatialdata_plot (`None`, "matplotlib", or "datashader").
        title: Title of the plot.
        title_fontsize: Font size of the title in points.
        color: Color to plot the label layer in. Can be a string specifying a specific color or linking to a column in an annotating table.
        fill_alpha: Alpha value of the fill of the segmentation masks.
        cmap: Colormap to use for the labels.
        palette: Palette for discrete annotations. List of valid color names that should be used for the categories. Must match the number of groups. The list can contain multiple palettes (one per group) to be visualized. If groups is provided but not palette, palette is set to default “lightgray”.
        groups: When using color and the key represents discrete labels, groups can be used to show only a subset of them. Other values are set to NA.
        norm: Colormap normalization for continuous annotations.
        vectorized: Whether to plot a vectorized version of the labels.
        dpi: Dots per inch of the plot.
        ax: Matplotlib axis object to plot on.
        return_fig: Whether to return the figure.
        show_fig: Whether to show the figure.

    Returns:
        Matplotlib figure object if return_fig is True.

    Examples:
        >>> from spatialdata.datasets import blobs
        >>> from scportrait.plotting import sdata as splot
        >>> sdata = blobs()
        >>> fig = splot.plot_labels(
        ...     sdata=sdata,
        ...     label_layer="blobs_labels",
        ...     color="labelling_categorical",
        ...     return_fig=True,
        ...     show_fig=False,
        ... )
    """
    # check for spatialdata_plot
    _check_for_spatialdata_plot()

    if ax is not None:
        if dpi is not None:
            warnings.warn("DPI is ignored when an axis is provided.", stacklevel=2)
        fig = ax.figure
    else:
        # get size of spatialdata object to plot (required for calculating figure size if DPI is set)
        x, y = _get_shape_element(sdata, label_layer)

        # create figure and axis if it does not exist
        fig, ax = _create_figure_dpi(x=x, y=y, dpi=dpi)

    # plot selected segmentation masks
    annotation_columns = _annotation_columns_for_layer(sdata, label_layer)
    normalized_groups = _normalize_groups(groups)
    use_fixed_color_fallback = (
        isinstance(color, str) and color not in annotation_columns and _is_valid_matplotlib_color(color)
    )

    if vectorized:
        if use_fixed_color_fallback:
            _render_labels_as_fixed_color_shapes(
                sdata=sdata,
                label_layer=label_layer,
                color=color,
                fill_alpha=fill_alpha,
                ax=ax,
                coordinate_systems=coordinate_systems,
                method=method,
            )
        else:
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
                            if normalized_groups is not None:
                                annotating_table.obs[color] = annotating_table.obs[color].astype("string")
                                annotating_table.obs[color] = annotating_table.obs[color].astype("category")
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
                had_annotation = "_annotation" in sdata
                prev_annotation = sdata["_annotation"] if had_annotation else None
                sdata["_annotation"] = annotating_table
                try:
                    sdata.pl.render_shapes(
                        f"{label_layer}_vectorized",
                        color=color,
                        fill_alpha=fill_alpha,
                        outline_alpha=0,
                        cmap=cmap,
                        palette=palette,
                        groups=normalized_groups,
                        norm=norm,
                        method=method,
                    ).pl.show(coordinate_systems=coordinate_systems, ax=ax)
                finally:
                    if had_annotation:
                        sdata["_annotation"] = prev_annotation
                    else:
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
                        groups=normalized_groups,
                        norm=norm,
                        method=method,
                    ).pl.show(coordinate_systems=coordinate_systems, ax=ax)
                except Exception as err:
                    raise Exception from err
    else:
        if use_fixed_color_fallback:
            _render_labels_as_fixed_color_shapes(
                sdata=sdata,
                label_layer=label_layer,
                color=color,
                fill_alpha=fill_alpha,
                ax=ax,
                coordinate_systems=coordinate_systems,
                method=method,
            )
        else:
            try:
                sdata.pl.render_labels(
                    f"{label_layer}",
                    color=color,
                    fill_alpha=fill_alpha,
                    outline_alpha=1,
                    cmap=cmap,
                    palette=palette,
                    groups=normalized_groups,
                    norm=norm,
                    method=method,
                ).pl.show(coordinate_systems=coordinate_systems, ax=ax)
            except Exception as err:
                raise Exception from err

    # configure axes
    ax.axis("off")
    ax.set_title(title, fontsize=title_fontsize)

    # return elements
    if return_fig:
        return fig
    elif show_fig:
        plt.show()
    return None
