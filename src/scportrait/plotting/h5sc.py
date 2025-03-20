"""Collection of plotting functions for scPortrait's standardized single-cell image format"""

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scportrait.pipeline._utils.constants import DEFAULT_CELL_ID_NAME
from scportrait.tools.h5sc import get_image_with_cellid


def _reshape_image_array(arr: np.ndarray) -> np.ndarray:
    """
    Reshape an array from (n, c, x, y) to (n*c, x, y) while maintaining the order:
    n1 c1, n1 c2, ..., n1 cy, n2 c1, ..., nx cy.
    """
    if len(arr.shape) == 3:
        return arr
    else:
        n, c, x, y = arr.shape
        return arr.reshape(n * c, x, y)


def _plot_image_grid(
    ax: Axes,
    images: np.ndarray,
    nrows: int,
    ncols: int,
    spacing: float = 0.01,
    image_titles: list[str] | None = None,
    image_titles_fontsize: int = 10,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    axs_title: str | None = None,
    axs_title_padding: float = 0,
    axs_title_fontsize: int = 12,
    cmap="viridis",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Helper function to plot an image grid with consistent spacing between rows and columns.

    Args:
        ax: The matplotlib axes object to plot on.
        images: The images to plot.
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        spacing: The spacing between cells in the grid expressed as fraction of the cell image size.
        image_titles: The titles for each image in the grid.
        image_titles_fontsize: The fontsize of the image titles.
        row_labels: The labels for each row in the grid.
        col_labels: The labels for each column in the grid.
        axs_title: The title of the axes.
        axs_title_padding: The padding of the axes title.
        axs_title_fontsize: The fontsize of the axes title.
        cmap: The colormap to use for the images.
        vmin: The minimum value for the colormap.
        vmax: The maximum value for the colormap.

    Returns:
        None
    """

    ax.set_title(axs_title, fontsize=axs_title_fontsize, pad=axs_title_padding)
    ax.axis("off")

    # Adjust row spacing if image titles are provided
    title_adjustment = 0.01 if image_titles is not None else 0

    # Calculate effective cell width and height considering spacing
    spacing = spacing / max(nrows, ncols)
    cell_width = (1 - (ncols + 1) * spacing) / ncols
    cell_height = (1 - (nrows + 1) * spacing - title_adjustment) / nrows  # Adjust height

    for i, img in enumerate(images):
        row = i // ncols
        col = i % ncols

        ax_sub = ax.inset_axes(
            [
                spacing + col * (cell_width + spacing),  # X position
                1 - (spacing + (row + 1) * (cell_height + spacing + title_adjustment)),  # Y position
                cell_width,  # Width
                cell_height,  # Height
            ]
        )

        ax_sub.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_sub.yaxis.set_visible(False)
        ax_sub.xaxis.set_visible(False)

        if row_labels is not None and col == 0:
            ax_sub.set_ylabel(row_labels[row], fontsize=image_titles_fontsize)
            ax_sub.yaxis.set_visible(True)
            ax_sub.tick_params(left=False, labelleft=False)

        if col_labels is not None and row == 0:
            ax_sub.set_title(f"{col_labels[col]}", fontsize=image_titles_fontsize)

        if image_titles is not None:
            ax_sub.set_title(f"{image_titles[i]}", fontsize=image_titles_fontsize, pad=2)


def cell_grid_single_channel(
    adata,
    select_channel: int | str,
    n_cells: int = 16,
    cell_ids: int | list[int] | None = None,
    cell_labels: list[str] | None = None,
    show_cell_id: bool = False,
    title: str | None = None,
    show_title: bool = True,
    cmap="viridis",
    ncols: int | None = None,
    nrows: int | None = None,
    single_cell_size: int = 2,
    spacing: float = 0.025,
    ax: Axes = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> None | Figure:
    """Visualize a single channel from h5sc object in a grid.

    Args:
        adata: An scPortrait single-cell image dataset.
        select_channel: The channel to visualize.
        n_cells: The number of cells to visualize. This number of cells will randomly be selected. If `None`, `cell_ids` must be provided.
        cell_ids: cell IDs for the specific cells that should be visualized. If `None`, `n_cells` must be provided.
        cell_labels: Label to plot as title for each single-cell image if provided.
        show_cell_id: Whether to show the cell ID as title for each single-cell image. Can not be used together with `cell_labels`.
        title: The title of the plot.
        show_title: Whether to show the title.
        cmap: The colormap to use for the images.
        ncols: The number of columns in the grid. If not specified will be automatically calculated to make a square grid.
        nrows: The number of rows in the grid. If not specified will be automatically calculated to make a square grid.
        single_cell_size: The size of each cell in the grid.
        spacing: The spacing between cells in the grid expressed as fraction of the cell image size.
        axs: The matplotlib axes object to plot on. If `None`, a new figure is created.
        return_fig: If `True`, the function returns the figure object instead of displaying it.

    Returns:
        If `return_fig=True`, the figure object is returned. Otherwise, the figure is displayed.
    """
    # ensure that cell_ids are passed as a list
    if isinstance(cell_ids, int):
        _cell_ids = [cell_ids]
    else:
        _cell_ids = cell_ids

    # ensure that this parameter does not need to be set
    if _cell_ids is not None:
        n_cells = None

    if isinstance(select_channel, str):
        channel_id = adata.uns["single_cell_images"]["channel_names"].tolist().index(select_channel)
        channel_name = select_channel
    elif isinstance(select_channel, int):
        channel_id = select_channel
        channel_name = adata.uns["single_cell_images"]["channel_names"].tolist()[select_channel]

    if n_cells is None:
        assert _cell_ids is not None, "Either `n_cells` or `cell_ids` must be provided."
        n_cells = len(_cell_ids)
    else:
        # get random cells if no specific IDs are provided
        _cell_ids = adata.obs[DEFAULT_CELL_ID_NAME].sample(n_cells).values
        assert cell_labels is None, "can not provide labels for randomly sampled cells."

    if cell_labels is not None:
        assert show_cell_id is False, "can not show cell IDs and labels at the same time."
        spacing = 0.05

    if show_cell_id:
        cell_labels = [f"Cell ID {x}" for x in _cell_ids]
        spacing = 0.05

    # configure size for resulting single-cell image grid
    if ncols is None and nrows is None:
        ncols = int(np.ceil(np.sqrt(n_cells)))
        nrows = int(np.ceil(n_cells / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_cells / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n_cells / ncols))

    # set up image size
    fig_height = nrows * single_cell_size
    fig_width = ncols * single_cell_size

    # configure title
    if title is None:
        title = f"Channel {channel_name}"
    if not show_title:
        title = None

    # create figure object
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    else:
        fig = ax.get_figure()

    spacing = spacing * single_cell_size
    images = get_image_with_cellid(adata, _cell_ids, channel_id)
    _plot_image_grid(
        ax, images, nrows=nrows, ncols=ncols, axs_title=title, image_titles=cell_labels, cmap=cmap, spacing=spacing
    )

    if return_fig:
        return fig
    elif show_fig:
        plt.show()
        return None
    else:
        return None


def cell_grid_multi_channel(
    adata,
    n_cells: int = 5,
    cell_ids: int | list[int] | None = None,
    title: str | None = None,
    show_cell_id: bool = True,
    label_channels: bool = True,
    cmap="viridis",
    spacing: float = 0.025,
    single_cell_size: int = 2,
    axs: Axes = None,
    return_fig: bool = False,
    show_fig: bool = True,
) -> None | Figure:
    """Plot a grid of single-cell images where each row is a unique cell and each column contains a different channel.

    Args:
        adata: An scPortrait single-cell image dataset.
        n_cells: The number of cells to visualize. This number of cells will randomly be selected. If `None`, `cell_ids` must be provided.
        cell_ids: cell IDs for the specific cells that should be visualized. If `None`, `n_cells` must be provided.
        title: The title of the plot.
        show_cell_id: Whether to show the cell ID as row label for each cell in the image grid.
        label_channels: Whether to show the channel names as titles for column in the image grid.
        cmap: The colormap to use for the images.
        spacing: The spacing between cells in the grid expressed as fraction of the cell image size.
        single_cell_size: The size of each cell in the grid.
        axs: The matplotlib axes object to plot on. If `None`, a new figure is created.
        return_fig: If `True`, the function returns the figure object instead of displaying it.

    Returns:
        If `return_fig=True`, the figure object is returned. Otherwise, the figure is displayed.
    """
    # ensure that cell_ids are passed as a list
    if isinstance(cell_ids, Iterable):
        _cell_ids = cell_ids
    else:
        _cell_ids = [cell_ids]

    if cell_ids is not None:
        n_cells = None

    if n_cells is None:
        assert _cell_ids is not None, "Either `n_cells` or `cell_ids` must be provided."
        n_cells = len(_cell_ids)
    else:
        # Get random cells if no specific IDs are provided
        _cell_ids = adata.obs[DEFAULT_CELL_ID_NAME].sample(n_cells).values

    n_channels = adata.uns["single_cell_images"]["n_channels"]
    channel_names = adata.uns["single_cell_images"]["channel_names"]

    # Determine grid size
    nrows = n_cells
    ncols = n_channels

    # Configure plot size
    fig_width = ncols * single_cell_size
    fig_height = nrows * single_cell_size

    # Create figure object
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    else:
        fig = axs.get_figure()

    # Collect images in a list
    images = get_image_with_cellid(adata, _cell_ids)
    images = _reshape_image_array(images)

    # Call the image grid function
    spacing = spacing * single_cell_size
    _plot_image_grid(
        ax=axs,
        images=images,
        nrows=nrows,
        ncols=ncols,
        spacing=spacing,
        row_labels=[f"cell ID {_id}" for _id in _cell_ids] if show_cell_id else None,
        col_labels=channel_names if label_channels else None,
        axs_title=title,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    if return_fig:
        return fig
    elif show_fig:
        plt.show()
        return None
    else:
        return None


def cell_grid(
    adata: AnnData,
    select_channel: int | None = None,
    n_cells: int | None = None,
    cell_ids: int | list[int] | None = None,
    cmap="viridis",
    return_fig: bool = False,
    show_fig: bool = True,
):
    """Visualize single-cell images of cells in an AnnData object.

    Uses either `cell_grid_single_channel` or `cell_grid_multi_channel` depending on the input.
    Use these functions if you would like more control over the visualization.

    Args:
        adata: An scPortrait single-cell image dataset.
        n_cells: The number of cells to visualize. This number of cells will randomly be selected. If set to `None`, `cell_ids` must be provided.
        cell_ids: cell IDs of the cells that are to be visualiazed. If `None`, `n_cells` must be provided.
        cmap: The colormap to use for the images.
        return_fig: If `True`, the function returns the figure object instead of displaying it.

    Returns:
        If `return_fig=True`, the figure object is returned. Otherwise, the figure is displayed.
    """

    if select_channel is None:
        if cell_ids is None:
            if n_cells is None:
                n_cells = 5
        return cell_grid_multi_channel(
            adata, n_cells=n_cells, cell_ids=cell_ids, cmap=cmap, return_fig=return_fig, show_fig=show_fig
        )
    else:
        if cell_ids is None:
            if n_cells is None:
                n_cells = 16
        return cell_grid_single_channel(
            adata,
            select_channel,
            n_cells=n_cells,
            cell_ids=cell_ids,
            cmap=cmap,
            return_fig=return_fig,
            show_fig=show_fig,
        )
