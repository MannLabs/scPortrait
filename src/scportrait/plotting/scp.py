"""Collection of plotting functions for scPortrait's standardized single-cell image format"""

import matplotlib.pyplot as plt
from anndata import AnnData

from scportrait.pipeline._utils.constants import DEFAULT_CELL_ID_NAME
from scportrait.tools.scp.operations import get_scp_images


def cell_images(
    adata: AnnData, n_cells: int | None = 5, cell_ids: list[int] | None = None, cmap="viridis", return_fig: bool = False
):
    """Visualize single-cell images of cells in an AnnData object.

    Args:
        adata: An scPortrait single-cell image dataset.
        n_cells: The number of cells to visualize. This number of cells will randomly be selected. If `None`, `cell_ids` must be provided.
        cell_ids: cell IDs for the specific cells that should be visualized. If `None`, `n_cells` must be provided.
        cmap: The colormap to use for the images.
        return_fig: If `True`, the function returns the figure object instead of displaying it.

    Returns:
        If `return_fig=True`, the figure object is returned. Otherwise, the figure is displayed.
    """

    if n_cells is None:
        assert cell_ids is not None, "Either `n_cells` or `cell_ids` must be provided."
        n_cells = len(cell_ids)
    else:
        # get random cells if no specific IDs are provided
        cell_ids = adata.obs[DEFAULT_CELL_ID_NAME].sample(n_cells).values

    n_channels = adata.uns["single_cell_images"]["n_channels"]
    channel_names = adata.uns["single_cell_images"]["channel_names"]

    fig, axs = plt.subplots(n_cells, n_channels, figsize=(n_channels * 2, n_cells * 2))
    for i, _id in enumerate(cell_ids):
        images = get_scp_images(adata, [_id])
        for n, _img in enumerate(images):
            axs[i, n].imshow(_img, vmin=0, vmax=1, cmap=cmap)
            if n == 0:
                axs[i, n].set_ylabel(f"cell {_id}", fontsize=10, rotation=0, labelpad=25)
                axs[i, n].xaxis.set_visible(False)
                axs[i, n].tick_params(left=False, labelleft=False)
            else:
                axs[i, n].axis("off")

            if i == 0:
                axs[i, n].set_title(channel_names[n], fontsize=10)
    fig.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        return None
