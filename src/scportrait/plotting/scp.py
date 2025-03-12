"""Collection of plotting functions for scPortrait's standardized single-cell image format"""

import matplotlib.pyplot as plt
from anndata import AnnData

from scportrait.tools.scp.operations import get_scp_images


def cell_images(adata: AnnData, cell_ids: list[int], cmap="Greys_r", return_fig: bool = False):
    n_cells = len(cell_ids)
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
