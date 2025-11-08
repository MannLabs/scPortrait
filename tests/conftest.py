import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

rng = np.random.default_rng()


@pytest.fixture
def h5sc_object():
    # Two cells, two channels, small images
    n_cells = 2
    channel_names = np.array(["seg_all_nucleus", "ch0", "ch1"])
    channel_mapping = np.array(["mask", "image", "image"])  # or whatever mapping your code expects
    n_channels = len(channel_names)
    H, W = 10, 10

    # --- obs ---
    obs = pd.DataFrame({"scportrait_cell_id": [101, 102]}, index=[0, 1])

    # --- var (channel metadata) ---
    var = pd.DataFrame(index=np.arange(n_channels).astype("str"))
    var["channels"] = channel_names
    var["channel_mapping"] = channel_mapping

    adata = AnnData(obs=obs, var=var)
    adata.obsm["single_cell_images"] = rng.random((n_cells, n_channels, H, W))
    adata.uns["single_cell_images"] = {
        "channel_mapping": channel_mapping,
        "channel_names": channel_names,
        "compression": "lzf",
        "image_size": np.int64(H),
        "n_cells": np.int64(n_cells),
        "n_channels": np.int64(n_channels),
        "n_image_channels": np.int64(n_channels - 1),
        "n_masks": np.int64(1),
    }

    return adata
