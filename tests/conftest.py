import shutil

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel, ShapesModel

from scportrait.tools.sdata.write._helper import _normalize_anndata_strings

rng = np.random.default_rng()


@pytest.fixture(autouse=True)
def _disable_matplotlib_show(monkeypatch):
    # Force a non-interactive backend for e2e runs
    matplotlib.use("Agg", force=True)

    # Disable any implicit rendering during tests
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(Figure, "show", lambda *args, **kwargs: None)


@pytest.fixture
def h5sc_object() -> AnnData:
    # Two cells, two channels, small images
    cell_ids = [101, 102, 107, 109]
    n_cells = 4
    channel_names = np.array(["seg_all_nucleus", "ch0", "ch1"])
    channel_mapping = np.array(["mask", "image", "image"])  # or whatever mapping your code expects
    n_channels = len(channel_names)
    H, W = 10, 10

    # --- obs ---
    obs = pd.DataFrame({"scportrait_cell_id": cell_ids}, index=np.arange(n_cells))

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

    yield adata


@pytest.fixture()
def sdata(tmp_path) -> SpatialData:
    sdata = blobs()
    _normalize_anndata_strings(sdata["table"])
    # Write to temporary location
    sdata_path = tmp_path / "sdata.zarr"
    sdata.write(sdata_path)
    yield sdata
    shutil.rmtree(sdata_path)


@pytest.fixture
def sdata_with_labels() -> SpatialData:
    sdata = blobs()
    _normalize_anndata_strings(sdata["table"])
    sdata["table"].obs["labelling_categorical"] = sdata["table"].obs["instance_id"].astype("category")
    sdata["table"].obs["labelling_continous"] = (sdata["table"].obs["instance_id"] > 10).astype(float)
    return sdata


@pytest.fixture
def sdata_with_selected_region():
    image = np.ones((1, 10, 10), dtype=np.uint16)
    shape = box(2, 3, 7, 8)
    image_model = Image2DModel.parse(image, dims=("c", "y", "x"))
    shapes_gdf = gpd.GeoDataFrame({"geometry": [shape]})
    shapes_model = ShapesModel.parse(shapes_gdf)
    sdata = SpatialData(images={"input_image": image_model}, shapes={"select_region": shapes_model})
    return sdata


@pytest.fixture
def sdata_builder():
    def _build(
        image,
        shapes,
        image_name="input_image",
        shape_name="select_region",
        dims=("c", "y", "x"),
    ):
        image_model = Image2DModel.parse(image, dims=dims)
        shapes_gdf = gpd.GeoDataFrame({"geometry": shapes})
        shapes_model = ShapesModel.parse(shapes_gdf)
        return SpatialData(images={image_name: image_model}, shapes={shape_name: shapes_model})

    return _build
