from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

rng = np.random.default_rng()

from scportrait.plotting.h5sc import (
    _plot_image_grid,
    _reshape_image_array,
    cell_grid,
    cell_grid_multi_channel,
    cell_grid_single_channel,
)

# ---------- _reshape_image_array ----------


def test_reshape_image_array_3d():
    arr = rng.random((10, 64, 64))
    reshaped = _reshape_image_array(arr)
    assert reshaped.shape == (10, 64, 64)


def test_reshape_image_array_4d():
    arr = rng.random((5, 3, 64, 64))
    reshaped = _reshape_image_array(arr)
    assert reshaped.shape == (15, 64, 64)


# ---------- _plot_image_grid ----------


def test_plot_image_grid_runs():
    ax = MagicMock(spec=Axes)
    images = rng.random((4, 10, 10))
    _plot_image_grid(ax, images, nrows=2, ncols=2)
    assert ax.set_title.called


# ---------- cell_grid_single_channel ----------


@patch("scportrait.plotting.h5sc.get_image_with_cellid")
def test_cell_grid_single_channel_returns_figure(mock_get_img):
    mock_adata = MagicMock()
    mock_adata.uns = {"single_cell_images": {"channel_names": np.array(["ch0", "ch1"])}}
    mock_adata.obs = MagicMock()
    mock_adata.obs.__getitem__.return_value.sample.return_value.values = [101, 102]

    mock_get_img.return_value = rng.random((2, 10, 10))

    fig = cell_grid_single_channel(adata=mock_adata, select_channel=0, n_cells=2, return_fig=True, show_fig=False)

    assert isinstance(fig, Figure)


# ---------- cell_grid_multi_channel ----------


@patch("scportrait.plotting.h5sc.get_image_with_cellid")
def test_cell_grid_multi_channel_returns_figure(mock_get_img):
    mock_adata = MagicMock()
    mock_adata.uns = {"single_cell_images": {"channel_names": ["ch0", "ch1"], "n_channels": 2}}
    mock_adata.obs = MagicMock()
    mock_adata.obs.__getitem__.return_value.sample.return_value.values = [101, 102]

    mock_get_img.return_value = rng.random((2, 2, 10, 10))

    fig = cell_grid_multi_channel(adata=mock_adata, n_cells=2, return_fig=True, show_fig=False)

    assert isinstance(fig, Figure)


# ---------- cell_grid ----------


@patch("scportrait.plotting.h5sc.cell_grid_single_channel")
def test_cell_grid_dispatches_to_single_channel(mock_single):
    mock_adata = MagicMock()
    cell_grid(adata=mock_adata, select_channel=1, n_cells=1, show_fig=False)
    assert mock_single.called


@patch("scportrait.plotting.h5sc.cell_grid_multi_channel")
def test_cell_grid_dispatches_to_multi_channel(mock_multi):
    mock_adata = MagicMock()
    cell_grid(adata=mock_adata, select_channel=[0, 1], n_cells=1, show_fig=False)
    assert mock_multi.called
