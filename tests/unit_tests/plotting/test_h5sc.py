import matplotlib.pyplot as plt
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
@pytest.mark.parametrize(
    "input_shape, expected_shape",
    [
        ((10, 64, 64), (10, 64, 64)),  # 3D array
        ((5, 3, 64, 64), (15, 64, 64)),  # 4D array
        ((1, 3, 64, 64), (3, 64, 64)),  # Single image in a batch
    ],
)
def test_reshape_image_array(input_shape, expected_shape):
    arr = rng.random(input_shape)
    reshaped = _reshape_image_array(arr)
    assert reshaped.shape == expected_shape


# ---------- _plot_image_grid ----------
@pytest.mark.parametrize(
    "input_shape, nrows, ncols, col_labels, col_labels_rotation",
    [
        ((4, 10, 10), 2, 2, ["A", "B"], 45),  # 45 degree rotation
        ((1, 10, 10), 1, 1, ["A"], 0),  # 0 degree rotation
        ((10, 10, 10), 3, 3, None, 0),  # No labels, no rotation
    ],
)
def test_plot_image_grid(input_shape, nrows, ncols, col_labels, col_labels_rotation):
    arr = rng.random(input_shape)
    fig, ax = plt.subplots(1, 1)
    _plot_image_grid(
        ax,
        arr,
        nrows=nrows,
        ncols=ncols,
        col_labels=col_labels,
        col_labels_rotation=col_labels_rotation,
    )
    assert len(ax.child_axes) == nrows * ncols
    # If col_labels is set, check some title rotation attribute
    if col_labels is not None:
        # Just check one subplot for the label rotation
        for ax_sub in ax.child_axes:
            if ax_sub.get_title():
                assert any(
                    [
                        getattr(ax_sub.title, "get_rotation", lambda: None)() == col_labels_rotation,
                        ax_sub.title.get_rotation() == col_labels_rotation,
                    ]
                )
                break


# ---------- cell_grid_single_channel ----------
def test_cell_grid_single_channel_returns_figure_with_title_rotation(h5sc_object):
    fig = cell_grid_single_channel(
        adata=h5sc_object,
        select_channel="ch0",
        n_cells=2,
        return_fig=True,
        show_fig=False,
        title_rotation=30,  # new parameter
    )
    assert isinstance(fig, Figure)


# ---------- cell_grid_multi_channel ----------
def test_cell_grid_multi_channel_returns_figure_with_channel_label_rotation(h5sc_object):
    fig = cell_grid_multi_channel(
        adata=h5sc_object,
        n_cells=2,
        return_fig=True,
        show_fig=False,
        channel_label_rotation=60,  # new parameter
    )
    assert isinstance(fig, Figure)
    # You could extend this to check subplot titles for correct rotation if desired


# ---------- cell_grid ----------
def test_cell_grid_dispatches_to_single_channel(h5sc_object):
    cell_grid(adata=h5sc_object, select_channel="ch1", n_cells=1, show_fig=False)


def test_cell_grid_dispatches_to_multi_channel(h5sc_object):
    cell_grid(adata=h5sc_object, select_channel=["ch0", "ch1"], n_cells=1, show_fig=False)
