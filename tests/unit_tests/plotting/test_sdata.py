import matplotlib

matplotlib.use("Agg")  # Ensure no GUI rendering during tests

import matplotlib.pyplot as plt
import pytest
import spatialdata as sd
from spatialdata.datasets import blobs

from scportrait.plotting import sdata as plotting


@pytest.mark.parametrize(
    "channel_names, palette, return_fig, show_fig",
    [
        (None, None, True, False),
        ([0], ["red"], True, False),
        ([0, 1], None, False, False),
    ],
)
def test_plot_image(sdata_with_labels, channel_names, palette, return_fig, show_fig):
    fig = plotting.plot_image(
        sdata=sdata_with_labels,
        image_name="blobs_image",
        channel_names=channel_names,
        palette=palette,
        title="Test Image",
        return_fig=return_fig,
        show_fig=show_fig,
    )
    if return_fig:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    else:
        assert fig is None


def test_plot_image_with_ax_returns_fig(sdata_with_labels):
    fig, ax = plt.subplots()
    returned = plotting.plot_image(
        sdata=sdata_with_labels,
        image_name="blobs_image",
        channel_names=[0],
        palette=["red"],
        return_fig=True,
        show_fig=False,
        ax=ax,
    )
    assert returned is fig
    plt.close(fig)


@pytest.mark.parametrize(
    "selected_channels, background_image",
    [
        (None, "blobs_image"),
        ([0], "blobs_image"),
        (None, None),  # test only mask overlay without image
    ],
)
def test_plot_segmentation_mask(sdata_with_labels, selected_channels, background_image):
    fig = plotting.plot_segmentation_mask(
        sdata=sdata_with_labels,
        masks=["blobs_labels"],
        background_image=background_image,
        selected_channels=selected_channels,
        return_fig=True,
        show_fig=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_segmentation_mask_selected_channels_out_of_range(sdata_with_labels):
    with pytest.raises(ValueError):
        plotting.plot_segmentation_mask(
            sdata=sdata_with_labels,
            masks=["blobs_labels"],
            background_image="blobs_image",
            selected_channels=[999],
            return_fig=False,
            show_fig=False,
        )


def test_plot_segmentation_mask_with_ax_returns_fig(sdata_with_labels):
    fig, ax = plt.subplots()
    returned = plotting.plot_segmentation_mask(
        sdata=sdata_with_labels,
        masks=["blobs_labels"],
        background_image="blobs_image",
        selected_channels=[0],
        return_fig=True,
        show_fig=False,
        ax=ax,
    )
    assert returned is fig
    plt.close(fig)


@pytest.mark.parametrize(
    "vectorized, color",
    [
        (False, "labelling_categorical"),
        (True, "labelling_categorical"),
        (False, "labelling_continous"),
        (True, "labelling_continous"),
        (False, "red"),
        (True, "red"),
    ],
)
def test_plot_labels(sdata_with_labels, vectorized, color):
    fig = plotting.plot_labels(
        sdata=sdata_with_labels,
        label_layer="blobs_labels",
        vectorized=vectorized,
        color=color,
        return_fig=True,
        show_fig=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_labels_with_ax_returns_fig(sdata_with_labels):
    fig, ax = plt.subplots()
    returned = plotting.plot_labels(
        sdata=sdata_with_labels,
        label_layer="blobs_labels",
        vectorized=False,
        color="labelling_categorical",
        return_fig=True,
        show_fig=False,
        ax=ax,
    )
    assert returned is fig
    plt.close(fig)


def test_plot_shapes_with_ax_returns_fig(sdata_with_labels):
    fig, ax = plt.subplots()
    returned = plotting.plot_shapes(
        sdata=sdata_with_labels,
        shapes_layer="blobs_polygons",
        return_fig=True,
        show_fig=False,
        ax=ax,
    )
    assert returned is fig
    plt.close(fig)
