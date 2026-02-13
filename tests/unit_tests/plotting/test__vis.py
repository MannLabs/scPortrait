#######################################################
# Unit tests for ../plotting/vis.py
#######################################################
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scportrait.plotting._vis import colorize, generate_composite, plot_image_array, visualize_class


def test_visualize_class():
    class_ids = [1, 2]
    seg_map = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 1]])
    rng = np.random.default_rng()
    background = rng.random((3, 3)) * 255
    # Since this function does not return anything, we just check if it produces any exceptions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            visualize_class(class_ids, seg_map, background)
        except (ValueError, TypeError) as e:
            pytest.fail(f"visualize_class raised exception: {str(e)}")


def test_visualize_class_returns_fig():
    class_ids = [1, 2]
    seg_map = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 1]])
    rng = np.random.default_rng()
    background = rng.random((3, 3)) * 255
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fig = visualize_class(class_ids, seg_map, background, return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_image_array(tmpdir):
    rng = np.random.default_rng()
    array = rng.random((10, 10))
    save_name = tmpdir.join("test_plot_image")

    # Since this function does not return anything, we just check if it produces any exceptions
    try:
        plot_image_array(array, size=(5, 5), save_name=save_name)
    except (OSError, ValueError, TypeError) as e:
        pytest.fail(f"plot_image raised exception: {str(e)}")
    assert os.path.isfile(str(save_name) + ".png")


def test_plot_image_array_return_fig():
    rng = np.random.default_rng()
    array = rng.random((10, 10))
    fig = plot_image_array(array, return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_colorize_normalized_output():
    rng = np.random.default_rng()
    im = rng.random((16, 16))
    rgb = colorize(im, color=(0, 1, 0), normalize_image=True)
    assert rgb.shape == (16, 16, 3)
    assert rgb.min() >= 0
    assert rgb.max() <= 1


def test_colorize_invalid_shape_raises():
    rng = np.random.default_rng()
    im = rng.random((4, 4, 3))
    with pytest.raises(ValueError):
        colorize(im)


def test_generate_composite_basic():
    rng = np.random.default_rng()
    images = rng.random((2, 8, 8))
    composite = generate_composite(images)
    assert composite.shape == (8, 8, 3)
    assert composite.min() >= 0
    assert composite.max() <= 1
