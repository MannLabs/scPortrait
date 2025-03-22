#######################################################
# Unit tests for ../plotting/vis.py
#######################################################
import os

import numpy as np
import pytest

from scportrait.plotting.vis import plot_image, visualize_class


def test_visualize_class():
    class_ids = [1, 2]
    seg_map = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 1]])
    rng = np.random.default_rng()
    background = rng.random((3, 3)) * 255
    # Since this function does not return anything, we just check if it produces any exceptions
    try:
        visualize_class(class_ids, seg_map, background)
    except (ValueError, TypeError) as e:
        pytest.fail(f"visualize_class raised exception: {str(e)}")


def test_plot_image(tmpdir):
    rng = np.random.default_rng()
    array = rng.random((10, 10))
    save_name = tmpdir.join("test_plot_image")

    # Since this function does not return anything, we just check if it produces any exceptions
    try:
        plot_image(array, size=(5, 5), save_name=save_name)
    except (OSError, ValueError, TypeError) as e:
        pytest.fail(f"plot_image raised exception: {str(e)}")
    assert os.path.isfile(str(save_name) + ".png")
