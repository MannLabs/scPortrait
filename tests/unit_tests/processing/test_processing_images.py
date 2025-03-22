#######################################################
# Unit tests for ../proccessing/images
#######################################################

# import general packages for testing
import numpy as np
import pytest

from scportrait.processing.images._image_processing import (
    MinMax,
    _percentile_norm,
    downsample_img_padding,
    percentile_normalization,
    rolling_window_mean,
)

rng = np.random.default_rng(seed=42)


def make_random_image_float(shape):
    return rng.random((10, 10))


def make_random_image_int(shape):
    return rng.integers(2, size=shape)


# @pytest.mark.parametrize(
#     ("shape", "N", "expected_shape"),
#     [
#         ((3, 4, 4), 2, (3, 2, 2)),
#         ((3, 5, 5), 2, (3, 3, 3)),
#         ((5, 5), 2, (3, 3)),
#     ],
# )
# def test_downsample_img_padding(shape, N, expected_shape):
#     img = make_random_image_float(shape)
#     downsampled_img = downsample_img_padding(img, N)
#     assert (
#         downsampled_img.shape == expected_shape
#     ), "Downsampled image shape is {downsampled_img.shape} instead of {expected_shape}."


@pytest.mark.parametrize(
    ("image_shape", "percentile_low", "percentile_high"),
    [
        ((4, 4), 0.1, 0.9),
        ((3, 4, 4), 0.1, 0.9),
    ],
)
def test_percentile_norm(image_shape, percentile_low, percentile_high):
    img = make_random_image_float(image_shape)
    norm_img = _percentile_norm(img, percentile_low, percentile_high)
    assert np.min(norm_img) == pytest.approx(0)
    assert np.max(norm_img) == pytest.approx(1)


def test_percentile_normalization_C_H_W():
    rng = np.random.default_rng()
    test_array = rng.integers(2, size=(3, 100, 100))
    test_array[:, 10:11, 10:11] = -1
    test_array[:, 12:13, 12:13] = 3

    normalized = percentile_normalization(test_array, 0.05, 0.95)
    assert np.max(normalized) == pytest.approx(1)
    assert np.min(normalized) == pytest.approx(0)


def test_percentile_normalization_H_W():
    rng = np.random.default_rng()
    test_array = rng.integers(2, size=(100, 100))
    test_array[10:11, 10:11] = -1
    test_array[12:13, 12:13] = 3

    normalized = percentile_normalization(test_array, 0.05, 0.95)
    assert np.max(normalized) == pytest.approx(1)
    assert np.min(normalized) == pytest.approx(0)


def test_rolling_window_mean():
    rng = np.random.default_rng()
    array = rng.random((10, 10))
    rolling_array = rolling_window_mean(array, size=5, scaling=False)
    assert np.all(array.shape == rolling_array.shape)


def test_MinMax():
    rng = np.random.default_rng()
    array = rng.random((10, 10))
    normalized_array = MinMax(array)
    assert np.min(normalized_array) == 0
    assert np.max(normalized_array) == 1
