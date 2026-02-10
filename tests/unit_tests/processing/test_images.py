#######################################################
# Unit tests for ../proccessing/images
#######################################################

# import general packages for testing
import numpy as np
import pytest

from scportrait.processing.images._image_processing import (
    MinMax,
    _percentile_norm,
    downsample_img,
    downsample_img_padding,
    percentile_normalization,
    rescale_image,
    rolling_window_mean,
    value_range_normalization,
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


def test_rolling_window_mean_scaling_constant():
    array = np.zeros((6, 6), dtype=np.float32)
    rolling_array = rolling_window_mean(array, size=3, scaling=True)
    assert np.all(np.isfinite(rolling_array))
    assert np.max(rolling_array) == 0


def test_MinMax():
    rng = np.random.default_rng()
    array = rng.random((10, 10))
    normalized_array = MinMax(array)
    assert np.min(normalized_array) == 0
    assert np.max(normalized_array) == 1


def test_downsample_img_zero_max():
    img = np.zeros((1, 4, 4), dtype=np.float32)
    downsampled = downsample_img(img, N=2, return_dtype=np.uint16)
    assert downsampled.dtype == np.uint16
    assert np.all(downsampled == 0)


def test_downsample_img_padding_3d_shape():
    img = np.zeros((2, 5, 5), dtype=np.float32)
    downsampled = downsample_img_padding(img, N=2)
    assert downsampled.shape == (2, 3, 3)


def test_rescale_image_invalid_dtype_raises():
    img = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    with pytest.raises(ValueError):
        rescale_image(img, (1, 99), dtype="float32")


def test_rescale_image_return_float_range():
    img = np.array([[0, 10], [20, 30]], dtype=np.uint16)
    out = rescale_image(img, (0, 100), return_float=True)
    assert np.issubdtype(out.dtype, np.floating)
    assert np.min(out) == pytest.approx(0)
    assert np.max(out) == pytest.approx(1)


def test_value_range_normalization_returns_float():
    img = np.array([[0, 5], [10, 15]], dtype=np.float32)
    norm = value_range_normalization(img, 5, 10, return_float=True)
    assert norm.dtype == np.float32
    assert np.min(norm) == pytest.approx(0)
    assert np.max(norm) == pytest.approx(1)


def test_value_range_normalization_uint16_default_for_float_input():
    img = np.array([[0, 5], [10, 15]], dtype=np.float32)
    norm = value_range_normalization(img, 5, 10)
    assert norm.dtype == np.uint16
    assert np.min(norm) == 0
    assert np.max(norm) == np.iinfo(np.uint16).max


def test_value_range_normalization_invalid_range_raises():
    img = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    with pytest.raises(ValueError):
        value_range_normalization(img, 5, 5)


def test_value_range_normalization_preserves_integer_dtype():
    img = np.array([[0, 5], [10, 15]], dtype=np.uint8)
    norm = value_range_normalization(img, 5, 10)
    assert norm.dtype == np.uint8


def test_value_range_normalization_out_dtype():
    img = np.array([[0, 5], [10, 15]], dtype=np.uint16)
    norm = value_range_normalization(img, 5, 10, out_dtype=np.uint8)
    assert norm.dtype == np.uint8
