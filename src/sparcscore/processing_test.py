# import general packages for testing
import pytest
import numpy as np
import os

#######################################################
# Unit tests for ../proccessing/segmentation.py
#######################################################

from sparcscore.processing.segmentation import (
    global_otsu,
    _segment_threshold,
    segment_global_threshold,
    segment_local_threshold,
)
from skimage import data  # test datasets for segmentation testing


def test_global_otsu():
    image = data.coins()
    threshold = global_otsu(image)
    assert isinstance(threshold, float), "The result is not a float"


def test_segment_threshold():
    image = data.coins()
    threshold = 100

    # Check if output is a numpy.ndarray
    labels = _segment_threshold(image, threshold)
    assert isinstance(labels, np.ndarray), "The result is not a numpy.ndarray"

    # Check if output has the same shape as the input image
    assert (
        labels.shape == image.shape
    ), "Output labels and input image shapes are not equal"

    # Check if values are non-negative
    assert np.all(labels >= 0), "Output labels contain negative values"


def test_segment_global_threshold():
    image = data.coins()

    # Check if output is a numpy.ndarray
    labels = segment_global_threshold(image)
    assert isinstance(labels, np.ndarray), "The result is not a numpy.ndarray"

    # Check if output has the same shape as the input image
    assert (
        labels.shape == image.shape
    ), "Output labels and input image shapes are not equal"

    # Check if values are non-negative
    assert np.all(labels >= 0), "Output labels contain negative values"


def test_segment_local_threshold():
    # Load sample image
    image = data.coins()

    # Test invalid speckle_kernel parameter
    with pytest.raises(ValueError):
        segment_local_threshold(image, speckle_kernel=0)

    # Test valid parameters
    labels = segment_local_threshold(
        image,
        dilation=4,
        thr=0.01,
        median_block=51,
        min_distance=10,
        peak_footprint=7,
        speckle_kernel=4,
        debug=False,
    )

    # Check the output label array shape and verify it's non-empty
    assert labels.shape == image.shape
    assert labels.max() > 0


from sparcscore.processing.segmentation import _return_edge_labels, shift_labels


def test_return_edge_labels():
    input_map = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    expected_edge_labels = [1, 3]
    edge_labels = _return_edge_labels(input_map)

    assert set(edge_labels) == set(expected_edge_labels)


def test_shift_labels():
    input_map = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    shift = 10
    expected_shifted_map = np.array([[11, 0, 0], [0, 12, 0], [0, 0, 13]])
    expected_edge_labels = [1, 3]

    shifted_map, edge_labels = shift_labels(input_map, shift, remove_edge_labels=False)

    expected_edge_labels_with_shift = np.array(expected_edge_labels) + shift

    shifted_map_with_shift, edge_labels_with_shift = shift_labels(
        input_map, shift, return_shifted_labels=True, remove_edge_labels=False
    )

    assert np.array_equal(shifted_map, expected_shifted_map)
    assert np.array_equal(shifted_map, shifted_map_with_shift)
    assert set(edge_labels) == set(expected_edge_labels)
    assert set(edge_labels_with_shift) == set(expected_edge_labels_with_shift)

    input_map_3d = np.array(
        [[[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[0, 1, 0], [2, 0, 0], [0, 3, 0]]]
    )

    expected_shifted_map_3d = np.array(
        [[[11, 0, 0], [0, 12, 0], [0, 0, 13]], [[0, 11, 0], [12, 0, 0], [0, 13, 0]]]
    )
    expected_edge_labels = [1, 2, 3]

    shifted_map_3d, edge_labels_3d = shift_labels(
        input_map_3d, shift, remove_edge_labels=False
    )

    assert np.array_equal(shifted_map_3d, expected_shifted_map_3d)
    assert set(edge_labels_3d) == set(expected_edge_labels)

    # test if removing edge labels works
    shifted_map_removed_edge_labels, edge_labels = shift_labels(
        input_map, shift, remove_edge_labels=True
    )
    expected_shifted_map_removed_edge_labels = np.array(
        [[0, 0, 0], [0, 12, 0], [0, 0, 0]]
    )
    assert np.array_equal(
        shifted_map_removed_edge_labels, expected_shifted_map_removed_edge_labels
    )


from sparcscore.processing.segmentation import _remove_classes, remove_classes


def test_remove_classes():
    label_in = np.array([[1, 2, 1], [1, 0, 2], [0, 2, 3]])
    to_remove = [1, 3]

    expected_output = np.array([[0, 2, 0], [0, 0, 2], [0, 2, 0]])

    # Test _remove_classes function
    result = _remove_classes(label_in, to_remove)
    assert np.array_equal(result, expected_output)

    # Test remove_classes function
    result = remove_classes(label_in, to_remove)
    assert np.array_equal(result, expected_output)

    # Test reindex parameter
    to_remove_reindex = [1]
    expected_output_reindex = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 2]])
    result_reindex = remove_classes(label_in, to_remove_reindex, reindex=True)
    assert np.array_equal(result_reindex, expected_output_reindex)

    # Test custom background parameter
    background_value = 1
    expected_output_custom_background = np.array([[1, 2, 1], [1, 0, 2], [0, 2, 1]])
    result_custom_background = remove_classes(
        label_in, to_remove, background=background_value
    )

    assert np.array_equal(result_custom_background, expected_output_custom_background)


from sparcscore.processing.segmentation import contact_filter_lambda, contact_filter


def test_contact_filter_lambda():
    label = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    result = contact_filter_lambda(label)
    expected_result = np.array([1.0, 1.0, 0.6])
    assert np.allclose(result, expected_result, atol=1e-6)


def test_contact_filter():
    inarr = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    result = contact_filter(inarr, threshold=0.7)
    expected_result = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    assert np.all(result == expected_result)


from sparcscore.processing.segmentation import _class_size, size_filter


def test_size_filter():
    label = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    limits = [1, 2]
    result = size_filter(label, limits)
    expected_result = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 2]])

    assert np.all(result == expected_result)


def test_class_size():
    mask = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])

    _, length = _class_size(mask)
    expected_length = np.array([float(np.nan), 3, 2])
    assert np.all(
        length[1:] == expected_length[1:]
    )  # only compare [1:] to ignore the nan in the first element
    assert np.isnan(length[0])


from sparcscore.processing.segmentation import _numba_subtract, numba_mask_centroid


def test_numba_subtract():
    array1 = np.array([[0, 2, 3], [0, 5, 6], [0, 0, 7]])
    min_number = 1
    result = _numba_subtract(array1, min_number)
    expected_result = np.array([[0, 1, 2], [0, 4, 5], [0, 0, 6]])
    assert np.all(result == expected_result)


def test_numba_mask_centroid():
    mask = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    centers, points_class, ids = numba_mask_centroid(mask)
    expected_centers = np.array([[0.33333333, 1.66666667], [1.5, 1.5]])
    expected_points_class = np.array([3, 2], dtype=np.uint32)
    expected_ids = np.array([1, 2], dtype=int)

    assert np.allclose(centers, expected_centers, atol=1e-6)
    assert np.all(points_class == expected_points_class)
    assert np.all(ids == expected_ids)


#######################################################
# Unit tests for ../proccessing/preprocessing.py
#######################################################

from sparcscore.processing.preprocessing import (
    _percentile_norm,
    percentile_normalization,
    rolling_window_mean,
    MinMax,
)


def test_percentile_norm():
    img = np.random.rand(4, 4)
    norm_img = _percentile_norm(img, 0.1, 0.9)
    assert np.min(norm_img) == pytest.approx(0)
    assert np.max(norm_img) == pytest.approx(1)


def test_percentile_normalization_C_H_W():
    test_array = np.random.randint(2, size=(3, 100, 100))
    test_array[:, 10:11, 10:11] = -1
    test_array[:, 12:13, 12:13] = 3

    normalized = percentile_normalization(test_array, 0.05, 0.95)
    assert np.max(normalized) == pytest.approx(1)
    assert np.min(normalized) == pytest.approx(0)


def test_percentile_normalization_H_W():
    test_array = np.random.randint(2, size=(100, 100))
    test_array[10:11, 10:11] = -1
    test_array[12:13, 12:13] = 3

    normalized = percentile_normalization(test_array, 0.05, 0.95)
    assert np.max(normalized) == pytest.approx(1)
    assert np.min(normalized) == pytest.approx(0)


def test_rolling_window_mean():
    array = np.random.rand(10, 10)
    rolling_array = rolling_window_mean(array, size=5, scaling=False)
    assert np.all(array.shape == rolling_array.shape)


def test_MinMax():
    array = np.random.rand(10, 10)
    normalized_array = MinMax(array)
    assert np.min(normalized_array) == 0
    assert np.max(normalized_array) == 1


#######################################################
# Unit tests for ../proccessing/utils.py
#######################################################

from sparcscore.processing.utils import (
    plot_image,
    visualize_class,
    download_testimage,
    flatten,
)


def test_flatten():
    nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flatten(nested_list) == expected_output


def test_download_testimage(tmpdir):
    folder = tmpdir.mkdir("test_images")
    downloaded_images = download_testimage(folder)
    assert len(downloaded_images) == 2
    for img_path in downloaded_images:
        assert os.path.isfile(img_path)


def test_visualize_class():
    class_ids = [1, 2]
    seg_map = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 1]])
    background = np.random.random((3, 3)) * 255
    # Since this function does not return anything, we just check if it produces any exceptions
    try:
        visualize_class(class_ids, seg_map, background)
    except Exception as e:
        pytest.fail(f"visualize_class raised exception: {str(e)}")


def test_plot_image(tmpdir):
    array = np.random.rand(10, 10)
    save_name = tmpdir.join("test_plot_image")

    # Since this function does not return anything, we just check if it produces any exceptions
    try:
        plot_image(array, size=(5, 5), save_name=save_name)
    except Exception as e:
        pytest.fail(f"plot_image raised exception: {str(e)}")
    assert os.path.isfile(str(save_name) + ".png")


#######################################################
# Unit tests for ../pipeline/base.py
#######################################################

import tempfile
from sparcscore.pipeline.base import Logable, ProcessingStep


def test_logable_init():
    logable = Logable()
    assert not logable.debug


def test_logable_log():
    with tempfile.TemporaryDirectory() as temp_dir:
        logable = Logable(debug=True)
        logable.directory = temp_dir
        logable.log("Testing")

        log_path = os.path.join(temp_dir, logable.DEFAULT_LOG_NAME)
        assert os.path.isfile(log_path)

        with open(log_path, "r") as f:
            log_content = f.read()
            assert "Testing" in log_content


def test_processing_step_init():
    config = {"setting1": "value1"}
    with tempfile.TemporaryDirectory() as temp_dir:
        processing_step = ProcessingStep(
            config, f"{temp_dir}/test_step", temp_dir, debug=True
        )

        assert processing_step.debug
        assert config == processing_step.config

def test_processing_step_register_parameter():
    config = {"setting1": "value1"}
    with tempfile.TemporaryDirectory() as temp_dir:
        processing_step = ProcessingStep(config, f"{temp_dir}/test_step", temp_dir)

        # Test registering a new parameter
        processing_step.register_parameter("setting2", "value2")
        assert "setting2" in processing_step.config
        assert "value2" == processing_step.config["setting2"]


def test_processing_step_get_directory():
    config = {"setting1": "value1"}
    with tempfile.TemporaryDirectory() as temp_dir:
        processing_step = ProcessingStep(config, f"{temp_dir}/test_step", temp_dir)
        assert f"{temp_dir}/test_step" == processing_step.get_directory()


# general test to check that testing is working
def test_test():
    assert 1 == 1
