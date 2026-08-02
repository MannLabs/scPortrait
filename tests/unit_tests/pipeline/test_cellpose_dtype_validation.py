from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from scportrait.pipeline.segmentation.workflows._cellpose import DAPISegmentationCellpose

TEST_HELPER_DIR = Path(__file__).resolve().parent
if str(TEST_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_HELPER_DIR))

from cellpose_test_helpers import make_cellpose_workflow


def test_check_input_image_dtype_accepts_uint16(tmp_path):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    input_image = np.zeros((1, 4, 5), dtype=np.uint16)

    workflow._check_input_image_dtype(input_image)


@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_check_input_image_dtype_rejects_non_uint16(tmp_path, dtype):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    input_image = np.zeros((1, 4, 5), dtype=dtype)
    expected_dtype = np.dtype(workflow.DEFAULT_IMAGE_DTYPE)

    with pytest.raises(
        ValueError,
        match=rf"expects input images with dtype {expected_dtype}, got {np.dtype(dtype)}",
    ):
        workflow._check_input_image_dtype(input_image)


def test_check_seg_dtype_returns_mask_unchanged_when_dtype_matches(tmp_path):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    expected_dtype = np.dtype(workflow.DEFAULT_SEGMENTATION_DTYPE)
    mask = np.array([[0, 1], [2, 3]], dtype=expected_dtype)

    returned = workflow._check_seg_dtype(mask=mask, mask_name="nucleus")

    assert returned is mask
    assert returned.dtype == expected_dtype
    assert np.array_equal(returned, mask)


def test_check_seg_dtype_converts_other_integer_dtype_and_warns(tmp_path):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    expected_dtype = np.dtype(workflow.DEFAULT_SEGMENTATION_DTYPE)
    mask = np.array([[0, 1], [2, 3]], dtype=np.int32)

    with pytest.warns(UserWarning, match=rf"expected {expected_dtype}"):
        converted = workflow._check_seg_dtype(mask=mask, mask_name="cytosol")

    assert converted.dtype == expected_dtype
    assert converted.shape == mask.shape
    assert np.array_equal(converted, mask.astype(expected_dtype))
