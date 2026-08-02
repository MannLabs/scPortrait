from __future__ import annotations

import numpy as np
import pytest

from scportrait.pipeline.segmentation.workflows._base_segmentation_workflow import _BaseSegmentation
from scportrait.pipeline.segmentation.workflows._cellpose import (
    CytosolOnlySegmentationCellpose,
    CytosolSegmentationCellpose,
    DAPISegmentationCellpose,
)


class _DummySegmentationWorkflow(_BaseSegmentation):
    DEFAULT_NUCLEI_CHANNEL_IDS = [0]
    DEFAULT_CYTOSOL_CHANNEL_IDS = [1]
    N_INPUT_CHANNELS = 2
    MASK_NAMES = ["nucleus", "cytosol"]


def _make_workflow(config: dict) -> _DummySegmentationWorkflow:
    workflow = _DummySegmentationWorkflow.__new__(_DummySegmentationWorkflow)
    workflow.config = config
    workflow.maximum_project_nucleus = "combine_nucleus_channels" in config
    workflow.maximum_project_cytosol = "combine_cytosol_channels" in config
    workflow.combine_nucleus_channels = config.get("combine_nucleus_channels")
    workflow.combine_cytosol_channels = config.get("combine_cytosol_channels")
    return workflow


def _make_cellpose_workflow(workflow_cls, config: dict):
    workflow = workflow_cls.__new__(workflow_cls)
    workflow.config = config
    workflow.maximum_project_nucleus = "combine_nucleus_channels" in config
    workflow.maximum_project_cytosol = "combine_cytosol_channels" in config
    workflow.combine_nucleus_channels = config.get("combine_nucleus_channels")
    workflow.combine_cytosol_channels = config.get("combine_cytosol_channels")
    return workflow


def test_define_channels_uses_configured_channel_ids():
    workflow = _make_workflow(
        {
            "segmentation_channel_nuclei": 4,
            "segmentation_channel_cytosol": 7,
        }
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [4, 7]
    assert workflow.original_nucleus_segmentation_channel == [4]
    assert workflow.original_cytosol_segmentation_channel == [7]
    assert workflow.nucleus_segmentation_channel == [0]
    assert workflow.cytosol_segmentation_channel == [1]


def test_define_channels_falls_back_to_defaults():
    workflow = _make_workflow({})

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [0, 1]
    assert workflow.original_nucleus_segmentation_channel == [0]
    assert workflow.original_cytosol_segmentation_channel == [1]
    assert workflow.nucleus_segmentation_channel == [0]
    assert workflow.cytosol_segmentation_channel == [1]


def test_define_channels_accepts_combined_channel_selection():
    workflow = _make_workflow(
        {
            "combine_cytosol_channels": [3, 5],
            "segmentation_channel_nuclei": 8,
        }
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [8, 3, 5]
    assert workflow.original_nucleus_segmentation_channel == [8]
    assert workflow.original_cytosol_segmentation_channel == [3, 5]
    assert workflow.original_combine_cytosol_channels == [3, 5]
    assert workflow.nucleus_segmentation_channel == [0]
    assert workflow.cytosol_segmentation_channel == [1, 2]
    assert workflow.combine_cytosol_channels == [1, 2]


def test_dapi_cellpose_default_nucleus_channel_selected():
    workflow = _make_cellpose_workflow(DAPISegmentationCellpose, {})

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [0]
    assert workflow.original_nucleus_segmentation_channel == [0]
    assert workflow.nucleus_segmentation_channel == [0]


def test_dapi_cellpose_configured_nucleus_channel_respected():
    workflow = _make_cellpose_workflow(DAPISegmentationCellpose, {"segmentation_channel_nuclei": 7})

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [7]
    assert workflow.original_nucleus_segmentation_channel == [7]
    assert workflow.nucleus_segmentation_channel == [0]


def test_dapi_cellpose_more_channels_fail_without_maximum_projection():
    workflow = _make_cellpose_workflow(DAPISegmentationCellpose, {"segmentation_channel_nuclei": [1, 2]})

    with pytest.raises(ValueError, match="More input channels provided than accepted by the segmentation method"):
        workflow._define_channels_to_extract_for_segmentation()


def test_dapi_cellpose_more_channels_allowed_with_maximum_projection():
    workflow = _make_cellpose_workflow(DAPISegmentationCellpose, {"combine_nucleus_channels": [1, 2]})

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [1, 2]
    assert workflow.original_combine_nucleus_channels == [1, 2]
    assert workflow.combine_nucleus_channels == [0, 1]


def test_cytosol_cellpose_default_channels_selected():
    workflow = _make_cellpose_workflow(CytosolSegmentationCellpose, {})

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [0, 1]
    assert workflow.original_nucleus_segmentation_channel == [0]
    assert workflow.original_cytosol_segmentation_channel == [1]
    assert workflow.nucleus_segmentation_channel == [0]
    assert workflow.cytosol_segmentation_channel == [1]


def test_cytosol_cellpose_configured_channels_preserve_order():
    workflow = _make_cellpose_workflow(
        CytosolSegmentationCellpose,
        {"segmentation_channel_nuclei": 9, "segmentation_channel_cytosol": 4},
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [9, 4]
    assert workflow.original_nucleus_segmentation_channel == [9]
    assert workflow.original_cytosol_segmentation_channel == [4]
    assert workflow.nucleus_segmentation_channel == [0]
    assert workflow.cytosol_segmentation_channel == [1]


def test_cytosol_cellpose_duplicate_channels_are_deduplicated_preserving_order():
    workflow = _make_cellpose_workflow(
        CytosolSegmentationCellpose,
        {"segmentation_channel_nuclei": 5, "segmentation_channel_cytosol": [5, 8, 5]},
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [5, 8]
    assert workflow.original_cytosol_segmentation_channel == [5, 8, 5]
    assert workflow.cytosol_segmentation_channel == [0, 1, 0]


def test_cytosol_cellpose_maximum_projection_remaps_to_transformed_indices():
    workflow = _make_cellpose_workflow(
        CytosolSegmentationCellpose,
        {"combine_nucleus_channels": [4, 1], "combine_cytosol_channels": [3, 1]},
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [4, 1, 3]
    assert workflow.original_combine_nucleus_channels == [4, 1]
    assert workflow.original_combine_cytosol_channels == [3, 1]
    assert workflow.combine_nucleus_channels == [0, 1]
    assert workflow.combine_cytosol_channels == [2, 1]
    assert workflow.nucleus_segmentation_channel == [0, 1]
    assert workflow.cytosol_segmentation_channel == [2, 1]


def test_cytosol_only_cellpose_default_channels_are_two_channel_input():
    workflow = _make_cellpose_workflow(CytosolOnlySegmentationCellpose, {})

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [0, 1]
    assert workflow.original_cytosol_segmentation_channel == [0, 1]
    assert workflow.cytosol_segmentation_channel == [0, 1]
    assert workflow.N_INPUT_CHANNELS == 2


def test_cytosol_only_cellpose_one_explicit_channel_is_allowed():
    workflow = _make_cellpose_workflow(CytosolOnlySegmentationCellpose, {"segmentation_channel_cytosol": 6})

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [6]
    assert workflow.original_cytosol_segmentation_channel == [6]
    assert workflow.cytosol_segmentation_channel == [0]
    assert workflow.N_INPUT_CHANNELS == 1


def test_cytosol_only_cellpose_two_explicit_channels_are_allowed():
    workflow = _make_cellpose_workflow(
        CytosolOnlySegmentationCellpose,
        {"segmentation_channel_cytosol": [6, 8]},
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [6, 8]
    assert workflow.original_cytosol_segmentation_channel == [6, 8]
    assert workflow.cytosol_segmentation_channel == [0, 1]
    assert workflow.N_INPUT_CHANNELS == 2


def test_cytosol_only_cellpose_three_explicit_channels_raise_value_error():
    workflow = _make_cellpose_workflow(
        CytosolOnlySegmentationCellpose,
        {"segmentation_channel_cytosol": [1, 2, 3]},
    )

    with pytest.raises(
        ValueError,
        match="CytosolOnlySegmentationCellpose requires 1 or 2 selected channels",
    ):
        workflow._define_channels_to_extract_for_segmentation()


def test_cytosol_only_cellpose_nucleus_none_does_not_add_nucleus_cue():
    workflow = _make_cellpose_workflow(
        CytosolOnlySegmentationCellpose,
        {"segmentation_channel_cytosol": 4, "segmentation_channel_nuclei": None},
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [4]
    assert workflow.original_cytosol_segmentation_channel == [4]
    assert workflow.cytosol_segmentation_channel == [0]
    assert workflow.N_INPUT_CHANNELS == 1


def test_cytosol_only_cellpose_nucleus_string_none_does_not_add_nucleus_cue():
    workflow = _make_cellpose_workflow(
        CytosolOnlySegmentationCellpose,
        {"segmentation_channel_cytosol": 4, "segmentation_channel_nuclei": "none"},
    )

    workflow._define_channels_to_extract_for_segmentation()
    workflow._remap_maximum_intensity_projection_channels()

    assert workflow.segmentation_channels == [4]
    assert workflow.original_cytosol_segmentation_channel == [4]
    assert workflow.cytosol_segmentation_channel == [0]
    assert workflow.N_INPUT_CHANNELS == 1


def test_cytosol_only_cellpose_requires_cytosol_when_nucleus_channel_is_set():
    workflow = _make_cellpose_workflow(CytosolOnlySegmentationCellpose, {"segmentation_channel_nuclei": 2})

    with pytest.raises(
        ValueError,
        match="segmentation_channel_cytosol must be provided when segmentation_channel_nuclei is set",
    ):
        workflow._define_channels_to_extract_for_segmentation()


def test_cytosol_only_cellpose_resolve_cellpose_channels_for_one_channel_input():
    workflow = _make_cellpose_workflow(CytosolOnlySegmentationCellpose, {})
    input_image = np.zeros((1, 5, 5), dtype=np.uint16)

    assert workflow._resolve_cellpose_channels(input_image) == [1, 0]


def test_cytosol_only_cellpose_resolve_cellpose_channels_for_two_channel_input():
    workflow = _make_cellpose_workflow(CytosolOnlySegmentationCellpose, {})
    input_image = np.zeros((2, 5, 5), dtype=np.uint16)

    assert workflow._resolve_cellpose_channels(input_image) == [2, 1]


def test_cytosol_only_cellpose_resolve_cellpose_channels_raises_for_unsupported_channel_count():
    workflow = _make_cellpose_workflow(CytosolOnlySegmentationCellpose, {})
    input_image = np.zeros((3, 5, 5), dtype=np.uint16)

    with pytest.raises(ValueError, match="Unsupported number of channels for Cellpose"):
        workflow._resolve_cellpose_channels(input_image)
