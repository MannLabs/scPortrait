from __future__ import annotations

from scportrait.pipeline.segmentation.workflows._base_segmentation_workflow import _BaseSegmentation


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
