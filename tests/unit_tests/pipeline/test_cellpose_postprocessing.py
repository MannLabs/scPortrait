from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scportrait.pipeline._utils.constants import DEFAULT_SEGMENTATION_DTYPE
from scportrait.pipeline.segmentation.workflows._cellpose import (
    CytosolOnlySegmentationCellpose,
    CytosolSegmentationCellpose,
    DAPISegmentationCellpose,
)

TEST_HELPER_DIR = Path(__file__).resolve().parent
if str(TEST_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_HELPER_DIR))

from cellpose_test_helpers import make_cellpose_workflow, make_input_image


class StaticMaskModel:
    def __init__(self, returned_mask: np.ndarray) -> None:
        self.returned_mask = np.asarray(returned_mask)

    def eval(self, *args, **kwargs):
        return [self.returned_mask.copy()]


def _edge_and_internal_mask(
    height: int = 8, width: int = 8, edge_label: int = 9, internal_label: int = 3
) -> np.ndarray:
    mask = np.zeros((1, height, width), dtype=np.uint32)
    mask[:, 0, :] = edge_label
    mask[:, :, 0] = edge_label
    mask[:, 2:6, 2:6] = internal_label
    return mask


def _internal_mask(height: int = 8, width: int = 8, label: int = 1) -> np.ndarray:
    mask = np.zeros((1, height, width), dtype=np.uint32)
    mask[:, 2:6, 2:6] = label
    return mask


def _set_cpu(workflow) -> None:
    workflow.use_GPU = False
    workflow.device = "cpu"


def _seed_eval_attributes(workflow) -> None:
    workflow.rescale = None
    workflow.resample = True
    workflow.normalize = True
    workflow.diameter = None
    workflow.flow_threshold = 0.4
    workflow.cellprob_threshold = 0.0


def test_dapi_postprocessing_removes_edge_labels_and_finalize_returns_expected_shape_and_dtype(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    _seed_eval_attributes(workflow)
    model = StaticMaskModel(_edge_and_internal_mask())
    input_image = make_input_image(channels=1, height=8, width=8)

    monkeypatch.setattr(workflow, "_check_gpu_status", lambda: _set_cpu(workflow))
    monkeypatch.setattr(workflow, "_load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(workflow, "_clear_cache", lambda vars_to_delete=None: None)

    raw_mask = workflow.cellpose_segmentation(input_image)

    assert raw_mask.ndim == 2
    assert raw_mask.shape == (8, 8)
    assert np.all(raw_mask[0, :] == 0)
    assert np.all(raw_mask[:, 0] == 0)
    assert 0 in np.unique(raw_mask)
    assert 3 in np.unique(raw_mask)
    assert 9 not in np.unique(raw_mask)

    final = workflow._finalize_segmentation_results(raw_mask)
    assert final.shape == (1, 8, 8)
    assert final.dtype == np.dtype(DEFAULT_SEGMENTATION_DTYPE)
    assert np.array_equal(final[0], raw_mask.astype(DEFAULT_SEGMENTATION_DTYPE))


def test_cytosol_finalize_stacks_masks_in_nucleus_then_cytosol_order_and_converts_dtype(tmp_path):
    workflow = make_cellpose_workflow(CytosolSegmentationCellpose, tmp_path=tmp_path)
    nucleus = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    cytosol = np.array(
        [
            [0, 0, 0, 0],
            [0, 5, 5, 0],
            [0, 5, 5, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    final = workflow._finalize_segmentation_results(mask_nucleus=nucleus, mask_cytosol=cytosol)

    assert final.shape == (2, 4, 4)
    assert final.dtype == np.dtype(DEFAULT_SEGMENTATION_DTYPE)
    assert np.array_equal(final[0], nucleus.astype(DEFAULT_SEGMENTATION_DTYPE))
    assert np.array_equal(final[1], cytosol.astype(DEFAULT_SEGMENTATION_DTYPE))


@pytest.mark.parametrize("filter_masks_size", [False, True])
def test_cytosol_size_filtering_orchestration_respects_filter_masks_size_flag(tmp_path, monkeypatch, filter_masks_size):
    workflow = make_cellpose_workflow(
        CytosolSegmentationCellpose,
        tmp_path=tmp_path,
        config={"filter_masks_size": filter_masks_size, "match_masks": False},
    )
    _seed_eval_attributes(workflow)
    input_image = make_input_image(channels=2, height=8, width=8)
    nucleus_model = StaticMaskModel(_internal_mask(label=1))
    cytosol_model = StaticMaskModel(_internal_mask(label=2))
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow, "_check_gpu_status", lambda: _set_cpu(workflow))
    monkeypatch.setattr(
        workflow,
        "_load_model",
        lambda model_type, gpu, device: nucleus_model if model_type == "nucleus" else cytosol_model,
    )
    monkeypatch.setattr(workflow, "_clear_cache", lambda vars_to_delete=None: None)

    def _fake_size_filtering(**kwargs):
        calls.append(kwargs)
        return kwargs["mask"]

    monkeypatch.setattr(workflow, "_perform_size_filtering", _fake_size_filtering)
    workflow.cellpose_segmentation(input_image)

    if filter_masks_size:
        assert len(calls) == 2
        assert [call["mask_name"] for call in calls] == ["nucleus", "cytosol"]
    else:
        assert calls == []


def test_cytosol_size_filtering_uses_mask_specific_thresholds_and_confidence_intervals(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(
        CytosolSegmentationCellpose,
        tmp_path=tmp_path,
        config={
            "filter_masks_size": True,
            "match_masks": False,
            "nucleus_segmentation": {"model": "nuclei", "min_size": 10, "max_size": 20},
            "cytosol_segmentation": {"model": "cyto3", "confidence_interval": 0.88},
        },
    )
    _seed_eval_attributes(workflow)
    input_image = make_input_image(channels=2, height=8, width=8)
    nucleus_model = StaticMaskModel(_internal_mask(label=1))
    cytosol_model = StaticMaskModel(_internal_mask(label=2))
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow, "_check_gpu_status", lambda: _set_cpu(workflow))
    monkeypatch.setattr(
        workflow,
        "_load_model",
        lambda model_type, gpu, device: nucleus_model if model_type == "nucleus" else cytosol_model,
    )
    monkeypatch.setattr(workflow, "_clear_cache", lambda vars_to_delete=None: None)

    def _fake_size_filtering(**kwargs):
        calls.append(kwargs)
        return kwargs["mask"]

    monkeypatch.setattr(workflow, "_perform_size_filtering", _fake_size_filtering)

    workflow.cellpose_segmentation(input_image)

    nucleus_call = next(call for call in calls if call["mask_name"] == "nucleus")
    cytosol_call = next(call for call in calls if call["mask_name"] == "cytosol")

    assert nucleus_call["thresholds"] == [10, 20]
    assert nucleus_call["confidence_interval"] is None
    assert cytosol_call["thresholds"] is None
    assert cytosol_call["confidence_interval"] == 0.88


@pytest.mark.parametrize(
    ("config", "expected_threshold"),
    [
        ({"match_masks": True}, 0.95),
        ({"match_masks": True, "filtering_threshold_mask_matching": 0.73}, 0.73),
    ],
)
def test_cytosol_mask_matching_orchestration_uses_expected_threshold(tmp_path, monkeypatch, config, expected_threshold):
    workflow = make_cellpose_workflow(
        CytosolSegmentationCellpose,
        tmp_path=tmp_path,
        config={"filter_masks_size": False, **config},
    )
    _seed_eval_attributes(workflow)
    input_image = make_input_image(channels=2, height=8, width=8)
    nucleus_model = StaticMaskModel(_internal_mask(label=1))
    cytosol_model = StaticMaskModel(_internal_mask(label=2))
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow, "_check_gpu_status", lambda: _set_cpu(workflow))
    monkeypatch.setattr(
        workflow,
        "_load_model",
        lambda model_type, gpu, device: nucleus_model if model_type == "nucleus" else cytosol_model,
    )
    monkeypatch.setattr(workflow, "_clear_cache", lambda vars_to_delete=None: None)
    monkeypatch.setattr(workflow, "_perform_size_filtering", lambda **kwargs: kwargs["mask"])

    def _fake_match_filtering(**kwargs):
        calls.append(kwargs)
        return kwargs["nucleus_mask"], kwargs["cytosol_mask"]

    monkeypatch.setattr(workflow, "_perform_mask_matching_filtering", _fake_match_filtering)

    workflow.cellpose_segmentation(input_image)

    assert len(calls) == 1
    assert calls[0]["filtering_threshold"] == expected_threshold


def test_cytosol_mask_matching_orchestration_skips_when_disabled(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(
        CytosolSegmentationCellpose,
        tmp_path=tmp_path,
        config={"filter_masks_size": False, "match_masks": False},
    )
    _seed_eval_attributes(workflow)
    input_image = make_input_image(channels=2, height=8, width=8)
    nucleus_model = StaticMaskModel(_internal_mask(label=1))
    cytosol_model = StaticMaskModel(_internal_mask(label=2))
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow, "_check_gpu_status", lambda: _set_cpu(workflow))
    monkeypatch.setattr(
        workflow,
        "_load_model",
        lambda model_type, gpu, device: nucleus_model if model_type == "nucleus" else cytosol_model,
    )
    monkeypatch.setattr(workflow, "_clear_cache", lambda vars_to_delete=None: None)
    monkeypatch.setattr(workflow, "_perform_size_filtering", lambda **kwargs: kwargs["mask"])
    monkeypatch.setattr(workflow, "_perform_mask_matching_filtering", lambda **kwargs: calls.append(kwargs))

    workflow.cellpose_segmentation(input_image)

    assert calls == []


@pytest.mark.parametrize("filter_masks_size", [False, True])
def test_cytosol_only_postprocessing_and_size_filter_orchestration(tmp_path, monkeypatch, filter_masks_size):
    workflow = make_cellpose_workflow(
        CytosolOnlySegmentationCellpose,
        tmp_path=tmp_path,
        config={"filter_masks_size": filter_masks_size},
    )
    _seed_eval_attributes(workflow)
    input_image = make_input_image(channels=2, height=8, width=8)
    model = StaticMaskModel(_internal_mask(label=7))
    size_filter_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow, "_setup_processing", lambda: _set_cpu(workflow))
    monkeypatch.setattr(workflow, "_load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(workflow, "_clear_cache", lambda vars_to_delete=None: None)

    def _fake_size_filtering(**kwargs):
        size_filter_calls.append(kwargs)
        return kwargs["mask"]

    monkeypatch.setattr(workflow, "_perform_size_filtering", _fake_size_filtering)

    raw_mask = workflow.cellpose_segmentation(input_image)

    assert raw_mask.ndim == 2
    assert raw_mask.shape == (8, 8)

    final = workflow._finalize_segmentation_results(raw_mask)
    assert final.shape == (1, 8, 8)
    assert final.dtype == np.dtype(DEFAULT_SEGMENTATION_DTYPE)

    if filter_masks_size:
        assert len(size_filter_calls) == 1
        assert size_filter_calls[0]["mask_name"] == "cytosol"
    else:
        assert size_filter_calls == []


def test_dapi_execute_segmentation_uses_transformed_input_and_saves_expected_masks_and_classes(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    source = make_input_image(channels=1, height=8, width=8)
    transformed = np.full((1, 8, 8), fill_value=5, dtype=np.uint16)
    raw_mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 4, 4, 0],
            [0, 4, 4, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(workflow, "_transform_input_image", lambda image: transformed)
    monkeypatch.setattr(
        workflow,
        "cellpose_segmentation",
        lambda image: raw_mask if image is transformed else np.zeros((4, 4), dtype=np.uint32),
    )

    def _fake_save(segmentation, classes, masks):
        captured["segmentation"] = segmentation
        captured["classes"] = classes
        captured["masks"] = masks

    monkeypatch.setattr(workflow, "_save_segmentation_sdata", _fake_save)

    workflow._execute_segmentation(source)

    assert captured["masks"] == ["nucleus"]
    assert captured["classes"] == {4}
    assert captured["segmentation"].shape == (1, 4, 4)
    assert captured["segmentation"].dtype == np.dtype(DEFAULT_SEGMENTATION_DTYPE)
