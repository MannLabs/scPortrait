from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scportrait.pipeline.segmentation.workflows._cellpose import (
    CytosolOnlySegmentationCellpose,
    CytosolSegmentationCellpose,
    DAPISegmentationCellpose,
)

TEST_HELPER_DIR = Path(__file__).resolve().parent
if str(TEST_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_HELPER_DIR))

from cellpose_test_helpers import make_cellpose_workflow, make_input_image

EXPECTED_EVAL_KWARGS = {
    "rescale",
    "resample",
    "normalize",
    "diameter",
    "flow_threshold",
    "cellprob_threshold",
    "channels",
}


class RecordingFakeModel:
    def __init__(self, name: str, returned_mask: np.ndarray, events: list[tuple[str, Any]] | None = None) -> None:
        self.name = name
        self.returned_mask = np.asarray(returned_mask)
        self.events = events
        self.eval_calls: list[dict[str, Any]] = []

    def eval(self, *args, **kwargs):
        self.eval_calls.append({"args": args, "kwargs": kwargs})
        if self.events is not None:
            self.events.append(("eval", self.name))
        return [self.returned_mask.copy()]


def _configure_eval_parameters(workflow) -> None:
    workflow.rescale = 1.25
    workflow.resample = False
    workflow.normalize = True
    workflow.diameter = 31
    workflow.flow_threshold = 0.65
    workflow.cellprob_threshold = -0.2


def _edge_touching_mask(height: int, width: int, edge_label: int = 7, interior_label: int = 3) -> np.ndarray:
    mask = np.zeros((1, height, width), dtype=np.uint32)
    mask[:, 0, :] = edge_label
    mask[:, :, 0] = edge_label
    mask[:, 2:-2, 2:-2] = interior_label
    return mask


def _interior_mask(height: int, width: int, label: int) -> np.ndarray:
    mask = np.zeros((1, height, width), dtype=np.uint32)
    mask[:, 2:-2, 2:-2] = label
    return mask


def _assert_eval_call_kwargs(call_kwargs: dict[str, Any], expected_channels: list[int], workflow) -> None:
    assert EXPECTED_EVAL_KWARGS.issubset(call_kwargs.keys())
    assert call_kwargs["rescale"] == workflow.rescale
    assert call_kwargs["resample"] == workflow.resample
    assert call_kwargs["normalize"] == workflow.normalize
    assert call_kwargs["diameter"] == workflow.diameter
    assert call_kwargs["flow_threshold"] == workflow.flow_threshold
    assert call_kwargs["cellprob_threshold"] == workflow.cellprob_threshold
    assert call_kwargs["channels"] == expected_channels


def test_dapi_eval_contract_and_edge_label_cleanup(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    _configure_eval_parameters(workflow)

    input_image = make_input_image(channels=1, height=8, width=8)
    fake_model = RecordingFakeModel("nucleus", _edge_touching_mask(height=8, width=8))

    monkeypatch.setattr(workflow, "_load_model", lambda *args, **kwargs: fake_model)

    def _fake_check_gpu_status() -> None:
        workflow.use_GPU = False
        workflow.device = "cpu"

    monkeypatch.setattr(workflow, "_check_gpu_status", _fake_check_gpu_status)

    mask = workflow.cellpose_segmentation(input_image)

    assert len(fake_model.eval_calls) == 1
    eval_call = fake_model.eval_calls[0]
    assert len(eval_call["args"]) == 1
    assert isinstance(eval_call["args"][0], list)
    assert len(eval_call["args"][0]) == 1
    assert eval_call["args"][0][0] is input_image
    _assert_eval_call_kwargs(eval_call["kwargs"], expected_channels=[1, 0], workflow=workflow)

    assert mask.ndim == 2
    assert mask.shape == (8, 8)
    assert np.all(mask[0, :] == 0)
    assert np.all(mask[:, 0] == 0)
    assert 7 not in np.unique(mask)
    assert 3 in np.unique(mask)


def test_cytosol_eval_contract_load_order_channel_mapping_and_cache_cleanup(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(CytosolSegmentationCellpose, tmp_path=tmp_path, config={"match_masks": False})
    _configure_eval_parameters(workflow)
    input_image = make_input_image(channels=2, height=8, width=8)

    load_order: list[str] = []
    clear_calls: list[Any] = []
    eval_order: list[str] = []
    nucleus_model = RecordingFakeModel("nucleus", _interior_mask(height=8, width=8, label=1), events=[])
    cytosol_model = RecordingFakeModel("cytosol", _interior_mask(height=8, width=8, label=2), events=[])

    def _recording_eval(model: RecordingFakeModel):
        original_eval = model.eval

        def _wrapped(*args, **kwargs):
            eval_order.append(model.name)
            return original_eval(*args, **kwargs)

        model.eval = _wrapped  # type: ignore[assignment]

    _recording_eval(nucleus_model)
    _recording_eval(cytosol_model)

    def _fake_load_model(model_type: str, gpu, device):
        load_order.append(model_type)
        return nucleus_model if model_type == "nucleus" else cytosol_model

    def _fake_check_gpu_status() -> None:
        workflow.use_GPU = False
        workflow.device = "cpu"

    def _fake_clear_cache(vars_to_delete=None) -> None:
        clear_calls.append(None if vars_to_delete is None else vars_to_delete[0])

    monkeypatch.setattr(workflow, "_load_model", _fake_load_model)
    monkeypatch.setattr(workflow, "_check_gpu_status", _fake_check_gpu_status)
    monkeypatch.setattr(workflow, "_clear_cache", _fake_clear_cache)

    masks_nucleus, masks_cytosol = workflow.cellpose_segmentation(input_image)

    assert len(nucleus_model.eval_calls) == 1
    assert len(cytosol_model.eval_calls) == 1
    _assert_eval_call_kwargs(nucleus_model.eval_calls[0]["kwargs"], expected_channels=[1, 0], workflow=workflow)
    _assert_eval_call_kwargs(cytosol_model.eval_calls[0]["kwargs"], expected_channels=[2, 1], workflow=workflow)

    assert load_order == ["nucleus", "cytosol"]
    assert eval_order == ["nucleus", "cytosol"]
    assert clear_calls.count(None) >= 1
    assert nucleus_model in clear_calls
    assert cytosol_model in clear_calls

    assert masks_nucleus.ndim == 2
    assert masks_cytosol.ndim == 2
    assert masks_nucleus.shape == (8, 8)
    assert masks_cytosol.shape == (8, 8)


@pytest.mark.parametrize(
    ("config", "expected_channels", "expected_input_channels"),
    [
        ({"segmentation_channel_cytosol": 0}, [1, 0], 1),
        ({}, [2, 1], 2),
    ],
)
def test_cytosol_only_eval_contract_tracks_selected_channel_count(
    tmp_path, monkeypatch, config, expected_channels, expected_input_channels
):
    workflow = make_cellpose_workflow(CytosolOnlySegmentationCellpose, tmp_path=tmp_path, config=config)
    _configure_eval_parameters(workflow)

    source_image = make_input_image(channels=2, height=8, width=8)
    input_image = workflow._transform_input_image(source_image)
    assert input_image.shape[0] == expected_input_channels

    fake_model = RecordingFakeModel("cytosol", _interior_mask(height=8, width=8, label=5))

    def _fake_setup_processing() -> None:
        workflow.use_GPU = False
        workflow.device = "cpu"

    monkeypatch.setattr(workflow, "_setup_processing", _fake_setup_processing)
    monkeypatch.setattr(workflow, "_load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(workflow, "_clear_cache", lambda vars_to_delete=None: None)

    mask = workflow.cellpose_segmentation(input_image)

    assert len(fake_model.eval_calls) == 1
    _assert_eval_call_kwargs(fake_model.eval_calls[0]["kwargs"], expected_channels=expected_channels, workflow=workflow)
    assert mask.ndim == 2
    assert mask.shape == (8, 8)
