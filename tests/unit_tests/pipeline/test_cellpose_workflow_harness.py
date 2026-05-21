from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scportrait.pipeline.segmentation.workflows._cellpose import (
    CytosolOnlySegmentationCellpose,
    CytosolSegmentationCellpose,
    DAPISegmentationCellpose,
)

TEST_HELPER_DIR = Path(__file__).resolve().parent
if str(TEST_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_HELPER_DIR))

from cellpose_test_helpers import (
    DummyFileHandler,
    FakeCellposeModel,
    make_cellpose_workflow,
    make_input_image,
)


@pytest.mark.parametrize(
    ("workflow_cls", "channels", "expected_model_calls", "expected_write_labels"),
    [
        pytest.param(DAPISegmentationCellpose, 1, 1, ["nucleus"], id="dapi"),
        pytest.param(CytosolSegmentationCellpose, 2, 2, ["nucleus", "cytosol"], id="cytosol-paired"),
        pytest.param(CytosolOnlySegmentationCellpose, 2, 1, ["cytosol"], id="cytosol-only"),
    ],
)
def test_cellpose_helpers_smoke_with_fake_model(
    tmp_path, monkeypatch, workflow_cls, channels, expected_model_calls, expected_write_labels
):
    filehandler = DummyFileHandler()
    workflow = make_cellpose_workflow(workflow_cls=workflow_cls, tmp_path=tmp_path, filehandler=filehandler)
    fake_model = FakeCellposeModel()

    monkeypatch.setattr(workflow, "_read_cellpose_model", lambda *args, **kwargs: fake_model)

    input_image = make_input_image(channels=channels, height=14, width=11)
    transformed = workflow._transform_input_image(input_image)

    workflow._execute_segmentation(input_image)

    assert transformed.shape[0] == workflow.N_INPUT_CHANNELS
    assert len(fake_model.eval_calls) == expected_model_calls

    recorded_labels = [call["segmentation_label"] for call in filehandler.write_calls]
    assert recorded_labels == expected_write_labels
    assert len(filehandler.center_calls) == len(expected_write_labels)
