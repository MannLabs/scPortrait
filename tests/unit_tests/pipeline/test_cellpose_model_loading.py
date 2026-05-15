from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import scportrait.pipeline.segmentation.workflows._cellpose as cellpose_workflow_module
from scportrait.pipeline.segmentation.workflows._cellpose import DAPISegmentationCellpose

TEST_HELPER_DIR = Path(__file__).resolve().parent
if str(TEST_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_HELPER_DIR))

from cellpose_test_helpers import make_cellpose_workflow


def test_load_model_pretrained_uses_download_and_cellpose_constructor(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    fake_model = object()

    download_calls: list[str] = []
    constructor_calls: list[dict] = []

    def _fake_download(name: str) -> None:
        download_calls.append(name)

    def _fake_cellpose(*args, **kwargs):
        constructor_calls.append({"args": args, "kwargs": kwargs})
        return fake_model

    monkeypatch.setattr(cellpose_workflow_module, "_download_model", _fake_download)
    monkeypatch.setattr(cellpose_workflow_module.models, "Cellpose", _fake_cellpose)

    loaded_model = workflow._load_model(model_type="nucleus", gpu=False, device="cpu")

    assert loaded_model is fake_model
    assert download_calls == ["nuclei"]
    assert constructor_calls == [{"args": (), "kwargs": {"model_type": "nuclei", "gpu": False, "device": "cpu"}}]


def test_load_model_custom_uses_cellposemodel_and_converts_pathlike_to_str(tmp_path, monkeypatch):
    model_path = tmp_path / "custom_model.cpkt"
    model_path.write_text("fake model bytes", encoding="utf-8")

    workflow = make_cellpose_workflow(
        DAPISegmentationCellpose,
        tmp_path=tmp_path,
        config={"nucleus_segmentation": {"model_path": model_path}},
    )
    workflow.config["nucleus_segmentation"] = {"model_path": model_path}
    fake_model = object()
    constructor_calls: list[dict] = []

    def _fake_download(_name: str) -> None:
        pytest.fail("_download_model should not be called for custom model loading")

    def _fake_cellpose_model(*args, **kwargs):
        constructor_calls.append({"args": args, "kwargs": kwargs})
        return fake_model

    monkeypatch.setattr(cellpose_workflow_module, "_download_model", _fake_download)
    monkeypatch.setattr(cellpose_workflow_module.models, "CellposeModel", _fake_cellpose_model)

    loaded_model = workflow._load_model(model_type="nucleus", gpu=True, device="cuda:0")

    assert loaded_model is fake_model
    assert constructor_calls == [
        {
            "args": (),
            "kwargs": {
                "pretrained_model": str(model_path),
                "gpu": True,
                "device": "cuda:0",
            },
        }
    ]
    assert isinstance(constructor_calls[0]["kwargs"]["pretrained_model"], str)


def test_load_model_custom_missing_path_raises_helpful_error(tmp_path):
    missing_model = tmp_path / "missing_custom_model.cpkt"
    workflow = make_cellpose_workflow(
        DAPISegmentationCellpose,
        tmp_path=tmp_path,
        config={"nucleus_segmentation": {"model_path": missing_model}},
    )
    workflow.config["nucleus_segmentation"] = {"model_path": missing_model}

    with pytest.raises(FileNotFoundError, match=r"custom trained model .* does not exist"):
        workflow._load_model(model_type="nucleus", gpu=False, device="cpu")


def test_load_model_raises_clear_error_when_model_and_model_path_are_missing(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    workflow.config["nucleus_segmentation"] = {}

    def _should_not_run(*args, **kwargs):
        pytest.fail("_read_cellpose_model should not be called when model config is invalid")

    monkeypatch.setattr(workflow, "_read_cellpose_model", _should_not_run)

    with pytest.raises(
        ValueError,
        match=re.escape("No Cellpose model configured for 'nucleus' segmentation."),
    ):
        workflow._load_model(model_type="nucleus", gpu=False, device="cpu")


def test_load_model_assigns_default_parameters_when_optional_values_missing(tmp_path, monkeypatch):
    workflow = make_cellpose_workflow(
        DAPISegmentationCellpose,
        tmp_path=tmp_path,
        config={"nucleus_segmentation": {"model": "nuclei"}},
    )
    monkeypatch.setattr(workflow, "_read_cellpose_model", lambda *args, **kwargs: object())

    workflow._load_model(model_type="nucleus", gpu=False, device="cpu")

    assert workflow.diameter is None
    assert workflow.resample is True
    assert workflow.flow_threshold == 0.4
    assert workflow.cellprob_threshold == 0.0
    assert workflow.normalize is True
    assert workflow.rescale is None


def test_load_model_honors_parameter_overrides_exactly(tmp_path, monkeypatch):
    config = {
        "nucleus_segmentation": {
            "model": "nuclei",
            "diameter": 37,
            "resample": False,
            "flow_threshold": 0.75,
            "cellprob_threshold": -0.1,
            "normalize": False,
            "rescale": 1.5,
        }
    }
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path, config=config)
    monkeypatch.setattr(workflow, "_read_cellpose_model", lambda *args, **kwargs: object())

    workflow._load_model(model_type="nucleus", gpu=False, device="cpu")

    assert workflow.diameter == 37
    assert workflow.resample is False
    assert workflow.flow_threshold == 0.75
    assert workflow.cellprob_threshold == -0.1
    assert workflow.normalize is False
    assert workflow.rescale == 1.5


@pytest.mark.parametrize(
    ("model_type", "model_name"),
    [("nucleus", "nuclei"), ("cytosol", "cyto3")],
)
def test_load_model_writes_cellpose_parameter_file(tmp_path, monkeypatch, model_type, model_name):
    workflow = make_cellpose_workflow(DAPISegmentationCellpose, tmp_path=tmp_path)
    monkeypatch.setattr(workflow, "_read_cellpose_model", lambda *args, **kwargs: object())

    workflow._load_model(model_type=model_type, gpu=False, device="cpu")

    params_file = Path(workflow.directory) / f"cellpose_params_{model_type}.txt"
    assert params_file.exists()

    text = params_file.read_text(encoding="utf-8")
    assert f"Model: {model_name}" in text
    assert f"Model Type: {model_type}" in text
    assert "Diameter: None" in text
    assert "Resample: True" in text
    assert "Flow Threshold: 0.4" in text
    assert "Cellprob Threshold: 0.0" in text
    assert "Normalize: True" in text
    assert "Rescale: None" in text
