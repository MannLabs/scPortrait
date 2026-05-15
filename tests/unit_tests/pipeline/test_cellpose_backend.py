from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scportrait.pipeline.segmentation.workflows._cellpose_backend import (
    CellposeBackend,
    CellposeEvalParameters,
    CellposeModelSpec,
)


def test_load_model_pretrained_delegates_download_and_cellpose_model_constructor():
    download_calls: list[str] = []
    constructor_calls: list[dict[str, Any]] = []
    fake_model = object()

    def _fake_download(name: str) -> str:
        download_calls.append(name)
        return f"/cache/{name}"

    def _fake_cellpose_model(*args, **kwargs):
        constructor_calls.append({"args": args, "kwargs": kwargs})
        return fake_model

    backend = CellposeBackend(
        download_model=_fake_download,
        cellpose_model_ctor=_fake_cellpose_model,
    )
    spec = CellposeModelSpec(model_type="pretrained", name="nuclei", gpu=False, device="cpu")

    loaded = backend.load_model(spec)

    assert loaded is fake_model
    assert download_calls == ["nuclei"]
    assert constructor_calls == [
        {"args": (), "kwargs": {"pretrained_model": "/cache/nuclei", "gpu": False, "device": "cpu"}}
    ]


def test_load_model_custom_missing_path_raises_helpful_error(tmp_path):
    missing_path = tmp_path / "missing_custom_model.cpkt"
    backend = CellposeBackend(
        download_model=lambda *args, **kwargs: "",
        cellpose_model_ctor=lambda *args, **kwargs: None,
    )
    spec = CellposeModelSpec(model_type="custom", name=str(missing_path), gpu=False, device="cpu")

    with pytest.raises(FileNotFoundError, match=r"custom trained model .* does not exist"):
        backend.load_model(spec)


def test_load_model_custom_instantiates_cellpose_model_for_existing_path(tmp_path):
    model_path = tmp_path / "custom_model.cpkt"
    model_path.write_text("fake model bytes", encoding="utf-8")

    constructor_calls: list[dict[str, Any]] = []
    fake_model = object()

    def _fake_cellpose_model(*args, **kwargs):
        constructor_calls.append({"args": args, "kwargs": kwargs})
        return fake_model

    backend = CellposeBackend(
        download_model=lambda *args, **kwargs: "",
        cellpose_model_ctor=_fake_cellpose_model,
    )
    spec = CellposeModelSpec(model_type="custom", name=str(model_path), gpu=True, device="cuda:0")

    loaded = backend.load_model(spec)

    assert loaded is fake_model
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


def test_eval_returns_masks_and_forwards_expected_kwargs():
    input_image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    expected_mask = np.arange(4 * 5, dtype=np.uint32).reshape(4, 5)
    flows = np.zeros((4, 5), dtype=np.float32)
    styles = np.zeros((256,), dtype=np.float32)

    recorded_calls: list[dict[str, Any]] = []

    class _FakeModel:
        def eval(self, *args, **kwargs):
            recorded_calls.append({"args": args, "kwargs": kwargs})
            return expected_mask, flows, styles

    backend = CellposeBackend(
        download_model=lambda *args, **kwargs: "",
        cellpose_model_ctor=lambda *args, **kwargs: None,
    )
    params = CellposeEvalParameters(
        rescale=1.5,
        resample=False,
        normalize=True,
        diameter=30,
        flow_threshold=0.6,
        cellprob_threshold=-0.2,
        channels=[2, 1],
    )

    result = backend.eval(_FakeModel(), input_image, params)

    assert len(recorded_calls) == 1
    assert len(recorded_calls[0]["args"]) == 1
    assert isinstance(recorded_calls[0]["args"][0], list)
    assert recorded_calls[0]["args"][0][0] is input_image
    assert recorded_calls[0]["kwargs"] == {
        "rescale": params.rescale,
        "resample": params.resample,
        "normalize": params.normalize,
        "diameter": params.diameter,
        "flow_threshold": params.flow_threshold,
        "cellprob_threshold": params.cellprob_threshold,
        "channels": params.channels,
    }
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4, 5)
    assert np.array_equal(result[0], expected_mask)


def test_eval_omits_unsupported_legacy_kwargs():
    recorded_calls: list[dict[str, Any]] = []

    class _FakeModel:
        def eval(self, images, normalize, diameter):
            recorded_calls.append({"images": images, "normalize": normalize, "diameter": diameter})
            return np.ones((6, 7), dtype=np.uint32), None, None

    backend = CellposeBackend(
        download_model=lambda *args, **kwargs: "", cellpose_model_ctor=lambda *args, **kwargs: None
    )
    params = CellposeEvalParameters(
        rescale=1.0,
        resample=True,
        normalize=False,
        diameter=27,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        channels=[1, 0],
    )

    result = backend.eval(_FakeModel(), np.zeros((1, 6, 7), dtype=np.uint16), params)

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["normalize"] is False
    assert recorded_calls[0]["diameter"] == 27
    assert result.shape == (1, 6, 7)
