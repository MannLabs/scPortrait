from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scportrait.pipeline.segmentation.workflows._cellpose_backend import (
    CellposeBackend,
    CellposeEvalParameters,
    CellposeModelSpec,
)


def test_load_model_pretrained_delegates_download_and_cellpose_constructor():
    download_calls: list[str] = []
    constructor_calls: list[dict[str, Any]] = []
    fake_model = object()

    def _fake_download(name: str) -> None:
        download_calls.append(name)

    def _fake_cellpose(*args, **kwargs):
        constructor_calls.append({"args": args, "kwargs": kwargs})
        return fake_model

    backend = CellposeBackend(
        download_model=_fake_download,
        cellpose_ctor=_fake_cellpose,
        cellpose_model_ctor=lambda *args, **kwargs: None,
    )
    spec = CellposeModelSpec(model_type="pretrained", name="nuclei", gpu=False, device="cpu")

    loaded = backend.load_model(spec)

    assert loaded is fake_model
    assert download_calls == ["nuclei"]
    assert constructor_calls == [{"args": (), "kwargs": {"model_type": "nuclei", "gpu": False, "device": "cpu"}}]


def test_load_model_custom_missing_path_raises_helpful_error(tmp_path):
    missing_path = tmp_path / "missing_custom_model.cpkt"
    backend = CellposeBackend(
        download_model=lambda *args, **kwargs: None,
        cellpose_ctor=lambda *args, **kwargs: None,
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
        download_model=lambda *args, **kwargs: None,
        cellpose_ctor=lambda *args, **kwargs: None,
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


def test_eval_returns_first_mask_and_forwards_expected_kwargs():
    input_image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    expected_mask = np.arange(4 * 5, dtype=np.uint32).reshape(1, 4, 5)
    extra_mask = np.zeros((1, 4, 5), dtype=np.uint32)

    recorded_calls: list[dict[str, Any]] = []

    class _FakeModel:
        def eval(self, *args, **kwargs):
            recorded_calls.append({"args": args, "kwargs": kwargs})
            return [expected_mask, extra_mask]

    backend = CellposeBackend(
        download_model=lambda *args, **kwargs: None,
        cellpose_ctor=lambda *args, **kwargs: None,
        cellpose_model_ctor=lambda *args, **kwargs: None,
    )
    params = CellposeEvalParameters(
        rescale=1.5,
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
        "normalize": params.normalize,
        "diameter": params.diameter,
        "flow_threshold": params.flow_threshold,
        "cellprob_threshold": params.cellprob_threshold,
        "channels": params.channels,
    }
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected_mask)
    assert not np.array_equal(result, extra_mask)
