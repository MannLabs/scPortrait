from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scportrait.pipeline.segmentation.workflows._cellpose_backend import (
    CellposeBackend,
    CellposeEvalParameters,
    CellposeModelSpec,
    CellposeRuntime,
)


def _assert_contains_items(actual: dict[str, Any], expected_subset: dict[str, Any]) -> None:
    for key, value in expected_subset.items():
        assert key in actual
        assert actual[key] == value


def _make_backend(
    *,
    runtime: CellposeRuntime,
    download_model=lambda _name: "/cache/model",
    cellpose_model_ctor=lambda **_kwargs: object(),
    cellpose_ctor=lambda **_kwargs: object(),
) -> CellposeBackend:
    return CellposeBackend(
        runtime=runtime,
        download_model=download_model,
        cellpose_model_ctor=cellpose_model_ctor,
        cellpose_ctor=cellpose_ctor,
    )


def test_runtime_detection_maps_cellpose_3_to_v3():
    backend = CellposeBackend(
        runtime=None,
        download_model=lambda _name: "/cache/model",
        cellpose_model_ctor=lambda **_kwargs: object(),
        cellpose_ctor=lambda **_kwargs: object(),
        cellpose_version_getter=lambda _name: "3.1.1",
    )
    assert backend._runtime == CellposeRuntime.V3


def test_runtime_detection_maps_cellpose_4_to_v4():
    backend = CellposeBackend(
        runtime=None,
        download_model=lambda _name: "/cache/model",
        cellpose_model_ctor=lambda **_kwargs: object(),
        cellpose_ctor=lambda **_kwargs: object(),
        cellpose_version_getter=lambda _name: "4.1.2",
    )
    assert backend._runtime == CellposeRuntime.V4


def test_runtime_detection_raises_for_unsupported_major_version():
    with pytest.raises(RuntimeError, match="Unsupported Cellpose major version '2'"):
        CellposeBackend(
            runtime=None,
            download_model=lambda _name: "/cache/model",
            cellpose_model_ctor=lambda **_kwargs: object(),
            cellpose_ctor=lambda **_kwargs: object(),
            cellpose_version_getter=lambda _name: "2.0.0",
        )


def test_runtime_can_be_injected_without_version_lookup():
    def _unexpected_version_lookup(_name: str) -> str:
        raise AssertionError("version getter should not be called when runtime is injected")

    backend = CellposeBackend(
        runtime=CellposeRuntime.V3,
        download_model=lambda _name: "/cache/model",
        cellpose_model_ctor=lambda **_kwargs: object(),
        cellpose_ctor=lambda **_kwargs: object(),
        cellpose_version_getter=_unexpected_version_lookup,
    )
    assert backend._runtime == CellposeRuntime.V3


@pytest.mark.parametrize("model_name", ["cyto", "cyto2", "cyto3", "nuclei"])
def test_cellpose3_full_builtin_pretrained_models_use_models_cellpose(model_name):
    download_calls: list[str] = []
    cellpose_calls: list[dict[str, Any]] = []
    cellpose_model_calls: list[dict[str, Any]] = []
    fake_model = object()

    backend = _make_backend(
        runtime=CellposeRuntime.V3,
        download_model=lambda name: (download_calls.append(name), f"/cache/{name}")[1],
        cellpose_ctor=lambda **kwargs: (cellpose_calls.append(kwargs), fake_model)[1],
        cellpose_model_ctor=lambda **kwargs: (cellpose_model_calls.append(kwargs), object())[1],
    )

    loaded = backend.load_model(CellposeModelSpec(model_type="pretrained", name=model_name, gpu=False, device="cpu"))

    assert loaded is fake_model
    assert download_calls == [model_name]
    assert cellpose_calls == [{"model_type": model_name, "gpu": False, "device": "cpu"}]
    assert cellpose_model_calls == []


def test_cellpose3_full_builtin_model_raises_clear_error_if_models_cellpose_missing():
    backend = _make_backend(
        runtime=CellposeRuntime.V3,
        download_model=lambda name: f"/cache/{name}",
        cellpose_ctor=None,
        cellpose_model_ctor=lambda **_kwargs: object(),
    )

    with pytest.raises(RuntimeError, match=r"does not expose models\.Cellpose.*'cyto3' requires"):
        backend.load_model(CellposeModelSpec(model_type="pretrained", name="cyto3", gpu=False, device="cpu"))


@pytest.mark.parametrize("model_name", ["tissuenet_cp3", "livecell_cp3", "my_gui_model"])
def test_cellpose3_dataset_specific_pretrained_models_use_cellposemodel_model_type(model_name):
    download_calls: list[str] = []
    cellpose_calls: list[dict[str, Any]] = []
    cellpose_model_calls: list[dict[str, Any]] = []
    fake_model = object()

    backend = _make_backend(
        runtime=CellposeRuntime.V3,
        download_model=lambda name: (download_calls.append(name), f"/cache/{name}")[1],
        cellpose_ctor=lambda **kwargs: (cellpose_calls.append(kwargs), object())[1],
        cellpose_model_ctor=lambda **kwargs: (cellpose_model_calls.append(kwargs), fake_model)[1],
    )

    loaded = backend.load_model(CellposeModelSpec(model_type="pretrained", name=model_name, gpu=False, device="cpu"))

    assert loaded is fake_model
    assert download_calls == []
    assert cellpose_calls == []
    assert cellpose_model_calls == [{"model_type": model_name, "gpu": False, "device": "cpu"}]


def test_cellpose3_custom_model_missing_path_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing_custom_model.cpkt"
    backend = _make_backend(runtime=CellposeRuntime.V3)
    spec = CellposeModelSpec(model_type="custom", name=str(missing_path), gpu=False, device="cpu")

    with pytest.raises(FileNotFoundError, match=r"custom trained model .* does not exist"):
        backend.load_model(spec)


@pytest.mark.parametrize("runtime", [CellposeRuntime.V3, CellposeRuntime.V4])
def test_custom_model_pathlike_is_normalized_and_loaded_via_pretrained_model(runtime, tmp_path):
    model_path = tmp_path / "custom_model.cpkt"
    model_path.write_text("fake model bytes", encoding="utf-8")

    constructor_calls: list[dict[str, Any]] = []
    fake_model = object()

    backend = _make_backend(
        runtime=runtime,
        cellpose_model_ctor=lambda **kwargs: (constructor_calls.append(kwargs), fake_model)[1],
    )

    loaded = backend.load_model(CellposeModelSpec(model_type="custom", name=model_path, gpu=True, device="cuda:0"))  # type: ignore[arg-type]

    assert loaded is fake_model
    assert constructor_calls == [{"pretrained_model": str(model_path), "gpu": True, "device": "cuda:0"}]
    assert isinstance(constructor_calls[0]["pretrained_model"], str)


def test_cellpose4_pretrained_cpsam_uses_downloaded_reference():
    download_calls: list[str] = []
    constructor_calls: list[dict[str, Any]] = []
    fake_model = object()

    backend = _make_backend(
        runtime=CellposeRuntime.V4,
        download_model=lambda name: (download_calls.append(name), f"/cache/{name}")[1],
        cellpose_model_ctor=lambda **kwargs: (constructor_calls.append(kwargs), fake_model)[1],
    )

    loaded = backend.load_model(CellposeModelSpec(model_type="pretrained", name="cpsam", gpu=False, device="cpu"))

    assert loaded is fake_model
    assert download_calls == ["cpsam"]
    assert constructor_calls == [{"pretrained_model": "/cache/cpsam", "gpu": False, "device": "cpu"}]


@pytest.mark.parametrize("legacy_name", ["cyto3", "nuclei"])
def test_cellpose4_legacy_cellpose3_names_raise_explicit_error(legacy_name):
    backend = _make_backend(runtime=CellposeRuntime.V4)

    with pytest.raises(ValueError, match=rf"Model '{legacy_name}' is a legacy Cellpose 3 model name"):
        backend.load_model(CellposeModelSpec(model_type="pretrained", name=legacy_name, gpu=False, device="cpu"))


def test_load_model_pretrained_download_error_has_helpful_message():
    backend = _make_backend(
        runtime=CellposeRuntime.V4,
        download_model=lambda _name: (_ for _ in ()).throw(FileNotFoundError("download failed")),
    )

    with pytest.raises(FileNotFoundError, match="Could not download the requested Cellpose model 'cpsam'"):
        backend.load_model(CellposeModelSpec(model_type="pretrained", name="cpsam", gpu=False, device="cpu"))


def test_load_model_unsupported_spec_type_raises():
    backend = _make_backend(runtime=CellposeRuntime.V4)

    with pytest.raises(ValueError, match="Unsupported Cellpose model type"):
        backend.load_model(CellposeModelSpec(model_type="unknown", name="x", gpu=False, device="cpu"))  # type: ignore[arg-type]


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

    backend = _make_backend(runtime=CellposeRuntime.V4)
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
    _assert_contains_items(
        recorded_calls[0]["kwargs"],
        {
            "rescale": params.rescale,
            "resample": params.resample,
            "normalize": params.normalize,
            "diameter": params.diameter,
            "flow_threshold": params.flow_threshold,
            "cellprob_threshold": params.cellprob_threshold,
            "channels": params.channels,
        },
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4, 5)
    assert np.array_equal(result[0], expected_mask)


def test_extract_masks_from_eval_result_accepts_2d_mask():
    backend = _make_backend(runtime=CellposeRuntime.V4)
    expected_mask = np.arange(12, dtype=np.uint32).reshape(3, 4)

    result = backend._extract_masks_from_eval_result((expected_mask, None, None))

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_mask.shape
    assert np.array_equal(result, expected_mask)


def test_extract_masks_from_eval_result_accepts_3d_mask():
    backend = _make_backend(runtime=CellposeRuntime.V4)
    expected_mask = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)

    result = backend._extract_masks_from_eval_result((expected_mask, None, None))

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_mask.shape
    assert np.array_equal(result, expected_mask)


@pytest.mark.parametrize("empty_result", [(), []])
def test_extract_masks_from_eval_result_rejects_empty_sequence(empty_result):
    backend = _make_backend(runtime=CellposeRuntime.V4)

    with pytest.raises(ValueError, match="non-empty tuple/list"):
        backend._extract_masks_from_eval_result(empty_result)


@pytest.mark.parametrize("invalid_first_element", [None, "mask", object(), 7])
def test_extract_masks_from_eval_result_rejects_non_array_first_element(invalid_first_element):
    backend = _make_backend(runtime=CellposeRuntime.V4)

    with pytest.raises(ValueError, match=r"expected 2D or 3D masks"):
        backend._extract_masks_from_eval_result((invalid_first_element, None, None))


@pytest.mark.parametrize(
    "invalid_mask",
    [np.array([1, 2, 3], dtype=np.uint32), np.zeros((1, 2, 3, 4), dtype=np.uint32)],
)
def test_extract_masks_from_eval_result_rejects_invalid_mask_dimensionality(invalid_mask):
    backend = _make_backend(runtime=CellposeRuntime.V4)

    with pytest.raises(ValueError, match=r"shape") as excinfo:
        backend._extract_masks_from_eval_result((invalid_mask, None, None))
    assert str(invalid_mask.shape) in str(excinfo.value)


@pytest.mark.parametrize(
    ("eval_result", "expected_error"),
    [
        ((), "non-empty tuple/list"),
        ([], "non-empty tuple/list"),
        ((None, None, None), "expected 2D or 3D masks"),
        ((np.array([1, 2, 3], dtype=np.uint32), None, None), "expected 2D or 3D masks"),
    ],
)
def test_eval_rejects_malformed_cellpose_results(eval_result, expected_error):
    class _FakeModel:
        def eval(self, *args, **kwargs):
            return eval_result

    backend = _make_backend(runtime=CellposeRuntime.V4)
    params = CellposeEvalParameters(
        rescale=1.0,
        resample=True,
        normalize=True,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        channels=[1, 0],
    )

    with pytest.raises(ValueError, match=expected_error):
        backend.eval(_FakeModel(), np.zeros((1, 6, 7), dtype=np.uint16), params)


def test_eval_omits_unsupported_kwargs_for_legacy_style_eval_signature():
    recorded_calls: list[dict[str, Any]] = []

    class _FakeModel:
        def eval(self, images, normalize, diameter):
            recorded_calls.append({"images": images, "normalize": normalize, "diameter": diameter})
            return np.ones((6, 7), dtype=np.uint32), None, None

    backend = _make_backend(runtime=CellposeRuntime.V3)
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
