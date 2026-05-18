from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

RUN_CELLPOSE_INTEGRATION_ENV = "SCPORTRAIT_RUN_CELLPOSE_INTEGRATION"
ALLOW_CELLPOSE_MODEL_DOWNLOAD_ENV = "SCPORTRAIT_ALLOW_CELLPOSE_MODEL_DOWNLOAD"
CUSTOM_MODEL_PATH_ENV = "SCPORTRAIT_CELLPOSE_CUSTOM_MODEL_PATH"
LEGACY_PRETRAINED_NAME = "nuclei"
CP4_PRETRAINED_NAME = "cpsam"


def _require_cellpose_integration_opt_in() -> None:
    if os.getenv(RUN_CELLPOSE_INTEGRATION_ENV) != "1":
        pytest.skip(f"Set {RUN_CELLPOSE_INTEGRATION_ENV}=1 to run optional Cellpose integration smoke tests.")


def _allow_model_download() -> bool:
    return os.getenv(ALLOW_CELLPOSE_MODEL_DOWNLOAD_ENV) == "1"


def _import_cellpose_runtime():
    pytest.importorskip("cellpose", reason="Cellpose is not installed in this test environment.")
    torch = pytest.importorskip("torch", reason="PyTorch is required by Cellpose.")
    from cellpose import models

    from scportrait.pipeline.segmentation.workflows import _model_caches
    from scportrait.pipeline.segmentation.workflows._cellpose_backend import (
        CellposeBackend,
        CellposeEvalParameters,
        CellposeModelSpec,
    )

    return torch, models, _model_caches, CellposeBackend, CellposeEvalParameters, CellposeModelSpec


def _tiny_input_image() -> np.ndarray:
    image = np.zeros((1, 32, 32), dtype=np.uint16)
    image[0, 10:22, 10:22] = 50000
    return image


def _default_eval_params(cellpose_eval_parameters_cls):
    return cellpose_eval_parameters_cls(
        rescale=None,
        resample=True,
        normalize=True,
        diameter=20,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        channels=[1, 0],
    )


def _assert_structural_mask_output(mask: np.ndarray, input_image: np.ndarray) -> None:
    mask_array = np.asarray(mask)

    assert isinstance(mask_array, np.ndarray)
    assert mask_array.ndim in (2, 3)
    if mask_array.ndim == 2:
        assert mask_array.shape == input_image.shape[1:]
    else:
        assert mask_array.shape[0] >= 1
        assert mask_array.shape[1:] == input_image.shape[1:]

    # Labels are expected to be integer-like, but we also accept numeric dtypes
    # compatible with downstream conversion checks.
    assert np.issubdtype(mask_array.dtype, np.integer) or np.issubdtype(mask_array.dtype, np.number)


def _legacy_cached_files(model_dir: Path, model_name: str) -> list[Path]:
    if model_name == "nuclei":
        return [model_dir / "nucleitorch_0", model_dir / "size_nucleitorch_0.npy"]
    if model_name == "cyto3":
        return [model_dir / "cyto3", model_dir / "size_cyto3.npy"]
    return [model_dir / model_name]


def _has_local_pretrained_cache(model_dir: Path, model_name: str) -> bool:
    if model_name == CP4_PRETRAINED_NAME:
        return (model_dir / CP4_PRETRAINED_NAME).exists()
    required = _legacy_cached_files(model_dir, model_name)
    return all(path.exists() for path in required)


def _resolve_custom_model_path(model_dir: Path, model_caches_module, allow_download: bool) -> Path:
    explicit_path = os.getenv(CUSTOM_MODEL_PATH_ENV)
    if explicit_path:
        resolved = Path(explicit_path)
        if not resolved.exists():
            pytest.skip(
                f"Custom model path from {CUSTOM_MODEL_PATH_ENV} does not exist: {resolved}. "
                "Provide a valid local path to run the custom model integration smoke test."
            )
        return resolved

    cached_cpsam = model_dir / CP4_PRETRAINED_NAME
    if cached_cpsam.exists():
        return cached_cpsam

    if not allow_download:
        pytest.skip(
            "Custom model smoke test requires a local model file path. "
            f"Set {CUSTOM_MODEL_PATH_ENV} to an existing path, pre-cache '{CP4_PRETRAINED_NAME}', "
            f"or set {ALLOW_CELLPOSE_MODEL_DOWNLOAD_ENV}=1 to allow one-time model cache download."
        )

    try:
        resolved = Path(model_caches_module._download_model(CP4_PRETRAINED_NAME))
    except FileNotFoundError as exc:
        pytest.fail(
            "Opt-in Cellpose integration failed while resolving a cacheable custom model path "
            f"for '{CP4_PRETRAINED_NAME}': {exc}"
        )

    if not resolved.exists():
        pytest.fail(f"Opt-in Cellpose integration resolved a custom model path but the file does not exist: {resolved}")
    return resolved


def _load_pretrained_model_for_smoke(
    backend, model_dir: Path, model_name: str, model_spec_cls, device, allow_download: bool
):
    if not allow_download and not _has_local_pretrained_cache(model_dir, model_name):
        pytest.skip(
            f"Pretrained model '{model_name}' is not available in local Cellpose cache ({model_dir}) and "
            f"{ALLOW_CELLPOSE_MODEL_DOWNLOAD_ENV} is not set to 1."
        )

    spec = model_spec_cls(model_type="pretrained", name=model_name, gpu=False, device=device)
    try:
        return backend.load_model(spec)
    except FileNotFoundError as exc:
        pytest.fail(
            f"Pretrained Cellpose integration load failed for '{model_name}' in opt-in mode "
            f"({RUN_CELLPOSE_INTEGRATION_ENV}=1): {exc}"
        )
    except ValueError as exc:
        if model_name == LEGACY_PRETRAINED_NAME and "does not appear to be a CP4 model" in str(exc):
            pytest.skip(
                f"Pretrained legacy model '{model_name}' resolved from local cache ({model_dir}) but is not "
                "compatible with the installed Cellpose 4 runtime."
            )
        raise


@pytest.mark.integration
@pytest.mark.cellpose
def test_cellpose_backend_smoke_custom_pre_cached_model_path():
    """
    Manual smoke test for real Cellpose custom model loading path.

    Run explicitly:
        SCPORTRAIT_RUN_CELLPOSE_INTEGRATION=1 pytest tests -m "cellpose or integration" -q
    """
    _require_cellpose_integration_opt_in()
    torch, models, model_caches, cellpose_backend_cls, eval_params_cls, model_spec_cls = _import_cellpose_runtime()

    model_dir = Path(getattr(models, "MODEL_DIR", Path.home() / ".cellpose" / "models"))
    model_path = _resolve_custom_model_path(
        model_dir=model_dir, model_caches_module=model_caches, allow_download=_allow_model_download()
    )

    backend = cellpose_backend_cls()
    model_spec = model_spec_cls(model_type="custom", name=str(model_path), gpu=False, device=torch.device("cpu"))
    model = backend.load_model(model_spec)

    input_image = _tiny_input_image()
    eval_params = _default_eval_params(eval_params_cls)
    mask = backend.eval(model, input_image, eval_params)

    _assert_structural_mask_output(mask, input_image)


@pytest.mark.integration
@pytest.mark.cellpose
@pytest.mark.parametrize("pretrained_name", [LEGACY_PRETRAINED_NAME, CP4_PRETRAINED_NAME])
def test_cellpose_backend_smoke_pretrained_name_loading(pretrained_name: str):
    """
    Manual smoke test for real Cellpose pretrained-name loading path.

    Run explicitly:
        SCPORTRAIT_RUN_CELLPOSE_INTEGRATION=1 pytest tests -m "cellpose or integration" -q
    """
    _require_cellpose_integration_opt_in()
    torch, models, _, cellpose_backend_cls, eval_params_cls, model_spec_cls = _import_cellpose_runtime()

    model_dir = Path(getattr(models, "MODEL_DIR", Path.home() / ".cellpose" / "models"))
    backend = cellpose_backend_cls()
    model = _load_pretrained_model_for_smoke(
        backend=backend,
        model_dir=model_dir,
        model_name=pretrained_name,
        model_spec_cls=model_spec_cls,
        device=torch.device("cpu"),
        allow_download=_allow_model_download(),
    )

    input_image = _tiny_input_image()
    eval_params = _default_eval_params(eval_params_cls)
    mask = backend.eval(model, input_image, eval_params)

    _assert_structural_mask_output(mask, input_image)
