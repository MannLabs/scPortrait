from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from inspect import Parameter, signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cellpose import models

from scportrait.pipeline.segmentation.workflows._model_caches import _download_model

if TYPE_CHECKING:
    from collections.abc import Callable


class CellposeRuntime(Enum):
    V3 = "v3"
    V4 = "v4"


_CELLPOSE3_FULL_BUILTIN_MODELS = {"cyto", "cyto2", "cyto3", "nuclei"}


@dataclass(frozen=True)
class CellposeModelSpec:
    model_type: Literal["pretrained", "custom"]
    name: str
    gpu: str | bool
    device: object


@dataclass(frozen=True)
class CellposeEvalParameters:
    rescale: float | None
    resample: bool
    normalize: bool
    diameter: float | int | None
    flow_threshold: float
    cellprob_threshold: float
    channels: list[int]


class CellposeBackend:
    def __init__(
        self,
        *,
        runtime: CellposeRuntime | None = None,
        download_model: Callable[[str], str] = _download_model,
        cellpose_model_ctor: Callable[..., object] = models.CellposeModel,
        cellpose_ctor: Callable[..., object] | None = getattr(models, "Cellpose", None),
        cellpose_version_getter: Callable[[str], str] = package_version,
    ) -> None:
        self._runtime = runtime or self._detect_runtime(cellpose_version_getter)
        self._download_model = download_model
        self._cellpose_model_ctor = cellpose_model_ctor
        self._cellpose_ctor = cellpose_ctor

    @staticmethod
    def _detect_runtime(cellpose_version_getter: Callable[[str], str]) -> CellposeRuntime:
        try:
            installed_version = cellpose_version_getter("cellpose")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Cellpose is not installed. Please install `cellpose` to use Cellpose workflows."
            ) from exc

        major_version_token = installed_version.split(".", 1)[0]
        try:
            major_version = int(major_version_token)
        except ValueError as exc:
            raise RuntimeError(f"Could not parse installed Cellpose version '{installed_version}'.") from exc

        if major_version == 3:
            return CellposeRuntime.V3
        if major_version >= 4:
            return CellposeRuntime.V4

        raise RuntimeError(
            f"Unsupported Cellpose major version '{major_version}' (installed: '{installed_version}'). "
            "Supported major versions are 3 and 4+."
        )

    def _normalize_custom_model_path(self, model_name: str) -> str:
        model_path = Path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"The file containing the custom trained model {model_name} does not exist. "
                "Please provide a valid path."
            )
        return os.fspath(model_name)

    def _resolve_pretrained_model_reference(self, model_name: str) -> str:
        try:
            return os.fspath(self._download_model(model_name))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not download the requested Cellpose model '{model_name}'. "
                "Please check the model name or ensure that the Cellpose model server is available."
            ) from e

    def _load_model_cellpose3(self, spec: CellposeModelSpec) -> object:
        if spec.model_type == "custom":
            model_path = self._normalize_custom_model_path(spec.name)
            return self._cellpose_model_ctor(pretrained_model=model_path, gpu=spec.gpu, device=spec.device)

        if spec.model_type != "pretrained":
            raise ValueError(
                f"Unsupported Cellpose model type '{spec.model_type}'. Expected one of: 'pretrained', 'custom'."
            )

        if spec.name in _CELLPOSE3_FULL_BUILTIN_MODELS:
            self._resolve_pretrained_model_reference(spec.name)
            if self._cellpose_ctor is None:
                raise RuntimeError(
                    "The installed Cellpose runtime does not expose models.Cellpose, "
                    f"but model '{spec.name}' requires the Cellpose 3 full-model API."
                )
            return self._cellpose_ctor(model_type=spec.name, gpu=spec.gpu, device=spec.device)

        return self._cellpose_model_ctor(model_type=spec.name, gpu=spec.gpu, device=spec.device)

    def _load_model_cellpose4(self, spec: CellposeModelSpec) -> object:
        if spec.model_type == "custom":
            model_path = self._normalize_custom_model_path(spec.name)
            return self._cellpose_model_ctor(pretrained_model=model_path, gpu=spec.gpu, device=spec.device)

        if spec.model_type != "pretrained":
            raise ValueError(
                f"Unsupported Cellpose model type '{spec.model_type}'. Expected one of: 'pretrained', 'custom'."
            )

        if spec.name in _CELLPOSE3_FULL_BUILTIN_MODELS:
            raise ValueError(
                f"Model '{spec.name}' is a legacy Cellpose 3 model name. "
                "It is not automatically mapped to Cellpose 4's 'cpsam' model because that can change segmentation "
                "results. Use a Python < 3.13 environment with Cellpose 3, or provide a compatible custom model_path."
            )

        model_ref = self._resolve_pretrained_model_reference(spec.name)
        return self._cellpose_model_ctor(pretrained_model=model_ref, gpu=spec.gpu, device=spec.device)

    def load_model(self, spec: CellposeModelSpec) -> object:
        if self._runtime == CellposeRuntime.V3:
            return self._load_model_cellpose3(spec)
        if self._runtime == CellposeRuntime.V4:
            return self._load_model_cellpose4(spec)
        raise RuntimeError(f"Unsupported Cellpose runtime '{self._runtime}'.")

    def _filter_eval_kwargs_for_model(self, model: Any, eval_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Keep eval compatibility logic at the backend boundary.

        Cellpose 4 still accepts `channels` in the API, but channels are ignored by Cellpose-SAM.
        We still pass the selected channel tuple for channel-aware legacy models and only omit kwargs
        when a custom model's eval signature does not accept them.
        """
        try:
            model_signature = signature(model.eval)
        except (TypeError, ValueError):
            return eval_kwargs

        params = model_signature.parameters.values()
        if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in params):
            return eval_kwargs

        supported = {
            name for name, parameter in model_signature.parameters.items() if parameter.kind != Parameter.VAR_POSITIONAL
        }
        return {name: value for name, value in eval_kwargs.items() if name in supported}

    def _extract_masks(self, eval_result: Any) -> np.ndarray:
        masks_array = self._extract_masks_from_eval_result(eval_result)
        if masks_array.ndim == 2:
            return masks_array[np.newaxis, ...]
        return masks_array

    def _extract_masks_from_eval_result(self, eval_result: Any) -> np.ndarray:
        if isinstance(eval_result, (tuple, list)):
            if len(eval_result) == 0:
                raise ValueError(
                    "Invalid Cellpose eval result: expected a non-empty tuple/list with masks at index 0, "
                    f"got {type(eval_result).__name__} with length 0."
                )
            masks = eval_result[0]
        else:
            masks = eval_result

        try:
            masks_array = np.asarray(masks)
        except Exception as exc:
            raise ValueError(
                "Invalid Cellpose eval masks output: expected array-like masks with shape (H, W) or (N, H, W), "
                f"got value of type {type(masks).__name__}."
            ) from exc

        if masks_array.ndim not in (2, 3):
            raise ValueError(
                "Invalid Cellpose eval masks output: expected 2D or 3D masks with shape (H, W) or (N, H, W), "
                f"got type {type(masks).__name__}, shape {masks_array.shape}, ndim {masks_array.ndim}."
            )

        return masks_array

    def eval(self, model: Any, input_image: np.ndarray, params: CellposeEvalParameters) -> np.ndarray:
        eval_kwargs = {
            "rescale": params.rescale,
            "resample": params.resample,
            "normalize": params.normalize,
            "diameter": params.diameter,
            "flow_threshold": params.flow_threshold,
            "cellprob_threshold": params.cellprob_threshold,
            "channels": params.channels,
        }
        compatible_kwargs = self._filter_eval_kwargs_for_model(model, eval_kwargs)
        result = model.eval([input_image], **compatible_kwargs)
        return self._extract_masks(result)
