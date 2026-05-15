from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cellpose import models

from scportrait.pipeline.segmentation.workflows._model_caches import _download_model

if TYPE_CHECKING:
    from collections.abc import Callable


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
        download_model: Callable[[str], str] = _download_model,
        cellpose_model_ctor: Callable[..., object] = models.CellposeModel,
    ) -> None:
        self._download_model = download_model
        self._cellpose_model_ctor = cellpose_model_ctor

    def load_model(self, spec: CellposeModelSpec) -> object:
        if spec.model_type == "pretrained":
            try:
                model_ref = self._download_model(spec.name)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Could not download the requested Cellpose model '{spec.name}'. "
                    "Please check the model name or ensure that the Cellpose model server is available."
                ) from e
            return self._cellpose_model_ctor(pretrained_model=model_ref, gpu=spec.gpu, device=spec.device)

        if spec.model_type == "custom":
            if not Path(spec.name).exists():
                raise FileNotFoundError(
                    f"The file containing the custom trained model {spec.name} does not exist. "
                    "Please provide a valid path."
                )
            return self._cellpose_model_ctor(pretrained_model=spec.name, gpu=spec.gpu, device=spec.device)

        raise ValueError(
            f"Unsupported Cellpose model type '{spec.model_type}'. Expected one of: 'pretrained', 'custom'."
        )

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
        masks = eval_result[0] if isinstance(eval_result, (tuple, list)) else eval_result
        masks_array = np.asarray(masks)
        if masks_array.ndim == 2:
            return masks_array[np.newaxis, ...]
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
