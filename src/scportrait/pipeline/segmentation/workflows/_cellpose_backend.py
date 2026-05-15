from __future__ import annotations

from dataclasses import dataclass
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
    normalize: bool
    diameter: float | int | None
    flow_threshold: float
    cellprob_threshold: float
    channels: list[int]


class CellposeBackend:
    def __init__(
        self,
        *,
        download_model: Callable[[str], None] = _download_model,
        cellpose_ctor: Callable[..., object] = models.Cellpose,
        cellpose_model_ctor: Callable[..., object] = models.CellposeModel,
    ) -> None:
        self._download_model = download_model
        self._cellpose_ctor = cellpose_ctor
        self._cellpose_model_ctor = cellpose_model_ctor

    def load_model(self, spec: CellposeModelSpec) -> object:
        if spec.model_type == "pretrained":
            try:
                self._download_model(spec.name)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Could not download the requested Cellpose model '{spec.name}'. "
                    "Please check the model name or ensure that the Cellpose model server is available."
                ) from e
            return self._cellpose_ctor(model_type=spec.name, gpu=spec.gpu, device=spec.device)

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

    def eval(self, model: Any, input_image: np.ndarray, params: CellposeEvalParameters) -> np.ndarray:
        result = model.eval(
            [input_image],
            rescale=params.rescale,
            normalize=params.normalize,
            diameter=params.diameter,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
            channels=params.channels,
        )
        return np.array(result[0])
