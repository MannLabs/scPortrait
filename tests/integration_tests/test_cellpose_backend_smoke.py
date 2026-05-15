from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scportrait.pipeline.segmentation.workflows._cellpose_backend import (
    CellposeBackend,
    CellposeEvalParameters,
    CellposeModelSpec,
)

RUN_CELLPOSE_INTEGRATION_ENV = "SCPORTRAIT_RUN_CELLPOSE_INTEGRATION"


@pytest.mark.integration
@pytest.mark.cellpose
def test_cellpose_backend_smoke_uses_real_cellpose_dependency():
    if os.getenv(RUN_CELLPOSE_INTEGRATION_ENV) != "1":
        pytest.skip(f"Set {RUN_CELLPOSE_INTEGRATION_ENV}=1 to run optional Cellpose integration smoke tests.")
    cellpose = pytest.importorskip("cellpose", reason="Cellpose is not installed in this test environment.")
    models = cellpose.models
    torch = pytest.importorskip("torch", reason="PyTorch is required by Cellpose.")

    backend = CellposeBackend()
    cpsam_path = Path(getattr(models, "MODEL_DIR", Path.home() / ".cellpose" / "models")) / "cpsam"
    if not cpsam_path.exists():
        pytest.skip(
            "Cellpose CP4 model cache is missing. Pre-cache the 'cpsam' model before running this optional smoke test."
        )

    model_spec = CellposeModelSpec(model_type="custom", name=str(cpsam_path), gpu=False, device=torch.device("cpu"))

    try:
        model = backend.load_model(model_spec)
    except FileNotFoundError as exc:
        pytest.skip(f"Cellpose model could not be resolved for smoke test: {exc}")

    input_image = np.zeros((1, 32, 32), dtype=np.uint16)
    input_image[0, 10:22, 10:22] = 50000

    eval_params = CellposeEvalParameters(
        rescale=None,
        resample=True,
        normalize=True,
        diameter=20,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        channels=[1, 0],
    )

    mask = backend.eval(model, input_image, eval_params)
    mask_array = np.asarray(mask)

    assert mask_array.ndim == 3
    assert mask_array.shape[0] == 1
    assert mask_array.shape[1:] == input_image.shape[1:]
    assert np.issubdtype(mask_array.dtype, np.number)
