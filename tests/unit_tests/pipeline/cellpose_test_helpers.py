from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


class DummyFileHandler:
    """Minimal in-memory stand-in for segmentation output writing."""

    def __init__(self) -> None:
        self.write_calls: list[dict[str, Any]] = []
        self.center_calls: list[dict[str, Any]] = []

    def _write_segmentation_sdata(self, labels, segmentation_label: str, overwrite: bool = False, **kwargs) -> None:
        self.write_calls.append(
            {
                "labels": np.array(labels),
                "segmentation_label": segmentation_label,
                "overwrite": overwrite,
                "kwargs": kwargs,
            }
        )

    def _add_centers(self, segmentation_label: str, overwrite: bool = False, **kwargs) -> None:
        self.center_calls.append(
            {
                "segmentation_label": segmentation_label,
                "overwrite": overwrite,
                "kwargs": kwargs,
            }
        )


class FakeCellposeModel:
    """Fake Cellpose model that records eval calls and returns deterministic masks."""

    def __init__(self) -> None:
        self.eval_calls: list[dict[str, Any]] = []

    def eval(self, images, *args, **kwargs):
        self.eval_calls.append({"images": images, "args": args, "kwargs": kwargs})

        input_image = np.asarray(images[0])
        height, width = int(input_image.shape[-2]), int(input_image.shape[-1])

        mask = np.zeros((1, height, width), dtype=np.uint32)
        y0, y1 = max(1, height // 4), min(height - 1, max(2, (3 * height) // 4))
        x0, x1 = max(1, width // 4), min(width - 1, max(2, (3 * width) // 4))
        label_value = len(self.eval_calls)
        mask[:, y0:y1, x0:x1] = label_value

        return [mask]


def make_input_image(channels: int = 2, height: int = 12, width: int = 10) -> np.ndarray:
    """Create a deterministic uint16 image shaped as (C, H, W)."""
    if channels < 1:
        raise ValueError("channels must be >= 1")
    values = np.arange(channels * height * width, dtype=np.uint16).reshape(channels, height, width)
    return values


def _deep_update(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return base
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def make_cellpose_workflow(
    workflow_cls,
    tmp_path: Path,
    config: dict[str, Any] | None = None,
    debug: bool = False,
    nuc_seg_name: str = "nucleus",
    cyto_seg_name: str = "cytosol",
    filehandler: DummyFileHandler | None = None,
):
    """Instantiate a Cellpose workflow with realistic lightweight test arguments."""
    default_config = {
        "cache": str(tmp_path / "cache"),
        "nucleus_segmentation": {"model": "nuclei"},
        "cytosol_segmentation": {"model": "cyto3"},
    }
    merged_config = _deep_update(deepcopy(default_config), config)
    if filehandler is None:
        filehandler = DummyFileHandler()

    workflow = workflow_cls(
        config=merged_config,
        directory=str(tmp_path / "segmentation"),
        nuc_seg_name=nuc_seg_name,
        cyto_seg_name=cyto_seg_name,
        _tmp_image_path=str(tmp_path / "tmp_image.dat"),
        project_location=str(tmp_path),
        debug=debug,
        overwrite=False,
        project=None,
        filehandler=filehandler,
        from_project=True,
    )
    return workflow
