from __future__ import annotations

import numpy as np
import yaml
import zarr

from scportrait.pipeline.project import Project


def _write_test_omezarr(path, channel_names: list[str], shape: tuple[int, int, int] = (2, 64, 64)) -> None:
    root = zarr.open_group(path, mode="w")
    root.attrs["omero"] = {"channels": [{"label": channel_name} for channel_name in channel_names]}
    root.create_array("0", data=np.arange(np.prod(shape), dtype=np.uint16).reshape(shape))


def test_project_load_input_from_omezarr_reads_level_zero_component(tmp_path):
    ome_zarr_path = tmp_path / "input_image.ome.zarr"
    channel_names = ["dna", "rna"]
    _write_test_omezarr(ome_zarr_path, channel_names=channel_names)

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"name": "Test segmentation", "cache": str(tmp_path / "cache")}))

    project = Project(
        project_location=str(tmp_path / "project"),
        config_path=str(config_path),
        overwrite=True,
        debug=True,
    )

    project.load_input_from_omezarr(ome_zarr_path)

    assert project.input_image_status is True
    assert project.channel_names == channel_names
    assert project.input_image.shape == (2, 64, 64)
    assert project.input_image.c.values.tolist() == channel_names
