import os
import shutil

import pytest
import spatialdata as sd
import yaml
from spatialdata.datasets import blobs

from scportrait.pipeline.project import Project


@pytest.fixture()
def sdata_path(tmp_path):
    sdata = blobs()

    # Write to temporary location
    sdata_path = tmp_path / "sdata.zarr"
    sdata.write(sdata_path)
    yield sdata_path
    shutil.rmtree(sdata_path)


@pytest.fixture()
def config_path(tmp_path):
    config_path = tmp_path / "config.yml"
    config = {"name": "Test segmentation"}

    with open(config_path, "w") as f:
        f.write(yaml.safe_dump(config))

    yield str(config_path)

    os.remove(config_path)


@pytest.mark.parametrize("image_name", ["blobs_image"])
def test_project_load_input_from_sdata(sdata_path, config_path, tmp_path, image_name: str):
    project_path = str(tmp_path / "scportrait/project/")

    project = Project(
        project_location=project_path,
        config_path=config_path,
        overwrite=True,
        debug=True,
    )

    project.load_input_from_sdata(sdata_path, input_image_name=image_name)
