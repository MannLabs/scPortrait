import os
import shutil

import pytest
import spatialdata as sd
import yaml
from spatialdata.datasets import blobs

from scportrait.data._datasets import dataset_1_omezarr
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


@pytest.mark.parametrize("image_name, segmentation_name", [("blobs_image", "blobs_labels")])
def test_project_load_input_from_sdata(sdata_path, config_path, tmp_path, image_name: str, segmentation_name: str):
    project_path = str(tmp_path / "scportrait/project/")

    project = Project(
        project_location=project_path,
        config_path=config_path,
        overwrite=True,
        debug=True,
    )

    project.load_input_from_sdata(sdata_path, input_image_name=image_name, cytosol_segmentation_name=segmentation_name)


@pytest.mark.parametrize("image_name, segmentation_name", [("blobs_multiscale_image", "blobs_multiscale_labels")])
def test_project_load_input_from_sdata_multiscale_image(
    sdata_path, config_path, tmp_path, image_name: str, segmentation_name: str
):
    project_path = str(tmp_path / "scportrait/project/")

    project = Project(
        project_location=project_path,
        config_path=config_path,
        overwrite=True,
        debug=True,
    )

    project.load_input_from_sdata(sdata_path, input_image_name=image_name, cytosol_segmentation_name=segmentation_name)


def test_project_load_from_omezarr(config_path, tmp_path):
    project_path = str(tmp_path / "scportrait/project/")
    omezarr_path = dataset_1_omezarr()

    project = Project(
        project_location=project_path,
        config_path=config_path,
        overwrite=True,
        debug=True,
    )

    project.load_input_from_omezarr(omezarr_path)
