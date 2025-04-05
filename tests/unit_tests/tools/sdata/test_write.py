import shutil

import numpy as np
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.datasets import blobs

import scportrait.tl.sdata.write as write


@pytest.fixture()
def sdata(tmp_path):
    sdata = blobs()
    # Write to temporary location
    sdata_path = tmp_path / "sdata.zarr"
    sdata.write(sdata_path)
    yield sdata
    shutil.rmtree(sdata_path)


@pytest.fixture()
def sdata_path(tmp_path):
    # Write to temporary location
    sdata_path = tmp_path / "sdata_new.zarr"
    yield sdata_path
    shutil.rmtree(sdata_path)


### test scportrait.tools.sdata.write._helper._force_delete_object
@pytest.mark.parametrize(
    "element_name",
    [
        ("blobs_image"),
        ("blobs_labels"),
        ("blobs_multiscale_image"),
        ("blobs_multiscale_labels"),
        ("blobs_circles"),
        ("table"),
    ],
)
def test_force_delete_object(sdata, element_name):
    write._helper._force_delete_object(sdata, element_name)


### test scportrait.tools.sdata.write._helper.add_element_sdata
@pytest.mark.parametrize(
    "element_name",
    [
        ("blobs_image"),
        ("blobs_multiscale_image"),
        ("blobs_multiscale_labels"),
        ("blobs_circles"),
        ("blobs_labels"),
        ("table"),
    ],
)
def test_add_element_sdata(sdata, sdata_path, element_name):
    # create a new empty spatialdata object
    sdata_new = SpatialData()
    sdata_new.write(sdata_path)

    # add element to the new object from blobs
    write._helper.add_element_sdata(sdata_new, sdata[element_name], element_name, overwrite=True)

    # check if the element is in the new object
    assert element_name in sdata_new

    # ensure it was written to disk
    sdata = read_zarr(sdata_path)
    assert element_name in sdata


### test scportrait.tools.sdata.write._helper.rename_image_element
@pytest.mark.parametrize(
    "old_name, new_name",
    [
        ("blobs_image", "blobs_image_renamed"),
        ("blobs_multiscale_image", "blobs_multiscale_image_renamed"),
    ],
)
def test_rename_image_element(sdata, old_name, new_name):
    sdata = write._helper.rename_image_element(
        sdata,
        image_element=old_name,
        new_element_name=new_name,
    )

    assert new_name in sdata
    assert old_name not in sdata


### test scportrait.tools.sdata.write._write.image
def test_write_image_single_scale(sdata_path):
    sdata = SpatialData()
    sdata.write(sdata_path)

    image_model = blobs()["blobs_image"]
    write._write.image(
        sdata,
        image=image_model,
        image_name="image_model",
        overwrite=True,
    )

    image = blobs()["blobs_image"].data.compute()

    assert isinstance(image, np.ndarray)
    write._write.image(
        sdata,
        image=image,
        image_name="image",
        overwrite=True,
    )


def test_write_image_multi_scale(sdata_path):
    sdata = SpatialData()
    sdata.write(sdata_path)

    image_model = blobs()["blobs_multiscale_image"]
    write._write.image(
        sdata,
        image=image_model,
        image_name="image_model",
        overwrite=True,
    )

    image = sdata["image_model"].scale0.image.compute().values
    write._write.image(
        sdata,
        image=image,
        image_name="image",
        overwrite=True,
    )


### test scportrait.tools.sdata.write._write.labels
def test_write_labels(sdata_path):
    sdata = SpatialData()
    sdata.write(sdata_path)

    labels_model = blobs()["blobs_labels"]
    write._write.labels(
        sdata,
        labels=labels_model,
        labels_name="labels_model",
        overwrite=True,
    )

    labels = blobs()["blobs_labels"].data.compute()

    assert isinstance(labels, np.ndarray)
    write._write.labels(
        sdata,
        labels=labels,
        labels_name="labels",
        overwrite=True,
    )


def test_write_labels_multiscale(sdata_path):
    sdata = SpatialData()
    sdata.write(sdata_path)

    labels_model = blobs()["blobs_multiscale_labels"]
    write._write.labels(
        sdata,
        labels=labels_model,
        labels_name="labels_model",
        overwrite=True,
    )

    labels = sdata["labels_model"].scale0.image.compute().values
    write._write.labels(
        sdata,
        labels=labels,
        labels_name="labels",
        overwrite=True,
    )
