import shutil
import warnings
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scportrait.tl.sdata.write as write
import zarr
from anndata import AnnData
from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01
from spatialdata.datasets import blobs


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


def _make_table_from_dataframe() -> AnnData:
    obs = pd.DataFrame(
        {
            "region": pd.array(["1", "2", "3"], dtype="string"),
            "group": pd.Categorical(pd.array(["a", "b", "a"], dtype="string")),
        },
        index=pd.Index(["1", "2", "3"], dtype="string"),
    )
    var = pd.DataFrame(index=pd.Index(["feature_0"], dtype="string"))
    X = np.arange(3, dtype=np.float32).reshape(3, 1)
    return AnnData(X=X, obs=obs, var=var)


@pytest.mark.parametrize("legacy_settings", [False, True])
@pytest.mark.parametrize("legacy_format", [False, True])
def test_add_element_sdata_table_from_dataframe(sdata_path, monkeypatch, legacy_settings, legacy_format):
    sdata_new = SpatialData()
    sdata_new.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01() if legacy_format else None)

    table = _make_table_from_dataframe()
    if legacy_settings:
        monkeypatch.setattr(write._helper, "ad", SimpleNamespace(settings=SimpleNamespace()))
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        write._helper.add_element_sdata(sdata_new, table, "table", overwrite=True)

    if not legacy_settings and hasattr(ad.settings, "auto_shard_zarr_v3"):
        assert not any("zarr v3 autosharding" in str(w.message) for w in caught_warnings)

    assert "table" in sdata_new
    sdata = read_zarr(sdata_path)
    assert "table" in sdata
    np.testing.assert_array_equal(sdata["table"].X, table.X)
    pd.testing.assert_frame_equal(sdata["table"].obs, table.obs)
    pd.testing.assert_frame_equal(sdata["table"].var, table.var)
    assert zarr.open_group(sdata_path, mode="r")["tables/table/X"].metadata.zarr_format == (2 if legacy_format else 3)


@pytest.mark.parametrize("auto_sharding", [None, False, True])
@pytest.mark.parametrize("target_size", [None, 8 * 1024**2])
@pytest.mark.parametrize("write_fails", [False, True])
def test_table_sharding_respects_and_restores_settings(monkeypatch, auto_sharding, target_size, write_fails):
    if not hasattr(ad.settings, "auto_shard_zarr_v3") or not hasattr(zarr, "config"):
        pytest.skip("Auto-sharding settings require newer AnnData and Zarr versions.")
    if "target_shard_size_bytes" not in zarr.config.get("array"):
        pytest.skip("This Zarr version does not support a target shard size.")

    sdata = SpatialData()

    def check_write_settings(element_name):
        assert element_name == "table"
        assert ad.settings.auto_shard_zarr_v3 is (auto_sharding is not False)
        assert zarr.config.get("array.target_shard_size_bytes") == (target_size or 64 * 1024**2)
        if write_fails:
            raise OSError("write failed")

    monkeypatch.setattr(sdata, "write_element", check_write_settings)
    with (
        ad.settings.override(auto_shard_zarr_v3=auto_sharding),
        zarr.config.set({"array.target_shard_size_bytes": target_size}),
    ):
        if write_fails:
            with pytest.raises(OSError, match="write failed"):
                write._helper.add_element_sdata(sdata, _make_table_from_dataframe(), "table")
        else:
            write._helper.add_element_sdata(sdata, _make_table_from_dataframe(), "table")

        assert ad.settings.auto_shard_zarr_v3 is auto_sharding
        assert zarr.config.get("array.target_shard_size_bytes") == target_size


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
