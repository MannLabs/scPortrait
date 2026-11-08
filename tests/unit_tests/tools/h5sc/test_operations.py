# tests/test_operations.py

from pathlib import Path

import anndata as ad
import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import pytest
from scportrait.tl.h5sc import (
    add_spatial_coordinates,
    get_cell_id_index,
    subset_cells_region,
    subset_h5sc,
    update_obs_on_disk,
)

from scportrait.io import read_h5sc

rng = np.random.default_rng()


def test_update_obs_on_disk(h5sc_object, tmp_path):
    # Write h5ad
    p = tmp_path / "test.h5ad"
    h5sc_object.write(p)

    h5sc_object.uns["h5sc_source_path"] = str(p)
    size = h5sc_object.obs.shape[0]

    # Modify obs
    random_values = rng.integers(1, 10, size=size)
    h5sc_object.obs["new_col"] = random_values
    h5sc_object.obs["label"] = pd.Series(["a", "b", "a", "c"], index=h5sc_object.obs.index, dtype="string")
    h5sc_object.obs["category"] = pd.Series(
        pd.Categorical(["one", "two", "one", "two"], categories=["one", "two", "unused"]),
        index=h5sc_object.obs.index,
    )
    update_obs_on_disk(h5sc_object)

    # Reload and confirm updated
    reloaded = read_h5sc(p)
    assert "new_col" in reloaded.obs.columns
    assert np.all(reloaded.obs["new_col"] == random_values)
    assert reloaded.obs.index.tolist() == ["0", "1", "2", "3"]
    assert reloaded.obs["label"].tolist() == ["a", "b", "a", "c"]
    assert reloaded.obs["category"].cat.categories.tolist() == ["one", "two", "unused"]


def test_add_spatial_coordinates_preserves_obs_order(h5sc_object):
    centers = pd.DataFrame(
        {"x": [30.0, 10.0, 40.0, 20.0], "y": [300.0, 100.0, 400.0, 200.0]},
        index=pd.Index([107, 101, 109, 102], name="scportrait_cell_id"),
    )

    add_spatial_coordinates(h5sc_object, dd.from_pandas(centers, npartitions=1), update_on_disk=False)

    np.testing.assert_array_equal(h5sc_object.obs[["x", "y"]].to_numpy(), [[10, 100], [20, 200], [30, 300], [40, 400]])


def test_add_spatial_coordinates_rejects_incomplete_centers(h5sc_object):
    centers = pd.DataFrame({"x": [1.0], "y": [2.0]}, index=pd.Index([101]))

    with pytest.raises(ValueError, match="exactly match"):
        add_spatial_coordinates(h5sc_object, dd.from_pandas(centers, npartitions=1))


def test_get_cell_id_index_single(h5sc_object):
    idx = get_cell_id_index(h5sc_object, 107)
    assert idx == 2


def test_get_cell_id_index_list(h5sc_object):
    idx = get_cell_id_index(h5sc_object, [101, 109])
    assert idx == [0, 3]


def test_subset_h5sc(h5sc_object, tmp_path):
    out = tmp_path / "subset.h5sc"
    subset_h5sc(h5sc_object, [101, 102], out)

    assert out.exists()
