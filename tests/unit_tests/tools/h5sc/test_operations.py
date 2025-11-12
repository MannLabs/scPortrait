# tests/test_operations.py

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from scportrait.io import read_h5sc
from scportrait.tl.h5sc import (
    add_spatial_coordinates,
    get_cell_id_index,
    subset_cells_region,
    subset_h5sc,
    update_obs_on_disk,
)

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
    update_obs_on_disk(h5sc_object)

    # Reload and confirm updated
    reloaded = read_h5sc(p)
    assert "new_col" in reloaded.obs.columns
    assert np.all(reloaded.obs["new_col"] == random_values)


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
