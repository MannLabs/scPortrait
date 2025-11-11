# tests/test_sdata_get_df.py
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scportrait.tools.sdata import get_featurization_results_as_df


@pytest.fixture
def make_table():
    """Factory to create a minimal AnnData 'table' shaped like a SpatialData table."""

    def _make(n_obs=4, n_feats=3, id_key="scportrait_cell_id", region="seg_all_cyto", index_kind="str"):
        X = np.arange(n_obs * n_feats, dtype=np.float32).reshape(n_obs, n_feats)
        var = pd.DataFrame(index=[f"f{i}" for i in range(n_feats)])
        obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
        obs[id_key] = np.arange(10, 10 + n_obs, dtype=np.int64)
        obs["region"] = region

        ad = AnnData(X=X, var=var, obs=obs)

        # optional coords to mimic usual tables
        ad.obsm["spatial"] = np.stack([np.arange(n_obs), np.arange(n_obs) * 2], axis=1).astype(float)

        # vary index kinds
        if index_kind == "int":
            ad.obs.index = pd.Index(np.arange(n_obs))
        elif index_kind == "unsorted_int":
            ad.obs.index = pd.Index([101, 99, 105, 100][:n_obs])
        else:
            # "str": keep default string index
            pass

        return ad

    return _make


@pytest.mark.parametrize(
    "n_obs,n_feats,index_kind",
    [
        (3, 2, "str"),
        (5, 4, "int"),
        (4, 3, "unsorted_int"),
    ],
)
def test_get_df_happy_path(make_table, n_obs, n_feats, index_kind):
    table_key = "ConvNeXtFeaturizer_Ch4_cytosol"
    ad = make_table(n_obs=n_obs, n_feats=n_feats, index_kind=index_kind)

    # minimal sdata stand-in
    sdata = {table_key: ad}

    df = get_featurization_results_as_df(sdata, table_key)

    # shape: rows match n_obs; columns = n_feats + id column (region dropped)
    assert df.shape == (n_obs, n_feats + 1)
    assert "region" not in df.columns
    assert "scportrait_cell_id" in df.columns

    # feature columns preserved and ordered
    feat_cols = [f"f{i}" for i in range(n_feats)]
    assert list(df.columns[:n_feats]) == feat_cols

    # matrix values unchanged
    np.testing.assert_allclose(df[feat_cols].to_numpy(), ad.X)

    # id mapping preserved
    np.testing.assert_array_equal(df["scportrait_cell_id"].to_numpy(), ad.obs["scportrait_cell_id"].to_numpy())


def test_missing_table_key_raises(make_table):
    sdata = {"some_other_key": make_table()}
    with pytest.raises((KeyError, ValueError)):
        get_featurization_results_as_df(sdata, "ConvNeXtFeaturizer_Ch4_cytosol")


@pytest.mark.parametrize("region_value", ["seg_all_cyto", "custom_region"])
def test_region_column_is_removed(make_table, region_value):
    table_key = "ConvNeXtFeaturizer_Ch4_cytosol"
    ad = make_table(region=region_value)
    sdata = {table_key: ad}

    df = get_featurization_results_as_df(sdata, table_key)
    assert "region" not in df.columns
