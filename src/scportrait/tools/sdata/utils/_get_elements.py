import pandas as pd


def get_featurization_results_as_df(sdata, key: str, drop_region: bool = True) -> pd.DataFrame:
    """
    Return featurization results from a SpatialData container as a merged DataFrame.

    Args:
        sdata: spatialdata.SpatialData object.
        key: key for the featurization result (e.g. "ConvNeXtFeaturizer_Ch4_cytosol").
        drop_region: whether to drop the 'region' column. Default True.

    Returns:
        pd.DataFrame: merged features + obs.
    """
    if key not in sdata:
        raise KeyError(f"'{key}' not found in SpatialData object.")

    adata = sdata[key]
    df = adata.to_df().merge(adata.obs, left_index=True, right_index=True)
    if drop_region and "region" in df.columns:
        df = df.drop(columns="region")

    return df
