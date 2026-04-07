import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scportrait.io.h5sc import write_h5sc


def _make_adata(channel_mapping: list[str] | None = None, *, include_mapping: bool = True) -> AnnData:
    n_cells = 2
    channel_names = ["segmentation", "dna", "rna"]
    obs = pd.DataFrame({"scportrait_cell_id": [1, 2]})
    obs.index = pd.Index(np.arange(n_cells).astype(str), dtype=object)
    var = pd.DataFrame(
        {"channels": np.asarray(channel_names, dtype=object)},
        index=pd.Index(np.arange(len(channel_names)).astype(str), dtype=object),
    )
    if include_mapping:
        var["channel_mapping"] = np.asarray(channel_mapping, dtype=object)

    adata = AnnData(obs=obs, var=var)
    adata.obsm["single_cell_images"] = np.random.default_rng(0).random((n_cells, len(channel_names), 8, 8))
    return adata


def test_write_h5sc_requires_channel_mapping_column(tmp_path):
    adata = _make_adata(include_mapping=False)

    with pytest.raises(ValueError, match="must contain a 'channel_mapping' column"):
        write_h5sc(adata, tmp_path / "missing_mapping.h5sc")


def test_write_h5sc_rejects_missing_channel_mapping_values(tmp_path):
    adata = _make_adata(["mask", None, "image_channel"])

    with pytest.raises(ValueError, match="contains missing values"):
        write_h5sc(adata, tmp_path / "null_mapping.h5sc")


def test_write_h5sc_rejects_invalid_channel_mapping_values(tmp_path):
    adata = _make_adata(["mask", "image", "image_channel"])

    with pytest.raises(ValueError, match="may only contain"):
        write_h5sc(adata, tmp_path / "invalid_mapping.h5sc")


def test_write_h5sc_requires_at_least_one_mask_channel(tmp_path):
    adata = _make_adata(["image_channel", "image_channel", "image_channel"])

    with pytest.raises(ValueError, match="at least one 'mask' channel"):
        write_h5sc(adata, tmp_path / "no_mask.h5sc")


def test_write_h5sc_accepts_valid_channel_mapping(tmp_path):
    adata = _make_adata(["mask", "image_channel", "image_channel"])

    write_h5sc(adata, tmp_path / "valid.h5sc")
