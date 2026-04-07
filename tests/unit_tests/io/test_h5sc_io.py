import h5py
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scportrait.io.h5sc import legacy_h5_to_h5sc, read_h5sc, write_h5sc


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


def test_legacy_h5_to_h5sc_converts_channel_information_and_obs_metadata(tmp_path):
    legacy_path = tmp_path / "legacy.h5"
    output_path = tmp_path / "converted.h5sc"

    images = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    labelled_index = np.array(
        [
            ["0", "101", "WT", "1.5"],
            ["1", "102", "KO", "2.5"],
        ],
        dtype=object,
    )

    with h5py.File(legacy_path, "w") as handle:
        handle.create_dataset("single_cell_data", data=images, compression="lzf")
        handle.create_dataset("single_cell_index", data=np.array([[0, 101], [1, 102]], dtype=np.int64))
        handle.create_dataset(
            "single_cell_index_labelled",
            data=labelled_index.astype("S"),
        )
        handle.create_dataset("label_names", data=np.array(["condition", "score"], dtype="S"))
        handle.create_dataset(
            "channel_information",
            data=np.array(["dna"], dtype="S"),
        )

    legacy_h5_to_h5sc(legacy_path, output_path, ["dna"])

    adata = read_h5sc(output_path)
    np.testing.assert_array_equal(np.asarray(adata.obsm["single_cell_images"]), images)
    assert adata.obs["scportrait_cell_id"].tolist() == [101, 102]
    assert adata.obs["condition"].tolist() == ["WT", "KO"]
    assert adata.obs["score"].tolist() == [1.5, 2.5]
    assert adata.var["channels"].tolist() == ["seg_all_nucleus", "seg_all_cytosol", "dna"]
    assert adata.var["channel_mapping"].tolist() == ["mask", "mask", "image_channel"]


def test_legacy_h5_to_h5sc_uses_user_supplied_image_channel_order(tmp_path):
    legacy_path = tmp_path / "legacy_image_only_channels.h5"
    output_path = tmp_path / "converted_image_only_channels.h5sc"

    images = np.ones((1, 6, 2, 2), dtype=np.float16)

    with h5py.File(legacy_path, "w") as handle:
        handle.create_dataset("single_cell_data", data=images)
        handle.create_dataset("single_cell_index", data=np.array([[0, 7]], dtype=np.int64))
        handle.create_dataset(
            "channel_information",
            data=np.array(["Alexa488", "Alexa647", "HOECHST33342", "mCherry"], dtype="S"),
        )

    legacy_h5_to_h5sc(
        legacy_path,
        output_path,
        ["mCherry", "HOECHST33342", "Alexa647", "Alexa488"],
    )

    adata = read_h5sc(output_path)
    assert adata.var["channels"].tolist() == [
        "seg_all_nucleus",
        "seg_all_cytosol",
        "mCherry",
        "HOECHST33342",
        "Alexa647",
        "Alexa488",
    ]
    assert adata.var["channel_mapping"].tolist() == [
        "mask",
        "mask",
        "image_channel",
        "image_channel",
        "image_channel",
        "image_channel",
    ]


def test_legacy_h5_to_h5sc_rejects_mismatched_image_channel_order(tmp_path):
    legacy_path = tmp_path / "legacy_bad_order.h5"
    output_path = tmp_path / "converted_bad_order.h5sc"

    with h5py.File(legacy_path, "w") as handle:
        handle.create_dataset("single_cell_data", data=np.ones((1, 4, 2, 2), dtype=np.float16))
        handle.create_dataset("single_cell_index", data=np.array([[0, 7]], dtype=np.int64))
        handle.create_dataset(
            "channel_information",
            data=np.array(["Alexa488", "Alexa647"], dtype="S"),
        )

    with pytest.raises(ValueError, match="image_channel_order must contain exactly the same channel names"):
        legacy_h5_to_h5sc(legacy_path, output_path, ["Alexa488", "mCherry"])


def test_legacy_h5_to_h5sc_requires_image_channel_order(tmp_path):
    legacy_path = tmp_path / "legacy_missing_order.h5"
    output_path = tmp_path / "converted_missing_order.h5sc"

    with h5py.File(legacy_path, "w") as handle:
        handle.create_dataset("single_cell_data", data=np.ones((1, 4, 2, 2), dtype=np.float16))
        handle.create_dataset("single_cell_index", data=np.array([[0, 7]], dtype=np.int64))
        handle.create_dataset(
            "channel_information",
            data=np.array(["Alexa488", "Alexa647"], dtype="S"),
        )

    with pytest.raises(ValueError, match="image_channel_order is required"):
        legacy_h5_to_h5sc(legacy_path, output_path)
