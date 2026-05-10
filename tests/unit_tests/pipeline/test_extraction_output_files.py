from __future__ import annotations

import h5py
import numpy as np

from scportrait.pipeline.extraction import HDF5CellExtraction


def _make_extraction(tmp_path, config: dict | None = None) -> HDF5CellExtraction:
    base_config = {
        "threads": 4,
        "image_size": 128,
        "cache": str(tmp_path / "cache"),
    }
    if config is not None:
        base_config.update(config)
    return HDF5CellExtraction(config=base_config, directory=tmp_path / "extraction")


def test_initialize_empty_anndata_writes_nested_single_cell_images_uns_group(tmp_path):
    extraction = _make_extraction(tmp_path, {"normalize_output": True, "normalization_range": (0.1, 0.9)})
    extraction.masks = ["seg_all_nucleus"]
    extraction.channel_names = ["ch0", "ch1"]
    extraction.n_masks = len(extraction.masks)
    extraction.n_image_channels = len(extraction.channel_names)
    extraction.num_classes = 3
    extraction.normalization = True
    extraction.normalization_range = (0.1, 0.9)
    extraction.compression_type = "lzf"
    extraction.output_path = tmp_path / "single_cells.h5ad"

    extraction._initialize_empty_anndata()

    with h5py.File(extraction.output_path, "r") as f:
        metadata_group = f["uns"][extraction.DEFAULT_NAME_SINGLE_CELL_IMAGES]

        assert isinstance(metadata_group, h5py.Group)
        np.testing.assert_array_equal(
            metadata_group["channel_names"].asstr()[:],
            np.array(["seg_all_nucleus", "ch0", "ch1"]),
        )
        np.testing.assert_array_equal(
            metadata_group["channel_mapping"].asstr()[:],
            np.array(["mask", "image_channel", "image_channel"]),
        )
        assert metadata_group["n_cells"][()] == 3
        assert metadata_group["n_channels"][()] == 3
        assert metadata_group["n_masks"][()] == 1
        assert metadata_group["n_image_channels"][()] == 2
        assert metadata_group["image_size"][()] == 128
        assert bool(metadata_group["normalization"][()]) is True
        assert metadata_group["normalization_range_lower"][()] == 0.1
        assert metadata_group["normalization_range_upper"][()] == 0.9
        assert metadata_group["compression"].asstr()[()] == "lzf"
