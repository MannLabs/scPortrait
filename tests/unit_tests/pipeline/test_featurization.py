import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from alphabase.io import tempmmap
from anndata import AnnData

from scportrait.io import write_h5sc
from scportrait.pipeline.featurization import CellFeaturizer, CellFeaturizer_single_channel


def test_cell_featurizer(tmp_path):
    # temp directory unique to this test
    out_dir = tmp_path / "featurization"
    out_dir.mkdir()

    config = {
        "batch_size": 100,
        "dataloader_worker_number": 10,
    }

    f = CellFeaturizer(
        config=config,
        directory=str(out_dir),
        project_location=None,
        overwrite=True,
    )

    # Image: 1 batch, 1 channel, 4x4
    img = torch.zeros((1, 1, 4, 4), dtype=torch.float32)

    # Define a "cell" region and a "nucleus" region inside it
    nucleus_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    cell_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)

    cell_mask[..., 1:3, 1:3] = 1.0  # 2x2 cell block
    nucleus_mask[..., 1:2, 1:2] = 1.0  # 1x1 nucleus (top-left of cell)

    # Intensities: nucleus=10, cytosol=1
    img[cell_mask.bool()] = 1.0
    img[nucleus_mask.bool()] = 10.0

    # Build label stack
    labels = torch.cat([nucleus_mask, cell_mask], dim=1)  # shape (1, 2, 4, 4)

    # Run featurization on concatenated masks and images
    feats = f.calculate_statistics(torch.cat([labels, img], dim=1))
    column_names = f._generate_column_names(n_masks=2, channel_names=["ch0"])
    feat_map = dict(zip(column_names, feats[0].tolist(), strict=True))

    # Key regression assertion:
    # If masks accidentally become all-True, these will match (or be very close).
    nuc_mean = feat_map["ch0_mean_nucleus"]
    cyto_mean = feat_map["ch0_mean_cytosol"]

    assert nuc_mean > cyto_mean
    assert abs(nuc_mean - cyto_mean) > 1.0

    # Ensure masks are applied and sums are finite (no NaNs from masked sum).
    assert not np.isnan(feat_map["ch0_summed_intensity_nucleus"])
    assert not np.isnan(feat_map["ch0_summed_intensity_cytosol"])
    assert not np.isnan(feat_map["ch0_summed_intensity_cytosol_only"])

    # Check exact expected values for this synthetic setup.
    assert feat_map["ch0_summed_intensity_nucleus"] == pytest.approx(10.0)
    assert feat_map["ch0_summed_intensity_cytosol"] == pytest.approx(13.0)
    assert feat_map["ch0_summed_intensity_cytosol_only"] == pytest.approx(3.0)

    # Area-normalized sums should use each mask's own area (not the last mask).
    assert feat_map["ch0_summed_intensity_area_normalized_nucleus"] == pytest.approx(10.0)
    assert feat_map["ch0_summed_intensity_area_normalized_cytosol"] == pytest.approx(13.0 / 4.0)
    assert feat_map["ch0_summed_intensity_area_normalized_cytosol_only"] == pytest.approx(1.0)


@pytest.mark.parametrize("channel_selection", [2, [2, 3, 4]])
def test_convnext_initializes_channel_selection(tmp_path, monkeypatch, channel_selection):
    import scportrait.pipeline.featurization as featurization

    monkeypatch.setattr(featurization, "_import_transformers", lambda: None)
    featurizer = featurization.ConvNeXtFeaturizer(
        config={"batch_size": 2, "dataloader_worker_number": 0, "channel_selection": channel_selection},
        directory=tmp_path,
    )

    expected = [channel_selection] if isinstance(channel_selection, int) else channel_selection
    assert featurizer.channel_selection == expected
    assert featurizer.label == f"ConvNeXtFeaturizer_{'_'.join(f'Ch{i}' for i in expected)}"


@pytest.mark.parametrize("n_masks", [1, 2])
@pytest.mark.parametrize("featurizer_class", [CellFeaturizer, CellFeaturizer_single_channel])
def test_cell_featurizer_processes_h5sc_with_correct_feature_names(tmp_path, n_masks, featurizer_class):
    path = tmp_path / "cells.h5sc"
    channel_names = ["nucleus", "cytosol"][:n_masks] + ["DAPI", "aSMA"]
    images = np.ones((3, len(channel_names), 4, 4), dtype=np.float32)
    images[:, 0] = 0
    images[:, 0, 0, 0] = 1
    images[:, n_masks] = 2
    images[:, n_masks + 1] = 7
    adata = AnnData(
        obs=pd.DataFrame({"scportrait_cell_id": [10, 20, 30]}, index=["0", "1", "2"]),
        var=pd.DataFrame(
            {"channels": channel_names, "channel_mapping": ["mask"] * n_masks + ["image_channel"] * 2},
            index=channel_names,
        ),
    )
    adata.obsm["single_cell_images"] = images
    write_h5sc(adata, path)
    featurizer = featurizer_class(
        config={
            "batch_size": 2,
            "dataloader_worker_number": 0,
            "inference_device": "cpu",
            "channel_selection": n_masks + 1,
            "cache": str(tmp_path / "cache"),
        },
        directory=tmp_path,
    )

    try:
        result = featurizer.process(path, return_results=True)
        features = tempmmap.mmap_array_from_path(result.features_path)
        assert result.n_rows == 3
        assert features.shape[1] == len(result.var_names)
        np.testing.assert_allclose(features[:, result.var_names.index("aSMA_mean_cytosol")], 7)
        np.testing.assert_array_equal(tempmmap.mmap_array_from_path(result.cell_ids_path).ravel(), [10, 20, 30])
        if featurizer_class is CellFeaturizer:
            np.testing.assert_allclose(features[:, result.var_names.index("DAPI_mean_cytosol")], 2)
        else:
            assert not any(name.startswith("DAPI_") for name in result.var_names)
    finally:
        if hasattr(featurizer, "dataloader"):
            featurizer.dataloader.dataset.close()
        featurizer.clear_temp_dir()


@pytest.mark.parametrize(
    ("n_masks", "channel_selection", "expected_selection"),
    [
        (2, 4, [0, 1, 4]),
        (1, 4, [0, 4]),
        (2, [4], [0, 1, 4]),
        (1, [4], [0, 4]),
    ],
)
def test_single_channel_featurizer_uses_metadata_for_selection(
    tmp_path, n_masks, channel_selection, expected_selection
):
    featurizer = CellFeaturizer_single_channel(
        config={
            "batch_size": 2,
            "dataloader_worker_number": 0,
            "channel_selection": channel_selection,
        },
        directory=tmp_path,
    )

    assert featurizer.channel_selection == [4]
    assert featurizer.label == "CellFeaturizer_single_channel_Ch4"

    featurizer.n_masks = n_masks
    featurizer.channel_names = ["nucleus", "cellmask", "DAPI", "FITC", "aSMA"]
    featurizer.image_channel_locs = [2, 3, 4]
    featurizer._setup_channel_selection()

    assert featurizer.channel_selection == expected_selection
    assert featurizer.label == "CellFeaturizer_single_channel_Ch4"


@pytest.mark.parametrize("channel_selection", [-1, 0, 1, 5])
def test_single_channel_featurizer_rejects_non_image_channel(tmp_path, channel_selection):
    featurizer = CellFeaturizer_single_channel(
        config={
            "batch_size": 2,
            "dataloader_worker_number": 0,
            "channel_selection": channel_selection,
        },
        directory=tmp_path,
    )
    featurizer.n_masks = 2
    featurizer.channel_names = ["nucleus", "cellmask", "DAPI", "FITC", "aSMA"]
    featurizer.image_channel_locs = [2, 3, 4]

    with pytest.raises(ValueError, match="does not identify an image channel"):
        featurizer._setup_channel_selection()


@pytest.mark.parametrize("label", [None, "Ch3_Featurization"])
def test_single_channel_output_path_preserves_default_and_configured_labels(tmp_path, label):
    config = {
        "batch_size": 2,
        "dataloader_worker_number": 0,
        "inference_device": "cpu",
        "channel_selection": 4,
    }
    if label is not None:
        config["label"] = label
    featurizer = CellFeaturizer_single_channel(
        config=config,
        directory=tmp_path,
    )
    featurizer.data_type = "cytosol"

    def set_dataset_metadata():
        featurizer.n_masks = 2
        featurizer.channel_names = ["nucleus", "cellmask", "DAPI", "FITC", "aSMA"]
        featurizer.image_channel_locs = [2, 3, 4]

    featurizer._get_single_cell_datafile_specs = set_dataset_metadata
    featurizer._setup_transforms = lambda: None
    featurizer.create_temp_dir = lambda: None

    featurizer._setup(dataset_paths=tmp_path / "unused.h5", return_results=False)

    expected_label = label or "CellFeaturizer_single_channel_Ch4"
    assert featurizer.run_path.endswith(f"cytosol_{expected_label}")


def _write_extraction_metadata(path, *, n_masks=2, channel_names=None, channel_mapping=None, n_cells=3):
    channel_names = channel_names or ["nucleus", "cellmask", "aSMA"]
    channel_mapping = channel_mapping or ["mask", "mask", "image_channel"]
    with h5py.File(path, "w") as file:
        metadata = file.create_group("uns/single_cell_images")
        metadata["n_masks"] = n_masks
        metadata["n_channels"] = len(channel_names)
        metadata["n_image_channels"] = len(channel_names) - n_masks
        metadata["channel_names"] = np.asarray(channel_names, dtype=h5py.string_dtype())
        metadata["channel_mapping"] = np.asarray(channel_mapping, dtype=h5py.string_dtype())
        metadata["n_cells"] = n_cells
    return path


def test_single_cell_metadata_accepts_matching_files_and_tracks_cell_counts(tmp_path):
    files = [
        _write_extraction_metadata(tmp_path / "first.h5", n_cells=3),
        _write_extraction_metadata(tmp_path / "second.h5", n_cells=5),
    ]
    featurizer = CellFeaturizer(
        config={"batch_size": 2, "dataloader_worker_number": 0},
        directory=tmp_path,
    )
    featurizer.extraction_file = files

    featurizer._get_single_cell_datafile_specs()

    assert featurizer.n_masks == 2
    assert featurizer.segmentation_channel == "nucleus"
    assert featurizer.n_cells == [3, 5]

    featurizer._get_single_cell_datafile_specs()
    assert featurizer.n_cells == [3, 5]


def test_single_cell_metadata_rejects_empty_file_list(tmp_path):
    featurizer = CellFeaturizer(config={"batch_size": 2, "dataloader_worker_number": 0}, directory=tmp_path)
    featurizer.extraction_file = []

    with pytest.raises(ValueError, match="At least one single-cell extraction file"):
        featurizer._get_single_cell_datafile_specs()


@pytest.mark.parametrize(
    ("metadata_changes", "error"),
    [
        ({"n_masks": 1}, "metadata is not consistent"),
        ({"channel_names": ["nucleus", "cellmask", "other"]}, "metadata is not consistent"),
        ({"channel_mapping": ["mask", "image_channel", "image_channel"]}, "metadata is not consistent"),
    ],
)
def test_single_cell_metadata_rejects_inconsistent_files(tmp_path, metadata_changes, error):
    first = _write_extraction_metadata(tmp_path / "first.h5")
    second = _write_extraction_metadata(tmp_path / "second.h5", **metadata_changes)
    featurizer = CellFeaturizer(
        config={"batch_size": 2, "dataloader_worker_number": 0},
        directory=tmp_path,
    )
    featurizer.extraction_file = [first, second]

    with pytest.raises(ValueError, match=error):
        featurizer._get_single_cell_datafile_specs()
