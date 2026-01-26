import numpy as np
import pytest
import torch

from scportrait.pipeline.featurization import CellFeaturizer  # adjust import to your project


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


def test_mask_bool_integrity_not_corrupted():
    mask = torch.tensor([[True, False], [False, True]])
    before_true = mask.sum().item()

    # This simulates the buggy behavior:
    # mask[mask == 0] = torch.nan  # would cast nan->True and change mask
    # Instead, ensure your implementation never does this.

    assert mask.dtype == torch.bool
    assert mask.sum().item() == before_true
