import numpy as np
import pytest

import scportrait
from scportrait.pipeline.extraction import HDF5CellExtraction
from scportrait.pipeline.featurization import CellFeaturizer
from scportrait.pipeline.project import Project
from scportrait.pipeline.segmentation.workflows import CytosolSegmentationCellpose
from scportrait.pipeline.selection import LMDSelection


@pytest.mark.slow
@pytest.mark.requires_dataset("dataset_1_config", "test_dataset")
def test_full_pipeline_e2e(tmp_path):
    project_path = tmp_path / "test_project"
    config_path = scportrait.data.get_config_file("dataset_1_config")

    project = Project(
        project_location=str(project_path),
        config_path=config_path,
        segmentation_f=CytosolSegmentationCellpose,
        extraction_f=HDF5CellExtraction,
        featurization_f=CellFeaturizer,
        selection_f=LMDSelection,
        overwrite=True,
        debug=True,
    )

    dataset = scportrait.data._datasets._test_dataset()
    imgs = [
        f"{dataset}/Ch1.tif",
        f"{dataset}/Ch2.tif",
        f"{dataset}/Ch3.tif",
    ]
    project.load_input_from_tif_files(imgs)

    project.segment()
    project.extract()
    project.featurize(overwrite=True)

    feat = project.sdata["CellFeaturizer_cytosol"]
    results = feat.to_df().merge(feat.obs, left_index=True, right_index=True).drop(columns="region")

    large = results[results.cytosol_area > 4500]["scportrait_cell_id"].tolist()
    small = results[results.cytosol_area < 3000]["scportrait_cell_id"].tolist()

    cells_to_select = [
        {"name": "large_cells", "classes": large, "well": "A1"},
        {"name": "small_cells", "classes": small, "well": "B1"},
    ]

    markers = np.array([(0, 0), (2000, 0), (0, 2000)])
    project.select(cells_to_select, markers)

    # Basic sanity checks
    assert project.sdata is not None
    assert "CellFeaturizer_cytosol" in project.sdata.tables
    assert len(large) + len(small) > 0
