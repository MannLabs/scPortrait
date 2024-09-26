import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scportrait.pipeline._utils.segmentation import numba_mask_centroid
from scportrait.pipeline.classification import CellFeaturizer
from scportrait.pipeline.extraction import HDF5CellExtraction
from scportrait.pipeline.project import Project
from scportrait.pipeline.segmentation_workflows import CytosolSegmentationCellpose, ShardedCytosolSegmentationCellpose

if __name__ == "__main__":
    print(os.getcwd())
    project_location = f"example_data/example_4/benchmark"

    project = Project(
        os.path.abspath(project_location),
        config_path="example_data/example_4/config_example4.yml",
        overwrite=True,
        debug=True,
        segmentation_f=CytosolSegmentationCellpose,
        extraction_f=HDF5CellExtraction,
        classification_f=CellFeaturizer,
    )

    images = [
        "example_data/example_4/input_images/ch1.tif",
        "example_data/example_4/input_images/ch2.tif",
        "example_data/example_4/input_images/ch3.tif",
    ]

    project.load_input_from_file(images)  # type: ignore

    project.segment()
    project.extract()
    project.classify(accessory=[(), (), ()])  # type: ignore

    with h5py.File(f"{project.seg_directory}/segmentation.h5", "r") as hf:
        masks = hf["labels"][:]

    nucleus_ids = set(np.unique(masks[0])) - {0}
    cytosol_ids = set(np.unique(masks[1])) - {0}
    classes = set(
        pd.read_csv(f"{project.seg_directory}/classes.csv", header=None)[0]
        .astype(project.DEFAULT_SEGMENTATION_DTYPE)
        .to_list()
    )

    assert nucleus_ids == classes

    centers, size, _ids = numba_mask_centroid(masks[0])
    assert set(_ids) == nucleus_ids

    centers, size, _ids = numba_mask_centroid(masks[1])
    assert set(_ids) == cytosol_ids

    # load classification results
    classification_results = pd.read_csv(
        f"{project_location}/classification/0_featurization_Ch4/calculated_features.csv", index_col=0
    )

    classified_cells = set(classification_results.cell_id.unique())
    removed_ids = set(project.extraction_f.cell_ids_removed)

    assert nucleus_ids.difference(classified_cells).difference(removed_ids) == set()
    assert nucleus_ids == cytosol_ids
