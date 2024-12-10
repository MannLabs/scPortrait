"""
Run Segmentation Method on any input image
==========================================

Instantiate and run a segmentation workflow on any input image without needing to create a complete scPortrait project.

This can be usefull for debugging purposes or to test out different segmentation methods side by side without needing to save anything to file.
"""

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

from scportrait.data._datasets import dataset_3
from scportrait.pipeline.segmentation.workflows import CytosolOnlySegmentationCellpose

# load image
path = dataset_3()
image = imread(f"{path}/Ch2.tif")

# define config for method
config = {
    "CytosolOnlySegmentationCellpose": {
        "input_channels": 1,
        "output_masks": 1,
        "cache": ".",
        "cytosol_segmentation": {"model": "cyto2"},
        "match_masks": False,
        "filter_masks_size": False,
        "chunk_size": 100,
    }
}

# initialize method
method = CytosolOnlySegmentationCellpose(config=config)
method.config = method.config["CytosolOnlySegmentationCellpose"]

# create datastructure to save results to
method.maps = {}

# perform segmentation
method.cellpose_segmentation(image)

# access results
seg_mask = method.maps["cytosol_segmentation"]

# plot results
plt.imshow(seg_mask)
plt.axis("off")
