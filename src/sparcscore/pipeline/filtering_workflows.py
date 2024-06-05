from sparcscore.pipeline.filter_segmentation import (
    SegmentationFilter,
    TiledSegmentationFilter,
)

import numpy as np

from sparcscore.processing.filtering import MatchNucleusCytosolIds


class BaseFiltering(SegmentationFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_unique_ids(self, mask):
        return np.unique(mask)[1:]

    def return_empty_mask(self, input_image):
        # write out an empty entry
        self.save_classes(classes={})


class filtering_match_nucleus_to_cytosol(BaseFiltering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filtering_threshold = self.config["filtering_threshold"]

        # allow for optional downsampling to improve computation time
        if "downsampling_factor" in self.config.keys():
            self.N = self.config["downsampling_factor"]
            self.kernel_size = self.config["downsampling_smoothing_kernel_size"]
            self.erosion_dilation = self.config["downsampling_erosion_dilation"]
        else:
            self.N = None
            self.kernel_size = None
            self.erosion_dilation = False

    def process(self, input_masks):
        if isinstance(input_masks, str):
            input_masks = self.read_input_masks(input_masks)

        # perform filtering
        filter = MatchNucleusCytosolIds(
            filtering_threshold=self.filtering_threshold,
            downsampling_factor=self.N,
            erosion_dilation=self.erosion_dilation,
            smoothing_kernel_size=self.kernel_size,
        )
        nucleus_cytosol_pairs = filter.generate_lookup_table(
            input_masks[0], input_masks[1]
        )

        # save results
        self.save_classes(classes=nucleus_cytosol_pairs)


class multithreaded_filtering_match_nucleus_to_cytosol(TiledSegmentationFilter):
    method = filtering_match_nucleus_to_cytosol
