import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
from skfmm import travel_time as skfmm_travel_time
from skimage.color import label2rgb
from skimage.morphology import binary_erosion, dilation, disk
from skimage.segmentation import watershed

from scportrait.pipeline._utils.segmentation import (
    contact_filter,
    global_otsu,
    numba_mask_centroid,
    remove_edge_labels,
    segment_global_threshold,
    segment_local_threshold,
)
from scportrait.pipeline.segmentation.segmentation import (
    ShardedSegmentation,
)
from scportrait.pipeline.segmentation.workflows._base_segmentation_workflow import _BaseSegmentation


class _ClassicalSegmentation(_BaseSegmentation):
    def __init__(self, *args, **kwargs):
        self.maps = None  # just a typehint hack. Remove this and fix it properly.
        super().__init__(*args, **kwargs)

    def _visualize_final_masks(self):
        if self.segment_nuclei:
            image = label2rgb(
                self.maps["nucleus_segmentation"],
                self.maps["input_image"][0],
                bg_label=0,
                alpha=0.6,
            )

            fig = plt.figure(frameon=False, figsize=(10, 10))
            plt.imshow(image)
            plt.scatter(self.nucleus_centers[:, 1], self.nucleus_centers[:, 0], s=2, color="red")
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(self.directory, "finalized_nucleus_mask.png"))

        if self.segment_cytosol:
            image = label2rgb(
                self.maps["cytosol_segmentation"],
                self.maps["input_image"][1],
                bg_label=0,
                alpha=0.6,
            )

            fig = plt.figure(frameon=False, figsize=(10, 10))
            plt.imshow(image)
            plt.scatter(self.nucleus_centers[:, 1], self.nucleus_centers[:, 0], s=2, color="red")
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(self.directory, "cytosol_mask.png"))

    def _get_processing_parameters(self):
        # check if normalization is required
        if "lower_quantile_normalization" in self.config.keys():
            self.normalization_input_image = True
        elif "upper_quantile_normalization" in self.config.keys():
            self.normalization_input_image = True
        else:
            self.normalization_input_image = False
            self.log("No normalization will be performed")

        if self.normalization_input_image:
            self.log("Normalization of the input image will be performed.")

            # get lower quantile normalization range
            self.lower_quantile_normalization_input_image = 0
            self.upper_quantile_normalization_input_image = 1

            if isinstance(self.config["lower_quantile_normalization"], float):
                self.lower_quantile_normalization_input_image = self.config["lower_quantile_normalization"]

            elif isinstance(self.config["lower_quantile_normalization"], list):
                if len(self.config["lower_quantile_normalization"]) != self.input_image.shape[0]:
                    raise ValueError(
                        "Number of specified normalization ranges for lower quantile normalization does not match the number of input channels."
                    )
                self.lower_quantile_normalization_input_image = dict(
                    zip(
                        range(self.input_image.shape[0]),
                        self.config["lower_quantile_normalization"],
                        strict=False,
                    )
                )
            else:
                param = self.config["lower_quantile_normalization"]
                raise ValueError(
                    f"lower_quantile_normalization must be either a float or a list of floats. Instead recieved {param}"
                )

            # get upper quantile normalization range
            if isinstance(self.config["upper_quantile_normalization"], float):
                self.upper_quantile_normalization_input_image = self.config["upper_quantile_normalization"]

            elif isinstance(self.config["upper_quantile_normalization"], list):
                if len(self.config["upper_quantile_normalization"]) != self.input_image.shape[0]:
                    raise ValueError(
                        "Number of specified normalization ranges for upper quantile normalization does not match the number of input channels."
                    )
                self.upper_quantile_normalization_input_image = dict(
                    zip(
                        range(self.input_image.shape[0]),
                        self.config["upper_quantile_normalization"],
                        strict=False,
                    )
                )
            else:
                param = self.config["upper_quantile_normalization"]
                raise ValueError(
                    f"upper_quantile_normalization must be either a float or a list of floats. Instead recieved {param}"
                )

            # check that the normalization ranges are of the same type otherwise this will result in issues
            assert type(self.lower_quantile_normalization_input_image) == type(  # noqa: E721
                self.upper_quantile_normalization_input_image
            )  # these need to be the same types! So we need to circumvent the ruff linting rules here

        # check if median filtering is required
        if "median_filter_size" in self.config.keys():
            self.median_filtering_input_image = True
            self.median_filter_size_input_image = self.config["median_filter_size"]
        else:
            self.median_filtering_input_image = False

        # setup nucleus segmentation
        if "nucleus_segmentation" in self.config.keys():
            self.segment_nuclei = True

            # check for additional normalization before segmentation
            # set default values to 0 and 1 so no normalization occurs
            self.normalization_nucleus_lower_quantile = 0
            self.normalization_cytosol_upper_quantile = 1
            self.normalization_nucleus_segmentation = False

            if "lower_quantile_normalization" in self.config["nucleus_segmentation"]:
                self.normalization_nucleus_segmentation = True
                self.normalization_nucleus_lower_quantile = self.config["nucleus_segmentation"][
                    "lower_quantile_normalization"
                ]
            if "upper_quantile_normalization" in self.config["nucleus_segmentation"]:
                self.normalization_nucleus_segmentation = True
                self.normalization_nucleus_upper_quantile = self.config["nucleus_segmentation"][
                    "upper_quantile_normalization"
                ]

            # check if nuclei should be filtered based on size
            self._check_for_size_filtering(mask_types=["nucleus"])

            # check if nuclei should be filtered based on contact
            if "contact_filter" in self.config["nucleus_segmentation"]:
                self.contact_filter_nuclei = True
                self.contact_filter_nuclei_threshold = self.config["nucleus_segmentation"]["contact_filter"]
            else:
                self.contact_filter_nuclei = False

        if "cytosol_segmentation" in self.config.keys():
            self.segment_cytosol = True
            self.config_cytosol_segmentation = self.config["cytosol_segmentation"]
            # check for additional normalization before segmentation
            # set default values to 0 and 1 so no normalization occurs
            self.normalization_cytosol_lower_quantile = 0
            self.normalization_cytosol_upper_quantile = 1
            self.normalization_cytosol_segmentation = False

            if "lower_quantile_normalization" in self.config["cytosol_segmentation"]:
                self.normalization_cytosol_segmentation = True
                self.normalization_cytosol_lower_quantile = self.config["cytosol_segmentation"][
                    "lower_quantile_normalization"
                ]
            if "upper_quantile_normalization" in self.config_cytosol_segmentation.keys():
                self.normalization_cytosol_segmentation = True
                self.normalization_cytosol_upper_quantile = self.config["cytosol_segmentation"][
                    "upper_quantile_normalization"
                ]

            # check if cytosol should be filtered based on size
            self._check_for_size_filtering(mask_types=["cytosol"])

            # check if cytosol should be filtered based on contact
            if "contact_filter" in self.config["cytosol_segmentation"]:
                self.contact_filter_cytosol = True
                self.contact_filter_cytosol_threshold = self.config["cytosol_segmentation"]["contact_filter"]
            else:
                self.contact_filter_cytosol = False

            # check if cytosol should be filtered based on matching to nuclei
            self._check_for_mask_matching_filtering()

    def _nucleus_segmentation(self, input_image, debug: bool = False):
        if self.normalization_nucleus_segmentation:
            lower = self.normalization_nucleus_lower_quantile
            upper = self.normalization_nucleus_upper_quantile

            self.log(f"Percentile normalization of nucleus input image to range {lower}, {upper}")
            input_image = self._normalize_image(input_image, lower, upper, debug=debug)

        # perform thresholding to generate  a mask of the nuclei

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if "threshold" in self.config["nucleus_segmentation"] and "median_block" in self.config["nucleus_segmentation"]:
            threshold = self.config["nucleus_segmentation"]["threshold"]
            self.log(f"Using local thresholding with a threshold of {threshold} to calculate nucleus mask.")
            self.maps["nucleus_segmentation"] = segment_local_threshold(
                input_image,
                dilation=self.config["nucleus_segmentation"]["dilation"],
                thr=self.config["nucleus_segmentation"]["threshold"],
                median_block=self.config["nucleus_segmentation"]["median_block"],
                min_distance=self.config["nucleus_segmentation"]["min_distance"],
                peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"],
                speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"],
                median_step=self.config["nucleus_segmentation"]["median_step"],
                debug=self.debug,
            )
        else:
            self.log("Using global otsu to calculate threshold for nucleus mask generation.")
            self.maps["nucleus_segmentation"] = segment_global_threshold(
                input_image,
                dilation=self.config["nucleus_segmentation"]["dilation"],
                min_distance=self.config["nucleus_segmentation"]["min_distance"],
                peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"],
                speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"],
                debug=self.debug,
            )

        # save the thresholding approach as a mask
        self.maps["nucleus_mask"] = np.clip(self.maps["nucleus_segmentation"], 0, 1)

        # calculate nucleus centroids
        # important to do this before any filtering! Otherwise we wont know where to start looking for cytosols
        centers, _, _ = numba_mask_centroid(self.maps["nucleus_segmentation"])
        self.nucleus_centers = centers

        # ensure all edge labels are removed
        self.maps["nucleus_segmentation"] = remove_edge_labels(self.maps["nucleus_segmentation"])

        if self.filter_size:
            self.maps["nucleus_segmentation"] = self._perform_size_filtering(
                self.maps["nucleus_segmentation"],
                self.nucleus_thresholds,  # type: ignore
                self.nucleus_confidence_interval,  # type: ignore
                "nucleus",
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

        if self.contact_filter_nuclei:
            if self.debug:
                n_classes = len(set(np.unique(self.maps["nucleus_segmentation"])) - {0})

            self.maps["nucleus_segmentation"] = contact_filter(
                self.maps["nucleus_segmentation"],
                threshold=self.contact_filter_nuclei_threshold,
                reindex=False,
            )

            if self.debug:
                n_classes_post = len(set(np.unique(self.maps["nucleus_segmentation"])) - {0})
                self.log(f"Filtered out {n_classes - n_classes_post} nuclei due to contact filtering.")

    def _cytosol_segmentation(self, input_image, debug: bool = False):
        if not self.segment_nuclei:
            raise ValueError("Nucleus segmentation must be performed to be able to perform a cytosol segmentation.")

        if self.normalization_cytosol_segmentation:
            lower = self.normalization_cytosol_lower_quantile
            upper = self.normalization_cytosol_upper_quantile

            input_image = self._normalize_image(input_image, lower, upper, debug=debug)

        # perform thresholding to generate a mask of the cytosol

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if "threshold" in self.config["cytosol_segmentation"]:
            cytosol_mask = input_image < self.config["cytosol_segmentation"]["threshold"]
        else:
            self.log("No treshold for cytosol segmentation defined, global otsu will be used.")
            cytosol_mask = input_image < global_otsu(input_image)

        self._clear_cache(vars_to_delete=[input_image])

        # remove the nucleus mask from the cytosol mask
        cytosol_mask = cytosol_mask.astype(float)
        cytosol_mask -= self.maps["nucleus_mask"]
        cytosol_mask = np.clip(cytosol_mask, 0, 1)

        # Apply dilation and erosion
        cytosol_mask = dilation(cytosol_mask, footprint=disk(self.config["cytosol_segmentation"]["erosion"]))
        cytosol_mask = binary_erosion(
            cytosol_mask,
            footprint=disk(self.config["cytosol_segmentation"]["dilation"]),
        )

        self.maps["cytosol_mask"] = cytosol_mask

        if self.debug:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(self.maps["cytosol_mask"], cmap="magma")
            plt.axis("off")
            plt.title("Cytosol Mask")
            fig_path = os.path.join(self.directory, "cytosol_mask.png")
            fig.savefig(fig_path)

        self._clear_cache(vars_to_delete=[cytosol_mask])

        # calculate potential map for cytosol segmentation
        potential_map = (
            self.maps["median_corrected"][1] - np.quantile(self.maps["median_corrected"][1], 0.02)
        ) / np.quantile(self.maps["median_corrected"][1], 0.98)
        potential_map = np.clip(potential_map, 0, 1)

        # subtract nucleus mask from potential map
        potential_map = np.clip(potential_map - self.maps["nucleus_mask"], 0, 1)
        potential_map = 1 - potential_map

        # enhance potential map to generate speedmap
        min_clip = self.config["cytosol_segmentation"]["min_clip"]
        max_clip = self.config["cytosol_segmentation"]["max_clip"]
        potential_map = (np.clip(potential_map, min_clip, max_clip) - min_clip) / (max_clip - min_clip)
        potential_map = (potential_map * 0.9 + 0.1).astype(float)

        self.maps["potential_map"] = potential_map

        if self.debug:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(self.maps["potential_map"], cmap="magma")
            plt.axis("off")
            plt.title("Potential Map")
            fig_path = os.path.join(self.directory, "potential_map.png")
            fig.savefig(fig_path)

        self.log("Cytosol Potential Mask generated.")
        self._clear_cache(vars_to_delete=[potential_map])

        # perform fast marching and watershed segmentation
        fmm_marker = np.ones_like(self.maps["median_corrected"][0])
        px_center = np.round(self.nucleus_centers).astype(np.uint64)

        for center in px_center:
            fmm_marker[center[0], center[1]] = 0

        fmm_marker = np.ma.MaskedArray(fmm_marker, self.maps["cytosol_mask"])
        travel_time = skfmm_travel_time(fmm_marker, self.maps["potential_map"])

        if not isinstance(travel_time, np.ma.core.MaskedArray):
            raise TypeError(
                "Travel_time for WGA based segmentation returned no MaskedArray. This is most likely due to missing WGA background determination."
            )

        self.maps["travel_time"] = travel_time.filled(fill_value=np.max(travel_time))

        marker = np.zeros_like(self.maps["median_corrected"][1])

        for center in px_center:
            marker[center[0], center[1]] = self.maps["nucleus_segmentation"][center[0], center[1]]

        cytosol_labels = watershed(
            self.maps["travel_time"],
            marker.astype(np.int64),
            mask=(self.maps["cytosol_mask"] == 0).astype(np.int64),
        )

        cytosol_segmentation = np.where(self.maps["cytosol_mask"] > 0.5, 0, cytosol_labels)

        # ensure all edge labels are removed
        cytosol_segmentation = remove_edge_labels(cytosol_segmentation)
        self.maps["cytosol_segmentation"] = cytosol_segmentation

        if self.filter_size:
            self.maps["cytosol_segmentation"] = self._perform_size_filtering(
                self.maps["cytosol_segmentation"],
                self.cytosol_thresholds,  # type: ignore
                self.cytosol_confidence_interval,  # type: ignore
                "cytosol",
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

        if self.contact_filter_cytosol:
            if self.debug:
                n_classes = len(set(np.unique(self.maps["cytosol_segmentation"])) - {0})

            self.maps["cytosol_segmentation"] = contact_filter(
                self.maps["cytosol_segmentation"],
                threshold=self.contact_filter_cytosol_threshold,
                reindex=False,
            )

            if self.debug:
                n_classes_post = len(set(np.unique(self.maps["cytosol_segmentation"])) - {0})
                self.log(f"Filtered out {n_classes - n_classes_post} cytosols due to contact filtering.")

        unique_cytosol_ids = set(np.unique(self.maps["cytosol_segmentation"])) - {0}

        # remove any ids from nucleus mask that dont have a cytosol mask
        self.maps["nucleus_segmentation"][~np.isin(self.maps["nucleus_segmentation"], list(unique_cytosol_ids))] = 0


class WGASegmentation(_ClassicalSegmentation):
    N_MASKS = 2
    N_INPUT_CHANNELS = 2
    MASK_NAMES = ["nucleus", "cytosol"]
    DEFAULT_NUCLEI_CHANNEL_IDS = [0]
    DEFAULT_CYTOSOL_CHANNEL_IDS = [1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        segmentation = np.stack([self.maps["nucleus_segmentation"], self.maps["cytosol_segmentation"]]).astype(
            self.DEFAULT_SEGMENTATION_DTYPE
        )

        return segmentation

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()
        self._get_processing_parameters()

        # intialize maps for storing intermediate results
        self.maps = {}

        input_image = self._transform_input_image(input_image)

        self.maps["input_image"] = input_image

        # normalize input
        if self.normalization_input_image:
            self.maps["normalized"] = self._normalize_image(
                input_image,
                self.lower_quantile_normalization_input_image,
                self.upper_quantile_normalization_input_image,
                debug=self.debug,
            )
            # update input image to normalized image
            input_image = self.maps["normalized"]

        if self.median_filtering_input_image:
            self.maps["median_corrected"] = self._median_correct_image(
                input_image, self.median_filter_size_input_image, debug=self.debug
            )
            # update input image to median corrected image
            input_image = self.maps["median_corrected"]

        start_segmentation = timeit.default_timer()
        if self.segment_nuclei:
            image = input_image[0]
            self._nucleus_segmentation(image, debug=self.debug)

        if self.segment_cytosol:
            image = self.maps["input_image"][1]
            self._cytosol_segmentation(image, debug=self.debug)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        if self.debug:
            self._visualize_final_masks()

        all_classes = list(set(np.unique(self.maps["nucleus_segmentation"])) - {0})
        segmentation = self._finalize_segmentation_results()  # type: ignore

        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self.total_time = timeit.default_timer() - total_time_start


class ShardedWGASegmentation(ShardedSegmentation):
    method = WGASegmentation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DAPISegmentation(_ClassicalSegmentation):
    N_MASKS = 1
    N_INPUT_CHANNELS = 1
    MASK_NAMES = ["nucleus"]
    DEFAULT_NUCLEI_CHANNEL_IDS = [0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        segmentation = np.stack([self.maps["nucleus_segmentation"]]).astype(self.DEFAULT_SEGMENTATION_DTYPE)

        return segmentation

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()
        self._get_processing_parameters()

        # intialize maps for storing intermediate results
        self.maps = {}

        input_image = self._transform_input_image(input_image)
        self.maps["input_image"] = input_image.copy()

        # normalize input
        if self.normalization_input_image:
            self.maps["normalized"] = self._normalize_image(
                input_image,
                self.lower_quantile_normalization_input_image,
                self.upper_quantile_normalization_input_image,
                debug=self.debug,
            )
            # update input image to normalized image
            input_image = self.maps["normalized"]

        if self.median_filtering_input_image:
            self.maps["median_corrected"] = self._median_correct_image(
                input_image, self.median_filter_size_input_image, debug=self.debug
            )
            # update input image to median corrected image
            input_image = self.maps["median_corrected"]

        start_segmentation = timeit.default_timer()
        if self.segment_nuclei:
            self._nucleus_segmentation(input_image[0], debug=self.debug)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        all_classes = list(set(np.unique(self.maps["nucleus_segmentation"])) - {0})
        segmentation = self._finalize_segmentation_results()

        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self.total_time = timeit.default_timer() - total_time_start


class ShardedDAPISegmentation(ShardedSegmentation):
    method = DAPISegmentation
