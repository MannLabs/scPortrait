from sparcscore.pipeline.segmentation import (
    Segmentation,
    ShardedSegmentation,
    TimecourseSegmentation,
    MultithreadedSegmentation,
)
from sparcscore.processing.preprocessing import percentile_normalization, downsample_img
from sparcscore.processing.filtering import SizeFilter, MatchNucleusCytosolIds
from sparcscore.processing.utils import visualize_class
from sparcscore.processing.segmentation import (
    segment_local_threshold,
    segment_global_threshold,
    numba_mask_centroid,
    contact_filter,
    size_filter,
    _class_size,
    global_otsu,
    remove_edge_labels,
    _return_edge_labels,
)

import os
import sys
import gc
import time

# for typing
import xarray
from typing import Tuple, Union, List

import numpy as np
import matplotlib.pyplot as plt

# WGA Segmentation
from skfmm import travel_time as skfmm_travel_time
from skimage.filters import median
from skimage.segmentation import watershed
from skimage.color import label2rgb

# mask processing
from skimage.morphology import binary_erosion, disk, dilation, erosion

# multiprocessing/out-of-memory processing
import multiprocessing
from alphabase.io import tempmmap

# for cellpose segmentation
import torch
from cellpose import models


class BaseSegmentation(Segmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def return_empty_mask(self, input_image):
        n_channels, x, y = input_image.shape
        self.save_segmentation(input_image, np.zeros((2, x, y)), [])

    def _clear_cache(self, vars_to_delete=None):
        """Helper function to help clear memory usage. Mainly relevant for GPU based segmentations."""

        # delete all specified variables
        if vars_to_delete is not None:
            for var in vars_to_delete:
                del var

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def _check_seg_dtype(self, mask: np.array, mask_name: str) -> np.array:
        if not isinstance(mask, self.DEFAULT_SEGMENTATION_DTYPE):
            Warning(
                f"{mask_name} segmentation map is not of the correct dtype. \n Forcefully converting {mask.dtype} to {self.DEFAULT_SEGMENTATION_DTYPE}. \n This could lead to unexpected behaviour."
            )

            return mask.astype(self.DEFAULT_SEGMENTATION_DTYPE)

        else:
            return mask

    #### Downsampling ####
    def _get_downsampling_parameters(self) -> None:
        self.N = self.config["downsampling_factor"]

        if "smoothing_kernel_size" in self.config.keys():
            self.smoothing_kernel_size = self.config["smoothing_kernel_size"]

            if self.smoothing_kernel_size > self.N:
                self.log(
                    "Warning: Smoothing Kernel size is larger than the downsampling factor. This can lead to issues during smoothing where segmentation masks are lost. Please ensure to double check your results."
                )

        else:
            self.log(
                "Smoothing Kernel size not explicitly defined. Will calculate a default value based on the downsampling factor."
            )
            self.smoothing_kernel_size = self.N

        return None

    def _calculate_padded_image_size(self, img: np.ndarray) -> None:
        """prepare metrics for image downsampling. Calculates image padding required for downsampling and returns
        metrics for this as well as resulting downsampled image size.
        """

        self.input_image_size = img.shape

        # check if N fits perfectly into image shape if not calculate how much we need to pad
        _, x, y = self.input_image_size

        if x % self.N == 0:
            pad_x = (0, 0)
        else:
            pad_x = (0, self.N - x % self.N)

        if y % self.N == 0:
            pad_y = (0, 0)
        else:
            pad_y = (0, self.N - y % self.N)

        # calculate resulting image size for use when e.g. inititalizing empty arrays to save results to
        padded_image_size = (2, self.input_image_size[1] + pad_x[1], self.input_image_size[2] + pad_y[1])

        self.expected_padded_image_size = padded_image_size
        self.pad_x = pad_x
        self.pad_y = pad_y

        return None

    def _downsample_image(self, img: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Downsample image by a factor of N. Before downsampling this function will pad the image to ensure its compatible with the selected kernel size.

        Parameters
        ----------
        img
            image to be downsampled

        Returns
        -------
        downsampled image

        """

        self.log(
            f"Performing image padding to ensure that image is compatible with selected downsample kernel size of {self.N}."
        )

        # track original image size
        self.original_image_size = img.shape

        # perform image padding to ensure that image is compatible with downsample kernel size
        img = np.pad(img, ((0, 0), self.pad_x, self.pad_y))
        self.padded_image_size = img.shape

        if debug:
            self.log(
                f"Original image had size {self.expected_padded_image_size}, padded image is {self.padded_image_size}"
            )

        # sanity check to make sure padding worked as we wanted
        if self.expected_padded_image_size != self.padded_image_size:
            Warning(
                f"Expected a padded image of size {self.expected_padded_image_size} but got {self.padded_image_size}. Padding did not work as expted"
            )
            sys.exit(
                "Error. Image padding did not work as expected and returned an array of differing size."
            )

        self.log(f"Downsampling image by a factor of {self.N}x{self.N}")

        # actually perform downsampling
        img = downsample_img(img, N=self.N)

        self.downsampled_image_size = img.shape

        if debug:
            self.log(f"Downsampled image size {self.downsampled_image_size}")

        return img

    def _rescale_downsampled_mask(self, mask: np.ndarray, mask_name: str) -> np.ndarray:
        input_mask = mask.copy()

        # get number of objects in mask for sanity checking
        n_classes = len(np.unique(mask))

        # rescale segmentations masks to padded image size
        mask = mask.repeat(self.N, axis=0).repeat(self.N, axis=1)

        # perform erosion and dilation for smoothing
        mask = erosion(mask, footprint=disk(self.smoothing_kernel_size))
        mask = dilation(
            mask, footprint=disk(self.smoothing_kernel_size + 1)
        )  # dilate 1 more than eroded to ensure that we do not lose any pixels

        # sanity check to make sure that smoothing does not remove masks
        if len(np.unique(mask)) != n_classes:
            Warning(
                "Number of objects in segmentation mask changed after smoothing. This should not happen. Ensure that you have chosen adequate smoothing parameters."
            )

            self.log(
                f"Will recalculate upsampling of {mask_name} mask with lower smoothing value to prevent the number of segmented objects from changing. Please ensure to double check your results."
            )

            smoothing_kernel_size = self.smoothing_kernel_size

            while len(np.unique(mask)) != n_classes:
                smoothing_kernel_size = smoothing_kernel_size - 1

                if smoothing_kernel_size == 0:
                    # if we reach 0 then we do not perform any smoothing
                    # repeat rescaling of the original mask

                    mask = input_mask
                    mask = mask.repeat(self.N, axis=0).repeat(self.N, axis=1)
                    self.log(f"Did not perform smoothing of {mask_name} mask.")

                    break

                else:
                    mask = input_mask
                    mask = mask.repeat(self.N, axis=0).repeat(self.N, axis=1)

                    # perform erosion and dilation for smoothing
                    mask = erosion(mask, footprint=disk(smoothing_kernel_size))

                    mask = dilation(
                        mask, footprint=disk(smoothing_kernel_size + 1)
                    )  # dilate 1 more than eroded to ensure that we do not lose any pixels

            self.log(
                f"Recalculation of {mask_name} mask successful with smoothing kernel size of {smoothing_kernel_size}."
            )

        # remove padding from mask
        x_trim = self.padded_image_size[1] - self.original_image_size[1]
        y_trim = self.padded_image_size[2] - self.original_image_size[2]

        # sanity check to ensure that we are removing what we addded
        assert x_trim == self.pad_x[1]
        assert y_trim == self.pad_y[1]

        # actually perform trimming
        if x_trim > 0:
            if y_trim > 0:
                mask = mask[:, :-x_trim, :-y_trim]
            else:
                mask = mask[:, :-x_trim, :]
        else:
            if y_trim > 0:
                mask = mask[:, :, :-y_trim]
            else:
                mask = mask

        # check that mask has the correct shape and matches to input image
        assert (mask.shape[1] == self.original_image_size[1]) and (
            mask.shape[2] == self.original_image_size[2]
        )

        return mask

    ##### Filtering Functions #####

    # 1. Size Filtering
    def _check_for_size_filtering(self, mask_types=["nucleus", "cytosol"]) -> None:
        """
        Check if size filtering should be performed on the masks.
        If size filtering is turned on, the thresholds for filtering are loaded from the config file.
        """

        if "filter_masks_size" in self.config.keys():
            self.filter_size = self.config["filter_masks_size"]
        else:
            # default behaviour is this is turned off filtering can always be performed later and this preserves the whole segmentation mask
            self.filter_size = False

        # load parameters for cellsize filtering
        if self.filter_size:
            for mask_type in mask_types:
                thresholds, confidence_interval = self._get_params_cellsize_filtering(
                    type=mask_type
                )
            self[f"{mask_type}_thresholds"] = thresholds
            self[f"{mask_type}_confidence_interval"] = confidence_interval

    def _get_params_cellsize_filtering(
        self, type
    ) -> Tuple[Union[Tuple[float], None], Union[float, None]]:
        self.absolute_filter_status = False

        if "min_size" in self.config[f"{type}_segmentation"].keys():
            min_size = self.config[f"{type}_segmentation"]["min_size"]
            absolute_filter_status = True
        else:
            min_size = None

        if "max_size" in self.config[f"{type}_segmentation"].keys():
            max_size = self.config[f"{type}_segmentation"]["max_size"]
            self.absolute_filter_status = True
        else:
            max_size = None

        if absolute_filter_status:
            thresholds = [min_size, max_size]
            return (thresholds, None)
        else:
            thresholds = None

            # get confidence intervals to automatically calculate thresholds
            if "confidence_interval" in self.config[f"{type}_segmentation"].keys():
                confidence_interval = self.config[f"{type}_segmentation"][
                    "confidence_interval"
                ]
            else:
                # get default value
                self.log(
                    f"No confidence interval specified for {type} mask filtering, using default value of 0.95"
                )
                confidence_interval = 0.95

            return (thresholds, confidence_interval)

    def _perform_size_filtering(
        self,
        mask: np.array,
        thresholds: Tuple[float] | None,
        confidence_interval: float,
        mask_name: str,
        log: bool = True,
        debug: bool = False,
    ) -> np.array:
        """
        Remove elements from mask based on a size filter.

        Parameters
        ----------
        mask
            mask to be filtered
        """
        start_time = time()

        if self.debug:
            unfiltered_mask = mask.copy()

        if thresholds is not None:
            self.log(
                f"Performing filtering of {mask_name} with specified thresholds {thresholds} from config file."
            )
        else:
            self.log(
                f"Automatically calculating thresholds for filtering of {mask_name} based on a fitted normal distribution with a confidence interval of {confidence_interval * 100}%."
            )

        filter = SizeFilter(
            label=mask_name,
            log=log,
            plot_qc=self.debug,
            directory=self.directory,
            confidence_interval=confidence_interval,
            filter_threshold=thresholds,
        )

        filtered_mask = filter.filter(mask)
        self.log(
            f"Removed {len(filter.ids_to_remove)} nuclei as they fell outside of the threshold range {filter.filter_threshold}."
        )

        if self.debug:
            # plot mask before and after filtering to visualize the results

            fig, axs = plt.subplots(1, 2, figsize=(8, 8))
            axs[0].imshow(unfiltered_mask[0])
            axs[0].axis("off")
            axs[0].set_title("before filtering", fontsize=6)

            axs[1].imshow(filtered_mask[0])
            axs[1].axis("off")
            axs[1].set_title("after filtering", fontsize=6)
            fig.tight_layout()

            fig_path = os.path.join(self.directory, f"{mask_name}_size_filtering.png")
            fig.savefig(fig_path)

        end_time = time()

        self.log(
            f"Total time to perform {mask_name} size filtering: {end_time - start_time} seconds"
        )

        return filtered_mask

    # 2. matching masks
    def _check_for_mask_matching_filtering(self) -> None:
        """Check to see if the masks should be filtered for matching nuclei/cytosols within the segmentation run."""

        # check to see if the cells should be filtered for matching nuclei/cytosols within the segmentation run
        if "match_masks" in self.config.keys():
            self.filter_match_masks = self.config["match_masks"]
            if "filtering_threshold_mask_matching" in self.config.keys():
                self.mask_matching_filtering_threshold = self.config[
                    "filtering_threshold_mask_matching"
                ]
            else:
                self.mask_matching_filtering_threshold = 0.95  # set default parameter

        else:
            # add deprecation warning for old config setup
            if "filter_status" in self.config.keys():
                Warning(
                    "filter_status is deprecated, please use match_masks instead Will not perform filtering."
                )

            # default behaviour that this filtering should be performed, otherwise another additional step is required before extraction
            self.filter_match_masks = True

    def _perform_mask_matching_filtering(
        self,
        nucleus_mask: np.array,
        cytosol_mask: np.array,
        filtering_threshold: float,
        debug: bool = False,
    ) -> Tuple[np.array, np.array]:
        """
        Match the nuclei and cytosol masks to ensure that the same cells are present in both masks.

        Parameters
        ----------
        nucleus_mask
            nucleus mask to be matched
        cytosol_mask
            cytosol mask to be matched
        """
        start_time = time.time()
        self.log("Performing filtering to match Cytosol and Nucleus IDs.")

        if debug:
            masks_nucleus_unfiltered = nucleus_mask.copy()
            masks_cytosol_unfiltered = cytosol_mask.copy()

        # perform filtering to remove cytosols which do not have a corresponding nucleus
        filter = MatchNucleusCytosolIds(filtering_threshold=filtering_threshold)
        masks_nucleus, masks_cytosol = filter.filter(
            nucleus_mask=nucleus_mask, cytosol_mask=cytosol_mask
        )

        self.log(
            f"Removed {len(filter.nuclei_discard_list)} nuclei and {len(filter.cytosol_discard_list)} cytosols due to filtering."
        )
        self.log(
            f"After filtering, {len(filter.nucleus_lookup_dict)} matching nuclei and cytosol masks remain."
        )

        if debug:
            # plot nucleus and cytosol masks before and after filtering
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            axs[0, 0].imshow(masks_nucleus_unfiltered[0])
            axs[0, 0].axis("off")
            axs[0, 0].set_title("before filtering", fontsize=6)
            axs[0, 1].imshow(masks_nucleus[0])
            axs[0, 1].axis("off")
            axs[0, 1].set_title("after filtering", fontsize=6)

            axs[1, 0].imshow(masks_cytosol_unfiltered[0])
            axs[1, 0].axis("off")
            axs[1, 1].imshow(masks_cytosol[0])
            axs[1, 1].axis("off")
            fig.tight_layout()

            fig_path = os.path.join(self.directory, "mask_matching_filtering.png")
            fig.savefig(fig_path)

            # clearup memory
            self._clear_cache(
                vars_to_delete=[fig, masks_cytosol_unfiltered, masks_nucleus_unfiltered]
            )

        self.log(
            "Total time to perform nucleus and cytosol mask matching filtering: {:.2f} seconds".format(
                time.time() - start_time
            )
        )

        return masks_nucleus, masks_cytosol


class _cellpose_segmentation(BaseSegmentation):
    def _read_cellpose_model(
        self, modeltype: str, name: str, gpu: str, device
    ) -> models.Cellpose:
        """
        Reads cellpose model based on the modeltype and name. Will load to GPU if available as specified in self._use_gpu

        Parameters
        ----------
        modeltype
            either "pretrained" or "custom" depending on the model to load
        name
            name of the model to load

        Returns
        -------
        cellpose model

        """
        if modeltype == "pretrained":
            model = models.Cellpose(model_type=name, gpu=gpu, device=device)
        elif modeltype == "custom":
            model = models.CellposeModel(pretrained_model=name, gpu=gpu, device=device)
        return model

    def _load_model(
        self, model_type: str, gpu: str, device
    ) -> Tuple[float, models.Cellpose]:
        """
        Loads cellpose model

        Parameters
        ----------
        model_type
            either "cytosol" or "nucleus" depending on the model to load

        Returns
        -------
        tuple of expected diameter and the cellpose model
        """

        # load correct segmentation model for cytosol
        if "model" in self.config[f"{model_type}_segmentation"].keys():
            model_name = self.config[f"{model_type}_segmentation"]["model"]
            model = self._read_cellpose_model(
                "pretrained", model_name, gpu=gpu, device=device
            )

        elif "model_path" in self.config[f"{model_type}_segmentation"].keys():
            model_name = self.config[f"{model_type}_segmentation"]["model_path"]
            model = self._read_cellpose_model(
                "custom", model_name, gpu=gpu, device=device
            )

        if "diameter" in self.config[f"{model_type}_segmentation"].keys():
            diameter = self.config[f"{model_type}_segmentation"]["diameter"]
        else:
            diameter = None

        self.log(f"Segmenting {model_type} using the following model: {model_name}")
        return diameter, model

    def _check_input_image_dtype(self, input_image):
        if input_image.dtype != self.DEFAULT_IMAGE_DTYPE:
            if isinstance(input_image.dtype, int):
                ValueError(
                    "Default image dtype is no longer int. Cellpose expects int inputs. Please contact developers."
                )
            else:
                ValueError(
                    "Image is not of type uint16, cellpose segmentation expects int input images."
                )

    def _check_gpu_status(self):
        """
        Checks and updates the GPU status.
        If a multi-GPU setup is used, the function checks the current process and returns the GPU id to use for the segmentation.
        If no GPUs are available, the function defaults to CPU.
        """

        # get GPU information if run with workers
        try:
            current = multiprocessing.current_process()
            cpu_name = current.name
            gpu_id_list = current.gpu_id_list
            cpu_id = int(cpu_name[cpu_name.find("-") + 1 :]) - 1

            # track gpu_id and update GPU status
            self.gpu_id = gpu_id_list[cpu_id]
            self.status = "multi_GPU"

        except Exception:
            # default to single GPU
            self.gpu_id = 0
            self.status = "potentially_single_GPU"

        # check if cuda GPU is available
        if torch.cuda.is_available():
            if self.gpu_status == "multi_GPU":
                self.use_GPU = f"cuda:{self.gpu_id}"
                self.device = torch.device(self.use_GPU)
            else:
                self.use_GPU = True
                self.device = torch.device(
                    "cuda"
                )  # dont need to specify id, saying cuda will default to the one thats avaialable

        # check if MPS is available
        elif torch.backends.mps.is_available():
            self.use_GPU = True
            self.device = torch.device("mps")

        # default to CPU
        else:
            self.use_GPU = False
            self.device = torch.device("cpu")

        self.log(
            f"GPU Status for segmentation is {self.use_GPU} and will segment using the following device {self.device}."
        )


###### CLASSICAL SEGMENTATION METHODS #####

class WGASegmentation(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalization(self, input_image):
        self.log("Starting with normalized map")
        if isinstance(self.config["lower_quantile_normalization"], float):
            self.log("Normalizing each channel to the same range")
            self.maps["normalized"] = percentile_normalization(
                input_image,
                self.config["lower_quantile_normalization"],
                self.config["upper_quantile_normalization"],
            )
        else:
            if len(self.config["lower_quantile_normalization"]) != input_image.shape[0]:
                sys.exit("please specify normalization range for each input channel")
            self.log("Normalizing each channel individually.")
            normalized = []
            for i in range(input_image.shape[0]):
                lower = self.config["lower_quantile_normalization"][i]
                upper = self.config["upper_quantile_normalization"][i]
                normalized.append(
                    percentile_normalization(
                        input_image[i],
                        lower,
                        upper,
                    )
                )
            self.maps["normalized"] = np.array(normalized)
        self.save_map("normalized")
        self.log("Normalized map created")

    def _median_calculation(self):
        self.log("Started with median map")
        self.maps["median"] = np.copy(self.maps["normalized"])

        for i, channel in enumerate(self.maps["median"]):
            self.maps["median"][i] = median(
                channel, disk(self.config["median_filter_size"])
            )

        self.save_map("median")
        self.log("Median map created")

    def _nucleus_segmentation(self):
        self.log("Generating thresholded nucleus map.")

        nucleus_map_tr = percentile_normalization(
            self.maps["median"][0],
            self.config["nucleus_segmentation"]["lower_quantile_normalization"],
            self.config["nucleus_segmentation"]["upper_quantile_normalization"],
        )

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if (
            "threshold" in self.config["nucleus_segmentation"]
            and "median_block" in self.config["nucleus_segmentation"]
        ):
            self.maps["nucleus_segmentation"] = segment_local_threshold(
                nucleus_map_tr,
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
            self.log(
                "No threshold or median_block for nucleus segmentation defined, global otsu will be used."
            )
            self.maps["nucleus_segmentation"] = segment_global_threshold(
                nucleus_map_tr,
                dilation=self.config["nucleus_segmentation"]["dilation"],
                min_distance=self.config["nucleus_segmentation"]["min_distance"],
                peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"],
                speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"],
                debug=self.debug,
            )

        del nucleus_map_tr

        self.maps["nucleus_mask"] = np.clip(self.maps["nucleus_segmentation"], 0, 1)

        # ensure all edge labels are removed
        self.maps["nucleus_segmentation"] = remove_edge_labels(
            self.maps["nucleus_segmentation"]
        )

        self.save_map("nucleus_segmentation")
        self.log(
            "Nucleus mask map created with {} elements".format(
                np.max(self.maps["nucleus_segmentation"])
            )
        )

    def _filter_nuclei_classes(self):
        # filter nuclei based on size and contact
        center_nuclei, length = _class_size(
            self.maps["nucleus_segmentation"], debug=self.debug
        )
        all_classes = np.unique(self.maps["nucleus_segmentation"])

        # ids of all nucleis which are unconnected and can be used for further analysis
        labels_nuclei_unconnected = contact_filter(
            self.maps["nucleus_segmentation"],
            threshold=self.config["nucleus_segmentation"]["contact_filter"],
            reindex=False,
        )
        classes_nuclei_unconnected = np.unique(labels_nuclei_unconnected)

        self.log(
            "Filtered out due to contact limit: {} ".format(
                len(all_classes) - len(classes_nuclei_unconnected)
            )
        )

        labels_nuclei_filtered = size_filter(
            self.maps["nucleus_segmentation"],
            limits=[
                self.config["nucleus_segmentation"]["min_size"],
                self.config["nucleus_segmentation"]["max_size"],
            ],
        )

        classes_nuclei_filtered = np.unique(labels_nuclei_filtered)

        self.log(
            "Filtered out due to size limit: {} ".format(
                len(all_classes) - len(classes_nuclei_filtered)
            )
        )

        filtered_classes = set(classes_nuclei_unconnected).intersection(
            set(classes_nuclei_filtered)
        )
        self.log("Filtered out: {} ".format(len(all_classes) - len(filtered_classes)))

        if self.debug:
            self._plot_nucleus_size_distribution(length)
            self._visualize_nucleus_segmentation(
                classes_nuclei_unconnected, classes_nuclei_filtered
            )

        return (all_classes, filtered_classes, center_nuclei)

    def _cellmembrane_mask_calculation(self):
        self.log("Started with WGA mask map")

        if "wga_background_image" in self.config["wga_segmentation"]:
            if self.config["wga_segmentation"]["wga_background_image"]:
                # Perform percentile normalization
                wga_mask_comp = percentile_normalization(
                    self.maps["median"][-1],
                    self.config["wga_segmentation"]["lower_quantile_normalization"],
                    self.config["wga_segmentation"]["upper_quantile_normalization"],
                )
            else:
                # Perform percentile normalization
                wga_mask_comp = percentile_normalization(
                    self.maps["median"][1],
                    self.config["wga_segmentation"]["lower_quantile_normalization"],
                    self.config["wga_segmentation"]["upper_quantile_normalization"],
                )
        else:
            # Perform percentile normalization
            wga_mask_comp = percentile_normalization(
                self.maps["median"][1],
                self.config["wga_segmentation"]["lower_quantile_normalization"],
                self.config["wga_segmentation"]["upper_quantile_normalization"],
            )

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if "threshold" in self.config["wga_segmentation"]:
            wga_mask = wga_mask_comp < self.config["wga_segmentation"]["threshold"]
        else:
            self.log(
                "No treshold for cytosol segmentation defined, global otsu will be used."
            )
            wga_mask = wga_mask_comp < global_otsu(wga_mask_comp)

        wga_mask = wga_mask.astype(float)
        wga_mask -= self.maps["nucleus_mask"]
        wga_mask = np.clip(wga_mask, 0, 1)

        # Apply dilation and erosion
        wga_mask = dilation(
            wga_mask, footprint=disk(self.config["wga_segmentation"]["erosion"])
        )
        self.maps["wga_mask"] = binary_erosion(
            wga_mask, footprint=disk(self.config["wga_segmentation"]["dilation"])
        )

        self.save_map("wga_mask")
        self.log("WGA mask map created")

    def _cellmembrane_potential_mask(self):
        self.log("Started with WGA potential map")

        wga_mask_comp = self.maps["median"][1] - np.quantile(
            self.maps["median"][1], 0.02
        )

        nn = np.quantile(self.maps["median"][1], 0.98)
        wga_mask_comp = wga_mask_comp / nn
        wga_mask_comp = np.clip(wga_mask_comp, 0, 1)

        # subtract golgi and dapi channel from wga
        diff = np.clip(wga_mask_comp - self.maps["median"][0], 0, 1)
        diff = np.clip(diff - self.maps["nucleus_mask"], 0, 1)
        diff = 1 - diff

        # enhance WGA map to generate speedmap
        # WGA 0.7-0.9
        min_clip = self.config["wga_segmentation"]["min_clip"]
        max_clip = self.config["wga_segmentation"]["max_clip"]
        diff = (np.clip(diff, min_clip, max_clip) - min_clip) / (max_clip - min_clip)

        diff = diff * 0.9 + 0.1
        diff = diff.astype(dtype=float)

        self.maps["wga_potential"] = diff

        self.save_map("wga_potential")
        self.log("WGA mask potential created")

    def _cellmembrane_fastmarching(self, center_nuclei):
        self.log("Started with fast marching")
        fmm_marker = np.ones_like(self.maps["median"][0])
        px_center = np.round(center_nuclei).astype(np.uint64)

        for center in px_center[1:]:
            fmm_marker[center[0], center[1]] = 0

        fmm_marker = np.ma.MaskedArray(fmm_marker, self.maps["wga_mask"])
        travel_time = skfmm_travel_time(fmm_marker, self.maps["wga_potential"])

        if not isinstance(travel_time, np.ma.core.MaskedArray):
            raise TypeError(
                "Travel_time for WGA based segmentation returned no MaskedArray. This is most likely due to missing WGA background determination."
            )

        self.maps["travel_time"] = travel_time.filled(fill_value=np.max(travel_time))

        self.save_map("travel_time")
        self.log("Fast marching finished")

    def _cellmembrane_watershed(self, center_nuclei):
        self.log("Started with watershed")

        marker = np.zeros_like(self.maps["median"][1])
        px_center = np.round(center_nuclei).astype(np.uint64)
        for i, center in enumerate(px_center[1:]):
            marker[center[0], center[1]] = i + 1
        wga_labels = watershed(
            self.maps["travel_time"],
            marker.astype(np.int64),
            mask=(self.maps["wga_mask"] == 0).astype(np.int64),
        )
        self.maps["watershed"] = np.where(self.maps["wga_mask"] > 0.5, 0, wga_labels)

        # ensure all edge labels are removed
        self.maps["watershed"] = remove_edge_labels(self.maps["watershed"])

        if self.debug:
            self._visualize_watershed_results(center_nuclei)

        self.save_map("watershed")
        self.log("watershed finished")

    def _filter_cells_cytosol_size(self, all_classes, filtered_classes):
        # filter cells based on cytosol size
        center_cell, length, coords = numba_mask_centroid(
            self.maps["watershed"], debug=self.debug
        )

        all_classes_wga = np.unique(self.maps["watershed"])

        labels_wga_filtered = size_filter(
            self.maps["watershed"],
            limits=[
                self.config["wga_segmentation"]["min_size"],
                self.config["wga_segmentation"]["max_size"],
            ],
        )

        classes_wga_filtered = np.unique(labels_wga_filtered)

        self.log(
            "Cells filtered out due to cytosol size limit: {} ".format(
                len(all_classes_wga) - len(classes_wga_filtered)
            )
        )

        filtered_classes_wga = set(classes_wga_filtered)
        filtered_classes = set(filtered_classes).intersection(filtered_classes_wga)
        self.log("Filtered out: {} ".format(len(all_classes) - len(filtered_classes)))
        self.log("Remaining: {} ".format(len(filtered_classes)))

        if self.debug:
            self._plot_cytosol_size_distribution(length)
            self._visualize_cytosol_filtering(classes_wga_filtered)

        return filtered_classes

    # functions to generate quality control plots
    def _dapi_median_intensity_plot(self):
        # generate plot of dapi median intensity
        plt.hist(self.maps["median"][0].flatten(), bins=100, log=False)
        plt.xlabel("intensity")
        plt.ylabel("frequency")
        plt.yscale("log")
        plt.title("DAPI intensity distribution")
        plt.savefig(os.path.join(self.directory, "dapi_intensity_dist.png"))
        plt.show()

    def _cellmembrane_median_intensity_plot(self):
        # generate plot of median Cellmembrane Marker intensity
        plt.hist(self.maps["median"][1].flatten(), bins=100, log=False)
        plt.xlabel("intensity")
        plt.ylabel("frequency")
        plt.yscale("log")
        plt.title("WGA intensity distribution")
        plt.savefig(os.path.join(self.directory, "wga_intensity_dist.png"))
        plt.show()

    def _visualize_nucleus_segmentation(
        self, classes_nuclei_unconnected, classes_nuclei_filtered
    ):
        visualize_class(
            classes_nuclei_unconnected,
            self.maps["nucleus_segmentation"],
            self.maps["normalized"][0],
        )
        visualize_class(
            classes_nuclei_filtered,
            self.maps["nucleus_segmentation"],
            self.maps["normalized"][0],
        )

    def _plot_nucleus_size_distribution(self, length):
        plt.hist(length, bins=50)
        plt.xlabel("px area")
        plt.ylabel("number")
        plt.title("Nucleus size distribution")
        plt.savefig(os.path.join(self.directory, "nucleus_size_dist.png"))
        plt.show()

    def _plot_cytosol_size_distribution(self, length):
        plt.hist(length, bins=50)
        plt.xlabel("px area")
        plt.ylabel("number")
        plt.title("Cytosol size distribution")
        plt.savefig(os.path.join(self.directory, "cytosol_size_dist.png"))
        plt.show()

    def _visualize_watershed_results(self, center_nuclei):
        image = label2rgb(
            self.maps["watershed"], self.maps["normalized"][0], bg_label=0, alpha=0.2
        )

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10, 10)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        plt.scatter(center_nuclei[:, 1], center_nuclei[:, 0], color="red")
        plt.savefig(os.path.join(self.directory, "watershed.png"))
        plt.show()

    def _visualize_cytosol_filtering(self, classes_wga_filtered):
        visualize_class(
            classes_wga_filtered, self.maps["watershed"], self.maps["normalized"][1]
        )
        visualize_class(
            classes_wga_filtered, self.maps["watershed"], self.maps["normalized"][0]
        )

    def _read_cellpose_model(self, modeltype, name, use_GPU, device):
        if modeltype == "pretrained":
            model = models.Cellpose(model_type=name, gpu=use_GPU, device=device)
        elif modeltype == "custom":
            model = models.CellposeModel(
                pretrained_model=name, gpu=use_GPU, device=device
            )
        return model

    def return_empty_mask(self, input_image):
        n_channels, x, y = input_image.shape
        self.save_segmentation(input_image, np.zeros((2, x, y)), [])


class WGASegmentation(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are the nucelus channel and a membrane marker channel like WGA
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if "wga_background_image" in self.config["wga_segmentation"]:
            if self.config["wga_segmentation"]["wga_background_image"]:
                # remove last channel since this is a pseudo channel to perform the WGA background calculation on
                feature_maps = [element for element in self.maps["normalized"][2:-1]]
            else:
                feature_maps = [element for element in self.maps["normalized"][2:]]
        else:
            feature_maps = [element for element in self.maps["normalized"][2:]]

        channels = np.stack(required_maps + feature_maps).astype(np.float64)

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["watershed"]]
        ).astype(self.DEFAULT_SEGMENTATION_DTYPE)

        return channels, segmentation

    def process(self, input_image):
        self.maps = {
            "normalized": None,
            "median": None,
            "nucleus_segmentation": None,
            "nucleus_mask": None,
            "wga_mask": None,
            "wga_potential": None,
            "travel_time": None,
            "watershed": None,
        }

        start_from = self.load_maps_from_disk()

        if self.identifier is not None:
            self.log(
                f"Segmentation started shard {self.identifier}, starting from checkpoint {start_from}"
            )

        else:
            self.log(f"Segmentation started, starting from checkpoint {start_from}")

        # Normalization
        if start_from <= 0:
            self._normalization(input_image)

        # Median calculation
        if start_from <= 1:
            self._median_calculation()

            if self.debug:
                self._dapi_median_intensity_plot()
                self._cellmembrane_median_intensity_plot()

        # segment dapi channels based on local tresholding
        if start_from <= 2:
            self.log("Started performing nucleus segmentation.")
            self._nucleus_segmentation()

        all_classes, filtered_classes, center_nuclei = self._filter_nuclei_classes()

        # create background map based on WGA
        if start_from <= 4:
            self._cellmembrane_mask_calculation()

        # create WGA potential map
        if start_from <= 5:
            self._cellmembrane_potential_mask()

        # WGA cytosol segmentation by fast marching

        if start_from <= 6:
            self._cellmembrane_fastmarching(center_nuclei)

        if start_from <= 7:
            self._cellmembrane_watershed(center_nuclei)

        filtered_classes = self._filter_cells_cytosol_size(
            all_classes, filtered_classes
        )
        channels, segmentation = self._finalize_segmentation_results()

        results = self.save_segmentation(channels, segmentation, filtered_classes)

        # self.save_segmentation_zarr(channels, segmentation) #currently save both since we have not fully converted.
        return results


class ShardedWGASegmentation(ShardedSegmentation):
    method = WGASegmentation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DAPISegmentation(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        feature_maps = [element for element in self.maps["normalized"][1:]]

        channels = np.stack(required_maps + feature_maps).astype(np.float64)

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["nucleus_segmentation"]]
        ).astype(self.DEFAULT_SEGMENTATION_DTYPE)
        return (channels, segmentation)

    def process(self, input_image):
        self.maps = {
            "normalized": None,
            "median": None,
            "nucleus_segmentation": None,
            "nucleus_mask": None,
            "travel_time": None,
        }

        start_from = self.load_maps_from_disk()

        if self.identifier is not None:
            self.log(
                f"Segmentation started shard {self.identifier}, starting from checkpoint {start_from}"
            )

        else:
            self.log(f"Segmentation started, starting from checkpoint {start_from}")

        # Normalization
        if start_from <= 0:
            self._normalization(input_image)

        # Median calculation
        if start_from <= 1:
            self._median_calculation()

            if self.debug:
                self._dapi_median_intensity_plot()

        # segment dapi channels based on local thresholding
        if start_from <= 2:
            self.log("Started performing nucleus segmentation.")
            self._nucleus_segmentation()

        _, filtered_classes, _ = self._filter_nuclei_classes()
        channels, segmentation = self._finalize_segmentation_results()

        self.save_segmentation(channels, segmentation, filtered_classes)


class ShardedDAPISegmentation(ShardedSegmentation):
    method = DAPISegmentation


##### CELLPOSE BASED SEGMENTATION METHODS #####
class DAPISegmentationCellpose(_cellpose_segmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=["nucleus"])

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0]]
        
        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 1:
            feature_maps = [element for element in self.maps["normalized"][1:]]

            channels = np.stack(required_maps + feature_maps).astype(np.float64)
        else:
            channels = np.stack(required_maps).astype(np.float64)

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["nucleus_segmentation"]]
        ).astype(self.DEFAULT_SEGMENTATION_DTYPE)
        return (channels, segmentation)

    def cellpose_segmentation(self, input_image):
        self._check_gpu_status()
        self._clear_cache()  # ensure we start with an empty cache

        ################################
        ### Perform Nucleus Segmentation
        ################################

        diameter, model = self._load_model(
            model_type="nuclei", gpu=self.use_GPU, device=self.device
        )

        masks = model.eval([input_image], diameter=diameter, channels=[1, 0])[0]
        masks = np.array(masks)  # convert to array

        # ensure all edge classes are removed
        masks = remove_edge_labels(masks)

        # check if filtering is required
        self._setup_filtering()

        if self.filter_size:
            masks = self._perform_size_filtering(
                mask=masks,
                thresholds=self.nucleus_thresholds,
                confidence_interval=self.nucleus_confidence_interval,
                mask_name="nucleus",
                log=True,
            )

        # save segementation to maps for access from other subfunctions
        self.maps["nucleus_segmentation"] = masks.reshape(masks.shape[1:])

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model, diameter, masks])

    def process(self, input_image):
        
        #ensure that we have the correct dtype loaded
        self._check_input_image_dtype(input_image)

        # initialize location to save masks to
        self.maps = {"normalized": None, 
                     "nucleus_segmentation": None}

        # could add a normalization step here if so desired
        self.maps["normalized"] = input_image.copy()

        #only get first channel for cellpose segmentation to preserver GPU memory
        input_image = input_image[:1, :, :]

        self.log("Starting Cellpose DAPI Segmentation.")

        self.cellpose_segmentation(input_image)

        all_classes = np.unique(self.maps["nucleus_segmentation"])
        channels, segmentation = self._finalize_segmentation_results()

        results = self.save_segmentation(channels, segmentation, all_classes)
        return results


class ShardedDAPISegmentationCellpose(ShardedSegmentation):
    method = DAPISegmentationCellpose


class CytosolSegmentationCellpose(_cellpose_segmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(self.DEFAULT_IMAGE_DTYPE)
        else:
            channels = np.stack(required_maps).astype(self.DEFAULT_IMAGE_DTYPE)

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["cytosol_segmentation"]]
        ).astype(self.DEFAULT_SEGMENTATION_DTYPE)

        return channels, segmentation

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=["nucleus", "cytosol"])
        self._check_for_mask_matching_filtering()

    def cellpose_segmentation(self, input_image):
        self._check_gpu_status()
        self._clear_cache()  # ensure we start with an empty cache

        ################################
        ### Perform Nucleus Segmentation
        ################################

        diameter, model = self._load_model(
            model_type="nucleus", gpu=self.use_GPU, device=self.device
        )

        masks_nucleus = model.eval([input_image], diameter=diameter, channels=[1, 0])[0]
        masks_nucleus = np.array(masks_nucleus)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model, diameter])

        #remove edge labels from masks
        masks_nucleus= remove_edge_labels(masks_nucleus)

        #################################
        #### Perform Cytosol Segmentation
        #################################

        diameter, model = self._load_model(
            model_type="cytosol", gpu=self.use_GPU, device=self.device
        )

        masks_cytosol = model.eval([input_image], diameter=diameter, channels=[2, 1])[0]
        masks_cytosol = np.array(masks_cytosol)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model, diameter])

        #remove edge labels from masks
        masks_cytosol = remove_edge_labels(masks_cytosol)

        ######################
        ### Perform Filtering to remove too small/too large masks if applicable
        ######################

        # check if filtering is required
        self._setup_filtering()

        if self.filter_size:
            masks_nucleus = self._perform_size_filtering(
                mask=masks_nucleus,
                thresholds=self.nucleus_thresholds,
                confidence_interval=self.nucleus_confidence_interval,
                mask_name="nucleus",
                log=True,
                debug=self.debug,
            )

            masks_cytosol = self._perform_size_filtering(
                mask=masks_cytosol,
                thresholds=self.nucleus_thresholds,
                confidence_interval=self.nucleus_confidence_interval,
                mask_name="cytosol",
                log=True,
                debug=self.debug,
            )

        ######################
        ### Perform Filtering match cytosol and nucleus IDs if applicable
        ######################
        self._setup_filtering()
        
        if self.filter_match_masks:
            masks_nucleus, masks_cytosol = self._perform_mask_matching_filtering(
                nucleus_mask=masks_nucleus,
                cytosol_mask=masks_cytosol,
                filtering_threshold=self.mask_matching_filtering_threshold,
                debug=self.debug,
            )
        else:
            self.log("No filtering performed. Cytosol and Nucleus IDs in the two masks do not match. Before proceeding with extraction an additional filtering step needs to be performed")

        ######################
        ### Cleanup Generated Segmentation masks
        ######################

        # first when the masks are finalized save them to the maps
        self.maps["nucleus_segmentation"] = masks_nucleus.reshape(
            masks_nucleus.shape[1:]
        )

        self.maps["cytosol_segmentation"] = masks_cytosol.reshape(
            masks_cytosol.shape[1:]
        )

        self._clear_cache(vars_to_delete=[masks_nucleus, masks_cytosol])

    def process(self, input_image):
        #check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # initialize location to save masks to
        self.maps = {
            "normalized": tempmmap.array(
                shape=input_image.shape,
                dtype=float,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "nucleus_segmentation": tempmmap.array(
                shape=input_image.shape,
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "cytosol_segmentation": tempmmap.array(
                shape=input_image.shape,
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
        }

        # could add a normalization step here if so desired
        self.maps["normalized"] = input_image.copy()

        #only get the first two channels for cellpose segmentation to preserve GPU memory
        input_image = input_image[:2, :, :]
        gc.collect()

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(input_image)

        #free up memory
        del input_image 
        gc.collect()

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = set(np.unique(self.maps["nucleus_segmentation"])) - set([0]) # remove background as a class
        all_classes = list(all_classes)

        channels, segmentation = self._finalize_segmentation_results()
        results = self.save_segmentation(channels, segmentation, all_classes)

        # clean up memory
        del channels, segmentation, all_classes
        gc.collect()

        return results


class ShardedCytosolSegmentationCellpose(ShardedSegmentation):
    method = CytosolSegmentationCellpose


class CytosolSegmentationDownsamplingCellpose(CytosolSegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):

        # nuclear and cytosolic channels are required (used for segmentation)
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain additional phenotypes e.g. for classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(self.DEFAULT_IMAGE_DTYPE)
        else:
            channels = np.stack(required_maps).astype(self.DEFAULT_IMAGE_DTYPE)

        self.maps["fullsize_nucleus_segmentation"] = self._rescale_downsampled_mask(
            self.maps["nucleus_segmentation"], "nucleus_segmentation"
        )
        self.maps["fullsize_cytosol_segmentation"] = self._rescale_downsampled_mask(
            self.maps["cytosol_segmentation"], "cytosol_segmentation"
        )

        segmentation = np.stack([self.maps["fullsize_nucleus_segmentation"], self.maps["fullsize_cytosol_segmentation"]])

        return(channels, segmentation)

    def process(self, input_image):

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        input_image = self._downsample_image(input_image)

        # setup the memory mapped arrays to store the results
        self.maps = {
            "normalized": tempmmap.array(
                shape=input_image.shape,
                dtype=float,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "nucleus_segmentation": tempmmap.array(
                shape=(1, input_image.shape[1], input_image.shape[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "cytosol_segmentation": tempmmap.array(
                shape=(1, input_image.shape[1], input_image.shape[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "fullsize_nucleus_segmentation": tempmmap.array(
                shape=(1, self.original_image_size[1], self.original_image_size[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "fullsize_cytosol_segmentation": tempmmap.array(
                shape=(1, self.original_image_size[1], self.original_image_size[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
        }

        # could add a normalization step here if so desired
        # perform downsampling after saving input image to ensure that we have a duplicate preserving the original dimensions
        self.maps["normalized"] = input_image.copy()

        # only get the first 2 channels for segmentation (does not use excess space on the GPU this way)
        input_image = input_image[:2, :, :]  
        gc.collect()  # cleanup to ensure memory is freed up

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(input_image)

        del input_image
        gc.collect()

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = np.unique(self.maps["nucleus_segmentation"])

        channels, segmentation = self._finalize_segmentation_results()

        results = self.save_segmentation(channels, segmentation, all_classes)

        return results


class ShardedCytosolSegmentationDownsamplingCellpose(ShardedSegmentation):
    method = CytosolSegmentationDownsamplingCellpose


class CytosolOnlySegmentationCellpose(_cellpose_segmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=["cytosol"])

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(np.float64)
        else:
            channels = np.stack(required_maps).astype(np.float64)

        segmentation = np.stack(
            [self.maps["cytosol_segmentation"], self.maps["cytosol_segmentation"]]
        ).astype(self.DEFAULT_SEGMENTATION_DTYPE)
        return (channels, segmentation)

    def cellpose_segmentation(self, input_image):
        self._check_gpu_status()
        self._clear_cache()
        
        #####
        ### Perform Cytosol Segmentation
        #####

        diameter, model = self._load_model(
            model_type="cytosol", gpu=self.use_GPU, device=self.device
        )

        masks_cytosol = model.eval([input_image], diameter=diameter, channels=[2, 1])[0]
        masks_cytosol = np.array(masks_cytosol)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model, diameter])

        # ensure edge classes are removed
        masks_cytosol = remove_edge_labels(masks_cytosol)

        #####
        ### Perform Filtering to remove too small/too large masks if applicable
        #####

        self._setup_filtering()

        if self.filter_size:
            masks_cytosol = self._perform_size_filtering(
                mask=masks_cytosol,
                thresholds=self.nucleus_thresholds,
                confidence_interval=self.nucleus_confidence_interval,
                mask_name="cytosol",
                log=True,
                debug=self.debug,
            )

        self.maps["cytosol_segmentation"] = masks_cytosol.reshape(
            masks_cytosol.shape[1:]
        )  # add reshape to match shape to HDF5 shape

        # clear memory
        self._clear_cache(vars_to_delete=[masks_cytosol])

    def process(self, input_image) -> None:

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # initialize location to save masks to
        self.maps = {
            "normalized": tempmmap.array(
                shape=input_image.shape,
                dtype=float,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "cytosol_segmentation": tempmmap.array(
                shape=(1, input_image.shape[1], input_image.shape[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
        }

        # could add a normalization step here if so desired
        self.maps["normalized"] = input_image.copy()

        # only get the first two channels for segmentation (does not use excess space on the GPU this way)
        input_image = input_image[:2, :, :]

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(input_image)

        self._clear_cache(vars_to_delete=[input_image])

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = np.unique(self.maps["cytosol_segmentation"])

        channels, segmentation = self._finalize_segmentation_results()
        results = self.save_segmentation(channels, segmentation, all_classes)

        # clean up memory
        self._clear_cache(vars_to_delete=[channels, segmentation, all_classes])
        return results


class Sharded_CytosolOnly_Cellpose_Segmentation(ShardedSegmentation):
    method = CytosolOnlySegmentationCellpose


class CytosolOnly_Segmentation_Downsampling_Cellpose(CytosolOnlySegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, size_padding):
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(self.DEFAULT_IMAGE_DTYPE)
        else:
            channels = np.stack(required_maps).astype(self.DEFAULT_IMAGE_DTYPE)

        _seg_size = self.maps["cytosol_segmentation"].shape

        self.log(
            f"Segmentation size after downsampling before resize to original dimensions: {_seg_size}"
        )

        _, x, y = size_padding

        cyto_seg = self.maps["cytosol_segmentation"]
        cyto_seg = cyto_seg.repeat(self.config["downsampling_factor"], axis=0).repeat(
            self.config["downsampling_factor"], axis=1
        )

        # perform erosion and dilation for smoothing
        cyto_seg = erosion(
            cyto_seg, footprint=disk(self.config["smoothing_kernel_size"])
        )
        cyto_seg = dilation(
            cyto_seg, footprint=disk(self.config["smoothing_kernel_size"])
        )

        # combine masks into one stack
        segmentation = np.stack([cyto_seg, cyto_seg]).astype(
            self.DEFAULT_SEGMENTATION_DTYPE
        )
        del cyto_seg

        # rescale segmentation results to original size
        x_trim = x - self.project.input_image.shape[1]
        y_trim = y - self.project.input_image.shape[2]

        # if no padding was performed then we need to keep the same dimensions
        if x_trim > 0:
            if y_trim > 0:
                segmentation = segmentation[:, :-x_trim, :-y_trim]
            else:
                segmentation = segmentation[:, :-x_trim, :]
        else:
            if y_trim > 0:
                segmentation = segmentation[:, :, :-y_trim]
            else:
                segmentation = segmentation

        self.log(
            f"Segmentation size after resize to original dimensions: {segmentation.shape}"
        )

        if segmentation.shape[1] != self.project.input_image.shape[1]:
            sys.exit("Error. Segmentation mask and image have different shapes")
        if segmentation.shape[2] != self.project.input_image.shape[2]:
            sys.exit("Error. Segmentation mask and image have different shapes")

        return channels, segmentation

    def process(self, input_image) -> None:

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        # setup the memory mapped arrays to store the results
        self.maps = {
            "normalized": tempmmap.array(
                shape=input_image.shape,
                dtype=float,
            ),
            "cytosol_segmentation": tempmmap.array(
                shape=(1, input_image.shape[1], input_image.shape[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
            "fullsize_cytosol_segmentation": tempmmap.array(
                shape=(1, self.original_image_size[1], self.original_image_size[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
        }

        self.maps["normalized"] = input_image.copy()

        #perform image downsampling
        input_image = self._downsample_image(input_image)

        self.cellpose_segmentation(input_image)
        self._clear_cache(vars_to_delete=[input_image])

        #finalize segmentation results
        all_classes = np.unique(self.maps["cytosol_segmentation"])

        channels, segmentation = self._finalize_segmentation_results()

        results = self.save_segmentation(channels, segmentation, all_classes)

        self._clear_cache(vars_to_delete=[channels, segmentation, all_classes])

        return results

class Sharded_CytosolOnly_Segmentation_Downsampling_Cellpose(ShardedSegmentation):
    method = CytosolOnly_Segmentation_Downsampling_Cellpose


#### TIMECOURSE SEGMENTATION METHODS #####
#### THIS SHOULD BE SWITCHED TO THE BATCHED CLASS IMPLEMENTED BY TIM ####


class WGA_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file.
    """

    class WGASegmentation_Timecourse(WGASegmentation, TimecourseSegmentation):
        method = WGASegmentation

    method = WGASegmentation_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Multithreaded_WGA_TimecourseSegmentation(MultithreadedSegmentation):
    class WGASegmentation_Timecourse(WGASegmentation, TimecourseSegmentation):
        method = WGASegmentation

    method = WGASegmentation_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Cytosol_Cellpose_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
    """

    class CytosolSegmentationCellpose_Timecourse(
        CytosolSegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationCellpose

    method = CytosolSegmentationCellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Cytosol_Cellpose_Downsampling_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
    """

    class Cytosol_Segmentation_Downsampling_Cellpose_Timecourse(
        CytosolSegmentationDownsamplingCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationDownsamplingCellpose

    method = Cytosol_Segmentation_Downsampling_Cellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CytosolOnly_Cellpose_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
    """

    class CytosolOnly_Cellpose_TimecourseSegmentation(
        CytosolOnlySegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolOnlySegmentationCellpose

    method = CytosolOnly_Cellpose_TimecourseSegmentation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Multithreaded_Cytosol_Cellpose_TimecourseSegmentation(MultithreadedSegmentation):
    class CytosolSegmentationCellpose_Timecourse(
        CytosolSegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationCellpose

    method = CytosolSegmentationCellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Multithreaded_Cytosol_Cellpose_Downsampling_TimecourseSegmentation(
    MultithreadedSegmentation
):
    class Cytosol_Segmentation_Downsampling_Cellpose_Timecourse(
        CytosolSegmentationDownsamplingCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationDownsamplingCellpose

    method = Cytosol_Segmentation_Downsampling_Cellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Multithreaded_CytosolOnly_Cellpose_TimecourseSegmentation(
    MultithreadedSegmentation
):
    class CytosolOnly_SegmentationCellpose_Timecourse(
        CytosolOnlySegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolOnlySegmentationCellpose

    method = CytosolOnly_SegmentationCellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
