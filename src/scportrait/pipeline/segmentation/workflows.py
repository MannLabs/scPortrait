import os
import sys
import time
import timeit
from typing import Tuple, Union, List
import multiprocessing

import xarray
import numpy as np
import matplotlib.pyplot as plt
from alphabase.io import tempmmap
from skfmm import travel_time as skfmm_travel_time
from skimage.filters import median
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.morphology import binary_erosion, disk, dilation, erosion
import torch
from cellpose import models

from scportrait.plotting._utils import _custom_cmap
from scportrait.processing.images._image_processing import percentile_normalization, downsample_img
from scportrait.processing.masks.mask_filtering import SizeFilter, MatchNucleusCytosolIds
from scportrait.pipeline._utils.segmentation import (
    segment_local_threshold,
    segment_global_threshold,
    numba_mask_centroid,
    contact_filter,
    global_otsu,
    remove_edge_labels,
)
from scportrait.pipeline.segmentation.segmentation import (
    Segmentation,
    ShardedSegmentation,
)

class _BaseSegmentation(Segmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform_input_image(self, input_image):
        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data
        return input_image

    def return_empty_mask(self, input_image):
        n_channels, x, y = input_image.shape
        self._save_segmentation_sdata(np.zeros((2, x, y)), [])

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

    def _calculate_padded_image_size(self, img: np.array) -> None:
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
        padded_image_size = (
            2,
            self.input_image_size[1] + pad_x[1],
            self.input_image_size[2] + pad_y[1],
        )

        self.expected_padded_image_size = padded_image_size
        self.pad_x = pad_x
        self.pad_y = pad_y

        # track original image size
        self.original_image_size = img.shape

        return None

    def _downsample_image(self, img: np.array, debug: bool = False) -> np.array:
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
                f"Error. Image padding did not work as expected. Expected a padded image of size {self.expected_padded_image_size} but got {self.padded_image_size}."
            )

        self.log(f"Downsampling image by a factor of {self.N}x{self.N}")

        # actually perform downsampling
        img = downsample_img(img, N=self.N)

        self.downsampled_image_size = img.shape

        if debug:
            self.log(f"Downsampled image size {self.downsampled_image_size}")

        return img

    def _rescale_downsampled_mask(self, mask: np.array, mask_name: str) -> np.array:
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

        if len(mask.shape) == 3:
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
        elif len(mask.shape) == 2:
            if x_trim > 0:
                if y_trim > 0:
                    mask = mask[:-x_trim, :-y_trim]
                else:
                    mask = mask[:-x_trim, :]
            else:
                if y_trim > 0:
                    mask = mask[:, :-y_trim]
                else:
                    mask = mask

        # check that mask has the correct shape and matches to input image

        if len(mask.shape) == 2:
            assert (mask.shape[0] == self.original_image_size[1]) and (
                mask.shape[1] == self.original_image_size[2])
        elif len(mask.shape) == 3:
            assert (mask.shape[1] == self.original_image_size[1]) and (
                mask.shape[2] == self.original_image_size[2])

        return mask

    #### Image Processing #####

    def _normalize_image(
        self,
        input_image: np.array,
        lower: Union[float, dict],
        upper: Union[float, dict],
        debug: bool = False,
    ) -> np.array:
        # check that both are specified as the same type
        assert isinstance(lower, float) == isinstance(upper, float)
        assert isinstance(lower, dict) == isinstance(upper, dict)

        if isinstance(lower, float):
            self.log("Normalizing each channel to the same range")
            norm_image = percentile_normalization(input_image, lower, upper)

        elif isinstance(lower, dict):
            norm_image = []

            for i in range(input_image.shape[0]):
                _lower = lower[i]
                _upper = upper[i]

                norm_image.append(
                    percentile_normalization(input_image[i], _lower, _upper)
                )

            norm_image = np.array(norm_image)

        if debug:
            if len(norm_image.shape) == 2:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(norm_image)
                ax.axis("off")
                ax.set_title("Normalized Image")
            else:
                # visualize output if debug is turned on
                n_channels = norm_image.shape[0]
                fig, axs = plt.subplots(1, n_channels, figsize=(10 * n_channels, 10))
                for i in range(n_channels):
                    axs[i].imshow(norm_image[i])
                    axs[i].axis("off")
                    axs[i].set_title(f"Normalized Channel {i}")

            fig.tight_layout()
            fig_path = os.path.join(self.directory, "normalized_input.png")
            fig.savefig(fig_path)

        return norm_image

    def _median_correct_image(
        self, input_image, median_filter_size: int, debug: bool = False
    ):
        self.log("Performing median filtering on input image")

        for i in range(input_image.shape[0]):
            input_image[i] = median(input_image[i], disk(median_filter_size))

        if debug:
            if len(input_image.shape) == 2:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(input_image)
                ax.axis("off")
                ax.set_title("Median Corrected Image")
            else:
                # visualize output if debug is turned on
                n_channels = input_image.shape[0]

                fig, axs = plt.subplots(1, n_channels, figsize=(10 * n_channels, 10))
                for i in range(n_channels):
                    axs[i].imshow(input_image[i])
                    axs[i].axis("off")
                    axs[i].set_title(f"Median Corrected Channel {i}")

            fig.tight_layout()
            fig_path = os.path.join(self.directory, "median_corrected_input.png")
            fig.savefig(fig_path)

        return input_image

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

                #save attributes for use later
                setattr(self, f"{mask_type}_thresholds", thresholds)
                setattr(self, f"{mask_type}_confidence_interval", confidence_interval)

    def _get_params_cellsize_filtering(
        self, type
        ) -> Tuple[Union[Tuple[float], None], Union[float, None]]:
        
        absolute_filter_status = False

        if "min_size" in self.config[f"{type}_segmentation"].keys():
            min_size = self.config[f"{type}_segmentation"]["min_size"]
            absolute_filter_status = True
        else:
            min_size = None

        if "max_size" in self.config[f"{type}_segmentation"].keys():
            max_size = self.config[f"{type}_segmentation"]["max_size"]
            absolute_filter_status = True
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
        thresholds: Union[Tuple[float], None],
        confidence_interval: float,
        mask_name: str,
        log: bool = True,
        debug: bool = False,
        input_image: np.array = None,
        ) -> np.array:
        """
        Remove elements from mask based on a size filter.

        Parameters
        ----------
        mask
            mask to be filtered
        """
        start_time = time.time()

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
            
            #get visualization of the filtering results (2 = filtered, 1 = keep, 0 = background)
            mask = filter.visualize_filtering_results(
                plot_fig=False, return_maps=True
            )

            if not self.is_shard:
                if self.save_filter_results:

                    if len(mask.shape) == 2:
                        #add results to sdata object
                        self.filehandler._write_segmentation_sdata(mask, segmentation_label = f"debugging_seg_size_filter_results_{mask_name}", classes = filter.ids)
                    else:
                        self.filehandler._write_segmentation_sdata(mask[0], segmentation_label = f"debugging_seg_size_filter_results_{mask_name}", classes = filter.ids)
                    #then this does not need to be plotted as it can be visualized from there
                    plot_results = False
                else:
                    plot_results = True
            else:
                plot_results = True
            
            if plot_results:
                # get input image for visualization
                if input_image is None:
                    if "input_image" in self.__dict__.keys():
                        input_image = self.input_image
                
                if input_image is not None:
                    if len(input_image.shape) == 2:
                        image_map = input_image
                    else:
                        if mask_name == "nucleus":
                            image_map = input_image[0]
                        elif mask_name == "cytosol":
                            image_map = input_image[1]
                    
                    cmap, norm = _custom_cmap()
                    
                    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
                    axs.imshow(image_map, cmap="gray")
                    axs.imshow(mask, cmap=cmap, norm=norm)
                    axs.axis("off")
                    axs.set_title(f"Visualization of classes removed during {mask_name} size filtering")
                    fig_path = os.path.join(self.directory, f"Results_{mask_name}_size_filtering.png")
                    fig.savefig(fig_path)

                    self._clear_cache(vars_to_delete=[fig, input_image, cmap, norm])

            # clearup memory
            self._clear_cache(
                vars_to_delete=[mask, plot_results]
            )

        end_time = time.time()

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
        input_image: np.array = None,
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

            mask_nuc, mask_cyto = filter.visualize_filtering_results(plot_fig = False, return_maps = True)
            
            #check if image should be added to sdata or plotted and saved
            if not self.is_shard:
                if self.save_filter_results:
                    #add filtering results to sdata object
                    self.filehandler._write_segmentation_sdata(mask_nuc[0], segmentation_label = "debugging_seg_match_mask_results_nucleus", classes = None)
                    self.filehandler._write_segmentation_sdata(mask_cyto[0], segmentation_label = "debugging_seg_match_mask_result_cytosol", classes = None)

                    #then no plotting needs to be performed as the results can be viewed in the sdata object
                    plot_results = False
                else:
                    plot_results = True
            else:
                plot_results = True

            if plot_results: 
                if input_image is not None:
                    if "input_image" in self.__dict__.keys():
                        input_image = self.input_image

                if input_image is not None:
                    #convert input image from uint16 to uint8
                    input_image = (input_image / 256).astype(np.uint8)

                    cmap, norm = _custom_cmap()

                    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

                    axs[0].imshow(input_image[0], cmap="gray")
                    axs[0].imshow(mask_nuc[0], cmap=cmap, norm = norm)
                    axs[0].imshow(mask_cyto[0], cmap=cmap, norm = norm)
                    axs[0].axis("off")
                    axs[0].set_title("results overlayed nucleus channel")

                    axs[1].imshow(input_image[1], cmap="gray")
                    axs[1].imshow(mask_nuc[0], cmap=cmap, norm = norm)
                    axs[1].imshow(mask_cyto[0], cmap=cmap, norm = norm)
                    axs[1].axis("off")
                    axs[1].set_title("results overlayed cytosol channel")
                    
                    fig.tight_layout()
                    fig_path = os.path.join(self.directory, "Results_mask_matching.png")
                    fig.savefig(fig_path)

                    self._clear_cache(vars_to_delete=[fig, input_image, cmap, norm])

            
            # clearup memory
            self._clear_cache(
                vars_to_delete=[mask_nuc, mask_cyto, plot_results]
            )

        self.log(
            "Total time to perform nucleus and cytosol mask matching filtering: {:.2f} seconds".format(
                time.time() - start_time
            )
        )

        return masks_nucleus, masks_cytosol


###### CLASSICAL SEGMENTATION METHODS #####


class _ClassicalSegmentation(_BaseSegmentation):
    def __init__(self, *args, **kwargs):
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
            plt.scatter(
                self.nucleus_centers[:, 1], self.nucleus_centers[:, 0], s=2, color="red"
            )
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
            plt.scatter(
                self.nucleus_centers[:, 1], self.nucleus_centers[:, 0], s=2, color="red"
            )
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
                self.lower_quantile_normalization_input_image = self.config[
                    "lower_quantile_normalization"
                ]

            elif isinstance(self.config["lower_quantile_normalization"], list):
                if (
                    len(self.config["lower_quantile_normalization"])
                    != self.input_image.shape[0]
                ):
                    raise ValueError(
                        "Number of specified normalization ranges for lower quantile normalization does not match the number of input channels."
                    )
                self.lower_quantile_normalization_input_image = dict(
                    zip(
                        range(self.input_image.shape[0]),
                        self.config["lower_quantile_normalization"],
                    )
                )
            else:
                param = self.config["lower_quantile_normalization"]
                raise ValueError(
                    f"lower_quantile_normalization must be either a float or a list of floats. Instead recieved {param}"
                )

            # get upper quantile normalization range
            if isinstance(self.config["upper_quantile_normalization"], float):
                self.upper_quantile_normalization_input_image = self.config[
                    "upper_quantile_normalization"
                ]

            elif isinstance(self.config["upper_quantile_normalization"], list):
                if (
                    len(self.config["upper_quantile_normalization"])
                    != self.input_image.shape[0]
                ):
                    raise ValueError(
                        "Number of specified normalization ranges for upper quantile normalization does not match the number of input channels."
                    )
                self.upper_quantile_normalization_input_image = dict(
                    zip(
                        range(self.input_image.shape[0]),
                        self.config["upper_quantile_normalization"],
                    )
                )
            else:
                param = self.config["upper_quantile_normalization"]
                raise ValueError(
                    f"upper_quantile_normalization must be either a float or a list of floats. Instead recieved {param}"
                )

            # check that the normalization ranges are of the same type otherwise this will result in issues
            assert type(self.lower_quantile_normalization_input_image) == type(
                self.upper_quantile_normalization_input_image
            )

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
                self.normalization_nucleus_lower_quantile = self.config[
                    "nucleus_segmentation"
                ]["lower_quantile_normalization"]
            if "upper_quantile_normalization" in self.config["nucleus_segmentation"]:
                self.normalization_nucleus_segmentation = True
                self.normalization_nucleus_upper_quantile = self.config[
                    "nucleus_segmentation"
                ]["upper_quantile_normalization"]

            # check if nuclei should be filtered based on size
            self._check_for_size_filtering(mask_types=["nucleus"])

            # check if nuclei should be filtered based on contact
            if "contact_filter" in self.config["nucleus_segmentation"]:
                self.contact_filter_nuclei = True
                self.contact_filter_nuclei_threshold = self.config[
                    "nucleus_segmentation"
                ]["contact_filter"]
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
                self.normalization_cytosol_lower_quantile = self.config[
                    "cytosol_segmentation"
                ]["lower_quantile_normalization"]
            if (
                "upper_quantile_normalization"
                in self.config_cytosol_segmentation.keys()
            ):
                self.normalization_cytosol_segmentation = True
                self.normalization_cytosol_upper_quantile = self.config[
                    "cytosol_segmentation"
                ]["upper_quantile_normalization"]

            # check if cytosol should be filtered based on size
            self._check_for_size_filtering(mask_types=["cytosol"])

            # check if cytosol should be filtered based on contact
            if "contact_filter" in self.config["cytosol_segmentation"]:
                self.contact_filter_cytosol = True
                self.contact_filter_cytosol_threshold = self.config[
                    "cytosol_segmentation"
                ]["contact_filter"]
            else:
                self.contact_filter_cytosol = False

            # check if cytosol should be filtered based on matching to nuclei
            self._check_for_mask_matching_filtering()

    def _nucleus_segmentation(self, input_image, debug: bool = False):
        if self.normalization_nucleus_segmentation:
            lower = self.normalization_nucleus_lower_quantile
            upper = self.normalization_nucleus_upper_quantile

            self.log(
                f"Percentile normalization of nucleus input image to range {lower}, {upper}"
            )
            input_image = self._normalize_image(input_image, lower, upper, debug=debug)

        # perform thresholding to generate  a mask of the nuclei

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if (
            "threshold" in self.config["nucleus_segmentation"]
            and "median_block" in self.config["nucleus_segmentation"]
        ):
            threshold = self.config["nucleus_segmentation"]["threshold"]
            self.log(
                f"Using local thresholding with a threshold of {threshold} to calculate nucleus mask."
            )
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
            self.log(
                "Using global otsu to calculate threshold for nucleus mask generation."
            )
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
        self.maps["nucleus_segmentation"] = remove_edge_labels(
            self.maps["nucleus_segmentation"]
        )

        if self.filter_size:
            self.maps["nucleus_segmentation"] = self._perform_size_filtering(
                self.maps["nucleus_segmentation"],
                self.nucleus_thresholds,
                self.nucleus_confidence_interval,
                "nucleus",
                debug=self.debug,
                input_image = input_image if self.debug else None,
            )

        if self.contact_filter_nuclei:
            if self.debug:
                n_classes = len(
                    set(np.unique(self.maps["nucleus_segmentation"])) - set([0])
                )

            self.maps["nucleus_segmentation"] = contact_filter(
                self.maps["nucleus_segmentation"],
                threshold=self.contact_filter_nuclei_threshold,
                reindex=False,
            )

            if self.debug:
                n_classes_post = len(
                    set(np.unique(self.maps["nucleus_segmentation"])) - set([0])
                )
                self.log(
                    f"Filtered out {n_classes - n_classes_post} nuclei due to contact filtering."
                )

    def _cytosol_segmentation(self, input_image, debug: bool = False):
        if not self.segment_nuclei:
            raise ValueError(
                "Nucleus segmentation must be performed to be able to perform a cytosol segmentation."
            )

        if self.normalization_cytosol_segmentation:
            lower = self.normalization_cytosol_lower_quantile
            upper = self.normalization_cytosol_upper_quantile

            input_image = self._normalize_image(input_image, lower, upper, debug=debug)

        # perform thresholding to generate a mask of the cytosol

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if "threshold" in self.config["cytosol_segmentation"]:
            cytosol_mask = (
                input_image < self.config["cytosol_segmentation"]["threshold"]
            )
        else:
            self.log(
                "No treshold for cytosol segmentation defined, global otsu will be used."
            )
            cytosol_mask = input_image < global_otsu(input_image)

        self._clear_cache(vars_to_delete=[input_image])

        # remove the nucleus mask from the cytosol mask
        cytosol_mask = cytosol_mask.astype(float)
        cytosol_mask -= self.maps["nucleus_mask"]
        cytosol_mask = np.clip(cytosol_mask, 0, 1)

        # Apply dilation and erosion
        cytosol_mask = dilation(
            cytosol_mask, footprint=disk(self.config["cytosol_segmentation"]["erosion"])
        )
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
            self.maps["median_corrected"][1]
            - np.quantile(self.maps["median_corrected"][1], 0.02)
        ) / np.quantile(self.maps["median_corrected"][1], 0.98)
        potential_map = np.clip(potential_map, 0, 1)

        # subtract nucleus mask from potential map
        potential_map = np.clip(potential_map - self.maps["nucleus_mask"], 0, 1)
        potential_map = 1 - potential_map

        # enhance potential map to generate speedmap
        min_clip = self.config["cytosol_segmentation"]["min_clip"]
        max_clip = self.config["cytosol_segmentation"]["max_clip"]
        potential_map = (np.clip(potential_map, min_clip, max_clip) - min_clip) / (
            max_clip - min_clip
        )
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
            marker[center[0], center[1]] = self.maps["nucleus_segmentation"][
                center[0], center[1]
            ]

        cytosol_labels = watershed(
            self.maps["travel_time"],
            marker.astype(np.int64),
            mask=(self.maps["cytosol_mask"] == 0).astype(np.int64),
        )

        cytosol_segmentation = np.where(
            self.maps["cytosol_mask"] > 0.5, 0, cytosol_labels
        )

        # ensure all edge labels are removed
        cytosol_segmentation = remove_edge_labels(cytosol_segmentation)
        self.maps["cytosol_segmentation"] = cytosol_segmentation

        if self.filter_size:
            self.maps["cytosol_segmentation"] = self._perform_size_filtering(
                self.maps["cytosol_segmentation"],
                self.cytosol_thresholds,
                self.cytosol_confidence_interval,
                "cytosol",
                debug=self.debug,
                input_image = input_image if self.debug else None,
            )

        if self.contact_filter_cytosol:
            if self.debug:
                n_classes = len(
                    set(np.unique(self.maps["cytosol_segmentation"])) - set([0])
                )

            self.maps["cytosol_segmentation"] = contact_filter(
                self.maps["cytosol_segmentation"],
                threshold=self.contact_filter_cytosol_threshold,
                reindex=False,
            )

            if self.debug:
                n_classes_post = len(
                    set(np.unique(self.maps["cytosol_segmentation"])) - set([0])
                )
                self.log(
                    f"Filtered out {n_classes - n_classes_post} cytosols due to contact filtering."
                )

        unique_cytosol_ids = set(np.unique(self.maps["cytosol_segmentation"])) - set(
            [0]
        )

        # remove any ids from nucleus mask that dont have a cytosol mask
        self.maps["nucleus_segmentation"][
            ~np.isin(self.maps["nucleus_segmentation"], list(unique_cytosol_ids))
        ] = 0


class WGASegmentation(_ClassicalSegmentation):
    N_MASKS = 2
    MASK_NAMES = ["nuclei", "cytosol"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["cytosol_segmentation"]]
        ).astype(self.DEFAULT_SEGMENTATION_DTYPE)

        return segmentation

    def process(self, input_image):
        self._get_processing_parameters()

        # intialize maps for storing intermediate results
        self.maps = {}

        # save input image
        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data.compute()

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

        if self.segment_nuclei:
            image = input_image[0]
            self._nucleus_segmentation(image, debug=self.debug)

        if self.segment_cytosol:
            image = self.maps["input_image"][1]
            if isinstance(image, xarray.DataArray):
                image = image.data.compute()
            self._cytosol_segmentation(image, debug=self.debug)

        if self.debug:
            self._visualize_final_masks()

        all_classes = list(set(np.unique(self.maps["nucleus_segmentation"])) - set([0]))
        segmentation = self._finalize_segmentation_results()

        print("Channels shape: ", segmentation.shape)

        results = self._save_segmentation_sdata(segmentation, all_classes, masks = self.MASK_NAMES)
        return results


class ShardedWGASegmentation(ShardedSegmentation):
    method = WGASegmentation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DAPISegmentation(_ClassicalSegmentation):

    N_MASKS = 1
    MASK_NAMES = ["nuclei"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"]]
        ).astype(self.DEFAULT_SEGMENTATION_DTYPE)

        return segmentation

    def process(self, input_image):
        self._get_processing_parameters()

        # intialize maps for storing intermediate results
        self.maps = {}

        # save input image
        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data.compute()
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

        if self.segment_nuclei:
            self._nucleus_segmentation(input_image[0], debug=self.debug)

        all_classes = list(set(np.unique(self.maps["nucleus_segmentation"])) - set([0]))
        segmentation = self._finalize_segmentation_results()

        results = self._save_segmentation_sdata(segmentation, all_classes, masks = self.MASK_NAMES)
        return results


class ShardedDAPISegmentation(ShardedSegmentation):
    method = DAPISegmentation


##### CELLPOSE BASED SEGMENTATION METHODS #####


class _CellposeSegmentation(_BaseSegmentation):
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
            if self.status == "multi_GPU":
                self.use_GPU = f"cuda:{self.gpu_id}"
                self.device = torch.device(self.use_GPU)
                self.nGPUs = torch.cuda.device_count()
            else:
                self.use_GPU = True
                self.device = torch.device(
                    "cuda"
                )  # dont need to specify id, saying cuda will default to the one thats avaialable
                self.nGPUs = 1

        # check if MPS is available
        elif torch.backends.mps.is_available():
            self.use_GPU = True
            self.device = torch.device("mps")
            self.nGPUs = 1

        # default to CPU
        else:
            self.use_GPU = False
            self.device = torch.device("cpu")
            self.nGPUs = 0

        self.log(
            f"GPU Status for segmentation is {self.use_GPU} and will segment using the following device {self.device}."
        )

class DAPISegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 1
    MASK_NAMES = ["nuclei"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=["nucleus"])

    def _finalize_segmentation_results(self):
        # ensure correct dtype of the maps

        self.maps["nucleus_segmentation"] = self._check_seg_dtype(
            mask=self.maps["nucleus_segmentation"], mask_name="nucleus"
        )

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"]]
        )

        return segmentation

    def cellpose_segmentation(self, input_image):
        
        #ensure we have a numpy array
        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data.compute()
        
        self._check_gpu_status()
        self._clear_cache()  # ensure we start with an empty cache

        ################################
        ### Perform Nucleus Segmentation
        ################################

        diameter, model = self._load_model(
            model_type="nucleus", gpu=self.use_GPU, device=self.device
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
                input_image = input_image if self.debug else None,
            )

        # save segementation to maps for access from other subfunctions
        self.maps["nucleus_segmentation"] = masks.reshape(masks.shape[1:])

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model, diameter, masks])

    def process(self, input_image):

        # check that the correct level of input image is used
        self._transform_input_image(input_image)

        # check that the image is of the correct dtype
        self._check_input_image_dtype(input_image)

        # only get the first cannel for segmentation (does not use excess space on the GPU this way)
        input_image = input_image[:1, :, :]

        # initialize location to save masks to
        self.maps = {
            "nucleus_segmentation": tempmmap.array(
                shape=(1, input_image.shape[1], input_image.shape[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
        }

        self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(input_image)

        # finalize classes list
        all_classes = set(np.unique(self.maps["nucleus_segmentation"])) - set([0])

        segmentation = self._finalize_segmentation_results()
        self._save_segmentation_sdata(segmentation, all_classes, masks = self.MASK_NAMES)


class ShardedDAPISegmentationCellpose(ShardedSegmentation):
    method = DAPISegmentationCellpose


class CytosolSegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 2
    MASK_NAMES = ["nuclei", "cytosol"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # ensure correct dtype of maps

        self.maps["nucleus_segmentation"] = self._check_seg_dtype(
            mask=self.maps["nucleus_segmentation"], mask_name="nucleus"
        )
        self.maps["cytosol_segmentation"] = self._check_seg_dtype(
            mask=self.maps["cytosol_segmentation"], mask_name="cytosol"
        )

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["cytosol_segmentation"]]
        )

        return segmentation

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=["nucleus", "cytosol"])
        self._check_for_mask_matching_filtering()

    def cellpose_segmentation(self, input_image):

        #ensure we have a numpy array
        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data.compute()
        
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

        # remove edge labels from masks
        masks_nucleus = remove_edge_labels(masks_nucleus)

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

        # remove edge labels from masks
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
                input_image = input_image if self.debug else None,
            )

            masks_cytosol = self._perform_size_filtering(
                mask=masks_cytosol,
                thresholds=self.nucleus_thresholds,
                confidence_interval=self.nucleus_confidence_interval,
                mask_name="cytosol",
                log=True,
                debug=self.debug,
                input_image = input_image if self.debug else None,
            )

        ######################
        ### Perform Filtering match cytosol and nucleus IDs if applicable
        ######################
        self._setup_filtering()

        if self.filter_match_masks:
            masks_nucleus, masks_cytosol = self._perform_mask_matching_filtering(
                nucleus_mask=masks_nucleus,
                cytosol_mask=masks_cytosol,
                input_image = input_image if self.debug else None,
                filtering_threshold=self.mask_matching_filtering_threshold,
                debug=self.debug,
            )
        else:
            self.log(
                "No filtering performed. Cytosol and Nucleus IDs in the two masks do not match. Before proceeding with extraction an additional filtering step needs to be performed"
            )

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
        # ensure the correct level is selected for the input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # only get the first two input image channels to perform segmentation to optimize memory usage on the GPU
        input_image = input_image[:2, :, :]

        # initialize location to save masks to
        self.maps = {
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
        }

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(input_image)

        # finalize segmentation classes ensuring that background is removed
        all_classes = set(np.unique(self.maps["nucleus_segmentation"])) - set([0])

        segmentation = self._finalize_segmentation_results()
        self._save_segmentation_sdata(segmentation, all_classes, masks = self.MASK_NAMES)

        # clean up memory
        self._clear_cache(vars_to_delete=[segmentation, all_classes])


class ShardedCytosolSegmentationCellpose(ShardedSegmentation):
    method = CytosolSegmentationCellpose


class CytosolSegmentationDownsamplingCellpose(CytosolSegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        self.maps["fullsize_nucleus_segmentation"] = self._rescale_downsampled_mask(
            self.maps["nucleus_segmentation"], "nucleus_segmentation"
        )
        self.maps["fullsize_cytosol_segmentation"] = self._rescale_downsampled_mask(
            self.maps["cytosol_segmentation"], "cytosol_segmentation"
        )

        self.maps["fullsize_nucleus_segmentation"] = self._check_seg_dtype(mask=self.maps["fullsize_nucleus_segmentation"], mask_name="nucleus")
        self.maps["fullsize_cytosol_segmentation"] = self._check_seg_dtype(mask=self.maps["fullsize_cytosol_segmentation"], mask_name="cytosol")

        # combine masks into one stack
        segmentation = np.stack([self.maps["fullsize_nucleus_segmentation"], self.maps["fullsize_cytosol_segmentation"]])

        return segmentation

    def process(self, input_image):
        # ensure the correct level is selected for the input image
        self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # only get the first two channels to save memory consumption
        input_image = input_image[:2, :, :] 

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        # setup the memory mapped arrays to store the results
        self.maps = {
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

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(input_image)

        # finalize classes list
        all_classes = set(np.unique(self.maps["nucleus_segmentation"])) - set([0])

        segmentation = self._finalize_segmentation_results()

        self._save_segmentation_sdata(segmentation, all_classes, masks = self.MASK_NAMES)
        self._clear_cache(vars_to_delete=[segmentation, all_classes])


class ShardedCytosolSegmentationDownsamplingCellpose(ShardedSegmentation):
    method = CytosolSegmentationDownsamplingCellpose


class CytosolOnlySegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 1
    MASK_NAMES = ["cytosol"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=["cytosol"])

    def _finalize_segmentation_results(self):

        # ensure correct dtype of maps
        self.maps["cytosol_segmentation"] = self._check_seg_dtype(
            mask=self.maps["cytosol_segmentation"], mask_name="cytosol"
        )

        segmentation = np.stack(
            [self.maps["cytosol_segmentation"]]
        )

        return segmentation

    def cellpose_segmentation(self, input_image):
        
        #ensure we have a numpy array
        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data.compute()

        self._setup_processing()
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
                input_image = input_image if self.debug else None,
            )

        self.maps["cytosol_segmentation"] = masks_cytosol.reshape(
            masks_cytosol.shape[1:]
        )  # add reshape to match shape to HDF5 shape

        # clear memory
        self._clear_cache(vars_to_delete=[masks_cytosol])

    def _execute_segmentation(self, input_image) -> None:
        total_time_start = timeit.default_timer()

        # transform input image
        start_transform = timeit.default_timer()
        self._transform_input_image(input_image)
        stop_transform = timeit.default_timer()
        self.transform_time = stop_transform - start_transform

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # only get the first two channels for segmentation (does not use excess space on the GPU this way)
        input_image = input_image[:2, :, :]  # we still need both even though its cytosol only because the cytosol models optionally also take the nucleus channel for additional information

        # initialize location to save masks to
        self.maps = {
            "cytosol_segmentation": tempmmap.array(
                shape=(1, input_image.shape[1], input_image.shape[2]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            ),
        }

        # execute segmentation
        start_segmentation = timeit.default_timer()
        self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # get final classes list
        all_classes = set(np.unique(self.maps["cytosol_segmentation"])) - set([0])

        segmentation = self._finalize_segmentation_results()
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)

        # clean up memory
        self._clear_cache(vars_to_delete=[segmentation, all_classes])

        self.total_time = timeit.default_timer() - total_time_start

        return None


class Sharded_CytosolOnly_Cellpose_Segmentation(ShardedSegmentation):
    method = CytosolOnlySegmentationCellpose


class CytosolOnly_Segmentation_Downsampling_Cellpose(CytosolOnlySegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, size_padding):
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
        segmentation = np.stack([cyto_seg]).astype(
            self.DEFAULT_SEGMENTATION_DTYPE
        )
        del cyto_seg

        # rescale segmentation results to original size
        x_trim = x - self.input_image_size[1]
        y_trim = y - self.input_image_size[2]

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

        if segmentation.shape[1] != self.input_image_size[1]:
            sys.exit("Error. Segmentation mask and image have different shapes")
        if segmentation.shape[2] != self.input_image_size[2]:
            sys.exit("Error. Segmentation mask and image have different shapes")

        return segmentation

    def process(self, input_image) -> None:
        # ensure the correct level is selected for the input image
        self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        #get the relevant channels for processing
        input_image = input_image[:2, :, :]
        self.input_image_size = input_image.shape

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        input_image = self._downsample_image(input_image)

        # setup the memory mapped arrays to store the results
        self.maps = {
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

        self.cellpose_segmentation(input_image)

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = set(np.unique(self.maps["cytosol_segmentation"])) - set([0])

        segmentation = self._finalize_segmentation_results()

        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)

        return None



class Sharded_CytosolOnly_Segmentation_Downsampling_Cellpose(ShardedSegmentation):
    method = CytosolOnly_Segmentation_Downsampling_Cellpose

#### TIMECOURSE SEGMENTATION METHODS #####
#### THIS SHOULD BE SWITCHED TO THE BATCHED CLASS IMPLEMENTED BY TIM ####
#currently these are not functional with the new spatialdata format

# class WGA_TimecourseSegmentation(TimecourseSegmentation):
#     """
#     Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
#     No intermediate results are saved and everything is written to one .hdf5 file.
#     """

#     class WGASegmentation_Timecourse(WGASegmentation, TimecourseSegmentation):
#         method = WGASegmentation

#     method = WGASegmentation_Timecourse

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class Multithreaded_WGA_TimecourseSegmentation(MultithreadedSegmentation):
#     class WGASegmentation_Timecourse(WGASegmentation, TimecourseSegmentation):
#         method = WGASegmentation

#     method = WGASegmentation_Timecourse

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class Cytosol_Cellpose_TimecourseSegmentation(TimecourseSegmentation):
#     """
#     Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
#     No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
#     """

#     class CytosolSegmentationCellpose_Timecourse(
#         CytosolSegmentationCellpose, TimecourseSegmentation
#     ):
#         method = CytosolSegmentationCellpose

#     method = CytosolSegmentationCellpose_Timecourse

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class Cytosol_Cellpose_Downsampling_TimecourseSegmentation(TimecourseSegmentation):
#     """
#     Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
#     No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
#     """

#     class Cytosol_Segmentation_Downsampling_Cellpose_Timecourse(
#         CytosolSegmentationDownsamplingCellpose, TimecourseSegmentation
#     ):
#         method = CytosolSegmentationDownsamplingCellpose

#     method = Cytosol_Segmentation_Downsampling_Cellpose_Timecourse

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class CytosolOnly_Cellpose_TimecourseSegmentation(TimecourseSegmentation):
#     """
#     Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
#     No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
#     """

#     class CytosolOnly_Cellpose_TimecourseSegmentation(
#         CytosolOnlySegmentationCellpose, TimecourseSegmentation
#     ):
#         method = CytosolOnlySegmentationCellpose

#     method = CytosolOnly_Cellpose_TimecourseSegmentation

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class Multithreaded_Cytosol_Cellpose_TimecourseSegmentation(MultithreadedSegmentation):
#     class CytosolSegmentationCellpose_Timecourse(
#         CytosolSegmentationCellpose, TimecourseSegmentation
#     ):
#         method = CytosolSegmentationCellpose

#     method = CytosolSegmentationCellpose_Timecourse

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class Multithreaded_Cytosol_Cellpose_Downsampling_TimecourseSegmentation(
#     MultithreadedSegmentation
# ):
#     class Cytosol_Segmentation_Downsampling_Cellpose_Timecourse(
#         CytosolSegmentationDownsamplingCellpose, TimecourseSegmentation
#     ):
#         method = CytosolSegmentationDownsamplingCellpose

#     method = Cytosol_Segmentation_Downsampling_Cellpose_Timecourse

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class Multithreaded_CytosolOnly_Cellpose_TimecourseSegmentation(
#     MultithreadedSegmentation
# ):
#     class CytosolOnly_SegmentationCellpose_Timecourse(
#         CytosolOnlySegmentationCellpose, TimecourseSegmentation
#     ):
#         method = CytosolOnlySegmentationCellpose

#     method = CytosolOnly_SegmentationCellpose_Timecourse

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
