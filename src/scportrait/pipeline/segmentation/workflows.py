import multiprocessing
import os
import sys
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray
from alphabase.io import tempmmap
from cellpose import models
from skfmm import travel_time as skfmm_travel_time
from skimage.color import label2rgb
from skimage.filters import median
from skimage.morphology import binary_erosion, dilation, disk, erosion
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
    Segmentation,
    ShardedSegmentation,
)
from scportrait.plotting._utils import _custom_cmap
from scportrait.processing.images._image_processing import downsample_img, percentile_normalization
from scportrait.processing.masks.mask_filtering import MatchNucleusCytosolIds, SizeFilter


class _BaseSegmentation(Segmentation):
    MASK_NAMES = ["nucleus", "cytosol"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_channel_selection()
        self.nGPUs = None

    def _setup_channel_selection(self):
        self._setup_maximum_intensity_projection()
        self._define_channels_to_extract_for_segmentation()
        self._remap_maximum_intensity_projection_channels()

    def _setup_maximum_intensity_projection(self):
        # check if channels that should be maximum intensity projected and combined are defined in the config
        if "combine_cytosol_channels" in self.config.keys():
            self.combine_cytosol_channels = self.config["combine_cytosol_channels"]
            assert isinstance(
                self.combine_cytosol_channels, list
            ), "combine_cytosol_channels must be a list of integers specifying the indexes of the channels to combine."
            assert (
                len(self.combine_cytosol_channels) > 1
            ), "combine_cytosol_channels must contain at least two integers specifying the indexes of the channels to combine."
            self.maximum_project_cytosol = True
        else:
            self.combine_cytosol_channels = None
            self.maximum_project_cytosol = False

        if "combine_nucleus_channels" in self.config.keys():
            self.combine_nucleus_channels = self.config["combine_nucleus_channels"]
            assert isinstance(
                self.combine_nucleus_channels, list
            ), "combine_nucleus_channels must be a list of integers specifying the indexes of the channels to combine."
            assert (
                len(self.combine_nucleus_channels) > 1
            ), "combine_nucleus_channels must contain at least two integers specifying the indexes of the channels to combine."
            self.maximum_project_nucleus = True
        else:
            self.combine_nucleus_channels = None
            self.maximum_project_nucleus = False

    def _define_channels_to_extract_for_segmentation(self):
        self.segmentation_channels = []

        if "nucleus" in self.MASK_NAMES:
            if "segmentation_channel_nuclei" in self.config.keys():
                self.nucleus_segmentation_channel = self.config["segmentation_channel_nuclei"]
            elif "combine_nucleus_channels" in self.config.keys():
                self.nucleus_segmentation_channel = self.combine_nucleus_channels
            else:
                self.nucleus_segmentation_channel = self.DEFAULT_NUCLEI_CHANNEL_IDS

            self.segmentation_channels.extend(self.nucleus_segmentation_channel)

        if "cytosol" in self.MASK_NAMES:
            if "segmentation_channel_cytosol" in self.config.keys():
                self.cytosol_segmentation_channel = self.config["segmentation_channel_cytosol"]
            elif "combine_cytosol_channels" in self.config.keys():
                self.cytosol_segmentation_channel = self.combine_cytosol_channels
            else:
                self.cytosol_segmentation_channel = self.DEFAULT_CYTOSOL_CHANNEL_IDS

            self.segmentation_channels.extend(self.cytosol_segmentation_channel)

        # remove any duplicate entries and sort according to order
        self.segmentation_channels = list(set(self.segmentation_channels))

        # check validity of resulting list of segmentation channels
        assert len(self.segmentation_channels) > 0, "No segmentation channels specified in config file."
        assert (
            len(self.segmentation_channels) >= self.N_INPUT_CHANNELS
        ), f"Fewer segmentation channels {self.segmentation_channels} provided than expected by segmentation method {self.N_INPUT_CHANNELS}."

        if len(self.segmentation_channels) > self.N_INPUT_CHANNELS:
            assert (
                self.maximum_project_nucleus or self.maximum_project_cytosol
            ), "More input channels provided than accepted by the segmentation method and no maximum intensity projection performed on any of the input values."

    def _remap_maximum_intensity_projection_channels(self):
        """After selecting channels that are passed to the segmentation update indexes of the channels for maximum intensity projection so that they reflect the provided image subset"""
        if self.maximum_project_nucleus:
            self.original_combine_nucleus_channels = self.combine_nucleus_channels
            self.combine_nucleus_channels = [self.segmentation_channels.index(x) for x in self.combine_nucleus_channels]
        if self.maximum_project_cytosol:
            self.original_combine_cytosol_channels = self.combine_cytosol_channels
            self.combine_cytosol_channels = [self.segmentation_channels.index(x) for x in self.combine_cytosol_channels]

    def _transform_input_image(self, input_image):
        start_transform = timeit.default_timer()

        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data.compute()

        values = []
        # check if any channels need to be transformed
        if "nucleus" in self.MASK_NAMES:
            if self.maximum_project_nucleus:
                self.log(
                    f"For nucleus segmentation using the maximum intensity projection of channels {self.original_combine_nucleus_channels}."
                )
                nucleus_channel = self._maximum_project_channels(input_image, self.combine_nucleus_channels)
                if len(nucleus_channel.shape) == 2:
                    nucleus_channel = nucleus_channel[np.newaxis, ...]
                if len(nucleus_channel.shape) == 4:
                    nucleus_channel = nucleus_channel.squeeze()
            else:
                nucleus_channel = input_image[self.DEFAULT_NUCLEI_CHANNEL_IDS]
                if len(nucleus_channel.shape) == 2:
                    nucleus_channel = nucleus_channel[np.newaxis, ...]
                if len(nucleus_channel.shape) == 4:
                    nucleus_channel = nucleus_channel.squeeze()
            values.append(nucleus_channel)

        if "cytosol" in self.MASK_NAMES:
            if self.maximum_project_cytosol:
                self.log(
                    f"For cytosol segmentation using the maximum intensity projection of channels {self.original_combine_cytosol_channels}."
                )
                cytosol_channel = self._maximum_project_channels(input_image, self.combine_cytosol_channels)
                if len(cytosol_channel.shape) == 4:
                    cytosol_channel = cytosol_channel.squeeze()
                if len(cytosol_channel.shape) == 2:
                    cytosol_channel = cytosol_channel[np.newaxis, ...]
            else:
                cytosol_channel = input_image[self.DEFAULT_CYTOSOL_CHANNEL_IDS]
                if len(cytosol_channel.shape) == 4:
                    cytosol_channel = cytosol_channel.squeeze()
                if len(cytosol_channel.shape) == 2:
                    cytosol_channel = cytosol_channel[np.newaxis, ...]

            values.append(cytosol_channel)

        input_image = np.vstack(values)

        assert (
            input_image.shape[0] == self.N_INPUT_CHANNELS
        ), f"Number of channels in input image {input_image.shape[0]} does not match the number of channels expected by segmentation method {self.N_INPUT_CHANNELS}."

        stop_transform = timeit.default_timer()
        self.transform_time = stop_transform - start_transform
        return input_image

    def return_empty_mask(self, input_image):
        _, x, y = input_image.shape
        self._save_segmentation_sdata(np.zeros((self.N_MASKS, x, y)), [])

    def _check_seg_dtype(self, mask: np.ndarray, mask_name: str) -> np.ndarray:
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
            assert (mask.shape[0] == self.original_image_size[1]) and (mask.shape[1] == self.original_image_size[2])
        elif len(mask.shape) == 3:
            assert (mask.shape[1] == self.original_image_size[1]) and (mask.shape[2] == self.original_image_size[2])

        return mask

    #### Image Processing #####

    def _maximum_project_channels(self, input_image: np.array, channel_ids: list[int]) -> np.array:
        """add multiple channels together to generate a new channel for segmentation using maximum intensity projection"""

        dtype = input_image.dtype
        new_channel = np.zeros_like(input_image[0], dtype=dtype)

        for channel_id in channel_ids:
            new_channel = np.maximum(new_channel, input_image[channel_id])

        return new_channel

    def _normalize_image(
        self,
        input_image: np.array,
        lower: float | list,
        upper: float | list,
        debug: bool = False,
    ) -> np.array:
        if isinstance(lower, float) and isinstance(upper, float):
            self.log("Normalizing each channel to the same range")
            norm_image = percentile_normalization(input_image, lower, upper)

        elif isinstance(lower, list) and isinstance(upper, list):
            norm_image = []

            for i in range(input_image.shape[0]):
                _lower = lower[i]
                _upper = upper[i]  # type: ignore

                norm_image.append(percentile_normalization(input_image[i], _lower, _upper))

            norm_image = np.array(norm_image)
        else:
            raise ValueError(
                "Lower and upper quantile normalization values must be either floats or dictionary of floats."
            )
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

    def _median_correct_image(self, input_image, median_filter_size: int, debug: bool = False):
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
    def _check_for_size_filtering(self, mask_types: list[str]) -> None:
        """
        Check if size filtering should be performed on the masks.
        If size filtering is turned on, the thresholds for filtering are loaded from the config file.
        """

        assert all(
            mask_type in self.MASK_NAMES for mask_type in mask_types
        ), f"mask_types must be a list of strings that are valid mask names {self.MASK_NAMES}."

        if "filter_masks_size" in self.config.keys():
            self.filter_size = self.config["filter_masks_size"]
        else:
            # default behaviour is this is turned off filtering can always be performed later and this preserves the whole segmentation mask
            self.filter_size = False
            for mask_type in mask_types:
                # save attributes for use later
                setattr(self, f"{mask_type}_thresholds", None)
                setattr(self, f"{mask_type}_confidence_interval", None)

        # load parameters for cellsize filtering
        if self.filter_size:
            for mask_type in mask_types:
                thresholds, confidence_interval = self._get_params_cellsize_filtering(type=mask_type)

                # save attributes for use later
                setattr(self, f"{mask_type}_thresholds", thresholds)
                setattr(self, f"{mask_type}_confidence_interval", confidence_interval)

    def _get_params_cellsize_filtering(self, type) -> tuple[tuple[float] | None, float | None]:
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
            return (thresholds, None)  # type: ignore
        else:
            thresholds = None

            # get confidence intervals to automatically calculate thresholds
            if "confidence_interval" in self.config[f"{type}_segmentation"].keys():
                confidence_interval = self.config[f"{type}_segmentation"]["confidence_interval"]
            else:
                # get default value
                self.log(f"No confidence interval specified for {type} mask filtering, using default value of 0.95")
                confidence_interval = 0.95

            return (thresholds, confidence_interval)

    def _perform_size_filtering(
        self,
        mask: np.array,
        thresholds: tuple[float] | None,
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
            self.log(f"Performing filtering of {mask_name} with specified thresholds {thresholds} from config file.")
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
            # get visualization of the filtering results (2 = filtered, 1 = keep, 0 = background)
            mask = filter.visualize_filtering_results(plot_fig=False, return_maps=True)

            if not self.is_shard:
                if self.save_filter_results:
                    if len(mask.shape) == 2:
                        # add results to sdata object
                        self.filehandler._write_segmentation_sdata(
                            mask,
                            segmentation_label=f"debugging_seg_size_filter_results_{mask_name}",
                            classes=filter.ids,
                        )
                    else:
                        self.filehandler._write_segmentation_sdata(
                            mask[0],
                            segmentation_label=f"debugging_seg_size_filter_results_{mask_name}",
                            classes=filter.ids,
                        )
                    # then this does not need to be plotted as it can be visualized from there
                    plot_results = False
                else:
                    plot_results = True
            else:
                plot_results = True

            if plot_results:
                # get input image for visualization
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
            self._clear_cache(vars_to_delete=[mask, plot_results])

        end_time = time.time()

        self.log(f"Total time to perform {mask_name} size filtering: {end_time - start_time} seconds")

        return filtered_mask

    # 2. matching masks
    def _check_for_mask_matching_filtering(self) -> None:
        """Check to see if the masks should be filtered for matching nuclei/cytosols within the segmentation run."""

        DEFAULT_FILTERING_THRESHOLD_MASK_MATCHING = 0.95
        # check to see if the cells should be filtered for matching nuclei/cytosols within the segmentation run
        if "match_masks" in self.config.keys():
            self.filter_match_masks = self.config["match_masks"]
            if "filtering_threshold_mask_matching" in self.config.keys():
                self.mask_matching_filtering_threshold = self.config["filtering_threshold_mask_matching"]
            else:
                self.mask_matching_filtering_threshold = (
                    DEFAULT_FILTERING_THRESHOLD_MASK_MATCHING  # set default parameter
                )

        else:
            # default behaviour that this filtering should be performed, otherwise another additional step is required before extraction
            self.filter_match_masks = True
            self.mask_matching_filtering_threshold = DEFAULT_FILTERING_THRESHOLD_MASK_MATCHING

        # sanity check provided values
        assert isinstance(self.filter_match_masks, bool), "`match_masks` must be a boolean value."
        if self.filter_match_masks:
            assert isinstance(
                self.mask_matching_filtering_threshold, float
            ), "`filtering_threshold_mask_matching` for mask matching must be a float."

    def _perform_mask_matching_filtering(
        self,
        nucleus_mask: np.array,
        cytosol_mask: np.array,
        filtering_threshold: float,
        debug: bool = False,
        input_image: np.array = None,
    ) -> tuple[np.array, np.array]:
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
        masks_nucleus, masks_cytosol = filter.filter(nucleus_mask=nucleus_mask, cytosol_mask=cytosol_mask)

        self.log(
            f"Removed {len(filter.nuclei_discard_list)} nuclei and {len(filter.cytosol_discard_list)} cytosols due to filtering."
        )
        self.log(f"After filtering, {len(filter.nucleus_lookup_dict)} matching nuclei and cytosol masks remain.")

        if debug:
            mask_nuc, mask_cyto = filter.visualize_filtering_results(plot_fig=False, return_maps=True)

            # check if image should be added to sdata or plotted and saved
            if not self.is_shard:
                if self.save_filter_results:
                    # add filtering results to sdata object
                    self.filehandler._write_segmentation_sdata(
                        mask_nuc[0], segmentation_label="debugging_seg_match_mask_results_nucleus", classes=None
                    )
                    self.filehandler._write_segmentation_sdata(
                        mask_cyto[0], segmentation_label="debugging_seg_match_mask_result_cytosol", classes=None
                    )

                    # then no plotting needs to be performed as the results can be viewed in the sdata object
                    plot_results = False
                else:
                    plot_results = True
            else:
                plot_results = True

            if plot_results:
                if input_image is not None:
                    # convert input image from uint16 to uint8
                    input_image = (input_image / 256).astype(np.uint8)

                    cmap, norm = _custom_cmap()

                    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

                    axs[0].imshow(input_image[0], cmap="gray")
                    axs[0].imshow(mask_nuc[0], cmap=cmap, norm=norm)
                    axs[0].imshow(mask_cyto[0], cmap=cmap, norm=norm)
                    axs[0].axis("off")
                    axs[0].set_title("results overlayed nucleus channel")

                    axs[1].imshow(input_image[1], cmap="gray")
                    axs[1].imshow(mask_nuc[0], cmap=cmap, norm=norm)
                    axs[1].imshow(mask_cyto[0], cmap=cmap, norm=norm)
                    axs[1].axis("off")
                    axs[1].set_title("results overlayed cytosol channel")

                    fig.tight_layout()
                    fig_path = os.path.join(self.directory, "Results_mask_matching.png")
                    fig.savefig(fig_path)

                    self._clear_cache(vars_to_delete=[fig, input_image, cmap, norm])

            # clearup memory
            self._clear_cache(vars_to_delete=[mask_nuc, mask_cyto, plot_results])

        self.log(
            f"Total time to perform nucleus and cytosol mask matching filtering: {time.time() - start_time:.2f} seconds"
        )

        return masks_nucleus, masks_cytosol


###### CLASSICAL SEGMENTATION METHODS #####


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


##### CELLPOSE BASED SEGMENTATION METHODS #####


class _CellposeSegmentation(_BaseSegmentation):
    def _write_cellpose_seg_params_to_file(self, model_type: str, model_name: str) -> None:
        """
        Writes the cellpose segmentation parameters to a file for debugging/logging purposes
        """
        with open(os.path.join(self.directory, f"cellpose_params_{model_type}.txt"), "w") as f:
            f.write("Cellpose Parameters:\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Diameter: {self.diameter}\n")
            f.write(f"Resample: {self.resample}\n")
            f.write(f"Flow Threshold: {self.flow_threshold}\n")
            f.write(f"Cellprob Threshold: {self.cellprob_threshold}\n")
            f.write(f"Normalize: {self.normalize}\n")
            f.write(f"Rescale: {self.rescale}\n")

    def _read_cellpose_model(self, modeltype: str, name: str, gpu: str, device) -> models.Cellpose:
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

    def _load_model(self, model_type: str, gpu: str, device) -> models.Cellpose:
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
            model = self._read_cellpose_model("pretrained", model_name, gpu=gpu, device=device)

        elif "model_path" in self.config[f"{model_type}_segmentation"].keys():
            model_name = self.config[f"{model_type}_segmentation"]["model_path"]
            model = self._read_cellpose_model("custom", model_name, gpu=gpu, device=device)

        # get model parameters from config if not defined use default values
        if "diameter" in self.config[f"{model_type}_segmentation"].keys():
            self.diameter = self.config[f"{model_type}_segmentation"]["diameter"]
        else:
            self.diameter = None

        if "resample" in self.config[f"{model_type}_segmentation"].keys():
            self.resample = self.config[f"{model_type}_segmentation"]["resample"]
        else:
            self.resample = True

        if "flow_threshold" in self.config[f"{model_type}_segmentation"].keys():
            self.flow_threshold = self.config[f"{model_type}_segmentation"]["flow_threshold"]
        else:
            self.flow_threshold = 0.4

        if "cellprob_threshold" in self.config[f"{model_type}_segmentation"].keys():
            self.cellprob_threshold = self.config[f"{model_type}_segmentation"]["cellprob_threshold"]
        else:
            self.cellprob_threshold = 0.0

        if "normalize" in self.config[f"{model_type}_segmentation"].keys():
            self.normalize = self.config[f"{model_type}_segmentation"]["normalize"]
        else:
            self.normalize = True

        if "rescale" in self.config[f"{model_type}_segmentation"].keys():
            self.rescale = self.config[f"{model_type}_segmentation"]["rescale"]
        else:
            self.rescale = None

        self.log(f"Segmenting {model_type} using the following model: {model_name}")

        self._write_cellpose_seg_params_to_file(model_type=model_type, model_name=model_name)

        return model

    def _check_input_image_dtype(self, input_image: np.ndarray):
        if input_image.dtype != self.DEFAULT_IMAGE_DTYPE:
            if isinstance(input_image.dtype, int):
                ValueError(
                    "Default image dtype is no longer int. Cellpose expects int inputs. Please contact developers."
                )
            else:
                ValueError("Image is not of type uint16, cellpose segmentation expects int input images.")

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

            if cpu_id >= len(gpu_id_list):
                cpu_id = cpu_id % current.n_processes

            # track gpu_id and update GPU status
            self.gpu_id = gpu_id_list[cpu_id]
            self.status = "multi_GPU"

        except (AttributeError, ValueError):
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
    N_INPUT_CHANNELS = 1
    MASK_NAMES = ["nucleus"]
    DEFAULT_NUCLEI_CHANNEL_IDS = [0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=self.MASK_NAMES)

    def _finalize_segmentation_results(self, nucleus_mask: np.ndarray) -> np.ndarray:
        # ensure correct dtype of the maps

        nucleus_mask = self._check_seg_dtype(mask=nucleus_mask, mask_name="nucleus")

        segmentation = np.stack([nucleus_mask])

        return segmentation

    def cellpose_segmentation(self, input_image: np.ndarray) -> np.ndarray:
        self._check_gpu_status()
        self._clear_cache()  # ensure we start with an empty cache

        ################################
        ### Perform Nucleus Segmentation
        ################################

        model = self._load_model(model_type="nucleus", gpu=self.use_GPU, device=self.device)

        if self.normalize is False:
            input_image = (input_image - np.min(input_image)) / (
                np.max(input_image) - np.min(input_image)
            )  # min max normalize to 0-1 range as cellpose expects this

        masks = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[1, 0],
        )[0]
        masks = np.array(masks)

        # ensure all edge classes are removed
        masks = remove_edge_labels(masks)

        # check if filtering is required
        self._setup_filtering()

        if self.filter_size:
            masks = self._perform_size_filtering(
                mask=masks,
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="nucleus",
                log=True,
                input_image=input_image if self.debug else None,
            )

        masks = masks.reshape(masks.shape[1:])
        return masks

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()

        # check that the correct level of input image is used
        input_image = self._transform_input_image(input_image)

        self._check_input_image_dtype(input_image)

        start_segmentation = timeit.default_timer()
        nucleus_mask = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # finalize classes list
        all_classes = set(np.unique(nucleus_mask)) - {0}

        segmentation = self._finalize_segmentation_results(nucleus_mask=nucleus_mask)
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self.total_time = timeit.default_timer() - total_time_start


class ShardedDAPISegmentationCellpose(ShardedSegmentation):
    method = DAPISegmentationCellpose


class CytosolSegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 2
    N_INPUT_CHANNELS = 2
    MASK_NAMES = ["nucleus", "cytosol"]
    DEFAULT_NUCLEI_CHANNEL_IDS = [0]
    DEFAULT_CYTOSOL_CHANNEL_IDS = [1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, mask_nucleus: np.ndarray, mask_cytosol: np.ndarray) -> np.ndarray:
        # ensure correct dtype of maps

        mask_nucleus = self._check_seg_dtype(mask=mask_nucleus, mask_name="nucleus")
        mask_cytosol = self._check_seg_dtype(mask=mask_cytosol, mask_name="cytosol")

        segmentation = np.stack([mask_nucleus, mask_cytosol])
        return segmentation

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=self.MASK_NAMES)
        self._check_for_mask_matching_filtering()

    def cellpose_segmentation(self, input_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_gpu_status()
        self._clear_cache()  # ensure we start with an empty cache

        ################################
        ### Perform Nucleus Segmentation
        ################################

        model = self._load_model(model_type="nucleus", gpu=self.use_GPU, device=self.device)

        if self.normalize is False:
            input_image = (input_image - np.min(input_image)) / (
                np.max(input_image) - np.min(input_image)
            )  # min max normalize to 0-1 range as cellpose expects this

        masks_nucleus = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[1, 0],
        )[0]
        masks_nucleus = np.array(masks_nucleus)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model])

        # remove edge labels from masks
        masks_nucleus = remove_edge_labels(masks_nucleus)

        #################################
        #### Perform Cytosol Segmentation
        #################################

        model = self._load_model(model_type="cytosol", gpu=self.use_GPU, device=self.device)

        masks_cytosol = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[2, 1],
        )[0]
        masks_cytosol = np.array(masks_cytosol)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model])

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
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="nucleus",
                log=True,
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

            masks_cytosol = self._perform_size_filtering(
                mask=masks_cytosol,
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="cytosol",
                log=True,
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

        ######################
        ### Perform Filtering match cytosol and nucleus IDs if applicable
        ######################
        self._setup_filtering()

        if self.filter_match_masks:
            masks_nucleus, masks_cytosol = self._perform_mask_matching_filtering(
                nucleus_mask=masks_nucleus,
                cytosol_mask=masks_cytosol,
                input_image=input_image if self.debug else None,
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

        masks_nucleus = masks_nucleus.reshape(masks_nucleus.shape[1:])
        masks_cytosol = masks_cytosol.reshape(masks_cytosol.shape[1:])

        return (masks_nucleus, masks_cytosol)

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()

        # ensure the correct level is selected for the input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        start_segmentation = timeit.default_timer()
        masks_nucleus, masks_cytosol = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # finalize segmentation classes ensuring that background is removed
        all_classes = set(np.unique(masks_nucleus)) - {0}

        segmentation = self._finalize_segmentation_results(mask_nucleus=masks_nucleus, mask_cytosol=masks_cytosol)
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)

        # clean up memory
        self._clear_cache(vars_to_delete=[segmentation, all_classes])
        self.total_time = timeit.default_timer() - total_time_start


class ShardedCytosolSegmentationCellpose(ShardedSegmentation):
    method = CytosolSegmentationCellpose


class CytosolSegmentationDownsamplingCellpose(CytosolSegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, mask_nucleus: np.ndarray, mask_cytosol: np.ndarray) -> np.ndarray:
        mask_nucleus = self._rescale_downsampled_mask(mask_nucleus, "nucleus_segmentation")
        mask_cytosol = self._rescale_downsampled_mask(mask_cytosol, "cytosol_segmentation")

        mask_nucleus = self._check_seg_dtype(mask=mask_nucleus, mask_name="nucleus")
        mask_cytosol = self._check_seg_dtype(mask=mask_cytosol, mask_name="cytosol")

        # combine masks into one stack
        segmentation = np.stack([mask_nucleus, mask_cytosol])

        return segmentation

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()

        # ensure the correct level is selected for the input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # only get the first two channels to save memory consumption
        input_image = input_image[:2, :, :]

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        # downsample the image
        input_image = self._downsample_image(input_image)

        start_segmentation = timeit.default_timer()
        mask_nucleus, mask_cytosol = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # finalize classes list
        all_classes = set(np.unique(mask_nucleus)) - {0}

        segmentation = self._finalize_segmentation_results(mask_nucleus=mask_nucleus, mask_cytosol=mask_cytosol)

        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self._clear_cache(vars_to_delete=[segmentation, all_classes])
        self.total_time = timeit.default_timer() - total_time_start


class ShardedCytosolSegmentationDownsamplingCellpose(ShardedSegmentation):
    method = CytosolSegmentationDownsamplingCellpose


class CytosolOnlySegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 1
    N_INPUT_CHANNELS = 2
    MASK_NAMES = ["cytosol"]
    DEFAULT_CYTOSOL_CHANNEL_IDS = [0, 1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=self.MASK_NAMES)

    def _finalize_segmentation_results(self, cytosol_mask: np.ndarray) -> np.ndarray:
        # ensure correct dtype of maps
        cytosol_mask = self._check_seg_dtype(mask=cytosol_mask, mask_name="cytosol")

        segmentation = np.stack([cytosol_mask])
        return segmentation

    def cellpose_segmentation(self, input_image: np.ndarray) -> np.ndarray:
        self._setup_processing()
        self._clear_cache()

        #####
        ### Perform Cytosol Segmentation
        #####

        model = self._load_model(model_type="cytosol", gpu=self.use_GPU, device=self.device)

        if self.normalize is False:
            input_image = (input_image - np.min(input_image)) / (
                np.max(input_image) - np.min(input_image)
            )  # min max normalize to 0-1 range as cellpose expects this

        masks_cytosol = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[2, 1],
        )[0]
        masks_cytosol = np.array(masks_cytosol)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model])

        # ensure edge classes are removed
        masks_cytosol = remove_edge_labels(masks_cytosol)

        #####
        ### Perform Filtering to remove too small/too large masks if applicable
        #####

        self._setup_filtering()

        if self.filter_size:
            masks_cytosol = self._perform_size_filtering(
                mask=masks_cytosol,
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="cytosol",
                log=True,
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

        masks_cytosol = masks_cytosol.reshape(masks_cytosol.shape[1:])  # add reshape to match shape to HDF5 shape

        return masks_cytosol

    def _execute_segmentation(self, input_image) -> None:
        total_time_start = timeit.default_timer()

        # transform input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # execute segmentation
        start_segmentation = timeit.default_timer()
        cytosol_mask = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # get final classes list
        all_classes = set(np.unique(cytosol_mask)) - {0}

        segmentation = self._finalize_segmentation_results(cytosol_mask)
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)

        # clean up memory
        self._clear_cache(vars_to_delete=[segmentation, all_classes])

        self.total_time = timeit.default_timer() - total_time_start

        return None


class ShardedCytosolOnlySegmentationCellpose(ShardedSegmentation):
    method = CytosolOnlySegmentationCellpose


class CytosolOnlySegmentationDownsamplingCellpose(CytosolOnlySegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, cytosol_mask: np.ndarray) -> np.ndarray:
        cytosol_mask = self._rescale_downsampled_mask(cytosol_mask, "cytosol_segmentation")

        cytosol_mask = self._check_seg_dtype(mask=cytosol_mask, mask_name="cytosol")

        # combine masks into one stack
        segmentation = np.stack([cytosol_mask])

        return segmentation

    def _execute_segmentation(self, input_image) -> None:
        total_time_start = timeit.default_timer()

        # ensure the correct level is selected for the input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # get the relevant channels for processing
        input_image = input_image[:2, :, :]
        self.input_image_size = input_image.shape

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        # downsample the image
        input_image = self._downsample_image(input_image)

        start_segmentation = timeit.default_timer()
        cytosol_mask = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = set(np.unique(cytosol_mask)) - {0}

        segmentation = self._finalize_segmentation_results(cytosol_mask=cytosol_mask)  # type: ignore

        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self.total_time = timeit.default_timer() - total_time_start


class ShardedCytosolOnlySegmentationDownsamplingCellpose(ShardedSegmentation):
    method = CytosolOnlySegmentationDownsamplingCellpose
