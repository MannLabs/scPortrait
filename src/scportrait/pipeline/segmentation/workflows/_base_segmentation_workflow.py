import os
import sys
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import xarray
from skimage.filters import median
from skimage.morphology import dilation, disk, erosion

from scportrait.pipeline.segmentation.segmentation import (
    Segmentation,
)
from scportrait.plotting._utils import _custom_cmap
from scportrait.processing.images._image_processing import downsample_img, percentile_normalization
from scportrait.processing.masks.mask_filtering import MatchNucleusCytosolIds, SizeFilter


class _BaseSegmentation(Segmentation):
    """Base Segmentation Workflow class that implements methods that can be reused across different segmentation workflows."""

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
        thresholds: tuple[float, float] | None,
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
                        )
                    else:
                        self.filehandler._write_segmentation_sdata(
                            mask[0],
                            segmentation_label=f"debugging_seg_size_filter_results_{mask_name}",
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
                        mask_nuc[0], segmentation_label="debugging_seg_match_mask_results_nucleus"
                    )
                    self.filehandler._write_segmentation_sdata(
                        mask_cyto[0], segmentation_label="debugging_seg_match_mask_result_cytosol"
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
