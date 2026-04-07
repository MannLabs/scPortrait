import gc
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
import timeit
from functools import partial as func_partial
from pathlib import PosixPath
from typing import TypeAlias

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray
from alphabase.io.tempmmap import create_empty_mmap, mmap_array_from_path
from anndata import AnnData
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian
from spatialdata import SpatialData
from tqdm.auto import tqdm

from scportrait.pipeline._base import ProcessingStep
from scportrait.pipeline._utils.helper import flatten
from scportrait.processing.images._image_processing import percentile_normalization
from scportrait.tools.sdata.write._helper import _normalize_anndata_strings

ExtractionArg: TypeAlias = tuple[int, int, int, tuple[float, float]]
BatchedExtractionArgs: TypeAlias = list[list[ExtractionArg]]


class HDF5CellExtraction(ProcessingStep):
    """
    A class to extracts single cell images from a segmented scPortrait project and save the
    results to an HDF5 file.
    """

    CLEAN_LOG = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.CLEAN_LOG:
            self._clean_log_file()

        self._check_config()

        # setup base extraction workflow
        self._get_compression_type()

        # initialize base variables to save results to
        self.save_index_to_remove = []
        self.batch_size = None

        # set developer debug mode for super detailed output
        self.deep_debug = False

        # check for windows operating system and if so set threads to 1
        if platform.system() == "Windows":
            Warning("Windows detected. Multithreading not supported on windows so setting threads to 1.")
            self.threads = 1

        if "overwrite_run_path" not in self.__dict__.keys():
            self.overwrite_run_path = self.overwrite

        self.extraction_file = os.path.join(self.directory, self.DEFAULT_DATA_DIR, self.DEFAULT_EXTRACTION_FILE)
        self.output_path = None

    def _get_compression_type(self) -> None:
        """setup compression of single-cell images in HDF5 file based on config."""
        # default value for compression is "lzf" is nothing else is specified
        if (self.compression is True) or (self.compression == "lzf"):
            self.compression_type = "lzf"
        elif self.compression == "gzip":
            self.compression_type = "gzip"
        else:
            self.compression_type = None
        self.log(f"Compression algorithm for extracted single-cell images: {self.compression_type}")

    def _check_config(self):
        """Load parameters from config file and check for required parameters."""

        # check for required parameters
        assert "threads" in self.config, "Number of threads must be specified in the config file."
        assert "image_size" in self.config, "Image size must be specified in the config file."
        assert "cache" in self.config, "Cache directory must be specified in the config file."

        self.threads = self.config["threads"]
        self.image_size = self.config["image_size"]

        # optional parameters with default values that can be overridden by the values in the config file

        ## parameters for image extraction
        self.normalization = self._get_normalize_output_config()
        self.normalization_range = self._get_normalization_range_config()

        ## parameters for HDF5 file creates
        self.compression = self.config.get("compression", True)
        self.flush_every = self._get_optional_positive_int_config("flush_every")

        ## Deprecated parameters since we no longer directly read from HDF5
        ## Preservering here in case we see better performance by adjusting this behaviour in how we read data from memmapped arrays

        # if "hdf5_rdcc_nbytes" in self.config:
        #     self.hdf5_rdcc_nbytes = self.config["hdf5_rdcc_nbytes"]
        # else:
        #     self.hdf5_rdcc_nbytes = 5242880000 # 5gb 1024 * 1024 * 5000

        # if "hdf5_rdcc_w0" in self.config:
        #     self.hdf5_rdcc_w0 = self.config["hdf5_rdcc_w0"]
        # else:
        #     self.hdf5_rdcc_w0 = 1

        # if "hdf5_rdcc_nslots" in self.config:
        #     self.hdf5_rdcc_nslots = self.config["hdf5_rdcc_nslots"]
        # else:
        #     self.hdf5_rdcc_nslots = 50000

    def _get_optional_positive_int_config(self, key: str) -> int | None:
        """Return an optional positive integer config value.

        Args:
            key: Config entry to validate and read.

        Returns:
            The configured integer value or ``None`` if the key is absent.
        """
        value = self.config.get(key)
        if value is None:
            return None

        assert isinstance(value, int) and value >= 1, f"{key} must be an integer >= 1."
        return value

    def _get_normalize_output_config(self) -> bool:
        """Resolve normalization enablement from config."""
        normalize = self.config.get("normalize_output", True)
        assert normalize in [True, False, None, "None"], (
            "Normalization must be one of the following values [True, False, None, 'None']"
        )
        if normalize in [None, "None"]:
            return False
        return bool(normalize)

    def _get_normalization_range_config(self) -> tuple[float, float] | tuple[str, str]:
        """Resolve normalization percentile range from config."""
        if not self.normalization:
            return ("None", "None")

        normalization_range = self.config.get("normalization_range", (0.001, 0.999))
        if normalization_range == "None" or normalization_range is None:
            return (0.001, 0.999)

        assert len(normalization_range) == 2, "Normalization range must be a tuple or list of length 2."
        assert all(isinstance(x, float | int) and (0 <= x <= 1) for x in normalization_range), (
            "Normalization range must be defined as a float between 0 and 1."
        )

        if isinstance(normalization_range, list):
            normalization_range = tuple(normalization_range)

        return normalization_range

    def _setup_normalization(self) -> None:
        """Configure normalization of single-cell images based on config.

        Default behaviour is that images are 1-99 percentile normalized and converted to a float between 0 and 1.

        This default behaviour can be overridden by providing a boolean value to normalize_output to configure if the images should be normalized, or
        by providing a tuple to normalization_range to specify the lower and upper percentile values to use for normalization.

        If normalization is set to False, the images will be converted to a float between 0 and 1 without any normalization where the highest value is the maximum value of the image datatype.
        """
        if self.normalization:
            lower, upper = self.normalization_range

            def percentile_norm(img: np.ndarray, lower: float = lower, upper: float = upper) -> np.ndarray:
                return percentile_normalization(img, lower, upper)

            self.norm_function = percentile_norm

        else:

            def min_max(img: np.ndarray, lower=None, upper=None) -> np.ndarray:
                img = (
                    img / np.iinfo(self.DEFAULT_IMAGE_DTYPE).max
                )  # convert 16bit unsigned integer image to float between 0 and 1 without adjusting for the pixel values we have in the extracted single cell image
                return img

            self.norm_function = min_max

    def _get_output_path(self) -> str | PosixPath:
        """Get the output path for the extraction results."""
        return self.extraction_data_directory

    def _setup_output(self, folder_name: str | None = None) -> None:
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # if a foldername is not specified it uses the default value
        if folder_name is None:
            folder_name = self.DEFAULT_DATA_DIR

        self.extraction_data_directory = os.path.join(self.directory, folder_name)

        if not os.path.isdir(self.extraction_data_directory):
            os.makedirs(self.extraction_data_directory)
            self.log(f"Created new directory for extraction results: {self.extraction_data_directory}")
        elif self.overwrite_run_path:
            self.log(f"Output folder at {self.extraction_data_directory} already exists. Overwriting...")
            shutil.rmtree(self.extraction_data_directory)
            os.makedirs(self.extraction_data_directory)
            self.log(f"Created new directory for extraction results: {self.extraction_data_directory}")
        else:
            raise ValueError("Output folder already exists. Set overwrite_run_path to True to overwrite.")

        self.log(f"Setup output folder at {self.extraction_data_directory}")

    def _set_up_extraction(self, output_folder_name: str | None) -> bool:
        """execute all helper functions to setup extraction process.

        Returns:
            Boolean value indicating if cells exist to proceed with extraction workflow.
        """

        if output_folder_name is None:
            output_folder_name = self.DEFAULT_DATA_DIR

        if self.partial_processing:
            output_folder_name = f"partial_{output_folder_name}_ncells_{self.n_cells}_seed_{self.seed}"
        else:
            output_folder_name = output_folder_name

        self._setup_output(folder_name=output_folder_name)
        self._get_segmentation_info()
        self._get_input_image_info()

        # setup number of output channels
        self.n_output_channels = self.n_image_channels + self.n_masks

        # get size of images to extract
        self.extracted_image_size = self.image_size
        self.width_extraction = (
            self.extracted_image_size // 2
        )  # half of the extracted image size (this is what needs to be added on either side of the center)

        self._get_centers()
        self._get_classes_to_extract()

        if self.num_classes == 0:
            Warning("After removing cells to close to border of image to extract no cells remain.")
            return False

        # create output files for saving results to
        self._create_output_files()

        # print relevant information to log file
        self._verbalise_extraction_info()

        return True

    def _get_segmentation_info(self) -> None:
        """execute logic to determine which segmentation masks to use for extraction"""

        # get current version of spatialdata object
        _sdata = self.filehandler._read_sdata()

        segmentation_keys = list(_sdata.labels.keys())

        # specific segmentation masks can be extracted that do not need to conform to the default naming convention in scportrait
        # this behaviour can be achieved by passing a segmentation_key in the config file.
        if "segmentation_key" in self.config.keys():
            self.segmentation_key = self.config["segmentation_key"]
        else:
            self.segmentation_key = "seg_all"  # default segmentation key

        relevant_masks = [x for x in segmentation_keys if self.segmentation_key in x]

        # ensure that we have at least 1 mask to proceed with
        if len(relevant_masks) == 0:
            raise ValueError(
                f"Found no segmentation masks with key {self.segmentation_key}. Cannot proceed with extraction."
            )

        # intialize default values to track what should be extracted
        self.nucleus_key = None
        self.cytosol_key = None
        self.extract_nucleus_mask = False
        self.extract_cytosol_mask = False

        if "segmentation_mask" in self.config:
            if isinstance(self.config["segmentation_mask"], str):
                if "nucleus" in self.config["segmentation_mask"]:
                    self.nucleus_key = self.config["segmentation_mask"]
                    self.extract_nucleus_mask = True
                elif "cytosol" in self.config["segmentation_mask"]:
                    self.cytosol_key = self.config["segmentation_mask"]
                    self.extract_cytosol_mask = True
                else:
                    Warning(
                        "Unclear if the singular provided segmentation mask is a nucleus or cytosol mask. Will proceed assuming cytosol like behaviour."
                    )
                    self.cytosol_key = self.config["segmentation_mask"]
                    self.extract_cytosol_mask = True

            elif isinstance(self.config["segmentation_mask"], list):
                for x in self.config["segmentation_mask"]:
                    if "nucleus" in x:
                        self.nucleus_key = x
                        self.extract_nucleus_mask = True
                    elif "cytosol" in x:
                        self.cytosol_key = x
                        self.extract_cytosol_mask = True
                    else:
                        raise ValueError(
                            f"Segmentation mask {self.config['segmentation_mask']} does not indicate which mask is the nucleus and which is the cytosol. Cannot proceed with extraction."
                        )

        else:
            # get relevant segmentation masks to perform extraction on
            nucleus_key = f"{self.segmentation_key}_nucleus"

            if nucleus_key in relevant_masks:
                self.extract_nucleus_mask = True
                self.nucleus_key = nucleus_key

            cytosol_key = f"{self.segmentation_key}_cytosol"

            if cytosol_key in relevant_masks:
                self.extract_cytosol_mask = True
                self.cytosol_key = cytosol_key

        self.n_masks = np.sum([self.extract_nucleus_mask, self.extract_cytosol_mask])
        self.masks = [x for x in [self.nucleus_key, self.cytosol_key] if x is not None]

        # define the main segmentation mask to extract single-cell images from
        # this mask will be used to calculate the cell centers
        if self.n_masks == 2:
            # perform sanity check that the masks have the same ids
            # THIS NEEDS TO BE IMPLEMENTED HERE

            self.main_segmentation_mask = self.nucleus_key

        elif self.n_masks == 1:
            if self.extract_nucleus_mask:
                self.main_segmentation_mask = self.nucleus_key
            elif self.extract_cytosol_mask:
                self.main_segmentation_mask = self.cytosol_key

        self.log(
            f"Found {self.n_masks} segmentation masks for the given key in the sdata object. Will be extracting single-cell images based on these masks: {self.masks}"
        )
        self.log(f"Using {self.main_segmentation_mask} as the main segmentation mask to determine cell centers.")

    def _get_input_image_info(self) -> None:
        """get relevant information about the input image to be able to extract single-cell images"""
        # get channel information
        input_image = self.filehandler._get_input_image(self.filehandler.get_sdata())
        self.channel_names = input_image.c.values
        self.n_image_channels = len(self.channel_names)
        self.input_image_width = len(input_image.x)
        self.input_image_height = len(input_image.y)

    def _get_centers(self) -> None:
        """get the centers of the cells that should be extracted.
        If a nucleus and cytosol mask are used, the centers are calculated based on the nucleus mask.
        If only one mask is used, the centers are calculated based on that mask.
        """
        _sdata = self.filehandler._read_sdata()

        # calculate centers if they have not been calculated yet
        centers_name = f"{self.DEFAULT_CENTERS_NAME}_{self.main_segmentation_mask}"
        if centers_name not in _sdata:
            self.filehandler._add_centers(self.main_segmentation_mask, overwrite=self.overwrite)
            _sdata = self.filehandler._read_sdata()  # reread to ensure we have updated version

        centers = _sdata[centers_name].values.compute()

        # round to int so that we can use them as indices
        centers = np.round(centers).astype(int)

        self.centers = centers
        self.centers_cell_ids = _sdata[centers_name].index.values.compute()

        # ensure that the centers ids are unique
        assert len(self.centers_cell_ids) == len(set(self.centers_cell_ids)), (
            "Cell ids in centers are not unique. Cannot proceed with extraction."
        )

        # double check that the cell_ids contained in the seg masks match to those from centers
        # THIS NEEDS TO BE IMPLEMENTED HERE

    def _get_classes_to_extract(self):
        """Get final list of unique cell IDs that should be extracted.
        Cell IDs for cells which are too close to the image edges to extract are discarded.
        """
        if self.partial_processing:
            self.log(
                f"Partial extraction mode enabled. Randomly sampling {self.n_cells} cells to extract with seed {self.seed}."
            )

            # randomly sample n_cells from the centers
            rng = np.random.default_rng(self.seed)
            chosen_ids = rng.choice(list(range(len(self.centers_cell_ids))), self.n_cells, replace=False)

            self.classes = self.centers_cell_ids[chosen_ids]
            self.px_centers = self.centers[chosen_ids]
        else:
            self.classes = self.centers_cell_ids
            self.px_centers = self.centers

        # ensure that none of the classes are located too close to the image edges to extract
        ids_to_discard = self._check_location_of_cells_to_extract()
        self._save_removed_classes(ids_to_discard)  # write discarded ids to file

        # update classes and centers to only contain the classes that should be extracted
        indexes_keep = [i for i, x in enumerate(self.classes) if x not in ids_to_discard]
        self.classes = self.classes[indexes_keep]
        self.px_centers = self.px_centers[indexes_keep]
        self.num_classes = len(self.classes)  # get total number of IDs that need to be processed

    def _check_location_of_cells_to_extract(self) -> list[int]:
        """Ensure that the cell_ids that are to be extracted are not too close to image edges

        Returns:
            list: List of cell IDs that are too close to the image border to be extracted
        """

        def _check_location(
            id, center: tuple[float, float], width: int, image_width: int, image_height: int
        ) -> int | None:
            x, y = center

            if x <= width:
                return id
            if y <= width:
                return id
            if x >= image_width - width:
                return id
            if y >= image_height - width:
                return id

            return None

        f = func_partial(
            _check_location,
            width=self.width_extraction,
            image_width=self.input_image_width,
            image_height=self.input_image_height,
        )
        results = {f(id, center) for id, center in zip(self.classes, self.px_centers, strict=True)}
        results.discard(None)

        return list(results)

    def _verbalise_extraction_info(self) -> None:
        """Print relevant information about the extraction process to the log file."""
        # print some output information
        self.log("Extraction Details:")
        self.log("--------------------------------")
        self.log(f"Number of input image channels: {self.n_image_channels}")
        self.log(f"Number of segmentation masks used during extraction: {self.n_masks}")
        self.log(f"Number of generated output images per cell: {self.n_output_channels}")
        self.log(f"Number of unique cells to extract: {self.num_classes}")
        self.log(f"Extracted Image Dimensions: {self.extracted_image_size} x {self.extracted_image_size}")
        self.log(f"Normalization of extracted images: {self.normalization}")
        self.log(f"Percentile normalization range for single-cell images: {self.normalization_range}")

    def _generate_save_index_lookup(self, class_list: list) -> None:
        """Create a lookup index indicating at which save_index each cell_id should be saved to in the HDF5 file."""
        self.save_index_lookup = pd.DataFrame(index=class_list)

    def _get_arg(self, cell_ids: list[int], centers: list[tuple[float, float]]) -> list:
        """Helper function for _generate_batched_args.

        The logic here is kept as it's own function, to allow for the simple changing fo this logic in alternative extraction workflow impelementations.
        Assumption is that provided cell_ids and centers are matched and sorted identically.

        Args:
            cell_ids: List of cell IDs to extract.
            centers: List of pixel centers for the cells to extract.

        Returns:
            list: List of arguments to be used for multiprocessing
        """
        args = list(
            zip(
                range(len(cell_ids)),
                [self.save_index_lookup.index.get_loc(x) for x in cell_ids],
                cell_ids,
                centers,
                strict=False,
            )
        )
        return args

    def _generate_batched_args(self, args: list, max_batch_size: int = 1000, min_batch_size: int = 100) -> list[list]:
        """
        Helper function to generate batched arguments for multiprocessing.
        Batched args are mini-batches of the original arguments that are used to split the processing into smaller chunks to prevent memory issues.
        The batch size for each mini batch can be configured witht he max_batch_size and min_batch_size parameters.
        The final batch size will be between the two values ensuring an equal distribution of arguments to desired number of processes.

        Args:
            args: List of arguments that need to be batched generated by self._get_arg
            max_batch_size: upper limit for the mini batch size
            min_batch_size: lower limit fr the mini batch size

        Returns:
            list: List of batched arguments.
        """

        if "max_batch_size" in self.config:
            max_batch_size = self.config["max_batch_size"]
        else:
            max_batch_size = max_batch_size

        original_threads = self.threads
        theoretical_max = np.ceil(len(args) / self.threads)
        batch_size = np.int64(min(max_batch_size, theoretical_max))

        self.batch_size = np.int64(max(min_batch_size, batch_size))
        self.log(f"Using batch size of {self.batch_size} for multiprocessing.")

        # dynamically adjust the number of threads to ensure that we dont initiate more threads than we have arguments
        self.threads = np.int64(min(self.threads, np.ceil(len(args) / self.batch_size)))

        if self.threads != original_threads:
            self.log(f"Reducing number of threads to {self.threads} to match number of cell batches to process.")

        return [args[i : i + self.batch_size] for i in range(0, len(args), self.batch_size)]

    def _get_label_info(self, arg: tuple) -> tuple:
        """adjust the label information for the extraction based on the segmentation masks used"""
        index, save_index, cell_id, px_center = arg

        # no additional labelling required
        return (index, save_index, cell_id, px_center, None, None)

    def _save_removed_classes(self, classes: np.ndarray) -> None:
        """Save cell IDs that are too close to the image border to be extracted as a csv file.

        Args:
            classes: List of cell IDs that are too close to the image border to be extracted.
        """

        # define path where classes should be saved
        filtered_path = os.path.join(self.extraction_data_directory, self.DEFAULT_REMOVED_CLASSES_FILE)

        to_write = "\n".join([str(i) for i in list(classes)])

        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        self.log(
            f"A total of {len(classes)} cells were too close to the image border to be extracted. Their cell_ids were saved to file {filtered_path}."
        )

    def _extract_classes(
        self, arg: tuple[int, int, int, tuple[float, float], int | None, str | None], return_results: bool = False
    ) -> None | tuple[int, np.ndarray, int]:
        """
        extract single-cell images for a given cell_id. If return_results is True, the results are returned, otherwise they are directly saved to the HDF5 file.
        If running in multi-threading mode, return_results should be set to False since concurrent access to the HDF5 file is not supported.

        Args:
            arg: a tuple containing all of the information needed to extract the specified single-cell. The tuple contains the index, the save_index, the cell_id, the pixel center, the image index, and the label_info.
            return_results (bool): Whether to return the results or save them to the HDF5 file.

        Returns:
            None: If return_results is False.
            tuple: If return_results is True. The tuple contains the save_index, the stack of images, and the cell_id.
        """
        index, save_index, cell_id, px_center, image_index, label_info = self._get_label_info(
            arg
        )  # label_info not used in base case but relevant for flexibility for other classes

        # currently this defaults to the same id for both
        # in theory scPortrait would be compatible with mapping nuclues ids to cytosol ids instead of updating the mask and then this code would need to change
        # this is a placeholder for now where this code can be implemented in the future but so that the rest of the pipeline already works with this use case
        nucleus_id = cell_id
        cytosol_id = cell_id

        ids = []
        if self.extract_nucleus_mask:
            nucleus_id = cell_id
            ids.append(nucleus_id)
        if self.extract_cytosol_mask:
            cytosol_id = cell_id
            ids.append(cytosol_id)

        # get region that should be extracted
        window_y = slice(px_center[1] - self.width_extraction, px_center[1] + self.width_extraction)
        window_x = slice(px_center[0] - self.width_extraction, px_center[0] + self.width_extraction)

        # ensure that the cell is not too close to the image edge to be extracted
        condition = [
            self.width_extraction < px_center[0],
            px_center[0] < self.input_image_width - self.width_extraction,
            self.width_extraction < px_center[1],
            px_center[1] < self.input_image_height - self.width_extraction,
        ]
        if not condition:
            raise ValueError("Cell is too close to the image edge to be extracted.")

        masks = []

        # get the segmentation masks
        for mask_ix in range(self.n_masks):
            if image_index is None:
                # nuclei_mask = self.sdata[self.nucleus_key].data[window_y, window_x].compute()
                mask = self.seg_masks[mask_ix, window_y, window_x]
            else:
                # nuclei_mask = self.sdata[self.nucleus_key].data[image_index, window_y, window_x].compute()
                mask = self.seg_masks[image_index, mask_ix, window_y, window_x]

            # modify nucleus mask to only contain the nucleus of interest and perform some morphological operations to generate a good image representation of the mask
            mask = np.where(mask == ids[mask_ix], 1, 0)
            mask = binary_fill_holes(mask)
            mask = gaussian(mask, preserve_range=True, sigma=1)

            if self.deep_debug:
                if mask.shape != (self.extracted_image_size, self.extracted_image_size):
                    print("Width of window_x", window_x.stop - window_x.start)
                    print("Width of window_y", window_y.stop - window_y.start)
                    print("Width of mask", mask.shape)
                    print("px_center", px_center)
                    print("cell_id", ids[mask_ix])
            assert mask.shape == (
                self.extracted_image_size,
                self.extracted_image_size,
            ), "Mask shape does not match extracted image size."
            masks.append(mask)

        # get the image data
        if image_index is None:
            image_data = self.image_data[:, window_y, window_x]
        else:
            image_data = self.image_data[image_index, :, window_y, window_x]

        image_data = (
            image_data * masks[-1]
        )  # always uses the last available mask, in nucleus only seg its the nucleus, if both its the cytosol, if only cytosol its also the cytosol. This always is the mask we want to use to extract the channel information

        # this needs to be performed on a per channel basis!
        images = []
        for i in range(image_data.shape[0]):
            ix = self.norm_function(image_data[i])
            images.append(ix)

        inputs = masks + images
        # masks and images have the same dtype here as both have been converted to images scaled between 0 and 1
        # ensuring the same dtype for all images is essential as otherwise they can not be saved into the same HDF5 container
        # the masks have also been slighly modified from the original binary masks to include a slight gaussian blur at the mask edges to smooth the masking of the image data
        stack = np.stack(inputs, axis=0).astype(self.DEFAULT_SINGLE_CELL_IMAGE_DTYPE)

        if self.deep_debug:
            # visualize some cells for debugging purposes
            if index % 1000 == 0:
                print(f"Cell ID: {cell_id} has center at [{px_center[0]}, {px_center[1]}]")

                fig, axs = plt.subplots(1, stack.shape[0], figsize=(2 * stack.shape[0], 2))
                for i, img in enumerate(stack):
                    axs[i].imshow(img, vmin=0, vmax=1)
                    axs[i].axis("off")
                fig.tight_layout()
                fig.show()

        if return_results:
            return save_index, stack, cell_id
        else:
            self._single_cell_data_container[save_index] = stack
            self._single_cell_index_container[save_index] = cell_id
            return None

    def _extract_classes_multi(
        self,
        arg_list: list[tuple[int, int, int, tuple[float, float], int | None, str | None]],
    ) -> list[tuple[int, np.ndarray, int]]:
        """Wrapper function to process all single-cells from a mini-batch.

        Args:
            arg_list: List of arguments to process.

        Returns:
            list: List of results from processing the mini-batch.
        """
        # set up normalization function
        self._setup_normalization()

        self.seg_masks: np.ndarray = mmap_array_from_path(self.path_seg_masks)
        self.image_data: np.ndarray = mmap_array_from_path(self.path_image_data)

        # get processing results
        results = [self._extract_classes(arg, return_results=True) for arg in arg_list]

        return results

    def _write_to_hdf5(self, results: list[tuple[int, np.ndarray, int]], hdf5_lock) -> None:
        """Function for writing results to HDF5 file in a thread-safe manner.

        Args:
            results: List of results to write to the HDF5 file.
            hdf5_lock: Lock object to ensure thread safety.
        """
        with hdf5_lock:
            with h5py.File(self.output_path, "a") as hf:
                single_cell_data_container: h5py.Dataset = hf[self.IMAGE_DATACONTAINER_NAME]
                single_cell_index_container: h5py.Dataset = hf[self.INDEX_DATACONTAINER_NAME]
                for save_index, stack, cell_id in results:
                    single_cell_data_container[save_index] = stack
                    single_cell_index_container[save_index] = cell_id

    def _initialize_empty_anndata(self) -> None:
        """Initialize an AnnData object to store the extracted single-cell images."""

        mask_names = np.array(self.masks, dtype="<U15")
        channel_names = np.array(self.channel_names, dtype="<U15")
        channels = np.concatenate([mask_names, channel_names])
        channel_mapping = ["mask" for x in mask_names] + ["image_channel" for x in channel_names]

        # create var object with channel names and their mapping to mask or image channels
        vars = pd.DataFrame(index=np.arange(len(channels)).astype("str"))
        vars["channels"] = channels
        vars["channel_mapping"] = channel_mapping

        # create empty obs object
        obs = pd.DataFrame(
            {self.DEFAULT_CELL_ID_NAME: np.zeros(shape=(self.num_classes), dtype=self.DEFAULT_SEGMENTATION_DTYPE)}
        )
        obs.index = obs.index.values.astype("str")

        # create anndata object
        adata = AnnData(obs=obs, var=vars)

        # add additional metadata to `uns`
        adata.uns[f"{self.DEFAULT_NAME_SINGLE_CELL_IMAGES}"] = {
            "n_cells": self.num_classes,
            "n_channels": self.n_masks + self.n_image_channels,
            "n_masks": self.n_masks,
            "n_image_channels": self.n_image_channels,
            "image_size": self.image_size,
            "normalization": self.normalization,
            "normalization_range_lower": self.normalization_range[0],
            "normalization_range_upper": self.normalization_range[1],
            "channel_names": channels,
            "channel_mapping": np.array(channel_mapping, dtype="<U15"),
            "compression": self.compression_type,
        }

        # write to file
        _normalize_anndata_strings(adata)
        adata.write(self.output_path)

    def _create_output_files(self) -> None:
        """Initialize the output HDF5 results file."""

        # define shapes for the output containers
        # single cell data: [n_cells, n_masks + n_image_channels, image_size, image_size]

        single_cell_data_shape = (
            self.num_classes,
            (self.n_masks + self.n_image_channels),
            self.image_size,
            self.image_size,
        )

        self.output_path = os.path.join(self.extraction_data_directory, self.DEFAULT_EXTRACTION_FILE)
        self._initialize_empty_anndata()

        # add an empty HDF5 dataset to the obsm group of the anndata object
        with h5py.File(self.output_path, "a") as hf:
            hf.create_dataset(
                self.IMAGE_DATACONTAINER_NAME,
                shape=single_cell_data_shape,
                chunks=(1, 1, self.image_size, self.image_size),
                compression=self.compression_type,
                dtype=self.DEFAULT_SINGLE_CELL_IMAGE_DTYPE,
            )

            # add required metadata from anndata package
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["encoding-type"] = "array"
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["encoding-version"] = "0.2.0"

            # add relevant metadata to the single-cell image container
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["n_cells"] = self.num_classes
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["n_channels"] = self.n_masks + self.n_image_channels
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["n_masks"] = self.n_masks
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["n_image_channels"] = self.n_image_channels
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["image_size"] = self.image_size
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["normalization"] = self.normalization
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["normalization_range"] = self.normalization_range
            masks = [x.encode("utf-8") for x in self.masks]
            channel_names = [x.encode("utf-8") for x in self.channel_names]
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["channel_names"] = np.array(masks + channel_names)
            mapping_values = ["mask" for x in masks] + ["image_channel" for x in channel_names]
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["channel_mapping"] = np.array(
                [x.encode("utf-8") for x in mapping_values]
            )
            hf[self.IMAGE_DATACONTAINER_NAME].attrs["compression"] = self.compression_type

            self.log("Container for single-cell data created.")

    def _post_extraction_cleanup(self, vars_to_delete=None):
        """remove temporary directories and files created during extraction. Reset attributes that are no longer required."""
        # remove normalization functions becuase other subsequent multiprocessing calls will fail
        if "norm_function" in self.__dict__:
            del self.norm_function

        # delete segmentation masks and input images from self if present
        if "seg_masks" in self.__dict__:
            del self.seg_masks
        if "image_data" in self.__dict__:
            del self.image_data

        if "_single_cell_data_container" in self.__dict__:
            del self._single_cell_data_container
        if "_single_cell_index_container" in self.__dict__:
            del self._single_cell_index_container

        # remove no longer required variables
        if vars_to_delete is not None:
            self._clear_cache(vars_to_delete=vars_to_delete)

        # remove memory mapped temp files
        if os.path.exists(self.path_seg_masks):
            os.remove(self.path_seg_masks)
        if os.path.exists(self.path_image_data):
            os.remove(self.path_image_data)

        # ensure that the save_index_to_remove is deleted to clear up memory and prevent issues with subsequent calls
        if "save_index_to_remove" in self.__dict__:
            self.save_index_to_remove = []  # reinitalize as empty

        self._clear_cache()

    def _save_benchmarking_times(
        self,
        total_time: float,
        time_setup: float,
        time_arg_generation: float,
        time_data_transfer: float,
        time_extraction: float,
        rate_extraction: float,
    ) -> None:
        """Write benchmarking times to csv file.

        Args:
            total_time: Total time taken for the extraction.
            time_setup: Time taken to set up the extraction.
            time_arg_generation: Time taken to generate arguments.
            time_data_transfer: Time taken to transfer data to memory mapped arrays.
            time_extraction: Time taken to extract single cell images.
            rate_extraction: Rate of extraction.

        """
        # save benchmarking times to file
        benchmarking_path = os.path.join(self.directory, self.DEFAULT_BENCHMARKING_FILE)

        benchmarking = pd.DataFrame(
            {
                "Size of image extracted from": [
                    (
                        self.n_image_channels,
                        self.input_image_width,
                        self.input_image_height,
                    )
                ],
                "Number of classes extracted": [self.num_classes],
                "Number of masks used for extraction": [self.n_masks],
                "Size of extracted images": [self.extracted_image_size],
                "Number of threads used": [self.threads],
                "Mini_batch size": [self.batch_size],
                "Total extraction time": [total_time],
                "Time taken to set up extraction": [time_setup],
                "Time taken to generate arguments": [time_arg_generation],
                "Time taken to transfer data to memory mapped arrays": [time_data_transfer],
                "Time taken to extract single cell images": [time_extraction],
                "Rate of extraction": [rate_extraction],
            }
        )

        if os.path.exists(benchmarking_path):
            # append to existing file
            benchmarking.to_csv(benchmarking_path, mode="a", header=False, index=False)
        else:
            # create new file
            benchmarking.to_csv(benchmarking_path, index=False)
        self.log("Benchmarking times saved to file.")

    def _get_current_process_rss_bytes(self) -> int:
        """Return RSS for the current process only."""
        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss)

    def _estimate_returned_result_bytes(self, results: list[tuple[int, np.ndarray, int]]) -> int:
        """Estimate memory footprint of a returned multiprocessing batch payload."""
        array_bytes = sum(int(stack.nbytes) for _, stack, _ in results)
        container_bytes = sys.getsizeof(results) + sum(sys.getsizeof(result) for result in results)
        return array_bytes + container_bytes

    def _get_target_job_ram_bytes(self) -> int:
        """Return the configured total RAM budget for this extraction job."""
        target_utilization = float(self.config.get("target_ram_utilization", 0.85))
        target_utilization = min(max(target_utilization, 0.1), 0.95)
        return int(psutil.virtual_memory().total * target_utilization)

    def _get_configured_max_inflight_result_batches(self, n_total_batches: int) -> int | None:
        """Return explicit in-flight override if present."""
        if "max_inflight_result_batches" not in self.config:
            return None

        configured = int(self.config["max_inflight_result_batches"])
        configured = max(1, min(configured, n_total_batches))
        self.log(f"Using configured max_inflight_result_batches={configured}.")
        return configured

    def _resolve_flush_every(self, default_flush_every: int) -> int:
        """Resolve flush cadence using config override or a supplied default.

        Args:
            default_flush_every: Fallback flush interval to use when the config
                does not explicitly set ``flush_every``.

        Returns:
            Flush interval in processed units.
        """
        if self.flush_every is not None:
            resolved = self.flush_every
            self.log(f"Using flush_every={resolved}.")
            return resolved

        resolved = max(1, int(default_flush_every))
        self.log(f"Using derived flush_every={resolved}.")
        return resolved

    def _periodic_flush_and_collect(self, counter: int, hf: h5py.File | None, flush_every: int) -> None:
        """Flush HDF5 output and run garbage collection at a configured interval.

        Args:
            counter: Number of processed units so far.
            hf: Open HDF5 file handle to flush if available.
            flush_every: Number of processed units between flush operations.
        """
        if counter % flush_every != 0:
            return

        if hf is not None:
            hf.flush()
        gc.collect()

    def _prepare_extraction_args(self) -> tuple[list, float]:
        """Generate extraction args and return generation time."""
        start_arg_generation = timeit.default_timer()
        self._generate_save_index_lookup(self.classes)
        args = self._get_arg(self.classes, self.px_centers)
        time_arg_generation = timeit.default_timer() - start_arg_generation
        return args, time_arg_generation

    def _load_inputs_to_memmap(self) -> float:
        """Load segmentation masks and input images into temporary memmaps."""
        self.log("Loading input images to memory mapped arrays...")
        start_data_transfer = timeit.default_timer()

        self.path_seg_masks = self.filehandler._load_seg_to_memmap(
            seg_name=self.masks, tmp_dir_abs_path=self._tmp_dir_path
        )
        self.path_image_data = self.filehandler._load_input_image_to_memmap(tmp_dir_abs_path=self._tmp_dir_path)

        time_data_transfer = timeit.default_timer() - start_data_transfer
        if self.debug:
            self.log(
                f"Finished transferring data to memory mapped arrays. Time taken: {time_data_transfer:.2f} seconds."
            )
        return time_data_transfer

    def _finish_processed_batch(
        self,
        processed_count: int,
        flush_every: int,
        pbar: tqdm | None = None,
        hf: h5py.File | None = None,
    ) -> int:
        """Apply common bookkeeping after a cell or batch has been written."""
        processed_count += 1
        self._periodic_flush_and_collect(counter=processed_count, hf=hf, flush_every=flush_every)
        if pbar is not None:
            pbar.update(1)
        return processed_count

    def _iter_completed_batch_results(
        self,
        pool: mp.pool.Pool,
        args: BatchedExtractionArgs,
        max_inflight_result_batches: int,
        pending_results: list | None = None,
        next_submit_ix: int = 0,
    ):
        """Yield completed multiprocessing batch results while bounding in-flight tasks.

        This helper exists to keep the extraction pipeline from submitting an
        unbounded number of batch jobs whose results then accumulate in memory
        faster than the main process can write them to disk. Each completed
        worker task returns a batch of extracted single-cell image stacks,
        which can be large. If too many completed batches are allowed to remain
        outstanding at once, resident memory can grow steadily until the job
        runs out of memory.

        To avoid that, this helper keeps at most
        ``max_inflight_result_batches`` tasks submitted but not yet consumed.
        As soon as one completed batch is yielded back to the caller, the
        helper submits the next pending batch task, maintaining a bounded
        producer/consumer pipeline. This preserves multiprocessing throughput
        while placing an upper bound on the amount of result data that can be
        buffered in flight at one time.

        Args:
            pool: Active multiprocessing pool used for extraction.
            args: Batched extraction arguments to submit.
            max_inflight_result_batches: Maximum number of submitted but not yet
                consumed batch tasks.

        Yields:
            Completed batch extraction results as returned by
            ``_extract_classes_multi``.
        """
        pending_results = [] if pending_results is None else pending_results
        n_total_batches = len(args)

        while (
            len(pending_results) < min(max_inflight_result_batches, n_total_batches)
            and next_submit_ix < n_total_batches
        ):
            pending_results.append(pool.apply_async(self._extract_classes_multi, (args[next_submit_ix],)))
            next_submit_ix += 1

        while pending_results:
            ready_ix = None
            for i, async_result in enumerate(pending_results):
                if async_result.ready():
                    ready_ix = i
                    break

            if ready_ix is None:
                time.sleep(0.01)
                continue

            async_result = pending_results.pop(ready_ix)
            result = async_result.get()

            while (
                len(pending_results) < min(max_inflight_result_batches, n_total_batches)
                and next_submit_ix < n_total_batches
            ):
                pending_results.append(pool.apply_async(self._extract_classes_multi, (args[next_submit_ix],)))
                next_submit_ix += 1

            yield result

    def _calibrate_max_inflight_result_batches(
        self,
        pool: mp.pool.Pool,
        args: BatchedExtractionArgs,
    ) -> tuple[int, float, list[tuple[int, np.ndarray, int]], list, int]:
        """Calibrate max in-flight batches from the first submitted worker wave.

        The calibration submits one initial wave of work and measures the size
        of the first batch payload returned to the parent process. Together
        with the parent-process RSS observed at that point, these measurements
        are used to estimate how many outstanding batch results can be buffered
        in the writer process while staying within the configured RAM budget
        for the job.

        Args:
            pool: Active multiprocessing pool used for extraction.
            args: Batched extraction arguments to submit.

        Returns:
            Tuple containing the calibrated max in-flight batch count, the time
            until the first batch completed, the first completed result itself,
            the remaining pending async results, and the next argument index to
            submit.
        """
        self.log("Calibrating max_inflight_result_batches using the first batch of workers...")
        n_total_batches = len(args)
        warmup_submit = min(max(1, int(self.threads)), n_total_batches)

        pending_results: list = []
        next_submit_ix = 0
        while len(pending_results) < warmup_submit and next_submit_ix < n_total_batches:
            pending_results.append(pool.apply_async(self._extract_classes_multi, (args[next_submit_ix],)))
            next_submit_ix += 1

        startup_wait_start = timeit.default_timer()
        while True:
            ready_ix = None
            for i, async_result in enumerate(pending_results):
                if async_result.ready():
                    ready_ix = i
                    break

            if ready_ix is None:
                time.sleep(0.01)
                continue

            async_result = pending_results.pop(ready_ix)
            first_result = async_result.get()
            first_wait_s = timeit.default_timer() - startup_wait_start
            break

        result_batch_bytes = self._estimate_returned_result_bytes(first_result)
        target_job_ram_bytes = self._get_target_job_ram_bytes()
        current_process_rss = self._get_current_process_rss_bytes()
        queue_budget_bytes = max(0, target_job_ram_bytes - current_process_rss)
        min_inflight_result_batches = max(1, warmup_submit)

        if result_batch_bytes <= 0:
            calibrated_from_budget = min_inflight_result_batches
        else:
            calibrated_from_budget = max(1, min(int(queue_budget_bytes // result_batch_bytes), n_total_batches))

        if calibrated_from_budget < min_inflight_result_batches:
            self.log(
                "Warning: target RAM budget would limit max_inflight_result_batches to "
                f"{calibrated_from_budget}, below the active worker floor of {min_inflight_result_batches}. "
                "Using the worker-count floor instead. If stricter memory limiting is required, reduce `threads`."
            )

        calibrated_max = max(min_inflight_result_batches, calibrated_from_budget)

        bytes_to_gb = 1024**3
        self.log(
            "Calibrated max_inflight_result_batches="
            f"{calibrated_max} (budget_based_n={calibrated_from_budget}, "
            f"worker_floor_n={min_inflight_result_batches}, "
            f"target_job_ram_gb={target_job_ram_bytes / bytes_to_gb:.3f}, "
            f"parent_rss_gb={current_process_rss / bytes_to_gb:.3f}, "
            f"result_batch_gb={result_batch_bytes / bytes_to_gb:.3f}, "
            f"queue_budget_gb={queue_budget_bytes / bytes_to_gb:.3f})."
        )
        return calibrated_max, first_wait_s, first_result, pending_results, next_submit_ix

    def process(
        self, partial: bool = False, n_cells: int = None, seed: int = 42, output_folder_name: str | None = None
    ) -> None:
        """
        Extracts single cell images from a segmented scPortrait project and saves the results to a standardized HDF5 file.

        Args:
            input_segmentation_path : Path of the segmentation HDF5 file. If this class is used as part of a project processing workflow, this argument will be provided automatically.
            partial: if set to True only a random subset of n_cells will be extracted.
            n_cells: Number of cells to extract if partial is set to True.
            seed: Seed for random sampling of cells for reproducibility if partial is set to True.

        Important
        ---------
        If this class is used as part of a project processing workflow, all of the arguments will be provided by the ``Project`` class based on the previous segmentation.
        The Project class will automatically provide the most recent segmentation forward together with the supplied parameters.

        Examples
        --------
        .. code-block:: python

            # After project is initialized and input data has been loaded and segmented
            project.extract()

        Notes
        -----
        The following parameters are required in the config file when running this method:

        .. code-block:: yaml

            HDF5CellExtraction:
                # threads used in multithreading
                threads: 80

                # image size in pixels
                image_size: 128

                # directory where intermediate results should be saved
                cache: "/mnt/temp/cache"

        The following optional parameters can also be configured:

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Default
              - Description
            * - ``normalize_output``
              - ``True``
              - Enable percentile normalization of extracted image channels.
            * - ``normalization_range``
              - ``(0.001, 0.999)``
              - Lower and upper percentiles used for normalization.
            * - ``compression``
              - ``True``
              - Compression mode for the output HDF5 dataset. ``True`` maps to ``lzf``. ``gzip`` and ``False`` are also supported.
            * - ``target_ram_utilization``
              - ``0.85``
              - Fraction of total system RAM the extraction job should aim to stay within when calibrating buffered result batches.
            * - ``max_inflight_result_batches``
              - auto-calibrated
              - Explicit override for the number of buffered multiprocessing result batches. If omitted, scPortrait calibrates this from the first worker wave.
            * - ``flush_every``
              - derived from effective in-flight batch limit
              - Flush cadence for HDF5 output and garbage collection during extraction. If omitted, it is derived from the effective in-flight batch limit.
            * - ``max_batch_size``
              - ``1000``
              - Upper bound used when building multiprocessing mini-batches.

        Normalization settings deserve special attention because they directly
        affect the dynamic range of the extracted single-cell images that are
        stored for downstream analysis. If you are unsure how to choose
        ``normalize_output`` or ``normalization_range``, refer to the
        :ref:`single-cell extraction tutorial <single_cell_extraction_tutorial>`
        for a more detailed walkthrough.

        During extraction, scPortrait can use multiple worker processes to
        prepare single-cell image batches while the main process writes results
        to the output HDF5 file. On large datasets with multiple threads, preparing
        batches can be faster than writing them to disk, which would otherwise
        allow completed batch results to accumulate in memory. To keep this manageable,
        the extraction workflow can automatically limit how many completed batch
        results are allowed to be buffered in memory at the same time.

        In multiprocessing mode, ``max_inflight_result_batches`` is calibrated
        automatically when it is not provided explicitly. The calibration uses
        the first wave of worker batches to estimate returned batch payload
        size together with the parent-process RSS, then chooses an in-flight
        batch limit that aims to respect ``target_ram_utilization``. If the
        calculated value would fall below the active worker count, the worker
        count is used as a minimum and a warning is written to the log.

        """
        total_time_start = timeit.default_timer()

        start_setup = timeit.default_timer()

        # set up flag for partial processing
        self.partial_processing = partial
        if self.partial_processing:
            self.n_cells = n_cells
            self.seed = seed
            self.DEFAULT_LOG_NAME = "partial_processing.log"  # change log name so that the results are not written to the same log file as a complete extraction

        # run all of the extraction setup steps
        check = self._set_up_extraction(output_folder_name=output_folder_name)

        if not check:
            return None

        stop_setup = timeit.default_timer()
        time_setup = stop_setup - start_setup

        if self.partial_processing:
            self.log(f"Starting partial single-cell image extraction of {self.n_cells} cells...")
        else:
            self.log(f"Starting single-cell image extraction of {self.num_classes} cells...")

        args, time_arg_generation = self._prepare_extraction_args()
        time_data_transfer = self._load_inputs_to_memmap()

        # actually perform single-cell image extraction
        start_extraction = timeit.default_timer()
        if self.threads <= 1:
            # set up for single-threaded processing
            self._setup_normalization()
            single_thread_flush_default = max(1, int(self.config.get("max_batch_size", 1000)))
            flush_every = self._resolve_flush_every(single_thread_flush_default)

            self.seg_masks = mmap_array_from_path(self.path_seg_masks)
            self.image_data = mmap_array_from_path(self.path_image_data)

            with h5py.File(
                self.output_path,
                "a",
            ) as hf:
                # connect to final containers for saving computed results
                self._single_cell_data_container = hf[self.IMAGE_DATACONTAINER_NAME]
                self._single_cell_index_container = hf[self.INDEX_DATACONTAINER_NAME]

                self.log("Running in single threaded mode.")
                processed_cells = 0
                for arg in tqdm(args, total=len(args), desc="Extracting cell batches"):
                    self._extract_classes(arg)
                    processed_cells = self._finish_processed_batch(
                        processed_count=processed_cells,
                        flush_every=flush_every,
                        hf=hf,
                    )
        else:
            args = self._generate_batched_args(args)
            n_total_batches = len(args)

            self.log(f"Running in multiprocessing mode with {self.threads} threads.")

            with mp.Manager() as manager:
                lock = manager.Lock()  # Create lock via Manager to enable sharing

                self.log("Initializing multiprocessing pool for extraction.")
                with mp.get_context("fork").Pool(
                    processes=self.threads
                ) as pool:  # both spawn and fork work but fork is faster so forcing fork here
                    if n_total_batches > 0:
                        self.log("Workers initialized. Waiting for first completed extraction batch...")
                        configured_max_inflight = self._get_configured_max_inflight_result_batches(n_total_batches)
                        if configured_max_inflight is not None:
                            startup_wait_start = timeit.default_timer()
                            result_iterator = self._iter_completed_batch_results(
                                pool=pool,
                                args=args,
                                max_inflight_result_batches=configured_max_inflight,
                            )
                            first_result = next(result_iterator)
                            first_wait_s = timeit.default_timer() - startup_wait_start
                            max_inflight_result_batches = configured_max_inflight
                        else:
                            (
                                max_inflight_result_batches,
                                first_wait_s,
                                first_result,
                                pending_results,
                                next_submit_ix,
                            ) = self._calibrate_max_inflight_result_batches(pool=pool, args=args)
                            result_iterator = self._iter_completed_batch_results(
                                pool=pool,
                                args=args,
                                max_inflight_result_batches=max_inflight_result_batches,
                                pending_results=pending_results,
                                next_submit_ix=next_submit_ix,
                            )

                        flush_every = self._resolve_flush_every(max_inflight_result_batches)
                        self.log(f"Limiting in-flight result batches to {max_inflight_result_batches}.")
                        self.log(f"First batch received after {first_wait_s:.2f} seconds.")
                        processed_batches = 0

                        with tqdm(total=n_total_batches, desc="Extracting cell batches") as pbar:
                            self._write_to_hdf5(first_result, lock)
                            processed_batches = self._finish_processed_batch(
                                processed_count=processed_batches,
                                flush_every=flush_every,
                                pbar=pbar,
                            )

                            for result in result_iterator:
                                self._write_to_hdf5(result, lock)
                                processed_batches = self._finish_processed_batch(
                                    processed_count=processed_batches,
                                    flush_every=flush_every,
                                    pbar=pbar,
                                )
                    pool.close()
                    pool.join()

        # cleanup memory and remove any no longer required variables
        del args
        stop_extraction = timeit.default_timer()

        # calculate duration
        time_extraction = stop_extraction - start_extraction
        rate = self.num_classes / time_extraction

        # generate final log entries
        self.log(f"Finished extraction in {time_extraction:.2f} seconds ({rate:.2f} cells / second)")

        if self.partial_processing:
            self.DEFAULT_LOG_NAME = "processing.log"  # change log name back to default

        self._post_extraction_cleanup()
        total_time_stop = timeit.default_timer()
        total_time = total_time_stop - total_time_start

        self._save_benchmarking_times(
            total_time=total_time,
            time_setup=time_setup,
            time_arg_generation=time_arg_generation,
            time_data_transfer=time_data_transfer,
            time_extraction=time_extraction,
            rate_extraction=rate,
        )
