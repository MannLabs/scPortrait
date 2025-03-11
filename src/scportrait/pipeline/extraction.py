import multiprocessing as mp
import os
import platform
import shutil
import sys
import timeit
from functools import partial as func_partial
from pathlib import PosixPath

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        if "normalize_output" in self.config:
            normalize = self.config["normalize_output"]

            # check that is a valid value
            assert normalize in [
                True,
                False,
                None,
                "None",
            ], "Normalization must be one of the following values [True, False, None, 'None']"

            # convert to boolean
            if normalize == "None":
                normalize = False
            if normalize is None:
                normalize = False

            self.normalization = normalize

        else:
            self.normalization = True  # default value

        if "normalization_range" in self.config:
            normalization_range = self.config["normalization_range"]

            if normalization_range == "None":
                normalization_range = None

            if normalization_range is not None:
                assert len(normalization_range) == 2, "Normalization range must be a tuple or list of length 2."
                assert all(
                    isinstance(x, float | int) and (0 <= x <= 1) for x in normalization_range
                ), "Normalization range must be defined as a float between 0 and 1."

                # conver to tuple to ensure consistency
                if isinstance(normalization_range, list):
                    normalization_range = tuple(normalization_range)
            else:
                normalization_range = (0.01, 0.99)

            self.normalization_range = normalization_range

        else:
            self.normalization_range = (0.01, 0.99)

        if not self.normalization:
            self.normalization_range = None

        ## parameters for HDF5 file creates
        if "compression" in self.config:
            self.compression = self.config["compression"]
        else:
            self.compression = True

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
        else:
            if self.overwrite_run_path:
                self.log(f"Output folder at {self.extraction_data_directory} already exists. Overwriting...")
                shutil.rmtree(self.extraction_data_directory)
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

    def _set_up_extraction(self) -> None:
        """execute all helper functions to setup extraction process"""
        if self.partial_processing:
            output_folder_name = f"partial_{self.DEFAULT_DATA_DIR}_ncells_{self.n_cells}_seed_{self.seed}"
        else:
            output_folder_name = self.DEFAULT_DATA_DIR

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

        # create output files for saving results to
        self._create_output_files()

        # print relevant information to log file
        self._verbalise_extraction_info()

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
            allowed_mask_values = ["nucleus", "cytosol"]
            allowed_mask_values = [f"{self.segmentation_key}_{x}" for x in allowed_mask_values]

            if isinstance(self.config["segmentation_mask"], str):
                assert self.config["segmentation_mask"] in allowed_mask_values

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

            self.main_segmenation_mask = self.nucleus_key

        elif self.n_masks == 1:
            if self.extract_nucleus_mask:
                self.main_segmenation_mask = self.nucleus_key
            elif self.extract_cytosol_mask:
                self.main_segmenation_mask = self.cytosol_key

        self.log(
            f"Found {self.n_masks} segmentation masks for the given key in the sdata object. Will be extracting single-cell images based on these masks: {self.masks}"
        )
        self.log(f"Using {self.main_segmenation_mask} as the main segmentation mask to determine cell centers.")

    def _get_input_image_info(self) -> None:
        """get relevant information about the input image to be able to extract single-cell images"""
        # get channel information
        self.channel_names = self.project.input_image.c.values
        self.n_image_channels = len(self.channel_names)
        self.input_image_width = len(self.project.input_image.x)
        self.input_image_height = len(self.project.input_image.y)

    def _get_centers(self) -> None:
        """get the centers of the cells that should be extracted.
        If a nucleus and cytosol mask are used, the centers are calculated based on the nucleus mask.
        If only one mask is used, the centers are calculated based on that mask.
        """
        _sdata = self.filehandler._read_sdata()

        # calculate centers if they have not been calculated yet
        centers_name = f"{self.DEFAULT_CENTERS_NAME}_{self.main_segmenation_mask}"
        if centers_name not in _sdata:
            self.filehandler._add_centers(self.main_segmenation_mask, overwrite=self.overwrite)
            _sdata = self.filehandler._read_sdata()  # reread to ensure we have updated version

        centers = _sdata[centers_name].values.compute()

        # round to int so that we can use them as indices
        centers = np.round(centers).astype(int)

        self.centers = centers
        self.centers_cell_ids = _sdata[centers_name].index.values.compute()

        # ensure that the centers ids are unique
        assert len(self.centers_cell_ids) == len(
            set(self.centers_cell_ids)
        ), "Cell ids in centers are not unique. Cannot proceed with extraction."

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

        theoretical_max = np.ceil(len(args) / self.threads)
        batch_size = np.int64(min(max_batch_size, theoretical_max))

        self.batch_size = np.int64(max(min_batch_size, batch_size))
        self.log(f"Using batch size of {self.batch_size} for multiprocessing.")

        # dynamically adjust the number of threads to ensure that we dont initiate more threads than we have arguments
        self.threads = np.int64(min(self.threads, np.ceil(len(args) / self.batch_size)))

        if self.threads != self.threads:
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
        filtered_path = os.path.join(
            self.project_location,
            self.DEFAULT_EXTRACTION_DIR_NAME,
            self.DEFAULT_REMOVED_CLASSES_FILE,
        )

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
            msg = "Cell is too close to the image edge to be extracted."
            raise ValueError(msg)  # or a custom error

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
            self._single_cell_index_container[save_index] = [save_index, cell_id]
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
                self._single_cell_data_container: h5py.Dataset = hf[self.IMAGE_DATACONTAINTER_NAME]
                self._single_cell_index_container: h5py.Dataset = hf[self.INDEX_DATACONTAINER_NAME]

                for res in results:
                    save_index, stack, cell_id = res
                    self._single_cell_data_container[save_index] = stack
                    self._single_cell_index_container[save_index] = str(cell_id)

    def _initialize_empty_anndata(self) -> None:
        """Initialize an AnnData object to store the extracted single-cell images."""
        # create var object with channel names
        mask_names = self.masks
        channel_names = self.channel_names
        vars = pd.DataFrame(columns=list(mask_names) + list(channel_names))

        # create empty obs object
        obs = pd.DataFrame(index=range(self.num_classes))

        adata = AnnData(obs=obs, var=vars)
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
                self.IMAGE_DATACONTAINTER_NAME,
                shape=single_cell_data_shape,
                chunks=(1, 1, self.image_size, self.image_size),
                compression=self.compression_type,
                dtype=self.DEFAULT_SINGLE_CELL_IMAGE_DTYPE,
            )

            # add relevant attrs to the single-cell image container
            hf[self.IMAGE_DATACONTAINTER_NAME].attrs["n_masks"] = self.n_masks
            hf[self.IMAGE_DATACONTAINTER_NAME].attrs["image_size"] = self.image_size
            hf[self.IMAGE_DATACONTAINTER_NAME].attrs["normalization"] = self.normalization
            hf[self.IMAGE_DATACONTAINTER_NAME].attrs["normalization_range"] = self.normalization_range
            hf[self.IMAGE_DATACONTAINTER_NAME].attrs["channel_names"] = np.array(
                [x.encode("utf-8") for x in self.channel_names]
            )
            hf[self.IMAGE_DATACONTAINTER_NAME].attrs["mask_names"] = np.array([x.encode("utf-8") for x in self.masks])
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

    def process(self, partial: bool = False, n_cells: int = None, seed: int = 42) -> None:
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
        self._set_up_extraction()
        stop_setup = timeit.default_timer()
        time_setup = stop_setup - start_setup

        if self.partial_processing:
            self.log(f"Starting partial single-cell image extraction of {self.n_cells} cells...")
        else:
            self.log(f"Starting single-cell image extraction of {self.num_classes} cells...")

        # generate cell pairings to extract
        start_arg_generation = timeit.default_timer()

        self._generate_save_index_lookup(self.classes)
        args = self._get_arg(self.classes, self.px_centers)
        stop_arg_generation = timeit.default_timer()
        time_arg_generation = stop_arg_generation - start_arg_generation

        # convert input images to memory mapped temp arrays for faster reading
        self.log("Loading input images to memory mapped arrays...")
        start_data_transfer = timeit.default_timer()

        self.path_seg_masks = self.filehandler._load_seg_to_memmap(
            seg_name=self.masks, tmp_dir_abs_path=self._tmp_dir_path
        )
        self.path_image_data = self.filehandler._load_input_image_to_memmap(tmp_dir_abs_path=self._tmp_dir_path)

        stop_data_transfer = timeit.default_timer()
        time_data_transfer = stop_data_transfer - start_data_transfer

        if self.debug:
            self.log(
                f"Finished transferring data to memory mapped arrays. Time taken: {time_data_transfer:.2f} seconds."
            )

        # actually perform single-cell image extraction
        start_extraction = timeit.default_timer()

        if self.threads <= 1:
            # set up for single-threaded processing
            self._setup_normalization()

            self.seg_masks = mmap_array_from_path(self.path_seg_masks)
            self.image_data = mmap_array_from_path(self.path_image_data)

            with h5py.File(
                self.output_path,
                "a",
            ) as hf:
                # connect to final containers for saving computed results
                self._single_cell_data_container = hf[self.IMAGE_DATACONTAINTER_NAME]
                self._single_cell_index_container = hf[self.INDEX_DATACONTAINER_NAME]

                self.log("Running in single threaded mode.")
                for arg in tqdm(args, total=len(args), desc="Extracting cell batches"):
                    self._extract_classes(arg)
        else:
            args = self._generate_batched_args(args)

            self.log(f"Running in multiprocessing mode with {self.threads} threads.")

            with mp.Manager() as manager:
                lock = manager.Lock()  # Create lock via Manager to enable sharing

                with mp.get_context("fork").Pool(
                    processes=self.threads
                ) as pool:  # both spawn and fork work but fork is faster so forcing fork here
                    for result in list(
                        tqdm(
                            pool.imap(self._extract_classes_multi, args),
                            total=len(args),
                            desc="Extracting cell batches",
                        )
                    ):
                        self._write_to_hdf5(result, lock)
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
