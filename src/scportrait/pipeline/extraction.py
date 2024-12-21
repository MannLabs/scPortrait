import os
import numpy as np
import pandas as pd
import csv
import h5py
from tqdm.auto import tqdm
from itertools import compress
import timeit
import _pickle as cPickle
import matplotlib.pyplot as plt

from functools import partial as func_partial
import multiprocessing as mp
import platform

from alphabase.io.tempmmap import create_empty_mmap, mmap_array_from_path

from skimage.filters import gaussian
from scipy.ndimage import binary_fill_holes

from scportrait.processing.segmentation import numba_mask_centroid
from scportrait.processing.utils import flatten
from scportrait.processing.preprocessing import percentile_normalization
from scportrait.pipeline.base import ProcessingStep


# to perform garbage collection


class HDF5CellExtraction(ProcessingStep):
    """
    A class to extracts single cell images from a segmented SPARCSpy project and save the
    results to an HDF5 file.
    """

    SELECTED_DATA_DIR = "selected_data"
    CLEAN_LOG = False

    # new parameters to make workflow adaptable to other types of projects
    channel_label = "channels"
    segmentation_label = "labels"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        base_directory = self.directory.replace("extraction", "")

        self.input_segmentation_path = os.path.join(
            base_directory,
            self.DEFAULT_SEGMENTATION_DIR_NAME,
            self.DEFAULT_SEGMENTATION_FILE,
        )

        # get path to filtered classes
        if os.path.isfile(
            os.path.join(
                base_directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                "needs_additional_filtering.txt",
            )
        ):
            try:
                self.classes_path = os.path.join(
                    base_directory,
                    self.DEFAULT_SEGMENTATION_FILTERING_DIR_NAME,
                    self.DEFAULT_FILTERED_CLASSES_FILE,
                )

                self.log(
                    f"Loading classes from filtered classes path: {self.classes_path}"
                )
            except Exception:
                raise ValueError("Need to run segmentation_filtering method ")
        else:
            self.classes_path = os.path.join(
                base_directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                self.DEFAULT_CLASSES_FILE,
            )
            self.log(f"Loading classes from default classes path: {self.classes_path}")

        # extract required information for generating datasets
        self._get_compression_type()

        self.save_index_to_remove = []
        self.batch_size = None

        # set developer debug mode for super detailed output
        self.deep_debug = False

        #initialize default values
        self.extraction_data_directory = None

        # check for windows operating system and if so set threads to 1
        if platform.system() == "Windows":
            Warning(
                "Windows detected. Multithreading not supported on windows so setting threads to 1."
            )
            self.config["threads"] = 1

    def _get_compression_type(self):
        self.compression_type = "lzf" if self.config["compression"] else None
        return self.compression_type

    def _parse_remapping(self):
        self.remap = None
        if "channel_remap" in self.config:
            char_list = self.config["channel_remap"].split(",")
            self.log("channel remap parameter found:")
            self.log(char_list)

            self.remap = [int(el.strip()) for el in char_list]

    def _get_normalization(self):
        # get normalization parameters
        if "normalize_output" in self.config:
            normalize = self.config["normalize_output"]

            # check that is a valid value
            assert (
                normalize in [True, False, None, "None"]
            ), "Normalization must be one of the following values [True, False, None, 'None']"

            # convert to boolean
            if normalize == "None":
                normalize = False
            if normalize is None:
                normalize = False

            self.normalization = normalize
        else:
            self.normalization = True  # default value

        # setup normalization range
        if "normalization_range" in self.config:
            normalization_range = self.config["normalization_range"]

            if normalization_range == "None":
                normalization_range = None

            if normalization_range is not None:
                assert isinstance(
                    normalization_range, tuple
                ), "Normalization range must be a tuple."

            self.normalization_range = normalization_range

        else:
            self.normalization_range = None

        # get functions for normalization
        if self.normalization:
            if self.normalization_range is None:

                def norm_function(
                    img: np.array, lower: float = None, upper: float = None
                ) -> np.array:
                    return percentile_normalization(img)
            else:
                lower, upper = self.normalization_range

                def norm_function(
                    img: np.array, lower: float = lower, upper: float = upper
                ) -> np.array:
                    return percentile_normalization(img, lower, upper)

        elif not self.normalization:

            def norm_function(
                img: np.array, lower: float = None, upper: float = None
            ) -> np.array:
                img = (
                    img / np.iinfo(self.DEFAULT_IMAGE_DTYPE).max
                )  # convert 16bit unsigned integer image to float between 0 and 1 without adjusting for the pixel values we have in the extracted single cell image
                return img

        self.norm_function = norm_function

    def get_channel_info(self):
        with h5py.File(self.input_segmentation_path, "r") as hf:
            hdf_channels = hf.get(self.channel_label)
            hdf_labels = hf.get(self.segmentation_label)

            if len(hdf_channels.shape) == 3:
                self.n_channels_input = hdf_channels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_channels_input = hdf_channels.shape[1]

            self.input_image_width = hdf_channels.shape[-2]
            self.input_image_height = hdf_channels.shape[-1]

            self.log(f"Using channel label {hdf_channels}")
            self.log(f"Using segmentation label {hdf_labels}")

            if len(hdf_labels.shape) == 3:
                self.n_segmentation_channels = hdf_labels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_segmentation_channels = hdf_labels.shape[1]

            self.n_channels_output = (
                self.n_segmentation_channels + self.n_channels_input
            )

    def _get_output_path(self):
        if self.extraction_data_directory is None:
            self._setup_output()
        return self.extraction_data_directory

    def _setup_output(self, folder_name=None):
        if folder_name is None:
            folder_name = self.DEFAULT_DATA_DIR

        self.extraction_data_directory = os.path.join(self.directory, folder_name)

        if not os.path.isdir(self.extraction_data_directory):
            os.makedirs(self.extraction_data_directory)
            self.log("Created new data directory " + self.extraction_data_directory)

        self.log(f"Setup output folder at {self.extraction_data_directory}")

    def _initialize_tempmmap_array(self, index_len=2):
        single_cell_index_shape = (self.num_classes, index_len)
        single_cell_data_shape = (
            self.num_classes,
            self.n_output_channels,
            self.config["image_size"],
            self.config["image_size"],
        )

        # generate container for single_cell_data
        self._tmp_single_cell_data_path = create_empty_mmap(
            shape=single_cell_data_shape,
            dtype=np.float16,
            tmp_dir_abs_path=self._tmp_dir_path,
        )

        # # generate container for single_cell_index
        fixed_length = 200  # this is the maximum length of string that can be stored in the mmap array for the cell_ids
        dt = np.dtype(f"S{fixed_length}")
        self._tmp_single_cell_index_path = create_empty_mmap(
            shape=single_cell_index_shape,
            dtype=dt,
            tmp_dir_abs_path=self._tmp_dir_path,
        )

    def _setup_extraction(self):
        if self.partial_processing:
            output_folder_name = (
                f"partial_{self.DEFAULT_DATA_DIR}_{self.n_cells}_{self.seed}"
            )
        else:
            output_folder_name = self.DEFAULT_DATA_DIR

        self._setup_output(folder_name=output_folder_name)

        self._parse_remapping()
        self._get_segmentation_info()
        self._get_input_image_info()

        # setup number of output channels
        self.n_output_channels = self.n_image_channels + self.n_masks

        # get size of images to extract
        self.extracted_image_size = self.config["image_size"]
        self.width_extraction = (
            self.extracted_image_size // 2
        )  # half of the extracted image size (this is what needs to be added on either side of the center)

        self._get_classes()
        self._get_centers()
        self._get_classes_to_extract()

        # initialize temporary mmap arrays for saving results
        self._initialize_tempmmap_array()

        self._verbalise_extraction_info()

    def _get_segmentation_info(self):
        with h5py.File(self.input_segmentation_path, "r") as hf:
            hdf_labels = hf.get(self.segmentation_label)
            self.log(f"Using segmentation {hdf_labels}")

            if len(hdf_labels.shape) == 3:
                self.n_masks = hdf_labels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_masks = hdf_labels.shape[1]

    def _get_input_image_info(self) -> None:
        with h5py.File(self.input_segmentation_path, "r") as hf:
            hdf_channels = hf.get(self.channel_label)
            self.log(f"Using channel information {hdf_channels}")

            if len(hdf_channels.shape) == 3:
                self.n_image_channels = hdf_channels.shape[0]
            elif len(hdf_channels.shape) == 4:
                self.n_image_channels = hdf_channels.shape[1]

            self.channel_names = np.array(
                [f"channel_{i}" for i in range(self.n_image_channels)]
            )  # placeholder for channel names as they are currently not properly tracked
            self.input_image_width = hdf_channels.shape[-2]
            self.input_image_height = hdf_channels.shape[-1]

    def _get_centers(self) -> None:
        # define locations to look for center and cell_ids files
        center_path = os.path.join(self.directory, "center.pickle")
        cell_ids_path = os.path.join(self.directory, "_cell_ids.pickle")

        # check to see if file has already been calculated if so load it
        if (
            os.path.isfile(center_path)
            and os.path.isfile(cell_ids_path)
            and not self.overwrite
        ):
            self.log(
                "Cached version of calculated cell centers found, loading instead of recalculating."
            )
            with open(center_path, "rb") as input_file:
                center_nuclei = cPickle.load(input_file)
                px_centers = np.round(center_nuclei).astype(int)
            with open(cell_ids_path, "rb") as input_file:
                _cell_ids = cPickle.load(input_file)

            # delete variables to free up memory
            del center_nuclei

        # perform calculation and save results to file
        else:
            self.log("Started cell coordinate calculation")
            with h5py.File(self.input_segmentation_path, "r") as hf:
                hdf_labels = hf.get(self.segmentation_label)
                center_nuclei, length, _cell_ids = numba_mask_centroid(
                    hdf_labels[0].astype(self.DEFAULT_SEGMENTATION_DTYPE),
                    debug=self.debug,
                )

            px_centers = np.round(center_nuclei).astype(int)

            self.log("Finished cell coordinate calculation")

            with open(center_path, "wb") as output_file:
                cPickle.dump(center_nuclei, output_file)
            with open(cell_ids_path, "wb") as output_file:
                cPickle.dump(_cell_ids, output_file)
            with open(
                os.path.join(self.directory, "length.pickle"), "wb"
            ) as output_file:
                cPickle.dump(length, output_file)

            # delete variables to free up memory
            del length, center_nuclei

            self.log(
                f"Cell coordinates saved to file {center_path} and cell Ids saved to file {cell_ids_path}."
            )

        # save for later use
        self.centers = px_centers
        self.centers_cell_ids = _cell_ids

    def _get_classes(self):
        if self.filtered_classes_path is not None:
            self.log(
                f"Loading classes from provided filtered classes path: {self.filtered_classes_path}"
            )
            path = self.filtered_classes_path
        else:
            path = self.classes_path

        cr = csv.reader(
            open(path, "r"),
        )

        if "filtered" in path:
            filtered_classes = [
                el[0] for el in list(cr)
            ]  # do not do int transform here as we expect a str of format "nucleus_id:cytosol_id"
            filtered_classes = list(np.unique(filtered_classes))
        else:
            filtered_classes = [int(float(el[0])) for el in list(cr)]
            filtered_classes = list(
                np.unique(filtered_classes).astype(np.uint64)
            )  # make sure they are all unique
            if 0 in filtered_classes:
                filtered_classes.remove(0)  # remove background if still listed

        self.log(f"Loaded {len(filtered_classes)} cellIds to extract.")
        self.log(f"After removing duplicates {len(filtered_classes)} cells remain.")

        self.num_classes = len(filtered_classes)
        self.classes_loaded = filtered_classes

    def _get_classes_to_extract(self) -> None:
        if isinstance(self.classes_loaded[0], str):
            lookup_dict = {
                x.split(":")[0]: x.split(":")[1] for x in self.classes_loaded
            }
            nuclei_ids = set(list(lookup_dict.keys()))
        else:
            nuclei_ids = set([str(x) for x in self.classes_loaded])

        # filter cell ids found using center into those that we actually want to extract
        _cell_ids = list(self.centers_cell_ids)
        filter = [str(x) in nuclei_ids for x in _cell_ids]

        px_centers = np.array(list(compress(self.centers, filter)))
        _cell_ids = list(compress(_cell_ids, filter))

        # generate new class list
        if isinstance(self.classes_loaded[0], str):
            class_list = [f"{x}:{lookup_dict[str(x)]}" for x in _cell_ids]
            del lookup_dict
        else:
            class_list = _cell_ids

        self.classes_loaded = class_list
        self.centers = px_centers

        if self.partial_processing:
            self.log(
                "Partial extraction mode enabled. Randomly sampling {self.n_cells} cells to extract with seed {self.seed}."
            )

            # randomly sample n_cells from the centers
            np.random.seed(self.seed)
            chosen_ids = np.random.choice(
                list(range(len(self.classes_loaded))), self.n_cells, replace=False
            )
            chosen_ids.sort()
            self.classes = self.classes_loaded[chosen_ids]
            self.px_centers = self.centers[chosen_ids]
        else:
            self.classes = self.classes_loaded
            self.px_centers = self.centers

        # get number of classes that need to be extracted
        self.num_classes = len(self.classes)

    def _verbalise_extraction_info(self):
        # print some output information
        self.log("Extraction Details:")
        self.log("--------------------------------")
        self.log(f"Number of input image channels: {self.n_image_channels}")
        self.log(f"Number of segmentation masks used during extraction: {self.n_masks}")
        self.log(
            f"Number of generated output images per cell: {self.n_output_channels}"
        )
        self.log(f"Number of unique cells to extract: {self.num_classes}")
        self.log(
            f"Extracted Image Dimensions: {self.extracted_image_size} x {self.extracted_image_size}"
        )

    def _generate_save_index_lookup(self, class_list):
        self.save_index_lookup = pd.DataFrame(index=class_list)

    def _get_arg(self, cell_ids):
        args = list(
            zip(
                range(len(cell_ids)),
                [self.save_index_lookup.index.get_loc(x) for x in cell_ids],
                cell_ids,
            )
        )
        return args

    def _generate_batched_args(self, args, max_batch_size=3000, min_batch_size=100):
        """
        Helper function to generate batched arguments for multiprocessing.
        Batched args are mini-batches of the original arguments that are used to split the processing into smaller chunks to prevent memory issues.
        """

        if "max_batch_size" in self.config:
            max_batch_size = self.config["max_batch_size"]
        else:
            max_batch_size = max_batch_size

        theoretical_max = np.ceil(len(args) / self.config["threads"])
        batch_size = min(max_batch_size, theoretical_max)

        self.batch_size = np.int64(batch_size)
        self.log(f"Using batch size of {self.batch_size} for multiprocessing.")

        # dynamically adjust the number of threads to ensure that we dont initiate more threads than we have arguments
        self.threads = np.int64(
            min(self.config["threads"], np.ceil(len(args) / self.batch_size))
        )

        if self.threads != self.config["threads"]:
            self.log(
                f"Reducing number of threads to {self.threads} to match number of cell batches to process."
            )

        return [
            args[i : i + self.batch_size] for i in range(0, len(args), self.batch_size)
        ]

    def _get_label_info(self, arg):
        index, save_index, cell_id = arg

        # no additional labelling required
        return (index, save_index, cell_id, None, None)

    def _save_removed_classes(self, classes):
        # define path where classes should be saved
        filtered_path = os.path.join(
            self.project_location,
            self.DEFAULT_SEGMENTATION_DIR_NAME,
            self.DEFAULT_REMOVED_CLASSES_FILE,
        )

        to_write = "\n".join([str(i) for i in list(classes)])

        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        self.log(
            f"A total of {len(classes)} cells were too close to the image border to be extracted. Their cell_ids were saved to file {filtered_path}."
        )

    def _save_cell_info(self, save_index, cell_id, image_index, label_info, stack):
        """helper function to save the extracted cell information to the temporary datastructures

        Parameters
        ----------
        save_index : int
            index location in the temporary datastructures where the cell in question needs to be saved
        cell_id : int
            unique identifier of extracted cell
        image_index : int | None
            index of the source image that was processed. Only relevant for TimecourseProjects. Otherwise None.
        label_info : str | None
            additional information that is to be saved with the extracted cell. Only relevant for TimecourseProjects. Otherwise None.
        stack : np.array
            extracted single cell images that are too be saved
        """
        # label info is None so just ignore for the base case
        # image_index is none so just ignore for the base case

        # save single cell images
        self._tmp_single_cell_data[save_index] = stack
        self._tmp_single_cell_index[save_index] = [save_index, cell_id]

    def _save_failed_cell_info(self, save_index, cell_id, image_index, label_info):
        """save the relevant information for cells that are too close to the image edges to extract

        Parameters
        ----------
        save_index : int
            index location in the temporary datastructures where the cell in question should have been saved, this index
            location will later be deleted
        cell_id : int
            unique identifier of the cell which was unable to be extracted

        """

        # image index and label_info can be ignored for the base case is only relevant for the timecourse extraction
        self._tmp_single_cell_index[save_index] = [save_index, cell_id]

    def _transfer_tempmmap_to_hdf5(self):
        self.log("Transferring results to final HDF5 data container.")

        # reconnect to memory mapped temp arrays
        _tmp_single_cell_index = mmap_array_from_path(self._tmp_single_cell_index_path)
        _tmp_single_cell_data = mmap_array_from_path(self._tmp_single_cell_data_path)

        self.log(
            f"number of cells too close to image edges to extract: {len(self.save_index_to_remove)}"
        )

        # generate final index of all of the rows that we wish to keep out of the original array
        keep_index = np.setdiff1d(
            np.arange(_tmp_single_cell_index.shape[0]), self.save_index_to_remove
        )

        # get cell_ids of the cells that were successfully extracted
        _, cell_ids = _tmp_single_cell_index[keep_index].T
        _, cell_ids_removed = _tmp_single_cell_index[self.save_index_to_remove].T

        # convert to correct type
        cell_ids = cell_ids.astype(self.DEFAULT_SEGMENTATION_DTYPE)
        cell_ids_removed = cell_ids_removed.astype(self.DEFAULT_SEGMENTATION_DTYPE)

        self.cell_ids_removed = (
            cell_ids_removed  # save for potentially accessing at later time point
        )
        self._save_removed_classes(self.cell_ids_removed)

        if self.debug:
            # visualize some cells for debugging purposes
            # visualize a random cell for every 100 contained in the dataset
            n_cells = 100
            n_cells_to_visualize = len(keep_index) // n_cells

            random_indexes = np.random.choice(
                keep_index, n_cells_to_visualize, replace=False
            )

            for index in random_indexes:
                stack = _tmp_single_cell_data[index]

                fig, axs = plt.subplots(
                    1, stack.shape[0], figsize=(2 * stack.shape[0], 2)
                )
                for i, img in enumerate(stack):
                    axs[i].imshow(img, vmin=0, vmax=1)
                    axs[i].axis("off")
                fig.tight_layout()
                fig.show()

        self.log("Transferring extracted single cells to .hdf5")

        # create name for output file
        self.output_path = os.path.join(
            self.extraction_data_directory, self.DEFAULT_EXTRACTION_FILE
        )

        with h5py.File(self.output_path, "w") as hf:
            hf.create_dataset(
                "single_cell_index",
                data=list(zip(list(range(len(cell_ids))), cell_ids)),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
            )  # increase to 64 bit otherwise information may become truncated

            self.log("single-cell index created.")
            del cell_ids
            # self._clear_cache(vars_to_delete=[cell_ids])
        
        with h5py.File(self.output_path, "a") as hf:
            _, c, x, y = _tmp_single_cell_data.shape
            single_cell_data = hf.create_dataset(
                "single_cell_data",
                shape=(len(keep_index), c, x, y),
                chunks=(1, 1, self.config["image_size"], self.config["image_size"]),
                compression=self.compression_type,
                dtype=np.float16,
            )

            # populate dataset in loop to prevent loading of entire dataset into memory
            # this is required to process large datasets to not run into memory issues
            for ix, i in tqdm(enumerate(keep_index), 
                              total = len(keep_index), 
                              desc = "Transferring single-cell images"):
                single_cell_data[ix] = _tmp_single_cell_data[i]

            self.log("single-cell data created")
            del single_cell_data, _tmp_single_cell_data
            #self._clear_cache(vars_to_delete=[single_cell_data])

            # also transfer labelled index to HDF5
            index_labelled = _tmp_single_cell_index[keep_index]
            index_labelled = (
                pd.DataFrame(index_labelled).iloc[:, 1:].reset_index(drop=True)
            )  # need to reset the lookup index so that it goes up sequentially
            index_labelled = (
                index_labelled.reset_index()
            )  # do this twice to get the index to be the first column of values
            index_labelled = np.char.encode(index_labelled.values.astype(str))

            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset(
                "single_cell_index_labelled", data=index_labelled, chunks=None, dtype=dt
            )

            self.log("single-cell index labelled created.")
            del index_labelled, _tmp_single_cell_index
            #self._clear_cache(vars_to_delete=[index_labelled])

            hf.create_dataset(
                "channel_information",
                data=np.char.encode(self.channel_names.astype(str)),
                dtype=h5py.special_dtype(vlen=str),
            )

            self.log("channel information created.")

        # cleanup memory
        #self._clear_cache(vars_to_delete=[_tmp_single_cell_index])
        os.remove(self._tmp_single_cell_data_path)
        os.remove(self._tmp_single_cell_index_path)

    def _extract_classes(
        self, input_segmentation_path, px_center, arg, return_failed_ids=False
    ):
        """
        Processing for each individual cell that needs to be run for each center.
        """

        index, save_index, cell_id, image_index, label_info = self._get_label_info(
            arg
        )  # label_info not used in base case but relevant for flexibility for other classes

        if self.deep_debug:
            print("cellID type:", type(cell_id), "\n")

        if isinstance(cell_id, str):
            nucleus_id, cytosol_id = cell_id.split(":")
            nucleus_id = int(float(nucleus_id))  # convert to int for further processing
            cytosol_id = int(float(cytosol_id))  # convert to int for further processing

            if self.deep_debug:
                print(f"cell_id: {cell_id}")
                print(f"nucleus_id: {nucleus_id}")
                print(f"cytosol_id: {cytosol_id}")
        else:
            nucleus_id = cell_id
            cytosol_id = cell_id

        ids = [nucleus_id, cytosol_id]
        n_masks = np.int64(min(2, self.n_image_channels))

        with h5py.File(
            input_segmentation_path,
            "r",
            rdcc_nbytes=self.config["hdf5_rdcc_nbytes"],
            rdcc_w0=self.config["hdf5_rdcc_w0"],
            rdcc_nslots=self.config["hdf5_rdcc_nslots"],
        ) as input_hdf:
            hdf_channels = input_hdf.get(self.channel_label)
            hdf_labels = input_hdf.get(self.segmentation_label)

            # get region that should be extracted
            _px_center = px_center[index]
            window_y = slice(
                _px_center[0] - self.width_extraction,
                _px_center[0] + self.width_extraction,
            )
            window_x = slice(
                _px_center[1] - self.width_extraction,
                _px_center[1] + self.width_extraction,
            )

            # ensure that the cell is not too close to the image edge to be extracted
            condition = [
                self.width_extraction < _px_center[0],
                _px_center[0] < self.input_image_width - self.width_extraction,
                self.width_extraction < _px_center[1],
                _px_center[1] < self.input_image_height - self.width_extraction,
            ]

            if np.all(condition):
                masks = []
                image_data = []

                for mask_ix in range(n_masks):
                    # mask 0: nucleus mask
                    if image_index is None:
                        mask = hdf_labels[mask_ix, window_y, window_x]
                    else:
                        mask = hdf_labels[image_index, mask_ix, window_y, window_x]

                    if self.deep_debug:
                        if mask_ix == 0:
                            x, y = mask.shape
                            center_nuclei = mask[
                                slice(x // 2 - 3, x // 2 + 3),
                                slice(y // 2 - 3, y // 2 + 3),
                            ]
                            print("center of nucleus array \n", center_nuclei, "\n")

                    mask = np.where(mask == ids[mask_ix], 1, 0).astype(int)
                    mask = binary_fill_holes(mask)
                    mask = gaussian(mask, preserve_range=True, sigma=1)

                    masks.append(mask)

                for i in range(self.n_image_channels):
                    if image_index is None:
                        # image_data = self.input_image[:, window_y, window_x].compute()
                        channel = hdf_channels[i, window_y, window_x]
                    else:
                        # image_data = self.input_image[image_index, :, window_y, window_x].compute()
                        channel = hdf_channels[image_index, i, window_y, window_x]

                    channel = channel * masks[-1]
                    channel = self.norm_function(channel)

                    image_data.append(channel)

                stack = np.stack(masks + image_data, axis=0).astype(
                    self.DEFAULT_SINGLE_CELL_IMAGE_DTYPE
                )

                if self.remap is not None:
                    stack = stack[self.remap]

                self._save_cell_info(
                    save_index, nucleus_id, image_index, label_info, stack
                )  # to make more flexible for new datastructures with more labelling info

                if self.deep_debug:
                    # visualize some cells for debugging purposes
                    if index % 1000 == 0:
                        print(
                            f"Cell ID: {cell_id} has center at [{_px_center[0]}, {_px_center[1]}]"
                        )

                        fig, axs = plt.subplots(
                            1, stack.shape[0], figsize=(2 * stack.shape[0], 2)
                        )
                        for i, img in enumerate(stack):
                            axs[i].imshow(img, vmin=0, vmax=1)
                            axs[i].axis("off")
                        fig.tight_layout()
                        fig.show()

                if return_failed_ids:
                    return []
                else:
                    return None
            else:
                if self.deep_debug:
                    print(
                        f"cell id {cell_id} is too close to the image edge to extract. Skipping this cell."
                    )

                self.save_index_to_remove.append(save_index)
                self._save_failed_cell_info(
                    save_index,
                    nucleus_id,
                    image_index,
                    label_info,
                )

                if return_failed_ids:
                    return [save_index]
                else:
                    return None

    def _extract_classes_multi(self, input_segmentation_path, px_centers, arg_list):
        # setup normalization functions
        self._get_normalization()

        # connect to temporary storage for saving results
        self._tmp_single_cell_index = mmap_array_from_path(
            self._tmp_single_cell_index_path
        )
        self._tmp_single_cell_data = mmap_array_from_path(
            self._tmp_single_cell_data_path
        )

        results = []
        for arg in arg_list:
            x = self._extract_classes(
                input_segmentation_path, px_centers, arg, return_failed_ids=True
            )
            results.append(x)

        return flatten(results)

    def _post_extraction_cleanup(self, vars_to_delete=None):
        # delete normalizaton functions from self if present to ensure that subsequent multiprocessing runs still run correctly
        if "norm_function" in self.__dict__:
            del self.norm_function

        # delete segmentation masks and input images from self if present
        if "seg_masks" in self.__dict__:
            del self.seg_masks
        if "image_data" in self.__dict__:
            del self.image_data

        # remove no longer required variables
        if vars_to_delete is not None:
            self._clear_cache(vars_to_delete=vars_to_delete)

    def _save_benchmarking_times(
        self,
        total_time,
        time_setup,
        time_arg_generation,
        time_extraction,
        rate_extraction,
    ):
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
                "Number of threads used": [self.config["threads"]],
                "Mini_batch size": [self.batch_size],
                "Total extraction time": [total_time],
                "Time taken to set up extraction": [time_setup],
                "Time taken to generate arguments": [time_arg_generation],
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

    def process(
        self,
        input_segmentation_path,
        filtered_classes_path=None,
        partial=False,
        n_cells=None,
        seed=42,
    ):
        """
        Extracts single cell images from a segmented SPARCSpy project and saves the results to an HDF5 file.

        Parameters
        ----------
        input_segmentation_path : str
            Path of the segmentation HDF5 file. If this class is used as part of a project processing workflow, this argument will be provided automatically.
        filtered_classes_path : str, optional
            Path to the filtered classes that should be used for extraction. Default is None. If not provided, will use the automatically generated paths.

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

                compression: True

                # threads used in multithreading
                threads: 80

                # image size in pixels
                image_size: 128

                # directory where intermediate results should be saved
                cache: "/mnt/temp/cache"

                # specs to define how HDF5 data should be chunked and saved
                hdf5_rdcc_nbytes: 5242880000 # 5GB 1024 * 1024 * 5000
                hdf5_rdcc_w0: 1
                hdf5_rdcc_nslots: 50000
        """

        total_time_start = timeit.default_timer()

        # run all of the extraction setup steps
        start_setup = timeit.default_timer()

        self.input_segmentation_path = input_segmentation_path
        self.filtered_classes_path = filtered_classes_path

        if partial:
            self.partial_processing = True
            self.n_cells = n_cells
            self.seed = seed
        else:
            self.partial_processing = False

        self._setup_extraction()
        stop_setup = timeit.default_timer()
        time_setup = stop_setup - start_setup

        if self.partial_processing:
            self.log(
                f"Starting partial single-cell image extraction of {self.n_cells} cells..."
            )
        else:
            self.log(
                f"Starting single-cell image extraction of {self.num_classes} cells..."
            )

        # generate cell pairings to extract
        start_arg_generation = timeit.default_timer()

        self._generate_save_index_lookup(self.classes)
        args = self._get_arg(self.classes)
        stop_arg_generation = timeit.default_timer()
        time_arg_generation = stop_arg_generation - start_arg_generation

        # actually perform single-cell image extraction
        start_extraction = timeit.default_timer()

        if self.config["threads"] <= 1:
            # set up for single-threaded processing
            self._get_normalization()

            # connect to temporary storage for saving results
            self._tmp_single_cell_index = mmap_array_from_path(
                self._tmp_single_cell_index_path
            )
            self._tmp_single_cell_data = mmap_array_from_path(
                self._tmp_single_cell_data_path
            )

            f = func_partial(
                self._extract_classes, self.input_segmentation_path, self.px_centers
            )

            self.log("Running in single threaded mode.")
            results = []
            for arg in tqdm(args):
                x = f(arg)
                results.append(x)
        else:
            # set up function for multi-threaded processing
            f = func_partial(
                self._extract_classes_multi,
                self.input_segmentation_path,
                self.px_centers,
            )
            batched_args = self._generate_batched_args(args)

            self.log(f"Running in multiprocessing mode with {self.threads} threads.")
            with mp.get_context("fork").Pool(
                processes=self.threads
            ) as pool:  # both spawn and fork work but fork gives faster extraction speeds (probably less overhead than spawning a new process)
                results = list(
                    tqdm(
                        pool.imap(f, batched_args),
                        total=len(batched_args),
                        desc="Processing cell batches",
                    )
                )
                pool.close()
                pool.join()
                print("multiprocessing done.")

            self.save_index_to_remove = flatten(results)

        stop_extraction = timeit.default_timer()

        # calculate duration
        time_extraction = stop_extraction - start_extraction
        rate = self.num_classes / time_extraction

        # generate final log entries
        self.log(
            f"Finished extraction in {time_extraction:.2f} seconds ({rate:.2f} cells / second)"
        )

        # transfer results to hdf5
        self._transfer_tempmmap_to_hdf5()
        self._post_extraction_cleanup()

        total_time_stop = timeit.default_timer()
        total_time = total_time_stop - total_time_start

        self._save_benchmarking_times(
            total_time=total_time,
            time_setup=time_setup,
            time_arg_generation=time_arg_generation,
            time_extraction=time_extraction,
            rate_extraction=rate,
        )


class TimecourseHDF5CellExtraction(HDF5CellExtraction):
    """
    A class to extracts single cell images from a segmented SPARCSpy Timecourse project and save the
    results to an HDF5 file.

    Functionality is the same as the HDF5CellExtraction except that the class is able to deal with an additional dimension(t)
    in the input data.
    """

    DEFAULT_SEGMENTATION_FILE = "input_segmentation.h5"
    CLEAN_LOG = False

    # new parameters to make workflow adaptable to other types of projects
    channel_label = "input_images"
    segmentation_label = "segmentation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_extraction(self):
        if self.partial_processing:
            output_folder_name = (
                f"partial_{self.DEFAULT_DATA_DIR}_{self.n_cells}_{self.seed}"
            )
        else:
            output_folder_name = self.DEFAULT_DATA_DIR

        self._setup_output(folder_name=output_folder_name)

        self._parse_remapping()
        self._get_segmentation_info()
        self._get_input_image_info()

        # setup number of output channels
        self.n_output_channels = self.n_image_channels + self.n_masks

        # get size of images to extract
        self.extracted_image_size = self.config["image_size"]
        self.width_extraction = (
            self.extracted_image_size // 2
        )  # half of the extracted image size (this is what needs to be added on either side of the center)

        self._get_classes()
        self.classes = self.classes_loaded

        # initialize temporary mmap arrays for saving results
        self._initialize_tempmmap_array(index_len=len(self.column_labels))

        self._verbalise_extraction_info()

    def _get_labelling(self):
        with h5py.File(self.input_segmentation_path, "r") as hf:
            self.label_names = hf.get("label_names")[:]
            self.n_labels = len(self.label_names)

    def _get_arg(self):
        # need to extract ID for each cellnumber

        # generate lookuptable where we have all cellids for each tile id
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labels = hf.get("labels").asstr()[:]
            classes = hf.get("classes")

            results = pd.DataFrame(
                columns=["tileids", "cellids"], index=range(labels.shape[0])
            )

            self.log({"Extracting classes from each Segmentation Tile."})
            tile_ids = labels.T[1]

            for i, tile_id in enumerate(tile_ids):
                cellids = list(classes[i])

                # remove background
                if 0 in cellids:
                    cellids.remove(0)

                results.loc[int(i), "cellids"] = cellids
                results.loc[int(i), "tileids"] = tile_id

        # map each cell id to tile id and generate a tuple which can be passed to later functions
        return_results = [
            (xset, i, results.loc[i, "tileids"])
            for i, xset in enumerate(results.cellids)
        ]
        return return_results

    def _get_label_info(self, arg):
        arg_index, save_index, cell_id, image_index, label_info = arg
        return (arg_index, save_index, cell_id, image_index, label_info)

    def _transfer_tempmmap_to_hdf5(self):
        # reconnect to memory mapped temp arrays
        _tmp_single_cell_index = mmap_array_from_path(self._tmp_single_cell_index_path)
        _tmp_single_cell_data = mmap_array_from_path(self._tmp_single_cell_data_path)

        self.log("Transferring results to final HDF5 data container.")

        # generate final index of all of the rows that we wish to keep out of the original array
        keep_index = np.setdiff1d(
            np.arange(_tmp_single_cell_index.shape[0]), self.save_index_to_remove
        )

        # extract information about the annotation of cell ids
        column_labels = ["index", "cellid"] + list(self.label_names.astype("U13"))[1:]

        self.log("Creating HDF5 file to save results to.")

        # define output path
        self.output_path = os.path.join(
            self.extraction_data_directory, self.DEFAULT_DATA_FILE
        )

        with h5py.File(self.output_path, "w") as hf:
            self.log(
                "Transferring extended labelling to ['single_cell_index_labelled'] container."
            )
            # create special datatype for storing strings
            dt = h5py.special_dtype(vlen=str)

            # save label names so that you can always look up the column labelling
            hf.create_dataset("label_names", data=column_labels, chunks=None, dtype=dt)

            # generate index data container
            index_labelled = _tmp_single_cell_index[keep_index]
            index_labelled = (
                pd.DataFrame(index_labelled).iloc[:, 1:].reset_index(drop=True).values
            )  # need to reset the lookup index so that it goes up sequentially
            index_labelled = np.char.encode(index_labelled.astype(str))

            hf.create_dataset(
                "single_cell_index_labelled", data=index_labelled, chunks=None, dtype=dt
            )
            del index_labelled  # cleanup to free up memory

            self.log(
                "Transferring extracted single cells to ['single_cell_data'] container."
            )
            _, c, x, y = _tmp_single_cell_data.shape
            single_cell_data = hf.create_dataset(
                "single_cell_data",
                shape=(len(keep_index), c, x, y),
                chunks=(1, 1, self.config["image_size"], self.config["image_size"]),
                compression=self.compression_type,
                dtype="float16",
            )

            # populate dataset in loop to prevent loading of entire dataset into memory
            # this is required to process large datasets to not run into memory issues
            for ix, i in enumerate(keep_index):
                single_cell_data[ix] = _tmp_single_cell_data[i]

        with h5py.File(self.output_path, "a") as hf:
            self.log(
                "Transferring simple cell_id index to ['single_cell_index'] container."
            )
            # need to save this index seperately since otherwise we get issues with the classificaiton of the extracted cells
            cell_ids = self._tmp_single_cell_index[keep_index, 1]
            index = np.array(list(zip(range(len(cell_ids)), cell_ids)))
            index = index.astype("uint64")

            hf.create_dataset("single_cell_index", data=index, dtype="uint64")
            del index

    def _save_cell_info(self, save_index, cell_id, image_index, label_info, stack):
        # label info is None so just ignore for the base case

        # save single cell images
        self._tmp_single_cell_data[save_index] = stack

        # get label information
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labelling = hf.get("labels").asstr()[image_index][1:]
            save_value = [str(save_index), str(cell_id)]
            save_value = np.array(flatten([save_value, labelling]))

            self._tmp_single_cell_index[save_index] = save_value

            # double check that its really the same values
            if self._tmp_single_cell_index[save_index][2] != label_info:
                self.log("ISSUE INDEXES DO NOT MATCH.")
                self.log(f"index: {save_index}")
                self.log(f"image_index: {image_index}")
                self.log(f"label_info: {label_info}")
                self.log(
                    f"index it should be: {self._tmp_single_cell_index[save_index][2]}"
                )

    def _save_failed_cell_info(self, save_index, cell_id, image_index, label_info):
        """save the relevant information for cells that are too close to the image edges to extract

        Parameters
        ----------
        save_index : int
            index location in the temporary datastructures where the cell in question should have been saved, this index
            location will later be deleted
        cell_id : int
            unique identifier of the cell which was unable to be extracted

        """
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labelling = hf.get("labels").asstr()[image_index][1:]
            save_value = [str(save_index), str(cell_id)]
            save_value = np.array(flatten([save_value, labelling]))

        self._tmp_single_cell_index[save_index] = save_value

        # double check that its really the same values
        if self._tmp_single_cell_index[save_index][2] != label_info:
            self.log("ISSUE INDEXES DO NOT MATCH.")
            self.log(f"index: {save_index}")
            self.log(f"image_index: {image_index}")
            self.log(f"label_info: {label_info}")
            self.log(
                f"index it should be: {self._tmp_single_cell_index[save_index][2]}"
            )

    def process(self, input_segmentation_path, filtered_classes_path=None):
        """
        Process function to run the extraction method.

        Parameters
        ----------
        input_segmentation_path : str
            Path of the segmentation HDF5 file. If this class is used as part of a project processing workflow, this argument will be provided automatically.
        filtered_classes_path : str, optional
            Path to the filtered classes that should be used for extraction. Default is None. If not provided, will use the automatically generated paths.

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

                compression: True

                # threads used in multithreading
                threads: 80

                # image size in pixels
                image_size: 128

                # directory where intermediate results should be saved
                cache: "/mnt/temp/cache"

                # specs to define how HDF5 data should be chunked and saved
                hdf5_rdcc_nbytes: 5242880000 # 5GB 1024 * 1024 * 5000
                hdf5_rdcc_w0: 1
                hdf5_rdcc_nslots: 50000
        """
        timeit.default_timer()

        # run all of the extraction setup steps
        start_setup = timeit.default_timer()

        self.partial_processing = False
        self.input_segmentation_path = input_segmentation_path
        self.filtered_classes_path = filtered_classes_path

        self._get_labelling()

        # define column labels for the index
        self.column_labels = ["index", "cellid"] + list(
            self.label_names.astype("<U512")
        )[1:]

        self._setup_extraction()
        stop_setup = timeit.default_timer()
        stop_setup - start_setup

        # generate arg list
        self._generate_save_index_lookup(self.classes)
        arg_list = self._get_arg()

        self.log(
            f"Starting single-cell image extraction of {self.num_classes} cells..."
        )
        self._get_normalization()

        # reconnect to memory mapped temp arrays
        self._tmp_single_cell_index = mmap_array_from_path(
            self._tmp_single_cell_index_path
        )
        self._tmp_single_cell_data = mmap_array_from_path(
            self._tmp_single_cell_data_path
        )

        with h5py.File(self.input_segmentation_path, "r") as hf:
            start = timeit.default_timer()

            self.log(f"Loading segmentation data from {self.input_segmentation_path}")
            hdf_labels = hf.get(self.segmentation_label)

            for arg in tqdm(arg_list):
                cell_ids, image_index, label_info = arg

                if self.deep_debug:
                    print("image index:", image_index)
                    print("cell ids", cell_ids)
                    print("label info:", label_info)

                input_image = hdf_labels[image_index, 0, :, :]

                # check if image is an empty array
                if np.all(input_image == 0):
                    self.log(
                        f"Image with the image_index {image_index} only contains zeros. Skipping this image."
                    )
                    print(
                        f"Error: image with the index {image_index} only contains zeros!! Skipping this image."
                    )
                    continue
                else:
                    center_nuclei, _, _cell_ids = numba_mask_centroid(
                        input_image, debug=self.debug
                    )

                    if center_nuclei is not None:
                        px_centers = np.round(center_nuclei).astype(int)
                        _cell_ids = list(_cell_ids)

                        if self.deep_debug:
                            # plotting results for debugging
                            import matplotlib.pyplot as plt

                            fig, axs = plt.subplots(1, 3, figsize=(20, 10))
                            axs[0].imshow(hdf_labels[image_index, 0, :, :])
                            axs[0].axis("off")
                            axs[0].set_title("Input Image Nucleus Mask")

                            axs[1].imshow(hdf_labels[image_index, 0, :, :])
                            axs[1].axis("off")
                            axs[1].set_title("Input Image Cytosol Mask")

                            # show calculated centers
                            y, x = px_centers.T
                            axs[3].imshow(hdf_labels[image_index, 0, :, :])
                            axs[3].axis("off")
                            axs[3].scatter(x, y, color="red", s=5)
                            axs[3].set_title(
                                "Nuclei masks with calculated centers overlayed."
                            )

                        # filter lists to only include those cells which passed the final filters (i.e remove border cells)
                        filter = [x in cell_ids for x in _cell_ids]
                        px_centers = np.array(list(compress(px_centers, filter)))
                        _cell_ids = list(compress(_cell_ids, filter))

                        if self.deep_debug:
                            # visualize the centers that pass the filtering thresholds
                            y, x = px_centers.T
                            axs[3].scatter(x, y, color="blue", s=5)

                            # acutally display the figure
                            fig.show()
                            del fig, axs  # remove figure to free up memory

                        for centers_index, cell_id in enumerate(_cell_ids):
                            save_index = self.save_index_lookup.index.get_loc(cell_id)
                            x = self._extract_classes(
                                input_segmentation_path,
                                px_centers,
                                (
                                    centers_index,
                                    save_index,
                                    cell_id,
                                    image_index,
                                    label_info,
                                ),
                                return_failed_ids=True,
                            )
                            self.save_index_to_remove.extend(x)
                    else:
                        self.log(
                            f"Image with the image_index {image_index} doesn't contain any cells. Skipping this image."
                        )

                        if self.deep_debug:
                            failed_image = hf.get(self.channel_label)[
                                image_index, :, :, :
                            ]
                            n_channels = failed_image.shape[0]

                            fig, axs = plt.subplots(
                                1, n_channels, figsize=(10 * n_channels, 10)
                            )
                            for i in range(n_channels):
                                axs[i].imshow(failed_image[i, :, :])
                                axs[i].axis("off")
                                axs[i].set_title(
                                    f"Channel {i} from image with index {image_index} that did not result in any segmented cells"
                                )

                            fig.show()
                            del fig, axs  # remove figure to free up memory
                        continue

            stop = timeit.default_timer()

        duration = stop - start
        rate = self.num_classes / duration
        self.log(
            f"Finished parallel extraction in {duration:.2f} seconds ({rate:.2f} cells / second)"
        )

        self._transfer_tempmmap_to_hdf5()
        self.log("Extraction completed.")
