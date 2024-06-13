import os
import numpy as np
import pandas as pd
import sys
import csv
import h5py
from tqdm.auto import tqdm
from itertools import compress
import timeit
import _pickle as cPickle
import matplotlib.pyplot as plt

from functools import partial
import multiprocessing as mp

from alphabase.io import tempmmap

from skimage.filters import gaussian
from skimage.morphology import disk, dilation
from scipy.ndimage import binary_fill_holes

from sparcscore.processing.segmentation import numba_mask_centroid
from sparcscore.processing.utils import flatten
from sparcscore.processing.preprocessing import percentile_normalization, MinMax
from sparcscore.pipeline.base import ProcessingStep


# to perform garbage collection
import gc


class HDF5CellExtraction(ProcessingStep):
    """
    A class to extracts single cell images from a segmented SPARCSpy project and save the
    results to an HDF5 file.
    """

    DEFAULT_LOG_NAME = "processing.log"
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_SEGMENTATION_DIR = "segmentation"
    DEFAULT_SEGMENTATION_FILE = "segmentation.h5"
    DEFAULT_CLASSES_FILE = "classes.csv"
    DEFAULT_FILTERED_CLASSES_FILE = "filtering/filtered_classes.csv"
    DEFAULT_DATA_DIR = "data"
    SELECTED_DATA_DIR = "selected_data"
    CLEAN_LOG = False

    # new parameters to make workflow adaptable to other types of projects
    channel_label = "channels"
    segmentation_label = "labels"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        base_directory = self.directory.replace("/extraction", "")

        self.input_segmentation_path = os.path.join(
            base_directory,
            self.DEFAULT_SEGMENTATION_DIR,
            self.DEFAULT_SEGMENTATION_FILE,
        )

        # get path to filtered classes
        if os.path.isfile(
            os.path.join(
                base_directory,
                self.DEFAULT_SEGMENTATION_DIR,
                "needs_additional_filtering.txt",
            )
        ):
            try:
                self.classes_path = os.path.join(
                    base_directory,
                    self.DEFAULT_SEGMENTATION_DIR,
                    self.DEFAULT_FILTERED_CLASSES_FILE,
                )
                self.log(
                    f"Loading classes from filtered classes path: {self.classes_path}"
                )
            except Exception:
                raise ValueError("Need to run segmentation_filtering method ")
        else:
            self.classes_path = os.path.join(
                base_directory, self.DEFAULT_SEGMENTATION_DIR, self.DEFAULT_CLASSES_FILE
            )
            self.log(f"Loading classes from default classes path: {self.classes_path}")

        self.output_path = os.path.join(
            self.directory, self.DEFAULT_DATA_DIR, self.DEFAULT_DATA_FILE
        )

        # extract required information for generating datasets
        self.get_compression_type()
        self.get_normalization()

        self.save_index_to_remove = []

        # set developer debug mode for super detailed output
        self.deep_debug = False

    def get_compression_type(self):
        self.compression_type = "lzf" if self.config["compression"] else None
        return self.compression_type

    def get_normalization(self):
        global norm_function, MinMax_function

        if "normalization_range" in self.config:
            self.normalization = self.config["normalization_range"]
        else:
            self.normalization = True

        if self.normalization:

            def norm_function(img):
                return percentile_normalization(img)

            def MinMax_function(img):
                return MinMax(img)

        elif isinstance(self.normalization, tuple):
            lower, upper = self.normalization

            def norm_function(img, lower=lower, upper=upper):
                return percentile_normalization(img, lower, upper)

            def MinMax_function(img):
                return MinMax(img)

        elif self.normalization is None:

            def norm_function(img):
                return img

            def MinMax_function(img):
                img = (
                    img / 65535
                )  # convert 16bit unsigned integer image to float between 0 and 1 without adjusting for the pixel values we have in the extracted single cell image
                return img

        elif (
            self.normalization == "None"
        ):  # add additional check if if None is included as a string

            def norm_function(img):
                return img

            def MinMax_function(img):
                img = (
                    img / 65535
                )  # convert 16bit unsigned integer image to float between 0 and 1 without adjusting for the pixel values we have in the extracted single cell image
                return img

        else:
            self.log("Incorrect type of normalization_range defined.")
            sys.exit("Incorrect type of normalization_range defined.")

    def get_channel_info(self):
        with h5py.File(self.input_segmentation_path, "r") as hf:
            hdf_channels = hf.get(self.channel_label)
            hdf_labels = hf.get(self.segmentation_label)

            if len(hdf_channels.shape) == 3:
                self.n_channels_input = hdf_channels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_channels_input = hdf_channels.shape[1]

            self.log(f"Using channel label {hdf_channels}")
            self.log(f"Using segmentation label {hdf_labels}")

            if len(hdf_labels.shape) == 3:
                self.n_segmentation_channels = hdf_labels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_segmentation_channels = hdf_labels.shape[1]

            self.n_channels_output = (
                self.n_segmentation_channels + self.n_channels_input
            )

    def get_output_path(self):
        return self.extraction_data_directory

    def setup_output(self, folder_name=None):

        if folder_name is None:
            folder_name = self.DEFAULT_DATA_DIR

        self.extraction_data_directory = os.path.join(self.directory, folder_name)
        
        if not os.path.isdir(self.extraction_data_directory):
            os.makedirs(self.extraction_data_directory)
            self.log("Created new data directory " + self.extraction_data_directory)

    def parse_remapping(self):
        self.remap = None
        if "channel_remap" in self.config:
            char_list = self.config["channel_remap"].split(",")
            self.log("channel remap parameter found:")
            self.log(char_list)

            self.remap = [int(el.strip()) for el in char_list]

    def get_classes(self, filtered_classes_path=None):
        if filtered_classes_path is not None:
            self.log(
                f"Loading classes from provided filtered classes path: {filtered_classes_path}"
            )
            path = filtered_classes_path
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
        return filtered_classes

    def generate_save_index_lookup(self, class_list):
        lookup = pd.DataFrame(index=class_list)
        return lookup

    def verbalise_extraction_info(self):
        # print some output information
        self.log("Extraction Details:")
        self.log("--------------------------------")
        self.log(f"Input channels: {self.n_channels_input}")
        self.log(f"Input labels: {self.n_segmentation_channels}")
        self.log(f"Output channels: {self.n_channels_output}")
        self.log(f"Number of classes to extract: {self.num_classes}")
        self.log(
            f"Extracted Image Dimensions: {self.config['image_size']} x {self.config['image_size']}"
        )

    def _get_arg(self, cell_ids, lookup_saveindex):
        lookup_saveindex = self.generate_save_index_lookup(cell_ids)
        args = list(
            zip(
                range(len(cell_ids)),
                [lookup_saveindex.index.get_loc(x) for x in cell_ids],
                cell_ids,
            )
        )
        return args

    def _initialize_tempmmap_array(self, index_len=2):
        # define as global variables so that this is also avaialable in other functions
        global _tmp_single_cell_data, _tmp_single_cell_index

        self.single_cell_index_shape = (self.num_classes, index_len)
        self.single_cell_data_shape = (
            self.num_classes,
            self.n_channels_output,
            self.config["image_size"],
            self.config["image_size"],
        )

        # generate container for single_cell_data
        _tmp_single_cell_data = tempmmap.array(
            shape=self.single_cell_data_shape,
            dtype=np.float16,
            tmp_dir_abs_path=self._tmp_dir_path,
        )

        if index_len == 2:
            # assign dtype int to only save the index and the cell_id
            _tmp_single_cell_index = tempmmap.array(
                shape=self.single_cell_index_shape,
                dtype=np.int64,
                tmp_dir_abs_path=self._tmp_dir_path,
            )
        else:
            # use a regulary numpy array instead of a tempmmap array to be able to save strings as well as ints
            _tmp_single_cell_index = np.empty(
                self.single_cell_index_shape, dtype="<U64"
            )  # need to use U64 here otherwise information potentially becomes truncated

    def _transfer_tempmmap_to_hdf5(self):
        global _tmp_single_cell_data, _tmp_single_cell_index
        self.log("Transferring results to final HDF5 data container.")

        self.log(
            f"number of cells too close to image edges to extract: {len(self.save_index_to_remove)}"
        )

        # generate final index of all of the rows that we wish to keep out of the original array
        keep_index = np.setdiff1d(
            np.arange(_tmp_single_cell_index.shape[0]), self.save_index_to_remove
        )

        # get cell_ids of the cells that were successfully extracted
        _, cell_ids = _tmp_single_cell_index[keep_index].T

        self.log("Transferring extracted single cells to .hdf5")

        if self.debug:
            # visualize some cells for debugging purposes
            x, y = _tmp_single_cell_index.shape
            n_cells = 100
            n_cells_to_visualize = x // n_cells

            random_indexes = np.random.choice(x, n_cells_to_visualize, replace=False)

            for index in random_indexes:
                stack = _tmp_single_cell_data[index]

                fig, axs = plt.subplots(1, stack.shape[0])

                for i, img in enumerate(stack):
                    axs[i].imshow(img)
                    axs[i].axis("off")

                fig.tight_layout()
                fig.show()

        with h5py.File(self.output_path, "w") as hf:
            hf.create_dataset(
                "single_cell_index",
                data=list(zip(list(range(len(cell_ids))), cell_ids)),
                dtype=np.int64,
            )  # increase to 64 bit otherwise information may become truncated
            self.log("index created.")

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
            for ix, i in enumerate(keep_index):
                single_cell_data[ix] = _tmp_single_cell_data[i]

        del _tmp_single_cell_data, _tmp_single_cell_index

    def _get_label_info(self, arg):
        index, save_index, cell_id = arg

        # no additional labelling required
        return (index, save_index, cell_id, None, None)

    def _save_cell_info(self, save_index, cell_id, image_index, label_info, stack):
        # save index is irrelevant for this
        # label info is None so just ignore for the base case
        # image_index is none so just ignore for the base case
        global _tmp_single_cell_data, _tmp_single_cell_index

        # save single cell images
        _tmp_single_cell_data[save_index] = stack
        _tmp_single_cell_index[save_index] = [save_index, cell_id]

    def _extract_classes(self, input_segmentation_path, px_center, arg):
        """
        Processing for each individual cell that needs to be run for each center.
        """
        global norm_function, MinMax_function

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

        # generate some progress output every 10000 cells
        # relevant for benchmarking of time
        if save_index % 10000 == 0:
            self.log("Extracting dataset {}".format(save_index))

        with h5py.File(
            input_segmentation_path,
            "r",
            rdcc_nbytes=self.config["hdf5_rdcc_nbytes"],
            rdcc_w0=self.config["hdf5_rdcc_w0"],
            rdcc_nslots=self.config["hdf5_rdcc_nslots"],
        ) as input_hdf:
            hdf_channels = input_hdf.get(self.channel_label)
            hdf_labels = input_hdf.get(self.segmentation_label)

            width = self.config["image_size"] // 2

            image_width = hdf_channels.shape[
                -2
            ]  # adaptive to ensure that even with multiple stacks of input images this works correctly
            image_height = hdf_channels.shape[-1]
            n_channels = hdf_channels.shape[-3]

            _px_center = px_center[index]
            window_y = slice(_px_center[0] - width, _px_center[0] + width)
            window_x = slice(_px_center[1] - width, _px_center[1] + width)

            condition = [
                width < _px_center[0],
                _px_center[0] < image_width - width,
                width < _px_center[1],
                _px_center[1] < image_height - width,
            ]
            if np.all(condition):
                # mask 0: nucleus mask
                if image_index is None:
                    nuclei_mask = hdf_labels[0, window_y, window_x]
                else:
                    nuclei_mask = hdf_labels[image_index, 0, window_y, window_x]

                if self.deep_debug:
                    x, y = nuclei_mask.shape
                    center_nuclei = nuclei_mask[
                        slice(x // 2 - 3, x // 2 + 3), slice(y // 2 - 3, y // 2 + 3)
                    ]
                    print("center of nucleus array \n", center_nuclei, "\n")

                nuclei_mask = np.where(nuclei_mask == nucleus_id, 1, 0)

                nuclei_mask_extended = gaussian(
                    nuclei_mask, preserve_range=True, sigma=5
                )
                nuclei_mask = gaussian(nuclei_mask, preserve_range=True, sigma=1)

                # channel 0: nucleus
                if image_index is None:
                    channel_nucleus = hdf_channels[0, window_y, window_x]
                else:
                    channel_nucleus = hdf_channels[image_index, 0, window_y, window_x]

                channel_nucleus = norm_function(channel_nucleus)
                channel_nucleus = channel_nucleus * nuclei_mask_extended
                channel_nucleus = MinMax_function(channel_nucleus)

                if n_channels >= 2:
                    # mask 1: cell mask
                    if image_index is None:
                        cell_mask = hdf_labels[1, window_y, window_x]
                    else:
                        cell_mask = hdf_labels[image_index, 1, window_y, window_x]

                    if self.deep_debug:
                        x, y = nuclei_mask.shape
                        center_cytosol = cell_mask[
                            slice(x // 2 - 3, x // 2 + 3), slice(y // 2 - 3, y // 2 + 3)
                        ]
                        print("center of cytosol array \n", center_cytosol, "\n")

                    cell_mask = np.where(cell_mask == cytosol_id, 1, 0).astype(int)
                    cell_mask = binary_fill_holes(cell_mask)

                    cell_mask_extended = dilation(cell_mask, footprint=disk(6))

                    cell_mask = gaussian(cell_mask, preserve_range=True, sigma=1)
                    cell_mask_extended = gaussian(
                        cell_mask_extended, preserve_range=True, sigma=5
                    )

                    # channel 3: cellmask

                    if image_index is None:
                        channel_cytosol = hdf_channels[1, window_y, window_x]
                    else:
                        channel_cytosol = hdf_channels[
                            image_index, 1, window_y, window_x
                        ]

                    channel_cytosol = norm_function(channel_cytosol)
                    channel_cytosol = channel_cytosol * cell_mask_extended
                    channel_cytosol = MinMax_function(channel_cytosol)

                if n_channels == 1:
                    required_maps = [nuclei_mask, channel_nucleus]
                else:
                    required_maps = [
                        nuclei_mask,
                        cell_mask,
                        channel_nucleus,
                        channel_cytosol,
                    ]

                # extract variable feature channels
                feature_channels = []

                if image_index is None:
                    if hdf_channels.shape[0] > 2:
                        for i in range(2, hdf_channels.shape[0]):
                            feature_channel = hdf_channels[i, window_y, window_x]
                            feature_channel = norm_function(feature_channel)
                            feature_channel = feature_channel * cell_mask_extended
                            feature_channel = MinMax_function(feature_channel)

                            feature_channels.append(feature_channel)

                else:
                    if hdf_channels.shape[1] > 2:
                        for i in range(2, hdf_channels.shape[1]):
                            feature_channel = hdf_channels[
                                image_index, i, window_y, window_x
                            ]
                            feature_channel = norm_function(feature_channel)
                            feature_channel = feature_channel * cell_mask_extended
                            feature_channel = MinMax_function(feature_channel)

                            feature_channels.append(feature_channel)

                channels = required_maps + feature_channels
                stack = np.stack(channels, axis=0).astype("float16")

                if self.debug:
                    # visualize some cells for debugging purposes
                    if index % 300 == 0:
                        print(
                            f"Cell ID: {cell_id} has center at [{_px_center[0]}, {_px_center[1]}]"
                        )
                        print("Nucleus ID", nucleus_id)
                        print("Cytosol ID", cytosol_id)

                        plt.figure()
                        plt.imshow(nuclei_mask)
                        plt.title("Nucleus Mask")
                        plt.axis("off")
                        plt.show()

                        if n_channels > 2:
                            plt.figure()
                            plt.imshow(cell_mask)
                            plt.title("Cytosol Mask")
                            plt.axis("off")
                            plt.show()

                            plt.figure()
                            plt.imshow(channel_cytosol)
                            plt.title("Cytosol Channel")
                            plt.axis("off")
                            plt.show()

                        plt.figure()
                        plt.imshow(channel_nucleus)
                        plt.title("Nucleus Channel")
                        plt.axis("off")
                        plt.show()

                        for i, img in enumerate(feature_channels):
                            plt.figure()
                            plt.imshow(img)
                            plt.title(f"Feature Channel {i}")
                            plt.axis("off")
                            plt.show()

                        fig, axs = plt.subplots(1, stack.shape[0])

                        for i, img in enumerate(stack):
                            axs[i].imshow(img)
                            axs[i].axis("off")

                        fig.tight_layout()
                        fig.show()

                if self.remap is not None:
                    stack = stack[self.remap]

                self._save_cell_info(
                    save_index, nucleus_id, image_index, label_info, stack
                )  # to make more flexible for new datastructures with more labelling info
                return []
            else:
                if self.debug:
                    print(
                        f"cell id {cell_id} is too close to the image edge to extract. Skipping this cell."
                    )
                self.save_index_to_remove.append(save_index)
                return [save_index]

    def _calculate_centers(self, hdf_labels):
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
            center_nuclei, length, _cell_ids = numba_mask_centroid(
                hdf_labels[0].astype(np.uint32), debug=self.debug
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

        return (px_centers, _cell_ids)

    def process(self, input_segmentation_path, filtered_classes_path=None):
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
        # is called with the path to the segmented image

        self.get_channel_info()  # needs to be called here after the segmentation is completed
        self.setup_output()
        self.parse_remapping()

        self.log("Started extraction")
        self.log(f"Loading segmentation data from {input_segmentation_path}")

        hf = h5py.File(input_segmentation_path, "r")
        hdf_channels = hf.get(self.channel_label)
        hdf_labels = hf.get(self.segmentation_label)

        self.log("Finished loading channel data " + str(hdf_channels.shape))
        self.log("Finished loading label data " + str(hdf_labels.shape))
        self.n_masks = hdf_labels.shape[0]

        px_centers, _cell_ids = self._calculate_centers(hdf_labels)

        # get classes to extract
        class_list = self.get_classes(filtered_classes_path)
        if isinstance(class_list[0], str):
            lookup_dict = {x.split(":")[0]: x.split(":")[1] for x in class_list}
            nuclei_ids = list(lookup_dict.keys())
            nuclei_ids = set(nuclei_ids)
        else:
            nuclei_ids = set([str(x) for x in class_list])

        # filter cell ids found using center into those that we actually want to extract
        _cell_ids = list(_cell_ids)

        filter = [str(x) in nuclei_ids for x in _cell_ids]

        px_centers = np.array(list(compress(px_centers, filter)))
        _cell_ids = list(compress(_cell_ids, filter))

        # generate new class list
        if isinstance(class_list[0], str):
            class_list = [f"{x}:{lookup_dict[str(x)]}" for x in _cell_ids]
            del lookup_dict
        else:
            class_list = _cell_ids

        self.log(
            f"Number of classes found in filtered classes list {len(nuclei_ids)} vs number of classes for which centers were calculated {len(class_list)}"
        )
        del _cell_ids, filter, nuclei_ids

        # update number of classes
        self.num_classes = len(class_list)

        # setup cache
        self._initialize_tempmmap_array()

        # start extraction
        self.verbalise_extraction_info()

        self.log(f"Starting extraction of {self.num_classes} classes")
        start = timeit.default_timer()

        f = partial(self._extract_classes, input_segmentation_path, px_centers)

        # generate cell pairings to extract
        lookup_saveindex = self.generate_save_index_lookup(class_list)
        args = self._get_arg(class_list, lookup_saveindex)

        with mp.get_context("fork").Pool(processes=self.config["threads"]) as pool:
            x = list(tqdm(pool.imap(f, args), total=len(args)))
            pool.close()
            pool.join()
            print("multiprocessing done.")

        stop = timeit.default_timer()
        self.save_index_to_remove = flatten(x)

        # calculate duration
        duration = stop - start
        rate = self.num_classes / duration

        # generate final log entries
        self.log(
            f"Finished extraction in {duration:.2f} seconds ({rate:.2f} cells / second)"
        )

        # transfer results to hdf5
        self._transfer_tempmmap_to_hdf5()
        self.log("Finished cleaning up cache.")

    def process_partial(
        self, input_segmentation_path, filtered_classes_path=None, n_cells=100
    ):
        # setup output directory
        self.setup_output(folder_name=self.SELECTED_DATA_DIR)
        self.DEFAULT_LOG_NAME = "partial_processing.log"

        self.get_channel_info()
        self.setup_output()
        self.parse_remapping()

        self.log("Started parital extraction")
        self.log(f"Loading segmentation data from {input_segmentation_path}")

        hf = h5py.File(input_segmentation_path, "r")
        hdf_channels = hf.get(self.channel_label)
        hdf_labels = hf.get(self.segmentation_label)

        self.log("Finished loading channel data " + str(hdf_channels.shape))
        self.log("Finished loading label data " + str(hdf_labels.shape))
        self.n_masks = hdf_labels.shape[0]

        px_centers, _cell_ids = self._calculate_centers(hdf_labels)

        # get classes to extract
        class_list = self.get_classes(filtered_classes_path)
        if isinstance(class_list[0], str):
            lookup_dict = {x.split(":")[0]: x.split(":")[1] for x in class_list}
            nuclei_ids = list(lookup_dict.keys())
            nuclei_ids = set(nuclei_ids)
        else:
            nuclei_ids = set([str(x) for x in class_list])

        # filter cell ids found using center into those that we actually want to extract
        _cell_ids = list(_cell_ids)

        filter = [str(x) in nuclei_ids for x in _cell_ids]

        px_centers = np.array(list(compress(px_centers, filter)))
        _cell_ids = list(compress(_cell_ids, filter))

        # generate new class list
        if isinstance(class_list[0], str):
            class_list = [f"{x}:{lookup_dict[str(x)]}" for x in _cell_ids]
            del lookup_dict
        else:
            class_list = _cell_ids

        self.log(
            f"Number of classes found in filtered classes list {len(nuclei_ids)} vs number of classes for which centers were calculated {len(class_list)}"
        )
        del _cell_ids, filter, nuclei_ids

        # subset to only get the N_cells requested from this method
        np.random.seed(42)
        class_list = np.random.choice(class_list, n_cells, replace=False)

        self.log(f"Randomly selected {n_cells} cells to extract")

        # update number of classes
        self.num_classes = len(class_list)

        # setup cache
        self._initialize_tempmmap_array()

        # start extraction
        self.verbalise_extraction_info()

        self.log(f"Starting partial extraction of {self.num_classes} classes")
        start = timeit.default_timer()

        f = partial(self._extract_classes, input_segmentation_path, px_centers)

        # generate cell pairings to extract
        lookup_saveindex = self.generate_save_index_lookup(class_list)
        args = self._get_arg(class_list, lookup_saveindex)

        with mp.get_context("fork").Pool(processes=self.config["threads"]) as pool:
            x = list(tqdm(pool.imap(f, args), total=len(args)))
            pool.close()
            pool.join()
            print("multiprocessing done.")

        stop = timeit.default_timer()
        self.save_index_to_remove = flatten(x)

        # calculate duration
        duration = stop - start
        rate = self.num_classes / duration

        # generate final log entries
        self.log(
            f"Finished extraction in {duration:.2f} seconds ({rate:.2f} cells / second)"
        )

        # transfer results to hdf5
        self._transfer_tempmmap_to_hdf5()
        self.log("Finished cleaning up cache.")

        # reset variable to initial value
        self.setup_output()
        self.DEFAULT_LOG_NAME = "processing.log"


class TimecourseHDF5CellExtraction(HDF5CellExtraction):
    """
    A class to extracts single cell images from a segmented SPARCSpy Timecourse project and save the
    results to an HDF5 file.

    Functionality is the same as the HDF5CellExtraction except that the class is able to deal with an additional dimension(t)
    in the input data.
    """

    DEFAULT_LOG_NAME = "processing.log"
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_SEGMENTATION_DIR = "segmentation"
    DEFAULT_SEGMENTATION_FILE = "input_segmentation.h5"

    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False

    # new parameters to make workflow adaptable to other types of projects
    channel_label = "input_images"
    segmentation_label = "segmentation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_labelling(self):
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
        global _tmp_single_cell_data, _tmp_single_cell_index

        self.log("Transferring results to final HDF5 data container.")

        # generate final index of all of the rows that we wish to keep out of the original array
        keep_index = np.setdiff1d(
            np.arange(_tmp_single_cell_index.shape[0]), self.save_index_to_remove
        )

        # extract information about the annotation of cell ids
        column_labels = ["index", "cellid"] + list(self.label_names.astype("U13"))[1:]

        self.log("Creating HDF5 file to save results to.")

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
            cell_ids = _tmp_single_cell_index[keep_index, 1]
            index = np.array(list(zip(range(len(cell_ids)), cell_ids)))
            index = index.astype("uint64")

            hf.create_dataset("single_cell_index", data=index, dtype="uint64")
            del index

        del _tmp_single_cell_data, _tmp_single_cell_index, keep_index
        gc.collect()

    def _save_cell_info(self, index, cell_id, image_index, label_info, stack):
        global _tmp_single_cell_data, _tmp_single_cell_index
        # label info is None so just ignore for the base case

        # save single cell images
        _tmp_single_cell_data[index] = stack

        # #perform check to see if stack only contains zeros
        # if np.all(stack == 0):
        #     self.log(f"Cell with the index {index} only contains zeros. Skipping this cell.")
        #     self.save_index_to_remove.append(index)
        #     return

        # get label information
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labelling = hf.get("labels").asstr()[image_index][1:]
            save_value = [str(index), str(cell_id)]
            save_value = np.array(flatten([save_value, labelling]))

            _tmp_single_cell_index[index] = save_value

            # double check that its really the same values
            if _tmp_single_cell_index[index][2] != label_info:
                self.log("ISSUE INDEXES DO NOT MATCH.")
                self.log(f"index: {index}")
                self.log(f"image_index: {image_index}")
                self.log(f"label_info: {label_info}")
                self.log(f"index it should be: {_tmp_single_cell_index[index][2]}")

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
        # is called with the path to the segmented image

        self.get_labelling()
        self.get_channel_info()
        self.setup_output()
        self.parse_remapping()

        complete_class_list = self.get_classes(filtered_classes_path)
        arg_list = self._get_arg()
        lookup_saveindex = self.generate_save_index_lookup(complete_class_list)

        # define column labels for the index
        self.column_labels = ["index", "cellid"] + list(self.label_names.astype("U13"))[
            1:
        ]

        # setup cache
        self._initialize_tempmmap_array(index_len=len(self.column_labels))

        # start extraction
        self.log("Starting extraction.")
        self.verbalise_extraction_info()

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
                            save_index = lookup_saveindex.index.get_loc(cell_id)
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
