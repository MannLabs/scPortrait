import os
import gc
import sys
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import h5py
from multiprocessing import current_process
import multiprocessing as mp
import shutil
import torch
import time

import traceback
from PIL import Image
from skimage.color import label2rgb

from sparcscore.processing.segmentation import shift_labels, sc_any
from sparcscore.processing.utils import plot_image
from sparcscore.pipeline.base import ProcessingStep

# for export to ome.zarr
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_labels, write_label_metadata

from alphabase.io import tempmmap


class Segmentation(ProcessingStep):
    """Segmentation helper class used for creating segmentation workflows.

    Attributes:
        maps (dict(str)): Segmentation workflows based on the :class:`.Segmentation` class can use maps for saving and loading checkpoints and perform. Maps can be numpy arrays

        DEFAULT_OUTPUT_FILE (str, default ``segmentation.h5``)
        DEFAULT_FILTER_FILE (str, default ``classes.csv``)
        DEFAULT_FILTER_ADDTIONAL_FILE (str, default ``filtered_classes.csv``)
        PRINT_MAPS_ON_DEBUG (bool, default ``False``)

        identifier (int, default ``None``): Only set if called by :class:`ShardedSegmentation`. Unique index of the shard.
        window (list(tuple), default ``None``): Only set if called by :class:`ShardedSegmentation`. Defines the window which is assigned to the shard. The window will be applied to the input. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.
        input_path (str, default ``None``): Only set if called by :class:`ShardedSegmentation`. Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.

    Example:
        .. code-block:: python

            def process(self):
                # two maps are initialized
                self.maps = {"map0": None,
                             "map1": None}

                # its checked if the segmentation directory already contains these maps and they are then loaded. The index of the first map which has not been found is returned. It indicates the step where computation needs to resume
                current_step = self.load_maps_from_disk()

                if current_step <= 0:
                    # do stuff and generate map0
                    self.save_map("map0")

                if current_step <= 1:
                    # do stuff and generate map1
                    self.save_map("map1")

    """

    DEFAULT_OUTPUT_FILE = "segmentation.h5"
    DEFAULT_FILTER_FILE = "classes.csv"
    DEFAULT_FILTER_ADDTIONAL_FILE = "needs_additional_filtering.txt"
    PRINT_MAPS_ON_DEBUG = True
    DEFAULT_CHANNELS_NAME = "channels"
    DEFAULT_MASK_NAME = "labels"
    DEFAULT_INPUT_IMAGE_NAME = "input_image.ome.zarr"
    DEFAULT_SEGMENTATION_DTYPE = np.uint32

    channel_colors = [
        "#0000FF",
        "#00FF00",
        "#FF0000",
        "#FFE464",
        "#9b19f5",
        "#ffa300",
        "#dc0ab4",
        "#b3d4ff",
        "#00bfa0",
    ]

    def __init__(self, *args, **kwargs):
        # if _tmp_seg is passed as an argument execute this following code (this only applies to some cases)
        if "_tmp_seg_path" in kwargs.keys():
            self._tmp_seg_path = kwargs["_tmp_seg_path"]

            # remove _tmp_seg from kwargs so that underlying classes do not need to account for it
            kwargs.pop("_tmp_seg_path")

        super().__init__(*args, **kwargs)

        self.identifier = None
        self.window = None
        self.input_path = None

        self.deep_debug = False

    def save_classes(self, classes):
        # define path where classes should be saved
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)

        to_write = "\n".join([str(i) for i in list(classes)])

        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        self.log(f"Saved cell_id classes to file {filtered_path}.")

    def check_filter_status(self):
        # check filter status in config
        if "filter_status" in self.config.keys():
            filter_status = self.config["filter_status"]
        else:
            filter_status = True  # always assumes that filtering is performed by default. Needs to be manually turned off if not desired.

        self.log(f"Filtering status for this segmentation is set to {filter_status}.")

        if not filter_status:
            # define path where the empty file should be generated
            filtered_path = os.path.join(
                self.directory, self.DEFAULT_FILTER_ADDTIONAL_FILE
            )

            with open(filtered_path, "w") as myfile:
                myfile.write("\n")

            self.log(
                f"Generated empty file at {filtered_path}. This marks that no filtering has been performed during segmentation and an additional step needs to be performed to populate this file with nucleus_id:cytosol_id matchings before running the extraction."
            )
        elif filter_status:
            self.log(
                "Filtering has been performed during segmentation. Nucleus and Cytosol IDs match. No additional steps are required."
            )

    def initialize_as_shard(self, identifier, window, input_path, zarr_status=True):
        """Initialize Segmentation Step with further parameters needed for federated segmentation.

        Important:
            This function is intended for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

        Args:
            identifier (int): Unique index of the shard.
            window (list(tuple)): Defines the window which is assigned to the shard. The window will be applied to the input. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.
            input_path (str): Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.
        """
        self.identifier = identifier
        self.window = window
        self.input_path = input_path
        self.save_zarr = zarr_status
        self.create_temp_dir()

    def call_as_shard(self):
        """Wrapper function for calling a sharded segmentation.

        Important:
            This function is intended for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.
        """
        self.log(f"Beginning Segmentation of Shard with the slicing {self.window}")

        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get(self.DEFAULT_CHANNELS_NAME)

            # calculate shape of required datacontainer
            c, _, _ = hdf_input.shape
            x1 = self.window[0].start
            x2 = self.window[0].stop
            y1 = self.window[1].start
            y2 = self.window[1].stop

            x = x2 - x1
            y = y2 - y1

            # initialize directory and load data
            if self.deep_debug:
                self.log(
                    f"Generating a memory mapped temp array with the dimensions {(2, x, y)}"
                )
            input_image = tempmmap.array(
                shape=(2, x, y), dtype=np.uint16, tmp_dir_abs_path=self._tmp_dir_path
            )
            input_image = hdf_input[:2, self.window[0], self.window[1]]

        # perform check to see if any input pixels are not 0, if so perform segmentation, else return array of zeros.
        if sc_any(input_image):
            try:
                super().__call__(input_image)
                self.clear_temp_dir()
            except Exception:
                self.log(traceback.format_exc())
                self.clear_temp_dir()
        else:
            self.log(
                f"Shard in position [{self.window[0]}, {self.window[1]}] only contained zeroes."
            )
            try:
                super().__call_empty__(input_image)
                self.clear_temp_dir()
            except Exception:
                self.log(traceback.format_exc())
                self.clear_temp_dir()

        # cleanup generated temp dir and variables
        del input_image
        gc.collect()

        # write out window location
        if self.deep_debug:
            self.log(f"Writing out window location to file at {self.directory}/window.csv")
        with open(f"{self.directory}/window.csv", "w") as f:
            f.write(f"{self.window}\n")

        self.log(f"Segmentation of Shard with the slicing {self.window} finished")

    def save_segmentation(self, channels, labels, classes):
        """Saves the results of a segmentation at the end of the process.

        Args:
            channels (np.array): Numpy array of shape ``(height, width)`` or``(channels, height, width)``. Channels are all data which are saved as floating point values e.g. images.
            labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.
            classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.

        """
        if self.deep_debug:
            self.log("saving segmentation")

        # size (C, H, W) is expected
        # dims are expanded in case (H, W) is passed

        labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels

        map_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        hf = h5py.File(map_path, "a")

        # check if data container already exists and if so delete
        if self.DEFAULT_MASK_NAME in hf.keys():
            del hf[self.DEFAULT_MASK_NAME]
            self.log(
                "labels dataset already existed in hdf5, dataset was deleted and will be overwritten."
            )

        hf.create_dataset(
            self.DEFAULT_MASK_NAME,
            data=labels,
            chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
        )

        # check if data container already exists and if so delete
        if self.DEFAULT_CHANNELS_NAME in hf.keys():
            del hf[self.DEFAULT_CHANNELS_NAME]
            self.log(
                "channels dataset already existed in hdf5, dataset was deleted and will be overwritten."
            )

        # also save channels
        hf.create_dataset(
            self.DEFAULT_CHANNELS_NAME,
            data=channels,
            chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
        )

        hf.close()

        # save classes
        self.check_filter_status()
        self.save_classes(classes)

        self.log("=== finished segmentation ===")
        self.save_segmentation_zarr(labels=labels)

    def save_segmentation_zarr(self, labels=None):
        """Saves the results of a segemtnation at the end of the process to ome.zarr"""
        if hasattr(self, "save_zarr"):
            if self.save_zarr:
                self.log("adding segmentation to input_image.ome.zarr")
                path = os.path.join(
                    self.project_location, self.DEFAULT_INPUT_IMAGE_NAME
                )

                loc = parse_url(path, mode="w").store
                group = zarr.group(store=loc)

                segmentation_names = ["nucleus", "cytosol"]

                # check if segmentation names already exist if so delete
                for seg_names in segmentation_names:
                    path = os.path.join(
                        self.project_location,
                        self.DEFAULT_INPUT_IMAGE_NAME,
                        self.DEFAULT_MASK_NAME,
                        seg_names,
                    )
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        self.log(
                            f"removed existing {seg_names} segmentation from ome.zarr"
                        )

                # reading labels
                if labels is None:
                    path_labels = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

                    with h5py.File(path_labels, "r") as hf:
                        # initialize tempmmap array to save label results into
                        labels = tempmmap.array(
                            shape=hf[self.DEFAULT_MASK_NAME].shape,
                            dtype=hf[self.DEFAULT_MASK_NAME].dtype,
                            tmp_dir_abs_path=self._tmp_dir_path,
                        )

                        labels[0] = hf[self.DEFAULT_MASK_NAME][0]
                        labels[1] = hf[self.DEFAULT_MASK_NAME][1]

                segmentations = [np.expand_dims(seg, axis=0) for seg in labels]

                for seg, name in zip(segmentations, segmentation_names):
                    write_labels(
                        labels=seg.astype("uint16"), group=group, name=name, axes="cyx"
                    )
                    write_label_metadata(
                        group=group,
                        name=f"labels/{name}",
                        colors=[{"label-value": 0, "rgba": [0, 0, 0, 0]}],
                    )

                self.log("finished saving segmentation results to ome.zarr")
                del labels
            else:
                self.log(
                    "Not saving shard segmentation into ome.zarr. Will only save completely assembled image."
                )
                pass
        else:
            self.log("save_zarr attribute not found")

    def load_maps_from_disk(self):
        """Tries to load all maps which were defined in ``self.maps`` and returns the current state of processing.

        Returns
            (int): Index of the first map which could not be loaded. An index of zero indicates that computation needs to start at the first map.

        """

        if not hasattr(self, "maps"):
            raise AttributeError(
                "No maps have been defined. Therefore saving and loading of maps as checkpoints is not supported. Initialize maps in the process method of the segmentation like self.maps = {'map1': None,'map2': None}"
            )

        if not hasattr(self, "directory"):
            self.directory = self.get_directory(self)
            # raise AttributeError("No directory is defined where maps should be saved. Therefore saving and loading of maps as checkpoints is not supported.")

        # iterating over all maps
        for map_index, map_name in enumerate(self.maps.keys()):
            try:
                map_path = os.path.join(
                    self.directory, "{}_{}_map.npy".format(map_index, map_name)
                )

                if os.path.isfile(map_path):
                    map = np.load(map_path)
                    self.log(
                        "Loaded map {} {} from path {}".format(
                            map_index, map_name, map_path
                        )
                    )
                    self.maps[map_name] = map
                else:
                    self.log(
                        "No existing map {} {} found at path {}, new one will be created".format(
                            map_index, map_name, map_path
                        )
                    )
                    self.maps[map_name] = None

            except Exception:
                self.log(
                    "Error loading map {} {} from path {}".format(
                        map_index, map_name, map_path
                    )
                )
                self.maps[map_name] = None

        # determine where to start based on precomputed maps and parameters
        # get index of lowest map which could not be loaded
        # results in index of step where to start

        is_not_none = [el is not None for el in self.maps.values()]
        return np.argmin(is_not_none) if not all(is_not_none) else len(is_not_none)

    def save_map(self, map_name):
        """Saves newly computed map.

        Args
            map_name (str): name of the map to be saved, as defined in ``self.maps``.

        Example:

            .. code-block:: python

                # declare all intermediate maps
                self.maps = {"myMap": None}

                # load intermediate maps if possible and get current processing step
                current_step = self.load_maps_from_disk()

                if current_step <= 0:

                    # do some computations

                    self.maps["myMap"] = myNumpyArray

                    # save map
                    self.save_map("myMap")
        """

        if self.maps[map_name] is None:
            self.log("Error saving map {}, map is None".format(map_name))
        else:
            map_index = list(self.maps.keys()).index(map_name)

            if self.intermediate_output:
                map_path = os.path.join(
                    self.directory, "{}_{}_map.npy".format(map_index, map_name)
                )
                np.save(map_path, self.maps[map_name])
                self.log(
                    "Saved map {} {} under path {}".format(
                        map_index, map_name, map_path
                    )
                )

            # check if map contains more than one channel (3, 1024, 1024) vs (1024, 1024)

            if len(self.maps[map_name].shape) > 2:
                for i, channel in enumerate(self.maps[map_name]):
                    channel_name = "{}_{}_{}_map".format(map_index, map_name, i)
                    channel_path = os.path.join(self.directory, channel_name)

                    if self.debug and self.PRINT_MAPS_ON_DEBUG:
                        self.save_image(channel, save_name=channel_path)
            else:
                channel_name = "{}_{}_map".format(map_index, map_name)
                channel_path = os.path.join(self.directory, channel_name)

                if self.debug and self.PRINT_MAPS_ON_DEBUG:
                    self.save_image(self.maps[map_name], save_name=channel_path)

    def save_image(self, array, save_name="", cmap="magma", **kwargs):
        if np.issubdtype(array.dtype.type, np.integer):
            self.log(f"{save_name} will be saved as tif")
            data = array.astype(np.uint16)
            im = Image.fromarray(data)
            im.save(f"{save_name}.tif")

        fig = plt.figure(frameon=False)
        fig.set_size_inches((10, 10))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(array, cmap=cmap, **kwargs)

        if save_name != "":
            plt.savefig(f"{save_name}.png")
            plt.show()
            plt.close()

    def get_output(self):
        return os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)


class ShardedSegmentation(Segmentation):
    """object which can create log entries.

    Attributes:
        DEFAULT_OUTPUT_FILE (str, default ``segmentation.h5``): Default output file name for segmentations.

        DEFAULT_FILTER_FILE (str, default ``classes.csv``): Default file with filtered class IDs.

        DEFAULT_INPUT_IMAGE_NAME (str, default ``input_image.h5``): Default name for the input image, which is written to disk as hdf5 file.

        DEFAULT_SHARD_FOLDER (str, default ``tiles``): Date and time format used for logging.
    """

    DEFAULT_SHARD_FOLDER = "tiles"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "method"):
            raise AttributeError(
                "No Segmentation method defined, please set attribute ``method``"
            )

        self.save_zarr = False

    def save_input_image(self, input_image):
        output = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        input_image = input_image.astype("uint16")

        with h5py.File(output, "w") as hf:
            hf.create_dataset(
                self.DEFAULT_CHANNELS_NAME,
                data=input_image,
                chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
                dtype="uint16",
            )

        self.log(
            "Input image added to .h5. Provides data source for reading shard information."
        )

    def save_segmentation(self, channels, labels, classes):
        """Saves the results of a segmentation at the end of the process. For the sharded segmentation no channels are passed because they have already been saved

        Args:
            labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.

            classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.

        """
        if self.deep_debug:
            self.log("saving segmentation")

        # size (C, H, W) is expected
        # dims are expanded in case (H, W) is passed

        labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels

        map_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        hf = h5py.File(map_path, "w")

        # check if data container already exists and if so delete
        if self.DEFAULT_MASK_NAME in hf.keys():
            del hf[self.DEFAULT_MASK_NAME]
            self.log(
                "labels dataset already existed in hdf5, dataset was deleted and will be overwritten."
            )

        hf.create_dataset(
            self.DEFAULT_MASK_NAME,
            data=labels,
            chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
        )

        hf.close()

        self.check_filter_status()
        self.save_classes(classes)
        self.save_segmentation_zarr(labels=labels)
        self.log("=== finished segmentation ===")

    def initialize_shard_list(self, sharding_plan):
        _shard_list = []

        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        self.input_path = input_path

        for i, window in enumerate(sharding_plan):
            local_shard_directory = os.path.join(self.shard_directory, str(i))
            current_shard = self.method(
                self.config,
                local_shard_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
            current_shard.initialize_as_shard(
                i, window, self.input_path, zarr_status=False
            )
            _shard_list.append(current_shard)

        return _shard_list

    def initialize_shard_list_incomplete(self, sharding_plan, incomplete_indexes):
        _shard_list = []

        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        self.input_path = input_path

        for i, window in zip(incomplete_indexes, sharding_plan):
            local_shard_directory = os.path.join(self.shard_directory, str(i))
            current_shard = self.method(
                self.config,
                local_shard_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
            current_shard.initialize_as_shard(
                i, window, self.input_path, zarr_status=False
            )
            _shard_list.append(current_shard)

        return _shard_list

    def calculate_sharding_plan(self, image_size):
        # save sharding plan to file
        sharding_plan_path = f"{self.directory}/sharding_plan.csv"

        if os.path.isfile(sharding_plan_path):
            self.log(f"sharding plan already found in directory {sharding_plan_path}.")
            if self.overwrite:
                self.log("Overwriting existing sharding plan.")
                os.remove(sharding_plan_path)
            else:
                self.log("Reading existing sharding plan from file.")
                with open(sharding_plan_path, "r") as f:
                    _sharding_plan = [eval(line) for line in f.readlines()]
                    return _sharding_plan

        _sharding_plan = []
        side_size = np.floor(np.sqrt(int(self.config["shard_size"])))
        shards_side = np.round(image_size / side_size).astype(int)
        shard_size = image_size // shards_side

        self.log(f"input image {image_size[0]} px by {image_size[1]} px")
        self.log(f"target_shard_size: {self.config['shard_size']}")
        self.log("sharding plan:")
        self.log(f"{shards_side[0]} rows by {shards_side[1]} columns")
        self.log(f"{shard_size[0]} px by {shard_size[1]} px")

        for y in range(shards_side[0]):
            for x in range(shards_side[1]):
                last_row = y == shards_side[0] - 1
                last_column = x == shards_side[1] - 1

                lower_y = y * shard_size[0]
                lower_x = x * shard_size[1]

                upper_y = (y + 1) * shard_size[0]
                upper_x = (x + 1) * shard_size[1]

                # add px overlap to each shard
                lower_y = lower_y - self.config["overlap_px"]
                lower_x = lower_x - self.config["overlap_px"]
                upper_y = upper_y + self.config["overlap_px"]
                upper_x = upper_x + self.config["overlap_px"]

                # make sure that each limit stays within the slides
                if lower_y < 0:
                    lower_y = 0
                if lower_x < 0:
                    lower_x = 0

                if last_row:
                    upper_y = image_size[0]

                if last_column:
                    upper_x = image_size[1]

                shard = (slice(lower_y, upper_y), slice(lower_x, upper_x))
                _sharding_plan.append(shard)

        # write out newly generated sharding plan
        with open(sharding_plan_path, "w") as f:
            for shard in _sharding_plan:
                f.write(f"{shard}\n")
        self.log(f"Sharding plan written to file at {sharding_plan_path}")

        return _sharding_plan

    def cleanup_shards(self, sharding_plan):
        file_identifiers_plots = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".pdf"]

        self.log("Moving generated plots from shard directory to main directory.")
        for i, window in enumerate(sharding_plan):
            local_shard_directory = os.path.join(self.shard_directory, str(i))
            for file in os.listdir(local_shard_directory):
                if file.endswith(tuple(file_identifiers_plots)):
                    shutil.copyfile(
                        os.path.join(local_shard_directory, file),
                        os.path.join(self.directory, f"tile{i}_{file}"),
                    )
                    os.remove(os.path.join(local_shard_directory, file))
                    # self.log(f"Moved file {file} from {local_shard_directory} to {self.directory} and renamed it to {i}_{file}")

        # Add section here that cleans up the results from the tiles and deletes them to save memory
        self.log("Deleting intermediate tile results to free up storage space")
        shutil.rmtree(self.shard_directory, ignore_errors=True)

        gc.collect()

    def resolve_sharding(self, sharding_plan):
        """
        The function iterates over a sharding plan and generates a new stitched hdf5 based segmentation.
        """

        self.log("resolve sharding plan")

        output = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        if self.config["input_channels"] == 1:
            label_size = (1, self.image_size[0], self.image_size[1])
        elif self.config["input_channels"] >= 2:
            label_size = (2, self.image_size[0], self.image_size[1])

        with h5py.File(output, "a") as hf:
            # check if data container already exists and if so delete
            if self.DEFAULT_MASK_NAME in hf.keys():
                del hf[self.DEFAULT_MASK_NAME]
                self.log(
                    "labels dataset already existed in hdf5, dataset was deleted and will be overwritten."
                )

            hdf_labels = hf.create_dataset(
                self.DEFAULT_MASK_NAME,
                label_size,
                chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
                dtype="int32",
            )

            class_id_shift = 0

            filtered_classes_combined = set()

            for i, window in enumerate(sharding_plan):
                self.log(f"Stitching tile {i}")

                local_shard_directory = os.path.join(self.shard_directory, str(i))
                local_output = os.path.join(
                    local_shard_directory, self.DEFAULT_OUTPUT_FILE
                )
                local_classes = os.path.join(local_shard_directory, "classes.csv")

                #check if this file exists otherwise abort process
                if not os.path.isfile(local_classes):
                    sys.exit(f"File {local_classes} does not exist. Processing of Shard {i} seems to be incomplete. \nAborting process to resolve sharding. Please run project.complete_segmentation() to regenerate this shard information.")

                # check to make sure windows match
                with open(f"{local_shard_directory}/window.csv", "r") as f:
                    window_local = eval(f.read())
                
                if window_local != window:
                    Warning("Sharding plans do not match.")
                    self.log("Sharding plans do not match.")
                    self.log(f"Sharding plan found locally: {window_local}")
                    self.log(f"Sharding plan found in sharding plan: {window}")
                    self.log(
                        "Reading sharding window from local file and proceeding with that."
                    )
                    window = window_local

                local_hf = h5py.File(local_output, "r")
                local_hdf_labels = local_hf.get(self.DEFAULT_MASK_NAME)

                shifted_map, edge_labels = shift_labels(
                    local_hdf_labels, class_id_shift, return_shifted_labels=True, remove_edge_labels=True
                )

                orig_input = hdf_labels[:, window[0], window[1]]

                if orig_input.shape != shifted_map.shape:
                    Warning("Shapes do not match")
                    self.log("Shapes do not match")
                    self.log("window", window[0], window[1])
                    self.log("shifted_map shape:", shifted_map.shape)
                    self.log("orig_input shape:", orig_input.shape)

                    # dirty fix to get this to run until we can implement a better solution
                    shifted_map = np.zeros(orig_input.shape)

                # since shards are computed with overlap there potentially already exist segmentations in the selected area that we wish to keep
                # if orig_input has a value that is not 0 (i.e. background) and the new map would replace this with 0 then we should keep the original value, in all other cases we should overwrite the values with the
                # new ones from the second shard

                #since segmentations resulting from cellpose are not necessarily deterministic we can not do this lookup on a pixel by pixel basis but need to edit
                #the segmentation mask to remove unwanted shapes before merging

                start_time_step1 = time.time()

                ids_discard = np.unique(orig_input[np.where((orig_input != 0) & (shifted_map != 0))])
                orig_input[np.isin(orig_input, ids_discard)] = 0
                time_step1 = time.time() - start_time_step1

                if self.deep_debug: 
                    orig_input_manipulation = orig_input.copy()
                    shifted_map_manipulation = shifted_map.copy()

                    orig_input_manipulation[orig_input_manipulation>0] = 1
                    shifted_map_manipulation[shifted_map_manipulation>0] = 2

                    resulting_map = orig_input_manipulation + shifted_map_manipulation

                    plt.figure()
                    plt.imshow(resulting_map[0])
                    plt.title(f"Combined nucleus segmentation mask after\n resolving sharding for region {i}")
                    plt.colorbar()
                    plt.show()
                    plt.savefig(f"{self.directory}/combined_nucleus_segmentation_mask_{i}.png")

                    plt.figure()
                    plt.imshow(resulting_map[1])
                    plt.colorbar()
                    plt.title(f"Combined cytosol segmentation mask after\n resolving sharding for region {i}")
                    plt.show()
                    plt.savefig(f"{self.directory}/combined_cytosol_segmentation_mask_{i}.png")

                start_time_step2 = time.time()
                shifted_map = np.where(
                    (orig_input != 0) & (shifted_map == 0), orig_input, shifted_map
                )
                
                time_step2 = time.time() - start_time_step2
                total_time = time_step1 + time_step2
                self.log(f"Time taken to cleanup overlapping shard regions for shard {i}: {total_time}")

                # potential issue: this does not check if we create a cytosol without a matching nucleus? But this should have been implemented in altanas segmentation method
                # for other segmentation methods this could cause issues?? Potentially something to revisit in the future

                hdf_labels[:, window[0], window[1]] = shifted_map
                class_id_shift += np.max(shifted_map) #get highest existing cell id and add it to the shift 
                unique_ids = set(np.unique(shifted_map[0]))

                self.log(f"Number of classes contained in shard after processing: {len(unique_ids)}")
                filtered_classes_combined = filtered_classes_combined.union(unique_ids)  #get unique nucleus ids and add them to the combined filtered class    
                self.log(f"Number of Ids in filtered_classes after adding shard {i}: {len(filtered_classes_combined)}")

                local_hf.close()
                self.log(f"Finished stitching tile {i}")
            
            #remove background class
            filtered_classes_combined = filtered_classes_combined - set([0]) 
            
            self.log(f"Number of filtered classes in Dataset: {len(filtered_classes_combined)}")

            # check filtering classes to ensure that segmentation run is properly tagged
            self.check_filter_status()

            # save newly generated class list
            self.save_classes(list(filtered_classes_combined))

        self.log("resolved sharding plan.")

        # add segmentation results to ome.zarr
        self.save_zarr = True

        # reading labels
        path_labels = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        with h5py.File(path_labels, "r") as hf:
            # initialize tempmmap array to save label results into
            labels = tempmmap.array(
                shape=hf[self.DEFAULT_MASK_NAME].shape,
                dtype=hf[self.DEFAULT_MASK_NAME].dtype,
                tmp_dir_abs_path=self._tmp_dir_path,
            )

            labels[0] = hf[self.DEFAULT_MASK_NAME][0]
            labels[1] = hf[self.DEFAULT_MASK_NAME][1]

        self.save_segmentation_zarr(labels=labels)
        self.log(
            "finished saving segmentation results to ome.zarr from sharded segmentation."
        )

        #self.cleanup_shards(sharding_plan)

    def initializer_function(self, gpu_id_list):
        current_process().gpu_id_list = gpu_id_list

    def process(self, input_image):
        self.save_zarr = False
        self.save_input_image(input_image)
        self.shard_directory = os.path.join(self.directory, self.DEFAULT_SHARD_FOLDER)

        if not os.path.isdir(self.shard_directory):
            os.makedirs(self.shard_directory)
            self.log("Created new shard directory " + self.shard_directory)

        # calculate sharding plan
        self.image_size = input_image.shape[1:]

        if self.config["shard_size"] >= np.prod(self.image_size):
            target_size = self.config["shard_size"]
            self.log(
                f"target size {target_size} is equal or larger to input image {np.prod(self.image_size)}. Sharding will not be used."
            )

            sharding_plan = [
                (slice(0, self.image_size[0]), slice(0, self.image_size[1]))
            ]
        else:
            target_size = self.config["shard_size"]
            self.log(
                f"target size {target_size} is smaller than input image {np.prod(self.image_size)}. Sharding will be used."
            )
            sharding_plan = self.calculate_sharding_plan(self.image_size)

        # save sharding plan to file to be able to reload later
        self.log(f"Saving Sharding plan to file: {self.directory}/sharding_plan.csv")
        with open(f"{self.directory}/sharding_plan.csv", "w") as f:
            for shard in sharding_plan:
                f.write(f"{shard}\n")

        shard_list = self.initialize_shard_list(sharding_plan)
        self.log(
            f"sharding plan with {len(sharding_plan)} elements generated, sharding with {self.config['threads']} threads begins"
        )

        del input_image  # remove from memory to free up space
        gc.collect()  # perform garbage collection

        # check that that number of GPUS is actually available
        if "nGPUS" not in self.config.keys():
            self.config["nGPUs"] = torch.cuda.device_count()

        nGPUS = self.config["nGPUs"]
        available_GPUs = torch.cuda.device_count()
        self.log(f"found {available_GPUs} GPUs.")

        processes_per_GPU = self.config["threads"]

        if available_GPUs != self.config["nGPUs"]:
            self.log(f"Found {available_GPUs} but {nGPUS} specified in config.")

        if available_GPUs >= 1:
            n_processes = processes_per_GPU * available_GPUs
            self.log(
                f"Proceeding in segmentation with {n_processes} number of processes."
            )
        else:
            n_processes = self.config["threads"]
            available_GPUs = (
                1  # default to 1 GPU if non are available and a CPU only method is run
            )

        # initialize a list of available GPUs
        gpu_id_list = []
        for gpu_ids in range(available_GPUs):
            for _ in range(processes_per_GPU):
                gpu_id_list.append(gpu_ids)

        self.log(f"Beginning segmentation on {available_GPUs} GPUs.")

        if n_processes == 1:
            for shard in shard_list:
                shard.call_as_shard()
        else:
            with mp.get_context(self.context).Pool(
                processes=n_processes,
                initializer=self.initializer_function,
                initargs=[gpu_id_list],
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(self.method.call_as_shard, shard_list),
                        total=len(shard_list),
                    )
                )
                pool.close()
                pool.join()
                print("All segmentations are done.", flush=True)

        # free up memory
        del shard_list
        gc.collect()

        self.log("Finished parallel segmentation")
        self.resolve_sharding(sharding_plan)

        # make sure to cleanup temp directories
        self.log("=== finished sharded segmentation === ")

    def complete_segmentation(self, input_image):
        self.save_zarr = False
        self.shard_directory = os.path.join(self.directory, self.DEFAULT_SHARD_FOLDER)

        # check to make sure that the shard directory exisits, if not exit and return error
        if not os.path.isdir(self.shard_directory):
            sys.exit(
                "No Shard Directory found for the given project. Can not complete a segmentation which has not started. Please rerun the segmentation method."
            )

        #save input image to segmentation.h5
        self.save_input_image(input_image)

        # check to see which tiles are incomplete
        tile_directories = os.listdir(self.shard_directory)
        incomplete_indexes = []
        for tile in tile_directories:
            if not os.path.isfile(f"{self.shard_directory}/{tile}/classes.csv"):
                incomplete_indexes.append(int(tile))
                self.log(f"Shard with ID {tile} not completed.")

        # calculate sharding plan
        self.image_size = input_image.shape[1:]

        if self.config["shard_size"] >= np.prod(self.image_size):
            target_size = self.config["shard_size"]
            self.log(
                f"target size {target_size} is equal or larger to input image {np.prod(self.image_size)}. Sharding will not be used."
            )

            sharding_plan = [
                (slice(0, self.image_size[0]), slice(0, self.image_size[1]))
            ]
        else:
            target_size = self.config["shard_size"]
            self.log(
                f"target size {target_size} is smaller than input image {np.prod(self.image_size)}. Sharding will be used."
            )

            # read sharding plan from file
            with open(f"{self.directory}/sharding_plan.csv", "r") as f:
                sharding_plan = [eval(line) for line in f.readlines()]

            self.log(f"Sharding plan read from file {self.directory}/sharding_plan.csv")

        # check to make sure that calculated sharding plan matches to existing sharding results
        if len(sharding_plan) != len(tile_directories):
            sys.exit(
                "Calculated a different number of shards than found shard directories. This indicates a mismatch between the current loaded config file and the config file used to generate the exisiting partial segmentation. Please rerun the complete segmentation to ensure accurate results."
            )

        # select only those shards that did not complete successfully for further processing
        sharding_plan_complete = sharding_plan

        if len(incomplete_indexes) == 0:
            if os.path.isfile(f"{self.directory}/classes.csv"):
                self.log("All segmentations already done")
            else:
                self.log(
                    "Segmentations already completed. Performing Stitching of individual tiles."
                )
                self.resolve_sharding(sharding_plan_complete)

                # make sure to cleanup temp directories
                self.log("=== completed segmentation === ")
        else:
            sharding_plan = [
                shard
                for i, shard in enumerate(sharding_plan)
                if i in incomplete_indexes
            ]
            self.log(
                f"Adjusted sharding plan to only proceed with the {len(incomplete_indexes)} incomplete shards."
            )

            shard_list = self.initialize_shard_list_incomplete(
                sharding_plan, incomplete_indexes
            )
            self.log(f"sharding plan with {len(sharding_plan)} elements generated")

            del input_image  # remove from memory to free up space
            gc.collect()  # perform garbage collection

            # check that that number of GPUS is actually available
            if "nGPUS" not in self.config.keys():
                self.config["nGPUs"] = torch.cuda.device_count()

            nGPUS = self.config["nGPUs"]
            available_GPUs = torch.cuda.device_count()
            self.log(f"found {available_GPUs} GPUs.")

            processes_per_GPU = self.config["threads"]

            if available_GPUs != self.config["nGPUs"]:
                self.log(f"Found {available_GPUs} but {nGPUS} specified in config.")

            if available_GPUs >= 1:
                n_processes = processes_per_GPU * available_GPUs
            else:
                n_processes = self.config["threads"]
                available_GPUs = 1  # default to 1 GPU if non are available and a CPU only method is run

            # initialize a list of available GPUs
            gpu_id_list = []
            for gpu_ids in range(available_GPUs):
                for _ in range(processes_per_GPU):
                    gpu_id_list.append(gpu_ids)

            self.log(f"Beginning segmentation on {available_GPUs}.")

            with mp.get_context(self.context).Pool(
                processes=n_processes,
                initializer=self.initializer_function,
                initargs=[gpu_id_list],
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(self.method.call_as_shard, shard_list),
                        total=len(shard_list),
                    )
                )
                pool.close()
                pool.join()
                print("All segmentations are done.", flush=True)

            # free up memory
            del shard_list
            gc.collect()

            self.log("Finished parallel segmentation of missing shards")
            self.resolve_sharding(sharding_plan_complete)

            # make sure to cleanup temp directories
            self.log("=== completed sharded segmentation === ")


#############################################
###### TIMECOURSE/BATCHED METHODS ###########
#############################################


class TimecourseSegmentation(Segmentation):
    """Segmentation helper class used for creating segmentation workflows working with timecourse data."""

    DEFAULT_OUTPUT_FILE = "input_segmentation.h5"
    DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"
    PRINT_MAPS_ON_DEBUG = True
    DEFAULT_CHANNELS_NAME = "input_images"
    DEFAULT_MASK_NAME = "segmentation"

    channel_colors = [
        "#e60049",
        "#0bb4ff",
        "#50e991",
        "#e6d800",
        "#9b19f5",
        "#ffa300",
        "#dc0ab4",
        "#b3d4ff",
        "#00bfa0",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.index = None
        self.input_path = None

        if not hasattr(self, "method"):
            raise AttributeError(
                "No BaseSegmentationType defined, please set attribute ``BaseSegmentationMethod``"
            )

    def initialize_as_shard(self, index, input_path, _tmp_seg_path):
        """Initialize Segmentation Step with further parameters needed for federated segmentation.

        Important:
            This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

        Args:
            index (int): Unique indexes of the elements that need to be segmented.
            input_path (str): Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.
        """
        self.index = index
        self.input_path = input_path
        self._tmp_seg_path = _tmp_seg_path
        self.create_temp_dir()

    def call_as_shard(self):
        """Wrapper function for calling a sharded segmentation.

        Important:
            This function is intended for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

        """
        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get(self.DEFAULT_CHANNELS_NAME)

            if isinstance(self.index, int):
                self.index = [self.index]

            for index in self.index:
                self.current_index = index
                input_image = hdf_input[index, :, :, :]

                self.log(f"Segmentation on index {index} started.")
                try:
                    super().__call__(input_image)
                    self.clear_temp_dir()
                except Exception:
                    self.log(traceback.format_exc())
                    self.clear_temp_dir()
                self.log(f"Segmentation on index {index} completed.")

    def save_segmentation(
        self,
        input_image,
        labels,
        classes,
    ):
        """
        Saves the results of a segmentation at the end of the process by transferring it to the initialized
        memory mapped array

        Args:
            labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.
            classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.
        """
        # reconnect to existing HDF5 for memory mapping segmentation results
        _tmp_seg = tempmmap.mmap_array_from_path(self._tmp_seg_path)

        # size (C, H, W) is expected
        # dims are expanded in case (H, W) is passed
        labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels
        classes = np.array(list(classes))

        self.log(f"transferring {self.current_index} to temmporray memory mapped array")
        _tmp_seg[self.current_index] = labels

        # close connect to temmpmmap file again
        del _tmp_seg

    def _initialize_tempmmap_array(self):
        # create an empty HDF5 file prepared for using as a memory mapped temp array to save segmentation results to
        # this required when trying to segment so many images that the results can no longer fit into memory
        _tmp_seg_path = tempmmap.create_empty_mmap(
            shape=self.shape_segmentation,
            dtype=self.DEFAULT_SEGMENTATION_DTYPE,
            tmp_dir_abs_path=self._tmp_dir_path,
        )
        self._tmp_seg_path = _tmp_seg_path

    def _transfer_tempmmap_to_hdf5(self):
        _tmp_seg = tempmmap.mmap_array_from_path(self._tmp_seg_path)
        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        # create hdf5 datasets with temp_arrays as input
        with h5py.File(input_path, "a") as hf:
            # check if dataset already exists if so delete and overwrite
            if self.DEFAULT_MASK_NAME in hf.keys():
                del hf[self.DEFAULT_MASK_NAME]
                self.log(
                    "segmentation dataset already existe in hdf5, deleted and overwritten."
                )
            hf.create_dataset(
                self.DEFAULT_MASK_NAME,
                shape=_tmp_seg.shape,
                chunks=(1, 2, self.shape_input_images[2], self.shape_input_images[3]),
                dtype=self.DEFAULT_SEGMENTATION_DTYPE,
            )

            # using this loop structure ensures that not all results are loaded in memory at any one timepoint
            for i in range(_tmp_seg.shape[0]):
                hf[self.DEFAULT_MASK_NAME][i] = _tmp_seg[i]

            dt = h5py.special_dtype(vlen=self.DEFAULT_SEGMENTATION_DTYPE)

            if "classes" in hf.keys():
                del hf["classes"]
                self.log(
                    "classes dataset already existed in hdf5, deleted and overwritten."
                )

            hf.create_dataset(
                "classes",
                shape=self.shape_classes,
                maxshape=(None),
                chunks=None,
                dtype=dt,
            )

        gc.collect()

    def save_image(self, array, save_name="", cmap="magma", **kwargs):
        if np.issubdtype(array.dtype.type, np.integer):
            self.log(f"{save_name} will be saved as tif")
            data = array.astype(np.uint16)
            im = Image.fromarray(data)
            im.save(f"{save_name}.tif")

        fig = plt.figure(frameon=False)
        fig.set_size_inches((10, 10))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()

        fig.add_axes(ax)
        ax.imshow(array, cmap=cmap, **kwargs)

        if save_name != "":
            plt.savefig(f"{save_name}.png")
            plt.show()
            plt.close()

    def adjust_segmentation_indexes(self):
        """
        The function iterates over all present segmented files and adjusts the indexes so that they are unique throughout.
        """

        self.log("resolve segmentation indexes")

        path = os.path.join(self.directory, self.DEFAULT_INPUT_IMAGE_NAME)

        with h5py.File(path, "a") as hf:
            hdf_labels = hf.get(self.DEFAULT_MASK_NAME)
            hdf_classes = hf.get("classes")

            class_id_shift = 0
            filtered_classes_combined = []
            edge_classes_combined = []

            for i in tqdm(
                range(0, hdf_labels.shape[0]),
                total=hdf_labels.shape[0],
                desc="Adjusting Indexes",
            ):
                individual_hdf_labels = hdf_labels[i, :, :, :]
                num_shapes = np.max(individual_hdf_labels)
                shifted_map, edge_labels = shift_labels(
                    individual_hdf_labels, class_id_shift, return_shifted_labels=True
                )
                hdf_labels[i, :, :] = shifted_map

                if set(np.unique(shifted_map[0])) != set(np.unique(shifted_map[1])):
                    self.log(
                        "Warning: Different classes in different segmentatio channels. Please report this example to the developers"
                    )
                    self.log("set1 nucleus: set(np.unique(shifted_map[0]))")
                    self.log("set2 cytosol: set(np.unique(shifted_map[1]))")

                    self.log(
                        f"{set(np.unique(shifted_map[1]))- set(np.unique(shifted_map[0]))} not in nucleus mask"
                    )
                    self.log(
                        f"{set(np.unique(shifted_map[0]))- set(np.unique(shifted_map[1]))} not in cytosol mask"
                    )

                filtered_classes = set(np.unique(shifted_map[0])) - set(
                    [0]
                )  # remove background class
                final_classes = list(filtered_classes - set(edge_labels))

                hdf_labels[i, :, :] = shifted_map
                hdf_classes[i] = np.array(final_classes, dtype=self.DEFAULT_SEGMENTATION_DTYPE).reshape(
                    1, 1, -1
                )

                # save all cells in general
                filtered_classes_combined.extend(filtered_classes)
                edge_classes_combined.extend(edge_labels)

                # adjust class_id shift
                class_id_shift += num_shapes

            edge_classes_combined = set(edge_classes_combined)
            classes_after_edges = list(
                set(filtered_classes_combined) - edge_classes_combined
            )

            self.log("Number of filtered classes combined after segmentation:")
            self.log(len(filtered_classes_combined))

            self.log("Number of classes in contact with image edges:")
            self.log(len(edge_classes_combined))

            self.log("Number of classes after removing image edges:")
            self.log(len(classes_after_edges))

            # save newly generated class list
            self.save_classes(classes_after_edges)

            # sanity check of class reconstruction
            if self.debug:
                all_classes = set(hdf_labels[:].flatten())
                if set(edge_classes_combined).issubset(set(all_classes)):
                    self.log(
                        "Sharding sanity check: edge classes are a full subset of all classes"
                    )
                elif len(set(all_classes)) - len(set(edge_classes_combined)) == len(
                    set(classes_after_edges)
                ):
                    self.log(
                        "Sharding sanity check: sum of edge classes and classes after edges is equal to all classes."
                    )
                else:
                    self.log(
                        "Sharding sanity check: edge classes are NOT a full subset of all classes."
                    )

        self.log("resolved segmentation list")

    def process(self):
        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        with h5py.File(input_path, "r") as hf:
            input_images = hf.get(self.DEFAULT_CHANNELS_NAME)
            indexes = list(range(0, input_images.shape[0]))

            # initialize segmentation dataset
            self.shape_input_images = input_images.shape
            self.shape_segmentation = (
                input_images.shape[0],
                self.config["output_masks"],
                input_images.shape[2],
                input_images.shape[3],
            )
            self.shape_classes = input_images.shape[0]

        # initialize temp object to write segmentations too
        self._initialize_tempmmap_array()

        self.log("Beginning segmentation without multithreading.")

        # initialzie segmentation objects
        current_shard = self.method(
            self.config,
            self.directory,
            project_location=self.project_location,
            debug=self.debug,
            overwrite=self.overwrite,
            intermediate_output=self.intermediate_output,
        )

        current_shard.initialize_as_shard(
            indexes, input_path=input_path, _tmp_seg_path=self._tmp_seg_path
        )

        # calculate results for each shard
        results = [current_shard.call_as_shard()]

        # save results to hdf5
        self.log("Writing segmentation results to .hdf5 file.")
        self._transfer_tempmmap_to_hdf5()

        # adjust segmentation indexes
        self.adjust_segmentation_indexes()
        self.log("Adjusted Indexes.")


class MultithreadedSegmentation(TimecourseSegmentation):
    DEFAULT_OUTPUT_FILE = "input_segmentation.h5"
    DEFAULT_FILTER_FILE = "classes.csv"
    DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "method"):
            raise AttributeError(
                "No Segmentation method defined, please set attribute ``method``"
            )

    def initializer_function(self, gpu_id_list):
        current_process().gpu_id_list = gpu_id_list

    def process(self):
        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        with h5py.File(input_path, "r") as hf:
            input_images = hf.get(self.DEFAULT_CHANNELS_NAME)
            indexes = list(range(0, input_images.shape[0]))

            # initialize segmentation dataset
            self.shape_input_images = input_images.shape
            self.shape_segmentation = (
                input_images.shape[0],
                self.config["output_masks"],
                input_images.shape[2],
                input_images.shape[3],
            )
            self.shape_classes = input_images.shape[0]

        # initialize temp object to write segmentations too
        self._initialize_tempmmap_array()
        segmentation_list = self.initialize_shard_list(
            indexes, input_path=input_path, _tmp_seg_path=self._tmp_seg_path
        )

        # make more verbose output for troubleshooting and timing purposes.
        n_threads = self.config["threads"]
        self.log(f"Beginning segmentation with {n_threads} threads.")
        self.log(f"A total of {len(segmentation_list)} processes need to be executed.")

        # check that that number of GPUS is actually available
        if "nGPUS" not in self.config.keys():
            self.config["nGPUs"] = torch.cuda.device_count()

        nGPUS = self.config["nGPUs"]
        available_GPUs = torch.cuda.device_count()
        processes_per_GPU = self.config["threads"]

        if available_GPUs < self.config["nGPUs"]:
            self.log(f"Found {available_GPUs} but {nGPUS} specified in config.")

        if available_GPUs >= 1:
            n_processes = processes_per_GPU * available_GPUs
        else:
            n_processes = self.config["threads"]
            available_GPUs = (
                1  # default to 1 GPU if non are available and a CPU only method is run
            )

        # initialize a list of available GPUs
        gpu_id_list = []
        for gpu_ids in range(available_GPUs):
            for _ in range(processes_per_GPU):
                gpu_id_list.append(gpu_ids)

        self.log(f"Beginning segmentation on {available_GPUs} available GPUs.")

        with mp.get_context(self.context).Pool(
            processes=n_processes,
            initializer=self.initializer_function,
            initargs=[gpu_id_list],
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(self.method.call_as_shard, segmentation_list),
                    total=len(indexes),
                )
            )
            pool.close()
            pool.join()
            print("All segmentations are done.", flush=True)

        self.log("Finished parallel segmentation")
        self.log("Transferring results to array.")

        self._transfer_tempmmap_to_hdf5()
        self.adjust_segmentation_indexes()
        self.log("Adjusted Indexes.")

        # cleanup variables to make sure memory is cleared up again
        del results
        gc.collect()

    def initialize_shard_list(self, segmentation_list, input_path, _tmp_seg_path):
        _shard_list = []

        for i in tqdm(
            segmentation_list, total=len(segmentation_list), desc="Generating Shards"
        ):
            current_shard = self.method(
                self.config,
                self.directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )

            current_shard.initialize_as_shard(
                i, input_path, _tmp_seg_path=_tmp_seg_path
            )
            _shard_list.append(current_shard)

        self.log(f"Shard list created with {len(_shard_list)} elements.")

        return _shard_list

    def get_output(self):
        return os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
