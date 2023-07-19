import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import h5py
from functools import partial
from multiprocessing import Pool
import shutil

import traceback
from PIL import Image
from skimage.color import label2rgb

from sparcscore.processing.segmentation import shift_labels, sc_any
from sparcscore.processing.utils import plot_image, flatten
from sparcscore.pipeline.base import ProcessingStep

# for export to ome.zarr
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_labels, write_label_metadata 

# to show progress
from tqdm.auto import tqdm


class Segmentation(ProcessingStep):
    """Segmentation helper class used for creating segmentation workflows.
    Attributes:
        maps (dict(str)): Segmentation workflows based on the :class:`.Segmentation` class can use maps for saving and loading checkpoints and perform. Maps can be numpy arrays

        DEFAULT_OUTPUT_FILE (str, default ``segmentation.h5``)
        DEFAULT_FILTER_FILE (str, default ``classes.csv``)
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
    PRINT_MAPS_ON_DEBUG = True

    DEFAULT_INPUT_IMAGE_NAME = "input_image.ome.zarr"

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
        super().__init__(*args, **kwargs)

        self.identifier = None
        self.window = None
        self.input_path = None

    def initialize_as_shard(self, identifier, window, input_path, zarr_status = True):
        """Initialize Segmentation Step with further parameters needed for federated segmentation.

        Important:

            This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

        Args:
            identifier (int): Unique index of the shard.

            window (list(tuple)): Defines the window which is assigned to the shard. The window will be applied to the input. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.

            input_path (str): Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.

        """

        self.identifier = identifier
        self.window = window
        self.input_path = input_path
        self.save_zarr = zarr_status

    def call_as_shard(self):
        """Wrapper function for calling a sharded segmentation.

        Important:

            This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

        """

        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get("channels")
            input_image = hdf_input[:, self.window[0], self.window[1]]

        if input_image.dtype != float:
            input_image = input_image.astype(float)

        #perform check to see if any input pixels are not 0, if so perform segmentation, else return array of zeros.
        if sc_any(input_image):
            try:
                super().__call__(input_image)
            except Exception:
                self.log(traceback.format_exc())
        else:  
            print(f"Shard in position [{self.window[0]}, {self.window[1]}] only contained zeroes.")
            try:
                super().__call_empty__(input_image)
            except Exception:
                self.log(traceback.format_exc())

    def save_segmentation(self, channels, labels, classes):
        """Saves the results of a segmentation at the end of the process.

        Args:
            channels (np.array): Numpy array of shape ``(height, width)`` or``(channels, height, width)``. Channels are all data which are saved as floating point values e.g. images.
            labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.

            classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.

        """
        self.log("saving segmentation")

        # size (C, H, W) is expected
        # dims are expanded in case (H, W) is passed

        channels = (
            np.expand_dims(channels, axis=0) if len(channels.shape) == 2 else channels
        )
        labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels

        map_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        hf = h5py.File(map_path, "w")

        hf.create_dataset(
            "labels",
            data=labels,
            chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
        )
        hf.create_dataset(
            "channels",
            data=channels,
            chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
        )
        hf.close()

        # save classes
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)

        to_write = "\n".join([str(i) for i in list(classes)])
        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        self.log("=== finished segmentation ===")

        self.save_segmentation_zarr(labels = labels)

    def save_segmentation_zarr(self, labels = None):
        """Saves the results of a segemtnation at the end of the process to ome.zarr"""
        if hasattr(self, 'save_zarr'):
            if self.save_zarr:

                self.log("adding segmentation to input_image.ome.zarr")
                path = os.path.join(self.project_location, self.DEFAULT_INPUT_IMAGE_NAME) 

                loc = parse_url(path, mode="w").store
                group = zarr.group(store = loc)

                segmentation_names = ["nucleus", "cyotosol"]

                #check if segmentation names already exist if so delete
                for seg_names in segmentation_names:
                    path = os.path.join(self.project_location, self.DEFAULT_INPUT_IMAGE_NAME, "labels", seg_names)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        self.log(f"removed existing {seg_names} segmentation from ome.zarr")

                #reading labels
                if labels is None:
                    path_labels = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
                    
                    with h5py.File(path_labels, "r") as hf:
                        labels = hf["labels"][:]
                
                segmentations = [np.expand_dims(seg, axis = 0) for seg in labels]

                for seg, name in zip(segmentations, segmentation_names):
                    write_labels(labels = seg.astype("uint16"), group = group, name = name, axes = "cyx")
                    write_label_metadata(group = group, name = f"labels/{name}", colors = [{"label-value": 0, "rgba": [0, 0, 0, 0]}])

                self.log("finished saving segmentation results to ome.zarr")
            else:
                self.log("Not saving shard segmentation into ome.zarr. Will only save completely assembled image.")
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

            except:
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

            hdf_channels = hf.create_dataset(
                "channels",
                data = input_image,
                chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
                dtype="uint16",
            )

        self.log("Input image added to .h5. Provides data source for reading shard information.")

    def initialize_shard_list(self, sharding_plan):
        _shard_list = []

        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        self.input_path = input_path

        for i, window in enumerate(sharding_plan):
            local_shard_directory = os.path.join(self.shard_directory, str(i))
            current_shard = self.method(
                self.config,
                local_shard_directory,
                project_location = self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
            current_shard.initialize_as_shard(i, window, self.input_path, zarr_status = False)
            _shard_list.append(current_shard)

        return _shard_list

    def calculate_sharding_plan(self, image_size):
        _sharding_plan = []
        side_size = np.floor(np.sqrt(int(self.config["shard_size"])))
        shards_side = np.round(image_size / side_size).astype(int)
        shard_size = image_size // shards_side

        self.log(f"input image {image_size[0]} px by {image_size[1]} px")
        self.log(f"target_shard_size: {self.config['shard_size']}")
        self.log(f"sharding plan:")
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

                if last_row:
                    upper_y = image_size[0]

                if last_column:
                    upper_x = image_size[1]

                shard = (slice(lower_y, upper_y), slice(lower_x, upper_x))
                _sharding_plan.append(shard)
        return _sharding_plan

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

        # dirty fix to get this to run until we can impelement a better solution
        if ("wga_segmentation" in self.config):  # need to add this check because otherwise it sometimes throws errors need better solution
            if "wga_background_image" in self.config["wga_segmentation"]:
                if self.config["wga_segmentation"]["wga_background_image"]:
                    channel_size = (
                        self.config["input_channels"] - 1,
                        self.image_size[0],
                        self.image_size[1],
                    )
                else:
                    channel_size = (
                        self.config["input_channels"],
                        self.image_size[0],
                        self.image_size[1],
                    )
            else:
                channel_size = (
                    self.config["input_channels"],
                    self.image_size[0],
                    self.image_size[1],
                )
        else:
            channel_size = (
                self.config["input_channels"],
                self.image_size[0],
                self.image_size[1],
            )

        hf = h5py.File(output, "a")

        hdf_labels = hf.create_dataset(
            "labels",
            label_size,
            chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
            dtype="int32",
        )

        hdf_channels = hf.get("channels")

        class_id_shift = 0

        filtered_classes_combined = []
        edge_classes_combined = []
        for i, window in enumerate(sharding_plan):
            self.log(f"Stitching tile {i}")

            local_shard_directory = os.path.join(self.shard_directory, str(i))
            local_output = os.path.join(local_shard_directory, self.DEFAULT_OUTPUT_FILE)
            local_classes = os.path.join(local_shard_directory, "classes.csv")

            cr = csv.reader(open(local_classes, "r"))
            filtered_classes = [int(el[0]) for el in list(cr)]
            filtered_classes_combined += [
                class_id + class_id_shift
                for class_id in filtered_classes
                if class_id != 0
            ]

            local_hf = h5py.File(local_output, "r")
            local_hdf_channels = local_hf.get("channels")
            local_hdf_labels = local_hf.get("labels")

            shifted_map, edge_labels = shift_labels(
                local_hdf_labels, class_id_shift, return_shifted_labels=True
            )

            hdf_labels[:, window[0], window[1]] = shifted_map

            edge_classes_combined += edge_labels
            class_id_shift += np.max(local_hdf_labels[0])

            local_hf.close()
            self.log(f"Finished stitching tile {i}")

        classes_after_edges = [
            item
            for item in filtered_classes_combined
            if item not in edge_classes_combined
        ]

        self.log("Number of filtered classes combined after sharding:")
        self.log(len(filtered_classes_combined))

        self.log("Number of classes in contact with shard edges:")
        self.log(len(edge_classes_combined))

        self.log("Number of classes after removing shard edges:")
        self.log(len(classes_after_edges))

        # save newly generated class list
        # print filtered classes
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)
        to_write = "\n".join([str(i) for i in list(classes_after_edges)])
        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        # sanity check of class reconstruction
        if self.debug:
            all_classes = set(hdf_labels[:].flatten())
            if set(edge_classes_combined).issubset(set(all_classes)):
                self.log(
                    "Sharding sanity check: edge classes are a full subset of all classes"
                )
            else:
                self.log(
                    "Sharding sanity check: edge classes are NOT a full subset of all classes."
                )

            for i in range(len(hdf_channels)):
                plot_image(hdf_channels[i].astype(np.float64))

            for i in range(len(hdf_labels)):
                image = label2rgb(
                    hdf_labels[i],
                    hdf_channels[0].astype(np.float64)
                    / np.max(hdf_channels[0].astype(np.float64)),
                    alpha=0.5,
                    bg_label=0,
                )
                plot_image(image)

        hf.close()

        self.log("resolved sharding plan.")


        #add segmentation results to ome.zarr
        self.save_zarr = True

        #reading labels
        path_labels = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
            
        with h5py.File(path_labels, "r") as hf:
            labels = hf["labels"][:]

        self.save_segmentation_zarr(labels = labels)
        self.log("finished saving segmentation results to ome.zarr from sharded segmentation.")

        # Add section here that cleans up the results from the tiles and deletes them to save memory
        self.log("Deleting intermediate tile results to free up storage space")
        shutil.rmtree(self.shard_directory)

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
            self.log(f"target size {target_size} is equal or larger to input image {np.prod(self.image_size)}. Sharding will not be used.")

            sharding_plan = [
                (slice(0, self.image_size[0]), slice(0, self.image_size[1]))
            ]
        else:
            target_size = self.config["shard_size"]
            self.log(f"target size {target_size} is smaller than input image {np.prod(self.image_size)}. Sharding will be used.")
            sharding_plan = self.calculate_sharding_plan(self.image_size)

        shard_list = self.initialize_shard_list(sharding_plan)
        self.log(
            f"sharding plan with {len(sharding_plan)} elements generated, sharding with {self.config['threads']} threads begins"
        )

        del input_image #remove from memory to free up space

        with Pool(processes=self.config["threads"]) as pool:
            results = list(
                tqdm(
                    pool.imap(self.method.call_as_shard, shard_list),
                    total=len(shard_list),
                )
            )
            pool.close()
            pool.join()
            print("All segmentations are done.", flush=True)

        self.log("Finished parallel segmentation")

        self.resolve_sharding(sharding_plan)

        self.log("=== finished segmentation === ")


class TimecourseSegmentation(Segmentation):
    """Segmentation helper class used for creating segmentation workflows working with timecourse data."""

    DEFAULT_OUTPUT_FILE = "input_segmentation.h5"
    DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"
    PRINT_MAPS_ON_DEBUG = True
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

    def initialize_as_shard(self, index, input_path):
        """Initialize Segmentation Step with further parameters needed for federated segmentation.

        Important:

            This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

        Args:
            index (int): Unique indexes of the elements that need to be segmented.

            input_path (str): Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.
        """
        self.index = index
        self.input_path = input_path

    def call_as_shard(self):
        """Wrapper function for calling a sharded segmentation.

        Important:

            This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

        """
        global _tmp_seg

        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get("input_images")

            if type(self.index) == int:
                self.index = [self.index]

            results = []
            for index in self.index:
                self.current_index = index
                input_image = hdf_input[index, :, :, :]

                self.log(f"Segmentation on index {index} started.")
                try:
                    _result = super().__call__(input_image)
                except Exception:
                    self.log(traceback.format_exc())
                self.log(f"Segmentation on index {index} completed.")
            
        return results

    def save_segmentation(
        self,
        input_image,
        labels,
        classes,
    ):
        """Saves the results of a segmentation at the end of the process by transferring it to the initialized
        memory mapped array

        Args:
            labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.
            classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.

        """
        global _tmp_seg
        
        # size (C, H, W) is expected
        # dims are expanded in case (H, W) is passed
        labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels
        classes = np.array(list(classes))
        
        self.log(f"transferring {self.current_index} to temmporray memory mapped array")
        _tmp_seg[self.current_index] = labels

    def _initialize_tempmmap_array(self):
        global _tmp_seg
        # import tempmmap module and reset temp folder location
        from alphabase.io import tempmmap

        TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])
        self.TEMP_DIR_NAME = TEMP_DIR_NAME

        # initialize tempmmap array to save segmentation results to
        # this required when trying to segment so many images that the results can no longer fit into memory
        _tmp_seg = tempmmap.array(self.shape_segmentation, dtype=np.int32)

    def _transfer_tempmmap_to_hdf5(self):
        global _tmp_seg
        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        # create hdf5 datasets with temp_arrays as input
        with h5py.File(input_path, "a") as hf:
            # check if dataset already exists if so delete and overwrite
            if "segmentation" in hf.keys():
                del hf["segmentation"]
                self.log(
                    "segmentation dataset already existe in hdf5, deleted and overwritten."
                )
            hf.create_dataset(
                "segmentation",
                shape=_tmp_seg.shape,
                chunks=(1, 2, self.shape_input_images[2], self.shape_input_images[3]),
                dtype="uint32",
            )
            
            hf["segmentation"][:] = _tmp_seg
            
            dt = h5py.special_dtype(vlen=np.dtype("uint32"))

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


        # delete tempobjects (to cleanup directory)
        self.log(f"Tempmmap Folder location {self.TEMP_DIR_NAME} will now be removed.")
        shutil.rmtree(self.TEMP_DIR_NAME, ignore_errors=True)

        del _tmp_seg, self.TEMP_DIR_NAME

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

        output = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        path = os.path.join(self.directory, self.DEFAULT_INPUT_IMAGE_NAME)

        with h5py.File(path, "a") as hf:
            hdf_labels = hf.get("segmentation")
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
                cr = np.unique(individual_hdf_labels)

                filtered_classes = [int(el) for el in list(cr)]
                shifted_map, edge_labels = shift_labels(
                    individual_hdf_labels, class_id_shift, return_shifted_labels=True
                )
                filtered_classes = np.unique(shifted_map)

                edge_labels = set(edge_labels)
                final_classes = [
                    item for item in filtered_classes if item not in edge_labels
                ]

                hdf_labels[i, :, :] = shifted_map
                hdf_classes[i] = np.array(final_classes, dtype="int32").reshape(
                    1, 1, -1
                )

                # save all cells in general
                filtered_classes_combined += [
                    class_id for class_id in filtered_classes if class_id != 0
                ]
                edge_classes_combined += edge_labels

                # adjust class_id shift
                class_id_shift += num_shapes

            edge_classes_combined = set(edge_classes_combined)
            classes_after_edges = [
                item
                for item in filtered_classes_combined
                if item not in edge_classes_combined
            ]

            self.log("Number of filtered classes combined after segmentation:")
            self.log(len(filtered_classes_combined))

            self.log("Number of classes in contact with image edges:")
            self.log(len(edge_classes_combined))

            self.log("Number of classes after removing image edges:")
            self.log(len(classes_after_edges))

            # save newly generated class list
            # print filtered classes
            filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)
            to_write = "\n".join([str(i) for i in list(classes_after_edges)])

            with open(filtered_path, "w") as myfile:
                myfile.write(to_write)

            # sanity check of class reconstruction
            if self.debug:
                all_classes = set(hdf_labels[:].flatten())
                if set(edge_classes_combined).issubset(set(all_classes)):
                    self.log(
                        "Sharding sanity check: edge classes are a full subset of all classes"
                    )
                else:
                    self.log(
                        "Sharding sanity check: edge classes are NOT a full subset of all classes."
                    )

        self.log("resolved segmentation list")

    def process(self):
        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        with h5py.File(input_path, "r") as hf:
            input_images = hf.get("input_images")
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

        #initialzie segmentation objects
        current_shard = self.method(
            self.config,
            self.directory,
            debug=self.debug,
            overwrite=self.overwrite,
            intermediate_output=self.intermediate_output,
        )

        current_shard.initialize_as_shard(indexes, input_path=input_path)

        #calculate results for each shard
        results = [current_shard.call_as_shard()]
       
        # save results to hdf5
        self.log("Writing segmentation results to .hdf5 file.")
        self._transfer_tempmmap_to_hdf5()

        #adjust segmentation indexes
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

    def process(self):
        # global _tmp_seg

        input_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)

        with h5py.File(input_path, "r") as hf:
            input_images = hf.get("input_images")
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


        segmentation_list = self.initialize_shard_list(indexes, input_path=input_path)

        # make more verbose output for troubleshooting and timing purposes.
        n_threads = self.config["threads"]
        self.log(f"Beginning segmentation with {n_threads} threads.")
        self.log(f"A total of {len(segmentation_list)} processes need to be executed.")

        with Pool(processes=self.config["threads"]) as pool:
            results = list(
                tqdm(
                    pool.imap(self.method.call_as_shard, segmentation_list),
                    total=len(indexes),
                )
            )
            print("All segmentations are done.", flush=True)

        self.log("Finished parallel segmentation")
        self.log("Transferring results to array.")

        self._transfer_tempmmap_to_hdf5()
        self.adjust_segmentation_indexes()
        self.log("Adjusted Indexes.")

        # cleanup variables to make sure memory is cleared up again
        del results 

    def initialize_shard_list(self, segmentation_list, input_path):
        _shard_list = []

        for i in tqdm(
            segmentation_list, total=len(segmentation_list), desc="Generating Shards"
        ):
            current_shard = self.method(
                self.config,
                self.directory,
                project_location = self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )

            current_shard.initialize_as_shard(i, input_path)
            _shard_list.append(current_shard)

        self.log(f"Shard list created with {len(_shard_list)} elements.")

        return _shard_list

    def get_output(self):
        return os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
