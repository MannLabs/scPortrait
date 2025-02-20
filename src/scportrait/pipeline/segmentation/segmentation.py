import multiprocessing as mp
import os
import shutil
import sys
import time
import timeit
import traceback
from multiprocessing import current_process

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray
from alphabase.io import tempmmap
from dask.array.core import Array as daskArray
from PIL import Image
from tqdm.auto import tqdm

from scportrait.pipeline._base import ProcessingStep
from scportrait.pipeline._utils.segmentation import _return_edge_labels, sc_any, shift_labels


class Segmentation(ProcessingStep):
    """Segmentation helper class used for creating segmentation workflows.

    Attributes:
        maps (dict(str)): Segmentation workflows based on the :class:`.Segmentation` class can use maps for saving and loading checkpoints and perform. Maps can be numpy arrays

        DEFAULT_FILTER_ADDTIONAL_FILE (str, default ``filtered_classes.csv``)
        PRINT_MAPS_ON_DEBUG (bool, default ``False``)

        identifier (int, default ``None``): Only set if called by :class:`ShardedSegmentation`. Unique index of the shard.
        window (list(tuple), default ``None``): Only set if called by :class:`ShardedSegmentation`. Defines the window which is assigned to the shard. The window will be applied to the input. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.
        input_path (str, default ``None``): Only set if called by :class:`ShardedSegmentation`. Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.

    Example:
        .. code-block:: python

            def process(self):
                # two maps are initialized
                self.maps = {"map0": None, "map1": None}

                # its checked if the segmentation directory already contains these maps and they are then loaded. The index of the first map which has not been found is returned. It indicates the step where computation needs to resume
                current_step = self.load_maps_from_disk()

                if current_step <= 0:
                    # do stuff and generate map0
                    self.save_map("map0")

                if current_step <= 1:
                    # do stuff and generate map1
                    self.save_map("map1")

    """

    CLEAN_LOG = True
    DEFAULT_FILTER_ADDTIONAL_FILE = "needs_additional_filtering.txt"
    PRINT_MAPS_ON_DEBUG = True
    DEFAULT_CHANNELS_NAME = "channels"
    DEFAULT_MASK_NAME = "labels"

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

    def __init__(
        self,
        config,
        directory,
        nuc_seg_name,
        cyto_seg_name,
        _tmp_image_path,
        project_location,
        debug,
        overwrite,
        project,
        filehandler,
        from_project: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            directory,
            project_location,
            debug=debug,
            overwrite=overwrite,
            project=project,
            filehandler=filehandler,
            from_project=from_project,
        )

        if self.directory is not None:
            # only clean directory if a proper directoy is passed
            if self.CLEAN_LOG:
                self._clean_log_file()

        self._check_config()

        # if _tmp_seg is passed as an argument execute this following code (this only applies to some cases)
        if "_tmp_seg_path" in kwargs.keys():
            self._tmp_seg_path = kwargs["_tmp_seg_path"]

            # remove _tmp_seg from kwargs so that underlying classes do not need to account for it
            kwargs.pop("_tmp_seg_path")

        self.identifier = None
        self.window = None
        self.input_path = None
        self.is_shard = False

        # additional parameters to configure level of debugging for developers
        self.deep_debug = False
        self.save_filter_results = False
        self.nuc_seg_name = nuc_seg_name
        self.cyto_seg_name = cyto_seg_name
        self._tmp_image_path = _tmp_image_path
        self.processes_per_GPU = None
        self.n_processes = 1

    def _check_config(self):
        """Check if the configuration is valid."""

        # optional config parameters that can be overridden through the config file
        if "chunk_size" in self.config.keys():
            self.chunk_size = self.config["chunk_size"]
        else:
            self.chunk_size = 50

        if "match_masks" in self.config.keys():
            self.match_masks = self.config["match_masks"]
        else:
            self.match_masks = True

        if "filtering_threshold_mask_matching" in self.config.keys():
            self.filtering_threshold_mask_matching = self.config["filtering_threshold_mask_matching"]
        else:
            self.filtering_threshold_mask_matching = 0.95

    def _check_gpu_status(self):
        if torch.cuda.is_available():
            self.use_GPU = True
            self.device = "cuda"
            self.nGPUs = torch.cuda.device_count()
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

    def _setup_processing(self):
        """
        Checks and updates the GPU status.
        """
        self._check_gpu_status()

        # compare with config configuration
        if "nGPUs" in self.config.keys():
            nGPUs = self.config["nGPUs"]
            if nGPUs == "max":
                self.log(f"Segmentation will be performed with all {self.nGPUs} found GPUs.")
            elif self.nGPUs != nGPUs:
                self.log(f"Found {self.nGPUs} available GPUS but {nGPUs} GPUs specified in config.")
                if self.nGPUs >= 1 and nGPUs >= 1:
                    self.nGPUs = min(self.nGPUs, nGPUs)
                    self.log(f"Will proceed with the number of GPUs specified in config ({self.nGPUs}).")
                else:
                    self.log(f"Segmentation will be performed with all {self.nGPUs} found GPUs.")
        else:
            self.log(f"Segmentation will be performed wtih all {self.nGPUs} found GPUs.")

        # set up threading
        if "threads" in self.config.keys():
            self.processes_per_GPU = self.config["threads"]
        else:
            self.processes_per_GPU = 1

        if self.nGPUs >= 1:
            self.n_processes = self.processes_per_GPU * self.nGPUs
        else:
            self.n_processes = self.processes_per_GPU

        # initialize a list of available GPUs
        gpu_id_list = []
        for gpu_ids in range(self.nGPUs):
            for _ in range(self.processes_per_GPU):
                gpu_id_list.append(gpu_ids)
        self.gpu_id_list = gpu_id_list

        self.log(
            f"GPU Status for segmentation is {self.use_GPU} with {self.nGPUs} GPUs found. Segmentation will be performed on the device {self.device} with {self.processes_per_GPU} processes per device in parallel."
        )

    def _check_filter_status(self):
        # check filter status in config
        if "filter_status" in self.config.keys():
            filter_status = self.config["filter_status"]
        else:
            filter_status = True  # always assumes that filtering is performed by default. Needs to be manually turned off if not desired.

        self.log(f"Filtering status for this segmentation is set to {filter_status}.")

        if not filter_status:
            # define path where the empty file should be generated
            filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_ADDTIONAL_FILE)

            with open(filtered_path, "w") as myfile:
                myfile.write("\n")

            self.log(
                f"Generated empty file at {filtered_path}. This marks that no filtering has been performed during segmentation and an additional step needs to be performed to populate this file with nucleus_id:cytosol_id matchings before running the extraction."
            )
        elif filter_status:
            self.log(
                "Filtering has been performed during segmentation. Nucleus and Cytosol IDs match. No additional steps are required."
            )

    def _load_input_image(self) -> np.ndarray:
        """Loads the input image from the sdatafile.

        Returns:
            np.ndarray: Input image as a np.ndarray.
        """
        start = timeit.default_timer()
        input_image = tempmmap.mmap_array_from_path(self._tmp_image_path)
        self.log(f"Time taken to load input image: {timeit.default_timer() - start}")
        return input_image

    def _select_relevant_channels(self, input_image):
        """transform image dtype and select segmentation channels

        The relevant channels for subsequent segmentation are determined by the channel ID's saved
        in `self.segmentation_channels`.

        Args:
            input_image (np.ndarray): Input image as a np.ndarray.

        Returns:
            np.ndarray: Transformed input image.
        """
        return input_image[self.segmentation_channels]

    def _transform_input_image(self, input_image):
        if isinstance(input_image, xarray.DataArray):
            input_image = input_image.data
        return input_image

    def _save_segmentation(self, labels: np.array, classes: list) -> None:
        """Helper function to save the results of a segmentation to file when generating a segmentation of a shard.

        Args:
            labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.
            classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.

        """
        if self.deep_debug:
            self.log("saving segmentation")

        # size (C, H, W) is expected
        # dims are expanded in case (H, W) is passed

        labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels

        map_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_FILE)
        hf = h5py.File(map_path, "w")

        # check if data container already exists and if so delete
        if self.DEFAULT_MASK_NAME in hf.keys():
            del hf[self.DEFAULT_MASK_NAME]
            self.log("labels dataset already existed in hdf5, dataset was deleted and will be overwritten.")

        hf.create_dataset(
            self.DEFAULT_MASK_NAME,
            data=labels,
            chunks=(1, self.chunk_size, self.chunk_size),
        )

        hf.close()

        # save classes
        self._check_filter_status()
        self._save_classes(classes)

        self.log("=== Finished segmentation of shard ===")

    def _save_segmentation_sdata(self, labels, classes, masks=None):
        if masks is None:
            masks = ["nuclei", "cytosol"]
        if self.is_shard:
            self._save_segmentation(labels, classes)
        else:
            if "nuclei" in masks:
                ix = masks.index("nuclei")

                self.filehandler._write_segmentation_sdata(
                    labels[ix], self.nuc_seg_name, classes=classes, overwrite=self.overwrite
                )
                self.filehandler._add_centers(self.nuc_seg_name, overwrite=self.overwrite)

            if "cytosol" in masks:
                ix = masks.index("cytosol")
                self.filehandler._write_segmentation_sdata(
                    labels[ix], self.cyto_seg_name, classes=classes, overwrite=self.overwrite
                )
                self.filehandler._add_centers(self.cyto_seg_name, overwrite=self.overwrite)

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
            self.log(f"Error saving map {map_name}, map is None")
        else:
            map_index = list(self.maps.keys()).index(map_name)

            # check if map contains more than one channel (3, 1024, 1024) vs (1024, 1024)
            if len(self.maps[map_name].shape) > 2:
                for i, channel in enumerate(self.maps[map_name]):
                    channel_name = f"{map_index}_{map_name}_{i}_map"
                    channel_path = os.path.join(self.directory, channel_name)

                    if self.debug and self.PRINT_MAPS_ON_DEBUG:
                        self.save_image(channel, save_name=channel_path)
            else:
                channel_name = f"{map_index}_{map_name}_map"
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
        return os.path.join(self.directory, self.DEFAULT_SEGMENTATION_FILE)

    def _initialize_as_shard(self, identifier, window, input_path, zarr_status=True):
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
        self.is_shard = True

    def _call_as_shard(self):
        """Wrapper function for calling a sharded segmentation.

        Important:
            This function is intended for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.
        """
        self.log(f"Beginning Segmentation of Shard with the slicing {self.window}")

        input_image = self._load_input_image()
        input_image = self._select_relevant_channels(input_image)

        # select the part of the image that is relevant for this shard
        input_image = input_image[
            :, self.window[0], self.window[1]
        ]  # for some segmentation workflows potentially only the first channel is required this is further selected down in that segmentation workflow
        self.input_image = input_image  # track for potential plotting of intermediate results

        if self.deep_debug:
            self.log(
                f"Input image of dtype {type(input_image)} with dimensions {input_image.shape} passed to sharded segmentation method."
            )

        if sc_any(input_image):
            try:
                self._execute_segmentation(input_image)
                self.clear_temp_dir()
            except (RuntimeError, ValueError, TypeError) as e:
                self.log(f"An error occurred: {e}")
                self.log(traceback.format_exc())
                self.clear_temp_dir()
        else:
            self.log(f"Shard in position [{self.window[0]}, {self.window[1]}] only contained zeroes.")
            try:
                super().__call_empty__(input_image)
                self.clear_temp_dir()
            except (RuntimeError, ValueError, TypeError) as e:
                self.log(f"An error occurred: {e}")
                self.log(traceback.format_exc())
                self.clear_temp_dir()

        self._clear_cache(vars_to_delete=["input_image"])

        # write out window location
        if self.deep_debug:
            self.log(f"Writing out window location to file at {self.directory}/window.csv")
        with open(f"{self.directory}/window.csv", "w") as f:
            f.write(f"{self.window}\n")

        self.log(f"Segmentation of Shard with the slicing {self.window} finished")

    def _save_classes(self, classes: list) -> None:
        """Helper function to save classes to a file when generating a segmentation of a shard."""
        # define path where classes should be saved
        filtered_path = os.path.join(self.directory, self.DEFAULT_CLASSES_FILE)

        to_write = "\n".join([str(i) for i in list(classes)])

        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        self.log(f"Saved cell_id classes to file {filtered_path}.")

    def _save_benchmarking_times(
        self,
        image_size,
        transform_time,
        segmentation_time,
        total_time,
        max_shard_size=None,
        sharding_time=None,
        shard_resolving_time=None,
        time_per_shard=None,
    ):
        benchmarking_path = os.path.join(self.directory, self.DEFAULT_BENCHMARKING_FILE)

        benchmarking = pd.DataFrame(
            {
                "Size of the image": [image_size],
                "Number of GPUs used": [self.nGPUs],
                "Number of processes per GPU": [self.processes_per_GPU],
                "Total number of processes": [self.n_processes],
                "Shard max size": [max_shard_size if max_shard_size is not None else "N/A"],
                "Time taken for transformation": [transform_time],
                "Time taken for sharding": [sharding_time if sharding_time is not None else "N/A"],
                "Time taken for segmentation": [segmentation_time],
                "Time taken for shard resolving": [shard_resolving_time if shard_resolving_time is not None else "N/A"],
                "Time taken per shard": [time_per_shard if time_per_shard is not None else "N/A"],
                "Total time taken": [total_time],
            }
        )

        if os.path.exists(benchmarking_path):
            benchmarking.to_csv(benchmarking_path, mode="a", header=False, index=False)
        else:
            benchmarking.to_csv(benchmarking_path, index=False)

    def process(self, input_image):
        """Process the input image with the segmentation method."""
        image_size = input_image.shape
        input_image = self._select_relevant_channels(input_image)
        self._execute_segmentation(input_image)

        self._save_benchmarking_times(
            image_size=image_size,
            transform_time=self.transform_time,
            segmentation_time=self.segmentation_time,
            total_time=self.total_time,
        )


class ShardedSegmentation(Segmentation):
    """To perform a sharded segmentation where the input image is split into individual tiles (with overlap) that are processed idnividually before the results are joined back together."""

    DEFAULT_MASK_NAME = "labels"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "method"):
            raise AttributeError("No Segmentation method defined, please set attribute ``method``")

        # initialize a dummy instance of the segmentation method to determine which channels need to be loaded for segmentation
        test_method = self.method(
            self.config,
            directory=None,
            _tmp_image_path=None,
            nuc_seg_name=self.nuc_seg_name,
            cyto_seg_name=self.cyto_seg_name,
            filehandler=self.filehandler,
            project=None,
            project_location=None,
            debug=self.debug,
            overwrite=self.overwrite,
        )
        self.segmentation_channels = test_method.segmentation_channels

    def _check_config(self):
        super()._check_config()

        # required config parameters for a sharded segmentation
        assert "shard_size" in self.config.keys(), "No shard size specified in config."
        assert "overlap_px" in self.config.keys(), "No overlap specified in config."
        assert "threads" in self.config.keys(), "No threads specified in config."

    def _calculate_sharding_plan(self, image_size) -> list:
        """Calculate the sharding plan for the given input image size."""

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

        return _sharding_plan

    def _get_sharding_plan(self, overwrite, force_read: bool = False) -> list:
        # check if a sharding plan already exists
        sharding_plan_path = f"{self.directory}/sharding_plan.csv"

        if os.path.isfile(sharding_plan_path):
            self.log(f"sharding plan already found in directory {sharding_plan_path}.")
            if overwrite:
                self.log("Overwriting existing sharding plan.")
                os.remove(sharding_plan_path)
            else:
                self.log("Reading existing sharding plan from file.")
                with open(sharding_plan_path) as f:
                    sharding_plan = [eval(line) for line in f.readlines()]
                    return sharding_plan

        if force_read:
            raise FileNotFoundError(
                "No sharding plan found in directory. Please run project.shard_segmentation() to generate a sharding plan. Can not complete a segmentation without the original sharding plan."
            )

        if self.config["shard_size"] >= np.prod(self.image_size):
            target_size = self.config["shard_size"]
            self.log(
                f"target size {target_size} is equal or larger to input image {np.prod(self.image_size)}. Sharding will not be used."
            )

            sharding_plan = [(slice(0, self.image_size[0]), slice(0, self.image_size[1]))]
        else:
            target_size = self.config["shard_size"]
            self.log(
                f"target size {target_size} is smaller than input image {np.prod(self.image_size)}. Sharding will be used."
            )
            sharding_plan = self._calculate_sharding_plan(self.image_size)

        # save sharding plan to file to be able to reload later
        self.log(f"Saving Sharding plan to file: {self.directory}/sharding_plan.csv")
        with open(f"{self.directory}/sharding_plan.csv", "w") as f:
            for shard in sharding_plan:
                f.write(f"{shard}\n")

        return sharding_plan

    def _initialize_shard_list(self, sharding_plan):
        _shard_list = []

        self.input_path = self.filehandler.sdata_path

        for i, window in enumerate(sharding_plan):
            local_shard_directory = os.path.join(self.shard_directory, str(i))

            current_shard = self.method(
                self.config,
                directory=local_shard_directory,
                _tmp_image_path=self.input_image_path,
                nuc_seg_name=self.nuc_seg_name,
                cyto_seg_name=self.cyto_seg_name,
                filehandler=self.filehandler,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                project=None,
            )

            current_shard._initialize_as_shard(i, window, self.input_path, zarr_status=False)
            _shard_list.append(current_shard)

        return _shard_list

    def _cleanup_shards(self, sharding_plan, keep_plots=False):
        file_identifiers_plots = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".pdf"]

        if keep_plots:
            self.log("Moving generated plots from shard directory to main directory.")
            for i, _window in enumerate(sharding_plan):
                local_shard_directory = os.path.join(self.shard_directory, str(i))
                for file in os.listdir(local_shard_directory):
                    if file.endswith(tuple(file_identifiers_plots)):
                        shutil.copyfile(
                            os.path.join(local_shard_directory, file),
                            os.path.join(self.directory, f"tile{i}_{file}"),
                        )
                        os.remove(os.path.join(local_shard_directory, file))

        # Add section here that cleans up the results from the tiles and deletes them to save memory
        self.log("Deleting intermediate tile results to free up storage space")
        shutil.rmtree(self.shard_directory, ignore_errors=True)

        self._clear_cache()

    def _resolve_sharding(self, sharding_plan):
        """
        The function iterates over a sharding plan and generates a new stitched hdf5 based segmentation.
        """

        self.log("resolve sharding plan")

        # ensure a temp directory is creates
        if not hasattr(self, "_tmp_dir_path"):
            self.create_temp_dir()

        label_size = (self.method.N_MASKS, self.image_size[0], self.image_size[1])

        # initialize an empty hdf5 file that will be filled with the results of the sharded segmentation
        # this is a workaround because as of yet labels can not be incrementally updated in spatialdata objects while being backed to disk

        hdf_labels_path = tempmmap.create_empty_mmap(
            shape=label_size,
            dtype=self.DEFAULT_SEGMENTATION_DTYPE,
            tmp_dir_abs_path=self._tmp_dir_path,
        )

        # clear temp directory used for sharding
        os.remove(self.input_image_path)
        self.log("Cleared temporary directory containing input image used for sharding.")

        hdf_labels = tempmmap.mmap_array_from_path(hdf_labels_path)

        class_id_shift = 0
        filtered_classes_combined = set()

        for i, window in enumerate(sharding_plan):
            timer = time.time()

            self.log(f"Stitching tile {i}")

            local_shard_directory = os.path.join(self.shard_directory, str(i))
            local_output = os.path.join(local_shard_directory, self.DEFAULT_SEGMENTATION_FILE)
            local_classes = os.path.join(local_shard_directory, "classes.csv")

            # check if this file exists otherwise abort process
            if not os.path.isfile(local_classes):
                sys.exit(
                    f"File {local_classes} does not exist. Processing of Shard {i} seems to be incomplete. \nAborting process to resolve sharding. Please run project.complete_segmentation() to regenerate this shard information."
                )

            # check to make sure windows match
            with open(f"{local_shard_directory}/window.csv") as f:
                window_local = eval(f.read())

            if window_local != window:
                Warning("Sharding plans do not match.")
                self.log("Sharding plans do not match.")
                self.log(f"Sharding plan found locally: {window_local}")
                self.log(f"Sharding plan found in sharding plan: {window}")
                self.log("Reading sharding window from local file and proceeding with that.")
                window = window_local

            local_hf = h5py.File(local_output, "r")
            local_hdf_labels = local_hf.get(self.DEFAULT_MASK_NAME)[:]

            shifted_map, edge_labels = shift_labels(
                local_hdf_labels,
                class_id_shift,
                return_shifted_labels=True,
                remove_edge_labels=True,
            )

            orig_input = hdf_labels[:, window[0], window[1]]
            _, x, y = orig_input.shape
            c_shifted, x_shifted, y_shifted = shifted_map.shape

            if x != x_shifted or y != y_shifted or c_shifted != self.method.N_MASKS:
                Warning("Shapes do not match")
                self.log("Shapes do not match")
                self.log(f"window: {(window[0], window[1])}")
                self.log(f"shifted_map shape: {shifted_map.shape}")
                self.log(f"orig_input shape: {orig_input.shape}")

                raise ValueError("Shapes do not match. Please send this example to the developers for troubleshooting.")

            # since shards are computed with overlap there potentially already exist segmentations in the selected area that we wish to keep
            # if orig_input has a value that is not 0 (i.e. background) and the new map would replace this with 0 then we should keep the original value, in all other cases we should overwrite the values with the
            # new ones from the second shard

            # since segmentations resulting from cellpose are not necessarily deterministic we can not do this lookup on a pixel by pixel basis but need to edit
            # the segmentation mask to remove unwanted shapes before merging

            start_time_step1 = time.time()

            # assumptions: if a nucleus is discarded the cytosol must also be discarded and vice versa otherwise this could leave cells with only one of the two
            ids_discard = set(
                np.unique(orig_input[np.where((orig_input != 0) & (shifted_map != 0))])
            )  # gets all ids that potentially need to be discarded over both masks

            # we dont want to discard any id's' that touch the edge of the image
            # for these ids we can not ensure when processing this shard that the entire cell is present in the shard
            # if we would delete these ids we would potentially be deleting half of a cell
            edge_labels = set(_return_edge_labels(orig_input))
            ids_discard = ids_discard - edge_labels

            # set ids to 0 that we dont want to keep that are already in the final map
            orig_input[np.isin(orig_input, list(ids_discard))] = 0

            time_step1 = time.time() - start_time_step1

            start_time_step2 = time.time()

            # identify all ids from the new map that potentially could lead to problems
            problematic_ids = set(np.unique(shifted_map[np.where((orig_input != 0) & (shifted_map != 0))]))

            # remove all potentially problematic ids from the new map
            shifted_map[np.isin(shifted_map, list(problematic_ids))] = 0

            time_step2 = time.time() - start_time_step2

            if self.deep_debug:
                orig_input_manipulation = orig_input.copy()
                shifted_map_manipulation = shifted_map.copy()

                orig_input_manipulation[orig_input_manipulation > 0] = 1
                shifted_map_manipulation[shifted_map_manipulation > 0] = 2

                resulting_map = orig_input_manipulation + shifted_map_manipulation

                for _mask_ix in range(self.method.N_MASKS):
                    plt.figure()
                    plt.imshow(resulting_map[_mask_ix])
                    plt.title(
                        f"Combined segmentation mask {self.method.MASK_NAMES[_mask_ix]} after\n resolving sharding for region {i}"
                    )
                    plt.colorbar()
                    plt.show()
                    plt.savefig(
                        f"{self.directory}/combined_segmentation_mask_{self.method.MASK_NAMES[_mask_ix]}_{i}.png"
                    )

            start_time_step3 = time.time()
            shifted_map = np.where((orig_input != 0) & (shifted_map == 0), orig_input, shifted_map)

            time_step3 = time.time() - start_time_step3
            total_time = time_step1 + time_step2 + time_step3

            self.log(f"Time taken to cleanup overlapping shard regions for shard {i}: {total_time}s")

            # potential issue: this does not check if we create a cytosol without a matching nucleus? But this should have been implemented in altanas segmentation method
            # for other segmentation methods this could cause issues?? Potentially something to revisit in the future

            class_id_shift += np.max(shifted_map)  # get highest existing cell id and add it to the shift
            unique_ids = set(np.unique(shifted_map[0])[1:])  # get unique cellids in the shifted map

            # save results to hdf_labels
            hdf_labels[:, window[0], window[1]] = shifted_map

            # updated classes list
            filtered_classes_combined = (
                filtered_classes_combined - set(ids_discard)
            )  # ensure that the deleted ids are also removed from the classes list (class list is always in reference to the nucleus id!)
            filtered_classes_combined = filtered_classes_combined.union(
                unique_ids
            )  # get unique nucleus ids and add them to the combined filtered class

            if self.debug:
                self.log(f"Number of classes contained in shard after processing: {len(unique_ids)}")
                self.log(f"Number of Ids in filtered_classes after adding shard {i}: {len(filtered_classes_combined)}")

            # check if filtering of classes has been successfull
            # ideally the classes contained in the mask should be identical with those contained in the classes file
            # if this is not the case it is worth investigating and there it can be helpful to see which classes are contained in the mask but not in the classes file and vice versa
            if self.deep_debug:
                masks = hdf_labels[:, :, :]
                unique_ids = set(np.unique(masks[0])) - {0}
                self.log(f"Total number of classes in final segmentation after processing: {len(unique_ids)}")

                difference_classes = filtered_classes_combined.difference(unique_ids)
                self.log(
                    f"Classes contained in classes list that are not found in the segmentation mask: {difference_classes}"
                )

                difference_masks = unique_ids.difference(filtered_classes_combined)
                self.log(
                    f"Classes contained in segmentation mask that are not contained in classes list: {difference_masks}"
                )

                if len(difference_masks) > 0:
                    _masks = np.isin(masks[0], list(difference_masks))
                    plt.figure()
                    plt.imshow(_masks)
                    plt.set_title(
                        f"Classes in segmentation mask that are not contained in classes list after processing shard {i}"
                    )
                    plt.axis("off")
                    plt.show()

            local_hf.close()
            self.log(f"Finished stitching tile {i} in {time.time() - timer} seconds.")

            # remove background class
            filtered_classes_combined = filtered_classes_combined - {0}

            self.log(f"Number of filtered classes in Dataset: {len(filtered_classes_combined)}")

            # check filtering classes to ensure that segmentation run is properly tagged
            self._check_filter_status()

            # save newly generated class list
            self._save_classes(list(filtered_classes_combined))

            # ensure cleanup
            self.clear_temp_dir()

        self.log("resolved sharding plan.")

        # save final segmentation to sdata
        self._save_segmentation_sdata(hdf_labels, list(filtered_classes_combined), masks=self.method.MASK_NAMES)

        self.log("finished saving segmentation results to sdata object for sharded segmentation.")

        if not self.deep_debug:
            self._cleanup_shards(sharding_plan)

    def _initializer_function(self, gpu_id_list, n_processes):
        current_process().gpu_id_list = gpu_id_list
        current_process().n_processes = n_processes

    def _perform_segmentation(self, shard_list):
        # get GPU status
        self._setup_processing()

        # run in single-threaded mode
        if self.n_processes == 1:
            for shard in shard_list:
                shard._call_as_shard()

            self.log("Finished serial segmentation")

        # run in multithreaded mode
        elif self.n_processes > 1:
            with mp.get_context(self.context).Pool(
                processes=self.n_processes,
                initializer=self._initializer_function,
                initargs=[self.gpu_id_list, self.n_processes],
            ) as pool:
                list(
                    tqdm(
                        pool.imap(self.method._call_as_shard, shard_list),
                        total=len(shard_list),
                        desc="Segmenting Image Tiles",
                    )
                )
                pool.close()
                pool.join()
            self.log("Finished parallel segmentation")

    def process(self, input_image):
        """Process the input image with the sharded segmentation method.

        Important:
            This function is called automatically when a Segmentation Class is executed.


        Args:
            input_image (np.array): Input image to be processed. The input image should be a numpy array of shape (C, H, W) where C is the number of channels, H is the height of the image and W is the width of the image.

        """

        total_time_start = timeit.default_timer()

        start_transform = timeit.default_timer()

        # get proper level and shape of input image

        input_image = self._transform_input_image(input_image)
        input_image = self._select_relevant_channels(input_image)

        transform_time = timeit.default_timer() - start_transform

        self.input_image_path = self.filehandler._load_input_image_to_memmap(
            image=input_image, tmp_dir_abs_path=self._tmp_dir_path
        )
        self._clear_cache(vars_to_delete=[input_image])
        input_image = tempmmap.mmap_array_from_path(self.input_image_path)

        self.log("Mapped input image to memory-mapped array.")

        self.image_size = input_image.shape[1:]

        if self.deep_debug:
            self.log(
                f"Input image of dtype {type(input_image)} with dimensions {input_image.shape} passed to sharded segmentation method."
            )

        self.shard_directory = os.path.join(self.directory, self.DEFAULT_TILES_FOLDER)

        if not os.path.isdir(self.shard_directory):
            os.makedirs(self.shard_directory)
            self.log("Created new shard directory " + self.shard_directory)

        start_sharding = timeit.default_timer()

        # get sharding plan
        sharding_plan = self._get_sharding_plan(overwrite=self.overwrite)

        # generate shard list
        shard_list = self._initialize_shard_list(sharding_plan)
        self.log(
            f"sharding plan with {len(sharding_plan)} elements generated, sharding with {self.config['threads']} threads begins"
        )

        stop_sharding = timeit.default_timer()
        sharding_time = stop_sharding - start_sharding

        start_segmentation = timeit.default_timer()

        # perform segmentation
        self._perform_segmentation(shard_list)

        stop_segmentation = timeit.default_timer()
        segmentation_time = stop_segmentation - start_segmentation

        self._clear_cache(vars_to_delete=[shard_list])

        start_resolving = timeit.default_timer()

        self._resolve_sharding(sharding_plan)

        stop_resolving = timeit.default_timer()
        resolving_time = stop_resolving - start_resolving

        total_time_stop = timeit.default_timer()

        total_time = total_time_stop - total_time_start

        self.log(f"Total time taken for sharded segmentation: {total_time} seconds")

        # make sure to cleanup temp directories
        self.log("=== finished sharded segmentation === ")

        self._save_benchmarking_times(
            image_size=input_image.shape,
            transform_time=transform_time,
            segmentation_time=segmentation_time,
            total_time=total_time,
            max_shard_size=self.config["shard_size"],
            sharding_time=sharding_time,
            shard_resolving_time=resolving_time,
            time_per_shard=(total_time / len(sharding_plan)),
        )

    def complete_segmentation(self, input_image, force_run=False):
        """Complete an already started sharded segmentation of the provided input image.

        Args:
            input_image (np.array): Input image to be processed. The input image should be a numpy array of shape (C, H, W) where C is the number of channels, H is the height of the image and W is the width of the image.
            force_run (bool): If set to True the segmentation will be run even if a completed segmentation is already found in the sdata object. Default is False.
        """

        self.shard_directory = os.path.join(self.directory, self.DEFAULT_TILES_FOLDER)

        if "_tmp_dir_path" not in self.__dict__.keys():
            self.create_temp_dir()

        # check status of sdata object
        self.filehandler._check_sdata_status()

        if self.filehandler.nuc_seg_status:
            self.log("Nucleus segmentation already exists in sdata object.")
        if self.filehandler.cyto_seg_status:
            self.log("Cytosol segmentation already exists in sdata object.")

        # check to make sure that the shard directory exisits, if not exit and return error
        if not os.path.isdir(self.shard_directory):
            if not self.filehandler.nuc_seg_status or not self.filehandler.cyto_seg_status:
                raise FileNotFoundError(
                    "No shard directory found. Please run project.shard_segmentation() to generate a sharding instead of project.complete_segmentation()."
                )
            else:
                raise ValueError(
                    "Completed segmentation found and no shard directory found. Unclear what processing should be performed. If you believe this to be an error please send your example to the developers."
                )
        else:
            if not force_run:
                if self.filehandler.nuc_seg_status or self.filehandler.cyto_seg_status:
                    raise ValueError(
                        "Completed segmentation found in addition to shard directory. If you want to force overwrite of these segmentations with the sharded segmentation results found in the shard directory please rerurn the project.complete_segmentation() method with the force_run flag set to True."
                    )

        # check to see which tiles are incomplete
        tile_directories = os.listdir(self.shard_directory)
        incomplete_indexes = []
        for tile in tile_directories:
            if not os.path.isfile(f"{self.shard_directory}/{tile}/classes.csv"):
                incomplete_indexes.append(int(tile))
                self.log(f"Shard with ID {tile} not completed.")

        # get input image size
        input_image = self._transform_input_image(input_image)
        input_image = self._select_relevant_channels(input_image)

        self.input_image_path = self.filehandler._load_input_image_to_memmap(
            image=input_image, tmp_dir_abs_path=self._tmp_dir_path
        )
        self._clear_cache(vars_to_delete=[input_image])

        input_image = tempmmap.mmap_array_from_path(self.input_image_path)
        self.log("Mapped input image to memory-mapped array.")

        self.image_size = input_image.shape[1:]

        # load sharding plan
        sharding_plan = self._get_sharding_plan(overwrite=False, force_read=True)

        # check to make sure that calculated sharding plan matches to existing sharding results
        assert (
            len(sharding_plan) == len(tile_directories)
        ), "Calculated a different number of shards than found shard directories. This indicates a mismatch between the current loaded config file and the config file used to generate the exisiting partial segmentation. Please rerun the complete segmentation to ensure accurate results."

        # select only those shards that did not complete successfully for further processing
        sharding_plan_complete = sharding_plan

        if len(incomplete_indexes) > 0:
            # adjust current sharding plan to only contain incomplete elements
            sharding_plan = [shard for i, shard in enumerate(sharding_plan) if i in incomplete_indexes]

            self.log(f"Adjusted sharding plan to only proceed with the {len(incomplete_indexes)} incomplete shards.")

            shard_list = self._initialize_shard_list(sharding_plan)
            self.log(
                f"sharding plan with {len(sharding_plan)} elements generated, sharding with {self.config['threads']} threads begins"
            )

            # perform segmentation
            self._perform_segmentation(shard_list)
            self._clear_cache(vars_to_delete=[shard_list])

        self._resolve_sharding(sharding_plan_complete)
        self._cleanup_shards(sharding_plan_complete, keep_plots=False)

        self.log("=== completed sharded segmentation === ")


#############################################
###### TIMECOURSE/BATCHED METHODS ###########
#############################################

## These functions have not yet been adapted to the new spatial data format and are not yet functional.

# class TimecourseSegmentation(Segmentation):
#     """Segmentation helper class used for creating segmentation workflows working with timecourse data."""

#     DEFAULT_SEGMENTATION_FILE = "input_segmentation.h5"
#     DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"
#     PRINT_MAPS_ON_DEBUG = True
#     DEFAULT_CHANNELS_NAME = "input_images"
#     DEFAULT_MASK_NAME = "segmentation"

#     channel_colors = [
#         "#e60049",
#         "#0bb4ff",
#         "#50e991",
#         "#e6d800",
#         "#9b19f5",
#         "#ffa300",
#         "#dc0ab4",
#         "#b3d4ff",
#         "#00bfa0",
#     ]

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.index = None
#         self.input_path = None

#         if not hasattr(self, "method"):
#             raise AttributeError(
#                 "No BaseSegmentationType defined, please set attribute ``BaseSegmentationMethod``"
#             )

#     def initialize_as_shard(self, index, input_path, _tmp_seg_path):
#         """Initialize Segmentation Step with further parameters needed for federated segmentation.

#         Important:
#             This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

#         Args:
#             index (int): Unique indexes of the elements that need to be segmented.
#             input_path (str): Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.
#         """
#         self.index = index
#         self.input_path = input_path
#         self._tmp_seg_path = _tmp_seg_path
#         self.create_temp_dir()

#     def call_as_shard(self):
#         """Wrapper function for calling a sharded segmentation.

#         Important:
#             This function is intended for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.

#         """
#         with h5py.File(self.input_path, "r") as hf:
#             hdf_input = hf.get(self.DEFAULT_CHANNELS_NAME)

#             if isinstance(self.index, int):
#                 self.index = [self.index]

#             for index in self.index:
#                 self.current_index = index
#                 input_image = hdf_input[index, :, :, :]

#                 self.log(f"Segmentation on index {index} started.")
#                 try:
#                     super().__call__(input_image)
#                     self.clear_temp_dir()
#                 except Exception:
#                     self.log(traceback.format_exc())
#                     self.clear_temp_dir()
#                 self.log(f"Segmentation on index {index} completed.")

#     def save_segmentation(
#         self,
#         input_image,
#         labels,
#         classes,
#     ):
#         """
#         Saves the results of a segmentation at the end of the process by transferring it to the initialized
#         memory mapped array

#         Args:
#             labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.
#             classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.
#         """
#         # reconnect to existing HDF5 for memory mapping segmentation results
#         _tmp_seg = tempmmap.mmap_array_from_path(self._tmp_seg_path)

#         # size (C, H, W) is expected
#         # dims are expanded in case (H, W) is passed
#         labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels
#         classes = np.array(list(classes))

#         self.log(f"transferring {self.current_index} to temmporray memory mapped array")
#         _tmp_seg[self.current_index] = labels

#         # close connect to temmpmmap file again
#         del _tmp_seg

#     def _initialize_tempmmap_array(self):
#         # create an empty HDF5 file prepared for using as a memory mapped temp array to save segmentation results to
#         # this required when trying to segment so many images that the results can no longer fit into memory
#         _tmp_seg_path = tempmmap.create_empty_mmap(
#             shape=self.shape_segmentation,
#             dtype=self.DEFAULT_SEGMENTATION_DTYPE,
#             tmp_dir_abs_path=self._tmp_dir_path,
#         )
#         self._tmp_seg_path = _tmp_seg_path

#     def _transfer_tempmmap_to_hdf5(self):
#         _tmp_seg = tempmmap.mmap_array_from_path(self._tmp_seg_path)
#         input_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_FILE)

#         # create hdf5 datasets with temp_arrays as input
#         with h5py.File(input_path, "a") as hf:
#             # check if dataset already exists if so delete and overwrite
#             if self.DEFAULT_MASK_NAME in hf.keys():
#                 del hf[self.DEFAULT_MASK_NAME]
#                 self.log(
#                     "segmentation dataset already existe in hdf5, deleted and overwritten."
#                 )
#             hf.create_dataset(
#                 self.DEFAULT_MASK_NAME,
#                 shape=_tmp_seg.shape,
#                 chunks=(1, 2, self.shape_input_images[2], self.shape_input_images[3]),
#                 dtype=self.DEFAULT_SEGMENTATION_DTYPE,
#             )

#             # using this loop structure ensures that not all results are loaded in memory at any one timepoint
#             for i in range(_tmp_seg.shape[0]):
#                 hf[self.DEFAULT_MASK_NAME][i] = _tmp_seg[i]

#             dt = h5py.special_dtype(vlen=self.DEFAULT_SEGMENTATION_DTYPE)

#             if "classes" in hf.keys():
#                 del hf["classes"]
#                 self.log(
#                     "classes dataset already existed in hdf5, deleted and overwritten."
#                 )

#             hf.create_dataset(
#                 "classes",
#                 shape=self.shape_classes,
#                 maxshape=(None),
#                 chunks=None,
#                 dtype=dt,
#             )

#         gc.collect()

#     def save_image(self, array, save_name="", cmap="magma", **kwargs):
#         if np.issubdtype(array.dtype.type, np.integer):
#             self.log(f"{save_name} will be saved as tif")
#             data = array.astype(np.uint16)
#             im = Image.fromarray(data)
#             im.save(f"{save_name}.tif")

#         fig = plt.figure(frameon=False)
#         fig.set_size_inches((10, 10))
#         ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
#         ax.set_axis_off()

#         fig.add_axes(ax)
#         ax.imshow(array, cmap=cmap, **kwargs)

#         if save_name != "":
#             plt.savefig(f"{save_name}.png")
#             plt.show()
#             plt.close()

#     def adjust_segmentation_indexes(self):
#         """
#         The function iterates over all present segmented files and adjusts the indexes so that they are unique throughout.
#         """

#         self.log("resolve segmentation indexes")

#         path = os.path.join(self.directory, self.DEFAULT_INPUT_IMAGE_NAME)

#         with h5py.File(path, "a") as hf:
#             hdf_labels = hf.get(self.DEFAULT_MASK_NAME)
#             hdf_classes = hf.get("classes")

#             class_id_shift = 0
#             filtered_classes_combined = []
#             edge_classes_combined = []

#             for i in tqdm(
#                 range(0, hdf_labels.shape[0]),
#                 total=hdf_labels.shape[0],
#                 desc="Adjusting Indexes",
#             ):
#                 individual_hdf_labels = hdf_labels[i, :, :, :]
#                 num_shapes = np.max(individual_hdf_labels)
#                 shifted_map, edge_labels = shift_labels(
#                     individual_hdf_labels, class_id_shift, return_shifted_labels=True
#                 )
#                 hdf_labels[i, :, :] = shifted_map

#                 if set(np.unique(shifted_map[0])) != set(np.unique(shifted_map[1])):
#                     self.log(
#                         "Warning: Different classes in different segmentatio channels. Please report this example to the developers"
#                     )
#                     self.log("set1 nucleus: set(np.unique(shifted_map[0]))")
#                     self.log("set2 cytosol: set(np.unique(shifted_map[1]))")

#                     self.log(
#                         f"{set(np.unique(shifted_map[1]))- set(np.unique(shifted_map[0]))} not in nucleus mask"
#                     )
#                     self.log(
#                         f"{set(np.unique(shifted_map[0]))- set(np.unique(shifted_map[1]))} not in cytosol mask"
#                     )

#                 filtered_classes = set(np.unique(shifted_map[0])) - set(
#                     [0]
#                 )  # remove background class
#                 final_classes = list(filtered_classes - set(edge_labels))

#                 hdf_labels[i, :, :] = shifted_map
#                 hdf_classes[i] = np.array(final_classes, dtype=self.DEFAULT_SEGMENTATION_DTYPE).reshape(
#                     1, 1, -1
#                 )

#                 # save all cells in general
#                 filtered_classes_combined.extend(filtered_classes)
#                 edge_classes_combined.extend(edge_labels)

#                 # adjust class_id shift
#                 class_id_shift += num_shapes

#             edge_classes_combined = set(edge_classes_combined)
#             classes_after_edges = list(
#                 set(filtered_classes_combined) - edge_classes_combined
#             )

#             self.log("Number of filtered classes combined after segmentation:")
#             self.log(len(filtered_classes_combined))

#             self.log("Number of classes in contact with image edges:")
#             self.log(len(edge_classes_combined))

#             self.log("Number of classes after removing image edges:")
#             self.log(len(classes_after_edges))

#             # save newly generated class list
#             self.save_classes(classes_after_edges)

#             # sanity check of class reconstruction
#             if self.debug:
#                 all_classes = set(hdf_labels[:].flatten())
#                 if set(edge_classes_combined).issubset(set(all_classes)):
#                     self.log(
#                         "Sharding sanity check: edge classes are a full subset of all classes"
#                     )
#                 elif len(set(all_classes)) - len(set(edge_classes_combined)) == len(
#                     set(classes_after_edges)
#                 ):
#                     self.log(
#                         "Sharding sanity check: sum of edge classes and classes after edges is equal to all classes."
#                     )
#                 else:
#                     self.log(
#                         "Sharding sanity check: edge classes are NOT a full subset of all classes."
#                     )

#         self.log("resolved segmentation list")

#     def process(self):
#         input_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_FILE)

#         with h5py.File(input_path, "r") as hf:
#             input_images = hf.get(self.DEFAULT_CHANNELS_NAME)
#             indexes = list(range(0, input_images.shape[0]))

#             # initialize segmentation dataset
#             self.shape_input_images = input_images.shape
#             self.shape_segmentation = (
#                 input_images.shape[0],
#                 self.config["output_masks"],
#                 input_images.shape[2],
#                 input_images.shape[3],
#             )
#             self.shape_classes = input_images.shape[0]

#         # initialize temp object to write segmentations too
#         self._initialize_tempmmap_array()

#         self.log("Beginning segmentation without multithreading.")

#         # initialzie segmentation objects
#         current_shard = self.method(
#             self.config,
#             self.directory,
#             project_location=self.project_location,
#             debug=self.debug,
#             overwrite=self.overwrite,
#         )

#         current_shard.initialize_as_shard(
#             indexes, input_path=input_path, _tmp_seg_path=self._tmp_seg_path
#         )

#         # calculate results for each shard
#         results = [current_shard.call_as_shard()]

#         # save results to hdf5
#         self.log("Writing segmentation results to .hdf5 file.")
#         self._transfer_tempmmap_to_hdf5()

#         # adjust segmentation indexes
#         self.adjust_segmentation_indexes()
#         self.log("Adjusted Indexes.")


# class MultithreadedSegmentation(TimecourseSegmentation):
#     DEFAULT_SEGMENTATION_FILE = "input_segmentation.h5"
#     DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         if not hasattr(self, "method"):
#             raise AttributeError(
#                 "No Segmentation method defined, please set attribute ``method``"
#             )

#     def initializer_function(self, gpu_id_list):
#         current_process().gpu_id_list = gpu_id_list

#     def process(self):
#         input_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_FILE)

#         with h5py.File(input_path, "r") as hf:
#             input_images = hf.get(self.DEFAULT_CHANNELS_NAME)
#             indexes = list(range(0, input_images.shape[0]))

#             # initialize segmentation dataset
#             self.shape_input_images = input_images.shape
#             self.shape_segmentation = (
#                 input_images.shape[0],
#                 self.config["output_masks"],
#                 input_images.shape[2],
#                 input_images.shape[3],
#             )
#             self.shape_classes = input_images.shape[0]

#         # initialize temp object to write segmentations too
#         self._initialize_tempmmap_array()
#         segmentation_list = self.initialize_shard_list(
#             indexes, input_path=input_path, _tmp_seg_path=self._tmp_seg_path
#         )

#         # make more verbose output for troubleshooting and timing purposes.
#         n_threads = self.config["threads"]
#         self.log(f"Beginning segmentation with {n_threads} threads.")
#         self.log(f"A total of {len(segmentation_list)} processes need to be executed.")

#         # check that that number of GPUS is actually available
#         if "nGPUS" not in self.config.keys():
#             self.config["nGPUs"] = torch.cuda.device_count()

#         nGPUS = self.config["nGPUs"]
#         available_GPUs = torch.cuda.device_count()
#         processes_per_GPU = self.config["threads"]

#         if available_GPUs < self.config["nGPUs"]:
#             self.log(f"Found {available_GPUs} but {nGPUS} specified in config.")

#         if available_GPUs >= 1:
#             n_processes = processes_per_GPU * available_GPUs
#         else:
#             n_processes = self.config["threads"]
#             available_GPUs = (
#                 1  # default to 1 GPU if non are available and a CPU only method is run
#             )

#         # initialize a list of available GPUs
#         gpu_id_list = []
#         for gpu_ids in range(available_GPUs):
#             for _ in range(processes_per_GPU):
#                 gpu_id_list.append(gpu_ids)

#         self.log(f"Beginning segmentation on {available_GPUs} available GPUs.")

#         with mp.get_context(self.context).Pool(
#             processes=n_processes,
#             initializer=self.initializer_function,
#             initargs=[gpu_id_list],
#         ) as pool:
#             results = list(
#                 tqdm(
#                     pool.imap(self.method.call_as_shard, segmentation_list),
#                     total=len(indexes),
#                 )
#             )
#             pool.close()
#             pool.join()
#             print("All segmentations are done.", flush=True)

#         self.log("Finished parallel segmentation")
#         self.log("Transferring results to array.")

#         self._transfer_tempmmap_to_hdf5()
#         self.adjust_segmentation_indexes()
#         self.log("Adjusted Indexes.")

#         # cleanup variables to make sure memory is cleared up again
#         del results
#         gc.collect()

#     def initialize_shard_list(self, segmentation_list, input_path, _tmp_seg_path):
#         _shard_list = []

#         for i in tqdm(
#             segmentation_list, total=len(segmentation_list), desc="Generating Shards"
#         ):
#             current_shard = self.method(
#                 self.config,
#                 self.directory,
#                 project_location=self.project_location,
#                 debug=self.debug,
#                 overwrite=self.overwrite,
#             )

#             current_shard.initialize_as_shard(
#                 i, input_path, _tmp_seg_path=_tmp_seg_path
#             )
#             _shard_list.append(current_shard)

#         self.log(f"Shard list created with {len(_shard_list)} elements.")

#         return _shard_list

#     def get_output(self):
#         return os.path.join(self.directory, self.DEFAULT_SEGMENTATION_FILE)
