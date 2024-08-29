import os
import gc
import sys
import numpy as np
import csv
import h5py
from tqdm.auto import tqdm
import shutil
from collections import defaultdict
import traceback

from multiprocessing import Pool

from scportrait.pipeline._utils.segmentation import sc_any
from scportrait.pipeline._base import ProcessingStep

from alphabase.io import tempmmap

class SegmentationFilter(ProcessingStep):
    """SegmentationFilter helper class used for creating workflows to filter generated segmentation masks before extraction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identifier = None
        self.window = None
        self.input_path = None

    def read_input_masks(self, input_path):
        """
        Read input masks from a given HDF5 file.

        Parameters
        ----------
        input_path : str
            Path to the HDF5 file containing the input masks.

        Returns
        -------
        numpy.ndarray
            Array containing the input masks.
        """
        with h5py.File(input_path, "r") as hf:
            hdf_input = hf.get("labels")
            input_masks = tempmmap.array(
                shape=hdf_input.shape,
                dtype=np.uint16,
                tmp_dir_abs_path=self._tmp_dir_path,
            )
            input_masks = hdf_input[:2, :, :]
        return input_masks

    def save_classes(self, classes):
        """
        Save the filtered classes to a CSV file.

        Parameters
        ----------
        classes : dict
            Dictionary of classes to save.
        """
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTERED_CLASSES_FILE)
        to_write = "\n".join([f"{str(x)}:{str(y)}" for x, y in classes.items()])
        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)
        self.log(f"Saved nucleus_id:cytosol_id matchings of all cells that passed filtering to {filtered_path}.")

    def initialize_as_tile(self, identifier, window, input_path, zarr_status=True):
        """
        Initialize Filtering Step with further parameters needed for filtering segmentation results.

        Important:
            This function is intended for internal use by the :class:`TiledFilterSegmentation` helper class. In most cases it is not relevant to the creation of custom filtering workflows.

        Parameters
        ----------
        identifier : int
            Unique index of the tile.
        window : list of tuple
            Defines the window which is assigned to the tile. The window will be applied to the input. The first element refers to the first dimension of the image and so on.
        input_path : str
            Location of the input HDF5 file. During tiled segmentation the :class:`TiledSegmentation` derived helper class will save the input image in form of a HDF5 file.
        zarr_status : bool, optional
            Status of zarr saving, by default True.
        """
        self.identifier = identifier
        self.window = window
        self.input_path = input_path
        self.save_zarr = zarr_status

    def call_as_tile(self):
        """
        Wrapper function for calling segmentation filtering on an image tile.

        Important:
            This function is intended for internal use by the :class:`TiledSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.
        """
        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get("labels")

            c, _, _ = hdf_input.shape
            x1 = self.window[0].start
            x2 = self.window[0].stop
            y1 = self.window[1].start
            y2 = self.window[1].stop

            x = x2 - x1
            y = y2 - y1

            input_image = tempmmap.array(
                shape=(2, x, y), dtype=np.uint16, tmp_dir_abs_path=self._tmp_dir_path
            )
            input_image = hdf_input[:2, self.window[0], self.window[1]]

        if sc_any(input_image):
            try:
                self.log(f"Beginning filtering on tile in position [{self.window[0]}, {self.window[1]}]")
                super().__call__(input_image)
            except Exception:
                self.log(traceback.format_exc())
        else:
            print(f"Tile in position [{self.window[0]}, {self.window[1]}] only contained zeroes.")
            try:
                super().__call_empty__(input_image)
            except Exception:
                self.log(traceback.format_exc())

        del input_image
        gc.collect()

        self.log(f"Writing out window location to file at {self.directory}/window.csv")
        with open(f"{self.directory}/window.csv", "w") as f:
            f.write(f"{self.window}\n")
        self.log(f"Filtering of tile with the slicing {self.window} finished.")

    def get_output(self):
        """
        Get the output file path.

        Returns
        -------
        str
            Path to the output file.
        """
        return os.path.join(self.directory, self.DEFAULT_SEGMENTATION_FILE)


class TiledSegmentationFilter(SegmentationFilter):
    """TiledSegmentationFilter helper class used for creating workflows to filter generated segmentation masks using a tiled approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "method"):
            raise AttributeError("No SegmentationFilter method defined, please set attribute ``method``")

    def initialize_tile_list(self, tileing_plan, input_path):
        """
        Initialize the list of tiles for segmentation filtering.

        Parameters
        ----------
        tileing_plan : list of tuple
            List of windows defining the tiling plan.
        input_path : str
            Path to the input HDF5 file.

        Returns
        -------
        list
            List of initialized tiles.
        """
        _tile_list = []
        self.input_path = input_path

        for i, window in enumerate(tileing_plan):
            local_tile_directory = os.path.join(self.tile_directory, str(i))
            current_tile = self.method(
                self.config,
                local_tile_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
            current_tile.initialize_as_tile(i, window, self.input_path, zarr_status=False)
            _tile_list.append(current_tile)

        return _tile_list

    def initialize_tile_list_incomplete(self, tileing_plan, incomplete_indexes, input_path):
        """
        Initialize the list of incomplete tiles for segmentation filtering.

        Parameters
        ----------
        tileing_plan : list of tuple
            List of windows defining the tiling plan.
        incomplete_indexes : list of int
            List of indexes for incomplete tiles.
        input_path : str
            Path to the input HDF5 file.

        Returns
        -------
        list
            List of initialized incomplete tiles.
        """
        _tile_list = []
        self.input_path = input_path

        for i, window in zip(incomplete_indexes, tileing_plan):
            local_tile_directory = os.path.join(self.tile_directory, str(i))
            current_tile = self.method(
                self.config,
                local_tile_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
            current_tile.initialize_as_tile(i, window, self.input_path, zarr_status=False)
            _tile_list.append(current_tile)

        return _tile_list

    def calculate_tileing_plan(self, mask_size):
        """
        Calculate the tiling plan based on the mask size.

        Parameters
        ----------
        mask_size : tuple
            Size of the mask.

        Returns
        -------
        list of tuple
            List of windows defining the tiling plan.
        """
        tileing_plan_path = f"{self.directory}/tileing_plan.csv"

        if os.path.isfile(tileing_plan_path):
            self.log(f"tileing plan already found in directory {tileing_plan_path}.")
            if self.overwrite:
                self.log("Overwriting existing tileing plan.")
                os.remove(tileing_plan_path)
            else:
                self.log("Reading existing tileing plan from file.")
                with open(tileing_plan_path, "r") as f:
                    _tileing_plan = [eval(line) for line in f.readlines()]
                    return _tileing_plan

        _tileing_plan = []
        side_size = np.floor(np.sqrt(int(self.config["tile_size"])))
        tiles_side = np.round(mask_size / side_size).astype(int)
        tile_size = mask_size // tiles_side

        self.log(f"input image {mask_size[0]} px by {mask_size[1]} px")
        self.log(f"target_tile_size: {self.config['tile_size']}")
        self.log("tileing plan:")
        self.log(f"{tiles_side[0]} rows by {tiles_side[1]} columns")
        self.log(f"{tile_size[0]} px by {tile_size[1]} px")

        for y in range(tiles_side[0]):
            for x in range(tiles_side[1]):
                last_row = y == tiles_side[0] - 1
                last_column = x == tiles_side[1] - 1

                lower_y = y * tile_size[0]
                lower_x = x * tile_size[1]

                upper_y = (y + 1) * tile_size[0]
                upper_x = (x + 1) * tile_size[1]

                if last_row:
                    upper_y = mask_size[0]

                if last_column:
                    upper_x = mask_size[1]

                _tileing_plan.append(
                    (slice(lower_y, upper_y, None), slice(lower_x, upper_x, None))
                )

        with open(tileing_plan_path, "w") as f:
            for item in _tileing_plan:
                f.write(f"{item}\n")

        return _tileing_plan

    def execute_tile_list(self, tile_list, n_cpu=None):
        """
        Execute the filtering process for a list of tiles.

        Parameters
        ----------
        tile_list : list
            List of tiles to process.
        n_cpu : int, optional
            Number of CPU cores to use, by default None.

        Returns
        -------
        list
            List of output file paths for the processed tiles.
        """
        def f(x):
            try:
                x.call_as_tile()
            except Exception:
                self.log(traceback.format_exc())
            return x.get_output()

        if n_cpu == 1:
            self.log(f"Running sequentially on {n_cpu} CPU")
            return list(map(f, tile_list))
        else:
            n_processes = n_cpu if n_cpu else os.cpu_count()
            self.log(f"Running in parallel on {n_processes} CPUs")
            with Pool(n_processes) as pool:
                return list(pool.imap(f, tile_list))

    def execute_tile(self, tile):
        """
        Execute the filtering process for a single tile.

        Parameters
        ----------
        tile : object
            Tile to process.

        Returns
        -------
        str
            Output file path for the processed tile.
        """
        tile.call_as_tile()
        return tile.get_output()

    def initialize_tile_directory(self):
        """
        Initialize the directory for storing tile outputs.
        """
        self.tile_directory = os.path.join(self.directory, self.DEFAULT_TILES_FOLDER)
        if os.path.exists(self.tile_directory):
            self.log(f"Directory {self.tile_directory} already exists.")
            if self.overwrite:
                self.log("Overwriting existing tiles folder.")
                shutil.rmtree(self.tile_directory)
                os.makedirs(self.tile_directory)
        else:
            os.makedirs(self.tile_directory)

    def collect_results(self):
        """
        Collect the results from the processed tiles.

        Returns
        -------
        numpy.ndarray
            Array containing the combined results from all tiles.
        """
        self.log("Reading in all tile results")
        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get("labels")
            c, y, x = hdf_input.shape

        self.log(f"Output image will have shape {c, y, x}")
        output_image = np.zeros((c, y, x), dtype=np.uint16)
        classes = defaultdict(list)

        with open(f"{self.directory}/window.csv", "r") as f:
            _window_locations = [eval(line.strip()) for line in f.readlines()]

        self.log(f"Expecting {len(_window_locations)} tiles")

        for i, loc in tqdm(enumerate(_window_locations), total=len(_window_locations)):
            out_dir = os.path.join(self.tile_directory, str(i))
            with h5py.File(f"{out_dir}/segmentation.h5", "r") as hf:
                data = hf.get("labels")
                for cls, mappings in csv.reader(open(f"{out_dir}/filtered_classes.csv")):
                    classes[cls].append(mappings)
                output_image[:, loc[0], loc[1]] = data[:, :]

        self.save_classes(classes)

        return output_image
