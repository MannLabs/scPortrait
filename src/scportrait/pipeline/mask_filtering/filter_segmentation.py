import csv
import gc
import os
import shutil
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from alphabase.io import tempmmap
from tqdm.auto import tqdm

from scportrait.pipeline._base import ProcessingStep
from scportrait.pipeline._utils.segmentation import sc_any


class SegmentationFilter(ProcessingStep):
    """SegmentationFilter helper class used for creating workflows to filter generated segmentation masks before extraction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.identifier: int | None = None
        self.window: tuple[slice, slice] | None = None
        self.input_path: str | Path | None = None

    def read_input_masks(self, input_path: str | Path) -> npt.NDArray[np.uint16]:
        """Read input masks from a given HDF5 file.

        Args:
            input_path: Path to the HDF5 file containing the input masks.

        Returns:
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

    def save_classes(self, classes: dict[str, list[Any]]) -> None:
        """Save the filtered classes to a CSV file.

        Args:
            classes: Dictionary of classes to save.
        """
        filtered_path = Path(self.directory) / self.DEFAULT_REMOVED_CLASSES_FILE
        to_write = "\n".join([f"{str(x)}:{str(y)}" for x, y in classes.items()])
        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)
        self.log(f"Saved nucleus_id:cytosol_id matchings of all cells that passed filtering to {filtered_path}.")

    def initialize_as_tile(
        self, identifier: int, window: tuple[slice, slice], input_path: str | Path, zarr_status: bool = True
    ) -> None:
        """Initialize Filtering Step with further parameters needed for filtering segmentation results.

        Important:
            This function is intended for internal use by the :class:`TiledFilterSegmentation` helper class.
            In most cases it is not relevant to the creation of custom filtering workflows.

        Args:
            identifier: Unique index of the tile.
            window: Defines the window which is assigned to the tile. The window will be applied
                   to the input. The first element refers to the first dimension of the image and so on.
            input_path: Location of the input HDF5 file. During tiled segmentation the
                       :class:`TiledSegmentation` derived helper class will save the input image
                       in form of a HDF5 file.
            zarr_status: Status of zarr saving, by default True.
        """
        self.identifier = identifier
        self.window = window
        self.input_path = input_path
        self.save_zarr = zarr_status

    def call_as_tile(self) -> None:
        """Wrapper function for calling segmentation filtering on an image tile.

        Important:
            This function is intended for internal use by the :class:`TiledSegmentation` helper class.
            In most cases it is not relevant to the creation of custom segmentation workflows.
        """
        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get("labels")
            if hdf_input is None:
                raise ValueError("No 'labels' dataset found in HDF5 file")

            c, _, _ = hdf_input.shape
            x1 = self.window[0].start
            x2 = self.window[0].stop
            y1 = self.window[1].start
            y2 = self.window[1].stop

            if any(v is None for v in [x1, x2, y1, y2]):
                raise ValueError("Window slice boundaries cannot be None")

            x = x2 - x1
            y = y2 - y1

            input_image = tempmmap.array(shape=(2, x, y), dtype=np.uint16, tmp_dir_abs_path=self._tmp_dir_path)
            input_image = hdf_input[:2, self.window[0], self.window[1]]

        if sc_any(input_image):
            try:
                self.log(f"Beginning filtering on tile in position [{self.window[0]}, {self.window[1]}]")
                super().__call__(input_image)
            except (OSError, ValueError, RuntimeError) as e:
                self.log(f"An error occurred: {e}")
                self.log(traceback.format_exc())
        else:
            print(f"Tile in position [{self.window[0]}, {self.window[1]}] only contained zeroes.")
            try:
                super().__call_empty__(input_image)
            except (OSError, ValueError, RuntimeError) as e:
                self.log(f"An error occurred: {e}")
                self.log(traceback.format_exc())

        del input_image
        gc.collect()

        self.log(f"Writing out window location to file at {self.directory}/window.csv")
        with open(f"{self.directory}/window.csv", "w") as f:
            f.write(f"{self.window}\n")
        self.log(f"Filtering of tile with the slicing {self.window} finished.")

    def get_output_path(self) -> Path:
        return Path(self.directory) / self.DEFAULT_SEGMENTATION_FILE


class TiledSegmentationFilter(SegmentationFilter):
    """TiledSegmentationFilter helper class used for creating workflows to filter generated segmentation masks using a tiled approach."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self, "method"):
            raise AttributeError("No SegmentationFilter method defined, please set attribute ``method``")
        self.tile_directory: Path | None = None

    def initialize_tile_list(
        self, tileing_plan: list[tuple[slice, slice]], input_path: str | Path
    ) -> list[SegmentationFilter]:
        """Initialize the list of tiles for segmentation filtering.

        Args:
            tileing_plan: List of windows defining the tiling plan.
            input_path: Path to the input HDF5 file.

        Returns:
            List of initialized tiles.
        """
        _tile_list = []
        self.input_path = input_path

        for i, window in enumerate(tileing_plan):
            local_tile_directory = Path(self.tile_directory) / str(i)
            current_tile = self.method(  # type: ignore
                self.config,
                local_tile_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,  # type: ignore
            )
            current_tile.initialize_as_tile(i, window, self.input_path, zarr_status=False)  # type: ignore
            _tile_list.append(current_tile)

        return _tile_list

    def initialize_tile_list_incomplete(
        self, tileing_plan: list[tuple[slice, slice]], incomplete_indexes: list[int], input_path: str | Path
    ) -> list[SegmentationFilter]:
        """Initialize the list of incomplete tiles for segmentation filtering.

        Args:
            tileing_plan: List of windows defining the tiling plan.
            incomplete_indexes: List of indexes for incomplete tiles.
            input_path: Path to the input HDF5 file.

        Returns:
            List of initialized incomplete tiles.
        """
        _tile_list = []
        self.input_path = input_path

        for i, window in zip(incomplete_indexes, tileing_plan, strict=False):
            local_tile_directory = Path(self.tile_directory) / str(i)
            current_tile = self.method(  # type: ignore
                self.config,
                local_tile_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,  # type: ignore
            )
            current_tile.initialize_as_tile(i, window, self.input_path, zarr_status=False)  # type: ignore
            _tile_list.append(current_tile)

        return _tile_list

    def calculate_tileing_plan(self, mask_size: tuple[int, int]) -> list[tuple[slice, slice]]:
        """Calculate the tiling plan based on the mask size.

        Args:
            mask_size: Size of the mask.

        Returns:
            List of windows defining the tiling plan.
        """
        tileing_plan_path = f"{self.directory}/tileing_plan.csv"

        if Path(tileing_plan_path).is_file():
            self.log(f"tileing plan already found in directory {tileing_plan_path}.")
            if self.overwrite:
                self.log("Overwriting existing tileing plan.")
                Path(tileing_plan_path).unlink()
            else:
                self.log("Reading existing tileing plan from file.")
                with Path(tileing_plan_path).open() as f:
                    tileing_plan = [eval(line) for line in f.readlines()]
                return tileing_plan

        _tileing_plan: list[tuple[slice, slice]] = []
        side_size = np.floor(np.sqrt(int(self.config["tile_size"])))
        tiles_side = np.round(np.array(mask_size) / side_size).astype(int)
        tile_size = np.array(mask_size) // tiles_side

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

                _tileing_plan.append((slice(lower_y, upper_y, None), slice(lower_x, upper_x, None)))

        with open(tileing_plan_path, "w") as f:
            for item in _tileing_plan:
                f.write(f"{item}\n")

        return _tileing_plan

    def execute_tile_list(self, tile_list: list[SegmentationFilter], n_cpu: int | None = None) -> list[Path]:
        """Execute the filtering process for a list of tiles.

        Args:
            tile_list: List of tiles to process.
            n_cpu: Number of CPU cores to use, by default None.

        Returns:
            List of output file paths for the processed tiles.
        """

        def f(x: SegmentationFilter) -> Path:
            try:
                x.call_as_tile()
            except (OSError, ValueError, RuntimeError) as e:
                self.log(f"An error occurred: {e}")
                self.log(traceback.format_exc())
            return x.get_output_path()

        if n_cpu == 1:
            self.log(f"Running sequentially on {n_cpu} CPU")
            return list(map(f, tile_list))
        else:
            n_processes = n_cpu if n_cpu else os.cpu_count()
            self.log(f"Running in parallel on {n_processes} CPUs")
            with Pool(n_processes) as pool:
                return list(pool.imap(f, tile_list))

    def execute_tile(self, tile: SegmentationFilter) -> Path:
        """Execute the filtering process for a single tile.

        Args:
            tile: Tile to process.

        Returns:
            Output file path for the processed tile.
        """
        tile.call_as_tile()
        return tile.get_output_path()

    def initialize_tile_directory(self) -> None:
        """Initialize the directory for storing tile outputs."""
        self.tile_directory = Path(self.directory) / self.DEFAULT_TILES_FOLDER
        if self.tile_directory.exists():
            self.log(f"Directory {self.tile_directory} already exists.")
            if self.overwrite:
                self.log("Overwriting existing tiles folder.")
                shutil.rmtree(self.tile_directory)
                self.tile_directory.mkdir()
            else:
                os.makedirs(self.tile_directory)
        else:
            self.tile_directory.mkdir()

    def collect_results(self) -> npt.NDArray[np.uint16]:
        """Collect the results from the processed tiles.

        Returns:
            Array containing the combined results from all tiles.
        """
        self.log("Reading in all tile results")
        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get("labels")
            if hdf_input is None:
                raise ValueError("No 'labels' dataset found in HDF5 file")
            c, y, x = hdf_input.shape

        self.log(f"Output image will have shape {c, y, x}")
        output_image = np.zeros((c, y, x), dtype=np.uint16)
        classes: defaultdict[str, list[Any]] = defaultdict(list)

        with open(f"{self.directory}/window.csv") as f:
            _window_locations = [eval(line.strip()) for line in f.readlines()]

        self.log(f"Expecting {len(_window_locations)} tiles")

        for i, loc in tqdm(enumerate(_window_locations), total=len(_window_locations)):
            out_dir = os.path.join(self.tile_directory, str(i))
            with h5py.File(f"{out_dir}/segmentation.h5", "r") as hf:
                data = hf.get("labels")
                if data is None:
                    raise ValueError(f"No 'labels' dataset found in tile {i}")
                for cls, mappings in csv.reader(open(f"{out_dir}/filtered_classes.csv")):
                    classes[cls].append(mappings)
                output_image[:, loc[0], loc[1]] = data[:, :]

        self.save_classes(classes)

        return output_image
