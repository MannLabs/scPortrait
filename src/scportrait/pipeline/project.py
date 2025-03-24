"""
project
=======

At the core of scPortrait is the concept of a `Project`. A `Project` is a Python class that orchestrates all scPortrait processing steps, serving as the central element for all operations.
Each `Project` corresponds to a directory on the file system, which houses the input data for a specific scPortrait run along with the generated outputs.
The choice of the appropriate `Project` class depends on the structure of the data to be processed.

For more details, refer to :ref:`here <projects>`.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Literal

import dask.array as da
import dask.array as darray
import numpy as np
import psutil
import zarr
from alphabase.io import tempmmap
from spatialdata import SpatialData
from tifffile import imread

from scportrait.io import daskmmap
from scportrait.pipeline._base import Logable
from scportrait.pipeline._utils.helper import read_config
from scportrait.pipeline._utils.sdata_io import sdata_filehandler
from scportrait.pipeline._utils.spatialdata_helper import (
    get_chunk_size,
    rechunk_image,
)
from scportrait.tools.spdata.write._helper import _get_image, _get_shape

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData
    from matplotlib.pyplot import Figure

from scportrait.io import read_h5sc
from scportrait.pipeline._utils.constants import (
    DEFAULT_CELL_ID_NAME,
    DEFAULT_CENTERS_NAME,
    DEFAULT_CHUNK_SIZE_2D,
    DEFAULT_CHUNK_SIZE_3D,
    DEFAULT_CONFIG_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_EXTRACTION_DIR_NAME,
    DEFAULT_EXTRACTION_FILE,
    DEFAULT_FEATURIZATION_DIR_NAME,
    DEFAULT_IMAGE_DTYPE,
    DEFAULT_INPUT_IMAGE_NAME,
    DEFAULT_PREFIX_FILTERED_SEG,
    DEFAULT_PREFIX_MAIN_SEG,
    DEFAULT_PREFIX_SELECTED_SEG,
    DEFAULT_SDATA_FILE,
    DEFAULT_SEG_NAME_0,
    DEFAULT_SEG_NAME_1,
    DEFAULT_SEGMENTATION_DIR_NAME,
    DEFAULT_SEGMENTATION_DTYPE,
    DEFAULT_SELECTION_DIR_NAME,
    DEFAULT_SINGLE_CELL_IMAGE_DTYPE,
    ChunkSize2D,
    ChunkSize3D,
)
from scportrait.plotting.h5sc import cell_grid
from scportrait.processing.images._image_processing import percentile_normalization


class Project(Logable):
    """Base implementation for a scPortrait ``project``.

    This class is designed to handle single-timepoint, single-location data, like e.g. whole-slide images.

    Segmentation Methods should be based on :func:`Segmentation <scportrait.pipeline.segmentation.Segmentation>` or :func:`ShardedSegmentation <scportrait.pipeline.segmentation.ShardedSegmentation>`.
    Extraction Methods should be based on :func:`HDF5CellExtraction <scportrait.pipeline.extraction.HDF5CellExtraction>`.

    Attributes:
        config (dict): Dictionary containing the config file.
        nuc_seg_name (str): Name of the nucleus segmentation object.
        cyto_seg_name (str): Name of the cytosol segmentation object.
        sdata_path (str): Path to the spatialdata object.
        filehander (sdata_filehandler): Filehandler for the spatialdata object which manages all calls or updates to the spatialdata object.
    """

    # define cleanup behaviour
    CLEAN_LOG: bool = True

    # import all default values from constants
    DEFAULT_CONFIG_NAME = DEFAULT_CONFIG_NAME
    DEFAULT_INPUT_IMAGE_NAME = DEFAULT_INPUT_IMAGE_NAME
    DEFAULT_SDATA_FILE = DEFAULT_SDATA_FILE

    DEFAULT_PREFIX_MAIN_SEG = DEFAULT_PREFIX_MAIN_SEG
    DEFAULT_PREFIX_FILTERED_SEG = DEFAULT_PREFIX_FILTERED_SEG
    DEFAULT_PREFIX_SELECTED_SEG = DEFAULT_PREFIX_SELECTED_SEG

    DEFAULT_SEG_NAME_0: str = DEFAULT_SEG_NAME_0
    DEFAULT_SEG_NAME_1: str = DEFAULT_SEG_NAME_1

    DEFAULT_CENTERS_NAME: str = DEFAULT_CENTERS_NAME

    DEFAULT_CHUNK_SIZE_3D: ChunkSize3D = DEFAULT_CHUNK_SIZE_3D
    DEFAULT_CHUNK_SIZE_2D: ChunkSize2D = DEFAULT_CHUNK_SIZE_2D

    DEFAULT_SEGMENTATION_DIR_NAME = DEFAULT_SEGMENTATION_DIR_NAME
    DEFAULT_EXTRACTION_DIR_NAME = DEFAULT_EXTRACTION_DIR_NAME
    DEFAULT_DATA_DIR = DEFAULT_DATA_DIR
    DEFAULT_EXTRACTION_FILE = DEFAULT_EXTRACTION_FILE

    DEFAULT_FEATURIZATION_DIR_NAME = DEFAULT_FEATURIZATION_DIR_NAME
    DEFAULT_SELECTION_DIR_NAME = DEFAULT_SELECTION_DIR_NAME

    DEFAULT_IMAGE_DTYPE = DEFAULT_IMAGE_DTYPE
    DEFAULT_SEGMENTATION_DTYPE = DEFAULT_SEGMENTATION_DTYPE
    DEFAULT_SINGLE_CELL_IMAGE_DTYPE = DEFAULT_SINGLE_CELL_IMAGE_DTYPE
    DEFAULT_CELL_ID_NAME = DEFAULT_CELL_ID_NAME

    def __init__(
        self,
        project_location: str,
        config_path: str = None,
        segmentation_f=None,
        extraction_f=None,
        featurization_f=None,
        selection_f=None,
        overwrite: bool = False,
        debug: bool = False,
    ):
        """
        Args:
            project_location (str): Path to the project directory.
            config_path (str): Path to the config file.
            segmentation_f (optional): Segmentation method to be used for the project.
            extraction_f (optional): Extraction method to be used for the project.
            featurization_f (optional): Featurization method to be used for the project.
            selection_f (optional): Selection method to be used for the project.
            overwrite (optional): If set to ``True``, will overwrite existing files in the project directory. Default is ``False``.
            debug (optional): If set to ``True``, will print additional debug messages/plots. Default is ``False``.
        """
        super().__init__(directory=project_location, debug=debug)

        self.project_location = project_location
        self.overwrite = overwrite
        self.config = None
        self._get_config_file(config_path)

        self.nuc_seg_name = f"{self.DEFAULT_PREFIX_MAIN_SEG}_{self.DEFAULT_SEG_NAME_0}"
        self.cyto_seg_name = f"{self.DEFAULT_PREFIX_MAIN_SEG}_{self.DEFAULT_SEG_NAME_1}"

        self.segmentation_f = segmentation_f
        self.extraction_f = extraction_f
        self.featurization_f = featurization_f
        self.selection_f = selection_f

        # intialize containers to support interactive viewing of the spatialdata object
        self.interactive_sdata = None
        self.interactive = None

        if self.CLEAN_LOG:
            self._clean_log_file()

        # check if project directory exists, if it does not create
        if not os.path.isdir(self.project_location):
            os.makedirs(self.project_location)
        else:
            Warning("There is already a directory in the location path")

        # === setup sdata reader/writer ===
        self.filehandler = sdata_filehandler(
            directory=self.directory,
            sdata_path=self.sdata_path,
            input_image_name=self.DEFAULT_INPUT_IMAGE_NAME,
            nuc_seg_name=self.nuc_seg_name,
            cyto_seg_name=self.cyto_seg_name,
            centers_name=self.DEFAULT_CENTERS_NAME,
            default_cell_id_name=self.DEFAULT_CELL_ID_NAME,
            debug=self.debug,
        )
        self.filehandler._check_sdata_status()  # update sdata object and status

        # === setup segmentation ===
        self._setup_segmentation_f(segmentation_f)

        # === setup extraction ===
        self._setup_extraction_f(extraction_f)

        # === setup featurization ===
        self._setup_featurization_f(featurization_f)

        # ==== setup selection ===
        self._setup_selection(selection_f)

    def __exit__(self):
        self._clear_temp_dir()

    def __del__(self):
        self._clear_temp_dir()

    @property
    def sdata_path(self) -> str:
        return self._get_sdata_path()

    @property
    def sdata(self) -> SpatialData:
        """Shape of data matrix (:attr:`n_obs`, :attr:`n_vars`)."""
        return self.filehandler.get_sdata()

    @property
    def h5sc(self) -> AnnData:
        if self.extraction_f is None:
            raise ValueError("No extraction method has been set.")
        else:
            if self.extraction_f.output_path is None:
                path = self.extraction_f.extraction_file
            else:
                path = self.extraction_f.output_path
            return read_h5sc(path)

    ##### Setup Functions #####

    def _load_config_from_file(self, file_path):
        """
        Loads config from file and writes it to self.config

        Args:
            file_path (str): Path to the config.yml file that should be loaded.
        """
        self.log(f"Loading config from {file_path}")

        if not os.path.isfile(file_path):
            raise ValueError(f"Your config path {file_path} is invalid.")

        self.config = read_config(file_path)

    def _get_config_file(self, config_path: str | None = None) -> None:
        """Load the config file for the project. If no config file is passed the default config file in the project directory is loaded.

        Args:
            config_path (str, optional): Path to the config file. Default is ``None``.

        Returns:
            None: the config dictionary project.config is updated.
        """
        # load config file
        self.config_path = os.path.join(self.project_location, self.DEFAULT_CONFIG_NAME)

        if config_path is None:
            # Check if there is already a config file in the dataset folder in case no config file has been specified

            if os.path.isfile(self.config_path):
                self._load_config_from_file(self.config_path)
            else:
                raise ValueError("No config passed and no config found in project directory.")

        else:
            if not os.path.isfile(config_path):
                raise ValueError(f"Your config path {config_path} is invalid. Please specify a valid config path.")

            else:
                print("Updating project config file.")

                if os.path.isfile(self.config_path):
                    os.remove(self.config_path)

                # ensure that the project location exists
                if not os.path.isdir(self.project_location):
                    os.makedirs(self.project_location)

                # The blueprint config file is copied to the dataset folder and renamed to the default name
                shutil.copy(config_path, self.config_path)
                self._load_config_from_file(self.config_path)

    def _setup_segmentation_f(self, segmentation_f):
        """Configure the segmentation method for the project.

        Args:
            segmentation_f (Callable): Segmentation method to be used for the project.

        Returns:
            None: the segmentation method is updated in the project object.
        """
        if segmentation_f is not None:
            if segmentation_f.__name__ not in self.config:
                raise ValueError(f"Config for {segmentation_f.__name__} is missing from the config file.")

            seg_directory = os.path.join(self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME)

            self.seg_directory = seg_directory

            self.segmentation_f = segmentation_f(
                self.config[segmentation_f.__name__],
                self.seg_directory,
                nuc_seg_name=self.nuc_seg_name,
                cyto_seg_name=self.cyto_seg_name,
                _tmp_image_path=None,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                project=None,
                filehandler=self.filehandler,
                from_project=True,
            )

    def _update_segmentation_f(self, segmentation_f):
        self._setup_segmentation_f(segmentation_f)

    def _setup_extraction_f(self, extraction_f):
        """Configure the extraction method for the project.

        Args:
            extraction_f (Callable): Extraction method to be used for the project.

        Returns:
            None: the extraction method is updated in the project object.
        """

        if extraction_f is not None:
            extraction_directory = os.path.join(self.project_location, self.DEFAULT_EXTRACTION_DIR_NAME)

            self.extraction_directory = extraction_directory

            if extraction_f.__name__ not in self.config:
                raise ValueError(f"Config for {extraction_f.__name__} is missing from the config file")

            self.extraction_f = extraction_f(
                self.config[extraction_f.__name__],
                self.extraction_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                project=self,
                filehandler=self.filehandler,
                from_project=True,
            )

    def _update_extraction_f(self, extraction_f):
        self._setup_extraction_f(extraction_f)

    def _setup_featurization_f(self, featurization_f):
        """Configure the featurization method for the project.

        Args:
            featurization_f (Callable): Featurization method to be used for the project.

        Returns:
            None: the featurization method is updated in the project object.
        """
        if featurization_f is not None:
            if featurization_f.__name__ not in self.config:
                raise ValueError(f"Config for {featurization_f.__name__} is missing from the config file")

            featurization_directory = os.path.join(self.project_location, self.DEFAULT_FEATURIZATION_DIR_NAME)

            self.featurization_directory = featurization_directory

            self.featurization_f = featurization_f(
                self.config[featurization_f.__name__],
                self.featurization_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=False,  # this needs to be set to false as the featurization step should not remove previously created features
                project=self,
                filehandler=self.filehandler,
                from_project=True,
            )

    def update_featurization_f(self, featurization_f):
        """Update the featurization method chosen for the project without reinitializing the entire project.

        Args:
            featurization_f : The featurization method that should be used for the project.

        Returns:
            None : the featurization method is updated in the project object.

        Examples:
            Update the featurization method for a project::

                from scportrait.pipeline.featurization import CellFeaturizer

                project.update_featurization_f(CellFeaturizer)
        """
        self.log(f"Replacing current featurization method {self.featurization_f.__class__} with {featurization_f}")
        self._setup_featurization_f(featurization_f)

    def _setup_selection(self, selection_f):
        """Configure the selection method for the project.

        Args:
            selection_f (Callable): Selection method to be used for the project.

        Returns:
            None: the selection method is updated in the project object.
        """
        if self.selection_f is not None:
            if selection_f.__name__ not in self.config:
                raise ValueError(f"Config for {selection_f.__name__} is missing from the config file")

            selection_directory = os.path.join(self.project_location, self.DEFAULT_SELECTION_DIR_NAME)

            self.selection_directory = selection_directory

            self.selection_f = selection_f(
                self.config[selection_f.__name__],
                self.selection_directory,
                project_location=self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                project=self,
                filehandler=self.filehandler,
                from_project=True,
            )

    def _update_selection_f(self, selection_f):
        self._setup_selection(selection_f)

    ##### General small helper functions ####

    def _check_memory(self, item):
        """
        Check the memory usage of the given if it were completely loaded into memory using .compute().
        """
        array_size = item.nbytes
        available_memory = psutil.virtual_memory().available

        return array_size < available_memory

    def _check_chunk_size(self, elem, chunk_size):
        """
        Check if the chunk size of the element is the default chunk size. If not rechunk the element to the default chunk size.
        """

        # get chunk size of element
        chunk_size = get_chunk_size(elem)

        if isinstance(chunk_size, list):
            # check if all chunk sizes are the same otherwise rechunking needs to occur anyways
            if not all(x == chunk_size[0] for x in chunk_size):
                elem = rechunk_image(elem, chunk_size=chunk_size)
            else:
                # ensure that the chunk size is the default chunk size
                if chunk_size != chunk_size:
                    elem = rechunk_image(elem, chunk_size=chunk_size)
        else:
            # ensure that the chunk size is the default chunk size
            if chunk_size != chunk_size:
                elem = rechunk_image(elem, chunk_size=chunk_size)

    def _check_image_dtype(self, image: np.ndarray) -> None:
        """Check if the image dtype is the default image dtype.

        Args:
            image (np.ndarray): Image to be checked.

        Returns:
            None: If the image dtype is the default image dtype, no action is taken.

        Raises:
            Warning: If the image dtype is not the default image dtype.
        """

        if not image.dtype == self.DEFAULT_IMAGE_DTYPE:
            Warning(
                f"Image dtype is not {self.DEFAULT_IMAGE_DTYPE} but insteadt {image.dtype}. The workflow expects images to be of dtype {self.DEFAULT_IMAGE_DTYPE}. Proceeding with the incorrect dtype can lead to unexpected results."
            )
            self.log(
                f"Image dtype is not {self.DEFAULT_IMAGE_DTYPE} but insteadt {image.dtype}. The workflow expects images to be of dtype {self.DEFAULT_IMAGE_DTYPE}. Proceeding with the incorrect dtype can lead to unexpected results."
            )

    def _create_temp_dir(self, path) -> None:
        """
        Create a temporary directory in the specified directory with the name of the class.

        Args:
            path (str): Path to the directory where the temporary directory should be created.

        Returns:
            None: The temporary directory is created in the specified directory. The path to the temporary directory is stored in the project object as self._tmp_dir_path.

        """

        path = os.path.join(path, f"{self.__class__.__name__}_")
        self._tmp_dir = tempfile.TemporaryDirectory(prefix=path)
        self._tmp_dir_path = self._tmp_dir.name
        """str: Path to the temporary directory."""

        self.log(f"Initialized temporary directory at {self._tmp_dir_path} for {self.__class__.__name__}")

    def _clear_temp_dir(self):
        """Clear the temporary directory."""
        if "_tmp_dir" in self.__dict__.keys():
            shutil.rmtree(self._tmp_dir_path, ignore_errors=True)
            self.log(f"Cleaned up temporary directory at {self._tmp_dir}")

            del self._tmp_dir, self._tmp_dir_path
        else:
            self.log("Temporary directory not found, skipping cleanup")

    ##### Functions for handling sdata object #####

    def _cleanup_sdata_object(self):
        """
        Check if the output location exists and if it does cleanup if allowed, otherwise raise an error.
        """

        if os.path.exists(self.sdata_path):
            if self.overwrite:
                if not self.filehandler._check_empty_sdata():
                    self.log(f"Output location {self.sdata_path} already exists. Overwriting.")
                    shutil.rmtree(self.sdata_path, ignore_errors=True)
            else:
                # check to see if the sdata object is empty
                if not self.filehandler._check_empty_sdata():
                    raise ValueError(
                        f"Output location {self.sdata_path} already exists. Set overwrite=True to overwrite."
                    )
        self.filehandler._check_sdata_status()

    def _get_sdata_path(self):
        """
        Get the path to the spatialdata object.
        """
        return os.path.join(self.project_location, self.DEFAULT_SDATA_FILE)

    def print_project_status(self):
        """Print the current project status."""
        self.get_project_status(print_status=True)

    def get_project_status(self, print_status=False):
        self.filehandler._check_sdata_status()
        self.input_image_status = self.filehandler.input_image_status
        self.nuc_seg_status = self.filehandler.nuc_seg_status
        self.cyto_seg_status = self.filehandler.cyto_seg_status
        self.nuc_centers_status = self.filehandler.nuc_centers_status
        self.cyto_centers_status = self.filehandler.cyto_centers_status
        extraction_file = os.path.join(
            self.project_location, self.DEFAULT_EXTRACTION_DIR_NAME, self.DEFAULT_DATA_DIR, self.DEFAULT_EXTRACTION_FILE
        )
        self.extraction_status = True if os.path.isfile(extraction_file) else False

        if self.input_image_status:
            if self.DEFAULT_INPUT_IMAGE_NAME in self.sdata:
                self.input_image = _get_image(self.sdata[self.DEFAULT_INPUT_IMAGE_NAME])
            else:
                self.input_image = None

        if print_status:
            self.log("Current Project Status:")
            self.log("--------------------------------")
            self.log(f"Input Image in sdata: {self.input_image_status}")
            self.log(f"Nucleus Segmentation in sdata: {self.nuc_seg_status}")
            self.log(f"Cytosol Segmentation in sdata: {self.cyto_seg_status}")
            self.log(f"Nucleus Centers in sdata: {self.nuc_centers_status}")
            self.log(f"Cytosol Centers in sdata: {self.cyto_centers_status}")
            self.log(f"Extracted single-cell images saved to file: {self.extraction_status}")
            self.log("--------------------------------")

        return None

    def view_sdata(self):
        """Start an interactive napari viewer to look at the sdata object associated with the project.
        Note:
            This only works in sessions with a visual interface.
        """
        # open interactive viewer in napari
        try:
            from napari_spatialdata import Interactive
        except ImportError:
            raise ImportError(
                "napari-spatialdata must be installed to use the interactive viewer. Please install with `pip install scportrait[plotting]`."
            ) from None
        self.interactive_sdata = self.filehandler.get_sdata()
        self.interactive = Interactive(self.interactive_sdata)
        self.interactive.run()

    def _save_interactive_sdata(self):
        assert self.interactive_sdata is not None, "No interactive sdata object found."

        in_memory_only, _ = self.interactive_sdata._symmetric_difference_with_zarr_store()
        print(f"Writing the following manually added files to the sdata object: {in_memory_only}")

        dict_lookup = {}
        for elem in in_memory_only:
            key, name = elem.split("/")
            if key not in dict_lookup:
                dict_lookup[key] = []
            dict_lookup[key].append(name)
        for _, name in dict_lookup.items():
            self.interactive_sdata.write_element(name)  # replace with correct function once pulled in from sdata

    def close_interactive_viewer(self):
        if self.interactive is not None:
            self._save_interactive_sdata()
            self.interactive._viewer.close()

            # reset to none values to track next call of view_sdata
            self.interactive_sdata = None
            self.interactive = None

    def _check_for_interactive_session(self):
        if self.interactive is not None:
            Warning("Interactive viewer is still open. Will automatically close before proceeding with processing.")
            self.close_interactive_viewer()

    #### Functions to visualize results ####

    def plot_input_image(
        self,
        max_width: int = 1000,
        select_region: tuple[int, int] | None = None,
        channels: list[int] | list[str] | None = None,
        normalize: bool = False,
        normalization_percentile: tuple[float, float] = (0.01, 0.99),
        fontsize: int = 20,
        figsize_single_tile=(8, 8),
        return_fig: bool = False,
        image_name="input_image",
    ) -> Figure | None:
        """Plot the input image associated with the project. If the image is large it will automatically plot a subset in the center

        Args:
            max_size: Maximum size of the image to be plotted in pixels.
            select_region: Tuple containing the x and y coordinates of the center of the region to be plotted. If not set it will use the center of the image.
            channels: List of channel names or indices to be plotted. If not set, the first 4 channels will be plotted.
            fontsize: Fontsize of the title of the plot.
            figsize_single_tile: Size of the single tile in the plot.
            return_fig: If set to ``True``, the function returns the figure object instead of displaying it.

        Returns:
            A matplotlib figure object if return_fig is set to ``True``.

        Examples:
            Plot the input image of a project::

                project.plot_input_image()
        """

        try:
            import matplotlib.pyplot as plt
            import spatialdata_plot  # this does not have an explicit call put allows for sdata.pl calls
            from spatialdata import to_polygons

        except ImportError:
            raise ImportError(
                "spatialdata_plot must be installed to use the plotting capabilites. please install with `pip install scportrait[plotting]`."
            ) from None

        _sdata = self.sdata

        # remove points object as this makes it
        points_keys = list(_sdata.points.keys())
        if len(points_keys) > 0:
            for x in points_keys:
                del _sdata.points[x]

        palette = [
            "blue",
            "green",
            "red",
            "yellow",
            "purple",
            "orange",
            "pink",
            "cyan",
            "magenta",
            "lime",
            "teal",
            "lavender",
            "brown",
            "beige",
            "maroon",
            "mint",
            "olive",
            "apricot",
            "navy",
            "grey",
            "white",
            "black",
        ]
        c, x, y = _sdata[image_name].scale0.image.shape
        channel_names = _sdata["input_image"].scale0.c.values

        if channels is not None:
            if isinstance(channels[0], str):
                assert [
                    x in channel_names for x in channels
                ], "The specified channel names are not found in the spatialdata object."
                channel_names = channels
                palette = palette[:c]
            if isinstance(channels[0], int):
                assert [
                    x in range(c) for x in channels
                ], "The specified channel indices are not found in the spatialdata object."
                channel_names = list(channel_names[channels])
            c = len(channels)
            palette = palette[:c]
        else:
            # do not plot more than 4 channels per default
            if c > 4:
                c = 4
            palette = palette[:c]
            channel_names = list(channel_names[:c])

        # subset spatialdata object if its too large
        width = max_width // 2
        if x > max_width or y > max_width:
            if select_region is None:
                center_x = x // 2
                center_y = y // 2
            else:
                center_x, center_y = select_region

            _sdata = _sdata.query.bounding_box(
                axes=["x", "y"],
                min_coordinate=[center_x - width, center_y - width],
                max_coordinate=[center_x + width, center_y + width],
                target_coordinate_system="global",
            )

        if normalize:
            lower_percentile, upper_percentile = normalization_percentile

            # get percentile values to normalize viewing to
            for channel in channel_names:
                idx = list(_sdata[image_name].scale0.c.values).index(channel)
                for scale in _sdata[image_name]:
                    im = _sdata[image_name].get(scale).image[idx].compute()
                    _sdata[image_name].get(scale).image[idx] = (
                        percentile_normalization(im, lower_percentile, upper_percentile) * np.iinfo(np.uint16).max
                    ).astype(np.uint16)

        fig_size_x, fig_size_y = figsize_single_tile
        fig, axs = plt.subplots(1, len(channel_names) + 1, figsize=(fig_size_x * (len(channel_names) + 1), fig_size_y))
        _sdata.pl.render_images(image_name, channel=channel_names, palette=palette).pl.show(ax=axs[0])
        axs[0].set_title("overlayed", fontsize=fontsize)
        axs[0].axis("off")

        for i, channel in enumerate(channel_names):
            _sdata.pl.render_images(image_name, channel=channel, palette=palette[i]).pl.show(
                ax=axs[i + 1],
                colorbar=False,
            )
            axs[i + 1].set_title(channel, fontsize=fontsize)
            axs[i + 1].axis("off")
        fig.tight_layout()

        if return_fig:
            return fig
        else:
            return None
            plt.show()

    def plot_he_image(
        self,
        image_name: str = "he_image",
        max_width: int | None = None,
        select_region: tuple[int, int] | None = None,
        return_fig: bool = False,
        fontsize: int = 20,
    ) -> None | Figure:
        """Plot the hematoxylin and eosin (HE) channel of the input image.

        Args:
            image_name: Name of the element containing the H&E image in the spatialdata object.
            max_width: Maximum width of the image to be plotted in pixels.
            select_region: Tuple containing the x and y coordinates of the region to be plotted. If not set it will use the center of the image.

            return_fig: If set to ``True``, the function returns the figure object instead of displaying it.

        Returns:
            A matplotlib figure object if return_fig is set to ``True``.

        Examples:
            Plot the HE channel of a project::

                project.plot_he()
        """
        try:
            import matplotlib.pyplot as plt
            import spatialdata_plot  # this does not have an explicit call put allows for sdata.pl calls
            from spatialdata import to_polygons

        except ImportError:
            raise ImportError(
                "spatialdata_plot must be installed to use the plotting capabilites. please install with `pip install scportrait[plotting]`."
            ) from None

        _sdata = self.sdata

        # remove points object as this makes it
        points_keys = list(_sdata.points.keys())
        if len(points_keys) > 0:
            for x in points_keys:
                del _sdata.points[x]

        channel_names = list(_sdata[image_name].scale0.c.values)
        assert channel_names == ["r", "g", "b"], "The image is not an RGB image."

        # subset spatialdata object if its too large
        if max_width is not None:
            c, x, y = _sdata[image_name].scale0.image.shape
            width = max_width // 2
            if select_region is None:
                center_x = x // 2
                center_y = y // 2
            else:
                center_x, center_y = select_region

            if x > max_width or y > max_width:
                _sdata = _sdata.query.bounding_box(
                    axes=["x", "y"],
                    min_coordinate=[center_x - width, center_y - width],
                    max_coordinate=[center_x + width, center_y + width],
                    target_coordinate_system="global",
                )

        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        _sdata.pl.render_images(image_name).pl.show(ax=axs, title="H&E Image")
        axs.axis("off")
        fig.tight_layout()

        if return_fig:
            return fig
        else:
            return None
            plt.show()

    def plot_segmentation_masks(
        self,
        max_width: int = 1500,
        select_region: tuple[int, int] | None = None,
        normalize: bool = False,
        normalization_percentile: tuple[float, float] = (0.01, 0.99),
        image_name: str = "input_image",
        mask_names: list[str] | None = None,
        fontsize: int = 20,
        linewidth: int = 1,
        return_fig: bool = False,
    ) -> None | Figure:
        """Plot the generated segmentation masks. If the image is large it will automatically plot a subset cropped to the center of the spatialdata object.

        Args:
            return_fig: If set to ``True``, the function returns the figure object instead of displaying it.
            max_width: Maximum width of the image to be plotted in pixels.
            select_region: Tuple containing the x and y coordinates of the region to be plotted. If not set it will use the center of the image.

        Returns:
            A matplotlib figure object if return_fig is set to ``True``.

        Examples:
            Plot the segmentation masks of a project::

                project.plot_segmentation_masks()
        """
        # import relevant functions for this method
        import matplotlib.pyplot as plt

        from scportrait.plotting.sdata import _bounding_box_sdata, plot_segmentation_mask

        _sdata = self.sdata

        # get relevant information from spatialdata object
        _, x, y = _sdata["input_image"].scale0.image.shape
        channel_names = list(_sdata["input_image"].scale0.c.values)

        # get center coordinates
        if select_region is None:
            center_x = x // 2
            center_y = y // 2
        else:
            center_x, center_y = select_region

        # subset spatialdata object if its too large
        if x > max_width or y > max_width:
            _sdata = _bounding_box_sdata(_sdata, max_width, center_x, center_y)

        if normalize:
            lower_percentile, upper_percentile = normalization_percentile

            # get percentile values to normalize viewing to
            for channel in channel_names:
                idx = list(_sdata[image_name].scale0.c.values).index(channel)
                for scale in _sdata[image_name]:
                    im = _sdata[image_name].get(scale).image[idx].compute()
                    _sdata[image_name].get(scale).image[idx] = (
                        percentile_normalization(im, lower_percentile, upper_percentile) * np.iinfo(np.uint16).max
                    ).astype(np.uint16)

        # get relevant segmentation masks
        if mask_names is None:
            masks = []
            if self.filehandler.nuc_seg_status:
                masks.append("seg_all_nucleus")
            if self.filehandler.cyto_seg_status:
                masks.append("seg_all_cytosol")

            if len(masks) == 0:
                raise ValueError("No segmentation masks found in the sdata object.")
        else:
            for mask in mask_names:
                if mask not in _sdata:
                    raise ValueError(f"Mask {mask} not found in the spatialdata object.")
            masks = mask_names

        # create plot
        fig, axs = plt.subplots(1, len(masks) + 1, figsize=(8 * (len(masks) + 1), 8))
        plot_segmentation_mask(
            _sdata,
            masks,
            max_width=max_width,
            axs=axs[0],
            title="overlayed",
            font_size=fontsize,
            linewidth=linewidth,
            show_fig=False,
        )

        for mask in masks:
            idx = masks.index(mask)
            if "nucleus" in mask:
                channel = [0]
                name = "Nucleus Mask"
            elif "cytosol" in mask:
                channel = [1]
                name = "Cytosol Mask"
            else:
                channel = list(range(len(channel_names)))
                name = mask

            plot_segmentation_mask(
                _sdata,
                [mask],
                max_width=max_width,
                selected_channels=channel,
                axs=axs[idx + 1],
                title=name,
                font_size=fontsize,
                linewidth=linewidth,
                show_fig=False,
            )

        fig.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return None

    def plot_single_cell_images(
        self,
        n_cells: int | None = None,
        cell_ids: list[int] | None = None,
        select_channel: int | None = None,
        cmap="viridis",
        return_fig: bool = False,
    ) -> None | Figure:
        if cell_ids is not None:
            assert n_cells is None, "n_cells and cell_ids cannot be set at the same time."
        if n_cells is not None:
            assert cell_ids is None, "n_cells and cell_ids cannot be set at the same time."

        return cell_grid(
            self.h5sc,
            n_cells=n_cells,
            cell_ids=cell_ids,
            select_channel=select_channel,
            cmap=cmap,
            return_fig=return_fig,
        )

    #### Functions to load input data ####
    def load_input_from_array(
        self, array: np.ndarray, channel_names: list[str] = None, overwrite: bool | None = None, remap: list[int] = None
    ) -> None:
        """Load input image from a numpy array.

        In the array the channels should be specified in the following order: nucleus, cytosol other channels.

        Args:
            array (np.ndarray): Input image as a numpy array.
            channel_names: List of channel names. Default is ``["channel_0", "channel_1", ...]``.
            overwrite (bool, None, optional): If set to ``None``, will read the overwrite value from the associated project.
                Otherwise can be set to a boolean value to override project specific settings for image loading.
            remap: List of integers that can be used to shuffle the order of the channels. For example ``[1, 0, 2]`` to invert the first two channels. Default is ``None`` in which case no reordering is performed.
                This transform is also applied to the channel names.
        Returns:
            None: Image is written to the project associated sdata object.

            The input image can be accessed using the project object::

                    project.input_image

        Examples:
            Load input images from tif files and attach them to an scportrait project::

                from scportrait.pipeline.project import Project

                project = Project("path/to/project", config_path="path/to/config.yml", overwrite=True, debug=False)
                array = np.random.rand(3, 1000, 1000)
                channel_names = ["cytosol", "nucleus", "other_channel"]
                project.load_input_from_array(array, channel_names=channel_names, remap=[1, 0, 2])

        """
        # check if an input image was already loaded if so throw error if overwrite = False

        # setup overwrite
        original_overwrite = self.overwrite
        if overwrite is not None:
            self.overwrite = overwrite

        self._cleanup_sdata_object()

        # get channel names
        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(array.shape[0])]

        assert len(channel_names) == array.shape[0], "Number of channel names does not match number of input images."

        self.channel_names = channel_names

        # ensure the array is a dask array
        image = darray.from_array(array, chunks=self.DEFAULT_CHUNK_SIZE_3D)

        if remap is not None:
            image = image[remap]
            self.channel_names = [self.channel_names[i] for i in remap]

        # write to sdata object
        self.filehandler._write_image_sdata(
            image,
            channel_names=self.channel_names,
            scale_factors=[2, 4, 8],
            chunks=self.DEFAULT_CHUNK_SIZE_3D,
            image_name=self.DEFAULT_INPUT_IMAGE_NAME,
        )

        self.get_project_status()
        self.overwrite = original_overwrite  # reset to original value

    def load_input_from_tif_files(
        self,
        file_paths: list[str],
        channel_names: list[str] = None,
        crop: list[tuple[int, int]] | None = None,
        overwrite: bool | None = None,
        remap: list[int] = None,
        cache: str | None = None,
    ):
        """
        Load input image from a list of files. The channels need to be specified in the following order: nucleus, cytosol other channels.

        Args:
            file_paths: List containing paths to each channel tiff file, like
                ``["path1/img.tiff", "path2/img.tiff", "path3/img.tiff"]``
            channel_names: List of channel names. Default is ``["channel_0", "channel_1", ...]``.
            crop (None, List[Tuple], optional): When set, it can be used to crop the input image.
                The first element refers to the first dimension of the image and so on.
                For example use ``[(0,1000),(0,2000)]`` to crop the image to 1000 px height and 2000 px width from the top left corner.
            overwrite (bool, None, optional): If set to ``None``, will read the overwrite value from the associated project.
                Otherwise can be set to a boolean value to override project specific settings for image loading.
            remap: List of integers that can be used to shuffle the order of the channels. For example ``[1, 0, 2]`` to invert the first two channels. Default is ``None`` in which case no reordering is performed.
                This transform is also applied to the channel names.
            cache: path to a directory where the temporary files should be stored. Default is ``None`` then the current working directory will be used.

        Returns:
            None: Image is written to the project associated sdata object.

            The input image can be accessed using the project object::

                    project.input_image

        Examples:
            Load input images from tif files and attach them to an scportrait project::

                from scportrait.data._datasets import dataset_3
                from scportrait.pipeline.project import Project

                project = Project("path/to/project", config_path="path/to/config.yml", overwrite=True, debug=False)
                path = dataset_3()
                image_paths = [
                    f"{path}/Ch2.tif",
                    f"{path}/Ch1.tif",
                    f"{path}/Ch3.tif",
                ]
                channel_names = ["cytosol", "nucleus", "other_channel"]
                project.load_input_from_tif_files(image_paths, channel_names=channel_names, remap=[1, 0, 2])

        """

        if crop is None:
            crop = [(0, -1), (0, -1)]

        def extract_unique_parts(paths: list[str]):
            """helper function to get unique channel names from filepaths

            Parameters
            ----------
            paths : str
                _description_

            Returns
            -------
            List[str]

            """
            # Find the common base directory
            common_base = os.path.commonpath(paths)

            # Remove the common base from each path
            unique_paths = [os.path.relpath(path, common_base) for path in paths]

            unique_parts = []
            pattern = re.compile(r"(\d+)")  # Regex pattern to match numbers

            for file_name in unique_paths:
                match = pattern.search(file_name)
                if match:
                    unique_parts.append(match.group(1))  # Extract the matched number
                else:
                    unique_parts.append(file_name)  # If no match, return the whole name

            return unique_parts

        # setup overwrite
        original_overwrite = self.overwrite
        if overwrite is not None:
            self.overwrite = overwrite

        if self.config is None:
            raise ValueError("Dataset has no config file loaded")

        # check if an input image was already loaded if so throw error if overwrite = False
        self._cleanup_sdata_object()

        # save channel names
        if channel_names is None:
            channel_names = extract_unique_parts(file_paths)

        assert len(channel_names) == len(file_paths), "Number of channel names does not match number of input images."

        self.channel_names = channel_names

        # remap can be used to shuffle the order, for example [1, 0, 2] to invert the first two channels
        # default order that is expected: Nucleus channel, cell membrane channel, other channels
        if remap is not None:
            file_paths = [file_paths[i] for i in remap]
            self.channel_names = [self.channel_names[i] for i in remap]

        if cache is None:
            cache = os.getcwd()

        self._create_temp_dir(cache)

        for i, channel_path in enumerate(file_paths):
            im = imread(channel_path)

            # add automatic conversion for uint8 as this is another very common image format
            if im.dtype == np.uint8:
                im = (
                    im.astype(np.uint16) * np.iinfo(np.uint8).max
                )  # leave set to np.uint16 explicilty here as this conversion assumes going from uint8 to uint16 if the dtype is changed then this will throw a warning later ensuring that this line is fixed

            self._check_image_dtype(im)

            im = np.array(im, dtype=self.DEFAULT_IMAGE_DTYPE)[slice(*crop[0]), slice(*crop[1])]

            if i == 0:
                # define shape of required tempmmap array to read results to
                y, x = im.shape
                c = len(file_paths)

                # initialize temp array to save results to and then append to channels
                temp_image_path = tempmmap.create_empty_mmap(
                    shape=(c, y, x),
                    dtype=self.DEFAULT_IMAGE_DTYPE,
                    tmp_dir_abs_path=self._tmp_dir_path,
                )

                channels = tempmmap.mmap_array_from_path(temp_image_path)

            channels[i] = im

        channels = daskmmap.dask_array_from_path(temp_image_path)

        self.filehandler._write_image_sdata(
            channels,
            self.DEFAULT_INPUT_IMAGE_NAME,
            channel_names=self.channel_names,
            scale_factors=[2, 4, 8],
            chunks=self.DEFAULT_CHUNK_SIZE_3D,
        )

        self.overwrite = original_overwrite  # reset to original value

        # cleanup variables and temp dir
        self._clear_cache(vars_to_delete=[temp_image_path, im, channels])
        self._clear_temp_dir()

        # strange workaround that is required so that the sdata input image does not point to the dask array anymore but
        # to the image which was written to disk
        self.get_project_status()

    def load_input_from_omezarr(
        self,
        ome_zarr_path: str,
        overwrite: None | bool = None,
        channel_names: None | list[str] = None,
        remap: list[int] = None,
    ) -> None:
        """Load input image from an ome-zarr file.

        Args:
            ome_zarr_path: Path to the ome-zarr file.
            overwrite (bool, None, optional): If set to ``None``, will read the overwrite value from the associated project.
                Otherwise can be set to a boolean value to override project specific settings for image loading.
            remap: List of integers that can be used to shuffle the order of the channels. For example ``[1, 0, 2]`` to invert the first two channels. Default is ``None`` in which case no reordering is performed.
                This transform is also applied to the channel names.

        Returns:
            None: Image is written to the project associated sdata object.

            The input image can be accessed using the project object::

                    project.input_image

        Examples:
            Load input images from an ome-zarr file and attach them to an scportrait project::

                from scportrait.pipeline.project import Project

                project = Project("path/to/project", config_path="path/to/config.yml", overwrite=True, debug=False)
                ome_zarr_path = "path/to/ome.zarr"
                project.load_input_from_omezarr(ome_zarr_path, remap=[1, 0, 2])
        """
        # setup overwrite
        original_overwrite = self.overwrite
        if overwrite is not None:
            self.overwrite = overwrite

        # check if an input image was already loaded if so throw error if overwrite = False
        self._cleanup_sdata_object()

        # read the image data
        self.log(f"trying to read file from {ome_zarr_path}")
        image = da.from_zarr(str(ome_zarr_path), component=0)

        # Access the metadata to get channel names
        zarr_store = zarr.open(ome_zarr_path, mode="r")  # Adjust the path
        metadata = zarr_store.attrs.asdict()

        if "omero" in metadata and "channels" in metadata["omero"]:
            channels = metadata["omero"]["channels"]
            channel_names = [channel["label"] for channel in channels]
        else:
            if len(image.shape) == 3:
                _, _, n_channels = image.shape
            elif len(image.shape) == 2:
                n_channels = 1
            channel_names = [f"channel_{i}" for i in range(n_channels)]

        self.channel_names = channel_names

        if remap is not None:
            image = image[remap]
            self.channel_names = [self.channel_names[i] for i in remap]

        # write to sdata object
        self.filehandler._write_image_sdata(
            image,
            channel_names=self.channel_names,
            scale_factors=[2, 4, 8],
            chunks=self.DEFAULT_CHUNK_SIZE_3D,
            image_name=self.DEFAULT_INPUT_IMAGE_NAME,
        )

        self.get_project_status()
        self.overwrite = original_overwrite  # reset to original value

    def load_input_from_dask(self, dask_array, channel_names: list[str], overwrite: bool | None = None) -> None:
        """Load input image from a dask array.

        Args:
            dask_array: Dask array containing the input image.
            channel_names: List of channel names. Default is ``["channel_0", "channel_1", ...]``.
            overwrite (bool, None, optional): If set to ``None``, will read the overwrite value from the associated project.
                Otherwise can be set to a boolean value to override project specific settings for image loading.

        Returns:
            None: Image is written to the project associated sdata object.

            The input image can be accessed using the project object::

                    project.input_image

        Examples:
            Load input images from a dask array and attach them to an scportrait project::

                from scportrait.pipeline.project import Project

                project = Project("path/to/project", config_path="path/to/config.yml", overwrite=True, debug=False)
                dask_array = da.random.random((3, 1000, 1000))
                channel_names = ["cytosol", "nucleus", "other_channel"]
                project.load_input_from_dask(dask_array, channel_names=channel_names)

        """
        # setup overwrite
        original_overwrite = self.overwrite
        if overwrite is not None:
            self.overwrite = overwrite

        self._cleanup_sdata_object()

        assert (
            len(channel_names) == dask_array.shape[0]
        ), "Number of channel names does not match number of input images."

        self.channel_names = channel_names

        self.filehandler._write_image_sdata(
            dask_array,
            channel_names=self.channel_names,
            scale_factors=[2, 4, 8],
            chunks=self.DEFAULT_CHUNK_SIZE_3D,
            image_name=self.DEFAULT_INPUT_IMAGE_NAME,
        )

        self.get_project_status()
        self.overwrite = original_overwrite

    def load_input_from_sdata(
        self,
        sdata_path,
        input_image_name: str,
        nucleus_segmentation_name: str | None = None,
        cytosol_segmentation_name: str | None = None,
        cell_id_identifier: str | None = None,
        overwrite: bool | None = None,
        keep_all: bool = True,
        remove_duplicates: bool = True,
    ) -> None:
        """
        Load input image from a spatialdata object.

        Args:
            sdata_path: Path to the spatialdata object.
            input_image_name: Name of the element in the spatial data object containing the input image.
            nucleus_segmentation_name: Name of the element in the spatial data object containing the nucleus segmentation mask. Default is ``None``.
            cytosol_segmentation_name: Name of the element in the spatial data object containing the cytosol segmentation mask. Default is ``None``.
            cell_id_identifier: column of annotating tables that contain the values that match a segmentation mask. If not provided it will assume this column carries the same name as the segmentation mask before parsing.
            overwrite (bool, None, optional): If set to ``None``, will read the overwrite value from the associated project.
                Otherwise can be set to a boolean value to override project specific settings for image loading.
            keep_all: If set to ``True``, will keep all existing elements in the sdata object in addition to renaming the desired ones. Default is ``True``.
            remove_duplicates: If keep_all and remove_duplicates is True then only one copy of the spatialdata elements selected for use with scportrait processing steps will be kept. Otherwise, the element will be saved both under the original as well as the new name.

        Returns:
            None: Image is written to the project associated sdata object and self.sdata is updated.
        """

        # setup overwrite
        original_overwrite = self.overwrite
        if overwrite is not None:
            self.overwrite = overwrite

        # check if an input image was already loaded if so throw error if overwrite = False
        self._cleanup_sdata_object()

        # read input sdata object
        sdata_input = SpatialData.read(sdata_path)
        all_elements = [x.split("/")[1] for x in sdata_input.elements_paths_in_memory()]
        dict_elems = {self.DEFAULT_INPUT_IMAGE_NAME: sdata_input[input_image_name]}

        if nucleus_segmentation_name is not None:
            dict_elems[self.nuc_seg_name] = sdata_input[nucleus_segmentation_name]
            if remove_duplicates:
                all_elements.remove(nucleus_segmentation_name)

        if cytosol_segmentation_name is not None:
            dict_elems[self.cyto_seg_name] = sdata_input[cytosol_segmentation_name]
            if remove_duplicates:
                all_elements.remove(cytosol_segmentation_name)

        # set cell_id_identifier
        if cell_id_identifier is None:
            if nucleus_segmentation_name is not None and cytosol_segmentation_name is not None:
                cell_id_identifier = cytosol_segmentation_name
                new_name = self.cyto_seg_name
            elif nucleus_segmentation_name is not None:
                cell_id_identifier = nucleus_segmentation_name
                new_name = self.nuc_seg_name
            elif cytosol_segmentation_name is not None:
                cell_id_identifier = cytosol_segmentation_name
                new_name = self.cyto_seg_name
            else:
                cell_id_identifier = None
        else:
            if nucleus_segmentation_name is not None and cytosol_segmentation_name is not None:
                new_name = self.cyto_seg_name
            elif nucleus_segmentation_name is not None:
                new_name = self.nuc_seg_name
            elif cytosol_segmentation_name is not None:
                new_name = self.cyto_seg_name
            else:
                new_name = None

        # ensure that any annotating table objects are updated with the correct labels
        table_elements = [x.split("/")[1] for x in sdata_input.elements_paths_in_memory() if x.split("/")[1] == "table"]

        for table_elem in table_elements:
            table = sdata_input[table_elem]
            rename_columns = {}
            if self.DEFAULT_CELL_ID_NAME in table.obs:
                Warning(
                    f"Column {self.DEFAULT_CELL_ID_NAME} already exists in table. Renaming to `f{self.DEFAULT_CELL_ID_NAME}_orig` to preserve compatibility with scPortrait workflow."
                )
                rename_columns[self.DEFAULT_CELL_ID_NAME] = f"{self.DEFAULT_CELL_ID_NAME}_orig"
                self.log(
                    f"Renaming column `{self.DEFAULT_CENTERS_NAME}` to `f{self.DEFAULT_CELL_ID_NAME}_orig` in table {table_elem}"
                )
            if cell_id_identifier is not None:
                rename_columns[cell_id_identifier] = self.DEFAULT_CELL_ID_NAME
                self.log(f"Renaming column `{cell_id_identifier}` to `{DEFAULT_CELL_ID_NAME}` in table {table_elem}")
                table.uns["spatialdata_attrs"]["instance_key"] = self.DEFAULT_CELL_ID_NAME
                table.uns["spatialdata_attrs"]["region"] = new_name
                table.obs["region"] = new_name
                table.obs["region"] = table.obs["region"].astype("category")
            table.obs.rename(columns=rename_columns, inplace=True)

        if keep_all:
            shutil.rmtree(self.sdata_path, ignore_errors=True)
            for elem in all_elements:
                dict_elems[elem] = sdata_input[elem]

        sdata = SpatialData.init_from_elements(dict_elems)
        sdata.write(self.sdata_path, overwrite=True)

        # update project status
        self.get_project_status()
        _, x, y = _get_shape(sdata[self.DEFAULT_INPUT_IMAGE_NAME])

        self.overwrite = original_overwrite

        if self.nuc_seg_status:
            # check input size
            _, x_mask, y_mask = _get_shape(sdata[self.nuc_seg_name])
            assert x == x_mask and y == y_mask, "Input image and nucleus segmentation mask do not match in size."

            self.filehandler._add_centers(segmentation_label=self.nuc_seg_name)
        if self.cyto_seg_status:
            # check input size
            _, x_mask, y_mask = _get_shape(sdata[self.cyto_seg_name])
            assert x == x_mask and y == y_mask, "Input image and nucleus segmentation mask do not match in size."

            self.filehandler._add_centers(segmentation_label=self.cyto_seg_name)

        # ensure that if both an nucleus and cytosol segmentation mask are loaded that they match
        if self.nuc_seg_status and self.cyto_seg_status:
            ids_nuc = set(sdata[f"{self.DEFAULT_CENTERS_NAME}_{self.nuc_seg_name}"].index.values)
            ids_cyto = set(sdata[f"{self.DEFAULT_CENTERS_NAME}_{self.cyto_seg_name}"].index.values)
            assert ids_nuc == ids_cyto, "Nucleus and cytosol segmentation masks do not match."

        self.get_project_status()

    #### Functions to perform processing ####

    def segment(self, overwrite: bool | None = None):
        # check to ensure a method has been assigned
        if self.segmentation_f is None:
            raise ValueError("No segmentation method defined")

        self.get_project_status()
        self._check_for_interactive_session()

        # ensure that an input image has been loaded
        if not self.input_image_status:
            raise ValueError("No input image loaded. Please load an input image first.")

        # setup overwrite if specified in call
        original_overwrite = self.segmentation_f.overwrite
        if overwrite is not None:
            self.segmentation_f.overwrite = overwrite

        if self.nuc_seg_status or self.cyto_seg_status:
            if not self.segmentation_f.overwrite:
                raise ValueError("Segmentation already exists. Set overwrite=True to overwrite.")

        if self.input_image is not None:
            self.segmentation_f.process(self.input_image)

        self.get_project_status()
        self.segmentation_f.overwrite = original_overwrite  # reset to original value

    def complete_segmentation(self, overwrite: bool | None = None, force_run: bool = False):
        """If a sharded Segmentation was run but individual tiles failed to segment properly, this method can be called to repeat the segmentation on the failed tiles only.
        Already calculated segmentation masks will not be recalculated.

        Args:
            overwrite: If set to ``None``, will read the overwrite value from the associated project.
                Otherwise can be set to a boolean value to override project specific settings for image loading.
            force_run: If set to ``True``, will force complete_segmentation to run even if a finalized segmentation mask is already found in the spatialdata object.
        """
        # check to ensure a method has been assigned
        if self.segmentation_f is None:
            raise ValueError("No segmentation method defined")

        self.get_project_status()
        self._check_for_interactive_session()

        # ensure that an input image has been loaded
        if not self.input_image_status:
            raise ValueError("No input image loaded. Please load an input image first.")

        # setup overwrite if specified in call
        original_overwrite = self.segmentation_f.overwrite
        if overwrite is not None:
            self.segmentation_f.overwrite = overwrite

        if self.nuc_seg_status or self.cyto_seg_status:
            if not self.segmentation_f.overwrite:
                raise ValueError("Segmentation already exists. Set overwrite = True to overwrite.")

        if self.input_image is not None:
            self.segmentation_f.complete_segmentation(self.input_image, force_run=force_run)

        self.get_project_status()
        self.segmentation_f.overwrite = original_overwrite  # reset to original value

    def extract(self, partial=False, n_cells=None, seed: int = 42, overwrite: bool | None = None) -> None:
        """Extract single-cell images from the input image using the defined extraction method.

        Args:
            partial: If set to ``True``, will run the extraction on a subset of the image. Default is ``False``.
            n_cells: Number of cells to extract if partial is ``True``
            seed: Seed for the random number generator during a partial extraction. Default is ``42``.
            overwrite: If set to ``None``, will read the overwrite value from the associated project.
                Otherwise can be set to a boolean value to override project specific settings for image loading

        Returns:
            None: Single-cell images are written to HDF5 files in the project associated extraction directory. File path can be accessed via ``project.extraction_f.output_path``.
        """
        if self.extraction_f is None:
            raise ValueError("No extraction method defined")

        # ensure that a segmentation has been stored that can be extracted
        self.get_project_status()
        self._check_for_interactive_session()

        if not (self.nuc_seg_status or self.cyto_seg_status):
            raise ValueError("No nucleus or cytosol segmentation loaded. Please load a segmentation first.")

        # setup overwrite if specified in call
        if overwrite is not None:
            self.extraction_f.overwrite_run_path = overwrite

        self.extraction_f(partial=partial, n_cells=n_cells, seed=seed)
        self.get_project_status()

    def featurize(
        self,
        n_cells: int = 0,
        data_type: Literal["complete", "partial", "filtered"] = "complete",
        partial_seed: None | int = None,
        overwrite: bool | None = None,
    ):
        if self.featurization_f is None:
            raise ValueError("No featurization method defined")

        self.get_project_status()
        self._check_for_interactive_session()

        # check that prerequisits are fullfilled to featurize cells
        assert self.featurization_f is not None, "No featurization method defined."
        assert (
            self.nuc_seg_status or self.cyto_seg_status
        ), "No nucleus or cytosol segmentation loaded. Please load a segmentation first."
        assert self.extraction_status, "No single cell data extracted. Please extract single cell data first."

        extraction_dir = self.extraction_f.get_directory()

        if data_type == "complete":
            cells_path = f"{extraction_dir}/data/{DEFAULT_EXTRACTION_FILE}"

        if data_type == "partial":
            partial_runs = [x for x in os.listdir(extraction_dir) if x.startswith("partial_data")]
            selected_runs = [x for x in partial_runs if f"ncells_{n_cells}" in x]

            if len(selected_runs) == 0:
                raise ValueError(f"No partial data found for n_cells = {n_cells}.")

            if len(selected_runs) > 1:
                if partial_seed is None:
                    raise ValueError(
                        f"Multiple partial data runs found for n_cells = {n_cells} with varying seed number. Please select one by specifying partial_seed."
                    )
                else:
                    selected_run = [x for x in selected_runs if f"seed_{partial_seed}" in x]
                    if len(selected_run) == 0:
                        raise ValueError(f"No partial data found for n_cells = {n_cells} and seed = {partial_seed}.")
                    else:
                        cells_path = f"{extraction_dir}/{selected_run[0]}/{self.DEFAULT_EXTRACTION_FILE}"
            else:
                cells_path = f"{extraction_dir}/{selected_runs[0]}/{self.DEFAULT_EXTRACTION_FILE}"

        if data_type == "filtered":
            raise ValueError("Filtered data not yet implemented.")

        print("Using extraction directory:", cells_path)

        # setup overwrite if specified in call
        if overwrite is not None:
            self.featurization_f.overwrite_run_path = overwrite

        # update the number of masks that are available in the segmentation object
        self.featurization_f.n_masks = sum([self.nuc_seg_status, self.cyto_seg_status])
        self.featurization_f.data_type = data_type

        self.featurization_f(cells_path, size=n_cells)

        self.get_project_status()

    def select(
        self,
        cell_sets: list[dict],
        calibration_marker: np.ndarray | None = None,
        name: str | None = None,
    ):
        """
        Select specified classes using the defined selection method.
        """

        if self.selection_f is None:
            raise ValueError("No selection method defined")

        self.get_project_status()
        self._check_for_interactive_session()

        if not self.nuc_seg_status and not self.cyto_seg_status:
            raise ValueError("No nucleus or cytosol segmentation loaded. Please load a segmentation first.")

        assert len(self.sdata._shared_keys) > 0, "sdata object is empty."

        self.selection_f(
            cell_sets=cell_sets,
            calibration_marker=calibration_marker,
            name=name,
        )
        self.get_project_status()
