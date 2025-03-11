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
import warnings
from time import time
from typing import TYPE_CHECKING, Literal

import dask.array as darray
import numpy as np
import psutil
import xarray
from alphabase.io import tempmmap
from napari_spatialdata import Interactive
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from spatialdata import SpatialData
from tifffile import imread

from scportrait.io import daskmmap
from scportrait.pipeline._base import Logable
from scportrait.pipeline._utils.helper import read_config
from scportrait.pipeline._utils.sdata_io import sdata_filehandler
from scportrait.pipeline._utils.spatialdata_helper import (
    generate_region_annotation_lookuptable,
    get_chunk_size,
    rechunk_image,
    remap_region_annotation_table,
)

if TYPE_CHECKING:
    from collections.abc import Callable


from scportrait.pipeline._utils.constants import (
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
        self.centers_status = self.filehandler.centers_status
        extraction_file = os.path.join(
            self.project_location, self.DEFAULT_EXTRACTION_DIR_NAME, self.DEFAULT_DATA_DIR, self.DEFAULT_EXTRACTION_FILE
        )
        self.extraction_status = True if os.path.isfile(extraction_file) else False

        if self.input_image_status:
            if isinstance(self.sdata.images[self.DEFAULT_INPUT_IMAGE_NAME], xarray.DataTree):
                self.input_image = self.sdata.images[self.DEFAULT_INPUT_IMAGE_NAME]["scale0"].image
            elif isinstance(self.sdata.images[self.DEFAULT_INPUT_IMAGE_NAME], xarray.DataArray):
                self.input_image = self.sdata.images[self.DEFAULT_INPUT_IMAGE_NAME].image
            else:
                self.input_image = None

        if print_status:
            self.log("Current Project Status:")
            self.log("--------------------------------")
            self.log(f"Input Image in sdata: {self.input_image_status}")
            self.log(f"Nucleus Segmentation in sdata: {self.nuc_seg_status}")
            self.log(f"Cytosol Segmentation in sdata: {self.cyto_seg_status}")
            self.log(f"Centers in sdata: {self.centers_status}")
            self.log(f"Extracted single-cell images saved to file: {self.extraction_status}")
            self.log("--------------------------------")

        return None

    def view_sdata(self):
        """Start an interactive napari viewer to look at the sdata object associated with the project.
        Note:
            This only works in sessions with a visual interface.
        """
        # open interactive viewer in napari
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
        loc = parse_url(ome_zarr_path, mode="r")
        zarr_reader = Reader(loc).zarr
        image = zarr_reader.load("0")

        # Access the metadata to get channel names
        metadata = loc.root_attrs

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

    def load_input_from_sdata(
        self,
        sdata_path,
        input_image_name: str,
        nucleus_segmentation_name: str | None = None,
        cytosol_segmentation_name: str | None = None,
        overwrite: bool | None = None,
        keep_all: bool = True,
        remove_duplicates: bool = True,
        rechunk: bool = False,
    ) -> None:
        """
        Load input image from a spatialdata object.

        Args:
            sdata_path: Path to the spatialdata object.
            input_image_name: Name of the element in the spatial data object containing the input image.
            nucleus_segmentation_name: Name of the element in the spatial data object containing the nucleus segmentation mask. Default is ``None``.
            cytosol_segmentation_name: Name of the element in the spatial data object containing the cytosol segmentation mask. Default is ``None``.
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
        if keep_all:
            shutil.rmtree(self.sdata_path, ignore_errors=True)  # remove old sdata object
            sdata_input.write(self.sdata_path, overwrite=True)
            del sdata_input
            sdata_input = self.filehandler.get_sdata()

        self.get_project_status()

        # get input image and write it to the final sdata object
        image = sdata_input.images[input_image_name]
        self.log(f"Adding image {input_image_name} to sdata object as 'input_image'.")

        if isinstance(image, xarray.DataTree):
            image_c, image_x, image_y = image.scale0.image.shape

            # ensure chunking is correct
            if rechunk:
                for scale in image:
                    self._check_chunk_size(image[scale].image, chunk_size=self.DEFAULT_CHUNK_SIZE_3D)

            # get channel names
            channel_names = image.scale0.image.c.values

        elif isinstance(image, xarray.DataArray):
            image_c, image_x, image_y = image.shape

            # ensure chunking is correct
            if rechunk:
                self._check_chunk_size(image, chunk_size=self.DEFAULT_CHUNK_SIZE_3D)

            channel_names = image.c.values

        # Reset all transformations
        if image.attrs.get("transform"):
            self.log("Image contained transformations which which were removed.")
            image.attrs["transform"] = None

        # check coordinate system of input image
        ### PLACEHOLDER

        # check channel names
        self.log(
            f"Found the following channel names in the input image and saving in the spatialdata object: {channel_names}"
        )

        self.filehandler._write_image_sdata(image, self.DEFAULT_INPUT_IMAGE_NAME, channel_names=channel_names)

        # check if a nucleus segmentation exists and if so add it to the sdata object
        if nucleus_segmentation_name is not None:
            mask = sdata_input.labels[nucleus_segmentation_name]
            self.log(
                f"Adding nucleus segmentation mask '{nucleus_segmentation_name}' to sdata object as '{self.nuc_seg_name}'."
            )

            # if mask is multi-scale ensure we only use the scale 0
            if isinstance(mask, xarray.DataTree):
                mask = mask["scale0"].image

            # ensure that loaded masks are at the same scale as the input image
            mask_x, mask_y = mask.shape
            assert (mask_x == image_x) and (
                mask_y == image_y
            ), "Nucleus segmentation mask does not match input image size."

            if rechunk:
                self._check_chunk_size(mask, chunk_size=self.DEFAULT_CHUNK_SIZE_2D)  # ensure chunking is correct

            self.filehandler._write_segmentation_object_sdata(mask, self.nuc_seg_name)
            self.log(
                f"Calculating centers for nucleus segmentation mask {self.nuc_seg_name} and adding to spatialdata object."
            )
            self.filehandler._add_centers(segmentation_label=self.nuc_seg_name)

        # check if a cytosol segmentation exists and if so add it to the sdata object
        if cytosol_segmentation_name is not None:
            mask = sdata_input.labels[cytosol_segmentation_name]
            self.log(
                f"Adding cytosol segmentation mask '{cytosol_segmentation_name}' to sdata object as '{self.cyto_seg_name}'."
            )

            # if mask is multi-scale ensure we only use the scale 0
            if isinstance(mask, xarray.DataTree):
                mask = mask["scale0"].image

            # ensure that loaded masks are at the same scale as the input image
            mask_x, mask_y = mask.shape
            assert (mask_x == image_x) and (
                mask_y == image_y
            ), "Nucleus segmentation mask does not match input image size."

            if rechunk:
                self._check_chunk_size(mask, chunk_size=self.DEFAULT_CHUNK_SIZE_2D)  # ensure chunking is correct

            self.filehandler._write_segmentation_object_sdata(mask, self.cyto_seg_name)
            self.log(
                f"Calculating centers for cytosol segmentation mask {self.nuc_seg_name} and adding to spatialdata object."
            )
            self.filehandler._add_centers(segmentation_label=self.cyto_seg_name)

        self.get_project_status()

        # ensure that the provided nucleus and cytosol segmentations fullfill the scPortrait requirements
        # requirements are:
        # 1. The nucleus segmentation mask and the cytosol segmentation mask must contain the same ids
        # if self.nuc_seg_status and self.cyto_seg_status:
        # THIS NEEDS TO BE IMPLEMENTED HERE

        # 2. the nucleus segmentation ids and the cytosol segmentation ids need to match
        # THIS NEEDS TO BE IMPLEMENTED HERE

        # check if there are any annotations that match the nucleus/cytosol segmentations
        if self.nuc_seg_status or self.cyto_seg_status:
            region_annotation = generate_region_annotation_lookuptable(self.sdata)

            if self.nuc_seg_status:
                region_name = self.nuc_seg_name

                # add existing nucleus annotations if available
                if nucleus_segmentation_name in region_annotation.keys():
                    for x in region_annotation[nucleus_segmentation_name]:
                        table_name, table = x

                        new_table_name = f"annot_{region_name}_{table_name}"

                        table = remap_region_annotation_table(table, region_name=region_name)

                        self.filehandler._write_table_object_sdata(table, new_table_name)
                        self.log(
                            f"Added annotation {new_table_name} to spatialdata object for segmentation object {region_name}."
                        )

                        if keep_all and remove_duplicates:
                            self.log(
                                f"Deleting original annotation {table_name} for nucleus segmentation {nucleus_segmentation_name} from sdata object to prevent information duplication."
                            )
                            self.filehandler._force_delete_object(self.sdata, name=table_name, type="tables")
                else:
                    self.log(f"No region annotation found for the nucleus segmentation {nucleus_segmentation_name}.")

            # add cytosol segmentations if available
            if self.cyto_seg_status:
                if cytosol_segmentation_name in region_annotation.keys():
                    for x in region_annotation[cytosol_segmentation_name]:
                        table_name, table = x
                        region_name = self.cyto_seg_name
                        new_table_name = f"annot_{region_name}_{table_name}"

                        table = remap_region_annotation_table(table, region_name=region_name)
                        self.filehandler._write_table_object_sdata(table, new_table_name)

                        self.log(
                            f"Added annotation {new_table_name} to spatialdata object for segmentation object {region_name}."
                        )

                        if keep_all and remove_duplicates:
                            self.log(
                                f"Deleting original annotation {table_name} for cytosol segmentation {cytosol_segmentation_name} from sdata object to prevent information duplication."
                            )
                            self.filehandler._force_delete_object(self.sdata, name=table_name, type="tables")
                else:
                    self.log(f"No region annotation found for the cytosol segmentation {cytosol_segmentation_name}.")

        if keep_all and remove_duplicates:
            # remove input image
            self.log(f"Deleting input image '{input_image_name}' from sdata object to prevent information duplication.")
            self.filehandler._force_delete_object(self.sdata, name=input_image_name, type="images")

            if self.nuc_seg_status:
                self.log(
                    f"Deleting original nucleus segmentation mask '{nucleus_segmentation_name}' from sdata object to prevent information duplication."
                )
                self.filehandler._force_delete_object(self.sdata, name=nucleus_segmentation_name, type="labels")
            if self.cyto_seg_status:
                self.log(
                    f"Deleting original cytosol segmentation mask '{cytosol_segmentation_name}' from sdata object to prevent information duplication."
                )
                self.filehandler._force_delete_object(self.sdata, name=cytosol_segmentation_name, type="labels")

        self.get_project_status()
        self.overwrite = original_overwrite  # reset to original value

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
        if overwrite is None:
            self.featurization_f.overwrite_run_path = True

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
