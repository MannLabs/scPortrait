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
import yaml
from alphabase.io import tempmmap
from napari_spatialdata import Interactive
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from spatialdata import SpatialData
from tifffile import imread

from scportrait.io import daskmmap
from scportrait.pipeline._base import Logable
from scportrait.pipeline._utils.sdata_io import sdata_filehandler
from scportrait.pipeline._utils.spatialdata_helper import (
    calculate_centroids,
    generate_region_annotation_lookuptable,
    get_chunk_size,
    get_unique_cell_ids,
    rechunk_image,
    remap_region_annotation_table,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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

    CLEAN_LOG: bool = True
    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_INPUT_IMAGE_NAME = "input_image"
    DEFAULT_SDATA_FILE = "scportrait.sdata"

    DEFAULT_PREFIX_MAIN_SEG = "seg_all"
    DEFAULT_PREFIX_FILTERED_SEG = "seg_filtered"
    DEFAULT_PREFIX_SELECTED_SEG = "seg_selected"

    DEFAULT_SEG_NAME_0: str = "nucleus"
    DEFAULT_SEG_NAME_1: str = "cytosol"

    DEFAULT_CENTERS_NAME: str = "centers_cells"

    DEFAULT_CHUNK_SIZE = (1, 1000, 1000)

    DEFAULT_SEGMENTATION_DIR_NAME = "segmentation"
    DEFAULT_EXTRACTION_DIR_NAME = "extraction"
    DEFAULT_DATA_DIR = "data"

    DEFAULT_FEATURIZATION_DIR_NAME = "featurization"

    DEFAULT_SELECTION_DIR_NAME = "selection"

    DEFAULT_IMAGE_DTYPE = np.uint16
    DEFAULT_SEGMENTATION_DTYPE = np.uint32
    DEFAULT_SINGLE_CELL_IMAGE_DTYPE = np.float16

    def __init__(
        self,
        project_location: str,
        config_path: str,
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

        self.sdata_path = self._get_sdata_path()
        self.sdata = None

        self.nuc_seg_name = f"{self.DEFAULT_PREFIX_MAIN_SEG}_{self.DEFAULT_SEG_NAME_0}"
        self.cyto_seg_name = f"{self.DEFAULT_PREFIX_MAIN_SEG}_{self.DEFAULT_SEG_NAME_1}"

        self.segmentation_f = segmentation_f
        self.extraction_f = extraction_f
        self.featurization_f = featurization_f
        self.selection_f = selection_f

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
        self._read_sdata()
        self._check_sdata_status()

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

        with open(file_path) as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

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
            )

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
            )

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
                overwrite=self.overwrite,
                project=self,
                filehandler=self.filehandler,
            )

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

    ##### General small helper functions ####

    def _check_memory(self, item):
        """
        Check the memory usage of the given if it were completely loaded into memory using .compute().
        """
        array_size = item.nbytes
        available_memory = psutil.virtual_memory().available

        return array_size < available_memory

    def _check_chunk_size(self, elem):
        """
        Check if the chunk size of the element is the default chunk size. If not rechunk the element to the default chunk size.
        """

        # get chunk size of element
        chunk_size = get_chunk_size(elem)

        if isinstance(chunk_size, list):
            # check if all chunk sizes are the same otherwise rechunking needs to occur anyways
            if not all(x == chunk_size[0] for x in chunk_size):
                elem = rechunk_image(elem, chunks=self.DEFAULT_CHUNK_SIZE)
            else:
                # ensure that the chunk size is the default chunk size
                if chunk_size != self.DEFAULT_CHUNK_SIZE:
                    elem = rechunk_image(elem, chunks=self.DEFAULT_CHUNK_SIZE)
        else:
            # ensure that the chunk size is the default chunk size
            if chunk_size != self.DEFAULT_CHUNK_SIZE:
                elem = rechunk_image(elem, chunks=self.DEFAULT_CHUNK_SIZE)

        return elem

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
                self.log(f"Output location {self.sdata_path} already exists. Overwriting.")
                shutil.rmtree(self.sdata_path, ignore_errors=True)
            else:
                # check to see if the sdata object is empty
                if len(os.listdir(self.sdata_path)) == 0:
                    self.log(
                        f"Output location {self.sdata_path} already exists but does not contain any data. Overwriting."
                    )
                    shutil.rmtree(self.sdata_path, ignore_errors=True)
                else:
                    raise ValueError(
                        f"Output location {self.sdata_path} already exists. Set overwrite=True to overwrite."
                    )

        self._read_sdata()

    def _get_sdata_path(self):
        """
        Get the path to the spatialdata object.
        """
        return os.path.join(self.project_location, self.DEFAULT_SDATA_FILE)

    def _ensure_all_labels_habe_cell_ids(self):
        """Helper function to readd cell-ids to labels objects after reloading until a more permanent solution can be found"""
        for keys in list(self.sdata.labels.keys()):
            if not hasattr(self.sdata.labels[keys].attrs, "cell_ids"):
                self.sdata.labels[keys].attrs["cell_ids"] = get_unique_cell_ids(self.sdata.labels[keys])

    def print_project_status(self):
        """Print the current project status."""
        self._check_sdata_status(print_status=True)

    def _check_sdata_status(self, print_status=False):
        if self.sdata is None:
            self._read_sdata()
        else:
            self.sdata = self.filehandler._check_sdata_status(return_sdata=True)
            self.input_image_status = self.filehandler.input_image_status
            self.nuc_seg_status = self.filehandler.nuc_seg_status
            self.cyto_seg_status = self.filehandler.cyto_seg_status
            self.centers_status = self.filehandler.centers_status

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
            self.log(f"Input Image Status: {self.input_image_status}")
            self.log(f"Nucleus Segmentation Status: {self.nuc_seg_status}")
            self.log(f"Cytosol Segmentation Status: {self.cyto_seg_status}")
            self.log(f"Centers Status: {self.centers_status}")
            self.log("--------------------------------")

        return None

    def _read_sdata(self):
        self.sdata = self.filehandler.get_sdata()  # type: SpatialData
        self._check_sdata_status()

    def view_sdata(self):
        """Start an interactive napari viewer to look at the sdata object associated with the project.
        Note:
            This only works in sessions with a visual interface.
        """
        self.sdata = self.filehandler.get_sdata()  # ensure its up to date

        # open interactive viewer in napari
        interactive = Interactive(self.sdata)
        interactive.run()

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
        image = darray.from_array(array, chunks=self.DEFAULT_CHUNK_SIZE)

        if remap is not None:
            image = image[remap]
            self.channel_names = [self.channel_names[i] for i in remap]

        # write to sdata object
        self.filehandler._write_image_sdata(
            image,
            channel_names=self.channel_names,
            scale_factors=[2, 4, 8],
            chunks=self.DEFAULT_CHUNK_SIZE,
            image_name=self.DEFAULT_INPUT_IMAGE_NAME,
        )

        self._check_sdata_status()
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
            chunks=self.DEFAULT_CHUNK_SIZE,
        )

        self.sdata = None
        self.overwrite = original_overwrite  # reset to original value

        # cleanup variables and temp dir
        self._clear_cache(vars_to_delete=[temp_image_path, im, channels])
        self._clear_temp_dir()

        # strange workaround that is required so that the sdata input image does not point to the dask array anymore but
        # to the image which was written to disk
        self._check_sdata_status()

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

        # read entire data into memory
        time_start = time()
        input_image = np.array(
            zarr_reader.load("0").compute()
        )  ### adapt here to not read the entire image to memory TODO
        time_end = time()
        self.log(f"Read input image from file {ome_zarr_path} to numpy array in {(time_end - time_start)/60} minutes.")

        # Access the metadata to get channel names
        zarr_group = loc.zarr_group()
        metadata = zarr_group.attrs.asdict()

        if "omero" in metadata and "channels" in metadata["omero"]:
            channels = metadata["omero"]["channels"]
            channel_names = [channel["label"] for channel in channels]
        else:
            channel_names = [f"channel_{i}" for i in range(input_image.shape[0])]

        # write loaded array to sdata object
        self.load_input_from_array(input_image, channel_names=channel_names)

        self._check_sdata_status()
        self.overwrite = original_overwrite  # reset to original value

    def load_input_from_sdata(
        self,
        sdata_path,
        input_image_name="input_image",
        nucleus_segmentation_name=None,
        cytosol_segmentation_name=None,
        overwrite=None,
    ):
        """
        Load input image from a spatialdata object.
        """

        # setup overwrite
        original_overwrite = self.overwrite
        if overwrite is not None:
            self.overwrite = overwrite

        # check if an input image was already loaded if so throw error if overwrite = False
        self._cleanup_sdata_object()

        # read input sdata object
        sdata_input = SpatialData.read(sdata_path)

        # get input image and write it to the final sdata object
        image = sdata_input.images[input_image_name]

        # ensure chunking is correct
        image = self._check_chunk_size(image)

        # check coordinate system of input image
        ### PLACEHOLDER

        self.filehandler._write_image_sdata(image, self.DEFAULT_INPUT_IMAGE_NAME)
        self.input_image_status = True

        # check if a nucleus segmentation exists and if so add it to the sdata object
        if nucleus_segmentation_name is not None:
            mask = sdata_input.labels[nucleus_segmentation_name]
            mask = self._check_chunk_size(mask)  # ensure chunking is correct

            self.filehandler._write_segmentation_object_sdata(mask, self.nuc_seg_name)

            self.nuc_seg_status = True
            self.log("Nucleus segmentation saved under the label {nucleus_segmentation_name} added to sdata object.")

        # check if a cytosol segmentation exists and if so add it to the sdata object
        if cytosol_segmentation_name is not None:
            mask = sdata_input.labels[cytosol_segmentation_name]
            mask = self._check_chunk_size(mask)  # ensure chunking is correct

            self.filehandler_write_segmentation_object_sdata(mask, self.cyto_seg_name)

            self.cyto_seg_status = True
            self.log("Cytosol segmentation saved under the label {nucleus_segmentation_name} added to sdata object.")

        # ensure that the provided nucleus and cytosol segmentations fullfill the scPortrait requirements
        # requirements are:
        # 1. The nucleus segmentation mask and the cytosol segmentation mask must contain the same ids
        assert (
            self.sdata[self.nuc_seg_name].attrs["cell_ids"] == self.sdata[self.cyto_seg_name].attrs["cell_ids"]
        ), "The nucleus segmentation mask and the cytosol segmentation mask must contain the same ids."

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
                else:
                    self.log(f"No region annotation found for the nucleus segmentation {nucleus_segmentation_name}.")

                # add centers of cells for available nucleus map
                centroids = calculate_centroids(self.sdata.labels[region_name], coordinate_system="global")
                self._write_points_object_sdata(centroids, self.DEFAULT_CENTERS_NAME)

                self.centers_status = True

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
                else:
                    self.log(f"No region annotation found for the cytosol segmentation {cytosol_segmentation_name}.")

        self._check_sdata_status()
        self.overwrite = original_overwrite  # reset to original value

    #### Functions to perform processing ####

    def segment(self, overwrite: bool | None = None):
        # check to ensure a method has been assigned
        if self.segmentation_f is None:
            raise ValueError("No segmentation method defined")

        self._check_sdata_status()
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

        elif self.input_image is not None:
            self.segmentation_f(self.input_image)

        self._check_sdata_status()
        self.segmentation_f.overwrite = original_overwrite  # reset to original value
        self.sdata = self.filehandler.get_sdata()  # update

    def complete_segmentation(self, overwrite: bool | None = None):
        """If a sharded Segmentation was run but individual tiles failed to segment properly, this method can be called to repeat the segmentation on the failed tiles only.
        Already calculated segmentation masks will not be recalculated.
        """
        # check to ensure a method has been assigned
        if self.segmentation_f is None:
            raise ValueError("No segmentation method defined")

        self._check_sdata_status()
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

        elif self.input_image is not None:
            self.segmentation_f.complete_segmentation(self.input_image)

        self._check_sdata_status()
        self.segmentation_f.overwrite = original_overwrite  # reset to original value

    def extract(self, partial=False, n_cells=None, overwrite: bool | None = None):
        if self.extraction_f is None:
            raise ValueError("No extraction method defined")

        # ensure that a segmentation has been stored that can be extracted
        self._check_sdata_status()

        if not (self.nuc_seg_status or self.cyto_seg_status):
            raise ValueError("No nucleus or cytosol segmentation loaded. Please load a segmentation first.")

        # setup overwrite if specified in call
        if overwrite is not None:
            self.extraction_f.overwrite_run_path = overwrite

        self.extraction_f(partial=partial, n_cells=n_cells)
        self._check_sdata_status()

    def featurize(
        self,
        n_cells: int = 0,
        data_type: Literal["complete", "partial", "filtered"] = "complete",
        partial_seed: None | int = None,
        overwrite: bool | None = None,
    ):
        if self.featurization_f is None:
            raise ValueError("No featurization method defined")

        self._check_sdata_status()

        if not (self.nuc_seg_status or self.cyto_seg_status):
            raise ValueError("No nucleus or cytosol segmentation loaded. Please load a segmentation first.")

        extraction_dir = self.extraction_f.get_directory()

        if data_type == "complete":
            cells_path = f"{extraction_dir}/data/single_cells.h5"

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
                        cells_path = f"{extraction_dir}/{selected_run[0]}/single_cells.h5"
            else:
                cells_path = f"{extraction_dir}/{selected_runs[0]}/single_cells.h5"

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

        self._check_sdata_status()

    def select(
        self,
        cell_sets: list[dict],
        calibration_marker: np.ndarray | None = None,
        segmentation_name: str = "seg_all_nucleus",
        name: str | None = None,
    ):
        """
        Select specified classes using the defined selection method.
        """

        if self.selection_f is None:
            raise ValueError("No selection method defined")

        self._check_sdata_status()

        if not self.nuc_seg_status or not self.cyto_seg_status:
            raise ValueError("No nucleus or cytosol segmentation loaded. Please load a segmentation first.")

        assert self.sdata is not None, "No sdata object loaded."
        assert segmentation_name in self.sdata.labels, f"Segmentation {segmentation_name} not found in sdata object."

        self.selection_f(
            segmentation_name=segmentation_name,
            cell_sets=cell_sets,
            calibration_marker=calibration_marker,
            name=name,
        )
        self._check_sdata_status()


# this class has not yet been set up to be used with spatialdata
# class TimecourseProject(Project):
#     """
#     TimecourseProject class used to create a scPortrait project for datasets that have multiple fields of view that should be processed and analysed together.
#     It is also capable of handling multiple timepoints for the same field of view or a combiantion of both. Like the base scPortrait :func:`Project <sparcscore.pipeline.project.Project>`,
#     it manages all of the scPortrait processing steps. Because the input data has a different dimensionality than the base scPortrait :func:`Project <sparcscore.pipeline.project.Project>` class,
#     it requires the use of specialized processing classes that are able to handle this additional dimensionality.

#     Parameters
#     ----------
#     location_path : str
#         Path to the folder where to project should be created. The folder is created in case the specified folder does not exist.
#     config_path : str, optional, default ""
#         Path pointing to a valid configuration file. The file will be copied to the project directory and renamed to the name specified in ``DEFAULT_FEATURIZATION_DIR_NAME``. If no config is specified, the existing config in the project directory will be used, if possible. See the section configuration to find out more about the config file.
#     debug : bool, default False
#         When set to True debug outputs will be printed where applicable.
#     overwrite : bool, default False
#         When set to True, the processing step directory will be completely deleted and newly created when called.
#     segmentation_f : Class, default None
#         Class containing segmentation workflow.
#     extraction_f : Class, default None
#         Class containing extraction workflow.
#     featurization_f : Class, default None
#         Class containing featurization workflow.
#     selection_f : Class, default None
#         Class containing selection workflow.

#     Attributes
#     ----------
#     DEFAULT_CONFIG_NAME : str, default "config.yml"
#         Default config name which is used for the config file in the project directory. This name needs to be used when no config is supplied and the config is manually created in the project folder.
#     DEFAULT_INPUT_IMAGE_NAME: str, default "input_segmentation.h5"
#         Default file name for loading the input image.
#     DEFAULT_SEGMENTATION_DIR_NAME : str, default "segmentation"
#         Default foldername for the segmentation process.
#     DEFAULT_EXTRACTION_DIR_NAME : str, default "extraction"
#         Default foldername for the extraction process.
#     DEFAULT_FEATURIZATION_DIR_NAME : str, default "selection"
#         Default foldername for the featurization process.
#     DEFAULT_SELECTION_DIR_NAME : str, default "featurization"
#         Default foldername for the selection process.
#     """

#     DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def load_input_from_array(self, img, label, overwrite=False):
#         """
#         Function to load imaging data from an array into the TimecourseProject.

#         The provided array needs to fullfill the following conditions:
#         - shape: NCYX
#         - all images need to have the same dimensions and the same number of channels
#         - channels need to be in the following order: nucleus, cytosol other channels
#         - dtype uint16.

#         Parameters
#         ----------
#         img : numpy.ndarray
#             Numpy array of shape “[num_images, channels, height, width]”.
#         label : numpy.ndarray
#             Numpy array of shape “[num_images, num_labels]” containing the labels for each image. The labels need to have the following structure: "image_index", "unique_image_identifier", "..."
#         overwrite : bool, default False
#             If set to True, the function will overwrite the existing input image.
#         """

#         """
#         Function to load imaging data from an array into the TimecourseProject.

#         The provided array needs to fullfill the following conditions:
#         - shape: NCYX
#         - all images need to have the same dimensions and the same number of channels
#         - channels need to be in the following order: nucleus, cytosol other channels
#         - dtype uint16.

#         Parameters
#         ----------
#         img : numpy.ndarray
#             Numpy array of shape “[num_images, channels, height, width]”.
#         label : numpy.ndarray
#             Numpy array of shape “[num_images, num_labels]” containing the labels for each image. The labels need to have the following structure: "image_index", "unique_image_identifier", "..."
#         overwrite : bool, default False
#             If set to True, the function will overwrite the existing input image.
#         """

#         # check if already exists if so throw error message
#         if not os.path.isdir(
#             os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#         ):
#             os.makedirs(
#                 os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#             )

#         path = os.path.join(
#             self.directory,
#             self.DEFAULT_SEGMENTATION_DIR_NAME,
#             self.DEFAULT_INPUT_IMAGE_NAME,
#         )

#         if not overwrite:
#             if os.path.isfile(path):
#                 sys.exit("File already exists")
#             else:
#                 overwrite = True

#         if overwrite:
#             # column labels
#             column_labels = label.columns.to_list()

#             # create .h5 dataset to which all results are written
#             path = os.path.join(
#                 self.directory,
#                 self.DEFAULT_SEGMENTATION_DIR_NAME,
#                 self.DEFAULT_INPUT_IMAGE_NAME,
#             )
#             hf = h5py.File(path, "w")
#             dt = h5py.special_dtype(vlen=str)
#             hf.create_dataset("label_names", data=column_labels, chunks=None, dtype=dt)
#             hf.create_dataset(
#                 "labels", data=label.astype(str).values, chunks=None, dtype=dt
#             )
#             hf.create_dataset(
#                 "input_images", data=img, chunks=(1, 1, img.shape[2], img.shape[2])
#             )

#             hf.close()

#     def load_input_from_files(
#         self,
#         input_dir,
#         channels,
#         timepoints,
#         plate_layout,
#         img_size=1080,
#         overwrite=False,
#     ):
#         """
#         Function to load timecourse experiments recorded with an opera phenix into the TimecourseProject.

#         Before being able to use this function the exported images from the opera phenix first need to be parsed, sorted and renamed using the `sparcstools package <https://github.com/MannLabs/SPARCStools>`_.

#         In addition a plate layout file needs to be created that contains the information on imaged experiment and the experimental conditions for each well. This file needs to be in the following format,
#         using the well notation ``RowXX_WellXX``:

#         .. csv-table::
#             :header: "Well", "Condition1", "Condition2", ...
#             :widths: auto

#             "RowXX_WellXX", "A", "B", ...

#         A tab needs to be used as a seperator and the file saved as a .tsv file.

#         Parameters
#         ----------
#         input_dir : str
#             Path to the directory containing the sorted images from the opera phenix.
#         channels : list(str)
#             List containing the names of the channels that should be loaded.
#         timepoints : list(str)
#             List containing the names of the timepoints that should be loaded. Will return a warning if you try to load a timepoint that is not found in the data.
#         plate_layout : str
#             Path to the plate layout file. For the format please see above.
#         img_size : int, default 1080
#             Size of the images that should be loaded. All images will be cropped to this size.
#         overwrite : bool, default False
#             If set to True, the function will overwrite the existing input image.

#         Example
#         -------
#         >>> channels = ["DAPI", "Alexa488", "mCherry"]
#         >>> timepoints = ["Timepoint"+str(x).zfill(3) for x in list(range(1, 3))]
#         >>> input_dir = "path/to/sorted/outputs/from/sparcstools"
#         >>> plate_layout = "plate_layout.tsv"

#         >>> project.load_input_from_files(input_dir = input_dir,  channels = channels,  timepoints = timepoints, plate_layout = plate_layout, overwrite = True)

#         Function to load timecourse experiments recorded with an opera phenix into the TimecourseProject.

#         Before being able to use this function the exported images from the opera phenix first need to be parsed, sorted and renamed using the `sparcstools package <https://github.com/MannLabs/SPARCStools>`_.

#         In addition a plate layout file needs to be created that contains the information on imaged experiment and the experimental conditions for each well. This file needs to be in the following format,
#         using the well notation ``RowXX_WellXX``:

#         .. csv-table::
#             :header: "Well", "Condition1", "Condition2", ...
#             :widths: auto

#             "RowXX_WellXX", "A", "B", ...

#         A tab needs to be used as a seperator and the file saved as a .tsv file.

#         Parameters
#         ----------
#         input_dir : str
#             Path to the directory containing the sorted images from the opera phenix.
#         channels : list(str)
#             List containing the names of the channels that should be loaded.
#         timepoints : list(str)
#             List containing the names of the timepoints that should be loaded. Will return a warning if you try to load a timepoint that is not found in the data.
#         plate_layout : str
#             Path to the plate layout file. For the format please see above.
#         img_size : int, default 1080
#             Size of the images that should be loaded. All images will be cropped to this size.
#         overwrite : bool, default False
#             If set to True, the function will overwrite the existing input image.

#         Example
#         -------
#         >>> channels = ["DAPI", "Alexa488", "mCherry"]
#         >>> timepoints = ["Timepoint"+str(x).zfill(3) for x in list(range(1, 3))]
#         >>> input_dir = "path/to/sorted/outputs/from/sparcstools"
#         >>> plate_layout = "plate_layout.tsv"

#         >>> project.load_input_from_files(input_dir = input_dir,  channels = channels,  timepoints = timepoints, plate_layout = plate_layout, overwrite = True)

#         """

#         # check if already exists if so throw error message
#         if not os.path.isdir(
#             os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#         ):
#             os.makedirs(
#                 os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#             )

#         path = os.path.join(
#             self.directory,
#             self.DEFAULT_SEGMENTATION_DIR_NAME,
#             self.DEFAULT_INPUT_IMAGE_NAME,
#         )

#         if not overwrite:
#             if os.path.isfile(path):
#                 sys.exit("File already exists")
#             else:
#                 overwrite = True

#         if overwrite:
#             self.img_size = img_size

#             def _read_write_images(dir, indexes, h5py_path):
#                 # unpack indexes
#                 index_start, index_end = indexes

#                 # get information on directory
#                 well = re.search(
#                     "Row.._Well[0-9][0-9]", dir
#                 ).group()  # need to use re.search and not match sinde the identifier is not always at the beginning of the name
#                 region = re.search("r..._c...$", dir).group()

#                 # list all images within directory
#                 path = os.path.join(input_dir, dir)
#                 files = os.listdir(path)

#                 # filter to only contain the timepoints of interest
#                 files = np.sort([x for x in files if x.startswith(tuple(timepoints))])

#                 # checkt to make sure all timepoints are actually there
#                 _timepoints = np.unique(
#                     [re.search("Timepoint[0-9][0-9][0-9]", x).group() for x in files]
#                 )

#                 sum = 0
#                 for timepoint in timepoints:
#                     if timepoint in _timepoints:
#                         sum += 1
#                         continue
#                     else:
#                         print(f"No images found for Timepoint {timepoint}")

#                 self.log(
#                     f"{sum} different timepoints found of the total {len(timepoints)} timepoints given."
#                 )

#                 # read images for that region
#                 imgs = np.empty(
#                     (n_timepoints, n_channels, img_size, img_size), dtype="uint16"
#                 )
#                 for ix, channel in enumerate(channels):
#                     images = [x for x in files if channel in x]

#                     for i, im in enumerate(images):
#                         image = imread(os.path.join(path, im))

#                         if isinstance(image.dtype, np.uint8):
#                             image = image.astype("uint16") * np.iinfo(np.uint8).max

#                         self._check_image_dtype(image)
#                         imgs[i, ix, :, :] = image.astype("uint16")

#                 # create labelling
#                 column_values = []
#                 for column in plate_layout.columns:
#                     column_values.append(plate_layout.loc[well, column])

#                 list_input = [
#                     list(range(index_start, index_end)),
#                     [dir + "_" + x for x in timepoints],
#                     [dir] * n_timepoints,
#                     timepoints,
#                     [well] * n_timepoints,
#                     [region] * n_timepoints,
#                 ]
#                 list_input = [np.array(x) for x in list_input]

#                 for x in column_values:
#                     list_input.append(np.array([x] * n_timepoints))

#                 labelling = np.array(list_input).T

#                 input_images[index_start:index_end, :, :, :] = imgs
#                 labels[index_start:index_end] = labelling

#             # read plate layout
#             plate_layout = pd.read_csv(plate_layout, sep="\s+|;|,", engine="python")
#             plate_layout = plate_layout.set_index("Well")

#             column_labels = [
#                 "index",
#                 "ID",
#                 "location",
#                 "timepoint",
#                 "well",
#                 "region",
#             ] + plate_layout.columns.tolist()

#             # get information on number of timepoints and number of channels
#             n_timepoints = len(timepoints)
#             n_channels = len(channels)
#             wells = np.unique(plate_layout.index.tolist())

#             # get all directories contained within the input dir
#             directories = os.listdir(input_dir)
#             if ".DS_Store" in directories:
#                 directories.remove(
#                     ".DS_Store"
#                 )  # need to remove this because otherwise it gives errors
#             if ".ipynb_checkpoints" in directories:
#                 directories.remove(".ipynb_checkpoints")

#             # filter directories to only contain those listed in the plate layout
#             directories = [
#                 _dir
#                 for _dir in directories
#                 if re.search("Row.._Well[0-9][0-9]", _dir).group() in wells
#             ]

#             # check to ensure that imaging data is found for all wells listed in plate_layout
#             _wells = [
#                 re.search("Row.._Well[0-9][0-9]", _dir).group() for _dir in directories
#             ]
#             not_found = [well for well in _wells if well not in wells]
#             if len(not_found) > 0:
#                 print(
#                     "following wells listed in plate_layout not found in imaging data:",
#                     not_found,
#                 )
#                 self.log(
#                     f"following wells listed in plate_layout not found in imaging data: {not_found}"
#                 )

#             # check to make sure that timepoints given and timepoints found in data acutally match!
#             _timepoints = []

#             # create .h5 dataset to which all results are written
#             path = os.path.join(
#                 self.directory,
#                 self.DEFAULT_SEGMENTATION_DIR_NAME,
#                 self.DEFAULT_INPUT_IMAGE_NAME,
#             )

#             # for some reason this directory does not always exist so check to make sure it does otherwise the whole reading of stuff fails
#             if not os.path.isdir(
#                 os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#             ):
#                 os.makedirs(
#                     os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#                 )

#             with h5py.File(path, "w") as hf:
#                 dt = h5py.special_dtype(vlen=str)
#                 hf.create_dataset(
#                     "label_names", (len(column_labels)), chunks=None, dtype=dt
#                 )
#                 hf.create_dataset(
#                     "labels",
#                     (len(directories) * n_timepoints, len(column_labels)),
#                     chunks=None,
#                     dtype=dt,
#                 )

#                 hf.create_dataset(
#                     "input_images",
#                     (len(directories) * n_timepoints, n_channels, img_size, img_size),
#                     chunks=(1, 1, img_size, img_size),
#                     dtype="uint16",
#                 )

#                 label_names = hf.get("label_names")
#                 labels = hf.get("labels")
#                 input_images = hf.get("input_images")

#                 label_names[:] = column_labels

#                 # ------------------
#                 # start reading data
#                 # ------------------

#                 indexes = []
#                 # create indexes
#                 start_index = 0
#                 for i, _ in enumerate(directories):
#                     stop_index = start_index + n_timepoints
#                     indexes.append((start_index, stop_index))
#                     start_index = stop_index

#                 # iterate through all directories and add to .h5
#                 # this is not implemented with multithreaded processing because writing multi-threaded to hdf5 is hard
#                 # multithreaded reading is easier

#                 for dir, index in tqdm(
#                     zip(directories, indexes), total=len(directories)
#                 ):
#                     _read_write_images(dir, index, h5py_path=path)

#     def load_input_from_stitched_files(
#         self,
#         input_dir,
#         channels,
#         timepoints,
#         plate_layout,
#         overwrite=False,
#     ):
#         """
#         Function to load timecourse experiments recorded with opera phenix into .h5 dataformat for further processing.
#         Assumes that stitched images for all files have already been assembled.

#         Args:
#             input_dir (str): path to directory containing the stitched images
#             channels (list(str)): list of strings indicating which channels should be loaded
#             timepoints (list(str)): list of strings indicating which timepoints should be loaded
#             plate_layout (str): path to csv file containing the plate layout
#             overwrite (bool, optional): boolean indicating if existing files should be overwritten. Defaults to False.
#         """

#         # check if already exists if so throw error message
#         if not os.path.isdir(
#             os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#         ):
#             os.makedirs(
#                 os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#             )

#         path = os.path.join(
#             self.directory,
#             self.DEFAULT_SEGMENTATION_DIR_NAME,
#             self.DEFAULT_INPUT_IMAGE_NAME,
#         )

#         if not overwrite:
#             if os.path.isfile(path):
#                 sys.exit("File already exists")
#             else:
#                 overwrite = True

#         if overwrite:

#             def _read_write_images(well, indexes, h5py_path):
#                 # unpack indexes
#                 index_start, index_end = indexes

#                 # list all images for that well
#                 _files = [_file for _file in files if well in _file]

#                 # filter to only contain the timepoints of interest
#                 _files = np.sort([x for x in _files if x.startswith(tuple(timepoints))])

#                 # checkt to make sure all timepoints are actually there
#                 _timepoints = np.unique(
#                     [re.search("Timepoint[0-9][0-9][0-9]", x).group() for x in _files]
#                 )

#                 sum = 0
#                 for timepoint in timepoints:
#                     if timepoint in _timepoints:
#                         sum += 1
#                         continue
#                     else:
#                         print(f"No images found for Timepoint {timepoint}")

#                 self.log(
#                     f"{sum} different timepoints found of the total {len(timepoints)} timepoints given."
#                 )

#                 # read images for that region
#                 imgs = np.empty(
#                     (n_timepoints, n_channels, size1, size2), dtype="uint16"
#                 )
#                 for ix, channel in enumerate(channels):
#                     images = [x for x in _files if channel in x]

#                     for i, im in enumerate(images):
#                         image = imread(os.path.join(input_dir, im), 0).astype("uint16")

#                         # check if image is too small and if yes, pad the image with black pixels
#                         if image.shape[0] < size1 or image.shape[1] < size2:
#                             image = np.pad(
#                                 image,
#                                 (
#                                     (0, np.max((size1 - image.shape[0], 0))),
#                                     (0, np.max((size2 - image.shape[1], 0))),
#                                 ),
#                                 mode="constant",
#                                 constant_values=0,
#                             )
#                             self.log(
#                                 f"Image {im} with the index {i} is too small and was padded with black pixels. "
#                                 f"Image shape after padding: {image.shape}."
#                             )

#                         # perform cropping so that all stitched images have the same size
#                         x, y = image.shape
#                         diff1 = x - size1
#                         diff1x = int(np.floor(diff1 / 2))
#                         diff1y = int(np.ceil(diff1 / 2))
#                         diff2 = y - size2
#                         diff2x = int(np.floor(diff2 / 2))
#                         diff2y = int(np.ceil(diff2 / 2))

#                         cropped = image[
#                             slice(diff1x, x - diff1y), slice(diff2x, y - diff2y)
#                         ]

#                         imgs[i, ix, :, :] = cropped

#                 # create labelling
#                 column_values = []
#                 for column in plate_layout.columns:
#                     column_values.append(plate_layout.loc[well, column])

#                 list_input = [
#                     list(range(index_start, index_end)),
#                     [well + "_" + x for x in timepoints],
#                     [well] * n_timepoints,
#                     timepoints,
#                     [well] * n_timepoints,
#                 ]
#                 list_input = [np.array(x) for x in list_input]

#                 for x in column_values:
#                     list_input.append(np.array([x] * n_timepoints))

#                 labelling = np.array(list_input).T

#                 input_images[index_start:index_end, :, :, :] = imgs
#                 labels[index_start:index_end] = labelling

#             # read plate layout
#             plate_layout = pd.read_csv(plate_layout, sep="\s+|;|,", engine="python")
#             plate_layout = plate_layout.set_index("Well")

#             column_labels = [
#                 "index",
#                 "ID",
#                 "location",
#                 "timepoint",
#                 "well",
#             ] + plate_layout.columns.tolist()

#             # get information on number of timepoints and number of channels
#             n_timepoints = len(timepoints)
#             n_channels = len(channels)
#             wells = np.unique(plate_layout.index.tolist())

#             # get all files contained within the input dir
#             files = os.listdir(input_dir)
#             files = [file for file in files if file.endswith(".tif")]

#             # filter directories to only contain those listed in the plate layout
#             files = [
#                 _dir
#                 for _dir in files
#                 if re.search("Row.._Well[0-9][0-9]", _dir).group() in wells
#             ]

#             # check to ensure that imaging data is found for all wells listed in plate_layout
#             _wells = [re.search("Row.._Well[0-9][0-9]", _dir).group() for _dir in files]
#             not_found = [well for well in _wells if well not in wells]
#             if len(not_found) > 0:
#                 print(
#                     "following wells listed in plate_layout not found in imaging data:",
#                     not_found,
#                 )
#                 self.log(
#                     f"following wells listed in plate_layout not found in imaging data: {not_found}"
#                 )

#             # get image size and subtract 10 pixels from each edge
#             # will adjust all merged images to this dimension to ensure that they all have the same dimensions and can be loaded into the same hdf5 file
#             size1, size2 = imagesize.get(os.path.join(input_dir, files[0]))
#             size1 = size1 - 2 * 10
#             size2 = size2 - 2 * 10
#             self.img_size = (size1, size2)

#             # create .h5 dataset to which all results are written
#             path = os.path.join(
#                 self.directory,
#                 self.DEFAULT_SEGMENTATION_DIR_NAME,
#                 self.DEFAULT_INPUT_IMAGE_NAME,
#             )

#             # for some reason this directory does not always exist so check to make sure it does otherwise the whole reading of stuff fails
#             if not os.path.isdir(
#                 os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#             ):
#                 os.makedirs(
#                     os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#                 )

#             with h5py.File(path, "w") as hf:
#                 dt = h5py.special_dtype(vlen=str)
#                 hf.create_dataset(
#                     "label_names", (len(column_labels)), chunks=None, dtype=dt
#                 )
#                 hf.create_dataset(
#                     "labels",
#                     (len(wells) * n_timepoints, len(column_labels)),
#                     chunks=None,
#                     dtype=dt,
#                 )
#                 hf.create_dataset(
#                     "input_images",
#                     (len(wells) * n_timepoints, n_channels, size1, size2),
#                     chunks=(1, 1, size1, size2),
#                     dtype="uint16",
#                 )

#                 label_names = hf.get("label_names")
#                 labels = hf.get("labels")
#                 input_images = hf.get("input_images")

#                 label_names[:] = column_labels

#                 # ------------------
#                 # start reading data
#                 # ------------------

#                 indexes = []
#                 # create indexes
#                 start_index = 0
#                 for i, _ in enumerate(wells):
#                     stop_index = start_index + n_timepoints
#                     indexes.append((start_index, stop_index))
#                     start_index = stop_index

#                 # iterate through all directories and add to .h5
#                 # this is not implemented with multithreaded processing because writing multi-threaded to hdf5 is hard
#                 # multithreaded reading is easier

#                 for well, index in tqdm(zip(wells, indexes), total=len(wells)):
#                     _read_write_images(well, index, h5py_path=path)

#     def load_input_from_files_and_merge(
#         self,
#         input_dir,
#         channels,
#         timepoints,
#         plate_layout,
#         img_size=1080,
#         stitching_channel="Alexa488",
#         overlap=0.1,
#         max_shift=10,
#         overwrite=False,
#         nucleus_channel="DAPI",
#         cytosol_channel="Alexa488",
#     ):
#         """
#         Function to load timecourse experiments recorded with an opera phenix into a TimecourseProject. In addition to loading the images,
#         this wrapper function also stitches images acquired in the same well (this assumes that the tiles were aquired with overlap and in a rectangular shape)
#         using the `sparcstools package <https://github.com/MannLabs/SPARCStools>`_. Implementation of this function is currently still slow for many wells/timepoints as stitching
#         is handled consecutively and not in parallel. This will be fixed in the future.

#         Parameters
#         ----------
#         input_dir : str
#             Path to the directory containing the sorted images from the opera phenix.
#         channels : list(str)
#             List containing the names of the channels that should be loaded.
#         timepoints : list(str)
#             List containing the names of the timepoints that should be loaded. Will return a warning if you try to load a timepoint that is not found in the data.
#         plate_layout : str
#             Path to the plate layout file. For the format please see above.
#         img_size : int, default 1080
#             Size of the images that should be loaded. All images will be cropped to this size.
#         stitching_channel : str, default "Alexa488"
#             string indicated on which channel the stitching should be calculated.
#         overlap : float, default 0.1
#             float indicating the overlap between the tiles that were aquired.
#         max_shift : int, default 10
#             int indicating the maximum shift that is allowed when stitching the tiles. If a calculated shift is larger than this threshold
#             between two tiles then the position of these tiles is not updated and is set according to the calculated position based on the overlap.
#         overwrite : bool, default False
#             If set to True, the function will overwrite the existing input image.
#         nucleus_channel : str, default "DAPI"
#             string indicating the channel that should be used for the nucleus channel.
#         cytosol_channel : str, default "Alexa488"
#             string indicating the channel that should be used for the cytosol channel.

#         """

#         from sparcstools.stitch import generate_stitched

#         # check if already exists if so throw error message
#         if not os.path.isdir(
#             os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#         ):
#             os.makedirs(
#                 os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#             )

#         path = os.path.join(
#             self.directory,
#             self.DEFAULT_SEGMENTATION_DIR_NAME,
#             self.DEFAULT_INPUT_IMAGE_NAME,
#         )

#         if not overwrite:
#             if os.path.isfile(path):
#                 sys.exit("File already exists.")
#             else:
#                 overwrite = True

#         if overwrite:
#             self.img_size = img_size

#             self.log(f"Reading all images included in directory {input_dir}.")

#             images = os.listdir(input_dir)
#             images = [x for x in images if x.endswith((".tiff", ".tif"))]

#             _timepoints = np.sort(list(set([x.split("_")[0] for x in images])))
#             _wells = np.sort(
#                 list(
#                     set(
#                         [
#                             re.match(".*_Row[0-9][0-9]_Well[0-9][0-9]", x).group()[13:]
#                             for x in images
#                         ]
#                     )
#                 )
#             )

#             # apply filtering to only get those that are in the plate layout file
#             plate_layout = pd.read_csv(plate_layout, sep="\s+|;|,", engine="python")
#             plate_layout = plate_layout.set_index("Well")

#             column_labels = [
#                 "index",
#                 "ID",
#                 "location",
#                 "timepoint",
#                 "well",
#                 "region",
#             ] + plate_layout.columns.tolist()

#             # get information on number of timepoints and number of channels
#             n_timepoints = len(timepoints)
#             n_channels = len(channels)
#             wells = np.unique(plate_layout.index.tolist())

#             _wells = [x for x in _wells if x in wells]
#             _timepoints = [x for x in _timepoints if x in timepoints]

#             not_found_wells = [well for well in _wells if well not in wells]
#             not_found_timepoints = [
#                 timepoint for timepoint in _timepoints if timepoint not in timepoints
#             ]

#             if len(not_found_wells) > 0:
#                 print(
#                     "following wells listed in plate_layout not found in imaging data:",
#                     not_found_wells,
#                 )
#                 self.log(
#                     f"following wells listed in plate_layout not found in imaging data: {not_found_wells}"
#                 )

#             if len(not_found_timepoints) > 0:
#                 print(
#                     "following timepoints given not found in imaging data:",
#                     not_found_timepoints,
#                 )
#                 self.log(
#                     f"following timepoints given not found in imaging data: {not_found_timepoints}"
#                 )

#             self.log("Will perform merging over the following specs:")
#             self.log(f"Wells: {_wells}")
#             self.log(f"Timepoints: {_timepoints}")

#             # create .h5 dataset to which all results are written
#             path = os.path.join(
#                 self.directory,
#                 self.DEFAULT_SEGMENTATION_DIR_NAME,
#                 self.DEFAULT_INPUT_IMAGE_NAME,
#             )

#             # for some reason this directory does not always exist so check to make sure it does otherwise the whole reading of stuff fails
#             if not os.path.isdir(
#                 os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#             ):
#                 os.makedirs(
#                     os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
#                 )

#             with h5py.File(path, "w") as hf:
#                 dt = h5py.special_dtype(vlen=str)
#                 hf.create_dataset(
#                     "label_names", (len(column_labels)), chunks=None, dtype=dt
#                 )
#                 hf.create_dataset(
#                     "labels",
#                     (len(_wells) * n_timepoints, len(column_labels)),
#                     chunks=None,
#                     dtype=dt,
#                 )
#                 label_names = hf.get("label_names")
#                 labels = hf.get("labels")

#                 label_names[:] = column_labels

#                 run_number = 0
#                 for timepoint in tqdm(_timepoints):
#                     for well in tqdm(_wells):
#                         RowID = well.split("_")[0]
#                         WellID = well.split("_")[1]
#                         zstack_value = 1

#                         # define patter to recognize which slide should be stitched
#                         # remember to adjust the zstack value if you aquired zstacks and want to stitch a speciifc one in the parameters above

#                         pattern = (
#                             f"{timepoint}_{RowID}_{WellID}"
#                             + "_{channel}_"
#                             + "zstack"
#                             + str(zstack_value).zfill(3)
#                             + "_r{row:03}_c{col:03}.tif"
#                         )

#                         merged_images, channels = generate_stitched(
#                             input_dir,
#                             well,
#                             pattern,
#                             outdir="/",
#                             overlap=overlap,
#                             max_shift=max_shift,
#                             do_intensity_rescale=True,
#                             stitching_channel=stitching_channel,
#                             filetype="return_array",
#                             export_XML=False,
#                             plot_QC=False,
#                         )

#                         if run_number == 0:
#                             img_size1 = merged_images.shape[1] - 2 * 10
#                             img_size2 = merged_images.shape[2] - 2 * 10
#                             # create this after the first image is stitched and we have the dimensions
#                             hf.create_dataset(
#                                 "input_images",
#                                 (
#                                     len(_wells) * n_timepoints,
#                                     n_channels,
#                                     img_size1,
#                                     img_size2,
#                                 ),
#                                 chunks=(1, 1, img_size1, img_size2),
#                             )
#                             input_images = hf.get("input_images")

#                         # crop so that all images have the same size
#                         _, x, y = merged_images.shape
#                         diff1 = x - img_size1
#                         diff1x = int(np.floor(diff1 / 2))
#                         diff1y = int(np.ceil(diff1 / 2))
#                         diff2 = y - img_size2
#                         diff2x = int(np.floor(diff2 / 2))
#                         diff2y = int(np.ceil(diff2 / 2))
#                         cropped = merged_images[
#                             :, slice(diff1x, x - diff1y), slice(diff2x, y - diff2y)
#                         ]

#                         # create labelling
#                         column_values = []
#                         for column in plate_layout.columns:
#                             column_values.append(plate_layout.loc[well, column])

#                         list_input = [
#                             str(run_number),
#                             f"{well}_{timepoint}_all",
#                             f"{well}_all",
#                             timepoint,
#                             well,
#                             "stitched",
#                         ]

#                         for x in column_values:
#                             list_input.append(x)

#                         # reorder to fit to timecourse sorting
#                         allocated_channels = []
#                         allocated_indexes = []
#                         if nucleus_channel in channels:
#                             nucleus_index = channels.index(nucleus_channel)
#                             allocated_channels.append(nucleus_channel)
#                             allocated_indexes.append(nucleus_index)
#                         else:
#                             print("nucleus_channel not found in supplied channels!!!")

#                         if cytosol_channel in channels:
#                             cytosol_index = channels.index(cytosol_channel)
#                             allocated_channels.append(cytosol_channel)
#                             allocated_indexes.append(cytosol_index)
#                         else:
#                             print("cytosol_channel not found in supplied channels!!!")

#                         all_other_indexes = [
#                             channels.index(x)
#                             for x in channels
#                             if x not in allocated_channels
#                         ]
#                         all_other_indexes = list(np.sort(all_other_indexes))

#                         index_list = allocated_indexes + all_other_indexes
#                         cropped = np.array([cropped[x, :, :] for x in index_list])

#                         self.log(
#                             f"adjusted channels to the following order: {[channels[i] for i in index_list]}"
#                         )
#                         input_images[run_number, :, :, :] = cropped
#                         labels[run_number] = list_input
#                         run_number += 1
#                         self.log(
#                             f"finished stitching and saving well {well} for timepoint {timepoint}."
#                         )

#     def adjust_segmentation_indexes(self):
#         self.segmentation_f.adjust_segmentation_indexes()

#     def segment(self, overwrite=False, *args, **kwargs):
#         """
#         segment timecourse project with the defined segmentation method.
#         """

#         if overwrite:
#             # delete segmentation and classes from .hdf5 to be able to create new again
#             path = os.path.join(
#                 self.directory,
#                 self.DEFAULT_SEGMENTATION_DIR_NAME,
#                 self.DEFAULT_INPUT_IMAGE_NAME,
#             )
#             with h5py.File(path, "a") as hf:
#                 if "segmentation" in hf.keys():
#                     del hf["segmentation"]
#                 if "classes" in hf.keys():
#                     del hf["classes"]

#             # delete generated files to make clean
#             classes_path = os.path.join(
#                 self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv"
#             )
#             log_path = os.path.join(
#                 self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, "processing.log"
#             )
#             if os.path.isfile(classes_path):
#                 os.remove(classes_path)
#             if os.path.isfile(log_path):
#                 os.remove(log_path)

#             print("If Segmentation already existed removed.")

#         if self.segmentation_f is None:
#             raise ValueError("No segmentation method defined")

#         else:
#             self.segmentation_f(*args, **kwargs)

#     def extract(self, *args, **kwargs):
#         """
#         Extract single cells from a timecourse project with the defined extraction method.
#         """

#         if self.extraction_f is None:
#             raise ValueError("No extraction method defined")

#         input_segmentation = self.segmentation_f.get_output()
#         input_dir = os.path.join(
#             self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv"
#         )
#         self.extraction_f(input_segmentation, input_dir, *args, **kwargs)
