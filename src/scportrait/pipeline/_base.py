from __future__ import annotations

import gc
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Any

import torch

from scportrait.pipeline._utils.constants import (
    DEFAULT_BENCHMARKING_FILE,
    DEFAULT_CELL_ID_NAME,
    DEFAULT_CENTERS_NAME,
    DEFAULT_CHUNK_SIZE_2D,
    DEFAULT_CHUNK_SIZE_3D,
    DEFAULT_CLASSES_FILE,
    DEFAULT_CONFIG_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_EXTRACTION_DIR_NAME,
    DEFAULT_EXTRACTION_FILE,
    DEFAULT_FEATURIZATION_DIR_NAME,
    DEFAULT_FORMAT,
    DEFAULT_IMAGE_DTYPE,
    DEFAULT_INPUT_IMAGE_NAME,
    DEFAULT_LOG_NAME,
    DEFAULT_NAME_SINGLE_CELL_IMAGES,
    DEFAULT_PREFIX_FILTERED_SEG,
    DEFAULT_PREFIX_MAIN_SEG,
    DEFAULT_PREFIX_SELECTED_SEG,
    DEFAULT_REMOVED_CLASSES_FILE,
    DEFAULT_SDATA_FILE,
    DEFAULT_SEG_NAME_0,
    DEFAULT_SEG_NAME_1,
    DEFAULT_SEGMENTATION_DIR_NAME,
    DEFAULT_SEGMENTATION_DTYPE,
    DEFAULT_SEGMENTATION_FILE,
    DEFAULT_SELECTION_DIR_NAME,
    DEFAULT_SINGLE_CELL_IMAGE_DTYPE,
    DEFAULT_TILES_FOLDER,
    IMAGE_DATACONTAINER_NAME,
    INDEX_DATACONTAINER_NAME,
)
from scportrait.pipeline._utils.helper import read_config

if TYPE_CHECKING:
    from scportrait.pipeline._utils.sdata_io import sdata_filehandler
    from scportrait.pipeline.project import Project


class Logable:
    """Create log entries.

    Args:
        directory: path to a directory where the Logable entity should write it's log files.
        debug: When set to ``True`` log entries will be printed to the console otherwise they will only be written to the log file.

    Attributes:
        DEFAULT_LOG_NAME: Name of the log file.
        DEFAULT_FORMAT: Date and time format used for logging. See `datetime.strftime <https://docs.python.org/3/library/datetime.html#datetime.date.strftime>`_.
    """

    DEFAULT_LOG_NAME: str = DEFAULT_LOG_NAME
    DEFAULT_FORMAT: str = DEFAULT_FORMAT

    def __init__(self, directory: str | PosixPath, debug: bool = False):
        if isinstance(directory, PosixPath):
            directory = str(directory)
        self.directory = directory
        self.debug = debug

    def log(self, message: str | list[str] | dict[str, Any]):
        """log a message

        Args:
            message: strings which should be written to the log file. Can be a single string, a list of strings or a dictionary.
        """

        if not hasattr(self, "directory"):
            raise ValueError("Please define a valid self.directory in every descended of the Logable class")

        if isinstance(message, str):
            lines = message.split("\n")

        if isinstance(message, list):
            lines = message

        if isinstance(message, dict):
            lines = []
            for key, value in message.items():
                lines.append(f"{key}: {value}")

        else:
            try:
                lines = [str(message)]
            except (TypeError, ValueError):
                raise TypeError(
                    "Message must be a string, list of strings or a dictionary, but received type: ", type(message)
                ) from None

        for line in lines:
            log_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)

            # check that log path exists if not create
            if not os.path.isdir(self.directory):
                os.makedirs(self.directory)

            with open(log_path, "a") as myfile:
                myfile.write(self.get_timestamp() + line + " \n")

            if self.debug:
                print(self.get_timestamp() + line)

    def get_timestamp(self) -> str:
        """
        Get the current timestamp in the DEFAULT_FORMAT.

        Returns:
            Formatted timestamp.
        """

        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime(self.DEFAULT_FORMAT)
        return "[" + dt_string + "] "

    def _clean_log_file(self) -> None:
        """Helper function to clean up log files in the processing step directory.

        Returns:
            The log file is deleted on disk if it exists.
        """
        log_file_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)

        if os.path.exists(log_file_path):
            os.remove(log_file_path)

    ### THIS FUNCTION IS NOT WORKING AS INTENDED NEEDS TO BE FIXED
    # def _clear_cache(self, vars_to_delete=None):
    #     """Helper function to help clear memory usage. Mainly relevant for GPU based segmentations.

    #     Args:
    #         vars_to_delete (list): List of variable names (as strings) to delete.
    #     """

    #     # delete all specified variables
    #     if vars_to_delete is not None:
    #         for var_name in vars_to_delete:
    #             if var_name in globals():
    #                 del globals()[var_name]

    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     if torch.backends.mps.is_available():
    #         torch.mps.empty_cache()

    #     gc.collect()

    def _clear_cache(self, vars_to_delete=None):
        """Helper function to help clear memory usage. Mainly relevant for GPU based segmentations."""

        # delete all specified variables
        if vars_to_delete is not None:
            for var in vars_to_delete:
                del var

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        gc.collect()


class ProcessingStep(Logable):
    """Processing step. Can load a configuration file and create a subdirectory under the project class for the processing step.

    Args:
        config: Config file which is passed by the Project class when called. Is loaded from the project based on the name of the class.
        directory: Directory which should be used by the processing step. The directory will be newly created if it doesn't exist yet. When used with the :class:`scportrait.pipeline.project.Project` class, a subdirectory of the project directory is passed.
        debug: When set to True debug outputs will be printed where applicable. Otherwise they will only be written to file.
        overwrite: When set to True, the processing step directory will be completely deleted and newly created when called.
        project: Project class which is passed by the Project class when called.
        filehandler: Filehandler class which is passed by the Project class when called
        from_project: When set to True, the processing step is called from the Project class.
    """

    DEFAULT_CONFIG_NAME = DEFAULT_CONFIG_NAME
    DEFAULT_INPUT_IMAGE_NAME = DEFAULT_INPUT_IMAGE_NAME
    DEFAULT_SDATA_FILE = DEFAULT_SDATA_FILE

    DEFAULT_PREFIX_MAIN_SEG = DEFAULT_PREFIX_MAIN_SEG
    DEFAULT_PREFIX_FILTERED_SEG = DEFAULT_PREFIX_FILTERED_SEG
    DEFAULT_PREFIX_SELECTED_SEG = DEFAULT_PREFIX_SELECTED_SEG

    DEFAULT_SEG_NAME_0 = DEFAULT_SEG_NAME_0
    DEFAULT_SEG_NAME_1 = DEFAULT_SEG_NAME_1

    DEFAULT_CENTERS_NAME = DEFAULT_CENTERS_NAME

    DEFAULT_CHUNK_SIZE_3D = DEFAULT_CHUNK_SIZE_3D
    DEFAULT_CHUNK_SIZE_2D = DEFAULT_CHUNK_SIZE_2D

    DEFAULT_SEGMENTATION_DIR_NAME = DEFAULT_SEGMENTATION_DIR_NAME
    DEFAULT_TILES_FOLDER = DEFAULT_TILES_FOLDER

    DEFAULT_EXTRACTION_DIR_NAME = DEFAULT_EXTRACTION_DIR_NAME
    DEFAULT_DATA_DIR = DEFAULT_DATA_DIR

    DEFAULT_IMAGE_DTYPE = DEFAULT_IMAGE_DTYPE
    DEFAULT_SEGMENTATION_DTYPE = DEFAULT_SEGMENTATION_DTYPE
    DEFAULT_SINGLE_CELL_IMAGE_DTYPE = DEFAULT_SINGLE_CELL_IMAGE_DTYPE

    DEFAULT_SEGMENTATION_FILE = DEFAULT_SEGMENTATION_FILE
    DEFAULT_CLASSES_FILE = DEFAULT_CLASSES_FILE
    DEFAULT_REMOVED_CLASSES_FILE = DEFAULT_REMOVED_CLASSES_FILE
    DEFAULT_EXTRACTION_FILE = DEFAULT_EXTRACTION_FILE

    DEFAULT_BENCHMARKING_FILE = DEFAULT_BENCHMARKING_FILE

    DEFAULT_FEATURIZATION_DIR_NAME = DEFAULT_FEATURIZATION_DIR_NAME
    DEFAULT_SELECTION_DIR_NAME = DEFAULT_SELECTION_DIR_NAME

    IMAGE_DATACONTAINER_NAME = IMAGE_DATACONTAINER_NAME
    INDEX_DATACONTAINER_NAME = INDEX_DATACONTAINER_NAME
    DEFAULT_CELL_ID_NAME = DEFAULT_CELL_ID_NAME
    DEFAULT_NAME_SINGLE_CELL_IMAGES = DEFAULT_NAME_SINGLE_CELL_IMAGES

    deep_debug: bool = False  # flag to output more debug information

    def __init__(
        self,
        config: str | PosixPath | dict[str, Any],
        directory: str | PosixPath = None,
        project_location: str | PosixPath = None,
        debug: bool = False,
        overwrite: bool = False,
        project: Project | None = None,
        filehandler: sdata_filehandler | None = None,
        from_project: bool = False,
    ) -> None:
        super().__init__(directory=directory)

        self.debug = debug
        self.overwrite = overwrite
        if from_project:
            self.project_run = True
            self.project_location = project_location
            self.project = project
            self.filehandler = filehandler
        else:
            self.project_run = False
            self.project_location = None
            self.project = None
            self.filehandler = None

        raw_config: dict[str, Any]
        if isinstance(config, str | PosixPath):
            raw_config = read_config(config)
        else:
            raw_config = config

        class_config = raw_config.get(self.__class__.__name__)
        if isinstance(class_config, dict):
            self.config: dict[str, Any] = class_config
        else:
            self.config = raw_config
        self.overwrite = overwrite

        self.get_context()

        self.deep_debug = False

        if "cache" not in self.config:
            self.config["cache"] = os.path.abspath(os.getcwd())
            self.log(f"No cache directory specified in config using current working directory {self.config['cache']}.")

    def __call__(self, *args, debug: bool | None = None, overwrite: bool | None = None, **kwargs):
        """
        Call the processing step.

        Args:
            debug: Allows overriding the value set on initiation if not specified as None. When set to True debug outputs will be printed where applicable.
            overwrite: Allows overriding the value set on initiation if not specified as None. When set to True, the processing step directory will be completely deleted and newly created when called.
        """

        # set flags if provided
        self.debug = debug if debug is not None else self.debug
        self.overwrite = overwrite if overwrite is not None else self.overwrite

        # remove directory for processing step if overwrite is enabled
        if self.overwrite:
            if os.path.isdir(self.directory):
                shutil.rmtree(self.directory)

        # create directory for processing step
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # create a temporary directory for processing step
        self.create_temp_dir()
        if not os.path.isdir(self._tmp_dir_path):
            sys.exit("Temporary directory not found, exiting...")

        process = getattr(self, "process", None)
        if callable(process):
            x = self.process(*args, **kwargs)  # type: ignore[attr-defined]
            # clear temp directory after processing is completed
            if not self.deep_debug:
                self.clear_temp_dir()  # for deep debugging purposes, we keep the temp directory
            else:
                self.log("Deep debugging enabled, keeping temporary directory")
            return x
        else:
            self.clear_temp_dir()  # also ensure clearing if not callable just to make sure everything is cleaned up
            Warning("no process method defined.")

    def __call_empty__(self, *args, debug: bool | None = None, overwrite: bool | None = None, **kwargs):
        """Call the empty processing step.

        Args:
            debug: Allows overriding the value set on initiation if not specified as None. When set to True debug outputs will be printed where applicable.
            overwrite: Allows overriding the value set on initiation if not specified as None. When set to True, the processing step directory will be completely deleted and newly created when called.
        """
        # set flags if provided
        self.debug = debug if debug is not None else self.debug
        self.overwrite = overwrite if overwrite is not None else self.overwrite

        # remove directory for processing step if overwrite is enabled
        if self.overwrite:
            if os.path.isdir(self.directory):
                shutil.rmtree(self.directory)

        # create directory for processing step
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        process = getattr(self, "return_empty_mask", None)
        if callable(process):
            x = self.return_empty_mask(*args, **kwargs)  # type: ignore[attr-defined]
            return x
        else:
            Warning("no return_empty_mask method defined")

        # also clear empty temp directory here
        self.clear_temp_dir()

    def register_parameter(self, key: str, value: Any) -> None:
        """
        Registers a new parameter by updating the configuration dictionary with the provided key value pair.

        Args:
            key: Name of the parameter.
            value: Value of the parameter.
        """

        if isinstance(key, str):
            config_handle: dict[str, Any] = self.config

        elif isinstance(key, list):
            raise NotImplementedError("registration of parameters is not yet supported for nested parameters")
        else:
            raise TypeError("Key must be of string or a list of strings")

        if key not in config_handle:
            self.log(f"No configuration for {key} found, parameter will be set to {value}")
            config_handle[key] = value

    def get_directory(self) -> str:
        """
        Get the directory for this processing step.

        Returns:
            Directory path.
        """
        return self.directory

    def create_temp_dir(self) -> None:
        """
        Create a temporary directory at the location specified in `cache` of the config.
        If `cache` is not specified in the config for the method this will raise a ValueError.
        """
        if "cache" in self.config:
            path = os.path.join(str(self.config["cache"]), f"{self.__class__.__name__}_")
            self._tmp_dir = tempfile.TemporaryDirectory(prefix=path)
            self._tmp_dir_path = self._tmp_dir.name

            self.log(f"Initialized temporary directory at {self._tmp_dir_path} for {self.__class__.__name__}")
        else:
            raise ValueError("No cache directory specified in config.")

    def clear_temp_dir(self) -> None:
        """Delete created temporary directory."""

        if "_tmp_dir" in self.__dict__.keys():
            shutil.rmtree(self._tmp_dir_path)
            self.log(f"Cleaned up temporary directory at {self._tmp_dir}")

            del self._tmp_dir, self._tmp_dir_path
        else:
            if self.deep_debug:
                self.log("Temporary directory not found, skipping cleanup")

    def get_context(self) -> None:
        """
        Define context for multiprocessing steps that should be used.
        The context is platform dependent.
        """

        if platform.system() == "Windows":
            self.context = "spawn"
        elif platform.system() == "Darwin":
            self.context = "spawn"
        elif platform.system() == "Linux":
            self.context = "spawn"
