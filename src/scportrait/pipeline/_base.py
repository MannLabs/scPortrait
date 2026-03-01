from __future__ import annotations

import gc
import os
import platform
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import PosixPath
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

    Attributes:
        DEFAULT_LOG_NAME: Name of the log file.
        DEFAULT_FORMAT: Date and time format used for logging. See `datetime.strftime <https://docs.python.org/3/library/datetime.html#datetime.date.strftime>`_.
    """

    DEFAULT_LOG_NAME: str = DEFAULT_LOG_NAME
    DEFAULT_FORMAT: str = DEFAULT_FORMAT

    def __init__(self, directory: str | PosixPath, debug: bool = False):
        """Initialize logging configuration.

        Args:
            directory: Directory where log files should be written.
            debug: If ``True``, log entries are also printed to stdout.
        """
        if isinstance(directory, PosixPath):
            directory = str(directory)
        self.directory = directory
        self.debug = debug

    def log(self, message: str | list[str] | dict[str, Any] | Any):
        """Write one or more messages to the step log file.

        Args:
            message: Message payload to write. Strings are split on newlines, lists are written line-by-line,
                dictionaries are written as ``key: value`` pairs, and other objects are coerced to ``str``.
        """

        if not hasattr(self, "directory"):
            raise ValueError("Please define a valid self.directory in every descendant of the Logable class")

        if isinstance(message, str):
            lines = message.split("\n")
        elif isinstance(message, list):
            lines = message
        elif isinstance(message, dict):
            lines = []
            for key, value in message.items():
                lines.append(f"{key}: {value}")
        else:
            lines = [str(message)]

        log_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)

        # check that log path exists if not create
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        for line in lines:
            with open(log_path, "a") as myfile:
                myfile.write(self.get_timestamp() + str(line) + " \n")

            if self.debug:
                print(self.get_timestamp() + str(line))

    def get_timestamp(self) -> str:
        """Get the current timestamp in ``DEFAULT_FORMAT``.

        Returns:
            Formatted timestamp.
        """

        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime(self.DEFAULT_FORMAT)
        return "[" + dt_string + "] "

    def _clean_log_file(self) -> None:
        """Delete the current log file if it exists."""
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
        """Release GPU/MPS cache and trigger garbage collection.

        Args:
            vars_to_delete: Optional iterable of local variables to delete references for before cleanup.
        """

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
    """Base class for executable pipeline stages in scPortrait.

    A `ProcessingStep` is the common runtime wrapper around stage-specific
    implementations such as segmentation, extraction, featurization, and
    selection. It provides shared behavior that all stages rely on:

    - configuration loading and step-specific config scoping
    - directory and overwrite handling
    - temporary workspace lifecycle management
    - unified logging and debug behavior
    - project-aware execution context when called via `Project`

    Subclasses are expected to implement stage logic via `process(...)` for
    standard execution, or `return_empty_mask(...)` for workflows that need a
    structured empty result path.
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
        """Initialize a processing step and normalize configuration handling.

        Args:
            config: Parsed configuration dictionary or path to a config file.
                If the top-level config contains a key matching the concrete
                step class name, that sub-dictionary is used as the step config.
            directory: Working directory for this step.
            project_location: Project root when running as part of ``Project``.
            debug: Enable verbose stdout logging in addition to file logging.
            overwrite: If ``True``, existing step output may be removed before
                processing.
            project: Active ``Project`` instance when this step is project-managed.
            filehandler: Shared SpatialData file handler for project-managed runs.
            from_project: Whether this step is invoked from a project-managed
                execution context.
        """
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

        self.get_context()

        self.deep_debug = False

        if "cache" not in self.config:
            self.config["cache"] = os.path.abspath(os.getcwd())
            self.log(f"No cache directory specified in config using current working directory {self.config['cache']}.")

    def __call__(self, *args, debug: bool | None = None, overwrite: bool | None = None, **kwargs):
        """Execute the processing step.

        This method runs a processing step from start to finish.

        Execution order:
            1. Apply optional runtime ``debug``/``overwrite`` overrides.
            2. If ``overwrite=True``, remove any existing step output directory.
            3. Ensure the step output directory exists.
            4. Create a temporary working directory in the configured cache.
            5. Call subclass ``process(...)`` if implemented.
            6. Clean up temporary files unless ``deep_debug=True``.

        Args:
            debug: Optional runtime override for debug logging.
            overwrite: Optional runtime override for overwrite behavior.

        Raises:
            RuntimeError: If temporary directory setup fails unexpectedly.
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
            raise RuntimeError("Temporary directory not found after initialization.")

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
            warnings.warn("No process method defined.", UserWarning, stacklevel=2)

    def __call_empty__(self, *args, debug: bool | None = None, overwrite: bool | None = None, **kwargs):
        """Execute ``return_empty_mask`` for workflows without normal processing.

        This is used for code paths where a step needs to return a valid
        empty placeholder output while still participating in the standard
        directory/setup lifecycle.

        Args:
            debug: Optional runtime override for debug logging.
            overwrite: Optional runtime override for overwrite behavior.
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
            warnings.warn("No return_empty_mask method defined.", UserWarning, stacklevel=2)

        # also clear empty temp directory here
        self.clear_temp_dir()

    def register_parameter(self, key: str | list[str], value: Any) -> None:
        """Register a missing configuration parameter.

        Args:
            key: Name of the parameter. Nested key registration via list is not yet supported.
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
        """Return the configured working directory for this step.

        Returns:
            Directory path.
        """
        return self.directory

    def create_temp_dir(self) -> None:
        """Create a temporary directory inside the configured cache directory.

        Raises:
            ValueError: If ``cache`` is missing from the step configuration.
        """
        if "cache" in self.config:
            cache_dir = str(self.config["cache"])
            os.makedirs(cache_dir, exist_ok=True)
            self._tmp_dir = tempfile.TemporaryDirectory(prefix=f"{self.__class__.__name__}_", dir=cache_dir)
            self._tmp_dir_path = self._tmp_dir.name

            self.log(f"Initialized temporary directory at {self._tmp_dir_path} for {self.__class__.__name__}")
        else:
            raise ValueError("No cache directory specified in config.")

    def clear_temp_dir(self) -> None:
        """Delete the temporary directory if one is currently active."""

        if "_tmp_dir" in self.__dict__:
            tmp_dir_path = self._tmp_dir_path
            self._tmp_dir.cleanup()
            self.log(f"Cleaned up temporary directory at {tmp_dir_path}")

            del self._tmp_dir, self._tmp_dir_path
        else:
            if self.deep_debug:
                self.log("Temporary directory not found, skipping cleanup")

    def get_context(self) -> None:
        """Define multiprocessing context used by the pipeline.

        Context selection is explicitly split by platform to make targeted
        OS-specific changes straightforward when backend behavior changes.
        """
        system = platform.system()
        if system == "Windows":
            self.context = "spawn"
        elif system == "Darwin":
            self.context = "spawn"
        elif system == "Linux":
            self.context = "spawn"
        else:
            self.context = "spawn"
