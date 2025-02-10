import gc
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
import torch

from scportrait.pipeline._utils.helper import read_config


class Logable:
    """Create log entries.

    Args:
        debug (bool, default ``False``): When set to ``True`` log entries will be printed to the console.

    Attributes:
        directory (str): A directory must be set in every descendant before log can be called.
        DEFAULT_LOG_NAME (str, default ``processing.log``): Default log file name.
        DEFAULT_FORMAT (str): Date and time format used for logging. See `datetime.strftime <https://docs.python.org/3/library/datetime.html#datetime.date.strftime>`_.
    """

    DEFAULT_LOG_NAME = "processing.log"
    DEFAULT_FORMAT = "%d/%m/%Y %H:%M:%S"

    def __init__(self, directory, debug=False):
        self.directory = directory
        self.debug = debug

    def log(self, message):
        """log a message

        Args:
            message (str, list(str), dict(str)): Strings are s
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

    def get_timestamp(self):
        """
        Get the current timestamp in the DEFAULT_FORMAT.

        Returns:
            str: Formatted timestamp.
        """

        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime(self.DEFAULT_FORMAT)
        return "[" + dt_string + "] "

    def _clean_log_file(self):
        """Helper function to clean up log files in the processing step directory."""
        log_file_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)

        if os.path.exists(log_file_path):
            os.remove(log_file_path)

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

    Attributes:
        config (dict): Config file which is passed by the Project class when called. Is loaded from the project based on the name of the class.
        directory (str): Directory which should be used by the processing step. The directory will be newly created if it does not exist yet. When used with the :class:`sparcscore.pipeline.project.Project` class, a subdirectory of the project directory is passed.
        debug (bool, default ``False``): When set to True debug outputs will be printed where applicable.
        overwrite (bool, default ``False``): When set to True, the processing step directory will be completely deleted and newly created when called.
    """

    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_INPUT_IMAGE_NAME = "input_image"
    DEFAULT_SDATA_FILE = "sparcs.sdata"

    DEFAULT_PREFIX_MAIN_SEG = "seg_all"
    DEFAULT_PREFIX_FILTERED_SEG = "seg_filtered"
    DEFAULT_PREFIX_SELECTED_SEG = "seg_selected"

    DEFAULT_SEG_NAME_0 = "nucleus"
    DEFAULT_SEG_NAME_1 = "cytosol"

    DEFAULT_CENTERS_NAME = "centers"

    DEFAULT_CHUNK_SIZE = (1, 1000, 1000)

    DEFAULT_SEGMENTATION_DIR_NAME = "segmentation"
    DEFAULT_TILES_FOLDER = "tiles"

    DEFAULT_EXTRACTION_DIR_NAME = "extraction"
    DEFAULT_DATA_DIR = "data"

    DEFAULT_IMAGE_DTYPE = np.uint16
    DEFAULT_SEGMENTATION_DTYPE = np.uint32
    DEFAULT_SINGLE_CELL_IMAGE_DTYPE = np.float16

    DEFAULT_SEGMENTATION_FILE = "segmentation.h5"
    DEFAULT_CLASSES_FILE = "classes.csv"
    DEFAULT_REMOVED_CLASSES_FILE = "removed_classes.csv"
    DEFAULT_EXTRACTION_FILE = "single_cells.h5"

    DEFAULT_BENCHMARKING_FILE = "benchmarking.csv"

    DEFAULT_CLASSIFICATION_DIR_NAME = "classification"
    DEFAULT_SELECTION_DIR_NAME = "selection"

    def __init__(
        self,
        config,
        directory=None,
        project_location=None,
        debug=False,
        overwrite=False,
        project=None,
        filehandler=None,
        from_project: bool = False,
    ):
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

        if isinstance(config, str):
            config = read_config(config)
            if self.__class__.__name__ in config.keys():
                self.config = config[self.__class__.__name__]
            else:
                self.config = config
        else:
            self.config = config
        self.overwrite = overwrite

        self.get_context()

        self.deep_debug = False

        if "cache" not in self.config.keys():
            self.config["cache"] = os.path.abspath(os.getcwd())
            self.log(f"No cache directory specified in config using current working directory {self.config['cache']}.")

    def __call__(self, *args, debug=None, overwrite=None, **kwargs):
        """
        Call the processing step.

        Args:
            debug (bool, optional, default ``None``): Allows overriding the value set on initiation. When set to True debug outputs will be printed where applicable.
            overwrite (bool, optional, default ``None``): Allows overriding the value set on initiation. When set to True, the processing step directory will be completely deleted and newly created when called.
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
            x = self.process(*args, **kwargs)
            # clear temp directory after processing is completed
            if not self.deep_debug:
                self.clear_temp_dir()  # for deep debugging purposes, we keep the temp directory
            else:
                self.log("Deep debugging enabled, keeping temporary directory")
            return x
        else:
            self.clear_temp_dir()  # also ensure clearing if not callable just to make sure everything is cleaned up
            Warning("no process method defined.")

    def __call_empty__(self, *args, debug=None, overwrite=None, **kwargs):
        """Call the empty processing step.

        Args:
            debug (bool, optional, default ``None``): Allows overriding the value set on initiation. When set to True debug outputs will be printed where applicable.
            overwrite (bool, optional, default ``None``): Allows overriding the value set on initiation. When set to True, the processing step directory will be completely deleted and newly created when called.
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
            x = self.return_empty_mask(*args, **kwargs)
            return x
        else:
            Warning("no return_empty_mask method defined")

        # also clear empty temp directory here
        self.clear_temp_dir()

    def register_parameter(self, key, value):
        """
        Registers a new parameter by updating the configuration dictionary if the key didn't exist.

        Args:
            key (str): Name of the parameter.
            value: Value of the parameter.
        """

        if isinstance(key, str):
            config_handle = self.config

        elif isinstance(key, list):
            raise NotImplementedError("registration of parameters is not yet supported for nested parameters")

        else:
            raise TypeError("Key must be of string or a list of strings")

        if key not in config_handle:
            self.log(f"No configuration for {key} found, parameter will be set to {value}")
            config_handle[key] = value

    def get_directory(self):
        """
        Get the directory for this processing step.

        Returns:
            str: Directory path.
        """
        return self.directory

    def create_temp_dir(self):
        """
        Create a temporary directory in the cache directory specified in the config for saving all intermediate results.
        If "cache" not specified in the config for the method no directory will be created.
        """
        if "cache" in self.config.keys():
            path = os.path.join(self.config["cache"], f"{self.__class__.__name__}_")
            self._tmp_dir = tempfile.TemporaryDirectory(prefix=path)
            self._tmp_dir_path = self._tmp_dir.name

            self.log(f"Initialized temporary directory at {self._tmp_dir_path} for {self.__class__.__name__}")
        else:
            raise ValueError("No cache directory specified in config.")

    def clear_temp_dir(self):
        """Delete created temporary directory."""

        if "_tmp_dir" in self.__dict__.keys():
            shutil.rmtree(self._tmp_dir_path)
            self.log(f"Cleaned up temporary directory at {self._tmp_dir}")

            del self._tmp_dir, self._tmp_dir_path
        else:
            self.log("Temporary directory not found, skipping cleanup")

    def get_context(self):
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
