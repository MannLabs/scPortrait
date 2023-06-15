# -*- coding: utf-8 -*-
import warnings
import shutil
import os
import yaml
from PIL import Image
import PIL
import numpy as np
import sys

# packages for timecourse project
import pandas as pd
from cv2 import imread
import re
import h5py
from tqdm import tqdm

from sparcscore.pipeline.base import Logable


class Project(Logable):
    """
    Project base class used to create a SPARCSpy project.

    Parameters
    ----------
    location_path : str
        Path to the folder where to project should be created. The folder is created in case the specified folder does not exist.
    config_path : str, optional, default ""
        Path pointing to a valid configuration file. The file will be copied to the project directory and renamed to the name specified in ``DEFAULT_CLASSIFICATION_DIR_NAME``. If no config is specified, the existing config in the project directory will be used, if possible. See the section configuration to find out more about the config file.
    intermediate_output : bool, default False
        When set to True intermediate outputs will be saved where applicable.
    debug : bool, default False
        When set to True debug outputs will be printed where applicable.
    overwrite : bool, default False
        When set to True, the processing step directory will be completely deleted and newly created when called.
    segmentation_f : Class, default None
        Class containing segmentation workflow.
    extraction_f : Class, default None
        Class containing extraction workflow.
    classification_f : Class, default None
        Class containing classification workflow.
    selection_f : Class, default None
        Class containing selection workflow.

    Attributes
    ----------
    DEFAULT_CONFIG_NAME : str, default "config.yml"
        Default config name which is used for the config file in the project directory. This name needs to be used when no config is supplied and the config is manually created in the project folder.
    DEFAULT_SEGMENTATION_DIR_NAME : str, default "segmentation"
        Default foldername for the segmentation process.
    DEFAULT_EXTRACTION_DIR_NAME : str, default "extraction"
        Default foldername for the extraction process.
    DEFAULT_CLASSIFICATION_DIR_NAME : str, default "selection"
        Default foldername for the classification process.
    DEFAULT_SELECTION_DIR_NAME : str, default "classification"
        Default foldername for the selection process.

    """

    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_SEGMENTATION_DIR_NAME = "segmentation"
    DEFAULT_EXTRACTION_DIR_NAME = "extraction"
    DEFAULT_CLASSIFICATION_DIR_NAME = "classification"
    DEFAULT_SELECTION_DIR_NAME = "selection"

    # Project object is initialized, nothing is written to disk
    def __init__(
            self,
            location_path,
            config_path="",
            *args,
            intermediate_output=False,
            debug=False,
            overwrite=False,
            segmentation_f=None,
            extraction_f=None,
            classification_f=None,
            selection_f=None,
            **kwargs,
    ):
        super().__init__(debug=debug)

        self.debug = debug
        self.overwrite = overwrite
        self.intermediate_output = intermediate_output

        self.segmentation_f = segmentation_f
        self.extraction_f = extraction_f
        self.classification_f = classification_f
        self.selection_f = selection_f

        # PIL limit used to protect from large image attacks
        PIL.Image.MAX_IMAGE_PIXELS = 10000000000

        self.input_image = None
        self.config = None

        self.directory = location_path

        # handle location
        self.project_location = location_path

        # check if project dir exists and creates it if not
        if not os.path.isdir(self.project_location):
            os.makedirs(self.project_location)
        else:
            warnings.warn("Theres already a directory in the location path")

        # handle configuration file
        new_config_path = os.path.join(self.project_location, self.DEFAULT_CONFIG_NAME)

        if config_path == "":
            # Check if there is already a config file in the dataset folder in case no config file has been specified

            if os.path.isfile(new_config_path):
                self._load_config_from_file(new_config_path)

            else:
                warnings.warn(
                    f"You will need to add a config named {self.DEFAULT_CONFIG_NAME} file manually to the dataset"
                )

        else:
            if not os.path.isfile(config_path):
                raise ValueError("Your config path is invalid")

            else:
                print("modifying config")
                if os.path.isfile(new_config_path):
                    os.remove(new_config_path)

                # The blueprint config file is copied to the dataset folder and renamed to the default name
                shutil.copy(config_path, new_config_path)
                self._load_config_from_file(new_config_path)

        # === setup segmentation ===
        if self.segmentation_f is not None:
            if segmentation_f.__name__ not in self.config:
                raise ValueError(
                    f"Config for {segmentation_f.__name__} is missing from the config file"
                )

            seg_directory = os.path.join(
                self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME
            )
            self.segmentation_f = segmentation_f(
                self.config[segmentation_f.__name__],
                seg_directory,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
        else:
            self.segmentation_f = None

        # === setup extraction ===
        if extraction_f is not None:
            extraction_directory = os.path.join(
                self.project_location, self.DEFAULT_EXTRACTION_DIR_NAME
            )

            if extraction_f.__name__ not in self.config:
                raise ValueError(
                    f"Config for {extraction_f.__name__} is missing from the config file"
                )

            self.extraction_f = extraction_f(
                self.config[extraction_f.__name__],
                extraction_directory,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
        else:
            self.extraction_f = None

        # === setup classification ===
        if classification_f is not None:
            if classification_f.__name__ not in self.config:
                raise ValueError(
                    f"Config for {classification_f.__name__} is missing from the config file"
                )

            classification_directory = os.path.join(
                self.project_location, self.DEFAULT_CLASSIFICATION_DIR_NAME
            )
            self.classification_f = classification_f(
                self.config[classification_f.__name__],
                classification_directory,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
        else:
            self.classification_f = None

        # === setup selection ===
        if selection_f is not None:
            if selection_f.__name__ not in self.config:
                raise ValueError(
                    f"Config for {selection_f.__name__} is missing from the config file"
                )

            selection_directory = os.path.join(
                self.project_location, self.DEFAULT_SELECTION_DIR_NAME
            )
            self.selection_f = selection_f(
                self.config[selection_f.__name__],
                selection_directory,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
        else:
            self.selection_f = None

        # parse remapping
        self.remap = None
        if "channel_remap" in self.config:
            char_list = self.config["channel_remap"].split(",")
            self.log("channel remap parameter found:")
            self.log(char_list)

            self.remap = [int(el.strip()) for el in char_list]

    def _load_config_from_file(self, file_path):
        """
        loads config from file and writes it to self.config

        Parameters
        ----------
        file_path : str
            Path to the config.yml file that should be loaded.

        """
        self.log(f"Loading config from {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError("Your config path is invalid")

        with open(file_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def load_input_from_file(self, file_paths, crop=[(0, -1), (0, -1)]):
        """
        Load input image from a number of files.

        Parameters
        ----------
        file_paths : list(str)
            List containing paths to each channel like
            [“path1/img.tiff”, “path2/img.tiff”, “path3/img.tiff”].
            Expects a list of file paths with length “input_channel” as
            defined in the config.yml. Input data is NOT copied to the
            project folder by default. Different segmentation functions
            especially tiled segmentations might copy the input.

        crop : list(tuple), optional
            When set, it can be used to crop the input image. The first
            element refers to the first dimension of the image and so on.
            For example use “[(0,1000),(0,2000)]” to crop the image to
            1000 px height and 2000 px width from the top left corner.

        """
        if self.config is None:
            raise ValueError("Dataset has no config file loaded")

        # remap can be used to shuffle the order, for example [1, 0, 2] to invert the first two channels
        # default order that is expected: Nucleus channel, cell membrane channel, other channels

        if not len(file_paths) == self.config["input_channels"]:
            raise ValueError(
                "Expected {} image paths, only received {}".format(
                    self.config["input_channels"], len(file_paths)
                )
            )

        # append all images channel wise and remap them according to the supplied list
        channels = []

        for channel_path in file_paths:
            im = Image.open(channel_path)
            c = np.array(im, dtype="float64")[slice(*crop[0]), slice(*crop[1])]

            channels.append(c)

        self.input_image = np.stack(channels)

        print(self.input_image.shape)

        if self.remap is not None:
            self.input_image = self.input_image[self.remap]

    def load_input_from_array(self, array, remap=None):
        """
        Load input image from an already loaded numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            Numpy array of shape “[channels, height, width]”.

        remap : list(int), optional
            Define remapping of channels. For example use “[1, 0, 2]”
            to change the order of the first and the second channel.
            The expected order is Nucleus Channel, Cellmembrane Channel
            followed by other channels.

        """
        # input data is not copied to the project folder
        if self.config is None:
            raise ValueError("Dataset has no config file loaded")

        if not array.shape[0] == self.config["input_channels"]:
            raise ValueError(
                "Expected {} image paths, only received {}".format(
                    self.config["input_channels"], array.shape[0]
                )
            )

        self.input_image = np.array(array, dtype="float64")

        if self.remap is not None:
            self.input_image = self.input_image[self.remap]

    def segment(self, *args, **kwargs):
        """
        Segment project with the selected segmentation method.
        """

        if self.segmentation_f is None:
            raise ValueError("No segmentation method defined")
        elif type(self.input_image) is None:
            raise ValueError("No input image defined")
        else:
            self.segmentation_f(self.input_image, *args, **kwargs)

    def extract(self, *args, **kwargs):
        """
        Extract single cells with the defined extraction method.
        """

        if self.extraction_f is None:
            raise ValueError("No extraction method defined")

        input_segmentation = self.segmentation_f.get_output()
        input_dir = os.path.join(
            self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv"
        )
        self.extraction_f(input_segmentation, input_dir, *args, **kwargs)

    def classify(self, *args, **kwargs):
        """
        Classify extracted single cells with the defined classification method.
        """

        input_extraction = self.extraction_f.get_output_path()

        if not os.path.isdir(input_extraction):
            raise ValueError("input was not found at {}".format(input_extraction))

        self.classification_f(input_extraction, *args, **kwargs)

    def select(self, *args, **kwargs):
        """
        Select specified classes using the defined selection method.
        """

        if self.selection_f is None:
            raise ValueError("No selection method defined")
            pass

        input_selection = self.segmentation_f.get_output()

        self.selection_f(input_selection, *args, **kwargs)

    def process(self):
        self.segment()
        self.extract()
        self.classify()


class TimecourseProject(Project):
    """
    Timecourse Project used to create a SPARCSpy project for datasets that have multiple timepoints
    over the same field of view (add additional dimension in comparision to base SPARCSpy project).

    Parameters
    ----------
    location_path : str
        Path to the folder where to project should be created. The folder is created in case the specified folder does not exist.
    config_path : str, optional, default ""
        Path pointing to a valid configuration file. The file will be copied to the project directory and renamed to the name specified in ``DEFAULT_CLASSIFICATION_DIR_NAME``. If no config is specified, the existing config in the project directory will be used, if possible. See the section configuration to find out more about the config file.
    intermediate_output : bool, default False
        When set to True intermediate outputs will be saved where applicable.
    debug : bool, default False
        When set to True debug outputs will be printed where applicable.
    overwrite : bool, default False
        When set to True, the processing step directory will be completely deleted and newly created when called.
    segmentation_f : Class, default None
        Class containing segmentation workflow.
    extraction_f : Class, default None
        Class containing extraction workflow.
    classification_f : Class, default None
        Class containing classification workflow.
    selection_f : Class, default None
        Class containing selection workflow.

    Attributes
    ----------
    DEFAULT_CONFIG_NAME : str, default "config.yml"
        Default config name which is used for the config file in the project directory. This name needs to be used when no config is supplied and the config is manually created in the project folder.
    DEFAULT_INPUT_IMAGE_NAME: str, default "input_segmentation.h5"
        Default file name for loading the input image.
    DEFAULT_SEGMENTATION_DIR_NAME : str, default "segmentation"
        Default foldername for the segmentation process.
    DEFAULT_EXTRACTION_DIR_NAME : str, default "extraction"
        Default foldername for the extraction process.
    DEFAULT_CLASSIFICATION_DIR_NAME : str, default "selection"
        Default foldername for the classification process.
    DEFAULT_SELECTION_DIR_NAME : str, default "classification"
        Default foldername for the selection process.
    """

    DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_input_from_array(self, img, label, overwrite=False):
        # check if already exists if so throw error message
        if not os.path.isdir(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
        ):
            os.makedirs(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
            )

        path = os.path.join(
            self.directory,
            self.DEFAULT_SEGMENTATION_DIR_NAME,
            self.DEFAULT_INPUT_IMAGE_NAME,
        )

        if not overwrite:
            if os.path.isfile(path):
                sys.exit("File already exists")
            else:
                overwrite = True

        if overwrite:
            # column labels
            column_labels = ["label"]

            # create .h5 dataset to which all results are written
            path = os.path.join(
                self.directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                self.DEFAULT_INPUT_IMAGE_NAME,
            )
            hf = h5py.File(path, "w")
            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset("label_names", data=column_labels, chunks=None, dtype=dt)
            hf.create_dataset("labels", data=label, chunks=None, dtype=dt)
            hf.create_dataset(
                "input_images", data=img, chunks=(1, 1, img.shape[2], img.shape[2])
            )

            print(hf.keys())
            hf.close()

    def load_input_from_files(
            self,
            input_dir,
            channels,
            timepoints,
            plate_layout,
            img_size=1080,
            overwrite=False,
    ):
        """
        Function to load timecourse experiments recorded with opera phenix into .h5 dataformat for further processing.
        """

        # check if already exists if so throw error message
        if not os.path.isdir(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
        ):
            os.makedirs(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
            )

        path = os.path.join(
            self.directory,
            self.DEFAULT_SEGMENTATION_DIR_NAME,
            self.DEFAULT_INPUT_IMAGE_NAME,
        )

        if not overwrite:
            if os.path.isfile(path):
                sys.exit("File already exists")
            else:
                overwrite = True

        if overwrite:
            self.img_size = img_size

            def _read_write_images(dir, indexes, h5py_path):
                # unpack indexes
                index_start, index_end = indexes

                # get information on directory
                well = re.search(
                    "Row.._Well[0-9][0-9]", dir
                ).group()  # need to use re.search and not match sinde the identifier is not always at the beginning of the name
                region = re.search("r..._c...$", dir).group()

                # list all images within directory
                path = os.path.join(input_dir, dir)
                files = os.listdir(path)

                # filter to only contain the timepoints of interest
                files = np.sort([x for x in files if x.startswith(tuple(timepoints))])

                # checkt to make sure all timepoints are actually there
                _timepoints = np.unique(
                    [re.search("Timepoint[0-9][0-9][0-9]", x).group() for x in files]
                )

                sum = 0
                for timepoint in timepoints:
                    if timepoint in _timepoints:
                        sum += 1
                        continue
                    else:
                        print(f"No images found for Timepoint {timepoint}")
                # print(f"{sum} different timepoints found of the total {len(timepoints)} timepoints given.")
                self.log(
                    f"{sum} different timepoints found of the total {len(timepoints)} timepoints given."
                )

                # read images for that region
                imgs = np.empty(
                    (n_timepoints, n_channels, img_size, img_size), dtype="uint16"
                )
                for ix, channel in enumerate(channels):
                    images = [x for x in files if channel in x]

                    for i, im in enumerate(images):
                        image = imread(os.path.join(path, im), 0)
                        imgs[i, ix, :, :] = image

                # create labelling
                column_values = []
                for column in plate_layout.columns:
                    column_values.append(plate_layout.loc[well, column])

                list_input = [
                    list(range(index_start, index_end)),
                    [dir + "_" + x for x in timepoints],
                    [dir] * n_timepoints,
                    timepoints,
                    [well] * n_timepoints,
                    [region] * n_timepoints,
                ]
                list_input = [np.array(x) for x in list_input]

                for x in column_values:
                    list_input.append(np.array([x] * n_timepoints))

                labelling = np.array(list_input).T

                input_images[index_start:index_end, :, :, :] = imgs
                labels[index_start:index_end] = labelling

            # read plate layout
            plate_layout = pd.read_csv(plate_layout, sep="\s+|;|,", engine="python")
            plate_layout = plate_layout.set_index("Well")

            column_labels = [
                                "index",
                                "ID",
                                "location",
                                "timepoint",
                                "well",
                                "region",
                            ] + plate_layout.columns.tolist()

            # get information on number of timepoints and number of channels
            n_timepoints = len(timepoints)
            n_channels = len(channels)
            wells = np.unique(plate_layout.index.tolist())

            # get all directories contained within the input dir
            directories = os.listdir(input_dir)
            if ".DS_Store" in directories:
                directories.remove(
                    ".DS_Store"
                )  # need to remove this because otherwise it gives errors
            if ".ipynb_checkpoints" in directories:
                directories.remove(".ipynb_checkpoints")

            # filter directories to only contain those listed in the plate layout
            directories = [
                _dir
                for _dir in directories
                if re.search("Row.._Well[0-9][0-9]", _dir).group() in wells
            ]

            # check to ensure that imaging data is found for all wells listed in plate_layout
            _wells = [
                re.search("Row.._Well[0-9][0-9]", _dir).group() for _dir in directories
            ]
            not_found = [well for well in _wells if well not in wells]
            if len(not_found) > 0:
                print(
                    "following wells listed in plate_layout not found in imaging data:",
                    not_found,
                )
                self.log(
                    f"following wells listed in plate_layout not found in imaging data: {not_found}"
                )

            # check to make sure that timepoints given and timepoints found in data acutally match!
            _timepoints = []

            # create .h5 dataset to which all results are written
            path = os.path.join(
                self.directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                self.DEFAULT_INPUT_IMAGE_NAME,
            )

            # for some reason this directory does not always exist so check to make sure it does otherwise the whole reading of stuff fails
            if not os.path.isdir(
                    os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
            ):
                os.makedirs(
                    os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
                )

            with h5py.File(path, "w") as hf:
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset(
                    "label_names", (len(column_labels)), chunks=None, dtype=dt
                )
                hf.create_dataset(
                    "labels",
                    (len(directories) * n_timepoints, len(column_labels)),
                    chunks=None,
                    dtype=dt,
                )
                hf.create_dataset(
                    "input_images",
                    (len(directories) * n_timepoints, n_channels, img_size, img_size),
                    chunks=(1, 1, img_size, img_size),
                )

                label_names = hf.get("label_names")
                labels = hf.get("labels")
                input_images = hf.get("input_images")

                label_names[:] = column_labels

                # ------------------
                # start reading data
                # ------------------

                indexes = []
                # create indexes
                start_index = 0
                for i, _ in enumerate(directories):
                    stop_index = start_index + n_timepoints
                    indexes.append((start_index, stop_index))
                    start_index = stop_index

                # iterate through all directories and add to .h5
                # this is not implemented with multithreaded processing because writing multi-threaded to hdf5 is hard
                # multithreaded reading is easier

                for dir, index in tqdm(
                        zip(directories, indexes), total=len(directories)
                ):
                    _read_write_images(dir, index, h5py_path=path)

    def load_input_from_stitched_files(
            self,
            input_dir,
            channels,
            timepoints,
            plate_layout,
            overwrite=False,
    ):
        """
        Function to load timecourse experiments recorded with opera phenix into .h5 dataformat for further processing.
        Assumes that stitched images for all files have already been assembled.
        """

        # check if already exists if so throw error message
        if not os.path.isdir(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
        ):
            os.makedirs(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
            )

        path = os.path.join(
            self.directory,
            self.DEFAULT_SEGMENTATION_DIR_NAME,
            self.DEFAULT_INPUT_IMAGE_NAME,
        )

        import imagesize

        

        if not overwrite:
            if os.path.isfile(path):
                sys.exit("File already exists")
            else:
                overwrite = True

        if overwrite:

            def _read_write_images(well, indexes, h5py_path):
                # unpack indexes
                index_start, index_end = indexes

                # list all images for that well
                _files = [_file for _file in files if well in _file]

                # filter to only contain the timepoints of interest
                _files = np.sort([x for x in _files if x.startswith(tuple(timepoints))])

                # checkt to make sure all timepoints are actually there
                _timepoints = np.unique(
                    [re.search("Timepoint[0-9][0-9][0-9]", x).group() for x in _files]
                )

                sum = 0
                for timepoint in timepoints:
                    if timepoint in _timepoints:
                        sum += 1
                        continue
                    else:
                        print(f"No images found for Timepoint {timepoint}")
                # print(f"{sum} different timepoints found of the total {len(timepoints)} timepoints given.")
                self.log(
                    f"{sum} different timepoints found of the total {len(timepoints)} timepoints given."
                )

                # read images for that region
                imgs = np.empty(
                    (n_timepoints, n_channels, size1, size2), dtype="uint16"
                )
                for ix, channel in enumerate(channels):
                    images = [x for x in _files if channel in x]

                    for i, im in enumerate(images):
                        image = imread(os.path.join(input_dir, im), 0)

                        #perform cropping so that all stitched images have the same size
                        x, y = image.shape
                        diff1 = x - size1
                        diff1x = int(np.floor(diff1 / 2))
                        diff1y = int(np.ceil(diff1 / 2))
                        diff2 = y - size2
                        diff2x = int(np.floor(diff2 / 2))
                        diff2y = int(np.ceil(diff2 / 2))
                        cropped = image[slice(diff1x, x - diff1y), slice(diff2x, y - diff2y)]
                        print(image.shape, cropped.shape)

                        imgs[i, ix, :, :] = cropped
                
                # create labelling
                column_values = []
                for column in plate_layout.columns:
                    column_values.append(plate_layout.loc[well, column])

                list_input = [
                    list(range(index_start, index_end)),
                    [well+ "_" + x for x in timepoints],
                    [well] * n_timepoints,
                    timepoints,
                    [well] * n_timepoints,
                ]
                list_input = [np.array(x) for x in list_input]

                for x in column_values:
                    list_input.append(np.array([x] * n_timepoints))

                labelling = np.array(list_input).T

                input_images[index_start:index_end, :, :, :] = imgs
                labels[index_start:index_end] = labelling

            # read plate layout
            plate_layout = pd.read_csv(plate_layout, sep="\s+|;|,", engine="python")
            plate_layout = plate_layout.set_index("Well")

            column_labels = [
                                "index",
                                "ID",
                                "location",
                                "timepoint",
                                "well",
                            ] + plate_layout.columns.tolist()

            # get information on number of timepoints and number of channels
            n_timepoints = len(timepoints)
            n_channels = len(channels)
            wells = np.unique(plate_layout.index.tolist())

            # get all files contained within the input dir
            files = os.listdir(input_dir)
            files = [file for file in files if file.endswith(".tif")]

            # filter directories to only contain those listed in the plate layout
            files = [
                _dir
                for _dir in files
                if re.search("Row.._Well[0-9][0-9]", _dir).group() in wells
            ]

            # check to ensure that imaging data is found for all wells listed in plate_layout
            _wells = [
                re.search("Row.._Well[0-9][0-9]", _dir).group() for _dir in files
            ]
            not_found = [well for well in _wells if well not in wells]
            if len(not_found) > 0:
                print(
                    "following wells listed in plate_layout not found in imaging data:",
                    not_found,
                )
                self.log(
                    f"following wells listed in plate_layout not found in imaging data: {not_found}"
                )

            #get image size and subtract 10 pixels from each edge 
            # will adjust all merged images to this dimension to ensure that they all have the same dimensions and can be loaded into the same hdf5 file
            size1, size2 = imagesize.get(os.path.join(input_dir, files[0]))
            size1 = size1 - 2 * 10
            size2 = size2 - 2 * 10
            self.img_size = (size1, size2)

            # create .h5 dataset to which all results are written
            path = os.path.join(
                self.directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                self.DEFAULT_INPUT_IMAGE_NAME,
            )

            # for some reason this directory does not always exist so check to make sure it does otherwise the whole reading of stuff fails
            if not os.path.isdir(
                    os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
            ):
                os.makedirs(
                    os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
                )

            with h5py.File(path, "w") as hf:
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset(
                    "label_names", (len(column_labels)), chunks=None, dtype=dt
                )
                hf.create_dataset(
                    "labels",
                    (len(wells) * n_timepoints, len(column_labels)),
                    chunks=None,
                    dtype=dt,
                )
                hf.create_dataset(
                    "input_images",
                    (len(wells) * n_timepoints, n_channels, size1, size2),
                    chunks=(1, 1, size1, size2),
                )

                label_names = hf.get("label_names")
                labels = hf.get("labels")
                input_images = hf.get("input_images")

                label_names[:] = column_labels

                # ------------------
                # start reading data
                # ------------------

                indexes = []
                # create indexes
                start_index = 0
                for i, _ in enumerate(wells):
                    stop_index = start_index + n_timepoints
                    indexes.append((start_index, stop_index))
                    start_index = stop_index

                # iterate through all directories and add to .h5
                # this is not implemented with multithreaded processing because writing multi-threaded to hdf5 is hard
                # multithreaded reading is easier

                for well, index in tqdm(
                        zip(wells, indexes), total=len(wells)
                ):
                    _read_write_images(well, index, h5py_path=path)

  
    def load_input_from_files_and_merge(
            self,
            input_dir,
            channels,
            timepoints,
            plate_layout,
            img_size=1080,
            stitching_channel="Alexa488",
            overlap=0.1,
            max_shift=10,
            overwrite=False,
            nucleus_channel="DAPI",
            cytosol_channel="Alexa488",
    ):
        """
        Function to load timecourse experiments recorded with opera phenix into .h5 dataformat for further processing after merging all the regions from each teampoint in each well.
        """
        from sparcstools.stitch import generate_stitched

        # check if already exists if so throw error message
        if not os.path.isdir(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
        ):
            os.makedirs(
                os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
            )

        path = os.path.join(
            self.directory,
            self.DEFAULT_SEGMENTATION_DIR_NAME,
            self.DEFAULT_INPUT_IMAGE_NAME,
        )

        if not overwrite:
            if os.path.isfile(path):
                sys.exit("File already exists.")
            else:
                overwrite = True

        if overwrite:
            self.img_size = img_size

            self.log(f"Reading all images included in directory {input_dir}.")

            images = os.listdir(input_dir)
            images = [x for x in images if x.endswith((".tiff", ".tif"))]

            _timepoints = np.sort(list(set([x.split("_")[0] for x in images])))
            _wells = np.sort(
                list(
                    set(
                        [
                            re.match(".*_Row[0-9][0-9]_Well[0-9][0-9]", x).group()[13:]
                            for x in images
                        ]
                    )
                )
            )

            # apply filtering to only get those that are in the plate layout file
            plate_layout = pd.read_csv(plate_layout, sep="\s+|;|,", engine="python")
            plate_layout = plate_layout.set_index("Well")

            column_labels = [
                                "index",
                                "ID",
                                "location",
                                "timepoint",
                                "well",
                                "region",
                            ] + plate_layout.columns.tolist()

            # get information on number of timepoints and number of channels
            n_timepoints = len(timepoints)
            n_channels = len(channels)
            wells = np.unique(plate_layout.index.tolist())

            _wells = [x for x in _wells if x in wells]
            _timepoints = [x for x in _timepoints if x in timepoints]

            not_found_wells = [well for well in _wells if well not in wells]
            not_found_timepoints = [
                timepoint for timepoint in _timepoints if timepoint not in timepoints
            ]

            if len(not_found_wells) > 0:
                print(
                    "following wells listed in plate_layout not found in imaging data:",
                    not_found_wells,
                )
                self.log(
                    f"following wells listed in plate_layout not found in imaging data: {not_found_wells}"
                )

            if len(not_found_timepoints) > 0:
                print(
                    "following timepoints given not found in imaging data:",
                    not_found_timepoints,
                )
                self.log(
                    f"following timepoints given not found in imaging data: {not_found_timepoints}"
                )

            self.log(f"Will perform merging over the following specs:")
            self.log(f"Wells: {_wells}")
            self.log(f"Timepoints: {_timepoints}")

            # create .h5 dataset to which all results are written
            path = os.path.join(
                self.directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                self.DEFAULT_INPUT_IMAGE_NAME,
            )

            # for some reason this directory does not always exist so check to make sure it does otherwise the whole reading of stuff fails
            if not os.path.isdir(
                    os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
            ):
                os.makedirs(
                    os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)
                )

            with h5py.File(path, "w") as hf:
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset(
                    "label_names", (len(column_labels)), chunks=None, dtype=dt
                )
                hf.create_dataset(
                    "labels",
                    (len(_wells) * n_timepoints, len(column_labels)),
                    chunks=None,
                    dtype=dt,
                )
                label_names = hf.get("label_names")
                labels = hf.get("labels")

                label_names[:] = column_labels

                run_number = 0
                for timepoint in tqdm(_timepoints):
                    for well in tqdm(_wells):
                        RowID = well.split("_")[0]
                        WellID = well.split("_")[1]
                        zstack_value = 1

                        # define patter to recognize which slide should be stitched
                        # remember to adjust the zstack value if you aquired zstacks and want to stitch a speciifc one in the parameters above

                        pattern = (
                                f"{timepoint}_{RowID}_{WellID}"
                                + "_{channel}_"
                                + "zstack"
                                + str(zstack_value).zfill(3)
                                + "_r{row:03}_c{col:03}.tif"
                        )

                        merged_images, channels = generate_stitched(
                            input_dir,
                            well,
                            pattern,
                            outdir="/",
                            overlap=overlap,
                            max_shift=max_shift,
                            do_intensity_rescale=True,
                            stitching_channel=stitching_channel,
                            filetype="return_array",
                            export_XML=False,
                            plot_QC=False,
                        )

                        if run_number == 0:
                            img_size1 = merged_images.shape[1] - 2 * 10
                            img_size2 = merged_images.shape[2] - 2 * 10
                            # create this after the first image is stitched and we have the dimensions
                            hf.create_dataset(
                                "input_images",
                                (
                                    len(_wells) * n_timepoints,
                                    n_channels,
                                    img_size1,
                                    img_size2,
                                ),
                                chunks=(1, 1, img_size1, img_size2),
                            )
                            input_images = hf.get("input_images")

                        # crop so that all images have the same size
                        _, x, y = merged_images.shape
                        diff1 = x - img_size1
                        diff1x = int(np.floor(diff1 / 2))
                        diff1y = int(np.ceil(diff1 / 2))
                        diff2 = y - img_size2
                        diff2x = int(np.floor(diff2 / 2))
                        diff2y = int(np.ceil(diff2 / 2))
                        cropped = merged_images[
                                  :, slice(diff1x, x - diff1y), slice(diff2x, y - diff2y)
                                  ]
                        print(merged_images.shape, cropped.shape)

                        # create labelling
                        column_values = []
                        for column in plate_layout.columns:
                            column_values.append(plate_layout.loc[well, column])

                        list_input = [
                            str(run_number),
                            f"{well}_{timepoint}_all",
                            f"{well}_all",
                            timepoint,
                            well,
                            "stitched",
                        ]

                        for x in column_values:
                            list_input.append(x)

                        # reorder to fit to timecourse sorting
                        allocated_channels = []
                        allocated_indexes = []
                        if nucleus_channel in channels:
                            nucleus_index = channels.index(nucleus_channel)
                            allocated_channels.append(nucleus_channel)
                            allocated_indexes.append(nucleus_index)
                        else:
                            print("nucleus_channel not found in supplied channels!!!")

                        if cytosol_channel in channels:
                            cytosol_index = channels.index(cytosol_channel)
                            allocated_channels.append(cytosol_channel)
                            allocated_indexes.append(cytosol_index)
                        else:
                            print("cytosol_channel not found in supplied channels!!!")

                        all_other_indexes = [
                            channels.index(x)
                            for x in channels
                            if x not in allocated_channels
                        ]
                        all_other_indexes = list(np.sort(all_other_indexes))

                        index_list = allocated_indexes + all_other_indexes
                        cropped = np.array([cropped[x, :, :] for x in index_list])

                        self.log(
                            f"adjusted channels to the following order: {[channels[i] for i in index_list]}"
                        )
                        input_images[run_number, :, :, :] = cropped
                        labels[run_number] = list_input
                        run_number += 1
                        self.log(
                            f"finished stitching and saving well {well} for timepoint {timepoint}."
                        )

    def adjust_segmentation_indexes(self):
        self.segmentation_f.adjust_segmentation_indexes()

    def segment(self, overwrite=False, *args, **kwargs):
        """
        segment timecourse project with the defined segmentation method.
        """

        if overwrite == True:
            # delete segmentation and classes from .hdf5 to be able to create new again
            path = os.path.join(
                self.directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                self.DEFAULT_INPUT_IMAGE_NAME,
            )
            with h5py.File(path, "a") as hf:
                if "segmentation" in hf.keys():
                    del hf["segmentation"]
                if "classes" in hf.keys():
                    del hf["classes"]

            # delete generated files to make clean
            classes_path = os.path.join(
                self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv"
            )
            log_path = os.path.join(
                self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, "processing.log"
            )
            if os.path.isfile(classes_path):
                os.remove(classes_path)
            if os.path.isfile(log_path):
                os.remove(log_path)

            print("If Segmentation already existed removed.")

        if self.segmentation_f == None:
            raise ValueError("No segmentation method defined")

        else:
            self.segmentation_f(*args, **kwargs)

    def extract(self, *args, **kwargs):
        """
        extract single cells from a timecourse project with the defined extraction method.
        """

        if self.extraction_f == None:
            raise ValueError("No extraction method defined")

        input_segmentation = self.segmentation_f.get_output()
        input_dir = os.path.join(
            self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv"
        )
        self.extraction_f(input_segmentation, input_dir, *args, **kwargs)
