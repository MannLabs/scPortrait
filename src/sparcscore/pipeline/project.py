# -*- coding: utf-8 -*-
import warnings
import shutil
import os
import yaml
from PIL import Image
import PIL
import numpy as np
import numpy.ma as ma
import sys
import imagesize

# packages for timecourse project
import pandas as pd
from cv2 import imread
import re
import h5py
from tqdm.auto import tqdm
from time import time
import shutil

from sparcscore.pipeline.base import Logable
from sparcscore.processing.preprocessing import percentile_normalization, EDF, maximum_intensity_projection
from sparcscore.pipeline.utils import _read_napari_csv, _generate_mask_polygon

import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.reader import Reader

class Project(Logable):
    """
    Project base class used to create a SPARCSpy project.

    Args:
        location_path (str): Path to the folder where to project should be created. The folder is created in case the specified folder does not exist.
        config_path (str, optional, default ""): Path pointing to a valid configuration file. The file will be copied to the project directory and renamed to the name specified in ``DEFAULT_CLASSIFICATION_DIR_NAME``. If no config is specified, the existing config in the project directory will be used, if possible. See the section configuration to find out more about the config file.
        intermediate_output (bool, default False): When set to True intermediate outputs will be saved where applicable.
        debug (bool, default False): When set to True debug outputs will be printed where applicable.
        overwrite (bool, default False): When set to True, the processing step directory will be completely deleted and newly created when called.
        segmentation_f (Class, default None): Class containing segmentation workflow.
        extraction_f (Class, default None): Class containing extraction workflow.
        classification_f (Class, default None): Class containing classification workflow.
        selection_f (Class, default None): Class containing selection workflow.
    
    Attributes:
        DEFAULT_CONFIG_NAME (str, default "config.yml"): Default config name which is used for the config file in the project directory. This name needs to be used when no config is supplied and the config is manually created in the project folder.
        DEFAULT_SEGMENTATION_DIR_NAME (str, default "segmentation"): Default foldername for the segmentation process.
        DEFAULT_EXTRACTION_DIR_NAME (str, default "extraction"): Default foldername for the extraction process.
        DEFAULT_CLASSIFICATION_DIR_NAME (str, default "selection"): Default foldername for the classification process.
        DEFAULT_SELECTION_DIR_NAME (str, default "classification"): Default foldername for the selection process.
        DEFAULT_INPUT_IMAGE_NAME (str, default "input_image.ome.zarr"): Default filename for the input image.
        channel_colors (list(str)): List of colors used for the channel visualization in the segmentation output.
    """
    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_SEGMENTATION_DIR_NAME = "segmentation"
    DEFAULT_EXTRACTION_DIR_NAME = "extraction"
    DEFAULT_CLASSIFICATION_DIR_NAME = "classification"
    DEFAULT_SELECTION_DIR_NAME = "selection"
    DEFAULT_INPUT_IMAGE_NAME = "input_image.ome.zarr"
    channel_colors = [
        "#0000FF",
        "#00FF00",
        "#FF0000",
        "#FFE464", 
        "#9b19f5", 
        "#ffa300", 
        "#dc0ab4", 
        "#b3d4ff", 
        "#00bfa0"]

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
            warnings.warn("There is already a directory in the location path")

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
                project_location = self.project_location,
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
                project_location = self.project_location,
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
                project_location = self.project_location,
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
                project_location = self.project_location,
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
        Loads config from file and writes it to self.config

        Args:
            file_path (str): Path to the config.yml file that should be loaded.
        """
        self.log(f"Loading config from {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError("Your config path is invalid")

        with open(file_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def save_input_image(self, input_image):

        path = os.path.join(self.project_location, self.DEFAULT_INPUT_IMAGE_NAME)
        loc = parse_url(path, mode="w").store
        group = zarr.group(store = loc)

        n_channels = input_image.shape[0]
        channels = [f"Channel{i}" for i in range(n_channels)]
        
        group.attrs["omero"] = {
            "name": self.DEFAULT_INPUT_IMAGE_NAME,
            "channels": [{"label":channel, "color":self.channel_colors[i], "active":True} for i, channel in enumerate(channels)]
        }

        write_image(input_image, group = group, axes = "cyx", storage_options=dict(chunks=(1, 1024, 1024)))

        self.log(f"saved input_image: {path}")

    def load_input_image(self):
        path = os.path.join(self.project_location, self.DEFAULT_INPUT_IMAGE_NAME)
        # read the image data
        loc = parse_url(path, mode="r")
        zarr_reader = Reader(loc).zarr

        #read entire data into memory
        time_start = time()
        self.input_image = np.array(zarr_reader.load("0").compute())
        time_end = time()

        self.log(f"Read input image from file {path} to numpy array in {(time_end - time_start)/60} minutes.")

    def load_input_from_file(self, file_paths, crop=[(0, -1), (0, -1)]):
        """
        Load input image from a number of files.

        Args:
            file_paths (list(str)): List containing paths to each channel like [“path1/img.tiff”, “path2/img.tiff”, “path3/img.tiff”]. Expects a list of file paths with length “input_channel” as defined in the config.yml. Input data is NOT copied to the project folder by default. Different segmentation functions especially tiled segmentations might copy the input.
            crop (list(tuple), optional): When set, it can be used to crop the input image. The first element refers to the first dimension of the image and so on. For example use “[(0,1000),(0,2000)]” to crop the image to 1000 px height and 2000 px width from the top left corner.
        """
        if self.config is None:
            raise ValueError("Dataset has no config file loaded")
        
        #check if an input image was already loaded if so throw error if overwrite = False

        path = os.path.join(self.project_location, self.DEFAULT_INPUT_IMAGE_NAME)
        if os.path.isdir(path):
            if self.overwrite:
                shutil.rmtree(path)
                self.log("Overwrite is set to True. Existing input image was deleted.")
            else:
                raise ValueError(f"Overwrite is set to False but an input image already written to file. Either set ")

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
        
        self.save_input_image(self.input_image)

    def load_input_from_array(self, array, remap=None):
        """
        Load input image from an already loaded numpy array.

        Args:
            array (numpy.ndarray): Numpy array of shape “[channels, height, width]”.
            remap (list(int), optional): Define remapping of channels. For example use “[1, 0, 2]” to change the order of the first and the second channel. The expected order is Nucleus Channel, Cellmembrane Channel followed by other channels.
        """
        # input data is not copied to the project folder
        if self.config is None:
            raise ValueError("Dataset has no config file loaded")
        
        #check if an input image was already loaded if so throw error if overwrite = False

        path = os.path.join(self.project_location, self.DEFAULT_INPUT_IMAGE_NAME)
        if os.path.isdir(path):
            if self.overwrite:
                shutil.rmtree(path)
                self.log("Overwrite is set to True. Existing input image was deleted.")
            else:
                raise ValueError(f"Overwrite is set to False but an input image already written to file. Either set ")

        if not array.shape[0] == self.config["input_channels"]:
            raise ValueError(
                "Expected {} image paths, only received {}".format(
                    self.config["input_channels"], array.shape[0]
                )
            )

        self.input_image = np.array(array, dtype="float64")

        if remap is not None:
            self.input_image = self.input_image[remap]
        
        self.save_input_image(self.input_image)

    def load_input_from_czi(self, czi_path, intensity_rescale = True, scene = None, z_stack_projection = None, remap=None):
        """Load image input from .czi file

        Args:
            czi_path (path): path to .czi file that should be loaded
            intensity_rescale (bool, optional): boolean indicator if the read image should be intensity rescaled to the 0.5% and 99.5% quantile. Defaults to True.
            scene (int, optional): integer indicating which scene should be selected if the .czi contains several. Defaults to None.
            z_stack_projection (int or "maximum_intensity_projection" or "EDF"): if the .czi contains Z-stacks indicator which method should be used to integrate them. If an integer is passed the z-stack with that id is used. Can also pass a string indicating the implemented methods. Defaults to None.
            remap (list(int), optional): Define remapping of channels. For example use “[1, 0, 2]” to change the order of the first and the second channel. The expected order is Nucleus Channel, Cellmembrane Channel followed by other channels.
        """
        import aicspylibczi

        self.log(f"Reading CZI file from path {czi_path}")
        czi = aicspylibczi.CziFile(czi_path)

        # Get the shape of the data
        dimensions = czi.dims  # 'STCZMYX'
        shape = czi.get_dims_shape()  # (1, 1, 1, 1, 2, 624, 924)

        #check that we have a mosaic image else return an error as this method only support mosaic czi
        if not czi.is_mosaic():
            sys.exit("Only mosaic CZI files are supported. Please contact the developers.")

        n_scenes = len(shape)
        boxes = czi.get_all_mosaic_scene_bounding_boxes()

        if n_scenes > 1:
            if scene is None:
                sys.exit("For multi-scene CZI files you need to select one scene that you wish to load into SPARCSpy. Please pass an integer to the parameter scene indicating which scene to choose.")
            else:
                self.log(f"Reading scene {scene} from CZI file.")
        
        #if there is only one scene automatically select this one
        if n_scenes == 1:
            scene = 0

        
        box = boxes[scene]
        channels = shape[scene]["C"][1]

        #check if more than one zstack is contained
        if "Z" in dimensions:
            self.log("Found more than one Z-stack in CZI file.")
            
            if type(z_stack_projection) == int:
                self.log(f"Selection Z-stack {z_stack_projection}")
                _mosaic = np.array([czi.read_mosaic(region = (box.x, box.y, box.w, box.h),
                                                    C = c, 
                                                    Z = z_stack_projection).squeeze() for c in range(channels)])
            
            elif z_stack_projection is not None:
                
                #define method for aggregating z-stacks
                if z_stack_projection == "maximum_intensity_projection":
                    self.log("Using Maximum Intensity Projection to combine Z-stacks.")
                    method = maximum_intensity_projection
                elif z_stack_projection == "EDF":
                    self.log("Using EDF to combine Z-stacks.")
                    method = EDF
                else:
                    sys.exit("Please define a valid method for z_stack_projection.")
                
                #get number of zstacks
                zstacks = shape[scene]["Z"][1]
                
                #actually read data
                _mosaic = []
                for c in range(channels):
                    _img = []
                    for z in range(zstacks):
                        _img.append(czi.read_mosaic(region = (box.x, box.y, box.w, box.h), C = c, Z = z).squeeze())
                    _img = np.array(_img) #convert to numpy array
                    _mosaic.append(method(_img))
            
        else:
            _mosaic = np.array([czi.read_mosaic(region = (box.x, box.y, box.w, box.h),
                                                C = c).squeeze() for c in range(channels)])

        #perform intensity rescaling before loading images
        if intensity_rescale:
            self.log("Performing percentile normalization on the input image with lower_percentile=0.005 and upper_percentile=0.995")
            _mosaic = np.array([percentile_normalization(_mosaic[i], lower_percentile=0.005, upper_percentile=0.995) for i in range(channels)])
            _mosaic = (_mosaic * 65535).astype('uint16') #convert to 16bit images

        else:
            _mosaic = np.array(_mosaic).astype("uint16")
        self.log("finished loading. array.")
        self.load_input_from_array(np.fliplr(_mosaic), remap = remap)

    def load_input_from_czi_2(self, czi_path, intensity_rescale = True, scene = None, z_stack_projection = None, remap=None):
        """Load image input from .czi file. Slower than CZI 1 use that function instead.

        Args:
            czi_path (path): path to .czi file that should be loaded
            intensity_rescale (bool, optional): boolean indicator if the read image should be intensity rescaled to the 0.5% and 99.5% quantile. Defaults to True.
            scene (int, optional): integer indicating which scene should be selected if the .czi contains several. Defaults to None.
            z_stack_projection (int or "maximum_intensity_projection" or "EDF"): if the .czi contains Z-stacks indicator which method should be used to integrate them. If an integer is passed the z-stack with that id is used. Can also pass a string indicating the implemented methods. Defaults to None.
            remap (list(int), optional): Define remapping of channels. For example use “[1, 0, 2]” to change the order of the first and the second channel. The expected order is Nucleus Channel, Cellmembrane Channel followed by other channels.
        """
        from aicsimageio.aics_image import AICSImage

        self.log(f"Reading CZI file from path {czi_path}")
        czi = AICSImage(czi_path)

        n_scenes = len(czi.scenes)
        n_channels = czi.dims.C
        n_zstacks = czi.dims.Z

        if n_scenes > 1:
            if scene is None:
                sys.exit("For multi-scene CZI files you need to select one scene that you wish to load into SPARCSpy. Please pass an integer to the parameter scene indicating which scene to choose.")
            else:
                self.log(f"Reading scene {czi.scenes[scene]} from CZI file.")
                czi.set_scene(scene)
        
        #if there is only one scene automatically select this one
        if n_scenes == 1:
            scene = 0
            czi.set_scene(scene)

        #check if more than one zstack is contained
        if n_zstacks > 1:
            self.log(f"Found {n_zstacks} Z-stack in CZI file.")
            
            if type(z_stack_projection) == int:
                self.log(f"Selection Z-stack {z_stack_projection}")
                _mosaic = czi.get_image_dask_data("CYX", Z=z_stack_projection).compute()
            
            elif z_stack_projection is not None:
                
                #define method for aggregating z-stacks
                if z_stack_projection == "maximum_intensity_projection":
                    self.log("Using Maximum Intensity Projection to combine Z-stacks.")
                    method = maximum_intensity_projection
                elif z_stack_projection == "EDF":
                    self.log("Using EDF to combine Z-stacks.")
                    method = EDF
                else:
                    sys.exit("Please define a valid method for z_stack_projection.")
                
                #actually read data
                _mosaic = []
                for c in range(n_channels):
                    _img = czi.get_image_dask_data("ZYX", C = c).compute()
                    _mosaic.append(method(_img))
                _mosaic = np.array(_mosaic)
            
        else:
            _mosaic = czi.read_mosaic("CYX").compute()

        #perform intensity rescaling before loading images
        if intensity_rescale:
            self.log("Performing percentile normalization on the input image with lower_percentile=0.005 and upper_percentile=0.995")
            _mosaic = np.array([percentile_normalization(_mosaic[i], lower_percentile=0.005, upper_percentile=0.995) for i in range(n_channels)])
            _mosaic = (_mosaic * 65535).astype('uint16') #convert to 16bit images

        self.load_input_from_array(np.fliplr(_mosaic), remap = remap)


    def define_image_area_napari(self, napari_csv_path):
            if self.input_image is None:
                self.log("No input image loaded. Trying to read file from disk.")
            try:
                self.load_input_image()
            except:
                raise ValueError("No input image loaded and no file found to load image from.")
            
            #get name from naparipath
            name = os.path.basename(napari_csv_path).split(".")[0]

            # read napari csv
            polygons = _read_napari_csv(napari_csv_path)
    
            #determine size that mask needs to be generated for
            _, x, y = self.input_image.shape
            
            #generate mask indicating which areas of the image to use
            mask = _generate_mask_polygon(polygons, outshape = (x, y))
            mask = np.broadcast_to(mask, self.input_image.shape)
            
            masked = ma.masked_array(self.input_image, mask=~mask)

            #delete old input image
            path = os.path.join(self.project_location, self.DEFAULT_INPUT_IMAGE_NAME)
            shutil.rmtree(path)
            self.log("Removed old input image and writing new input image with masked areas set to 0 to file.")

            self.save_input_image(masked.filled(0))
            
    def segment(self, *args, **kwargs):
        """
        Segment project with the selected segmentation method.
        """

        if self.segmentation_f is None:
            raise ValueError("No segmentation method defined")
        elif self.input_image is None:
            self.log("No input image loaded. Trying to read file from disk.")
            try:
                self.load_input_image()
                self.segmentation_f(self.input_image, *args, **kwargs)
            except:
                raise ValueError("No input image loaded and no file found to load image from.")
        elif self.input_image is not None:
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

    def write_segmentation_to_omezarr(self):
        
        from sparcstools.segmentation_viz import write_zarr_with_seg

        input_dir = os.path.join(
            self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "segmentation.h5"
        )

        output_file = os.path.join(
            self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "segmentation.ome.zarr"
        )
       
        #read segmentation and images
        hf = h5py.File(input_dir, "r")

        labels = hf.get("labels")
        images = hf.get("channels")

        image = images[:]
        label = labels[:]
        
        hf.close()

        n_channels = self.config["input_channels"]

        write_zarr_with_seg(image, 
                            [np.expand_dims(seg, axis = 0) for seg in label],  #list of all sets you want to visualize
                            ["nucleus_segmentation", "cytosol_segmentation"], #list of what each cell set should be called
                            output_file, 
                            channels =["nucleus", "membrane"] + [f"Channel{i}" for i in range(n_channels-2)])

    def process(self):
        self.segment()
        self.extract()
        self.classify()


class TimecourseProject(Project):
    """
    Timecourse Project used to create a SPARCSpy project for datasets that have multiple timepoints
    over the same field of view (add additional dimension in comparision to base SPARCSpy project).

    Args:
        location_path (str): Path to the folder where to project should be created. The folder is created in case the specified folder does not exist.
        config_path (str, optional): Path pointing to a valid configuration file. The file will be copied to the project directory and renamed to the name specified in ``DEFAULT_CLASSIFICATION_DIR_NAME``. If no config is specified, the existing config in the project directory will be used, if possible. See the section configuration to find out more about the config file.
        intermediate_output (bool, optional): When set to True intermediate outputs will be saved where applicable. Defaults to False.
        debug (bool, optional): When set to True debug outputs will be printed where applicable. Defaults to False.
        overwrite (bool, optional): When set to True, the processing step directory will be completely deleted and newly created when called. Defaults to False.
        segmentation_f (Class, optional): Class containing segmentation workflow. Defaults to None.
        extraction_f (Class, optional): Class containing extraction workflow. Defaults to None.
        classification_f (Class, optional): Class containing classification workflow. Defaults to None.
        selection_f (Class, optional): Class containing selection workflow. Defaults to None.
    
    Attributes:
        DEFAULT_CONFIG_NAME (str): Default config name which is used for the config file in the project directory. This name needs to be used when no config is supplied and the config is manually created in the project folder.
        DEFAULT_INPUT_IMAGE_NAME (str): Default file name for loading the input image.
        DEFAULT_SEGMENTATION_DIR_NAME (str): Default foldername for the segmentation process.
        DEFAULT_EXTRACTION_DIR_NAME (str): Default foldername for the extraction process.
        DEFAULT_CLASSIFICATION_DIR_NAME (str): Default foldername for the classification process.
        DEFAULT_SELECTION_DIR_NAME (str): Default foldername for the selection process.
    """
    DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_input_from_array(self, img, label, overwrite=False):
        """img needs to be in the format: (tiles, C, X, Y). Channels need to be in the order: nucleus, cytosol, other channels """
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
            column_labels = label.columns.to_list()

            # create .h5 dataset to which all results are written
            path = os.path.join(
                self.directory,
                self.DEFAULT_SEGMENTATION_DIR_NAME,
                self.DEFAULT_INPUT_IMAGE_NAME,
            )
            hf = h5py.File(path, "w")
            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset("label_names", data=column_labels, chunks=None, dtype=dt)
            hf.create_dataset("labels", data=label.astype(str).values, chunks=None, dtype=dt)
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
            overwrite=False):
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
                        imgs[i, ix, :, :] = image.astype("uint16")

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
                    dtype = "uint16"
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

        Args:
            input_dir (str): path to directory containing the stitched images
            channels (list(str)): list of strings indicating which channels should be loaded
            timepoints (list(str)): list of strings indicating which timepoints should be loaded
            plate_layout (str): path to csv file containing the plate layout
            overwrite (bool, optional): boolean indicating if existing files should be overwritten. Defaults to False.
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
                        image = imread(os.path.join(input_dir, im), 0).astype("uint16")

                        # check if image is too small and if yes, pad the image with black pixels
                        if image.shape[0] < size1 or image.shape[1] < size2:
                            image = np.pad(
                                image,
                                ((0, np.max((size1 - image.shape[0], 0))),
                                 (0, np.max((size2 - image.shape[1], 0)))),
                                mode='constant',
                                constant_values=0
                            )
                            self.log(f"Image {im} with the index {i} is too small and was padded with black pixels. "
                                     f"Image shape after padding: {image.shape}.")

                        # perform cropping so that all stitched images have the same size
                        x, y = image.shape
                        diff1 = x - size1
                        diff1x = int(np.floor(diff1 / 2))
                        diff1y = int(np.ceil(diff1 / 2))
                        diff2 = y - size2
                        diff2x = int(np.floor(diff2 / 2))
                        diff2y = int(np.ceil(diff2 / 2))

                        cropped = image[
                            slice(diff1x, x - diff1y), slice(diff2x, y - diff2y)
                        ]

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
                    dtype = "uint16"
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

        Args:
            input_dir (str): path to directory containing the stitched images
            channels (list(str)): list of strings indicating which channels should be loaded
            timepoints (list(str)): list of strings indicating which timepoints should be loaded
            plate_layout (str): path to csv file containing the plate layout
            img_size (int, optional): size of the images that should be generated. Defaults to 1080.
            stitching_channel (str, optional): channel that should be used for stitching. Defaults to "Alexa488".
            overlap (float, optional): overlap between stitched images. Defaults to 0.1.
            max_shift (int, optional): maximum shift between stitched images. Defaults to 10.
            overwrite (bool, optional): boolean indicating if existing files should be overwritten. Defaults to False.
            nucleus_channel (str, optional): channel that should be used for nucleus segmentation. Defaults to "DAPI".
            cytosol_channel (str, optional): channel that should be used for cytosol segmentation. Defaults to "Alexa488".
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

        if overwrite:
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
        Extract single cells from a timecourse project with the defined extraction method.
        """

        if self.extraction_f == None:
            raise ValueError("No extraction method defined")

        input_segmentation = self.segmentation_f.get_output()
        input_dir = os.path.join(
            self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv"
        )
        self.extraction_f(input_segmentation, input_dir, *args, **kwargs)
