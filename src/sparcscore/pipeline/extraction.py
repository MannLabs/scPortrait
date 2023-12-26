from datetime import datetime
from operator import index
import os
import numpy as np
import pandas as pd
import csv
from functools import partial
from multiprocessing import Pool
import h5py
import sys
from tqdm import tqdm
from itertools import compress

from skimage.filters import gaussian
from skimage.morphology import disk, dilation

from scipy.ndimage import binary_fill_holes

from sparcscore.processing.segmentation import numba_mask_centroid, _return_edge_labels
from sparcscore.processing.utils import plot_image, flatten
from sparcscore.processing.preprocessing import percentile_normalization, MinMax
from sparcscore.pipeline.base import ProcessingStep

import uuid
import shutil
import timeit

import matplotlib.pyplot as plt

import _pickle as cPickle

class HDF5CellExtraction(ProcessingStep):
    """
    A class to extracts single cell images from a segmented SPARCSpy project and save the 
    results to an HDF5 file.
    """
    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_SEGMENTATION_DIR = "segmentation"
    DEFAULT_SEGMENTATION_FILE = "segmentation.h5"
    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False

    #new parameters to make workflow adaptable to other types of projects
    channel_label = "channels"
    segmentation_label = "labels"
    
    def __init__(self,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if not os.path.isdir(self.directory):
                os.makedirs(self.directory)

        base_directory = self.directory.replace("/extraction", "")

        self.input_segmentation_path = os.path.join(base_directory, self.DEFAULT_SEGMENTATION_DIR, self.DEFAULT_SEGMENTATION_FILE)
        self.filtered_classes_path = os.path.join(base_directory, self.DEFAULT_SEGMENTATION_DIR, "classes.csv")
        self.output_path = os.path.join(self.directory, self.DEFAULT_DATA_DIR, self.DEFAULT_DATA_FILE)

        #extract required information for generating datasets
        self.get_compression_type()
        self.get_classes_path()
        self.get_normalization()

        self.save_index_to_remove = []
        
                  
    def get_compression_type(self):
        self.compression_type = "lzf" if self.config["compression"] else None
        return(self.compression_type)

    def get_classes_path(self):
        self.classes_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR, "classes.csv")
        return self.classes_path
    
    def get_normalization(self):
        global norm_function, MinMax_function

        if "normalization_range" in self.config:
            self.normalization = self.config["normalization_range"]
        else:
            self.normalization = True
        
        if self.normalization == True:
            def norm_function(img):
                return(percentile_normalization(img))
            
            def MinMax_function(img):
                return(MinMax(img))
        
        elif isinstance(self.normalization, tuple):
            lower, upper = self.normalization
            
            def norm_function(img, lower = lower, upper = upper):
                return(percentile_normalization(img, lower, upper))
            
            def MinMax_function(img):
                return(MinMax(img))
        
        elif self.normalization is None:
            def norm_function(img):
                return(img)
            
            def MinMax_function(img):
                img = img/65535 #convert 16bit unsigned integer image to float between 0 and 1 without adjusting for the pixel values we have in the extracted single cell image
                return(img)
        
        elif self.normalization == "None": #add additional check if if None is included as a string
            def norm_function(img):
                return(img)
            
            def MinMax_function(img):
                img = img/65535 #convert 16bit unsigned integer image to float between 0 and 1 without adjusting for the pixel values we have in the extracted single cell image
                return(img)
            
        else:
            self.log("Incorrect type of normalization_range defined.")
            sys.exit("Incorrect type of normalization_range defined.")

    def get_channel_info(self):
        with h5py.File(self.input_segmentation_path, 'r') as hf:

            hdf_channels = hf.get(self.channel_label)
            hdf_labels = hf.get(self.segmentation_label)

            if len(hdf_channels.shape) == 3:
                self.n_channels_input = hdf_channels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_channels_input = hdf_channels.shape[1]

            self.log(f"Using channel label {hdf_channels}")
            self.log(f"Using segmentation label {hdf_labels}")

            if len(hdf_labels.shape) == 3:
                self.n_segmentation_channels = hdf_labels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_segmentation_channels = hdf_labels.shape[1]

            self.n_channels_output = self.n_segmentation_channels + self.n_channels_input

    def get_output_path(self):
        self.extraction_data_directory = os.path.join(self.directory, self.DEFAULT_DATA_DIR)
        return self.extraction_data_directory    

    def setup_output(self):
        self.extraction_data_directory = os.path.join(self.directory, self.DEFAULT_DATA_DIR)
        if not os.path.isdir(self.extraction_data_directory):
            os.makedirs(self.extraction_data_directory)
            self.log("Created new data directory " + self.extraction_data_directory)   
    
    def parse_remapping(self):
        self.remap = None
        if "channel_remap" in self.config:
            char_list = self.config["channel_remap"].split(",")
            self.log("channel remap parameter found:")
            self.log(char_list)
            
            self.remap = [int(el.strip()) for el in char_list]

    def get_classes(self, filtered_classes_path):
        self.log(f"Loading filtered classes from {filtered_classes_path}")
        cr = csv.reader(open(filtered_classes_path,'r'),    )
        filtered_classes = [int(float(el[0])) for el in list(cr)]

        self.log("Loaded {} filtered classes".format(len(filtered_classes)))
        filtered_classes = np.unique(filtered_classes) #make sure they are all unique
        filtered_classes.astype(np.uint64)
        self.log("After removing duplicates {} filtered classes remain.".format(len(filtered_classes)))

        class_list = list(filtered_classes)
        if 0 in class_list: class_list.remove(0)
        self.num_classes = len(class_list)

        return(class_list)
    
    def generate_save_index_lookup(self, class_list):
        lookup = pd.DataFrame(index = class_list)
        return(lookup)
    
    def verbalise_extraction_info(self):
        #print some output information
        self.log(f"Extraction Details:")
        self.log(f"--------------------------------")
        self.log(f"Input channels: {self.n_channels_input}")
        self.log(f"Input labels: {self.n_segmentation_channels}")
        self.log(f"Output channels: {self.n_channels_output}")
        self.log(f"Number of classes to extract: {self.num_classes}")
        self.log(f"Extracted Image Dimensions: {self.config['image_size']} x {self.config['image_size']}")
    
    def _get_arg(self, cell_ids, lookup_saveindex):
        lookup_saveindex = self.generate_save_index_lookup(cell_ids)  
        args = list(zip(range(len(cell_ids)), [lookup_saveindex.index.get_loc(x) for x in cell_ids], cell_ids))
        return(args)

    def _initialize_tempmmap_array(self):
        #define as global variables so that this is also avaialable in other functions
        global _tmp_single_cell_data, _tmp_single_cell_index

        self.single_cell_index_shape = (self.num_classes,2)
        self.single_cell_data_shape = (self.num_classes,
                                        self.n_channels_output,
                                        self.config["image_size"],
                                        self.config["image_size"]) 
        
        #import tempmmap module and reset temp folder location
        from alphabase.io import tempmmap
        TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])

        #generate container for single_cell_data
        _tmp_single_cell_data = tempmmap.array(shape = self.single_cell_data_shape, dtype = np.float16)
        _tmp_single_cell_index  = tempmmap.array(shape = self.single_cell_index_shape, dtype = np.int64)

        self.TEMP_DIR_NAME = TEMP_DIR_NAME

    def _transfer_tempmmap_to_hdf5(self):
        global _tmp_single_cell_data, _tmp_single_cell_index   
        
        self.log(f"number of cells too close to image edges to extract: {len(self.save_index_to_remove)}")
        self.log(f"{_tmp_single_cell_data.shape} shape of single-cell data before removing cells to close to image edges")
        _tmp_single_cell_data = np.delete(_tmp_single_cell_data, self.save_index_to_remove, axis=0)
        _tmp_single_cell_index = np.delete(_tmp_single_cell_index, self.save_index_to_remove, axis=0)
        self.log(f"{_tmp_single_cell_data.shape} shape of single-cell data after removing cells to close to image edges")

        _, cell_ids = _tmp_single_cell_index[:].T
        _tmp_single_cell_index[:] = list(zip(list(range(len(cell_ids))), cell_ids))

        self.log(f"Transferring extracted single cells to .hdf5")

        if self.debug:
            #visualize some cells for debugging purposes
            x, y = _tmp_single_cell_index.shape
            n_cells = 100
            n_cells_to_visualize = x//n_cells

            random_indexes = np.random.choice(x, n_cells_to_visualize, replace=False)

            for index in random_indexes:
                stack = _tmp_single_cell_data[index]

                fig, axs = plt.subplots(1, stack.shape[0])

                for i, img in enumerate(stack):
                    axs[i].imshow(img)
                    axs[i].axis("off")
                
                fig.tight_layout()
                fig.show()

        with h5py.File(self.output_path, 'w') as hf:
            hf.create_dataset('single_cell_index', data = _tmp_single_cell_index[:], dtype=np.int64) #increase to 64 bit otherwise information may become truncated
            self.log("index created.")
            hf.create_dataset('single_cell_data', data = _tmp_single_cell_data[:], 
                                                    chunks= (1,
                                                    1,
                                                    self.config["image_size"],
                                                    self.config["image_size"]),
                                            compression=self.compression_type,
                                            dtype=np.float16)  
        
        #delete tempobjects (to cleanup directory)
        self.log(f"Tempmmap Folder location {self.TEMP_DIR_NAME} will now be removed.")
        shutil.rmtree(self.TEMP_DIR_NAME, ignore_errors=True)

        del self.TEMP_DIR_NAME, _tmp_single_cell_data, _tmp_single_cell_index 

    def _get_label_info(self, arg):
        index, save_index, cell_id = arg
        
        # no additional labelling required
        return(index, save_index, cell_id, None, None)   

    def _save_cell_info(self, save_index, cell_id, image_index, label_info, stack):
        #save index is irrelevant for this
        #label info is None so just ignore for the base case
        #image_index is none so just ignore for the base case
        global _tmp_single_cell_data, _tmp_single_cell_index

        #save single cell images
        _tmp_single_cell_data[save_index] = stack
        _tmp_single_cell_index[save_index] = [save_index, cell_id]
            
    def _extract_classes(self, input_segmentation_path, px_center, arg):
        """
        Processing for each individual cell that needs to be run for each center.
        """
        global norm_function, MinMax_function

        index, save_index, cell_id, image_index, label_info = self._get_label_info(arg) #label_info not used in base case but relevant for flexibility for other classes

        #generate some progress output every 10000 cells
        #relevant for benchmarking of time
        if save_index % 10000 == 0:
            self.log("Extracting dataset {}".format(save_index))

        with h5py.File(input_segmentation_path, 'r', 
                       rdcc_nbytes=self.config["hdf5_rdcc_nbytes"], 
                       rdcc_w0=self.config["hdf5_rdcc_w0"],
                       rdcc_nslots=self.config["hdf5_rdcc_nslots"]) as input_hdf:
        
            hdf_channels = input_hdf.get(self.channel_label)
            hdf_labels = input_hdf.get(self.segmentation_label)
            
            width = self.config["image_size"]//2

            image_width = hdf_channels.shape[-2] #adaptive to ensure that even with multiple stacks of input images this works correctly
            image_height = hdf_channels.shape[-1]
            n_channels = hdf_channels.shape[-3]
            
            _px_center = px_center[index]
            window_y = slice(_px_center[0]-width, _px_center[0]+width)
            window_x = slice(_px_center[1]-width, _px_center[1]+width)
            
            condition = [width < _px_center[0], _px_center[0] < image_width-width, width < _px_center[1], _px_center[1] < image_height-width]
            if np.all(condition):
                # mask 0: nucleus mask
                if image_index is None:
                    nuclei_mask = hdf_labels[0, window_y, window_x]
                else:
                    nuclei_mask = hdf_labels[image_index, 0, window_y, window_x]

                nuclei_mask = np.where(nuclei_mask == cell_id, 1, 0)

                nuclei_mask_extended = gaussian(nuclei_mask, preserve_range=True, sigma=5)
                nuclei_mask = gaussian(nuclei_mask, preserve_range=True, sigma=1)
            
                # channel 0: nucleus
                if image_index is None:
                    channel_nucleus = hdf_channels[0, window_y, window_x]
                else:
                    channel_nucleus = hdf_channels[image_index, 0, window_y, window_x]
                
                channel_nucleus = norm_function(channel_nucleus)
                channel_nucleus = channel_nucleus * nuclei_mask_extended
                channel_nucleus = MinMax_function(channel_nucleus)

                if n_channels >= 2:
                    
                    # mask 1: cell mask
                    if image_index is None:
                        cell_mask = hdf_labels[1,window_y,window_x]
                    else:
                        cell_mask = hdf_labels[image_index, 1,window_y,window_x]

                    cell_mask = np.where(cell_mask == cell_id, 1, 0).astype(int)
                    cell_mask = binary_fill_holes(cell_mask)

                    cell_mask_extended = dilation(cell_mask, footprint=disk(6))

                    cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
                    cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)

                    # channel 3: cellmask
                    
                    if image_index is None:
                        channel_cytosol = hdf_channels[1, window_y, window_x]
                    else:
                        channel_cytosol = hdf_channels[image_index, 1,window_y,window_x]

                    channel_cytosol = norm_function(channel_cytosol)
                    channel_cytosol = channel_cytosol*cell_mask_extended
                    channel_cytosol = MinMax_function(channel_cytosol)
                
                if n_channels == 1:
                    required_maps = [nuclei_mask, channel_nucleus]
                else:
                    required_maps = [nuclei_mask, cell_mask, channel_nucleus, channel_cytosol]
                
                #extract variable feature channels
                feature_channels = []

                if image_index is None:
                    if hdf_channels.shape[0] > 2:  
                        for i in range(2, hdf_channels.shape[0]):
                            feature_channel = hdf_channels[i, window_y, window_x]
                            feature_channel = norm_function(feature_channel)
                            feature_channel = feature_channel*cell_mask_extended
                            feature_channel = MinMax_function(feature_channel)
                            
                            feature_channels.append(feature_channel)
        
                else:
                    if hdf_channels.shape[1] > 2:
                        for i in range(2, hdf_channels.shape[1]):
                            feature_channel = hdf_channels[image_index, i, window_y, window_x]
                            feature_channel = norm_function(feature_channel)
                            feature_channel = feature_channel*cell_mask_extended
                            feature_channel = MinMax_function(feature_channel)
                            
                            feature_channels.append(feature_channel)

                channels = required_maps + feature_channels
                stack = np.stack(channels, axis=0).astype("float16")

                if self.debug:

                    #visualize some cells for debugging purposes
                    if index % 100 == 0:
                        
                        print(f"Cell ID: {cell_id} has center at [{_px_center[0]}, {_px_center[1]}]")

                        plt.figure()
                        plt.imshow(nuclei_mask)
                        plt.title("Nucleus Mask")
                        plt.axis("off")
                        plt.show()

                        if n_channels > 2:
                            plt.figure()
                            plt.imshow(cell_mask)
                            plt.title("Cytosol Mask")
                            plt.axis("off")
                            plt.show()

                            plt.figure()
                            plt.imshow(channel_cytosol)
                            plt.title("Cytosol Channel")
                            plt.axis("off")
                            plt.show()

                        plt.figure()
                        plt.imshow(channel_nucleus)
                        plt.title("Nucleus Channel")
                        plt.axis("off")
                        plt.show()
                    
                        for i, img in enumerate(feature_channels):
                            plt.figure()
                            plt.imshow(img)
                            plt.title(f"Feature Channel {i}" )
                            plt.axis("off")
                            plt.show()
                        
                        fig, axs = plt.subplots(1, stack.shape[0])

                        for i, img in enumerate(stack):
                            axs[i].imshow(img)
                            axs[i].axis("off")
                        
                        fig.tight_layout()
                        fig.show()
                
                if self.remap is not None:
                    stack = stack[self.remap]

                self._save_cell_info(save_index, cell_id, image_index, label_info, stack) #to make more flexible for new datastructures with more labelling info
                return([])
            else:
                if self.debug:
                    print(f"cell id {cell_id} is too close to the image edge to extract. Skipping this cell.")
                self.save_index_to_remove.append(save_index)
                return([save_index])
    
    def process(self, input_segmentation_path, filtered_classes_path):
        """
        Process function to run the extraction method.

        Args:
            input_segmentation_path (str): Path of the segmentation hdf5 file. IF this class is used as part of a project processing workflow this argument will be provided automatically.
            filtered_classes_path (str): Path of the filtered classes resulting from segementation. If this class is used as part of a project processing workflow this argument will be provided automatically.

        Important:
        
            If this class is used as part of a project processing workflow, all of the arguments will be provided by the ``Project`` class based on the previous segmentation. 
            The Project class will automaticly provide the most recent segmentation forward together with the supplied parameters. 

        Example:

            .. code-block:: python

                #after project is initialized and input data has been loaded and segmented
                project.extract()

        Note:
        
            The following parameters are required in the config file when running this method:
            
            .. code-block:: yaml

                HDF5CellExtraction:

                    compression: True
                    
                    #threads used in multithreading
                    threads: 80 

                    # image size in pixel
                    image_size: 128 
                    
                    # directory where intermediate results should be saved
                    cache: "/mnt/temp/cache"

                    #specs to define how hdf5 data should be chunked and saved
                    hdf5_rdcc_nbytes: 5242880000 # 5gb 1024 * 1024 * 5000 
                    hdf5_rdcc_w0: 1
                    hdf5_rdcc_nslots: 50000
    
        """
        # is called with the path to the segmented image
        
        self.get_channel_info() # needs to be called here after the segmentation is completed
        self.setup_output()
        self.parse_remapping()
        
        # setup cache
        self.uuid = str(uuid.uuid4())
        self.extraction_cache = os.path.join(self.config["cache"],self.uuid)
        if not os.path.isdir(self.extraction_cache):
            os.makedirs(self.extraction_cache)
            self.log("Created new extraction cache " + self.extraction_cache)
            
        self.log("Started extraction")
        self.log("Loading segmentation data from {input_segmentation_path}")
    
        hf = h5py.File(input_segmentation_path, 'r')
        hdf_channels = hf.get(self.channel_label)
        hdf_labels = hf.get(self.segmentation_label)

        self.log(f"Using channel label {hdf_channels}")
        self.log(f"Using segmentation label {hdf_labels}")
        self.log("Finished loading channel data " + str(hdf_channels.shape))
        self.log("Finished loading label data " + str(hdf_labels.shape))
        self.n_masks = hdf_labels.shape[0]

        # Calculate centers
        self.log("Checked class coordinates")
        
        center_path = os.path.join(self.directory, "center.pickle")
        cell_ids_path = os.path.join(self.directory, "_cell_ids.pickle")

        if os.path.isfile(center_path) and os.path.isfile(cell_ids_path) and not self.overwrite:
            self.log("Cached version found, loading")
            with open(center_path, "rb") as input_file:
                center_nuclei = cPickle.load(input_file)
                px_centers = np.round(center_nuclei).astype(int)
            with open(cell_ids_path, "rb") as input_file:
                _cell_ids = cPickle.load(input_file)
        else:
            self.log("Started class coordinate calculation")
            center_nuclei, length, _cell_ids = numba_mask_centroid(hdf_labels[0].astype(np.uint32), debug=self.debug)
            px_centers = np.round(center_nuclei).astype(int)
            self.log("Finished class coordinate calculation")
            with open(center_path, "wb") as output_file:
                cPickle.dump(center_nuclei, output_file)
            with open(cell_ids_path, "wb") as output_file:
                cPickle.dump(_cell_ids, output_file)
            with open(os.path.join(self.directory,"length.pickle"), "wb") as output_file:
                cPickle.dump(length, output_file)
                
            del length

        class_list = self.get_classes(filtered_classes_path)    
        lookup_saveindex = self.generate_save_index_lookup(class_list)           
        
        #make into set to improve computational efficiency
        #needs to come after generating lookup index otherwise it will throw an error message
        class_list = set(class_list)
        
        #filter cell ids found using center into those that we actually want to extract
        _cell_ids = list(_cell_ids)
        filter = [x in class_list for x in _cell_ids]

        px_centers = np.array(list(compress(px_centers, filter)))
        _cell_ids = list(compress(_cell_ids, filter))

        #update number of classes

        self.log(f"Number of classes found in filtered classes list {len(class_list)} vs number of classes for which centers were calculated {len(_cell_ids)}")
        self.num_classes = len(_cell_ids)
        
        # setup cache
        self._initialize_tempmmap_array()
        
        #start extraction
        self.verbalise_extraction_info()

        self.log(f"Starting extraction of {self.num_classes} classes")
        start = timeit.default_timer()

        f = partial(self._extract_classes, input_segmentation_path, px_centers)
        args = self._get_arg(_cell_ids, lookup_saveindex)

        with Pool(processes = self.config["threads"]) as pool:
            x = list(tqdm(pool.imap(f, args), total = len(args)))
            pool.close()
            pool.join()
            print("multiprocessing done.")
        
        stop = timeit.default_timer()
        self.save_index_to_remove = flatten(x)

        #calculate duration
        duration = stop - start
        rate = self.num_classes/duration

        #generate final log entries
        self.log(f"Finished extraction in {duration:.2f} seconds ({rate:.2f} cells / second)")
        self.log("Collecting cells...")

        #make into set to improve computational efficiency
        #transfer results to hdf5
        self._transfer_tempmmap_to_hdf5()
        self.log("Finished cleaning up cache.")

class TimecourseHDF5CellExtraction(HDF5CellExtraction):
    """
    A class to extracts single cell images from a segmented SPARCSpy Timecourse project and save the 
    results to an HDF5 file.

    Functionality is the same as the HDF5CellExtraction except that the class is able to deal with an additional dimension(t)
    in the input data.
    """

    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_SEGMENTATION_DIR = "segmentation"
    DEFAULT_SEGMENTATION_FILE = "input_segmentation.h5"

    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False
    
    #new parameters to make workflow adaptable to other types of projects
    channel_label = "input_images"
    segmentation_label = "segmentation"

    def __init__(self, 
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

    def get_labelling(self):
        with h5py.File(self.input_segmentation_path, 'r') as hf:
            self.label_names = hf.get("label_names")[:]
            self.n_labels = len(self.label_names)

    def _get_arg(self):
        #need to extract ID for each cellnumber
        
        #generate lookuptable where we have all cellids for each tile id
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labels = hf.get("labels").asstr()[:]
            classes = hf.get("classes")

            results = pd.DataFrame(columns = ["tileids", "cellids"], index = range(labels.shape[0]))
        
            self.log({"Extracting classes from each Segmentation Tile."})
            # should be updated later when classes saved in segmentation automatically 
            # currently not working because of issue with datatypes
            
            for i, tile_id in zip(labels.T[0], labels.T[1]):
                #dirty fix for some strange problem with some of the datasets
                #FIX THIS
                if i == "":
                    continue
                cellids =list(classes[int(i)])
                
                #remove background
                if 0 in cellids:
                    cellids.remove(0)
                
                results.loc[int(i), "cellids"] = cellids
                results.loc[int(i), "tileids"] = tile_id

        #map each cell id to tile id and generate a tuple which can be passed to later functions
        return_results = [[(xset, i, results.loc[i, "tileids"]) for i, xset in enumerate(results.cellids)]]
        return_results = flatten(return_results)
        
        #required format 
        return(return_results)

    def _get_label_info(self, arg):
        arg_index, save_index, cell_id, image_index, label_info = arg
        return(arg_index, save_index, cell_id, image_index, label_info)  

    def _initialize_tempmmap_array(self):
        #define as global variables so that this is also avaialable in other functions
        global _tmp_single_cell_data, _tmp_single_cell_index
        
        #import tempmmap module and reset temp folder location
        from alphabase.io import tempmmap
        TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])

        #generate datacontainer for the single cell images
        column_labels = ['index', "cellid"] + list(self.label_names.astype("U13"))[1:]
        self.single_cell_index_shape =  (self.num_classes, len(column_labels))
        self.single_cell_data_shape = (self.num_classes,
                                                    self.n_channels_output,
                                                    self.config["image_size"],
                                                    self.config["image_size"])

        #generate container for single_cell_data
        print(self.single_cell_data_shape)
        _tmp_single_cell_data = tempmmap.array(self.single_cell_data_shape, dtype = np.float16)

        #generate container for single_cell_index
        #cannot be a temmmap array with object type as this doesnt work for memory mapped arrays
        #dt = h5py.special_dtype(vlen=str)
        _tmp_single_cell_index  = np.empty(self.single_cell_index_shape, dtype = "<U64") #need to use U64 here otherwise information potentially becomes truncated
        
        #_tmp_single_cell_index  = tempmmap.array(self.single_cell_index_shape, dtype = "<U32")

        self.TEMP_DIR_NAME = TEMP_DIR_NAME

    def _transfer_tempmmap_to_hdf5(self):
        global _tmp_single_cell_data, _tmp_single_cell_index   

        self.log(f"number of cells too close to image edges to extract: {len(self.save_index_to_remove)}")
        self.log(f"{_tmp_single_cell_data.shape} shape of single-cell data before removing cells to close to image edges")
        _tmp_single_cell_data = np.delete(_tmp_single_cell_data, self.save_index_to_remove, axis=0)
        _tmp_single_cell_index = np.delete(_tmp_single_cell_index, self.save_index_to_remove, axis=0)
        self.log(f"{_tmp_single_cell_data.shape} shape of single-cell data after removing cells to close to image edges")

        #extract information about the annotation of cell ids
        column_labels = ['index', "cellid"] + list(self.label_names.astype("U13"))[1:]
        
        self.log("Creating HDF5 file to save results to.")
        with h5py.File(self.output_path, 'w') as hf:
            #create special datatype for storing strings
            dt = h5py.special_dtype(vlen=str)

            #save label names so that you can always look up the column labelling
            hf.create_dataset('label_names', data = column_labels, chunks=None, dtype = dt)
            
            #generate index data container
            hf.create_dataset('single_cell_index_labelled', _tmp_single_cell_index.shape , chunks=None, dtype = dt)
            single_cell_labelled = hf.get("single_cell_index_labelled")
            single_cell_labelled[:] = _tmp_single_cell_index[:]

            hf.create_dataset('single_cell_index', (_tmp_single_cell_index.shape[0], 2), dtype="uint64")           

            hf.create_dataset('single_cell_data',data =  _tmp_single_cell_data,
                                                chunks=(1,
                                                        1,
                                                        self.config["image_size"],
                                                        self.config["image_size"]),
                                                compression=self.compression_type,
                                                dtype="float16")
            
        self.log(f"Transferring exracted single cells to .hdf5")
        with h5py.File(self.output_path, 'a') as hf:
            #need to save this index seperately since otherwise we get issues with the classificaiton of the extracted cells
            index = _tmp_single_cell_index[:, 0:2]
            _, cell_ids = index.T
            index = np.array(list(zip(range(len(cell_ids)), cell_ids)))
            index[index == ""] = "0" 
            index = index.astype("uint64")
            hf["single_cell_index"][:] = index

        #delete tempobjects (to cleanup directory)
        self.log(f"Tempmmap Folder location {self.TEMP_DIR_NAME} will now be removed.")
        shutil.rmtree(self.TEMP_DIR_NAME, ignore_errors=True)

        del _tmp_single_cell_data, _tmp_single_cell_index, self.TEMP_DIR_NAME 

    def _save_cell_info(self, index, cell_id, image_index, label_info, stack):
        global _tmp_single_cell_data, _tmp_single_cell_index
        #label info is None so just ignore for the base case
        
        #save single cell images
        _tmp_single_cell_data[index] = stack
        # print("index:", index)
        # import matplotlib.pyplot as plt
        
        # for i in stack:
        #         plt.figure()
        #         plt.imshow(i)
        #         plt.show()

        #get label information
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labelling = hf.get("labels").asstr()[image_index][1:]
            save_value = [str(index), str(cell_id)]
            save_value = np.array(flatten([save_value, labelling]))

            _tmp_single_cell_index[index] = save_value

            #double check that its really the same values
            if _tmp_single_cell_index[index][2] != label_info:
                self.log("ISSUE INDEXES DO NOT MATCH.")
                self.log(f"index: {index}")
                self.log(f"image_index: {image_index}")
                self.log(f"label_info: {label_info}")
                self.log(f"index it should be: {_tmp_single_cell_index[index][2]}")
    
    def process(self, input_segmentation_path, filtered_classes_path):
        
        """
        Process function to run the extraction method. 

        Args:
            input_segmentation_path: str
                Path of the segmentation hdf5 file. IF this class is used as part of a project processing workflow this argument will be provided automatically.
            filtered_classes_path: str
                Path of the filtered classes resulting from segementation. If this class is used as part of a project processing workflow this argument will be provided automatically.

        Important:
        
            If this class is used as part of a project processing workflow, all of the arguments will be provided by the ``Project`` class based on the previous segmentation. 
            The Project class will automaticly provide the most recent segmentation forward together with the supplied parameters. 

        Example:

            .. code-block:: python
            
                #after project is initialized and input data has been loaded and segmented
                project.extract()

        Note:
        
            The following parameters are required in the config file when running this method:
            
            .. code-block:: yaml

                HDF5CellExtraction:

                    compression: True
                    
                    #threads used in multithreading
                    threads: 80 

                    # image size in pixel
                    image_size: 128 
                    
                    # directory where intermediate results should be saved
                    cache: "/mnt/temp/cache"

                    #specs to define how hdf5 data should be chunked and saved
                    hdf5_rdcc_nbytes: 5242880000 # 5gb 1024 * 1024 * 5000 
                    hdf5_rdcc_w0: 1
                    hdf5_rdcc_nslots: 50000
    
        """
        # is called with the path to the segmented image
        
        self.get_labelling()
        self.get_channel_info()
        self.setup_output()
        self.parse_remapping()

        complete_class_list = self.get_classes(filtered_classes_path)
        arg_list = self._get_arg()
        lookup_saveindex = self.generate_save_index_lookup(complete_class_list)

        # setup cache
        self._initialize_tempmmap_array()

        #start extraction
        self.log("Starting extraction.")
        self.verbalise_extraction_info()

        with  h5py.File(self.input_segmentation_path, 'r') as hf:
            start = timeit.default_timer()

            self.log(f"Loading segmentation data from {self.input_segmentation_path}")
            hdf_labels = hf.get(self.segmentation_label)

            for arg in tqdm(arg_list):
                cell_ids, image_index, label_info = arg 
                # print("image index:", image_index)
                # print("cell ids", cell_ids)
                # print("label info:", label_info)
                
                input_image = hdf_labels[image_index, 0, :, :]

                #check if image is an empty array
                if np.all(input_image==0):
                    self.log(f"Image with the image_index {image_index} only contains zeros. Skipping this image.")
                    print(f"Error: image with the index {image_index} only contains zeros!! Skipping this image.")
                    continue
                else:
                    center_nuclei, _, _cell_ids = numba_mask_centroid(input_image, debug=self.debug)

                    if center_nuclei is not None:
                        px_centers = np.round(center_nuclei).astype(int)
                        _cell_ids = list(_cell_ids)

                        # #plotting results for debugging
                        # import matplotlib.pyplot as plt
                        # plt.figure(figsize = (10, 10))
                        # plt.imshow(hdf_labels[image_index, 1, :, :])
                        # plt.figure(figsize = (10, 10))
                        # plt.imshow(hdf_labels[image_index, 0, :, :])
                        # y, x = px_centers.T
                        # plt.scatter(x, y, color = "red", s = 5)
                        
                        #filter lists to only include those cells which passed the final filters (i.e remove border cells)
                        filter = [x in cell_ids for x in _cell_ids]
                        px_centers = np.array(list(compress(px_centers, filter)))
                        _cell_ids = list(compress(_cell_ids, filter))

                        # #plotting results for debugging
                        # y, x = px_centers.T
                        # plt.scatter(x, y, color = "blue", s = 5)
                        # plt.show()

                        for centers_index, cell_id in enumerate(_cell_ids):
                            save_index = lookup_saveindex.index.get_loc(cell_id)
                            self._extract_classes(input_segmentation_path, px_centers, (centers_index, save_index, cell_id, image_index, label_info))
                    else:
                        self.log(f"Image with the image_index {image_index} doesn't contain any cells. Skipping this image.")
                        print(f"Error: image with the index {image_index} doesn't contain any cells!! Skipping this image.")
                        continue

            stop = timeit.default_timer()

        duration = stop - start
        rate = self.num_classes/duration
        self.log(f"Finished parallel extraction in {duration:.2f} seconds ({rate:.2f} cells / second)")
        
        self.log("Collect cells")
        self._transfer_tempmmap_to_hdf5()
        self.log("Extraction completed.")