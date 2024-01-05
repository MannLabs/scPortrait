import os
import numpy as np
import csv
import h5py
from multiprocessing import Pool
import shutil
import pandas as pd

import traceback

from sparcscore.processing.segmentation import sc_any
from sparcscore.pipeline.base import ProcessingStep

# to show progress
from tqdm.auto import tqdm

#to perform garbage collection
import gc
import sys

class SegmentationFilter(ProcessingStep):
    """SegmentationFilter helper class used for creating workflows to filter generated segmentation masks before extraction.

    """
    DEFAULT_OUTPUT_FILE = "segmentation.h5"
    DEFAULT_FILTER_FILE = "filtered_classes.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.input_path = input_segmentation
        self.identifier = None
        self.window = None
        self.input_path = None
    
    def read_input_masks(self, input_path):

        with h5py.File(input_path, "r") as hf:
            hdf_input = hf.get("labels")

            #use a memory mapped numpy array to save the input image to better utilize memory consumption
            from alphabase.io import tempmmap
            TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])
            self.TEMP_DIR_NAME = TEMP_DIR_NAME #save for later to be able to remove cached folders

            input_masks = tempmmap.array(shape = hdf_input.shape, dtype = np.uint16)
            input_masks = hdf_input[:2,:, :]

        return(input_masks)
    
    def save_classes(self, classes):
        
        #define path where classes should be saved
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)

        to_write = "\n".join([f"{str(x)}:{str(y)}" for x, y in classes.items()])

        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        self.log(f"Saved nucleus_id:cytosol_id matchings of all cells that passed filtering to {filtered_path}.")

    def initialize_as_tile(self, identifier, window, input_path, zarr_status = True):
        """Initialize Filtering Step with further parameters needed for filtering segmentation results.

        Important:
            This function is intended for internal use by the :class:`TiledFilterSegmentation` helper class. In most cases it is not relevant to the creation of custom filtering workflows.

        Args:
            identifier (int): Unique index of the tile.
            window (list(tuple)): Defines the window which is assigned to the tile. The window will be applied to the input. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.
            input_path (str): Location of the input hdf5 file. During tiled segmentation the :class:`TiledSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.
        """
        self.identifier = identifier
        self.window = window
        self.input_path = input_path
        self.save_zarr = zarr_status

    def call_as_tile(self):
        """Wrapper function for calling a tiled segmentation.

        Important:
            This function is intended for internal use by the :class:`TiledSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.
        """
    
        with h5py.File(self.input_path, "r") as hf:
            hdf_input = hf.get("labels")

            #use a memory mapped numpy array to save the input image to better utilize memory consumption
            from alphabase.io import tempmmap
            TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])

            #calculate shape of required datacontainer
            c, _, _ = hdf_input.shape
            x1 = self.window[0].start
            x2 = self.window[0].stop
            y1 = self.window[1].start
            y2 = self.window[1].stop

            x = x2 - x1
            y = y2 - y1

            #initialize directory and load data
            input_image = tempmmap.array(shape = (2, x, y), dtype = np.uint16)
            input_image = hdf_input[:2, self.window[0], self.window[1]]

        #perform check to see if any input pixels are not 0, if so perform segmentation, else return array of zeros.
        if sc_any(input_image):
            try:
                self.log(f"Beginning filtering on tile in position [{self.window[0]}, {self.window[1]}]")
                super().__call__(input_image)
            except Exception:
                self.log(traceback.format_exc())
        else:  
            print(f"Tile in position [{self.window[0]}, {self.window[1]}] only contained zeroes.")
            try:
                super().__call_empty__(input_image)
            except Exception:
                self.log(traceback.format_exc())

        #cleanup generated temp dir and variables
        del input_image
        gc.collect()

        #write out window location
        self.log(f"Writing out window location to file at {self.directory}/window.csv")
        with open(f"{self.directory}/window.csv", "w") as f:
            f.write(f"{self.window}\n")  

        self.log(f"Filtering of tile with the slicing {self.window} finished.")

        #delete generate temp directory to cleanup space
        shutil.rmtree(TEMP_DIR_NAME, ignore_errors=True)

    def get_output(self):
        return os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)   
        
class TiledSegmentationFilter(SegmentationFilter):
    """"""

    DEFAULT_TILES_FOLDER = "tiles"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "method"):
            raise AttributeError(
                "No SegmentationFilter method defined, please set attribute ``method``"
            )
    
    def initialize_tile_list(self, tileing_plan, input_path):
        _tile_list = []

        self.input_path = input_path

        for i, window in enumerate(tileing_plan):
            local_tile_directory = os.path.join(self.tile_directory, str(i))
            current_tile = self.method(
                self.config,
                local_tile_directory,
                project_location = self.project_location,
                debug=self.debug,
                overwrite=self.overwrite,
                intermediate_output=self.intermediate_output,
            )
            current_tile.initialize_as_tile(i, window, self.input_path, zarr_status = False)
            _tile_list.append(current_tile)

        return _tile_list

    def calculate_tileing_plan(self, mask_size):
        #save tileing plan to file
        tileing_plan_path = f"{self.directory}/tileing_plan.csv"

        if os.path.isfile(tileing_plan_path):
            self.log(f"tileing plan already found in directory {tileing_plan_path}.")
            if self.overwrite:
                self.log("Overwriting existing tileing plan.")
                os.remove(tileing_plan_path)
            else:
                self.log("Reading existing tileing plan from file.")
                with open(tileing_plan_path, "r") as f:
                    _tileing_plan = [eval(line) for line in f.readlines()]
                    return(_tileing_plan)

        _tileing_plan = []
        side_size = np.floor(np.sqrt(int(self.config["tile_size"])))
        tiles_side = np.round(mask_size / side_size).astype(int)
        tile_size = mask_size // tiles_side

        self.log(f"input image {mask_size[0]} px by {mask_size[1]} px")
        self.log(f"target_tile_size: {self.config['tile_size']}")
        self.log(f"tileing plan:")
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

                #add px overlap to each tile
                lower_y = lower_y - self.config["overlap_px"]
                lower_x = lower_x - self.config["overlap_px"]
                upper_y = upper_y + self.config["overlap_px"]
                upper_x = upper_x + self.config["overlap_px"]

                #make sure that each limit stays within the slides
                if lower_y < 0:
                    lower_y = 0
                if lower_x < 0:
                    lower_x = 0

                if last_row:
                    upper_y = mask_size[0]

                if last_column:
                    upper_x = mask_size[1]

                tile = (slice(lower_y, upper_y), slice(lower_x, upper_x))
                _tileing_plan.append(tile)
    
        #write out newly generated tileing plan
        with open(tileing_plan_path, "w") as f:
            for tile in _tileing_plan:
                f.write(f"{tile}\n")
        self.log(f"Tileing plan written to file at {tileing_plan_path}")
        
        return _tileing_plan

    def resolve_tileing(self, tileing_plan):
        """
        The function iterates over a tileing plan and generates a converged list of all nucleus_id:cytosol_id matchings.
        """

        self.log("resolve tileing plan and joining generated lists together")

        #initialize empty list to save results to
        filtered_classes_combined = []

        for i, window in enumerate(tileing_plan):

            local_tile_directory = os.path.join(self.tile_directory, str(i))
            local_output = os.path.join(local_tile_directory, self.DEFAULT_OUTPUT_FILE)
            local_classes = os.path.join(local_tile_directory, "filtered_classes.csv")

            #check to make sure windows match
            with open(f"{local_tile_directory}/window.csv", "r") as f:
                window_local =  eval(f.read())
            if window_local != window:
                self.log("Tileing plans do not match. Aborting run.")
                self.log("Tileing plan found locally: ", window_local)
                self.log("Tileing plan found in tileing plan: ", window)
                sys.exit("tileing plans do not match!")

            cr = csv.reader(open(local_classes, "r"))
            filtered_classes = [el[0] for el in list(cr)]
            
            filtered_classes_combined += filtered_classes
            self.log(f"Finished stitching tile {i}")

        #remove duplicates from list (this only removes perfect duplicates)
        filtered_classes_combined = list(set(filtered_classes_combined))
        
        #perform sanity check that no cytosol_id is listed twice
        filtered_classes_combined = {int(k): int(v) for k, v in (s.split(":") for s in filtered_classes_combined)}
        if len(filtered_classes_combined.values()) != len(set(filtered_classes_combined.values())):
            print(pd.Series(filtered_classes_combined.values()).value_counts())
            print(filtered_classes_combined)
            sys.exit("Duplicate values found. Some issues with filtering. Please contact the developers.")

        # save newly generated class list to file
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)
        to_write = "\n".join([f"{str(x)}:{str(y)}" for x, y in filtered_classes_combined.items()])
        with open(filtered_path, "w") as myfile:
            myfile.write(to_write)

        # Add section here that cleans up the results from the tiles and deletes them to save memory
        self.log("Deleting intermediate tile results to free up storage space")
        shutil.rmtree(self.tile_directory, ignore_errors=True)

        gc.collect()

    def process(self, input_path):

        self.tile_directory = os.path.join(self.directory, self.DEFAULT_TILES_FOLDER)

        if not os.path.isdir(self.tile_directory):
            os.makedirs(self.tile_directory)
            self.log("Created new tile directory " + self.tile_directory)

        # calculate tileing plan
        with h5py.File(input_path, "r") as hf:
            self.mask_size = hf["labels"].shape[1:]
        
        if self.config["tile_size"] >= np.prod(self.mask_size):
            target_size = self.config["tile_size"]
            self.log(f"target size {target_size} is equal or larger to input mask {np.prod(self.mask_size)}. Tileing will not be used.")

            tileing_plan = [
                (slice(0, self.mask_size[0]), slice(0, self.mask_size[1]))
            ]

        else:
            target_size = self.config["tile_size"]
            self.log(f"target size {target_size} is smaller than input mask {np.prod(self.mask_size)}. Tileing will be used.")
            tileing_plan = self.calculate_tileing_plan(self.mask_size)

        #save tileing plan to file to be able to reload later
        self.log(f"Saving Tileing plan to file: {self.directory}/tileing_plan.csv")
        with open(f"{self.directory}/tileing_plan.csv", "w") as f:
            for tile in tileing_plan:
                f.write(f"{tile}\n")

        tile_list = self.initialize_tile_list(tileing_plan, input_path)
        
        self.log(
            f"tileing plan with {len(tileing_plan)} elements generated, tileing with {self.config['threads']} threads begins"
        )

        with Pool(processes=self.config['threads']) as pool:
            results = list(
                tqdm(
                    pool.imap(self.method.call_as_tile, tile_list),
                    total=len(tile_list),
                )
            )
            pool.close()
            pool.join()
            print("All Filtering Steps are done.", flush=True)
        
        #free up memory
        del tile_list
        gc.collect()

        self.log("Finished tiled filtering.")
        self.resolve_tileing(tileing_plan)

        #make sure to cleanup temp directories
        self.log("=== finished filtering === ")