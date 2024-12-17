
import os
import numpy as np
import h5py
import pickle
from tqdm.auto import tqdm
import timeit
import pandas as pd
from scipy.sparse import coo_array
from functools import partial as func_partial
import multiprocessing as mp
from scportrait.processing.utils import flatten
from scportrait.pipeline.base import ProcessingStep
from lmd.lib import SegmentationLoader
from pathlib import Path

class LMDSelection(ProcessingStep):
    """
    Select single cells from a segmented hdf5 file and generate cutting data for the Leica LMD microscope.
    This method class relies on the functionality of the pylmd library.
    """

    # define all valid path optimization methods used with the "path_optimization" argument in the configuration
    VALID_PATH_OPTIMIZERS = ["none", "hilbert", "greedy"]
    COORD_PICKLE_FILE = "coord_index.pkl"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #check config for required parameters
        self._check_config()

    def _check_config(self):
        #check mandatory config parameters
        assert "segmentation_channel" in self.config, "segmentation_channel not defined in config"
        self.segmentation_channel_to_select = self.config["segmentation_channel"]

        # check for optional config parameters
        if "cell_width" in self.config:
            self.cell_width = self.config["cell_width"]
        else:
            self.cell_radius = 100 
        
        if "threads" in self.config:
            self.threads = self.config["threads"]
            assert self.threads > 0, "threads must be greater than 0"
            assert isinstance(self.threads, int), "threads must be an integer"
        else:
            self.threads = 10
        
        if "batch_size_coordinate_extraction" in self.config:
            self.batch_size = self.config["batch_size_coordinate_extraction"]
            assert self.batch_size > 0, "batch_size_coordinate_extraction must be greater than 0"
            assert isinstance(self.batch_size, int), "batch_size_coordinate_extraction must be an integer"
        else:
            self.batch_size = 100
        
    def __get_coords(self, 
                     cell_ids: list, 
                     centers:list[tuple[int, int]], 
                     hdf_path:str, 
                     segmentation_channel:int, 
                     width:int = 60) -> list[tuple[int, np.ndarray]]:
        results = []
        
        with h5py.File(hdf_path, "r") as hf:
            hdf_labels = hf["labels"]
            for i, _id in enumerate(cell_ids):
                values = centers[i]

                x_start = np.max([int(values[0]) - width, 0])
                y_start = np.max([int(values[1]) - width, 0])
            
                x_end = x_start + width*2
                y_end = y_start + width*2
        
                _cropped = hdf_labels[segmentation_channel, slice(x_start, x_end), slice(y_start, y_end)]    
                sparse = coo_array(_cropped == _id)
            
                x = sparse.coords[0] + x_start
                y = sparse.coords[1] + y_start
                
                results.append((_id, np.array(list(zip(x, y)))))
        return(results)
    
    def _get_coords_multi(self, hdf_path:str, segmentation_channel:int, width:int, arg: tuple[list[int], np.ndarray]) -> list[tuple[int, np.ndarray]]:
        cell_ids, centers = arg
        results = self.__get_coords(cell_ids, centers, hdf_path, segmentation_channel, width)
        return(results)
    
    def _get_coords(self,
                    cell_ids: list, 
                    centers:list[tuple[int, int]], 
                    hdf_path:str, 
                    segmentation_channel:int, 
                    width:int = 60, 
                    batch_size:int = 100, 
                    threads:int = 10) -> dict:

        #create batches
        n_batches = int(np.ceil(len(cell_ids)/batch_size))
        slices = [(i*batch_size, i*batch_size + batch_size) for i in range(n_batches - 1)]
        slices.append(((n_batches - 1)*batch_size, len(cell_ids)))
        
        batched_args = [(cell_ids[start:end], centers[start:end]) for start, end in slices]

        f = func_partial(self._get_coords_multi,
                        hdf_path, 
                        segmentation_channel, 
                        width
            )
        
        if threads == 1: # if only one thread is used, the function is called directly to avoid the overhead of multiprocessing
            results = [f(arg) for arg in batched_args]
        else:
            with mp.get_context(self.context).Pool(processes=threads) as pool: 
                results = list(tqdm(
                        pool.imap(f, batched_args),
                        total=len(batched_args),
                        desc="Processing cell batches",
                    )
                )
                pool.close()
                pool.join()

        results = flatten(results)
        return(dict(results))
    
    def _get_cell_ids(self, cell_sets: list[dict]) -> list[int]:
        cell_ids = []
        for cell_set in cell_sets:
            if "classes" in cell_set:
                cell_ids.extend(cell_set["classes"])
            else:
                Warning(f"Cell set {cell_set['name']} does not contain any classes.")
        return(cell_ids)
    
    def _get_centers(self, cell_ids: list[int]) -> list[tuple[int, int]]:
        centers_path = Path(self.project_location) / "extraction" / "center.pickle"
        _ids_path = Path(self.project_location) / "extraction" / "_cell_ids.pickle"

        if centers_path.exists() and _ids_path.exists():
            with open(centers_path, "rb") as f:
                centers = pickle.load(f)
            with open(_ids_path, "rb") as f:
                _ids = pickle.load(f)
        else:
            raise ValueError("Center and cell id files not found.")
        
        centers = pd.DataFrame(centers, columns=["x", "y"])
        
        #convert coordinates to integers for compatibility with indexing in segmentation mask
        centers.x = centers.x.astype(int)
        centers.y = centers.y.astype(int)
        centers["cell_id"] = _ids
        centers.set_index("cell_id", inplace=True)
        
        centers = centers.loc[cell_ids, :]

        return(centers[["x", "y"]].values.tolist())

    def process(self, hdf_location, cell_sets, calibration_marker, name=None):
        """
        Process function for selecting cells and generating their XML.
        Under the hood this method relies on the pylmd library and utilizies its `SegmentationLoader` Class.

        Args:
            hdf_location (str): Path of the segmentation hdf5 file. If this class is used as part of a project processing workflow, this argument will be provided.
            cell_sets (list of dict): List of dictionaries containing the sets of cells which should be sorted into a single well.
            calibration_marker (numpy.array): Array of size ‘(3,2)’ containing the calibration marker coordinates in the ‘(row, column)’ format.
            name (str): Name of the output file. If not provided, the name will be generated based on the names of the cell sets or if also not specified set to "selected_cells".

        Important:

            If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project``
            class based on the previous segmentation. Therefore, only the second and third argument need to be provided. The Project
            class will automatically provide the most recent segmentation together with the supplied parameters.

        Example:

            .. code-block:: python

                # Calibration marker should be defined as (row, column).
                marker_0 = np.array([-10,-10])
                marker_1 = np.array([-10,1100])
                marker_2 = np.array([1100,505])

                # A numpy Array of shape (3, 2) should be passed.
                calibration_marker = np.array([marker_0, marker_1, marker_2])

                # Sets of cells can be defined by providing a name and a list of classes in a dictionary.
                cells_to_select = [{"name": "dataset1", "classes": [1,2,3]}]

                # Alternatively, a path to a csv file can be provided.
                # If a relative path is provided, it is accessed relativ to the projects base directory.
                cells_to_select += [{"name": "dataset2", "classes": "segmentation/class_subset.csv"}]

                # If desired, wells can be passed with the individual sets.
                cells_to_select += [{"name": "dataset3", "classes": [4,5,6], "well":"A1"}]

                project.select(cells_to_select, calibration_marker)

        Note:

            The following parameters are required in the config file:

            .. code-block:: yaml

                LMDSelection:
                    #the number of threads with which multithreaded tasks should be executed
                    threads: 10

                    # the number of parallel processes to use for generation of cell sets each set
                    # will be processed with the designated number of threads
                    processes_cell_sets: 1

                    # defines the channel used for generating cutting masks
                    # segmentation.hdf5 => labels => segmentation_channel
                    # When using WGA segmentation:
                    #    0 corresponds to nuclear masks
                    #    1 corresponds to cytosolic masks.
                    segmentation_channel: 0

                    # dilation of the cutting mask in pixel
                    shape_dilation: 10

                    # Cutting masks are transformed by binary dilation and erosion
                    binary_smoothing: 3

                    # number of datapoints which are averaged for smoothing
                    # the number of datapoints over an distance of n pixel is 2*n
                    convolution_smoothing: 25

                    # fold reduction of datapoints for compression
                    poly_compression_factor: 30

                    # Optimization of the cutting path inbetween shapes
                    # optimized paths improve the cutting time and the microscopes focus
                    # valid options are ["none", "hilbert", "greedy"]
                    path_optimization: "hilbert"

                    # Paramter required for hilbert curve based path optimization.
                    # Defines the order of the hilbert curve used, which needs to be tuned with the total cutting area.
                    # For areas of 1 x 1 mm we recommend at least p = 4,  for whole slides we recommend p = 7.
                    hilbert_p: 7

                    # Parameter required for greedy path optimization.
                    # Instead of a global distance matrix, the k nearest neighbours are approximated.
                    # The optimization problem is then greedily solved for the known set of nearest neighbours until the first set of neighbours is exhausted.
                    # Established edges are then removed and the nearest neighbour approximation is recursivly repeated.
                    greedy_k: 20

                    # The LMD reads coordinates as integers which leads to rounding of decimal places.
                    # Points spread between two whole coordinates are therefore collapsed to whole coordinates.
                    # This can be mitigated by scaling the entire coordinate system by a defined factor.
                    # For a resolution of 0.6 um / px a factor of 100 is recommended.
                    xml_decimal_transform: 100

                    # Overlapping shapes are merged based on a nearest neighbour heuristic.
                    # All selected shapes closer than distance_heuristic pixel are checked for overlap.
                    distance_heuristic: 300

        """

        self.log("Selection process started")

        #calculate a coordinate lookup file where for each cell id the coordinates for their location in the segmentation mask are stored
        self.log("Calculating coordinate lookup index for the specified cell ids.")
        start_time = timeit.default_timer()
        cell_ids = self._get_cell_ids(cell_sets)
        centers = self._get_centers(cell_ids)
        coord_index = self._get_coords(cell_ids = cell_ids, 
                                        centers = centers, 
                                        hdf_path = hdf_location, 
                                        segmentation_channel = self.segmentation_channel_to_select, 
                                        width = self.cell_radius, 
                                        batch_size = self.batch_size, 
                                        threads = self.threads)
        self.log(f"Coordinate lookup index calculation took {timeit.default_timer() - start_time} seconds.")

        #add default orientation transform
        self.config["orientation_transform"] = np.array([[0, -1], [1, 0]])

        sl = SegmentationLoader(
            config=self.config,
            verbose=self.debug,
            processes=self.config["processes_cell_sets"],
        )

        shape_collection = sl(None, cell_sets, calibration_marker, coords_lookup=coord_index)

        if self.debug:
            shape_collection.plot(calibration=True)
            shape_collection.stats()

        if name is None:
            try:
                name = "_".join([cell_set["name"] for cell_set in cell_sets])
            except Exception:
                name = "selected_cells"

        savename = name.replace(" ", "_") + ".xml"
        savepath = os.path.join(self.directory, savename)
        shape_collection.save(savepath)

        self.log(f"Saved output at {savepath}")
