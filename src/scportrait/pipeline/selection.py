from scportrait.pipeline.base import ProcessingStep
import os
import numpy as np
import h5py
import pickle
from lmd.lib import SegmentationLoader
from alphabase.io import tempmmap
from lmd.segmentation import _create_coord_index_sparse
import timeit

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
        assert "segmentation_channel" in self.config, "segmentation_channel not defined in config"

        self.segmentation_channel_to_select = self.config["segmentation_channel"]

        #the coord pickle file should be saved in the same directory as the segmentation results because it is based on that segmentation (if that segmentation is updated or changed it should also be recalculated)
        self.coord_pickle_file_path = os.path.join(self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, f"{self.segmentation_channel_to_select}_{self.COORD_PICKLE_FILE}")

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
        if os.path.exists(self.coord_pickle_file_path):
            self.log(f"Loading coordinate lookup index from file {self.coord_pickle_file_path}.")
            with open(self.coord_pickle_file_path, "rb") as f:
                coord_index = pickle.load(f)
            segmentation = None
        else:
            self.log("Calculating coordinate lookup index.")
            
            #start timer for performance evaluation
            start_time = timeit.default_timer()

            # load segmentation from hdf5
            with h5py.File(hdf_location, "r") as hf:
                hdf_labels = hf.get("labels")

                # create memory mapped temporary array for saving the segmentation
                c, x, y = hdf_labels.shape
                segmentation = tempmmap.array(
                    shape=(x, y), dtype=hdf_labels.dtype, tmp_dir_abs_path=self._tmp_dir_path
                )
                segmentation[:] = hdf_labels[self.config["segmentation_channel"], :, :]
                
            coord_index = dict(_create_coord_index_sparse(segmentation))
            
            with open(self.coord_pickle_file_path, "wb") as f:
                pickle.dump(coord_index, f)
            self.log(f"Coordinate lookup index saved to file {self.coord_pickle_file_path}.")
            self.log(f"Coordinate lookup index calculation took {timeit.default_timer() - start_time} seconds.")
            
        #add default orientation transform
        self.config["orientation_transform"] = np.array([[0, -1], [1, 0]])

        sl = SegmentationLoader(
            config=self.config,
            verbose=self.debug,
            processes=self.config["processes_cell_sets"],
        )

        shape_collection = sl(segmentation, cell_sets, calibration_marker, coords_lookup=coord_index)

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

        del segmentation

        self.log(f"Saved output at {savepath}")
