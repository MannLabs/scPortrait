import os
from typing import List, Dict, Union

import numpy as np
from lmd.lib import SegmentationLoader
from alphabase.io import tempmmap

from scportrait.pipeline.base import ProcessingStep


class LMDSelection(ProcessingStep):
    """
    Select single cells from a segmented sdata file and generate cutting data for the Leica LMD microscope.
    This method class relies on the functionality of the pylmd library.
    """

    # define all valid path optimization methods used with the "path_optimization" argument in the configuration
    VALID_PATH_OPTIMIZERS = ["none", "hilbert", "greedy"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = None
        self.cell_sets = None
        self.calibration_marker = None

    def _setup_selection(self):
        # set orientation transform
        self.config["orientation_transform"] = np.array([[0, -1], [1, 0]])

        # configure name of extraction
        if self.name is None:
            try:
                name = "_".join([cell_set["name"] for cell_set in self.cell_sets])
            except Exception:
                name = "selected_cells"

        # create savepath
        savename = name.replace(" ", "_") + ".xml"
        self.savepath = os.path.join(self.directory, savename)

    def _post_processing_cleanup(self, vars_to_delete: Union[List, None] = None):
        if vars_to_delete is not None:
            self._clear_cache(vars_to_delete=vars_to_delete)

        # remove temporary files
        if hasattr(self, "path_seg_mask"):
            os.remove(self.path_seg_mask)

        self._clear_cache()

    def process(
        self,
        segmentation_name: str,
        cell_sets: List[Dict],
        calibration_marker: np.array,
        name: Union[str, None] = None,
    ):
        """
        Process function for selecting cells and generating their XML.
        Under the hood this method relies on the pylmd library and utilizies its `SegmentationLoader` Class.

        Args:
            segmentation_name (str): Name of the segmentation to be used for shape generation in the sdata object.
            cell_sets (list of dict): List of dictionaries containing the sets of cells which should be sorted into a single well. Mandatory keys for each dictionary are: name, classes. Optional keys are: well.
            calibration_marker (numpy.array): Array of size ‘(3,2)’ containing the calibration marker coordinates in the ‘(row, column)’ format.

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
                    # will processed with the designated number of threads
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

        self.log("Selection process started.")

        self.name = name
        self.cell_sets = cell_sets
        self.calibration_marker = calibration_marker

        self._setup_selection()

        ## TO Do
        # check if classes and seglookup table already exist as pickle file
        # if not create them
        # else load them and proceed with selection

        # load segmentation from hdf5
        self.path_seg_mask = self.project.__load_seg_to_memmap(
            segmentation_name, tmp_dir_abs_path=self._tmp_dir_path
        )
        segmentation = tempmmap.mmap_array_from_path(self.path_seg_mask)

        # create segmentation loader
        sl = SegmentationLoader(
            config=self.config,
            verbose=self.debug,
            processes=self.config["processes_cell_sets"],
        )

        # get shape collections
        shape_collection = sl(segmentation, self.cell_sets, self.calibration_marker)

        if self.debug:
            shape_collection.plot(calibration=True)
            shape_collection.stats()

        shape_collection.save(self.savepath)

        self.log(f"Saved output at {self.savepath}")

        # perform post processing cleanup
        self._post_processing_cleanup(
            vars_to_delete=[shape_collection, sl, segmentation]
        )
