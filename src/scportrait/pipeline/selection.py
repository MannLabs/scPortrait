import multiprocessing as mp
import os
import pickle
import timeit
from functools import partial as func_partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alphabase.io import tempmmap
from lmd.lib import SegmentationLoader
from scipy.sparse import coo_array
from tqdm.auto import tqdm

from scportrait.pipeline._base import ProcessingStep
from scportrait.pipeline._utils.helper import flatten


class LMDSelection(ProcessingStep):
    """
    Select single cells from a segmented sdata file and generate cutting data for the Leica LMD microscope.
    This method class relies on the functionality of the pylmd library.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_config()

        self.name = None
        self.cell_sets = None
        self.calibration_marker = None

        self.deep_debug = False  # flag for deep debugging by developers

    def _check_config(self):
        assert "segmentation_channel" in self.config, "segmentation_channel not defined in config"
        self.segmentation_channel_to_select = self.config["segmentation_channel"]

        # check for optional config parameters

        # this defines how large the box mask around the center of a cell is for the coordinate extraction
        # assumption is that all pixels belonging to each mask are within the box otherwise they will be cut off during cutting contour generation

        if "cell_width" in self.config:
            self.cell_radius = self.config["cell_width"]
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

        if "orientation_transform" in self.config:
            self.orientation_transform = self.config["orientation_transform"]
        else:
            self.orientation_transform = np.array([[0, -1], [1, 0]])
            self.config["orientation_transform"] = (
                self.orientation_transform
            )  # ensure its also in config so its passed on to the segmentation loader

        if "processes_cell_sets" in self.config:
            self.processes_cell_sets = self.config["processes_cell_sets"]
            assert self.processes_cell_sets > 0, "processes_cell_sets must be greater than 0"
            assert isinstance(self.processes_cell_sets, int), "processes_cell_sets must be an integer"
        else:
            self.processes_cell_sets = 1

    def _setup_selection(self):
        # configure name of extraction
        if self.name is None:
            try:
                name = "_".join([cell_set["name"] for cell_set in self.cell_sets])
            except KeyError:
                Warning("No name provided for the selection. Will use default name.")
                name = "selected_cells"
        else:
            name = self.name

        # create savepath
        savename = name.replace(" ", "_") + ".xml"
        self.savepath = os.path.join(self.directory, savename)

        # check that the segmentation label exists
        assert (
            self.segmentation_channel_to_select in self.project.filehandler.get_sdata()._shared_keys
        ), f"Segmentation channel {self.segmentation_channel_to_select} not found in sdata."

    def __get_coords(
        self, cell_ids: list, centers: list[tuple[int, int]], width: int = 60
    ) -> list[tuple[int, np.ndarray]]:
        results = []

        _sdata = self.project.filehandler.get_sdata()
        for i, _id in enumerate(cell_ids):
            values = centers[i]

            x_start = np.max([int(values[0]) - width, 0])
            y_start = np.max([int(values[1]) - width, 0])

            x_end = x_start + width * 2
            y_end = y_start + width * 2

            _cropped = _sdata[self.segmentation_channel_to_select][
                slice(x_start, x_end), slice(y_start, y_end)
            ].compute()

            # optional plotting output for deep debugging
            if self.deep_debug:
                if self.threads == 1:
                    plt.figure()
                    plt.imshow(_cropped)
                    plt.show()
                else:
                    raise ValueError("Deep debug is not supported with multiple threads.")

            sparse = coo_array(_cropped == _id)

            if (
                0 in sparse.coords[0]
                or 0 in sparse.coords[1]
                or width * 2 - 1 in sparse.coords[0]
                or width * 2 - 1 in sparse.coords[1]
            ):
                Warning(
                    f"Cell {i} with id {_id} is potentially not fully contained in the bounding mask. Consider increasing the value for the 'cell_width' parameter in your config."
                )

            x = sparse.coords[0] + x_start
            y = sparse.coords[1] + y_start

            results.append((_id, np.array(list(zip(x, y, strict=True)))))

        return results

    def _get_coords_multi(self, width: int, arg: tuple[list[int], np.ndarray]) -> list[tuple[int, np.ndarray]]:
        cell_ids, centers = arg
        results = self.__get_coords(cell_ids, centers, width)
        return results

    def _get_coords(
        self, cell_ids: list, centers: list[tuple[int, int]], width: int = 60, batch_size: int = 100, threads: int = 10
    ) -> dict[int, np.ndarray]:
        # create batches
        n_batches = int(np.ceil(len(cell_ids) / batch_size))
        slices = [(i * batch_size, i * batch_size + batch_size) for i in range(n_batches - 1)]
        slices.append(((n_batches - 1) * batch_size, len(cell_ids)))

        batched_args = [(cell_ids[start:end], centers[start:end]) for start, end in slices]

        f = func_partial(self._get_coords_multi, width)

        if (
            threads == 1
        ):  # if only one thread is used, the function is called directly to avoid the overhead of multiprocessing
            results = [f(arg) for arg in batched_args]
        else:
            with mp.get_context(self.context).Pool(processes=threads) as pool:
                results = list(
                    tqdm(
                        pool.imap(f, batched_args),
                        total=len(batched_args),
                        desc="Processing cell batches",
                    )
                )
                pool.close()
                pool.join()

        results = flatten(results)  # type: ignore
        return dict(results)  # type: ignore

    def _get_cell_ids(self, cell_sets: list[dict]) -> list[int]:
        cell_ids = []
        for cell_set in cell_sets:
            if "classes" in cell_set:
                cell_ids.extend(cell_set["classes"])
            else:
                Warning(f"Cell set {cell_set['name']} does not contain any classes.")
        return cell_ids

    def _get_centers(self, cell_ids: list[int]) -> list[tuple[int, int]]:
        _sdata = self.project.filehandler.get_sdata()
        centers = _sdata[f"{self.DEFAULT_CENTERS_NAME}_{self.segmentation_channel_to_select}"].compute()
        centers = centers.loc[cell_ids, :]
        return centers[
            ["y", "x"]
        ].values.tolist()  # needs to be returned as yx to match the coordinate system as saved in spatialdataobjects

    def _post_processing_cleanup(self, vars_to_delete: list | None = None):
        if vars_to_delete is not None:
            self._clear_cache(vars_to_delete=vars_to_delete)

        # remove temporary files
        if hasattr(self, "path_seg_mask"):
            os.remove(self.path_seg_mask)

        self._clear_cache()

    def process(
        self,
        cell_sets: list[dict],
        calibration_marker: np.array,
        name: str | None = None,
    ):
        """
        Process function for selecting cells and generating their XML.
        Under the hood this method relies on the pylmd library and utilizies its `SegmentationLoader` Class.

        Args:
            cell_sets (list of dict): List of dictionaries containing the sets of cells which should be sorted into a single well. Mandatory keys for each dictionary are: name, classes. Optional keys are: well.
            calibration_marker (numpy.array): Array of size ‘(3,2)’ containing the calibration marker coordinates in the ‘(row, column)’ format.
            name (str, optional): Name of the output file. If not provided, the name will be generated based on the names of the cell sets or if also not specified set to "selected_cells".

        Example:

            .. code-block:: python

                # Calibration marker should be defined as (row, column).
                marker_0 = np.array([-10, -10])
                marker_1 = np.array([-10, 1100])
                marker_2 = np.array([1100, 505])

                # A numpy Array of shape (3, 2) should be passed.
                calibration_marker = np.array([marker_0, marker_1, marker_2])

                # Sets of cells can be defined by providing a name and a list of classes in a dictionary.
                cells_to_select = [{"name": "dataset1", "classes": [1, 2, 3]}]

                # Alternatively, a path to a csv file can be provided.
                # If a relative path is provided, it is accessed relativ to the projects base directory.
                cells_to_select += [{"name": "dataset2", "classes": "segmentation/class_subset.csv"}]

                # If desired, wells can be passed with the individual sets.
                cells_to_select += [{"name": "dataset3", "classes": [4, 5, 6], "well": "A1"}]

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
                    rdp: 0.7

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

        start_time = timeit.default_timer()
        cell_ids = self._get_cell_ids(cell_sets)
        centers = self._get_centers(cell_ids)
        coord_index = self._get_coords(
            cell_ids=cell_ids, centers=centers, width=self.cell_radius, batch_size=self.batch_size, threads=self.threads
        )
        self.log(f"Coordinate lookup index calculation took {timeit.default_timer() - start_time} seconds.")

        sl = SegmentationLoader(
            config=self.config,
            verbose=self.debug,
            processes=self.config["processes_cell_sets"],
        )

        shape_collection = sl(None, self.cell_sets, self.calibration_marker, coords_lookup=coord_index)

        if self.debug:
            shape_collection.plot(calibration=True)
            shape_collection.stats()

        shape_collection.save(self.savepath)

        self.log(f"Saved output at {self.savepath}")

        # perform post processing cleanup
        self._post_processing_cleanup(vars_to_delete=[shape_collection, sl, coord_index])
