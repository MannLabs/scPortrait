import _pickle as cPickle
import os
import shutil
import tempfile

import anndata
import h5py
import numpy as np
import pandas as pd
import psutil
from alphabase.io import tempmmap
from napari_spatialdata import Interactive
from sparcstools.base import daskmmap
from spatialdata import SpatialData, get_centroids
from spatialdata._core.operations.transform import transform
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import Identity

from scportrait.pipeline._base import Logable
from scportrait.pipeline._utils.segmentation import numba_mask_centroid

DEFAULT_IMAGE_DTYPE = np.uint16
DEFAULT_SEGMENTATION_DTYPE = np.uint32


class convert_SPARCSproject_to_spatialdata(Logable):
    def __init__(
        self,
        new_project_location,
        old_project_location,
        channel_names=None,
        cache=None,
        debug=False,
        overwrite=True,
    ):
        self.new_project_location = new_project_location
        self.directory = new_project_location  # This is the directory where the new spatialdata project will be saved and where log output will be saved. This key definition required for the Logable class.
        self.old_project_location = old_project_location
        self.channel_names = channel_names
        self.cache = cache
        self.debug = debug
        self.overwrite = overwrite

        self.input_image_status = None
        self.segmentation_status = None
        self.centers_status = None

    def _check_memory(self, item):
        """
        Check the memory usage of the given if it were completely loaded into memory using .compute().
        """
        array_size = item.nbytes
        available_memory = psutil.virtual_memory().available

        return array_size < available_memory

    def _create_temp_dir(self):
        """
        Create a temporary directory in the cache directory specified in the config for saving all intermediate results.
        """

        if self.cache is not None:
            path = os.path.join(self.config["cache"], f"{self.__class__.__name__}_")
        else:
            path = os.path.join(os.getcwd(), f"{self.__class__.__name__}_")

        self._tmp_dir = tempfile.TemporaryDirectory(prefix=path)
        self._tmp_dir_path = self._tmp_dir.name

        self.log(f"Initialized temporary directory at {self._tmp_dir_path} for {self.__class__.__name__}")

    def clear_temp_dir(self):
        """Delete created temporary directory."""

        if "_tmp_dir" in self.__dict__.keys():
            shutil.rmtree(self._tmp_dir_path)
            self.log(f"Cleaned up temporary directory at {self._tmp_dir}")

            del self._tmp_dir, self._tmp_dir_path
        else:
            self.log("Temporary directory not found, skipping cleanup")

    def __del__(self):
        self.clear_temp_dir()

    def _check_output_location(self):
        """
        Check if the output location exists and if it does cleanup if allowed, othwerwise raise an error.
        """
        if os.path.exists(self.new_project_location):
            if self.overwrite:
                self.log(f"Output location {self.new_project_location} already exists. Overwriting.")
                shutil.rmtree(self.new_project_location)
            else:
                raise ValueError(
                    f"Output location {self.new_project_location} already exists. Set overwrite=True to overwrite."
                )
        else:
            os.makedirs(self.new_project_location)

    def _get_sdata_path(self):
        """
        Get the path to the spatialdata object.
        """
        return os.path.join(self.new_project_location, "sparcs.sdata")

    def _load_input_image(self):
        """
        Load the input image from the project and save it to a temporary memory mapped array.
        """
        with h5py.File(f"{self.old_project_location}/segmentation/segmentation.h5", "r") as hf:
            input_image = hf["channels"]

            self.temp_image_path = tempmmap.create_empty_mmap(
                shape=input_image.shape,
                dtype=DEFAULT_IMAGE_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            )

            temp_image = tempmmap.mmap_array_from_path(self.temp_image_path)

            # write image in a batched fashion to lower memory overhead
            for i in range(input_image.shape[0]):
                for j in range(input_image.shape[1]):
                    temp_image[i, j] = input_image[i, j]

            self.log("Finished loading input image to memory mapped temp array.")

    def write_input_image_to_spatialdata(self, scale_factors=None):
        """
        Write the input image found under the label "channels" in the segmentation.h5 file to a spatialdata object.

        Parameters
        ----------
        scale_factors : list
            List of scale factors for the image. Default is [2, 4, 8]. This will load the image at 4 different resolutions to allow for fluid visualization.
        """

        # reconnect to temporary image as a dask array
        if scale_factors is None:
            scale_factors = [2, 4, 8]
        temp_image = daskmmap.dask_array_from_path(self.temp_image_path)

        if self.channel_names is None:
            self.channel_names = [f"channel_{i}" for i in range(temp_image.shape[0])]

        # transform to spatialdata image model
        transform_original = Identity()
        image = Image2DModel.parse(
            temp_image,
            dims=["c", "y", "x"],
            c_coords=self.channel_names,
            scale_factors=scale_factors,
            transformations={"global": transform_original},
            rgb=False,
        )

        sdata = SpatialData(images={"input_image": image})
        sdata.write(self._get_sdata_path(), overwrite=True)

        # track that input image has been loaded
        self.input_image_status = True

    def _load_segmentation(self):
        """
        Load the segmentation mask from the project and save it to a temporary memory mapped array.
        """
        with h5py.File(f"{self.old_project_location}/segmentation/segmentation.h5", "r") as hf:
            if "labels" in hf:
                segmentation = hf["labels"]

                self.temp_segmentation_path = tempmmap.create_empty_mmap(
                    shape=segmentation.shape,
                    dtype=DEFAULT_SEGMENTATION_DTYPE,
                    tmp_dir_abs_path=self._tmp_dir_path,
                )

                temp_image = tempmmap.mmap_array_from_path(self.temp_segmentation_path)

                # write image in a batched fashion to lower memory overhead
                for i in range(segmentation.shape[0]):
                    for j in range(segmentation.shape[1]):
                        temp_image[i, j] = segmentation[i, j]

                self.log("Finished loading segmentation mask to memory mapped temp array.")
            else:
                self.log("No segmentation found in project.")
                self.segmentation_status = False

    def write_segmentation_to_spatialdata(self, scale_factors=None):
        """
        Write the segmentation masks found under the label "labels" in the segmentation.h5 file to a spatialdata object.

        Parameters
        ----------
        scale_factors : list
            List of scale factors for the image. Default is []. This will only load the segmention masks at full resolution into the sdata object.
            In the future this behaviour may be changed but at the moment scPortrait is not designed to handle multiple resolutions for segmentation masks.
        """

        if scale_factors is None:
            scale_factors = []
        if self.segmentation_status is None:
            # reconnect to temporary image as a dask array
            temp_segmentation = daskmmap.dask_array_from_path(self.temp_segmentation_path)

            # transform to spatialdata image model
            transform_original = Identity()
            mask_0 = Labels2DModel.parse(
                temp_segmentation[0],
                dims=["y", "x"],
                scale_factors=scale_factors,
                transformations={"global": transform_original},
            )
            mask_1 = Labels2DModel.parse(
                temp_segmentation[1],
                dims=["y", "x"],
                scale_factors=scale_factors,
                transformations={"global": transform_original},
            )

            sdata = SpatialData.read(self._get_sdata_path())

            # write nucleus mask
            sdata.labels["seg_all_nucleus"] = mask_0
            sdata.write_element("seg_all_nucleus", overwrite=True)

            # write cytosol mask
            sdata.labels["seg_all_cytosol"] = mask_1
            sdata.write_element("seg_all_cytosol", overwrite=True)

            # track that segmentation has been loaded
            self.segmentation_status = True
            self.nucleus_segmentation = True
            self.cytosol_segmentation = True

    def _lookup_region_annotations(self):
        sdata = SpatialData.read(self._get_sdata_path())

        list(sdata.tables.keys())

        table_names = list(sdata.tables.keys())
        region_lookup = {}
        for table_name in table_names:
            table = sdata.tables[table_name]
            region = table.uns["spatialdata_attrs"]["region"]

            if region not in region_lookup:
                region_lookup[region] = [(table_name, table)]
            else:
                region_lookup[region].append((table_name, table))

        return region_lookup

    def add_multiscale_segmentation(self, region_keys=None, scale_factors=None):
        """
        Add multiscale segmentation to the spatialdata object.
        """
        if scale_factors is None:
            scale_factors = [2, 4, 8]
        if region_keys is None:
            region_keys = ["seg_all_nucleus", "seg_all_cytosol"]
        region_lookup = self._lookup_region_annotations()
        sdata = SpatialData.read(self._get_sdata_path())

        for region_key in region_keys:
            mask = sdata.labels[region_key]
            sdata.labels[f"{region_key}_multiscale"] = Labels2DModel.parse(mask, scale_factors=scale_factors)
            sdata.write_element(f"{region_key}_multiscale", overwrite=True)
            self.log(f"Added multiscaled segmentation for {region_key} to spatialdata object.")

            if region_key in region_lookup:
                for x in region_lookup[region_key]:
                    table_name, table = x
                    table = table.copy()
                    table.obs["region"] = f"{region_key}_multiscale"
                    table.obs["region"] = table.obs["region"].astype("category")
                    del table.uns[
                        "spatialdata_attrs"
                    ]  # remove the spatialdata attributes so that the table can be re-written

                    table = TableModel.parse(
                        table, region_key="region", region=f"{region_key}_multiscale", instance_key="cell_id"
                    )
                    sdata.tables[f"{table_name}_multiscale"] = table
                    sdata.write_element(f"{table_name}_multiscale", overwrite=True)
                    self.log(
                        f"Added annotation {table_name} to spatialdata object for multiscaled segmentation of {region_key}."
                    )

    def _make_centers_object(
        self,
        centers,
        ids,
        transformation,
        coordinate_system="global",
    ):
        """
        Create a spatialdata PointsModel object from the provided centers and ids.
        """
        coordinates = pd.DataFrame(centers, columns=["y", "x"], index=ids)
        centroids = PointsModel.parse(coordinates, transformations={coordinate_system: transformation})
        centroids = transform(centroids, to_coordinate_system=coordinate_system)

        return centroids

    def write_centers_to_spatialdata(self, coordinate_system="global"):
        """
        Write the centers of the cells (based on their nuclear segmentation masks) to a spatialdata object.

        Parameters
        ----------
        coordinate_system : str
            Coordinate system to use for the centers. Default is "global".
        """
        centers_path = f"{self.old_project_location}/extraction/center.pickle"
        cell_ids_path = f"{self.old_project_location}/extraction/_cell_ids.pickle"

        sdata = SpatialData.read(self._get_sdata_path())
        mask = sdata.labels["seg_all_nucleus"]
        transform = get_transformation(mask, coordinate_system)

        if os.path.exists(centers_path) and os.path.exists(cell_ids_path):
            self.log("Centers already precalculated. Loading from project.")
            with open(f"{self.old_project_location}/extraction/center.pickle", "rb") as input_file:
                centers = cPickle.load(input_file)
            with open(cell_ids_path, "rb") as input_file:
                _ids = cPickle.load(input_file)
            centroids = self._make_centers_object(centers, _ids, transform, coordinate_system=coordinate_system)
        else:
            if self.segementation_status:
                self.log("No centers found in project. Recalculating based on the provided segmentation mask.")

                if self.check_memory(mask):
                    self.log("Calculating centers using numba This should be quick.")
                    centers, _, _ids = numba_mask_centroid(mask.values)
                    centroids = self._make_centers_object(centers, _ids, transform, coordinate_system=coordinate_system)
                else:
                    self.log("Array larger than available memory, using dask-delayed calculation of centers.")
                    centroids = get_centroids(mask, coordinate_system)

        sdata.points["centers_cells"] = centroids
        sdata.write_element("centers_cells", overwrite=True)
        self.centers_status = True

    def _read_classification_results(self, classification_result):
        classification_dir = f"{self.old_project_location}/classification/{classification_result}/"
        filename = os.listdir(classification_dir)
        filename = [x for x in filename if x.endswith(".csv")]

        if len(filename) > 1:
            raise ValueError("Multiple classification files found in the classification directory.")
        elif len(filename) == 0:
            raise ValueError("No classification files found in the classification directory.")

        classification_file = f"{classification_dir}/{filename[0]}"

        # read classification results
        classification_results = pd.read_csv(classification_file, index_col=0)
        feature_matrix = classification_results.to_numpy()
        var_names = classification_results.columns

        obs = pd.DataFrame()
        obs["cell_id"] = classification_results.cell_id

        # map into an anndata object
        table = anndata.AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        return table

    def write_classification_result_to_spatialdata(self, classification_result, segmentation_regions=None):
        if segmentation_regions is None:
            segmentation_regions = ["seg_all_nucleus", "seg_all_cytosol"]
        class_result = self._read_classification_results(classification_result)
        sdata = SpatialData.read(self._get_sdata_path())

        for segmentation_region in segmentation_regions:
            table = class_result.copy()  # need to copy so that we can modify the object without changing the original
            table.obs["region"] = segmentation_region
            table.obs["region"] = table.obs["region"].astype("category")

            table = TableModel.parse(table, region_key="region", region=segmentation_region, instance_key="cell_id")

            sdata.tables[f"{classification_result}_{segmentation_region}"] = table
            sdata.write_element(f"{classification_result}_{segmentation_region}", overwrite=True)

        self.log(
            f"Added classification result from folder {classification_result} to spatialdata object. Annotated the following regions: {segmentation_regions}"
        )

    def process(self):
        # setup new project location correctly
        self._check_output_location()
        self.log(
            f"Transferring SPARCS project from {self.old_project_location} to spatialdata object at {self.new_project_location}."
        )

        # setup cache for tempmmap arrays
        self._create_temp_dir()

        # load input image and write to sdata object
        self._load_input_image()
        self.write_input_image_to_spatialdata()

        # write segmentation if it exists
        self._load_segmentation()
        self.write_segmentation_to_spatialdata()

        # write centers to sdata object
        self.write_centers_to_spatialdata()

        # write classification results to sdata object
        classification_results = os.listdir(f"{self.old_project_location}/classification/")
        classification_results = [
            x for x in classification_results if os.path.isdir(f"{self.old_project_location}/classification/{x}")
        ]

        if len(classification_results) > 0:
            for classification_result in classification_results:
                self.write_classification_result_to_spatialdata(classification_result)

        # add multiscale
        self.add_multiscale_segmentation()
        self.clear_temp_dir()
        self.log("Finished transferring SPARCS project to spatialdata object.")
