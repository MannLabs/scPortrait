from __future__ import annotations

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
        new_project_location: str,
        old_project_location: str,
        channel_names: list[str] | None = None,
        cache: str | None = None,
        debug: bool = False,
        overwrite: bool = True,
    ) -> None:
        """Initialize the SPARCS project to SpatialData converter.

        Args:
            new_project_location: Path where the new spatialdata project will be saved
            old_project_location: Path to the existing SPARCS project
            channel_names: List of channel names for the images
            cache: Path to cache directory
            debug: Enable debug mode
            overwrite: Whether to overwrite existing project
        """
        self.new_project_location = new_project_location
        self.directory = new_project_location
        self.old_project_location = old_project_location
        self.channel_names = channel_names
        self.cache = cache
        self.debug = debug
        self.overwrite = overwrite

        self.input_image_status = None
        self.segmentation_status = None
        self.centers_status = None

    def _check_memory(self, item) -> bool:
        """Check if item can fit in available memory.

        Args:
            item: Array-like object to check memory requirements for

        Returns:
            Whether the item can fit in available memory
        """
        array_size = item.nbytes
        available_memory = psutil.virtual_memory().available
        return array_size < available_memory

    def _create_temp_dir(self) -> None:
        """Create a temporary directory for intermediate results.

        Creates directory in cache location specified in config or current working directory.
        """
        if self.cache is not None:
            path = os.path.join(self.config["cache"], f"{self.__class__.__name__}_")
        else:
            path = os.path.join(os.getcwd(), f"{self.__class__.__name__}_")

        self._tmp_dir = tempfile.TemporaryDirectory(prefix=path)
        self._tmp_dir_path = self._tmp_dir.name

        self.log(f"Initialized temporary directory at {self._tmp_dir_path} for {self.__class__.__name__}")

    def clear_temp_dir(self) -> None:
        """Delete the temporary directory."""
        if "_tmp_dir" in self.__dict__.keys():
            shutil.rmtree(self._tmp_dir_path)
            self.log(f"Cleaned up temporary directory at {self._tmp_dir}")

            del self._tmp_dir, self._tmp_dir_path
        else:
            self.log("Temporary directory not found, skipping cleanup")

    def __del__(self) -> None:
        self.clear_temp_dir()

    def _check_output_location(self) -> None:
        """Check and prepare output location.

        Raises:
            ValueError: If output location exists and overwrite is False
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

    def _get_sdata_path(self) -> str:
        """Get path to spatialdata object.

        Returns:
            Path to the spatialdata object
        """
        return os.path.join(self.new_project_location, "sparcs.sdata")

    def _load_input_image(self) -> None:
        """Load input image to temporary memory mapped array."""
        with h5py.File(f"{self.old_project_location}/segmentation/segmentation.h5", "r") as hf:
            input_image = hf["channels"]

            self.temp_image_path = tempmmap.create_empty_mmap(
                shape=input_image.shape,
                dtype=DEFAULT_IMAGE_DTYPE,
                tmp_dir_abs_path=self._tmp_dir_path,
            )

            temp_image = tempmmap.mmap_array_from_path(self.temp_image_path)

            for i in range(input_image.shape[0]):
                for j in range(input_image.shape[1]):
                    temp_image[i, j] = input_image[i, j]

            self.log("Finished loading input image to memory mapped temp array.")

    def write_input_image_to_spatialdata(self, scale_factors: list[int] | None = None) -> None:
        """Write input image to spatialdata object.

        Args:
            scale_factors: List of scale factors for image resolution. Defaults to [2, 4, 8]
        """
        if scale_factors is None:
            scale_factors = [2, 4, 8]
        temp_image = daskmmap.dask_array_from_path(self.temp_image_path)

        if self.channel_names is None:
            self.channel_names = [f"channel_{i}" for i in range(temp_image.shape[0])]

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

        self.input_image_status = True

    def _load_segmentation(self) -> None:
        """Load segmentation mask to temporary memory mapped array."""
        with h5py.File(f"{self.old_project_location}/segmentation/segmentation.h5", "r") as hf:
            if "labels" in hf:
                segmentation = hf["labels"]

                self.temp_segmentation_path = tempmmap.create_empty_mmap(
                    shape=segmentation.shape,
                    dtype=DEFAULT_SEGMENTATION_DTYPE,
                    tmp_dir_abs_path=self._tmp_dir_path,
                )

                temp_image = tempmmap.mmap_array_from_path(self.temp_segmentation_path)

                for i in range(segmentation.shape[0]):
                    for j in range(segmentation.shape[1]):
                        temp_image[i, j] = segmentation[i, j]

                self.log("Finished loading segmentation mask to memory mapped temp array.")
            else:
                self.log("No segmentation found in project.")
                self.segmentation_status = False

    def write_segmentation_to_spatialdata(self, scale_factors: list[int] | None = None) -> None:
        """Write segmentation masks to spatialdata object.

        Args:
            scale_factors: List of scale factors for masks.
        """
        if scale_factors is None:
            scale_factors = []
        if self.segmentation_status is None:
            temp_segmentation = daskmmap.dask_array_from_path(self.temp_segmentation_path)

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

            sdata.labels["seg_all_nucleus"] = mask_0
            sdata.write_element("seg_all_nucleus", overwrite=True)

            sdata.labels["seg_all_cytosol"] = mask_1
            sdata.write_element("seg_all_cytosol", overwrite=True)

            self.segmentation_status = True
            self.nucleus_segmentation = True
            self.cytosol_segmentation = True

    def _lookup_region_annotations(self) -> dict[str, list[tuple[str, TableModel]]]:
        """Get mapping between regions and their annotations.

        Returns:
            Dictionary mapping region names to lists of (table_name, table) tuples
        """
        sdata = SpatialData.read(self._get_sdata_path())

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

    def add_multiscale_segmentation(
        self, region_keys: list[str] | None = None, scale_factors: list[int] | None = None
    ) -> None:
        """Add multiscale segmentation to spatialdata object.

        Args:
            region_keys: List of region keys to process.
                Defaults to ["seg_all_nucleus", "seg_all_cytosol"]
            scale_factors: List of scale factors.
                Defaults to [2, 4, 8]
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
                    del table.uns["spatialdata_attrs"]

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
        centers: np.ndarray,
        ids: np.ndarray,
        transformation: Identity,
        coordinate_system: str = "global",
    ) -> PointsModel:
        """Create PointsModel from cell centers and IDs.

        Args:
            centers: Array of center coordinates
            ids: Array of cell IDs
            transformation: Transformation to apply
            coordinate_system: Coordinate system name.

        Returns:
            PointsModel object containing cell centers
        """
        coordinates = pd.DataFrame(centers, columns=["y", "x"], index=ids)
        centroids = PointsModel.parse(coordinates, transformations={coordinate_system: transformation})
        centroids = transform(centroids, to_coordinate_system=coordinate_system)

        return centroids

    def write_centers_to_spatialdata(self, coordinate_system: str = "global") -> None:
        """Write cell centers to spatialdata object.

        Args:
            coordinate_system: Coordinate system name.
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

    def _read_classification_results(self, classification_result: str) -> anndata.AnnData:
        """Read classification results from file.

        Args:
            classification_result: Name of classification result folder

        Returns:
            AnnData object containing classification results

        Raises:
            ValueError: If multiple or no classification files found
        """
        classification_dir = f"{self.old_project_location}/classification/{classification_result}/"
        filename = os.listdir(classification_dir)
        filename = [x for x in filename if x.endswith(".csv")]

        if len(filename) > 1:
            raise ValueError("Multiple classification files found in the classification directory")
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

    def write_classification_result_to_spatialdata(
        self, classification_result: str, segmentation_regions: list[str] | None = None
    ) -> None:
        """Write classification results to spatialdata object.

        Args:
            classification_result: Name of classification result folder
            segmentation_regions: List of segmentation regions to annotate.
                Defaults to ["seg_all_nucleus", "seg_all_cytosol"]
        """
        if segmentation_regions is None:
            segmentation_regions = ["seg_all_nucleus", "seg_all_cytosol"]
        class_result = self._read_classification_results(classification_result)
        sdata = SpatialData.read(self._get_sdata_path())

        for segmentation_region in segmentation_regions:
            table = class_result.copy()
            table.obs["region"] = segmentation_region
            table.obs["region"] = table.obs["region"].astype("category")

            table = TableModel.parse(table, region_key="region", region=segmentation_region, instance_key="cell_id")

            sdata.tables[f"{classification_result}_{segmentation_region}"] = table
            sdata.write_element(f"{classification_result}_{segmentation_region}", overwrite=True)

        self.log(
            f"Added classification result from folder {classification_result} to spatialdata object. Annotated the following regions: {segmentation_regions}"
        )

    def process(self) -> None:
        """Process the SPARCS project and convert it to spatialdata format.

        1. Sets up the project location
        2. Creates temporary directory for cache
        3. Loads and writes input images
        4. Processes segmentation if available
        5. Calculates and writes cell centers
        6. Processes classification results if available
        7. Adds multiscale segmentation
        8. Cleans up temporary files
        """
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
