"""SpatialData file handling utilities for scPortrait."""

import os
import shutil
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import xarray
from alphabase.io import tempmmap
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations.transformations import Identity

from scportrait.pipeline._base import Logable
from scportrait.pipeline._utils.spatialdata_helper import (
    calculate_centroids,
    get_chunk_size,
)

ChunkSize2D: TypeAlias = tuple[int, int]
ChunkSize3D: TypeAlias = tuple[int, int, int]
ObjectType: TypeAlias = Literal["images", "labels", "points", "tables"]


class sdata_filehandler(Logable):
    def __init__(
        self,
        directory: str,
        sdata_path: str,
        input_image_name: str,
        nuc_seg_name: str,
        cyto_seg_name: str,
        centers_name: str,
        debug: bool = False,
    ) -> None:
        """Initialize the SpatialData file handler.

        Args:
            directory: Base directory for operations
            sdata_path: Path to SpatialData file
            input_image_name: Name of input image in SpatialData
            nuc_seg_name: Name of nuclear segmentation
            cyto_seg_name: Name of cytoplasm segmentation
            centers_name: Name for cell centers
            debug: Enable debug mode
        """
        super().__init__(directory=directory, debug=debug)

        self.sdata_path = sdata_path
        self.input_image_name = input_image_name
        self.nuc_seg_name = nuc_seg_name
        self.cyto_seg_name = cyto_seg_name
        self.centers_name = centers_name

    def _create_empty_sdata(self) -> SpatialData:
        """Create an empty SpatialData object.

        Returns:
            SpatialData object without any data
        """
        _sdata = SpatialData()
        _sdata.attrs["sdata_status"] = {
            "input_images": False,
            "nucleus_segmentation": False,
            "cytosol_segmentation": False,
            "centers": False,
        }
        return _sdata

    def _read_sdata(self) -> SpatialData:
        """Read or create SpatialData object.

        Returns:
            SpatialData object
        """
        if os.path.exists(self.sdata_path):
            if len(os.listdir(self.sdata_path)) == 0:
                shutil.rmtree(self.sdata_path, ignore_errors=True)
                _sdata = self._create_empty_sdata()
                _sdata.write(self.sdata_path, overwrite=True)
            else:
                _sdata = SpatialData.read(self.sdata_path)

        else:
            _sdata = self._create_empty_sdata()
            _sdata.write(self.sdata_path, overwrite=True)

        return _sdata

    def get_sdata(self) -> SpatialData:
        """Get the SpatialData object.

        Returns:
            SpatialData object
        """
        return self._read_sdata()

    def _force_delete_object(self, sdata: SpatialData, name: str, type: ObjectType) -> None:
        """Force delete an object from the SpatialData object and directory.

        Args:
            sdata: SpatialData object
            name: Name of object to delete
            type: Type of object ("images", "labels", "points", "tables")
        """
        if name in sdata:
            del sdata[name]

        path = os.path.join(self.sdata_path, type, name)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def _check_sdata_status(self, return_sdata: bool = False) -> SpatialData | None:
        """Check status of SpatialData objects.

        Args:
            return_sdata: Whether to return the SpatialData object

        Returns:
            SpatialData object if return_sdata is True, otherwise None
        """
        _sdata = self._read_sdata()

        self.input_image_status = self.input_image_name in _sdata.images
        self.nuc_seg_status = self.nuc_seg_name in _sdata.labels
        self.cyto_seg_status = self.cyto_seg_name in _sdata.labels
        self.centers_status = self.centers_name in _sdata.points

        _sdata.attrs["sdata_status"] = {
            "input_images": self.input_image_status,
            "nucleus_segmentation": self.nuc_seg_status,
            "cytosol_segmentation": self.cyto_seg_status,
            "centers": self.centers_status,
        }

        _sdata.write_metadata()  # ensure the metadata is updated on file

        if return_sdata:
            return _sdata
        return None

    def _get_input_image(self, sdata: SpatialData) -> xarray.DataArray:
        """Get input image from SpatialData object.

        Args:
            sdata: SpatialData object

        Returns:
            Input image as xarray DataArray

        Raises:
            ValueError: If input image not found
        """
        assert sdata.attrs["sdata_status"]["input_images"], "Input image not found in sdata object."
        if isinstance(sdata.images[self.input_image_name], xarray.DataTree):
            input_image = sdata.images[self.input_image_name]["scale0"].image
        elif isinstance(sdata.images[self.input_image_name], xarray.DataArray):
            input_image = sdata.images[self.input_image_name].image

        return input_image

    ## write elements to sdata object
    def _write_image_sdata(
        self,
        image,
        image_name: str,
        channel_names: list[str] = None,
        scale_factors: list[int] = None,
        chunks: ChunkSize3D = (1, 1000, 1000),
        overwrite=False,
    ):
        """
        Write the supplied image to the spatialdata object.

        Args:
            image (dask.array): Image to be written to the spatialdata object.
            image_name (str): Name of the image to be written to the spatialdata object.
            channel_names list[str]: List of channel names for the image. Default is None.
            scale_factors list[int]: List of scale factors for the image. Default is [2, 4, 8]. This will load the image at 4 different resolutions to allow for fluid visualization.
            chunks (tuple): Chunk size for the image. Default is (1, 1000, 1000).
            overwrite (bool): Whether to overwrite existing data. Default is False.
        """
        _sdata = self._read_sdata()

        # check if the image is already a multi-scale image
        if isinstance(image, xarray.DataTree):
            # if so only validate the model since this means we are getting the image from a spatialdata object already
            # image = Image2DModel.validate(image)
            # this appraoch is currently not functional but an issue was opened at https://github.com/scverse/spatialdata/issues/865
            if scale_factors is not None:
                Warning("Scale factors are ignored when passing a multi-scale image.")
            image = image.scale0.image

        if scale_factors is None:
            scale_factors = [2, 4, 8]
        if scale_factors is None:
            scale_factors = [2, 4, 8]

        if isinstance(image, xarray.DataArray):
            # if so first validate the model since this means we are getting the image from a spatialdata object already
            # then apply the scales transform
            # image = Image2DModel.validate(image)
            # this appraoch is currently not functional but an issue was opened at https://github.com/scverse/spatialdata/issues/865

            if channel_names is not None:
                Warning(
                    "Channel names are ignored when passing a single scale image in the DataArray format. Channel names are read directly from the DataArray."
                )

            image = Image2DModel.parse(
                image,
                scale_factors=scale_factors,
                rgb=False,
            )

        else:
            if channel_names is None:
                channel_names = [f"channel_{i}" for i in range(image.shape[0])]

            # transform to spatialdata image model
            transform_original = Identity()
            image = Image2DModel.parse(
                image,
                dims=["c", "y", "x"],
                chunks=chunks,
                c_coords=channel_names,
                scale_factors=scale_factors,
                transformations={"global": transform_original},
                rgb=False,
            )

        if overwrite:
            self._force_delete_object(_sdata, image_name, "images")

        _sdata.images[image_name] = image
        _sdata.write_element(image_name, overwrite=True)

        self.log(f"Image {image_name} written to sdata object.")
        self._check_sdata_status()

    def _write_segmentation_object_sdata(
        self,
        segmentation_object: Labels2DModel,
        segmentation_label: str,
        classes: set[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Write segmentation object to SpatialData.

        Args:
            segmentation_object: Segmentation object to write
            segmentation_label: Label for the segmentation
            classes: Set of class names
            overwrite: Whether to overwrite existing data
        """
        _sdata = self._read_sdata()

        if overwrite:
            self._force_delete_object(_sdata, segmentation_label, "labels")

        _sdata.labels[segmentation_label] = segmentation_object
        _sdata.write_element(segmentation_label, overwrite=True)

        self.log(f"Segmentation {segmentation_label} written to sdata object.")
        self._check_sdata_status()

    def _write_segmentation_sdata(
        self,
        segmentation: xarray.DataArray | np.ndarray,
        segmentation_label: str,
        classes: set[str] | None = None,
        chunks: ChunkSize2D = (1000, 1000),
        overwrite: bool = False,
    ) -> None:
        """Write segmentation data to SpatialData.

        Args:
            segmentation: Segmentation data to write
            segmentation_label: Label for the segmentation
            classes: Set of class names
            chunks: Chunk size for data storage
            overwrite: Whether to overwrite existing data
        """
        transform_original = Identity()
        mask = Labels2DModel.parse(
            segmentation,
            dims=["y", "x"],
            transformations={"global": transform_original},
            chunks=chunks,
        )

        if not get_chunk_size(mask) == chunks:
            mask.data = mask.data.rechunk(chunks)

        self._write_segmentation_object_sdata(mask, segmentation_label, classes=classes, overwrite=overwrite)

    def _write_points_object_sdata(self, points: PointsModel, points_name: str, overwrite: bool = False) -> None:
        """Write points object to SpatialData.

        Args:
            points: Points object to write
            points_name: Name for the points object
            overwrite: Whether to overwrite existing data
        """
        _sdata = self._read_sdata()

        if overwrite:
            self._force_delete_object(_sdata, points_name, "points")

        _sdata.points[points_name] = points
        _sdata.write_element(points_name, overwrite=True)

        self.log(f"Points {points_name} written to sdata object.")

    def _write_table_object_sdata(self, table: TableModel, table_name: str, overwrite: bool = False) -> None:
        """Write table object to SpatialData.

        Args:
            table: Table object to write
            table_name: Name for the table object
            overwrite: Whether to overwrite existing data
        """
        _sdata = self._read_sdata()

        if overwrite:
            self._force_delete_object(_sdata, table_name, "tables")

        _sdata.tables[table_name] = table
        _sdata.write_element(table_name, overwrite=True)

        self.log(f"Table {table_name} written to sdata object.")

    def _get_centers(self, sdata: SpatialData, segmentation_label: str) -> PointsModel:
        """Get cell centers from segmentation.

        Args:
            sdata: SpatialData object
            segmentation_label: Label of segmentation to use

        Returns:
            Points model containing cell centers

        Raises:
            ValueError: If segmentation not found
        """
        if segmentation_label not in sdata.labels:
            raise ValueError(f"Segmentation {segmentation_label} not found in sdata object.")

        mask = sdata.labels[segmentation_label]
        if isinstance(mask, xarray.DataTree):
            mask = mask.scale0.image
        centers = calculate_centroids(mask)
        return centers

    def _add_centers(self, segmentation_label: str, overwrite: bool = False) -> None:
        """Add cell centers from segmentation.

        Args:
            segmentation_label: Label of segmentation to use
            overwrite: Whether to overwrite existing centers
        """
        _sdata = self._read_sdata()
        centroids_object = self._get_centers(_sdata, segmentation_label)
        centers_name = f"{self.centers_name}_{segmentation_label}"
        self._write_points_object_sdata(centroids_object, centers_name, overwrite=overwrite)

    ## load elements from sdata to a memory mapped array
    def _load_input_image_to_memmap(
        self, tmp_dir_abs_path: str | Path, image: np.typing.NDArray[Any] | None = None
    ) -> str:
        """Helper function to load the input image from sdata to memory mapped temp arrays for faster access.

        Loading happens in a chunked manner to avoid memory issues.

        Args:
            tmp_dir_abs_path: Absolute path to the directory where the memory mapped arrays should be stored.
            image: Optional pre-loaded image array to process.

        Returns:
            Path to the memory mapped array. Can be reconneted to using the `mmap_array_from_path`
            function from the alphabase.io.tempmmap module.

        Raises:
            ValueError: If input image is not found in sdata object.
        """
        if image is None:
            _sdata = self._check_sdata_status(return_sdata=True)

            if not self.input_image_status:
                raise ValueError("Input image not found in sdata object.")

            image = self._get_input_image(_sdata)
        shape = image.shape

        # initialize empty memory mapped arrays to store the data
        path_input_image = tempmmap.create_empty_mmap(
            shape=shape,
            dtype=image.dtype,
            tmp_dir_abs_path=tmp_dir_abs_path,
        )

        input_image_mmap = tempmmap.mmap_array_from_path(path_input_image)

        Z: int | None = None
        if len(shape) == 3:
            C, Y, X = shape

        elif len(shape) == 4:
            Z, C, Y, X = shape

        if Z is not None:
            for z in range(Z):
                for c in range(C):
                    input_image_mmap[z][c] = image[z][c].compute()
        else:
            for c in range(C):
                input_image_mmap[c] = image[c].compute()

        # cleanup the cache
        del input_image_mmap, image

        return path_input_image

    def _load_seg_to_memmap(
        self,
        seg_name: list[str],
        tmp_dir_abs_path: str | Path,
    ) -> str:
        """Helper function to load segmentation masks from sdata to memory mapped temp arrays for faster access.

        Loading happens in a chunked manner to avoid memory issues.

        Args:
            seg_name: List of segmentation element names that should be loaded found in the sdata object.
                The segmentation elments need to have the same size.
            tmp_dir_abs_path: Absolute path to the directory where the memory mapped arrays should be stored.

        Returns:
            Path to the memory mapped array. Can be reconneted to using the `mmap_array_from_path`
            function from the alphabase.io.tempmmap module.

        Raises:
            AssertionError: If not all segmentation elements are found in sdata object or if shapes don't match.
        """
        _sdata = self._check_sdata_status(return_sdata=True)

        assert all(
            seg in _sdata.labels for seg in seg_name
        ), "Not all passed segmentation elements found in sdata object."

        seg_objects = [_sdata.labels[seg] for seg in seg_name]

        shapes = [seg.shape for seg in seg_objects]

        Z: int | None = None
        Y: int | None = None
        X: int | None = None
        for shape in shapes:
            if len(shape) == 2:
                if Y is None:
                    Y, X = shape
                else:
                    assert Y == shape[0]
                    assert X == shape[1]
            elif len(shape) == 3:
                if Z is None:
                    Z, Y, X = shape
                else:
                    assert Z == shape[0]
                    assert Y == shape[1]
                    assert X == shape[2]

        n_masks = len(seg_objects)

        if Z is not None and Y is not None and X is not None:
            shape = (n_masks, Z, Y, X)
        elif Y is not None and X is not None:
            shape = (n_masks, Y, X)
        else:
            raise ValueError("Unable to determine shape from segmentation masks")

        # initialize empty memory mapped arrays to store the data
        path_seg_masks = tempmmap.create_empty_mmap(
            shape=shape,
            dtype=seg_objects[0].data.dtype,
            tmp_dir_abs_path=tmp_dir_abs_path,
        )

        seg_masks = tempmmap.mmap_array_from_path(path_seg_masks)

        for i, seg in enumerate(seg_objects):
            if Z is not None:
                for z in range(Z):
                    seg_masks[i][z] = seg.data[z].compute()
            else:
                seg_masks[i] = seg.data.compute()

        # cleanup the cache
        del seg_masks, seg_objects, seg

        return path_seg_masks
