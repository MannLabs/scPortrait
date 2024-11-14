"""SpatialData file handling utilities for scPortrait."""

import os
import shutil
from typing import Literal, TypeAlias

import datatree
import xarray
from alphabase.io import tempmmap
from spatialdata import SpatialData
from spatialdata.models import PointsModel
from spatialdata.transformations.transformations import Identity

from scportrait.pipeline._base import Logable
from scportrait.pipeline._utils.spatialdata_classes import spLabels2DModel
from scportrait.pipeline._utils.spatialdata_helper import (
    calculate_centroids,
    get_chunk_size,
)

# Type aliases
ChunkSize: TypeAlias = tuple[int, int]
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

    def _read_sdata(self) -> SpatialData:
        """Read or create SpatialData object.

        Returns:
            SpatialData object
        """
        if os.path.exists(self.sdata_path):
            if len(os.listdir(self.sdata_path)) == 0:
                shutil.rmtree(self.sdata_path, ignore_errors=True)
                _sdata = SpatialData()
                _sdata.write(self.sdata_path, overwrite=True)
            else:
                _sdata = SpatialData.read(self.sdata_path)
        else:
            _sdata = SpatialData()
            _sdata.write(self.sdata_path, overwrite=True)

        for key in _sdata.labels:
            segmentation_object = _sdata.labels[key]
            if not hasattr(segmentation_object.attrs, "cell_ids"):
                segmentation_object = spLabels2DModel().convert(segmentation_object, classes=None)

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
        if self.input_image_status:
            if isinstance(sdata.images[self.input_image_name], datatree.DataTree):
                input_image = sdata.images[self.input_image_name]["scale0"].image
            elif isinstance(sdata.images[self.input_image_name], xarray.DataArray):
                input_image = sdata.images[self.input_image_name].image
        else:
            raise ValueError("Input image not found in sdata object.")

        return input_image

    def _write_segmentation_object_sdata(
        self,
        segmentation_object: spLabels2DModel,
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

        if not hasattr(segmentation_object.attrs, "cell_ids"):
            segmentation_object = spLabels2DModel().convert(segmentation_object, classes=classes)

        if overwrite:
            self._force_delete_object(_sdata, segmentation_label, "labels")

        _sdata.labels[segmentation_label] = segmentation_object
        _sdata.write_element(segmentation_label, overwrite=True)

        self.log(f"Segmentation {segmentation_label} written to sdata object.")
        self._check_sdata_status()

    def _write_segmentation_sdata(
        self,
        segmentation: xarray.DataArray,
        segmentation_label: str,
        classes: set[str] | None = None,
        chunks: ChunkSize = (1000, 1000),
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
        mask = spLabels2DModel.parse(
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

        centers = calculate_centroids(sdata.labels[segmentation_label])
        return centers

    def _add_centers(self, segmentation_label: str, overwrite: bool = False) -> None:
        """Add cell centers from segmentation.

        Args:
            segmentation_label: Label of segmentation to use
            overwrite: Whether to overwrite existing centers
        """
        _sdata = self._read_sdata()
        centroids_object = self._get_centers(_sdata, segmentation_label)
        self._write_points_object_sdata(centroids_object, self.centers_name, overwrite=overwrite)

    def _load_input_image_to_memmap(self, tmp_dir_abs_path: str, image: xarray.DataArray | None = None) -> str:
        """Load input image to memory mapped array.

        Args:
            tmp_dir_abs_path: Path for temporary storage
            image: Optional image data to load

        Returns:
            Path to memory mapped array

        Raises:
            ValueError: If input image not found
        """
        if image is None:
            _sdata = self._check_sdata_status(return_sdata=True)
            if not self.input_image_status:
                raise ValueError("Input image not found in sdata object.")
            image = self._get_input_image(_sdata)

        shape = image.shape
        path_input_image = tempmmap.create_empty_mmap(
            shape=shape,
            dtype=image.dtype,
            tmp_dir_abs_path=tmp_dir_abs_path,
        )

        input_image_mmap = tempmmap.mmap_array_from_path(path_input_image)

        if len(shape) == 3:
            C, Y, X = shape
            for c in range(C):
                input_image_mmap[c] = image[c].compute()
        elif len(shape) == 4:
            Z, C, Y, X = shape
            for z in range(Z):
                for c in range(C):
                    input_image_mmap[z][c] = image[z][c].compute()

        del input_image_mmap, image
        return path_input_image
