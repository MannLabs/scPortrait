import os
import shutil

import datatree
import xarray
from alphabase.io import tempmmap
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel
from spatialdata.transformations.transformations import Identity

from scportrait.pipeline._base import Logable
from scportrait.pipeline._utils.spatialdata_classes import spLabels2DModel
from scportrait.pipeline._utils.spatialdata_helper import (
    calculate_centroids,
    generate_region_annotation_lookuptable,
    get_chunk_size,
    get_unique_cell_ids,
    rechunk_image,
    remap_region_annotation_table,
)


class sdata_filehandler(Logable):
    def __init__(
        self,
        directory,
        sdata_path,
        input_image_name,
        nuc_seg_name,
        cyto_seg_name,
        centers_name,
        debug=False,
    ):
        super().__init__(directory=directory, debug=debug)

        self.sdata_path = sdata_path
        self.input_image_name = input_image_name
        self.nuc_seg_name = nuc_seg_name
        self.cyto_seg_name = cyto_seg_name
        self.centers_name = centers_name

    def _read_sdata(self):
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
        return self._read_sdata()

    def _force_delete_object(self, sdata, name: str, type: str):
        """
        Force delete an object from the sdata object and the corresponding directory.

        Parameters
        ----------
        name : str
            Name of the object to be deleted.
        type : str
            Type of the object to be deleted. Can be either "images", "labels", "points" or "tables".
        """
        if name in sdata:
            del sdata[name]

        # define path
        path = os.path.join(self.sdata_path, type, name)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def _check_sdata_status(self, return_sdata=False):
        _sdata = self._read_sdata()

        self.input_image_status = self.input_image_name in _sdata.images
        self.nuc_seg_status = self.nuc_seg_name in _sdata.labels
        self.cyto_seg_status = self.cyto_seg_name in _sdata.labels
        self.centers_status = self.centers_name in _sdata.points

        if return_sdata:
            return _sdata

    def _get_input_image(self, sdata):
        if self.input_image_status:
            if isinstance(sdata.images[self.input_image_name], datatree.DataTree):
                input_image = sdata.images[self.input_image_name]["scale0"].image
            elif isinstance(sdata.images[self.input_image_name], xarray.DataArray):
                input_image = sdata.images[self.input_image_name].image
        else:
            raise ValueError("Input image not found in sdata object.")

        return input_image

    ### Write new objects to sdata ###

    def _write_segmentation_object_sdata(
        self,
        segmentation_object,
        segmentation_label: str,
        classes: set = None,
        overwrite: bool = False,
    ):
        _sdata = self._read_sdata()

        # ensure that the segmentation object is converted to the scPortrait Labels2DModel
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
        segmentation,
        segmentation_label: str,
        classes: set = None,
        chunks: tuple[int, int] = (1000, 1000),
        overwrite: bool = False,
    ):
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

    def _write_points_object_sdata(self, points, points_name: str, overwrite: bool = False):
        _sdata = self._read_sdata()

        if overwrite:
            self._force_delete_object(_sdata, points_name, "points")

        _sdata.points[points_name] = points
        _sdata.write_element(points_name, overwrite=True)

        self.log(f"Points {points_name} written to sdata object.")

    def _write_table_object_sdata(self, table, table_name: str, overwrite: bool = False):
        # reconnect to sdata object
        _sdata = self._read_sdata()

        if overwrite:
            self._force_delete_object(_sdata, table_name, "tables")

        _sdata.tables[table_name] = table
        _sdata.write_element(table_name, overwrite=False)

        self.log(f"Table {table_name} written to sdata object.")

    ### Perform operations on sdata object ###
    def _get_centers(self, sdata, segmentation_label: str) -> PointsModel:
        if segmentation_label not in sdata.labels:
            raise ValueError(f"Segmentation {segmentation_label} not found in sdata object.")

        centers = calculate_centroids(sdata.labels[segmentation_label])

        return centers

    def _add_centers(self, segmentation_label: str, overwrite=False) -> None:
        _sdata = self._read_sdata()
        centroids_object = self._get_centers(_sdata, segmentation_label)
        self._write_points_object_sdata(centroids_object, self.centers_name, overwrite=overwrite)

    def _load_input_image_to_memmap(self, tmp_dir_abs_path: str, image=None):
        """
        Helper function to load the input image from sdata to memory mapped temp arrays for faster access.
        Loading happens in a chunked manner to avoid memory issues.

        The function will return the path to the memory mapped array.

        Parameters
        ----------
        tmp_dir_abs_path : str
            Absolute path to the directory where the memory mapped arrays should be stored.

        Returns
        -------
        str
            Path to the memory mapped array. Can be reconneted to using the `mmap_array_from_path`
            function from the alphabase.io.tempmmap module.
        """
        # ensure all elements are loaded
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

        # create the empty mmap array
        input_image_mmap = tempmmap.mmap_array_from_path(path_input_image)

        # load the data into the mmap array in chunks
        Z = None
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
        tmp_dir_abs_path: str,
    ):
        """
        Helper function to load segmentation masks from sdata to memory mapped temp arrays for faster access.
        Loading happens in a chunked manner to avoid memory issues.

        The function will return the path to the memory mapped array.

        Parameters
        ----------
        seg_name : list[str]
            List of segmentation element names that should be loaded found in the sdata object.
            The segmentation elments need to have the same size.
        tmp_dir_abs_path : str
            Absolute path to the directory where the memory mapped arrays should be stored.

        Returns
        -------
        str
            Path to the memory mapped array. Can be reconneted to using the `mmap_array_from_path`
            function from the alphabase.io.tempmmap module.
        """

        # ensure all elements are loaded
        _sdata = self._check_sdata_status(return_sdata=True)

        assert all(
            seg in _sdata.labels for seg in seg_name
        ), "Not all passed segmentation elements found in sdata object."

        seg_objects = [_sdata.labels[seg] for seg in seg_name]

        # get the shape of the segmentation
        shapes = [seg.shape for seg in seg_objects]

        Z, Y, X = None, None, None
        for shape in shapes:
            if len(shape) == 2:
                if Y is None:
                    Y, X = shape
                else:
                    # ensure that all seg masks have the same shape
                    assert Y == shape[0]
                    assert X == shape[1]
            elif len(shape) == 3:
                if Z is None:
                    Z, Y, X = shape
                else:
                    # ensure that all seg masks have the same shape
                    assert Z == shape[0]
                    assert Y == shape[1]
                    assert X == shape[2]

        # get the number of masks
        n_masks = len(seg_objects)

        if Z is not None:
            shape = (n_masks, Z, Y, X)
        else:
            shape = (n_masks, Y, X)

        # initialize empty memory mapped arrays to store the data
        path_seg_masks = tempmmap.create_empty_mmap(
            shape=shape,
            dtype=seg_objects[0].data.dtype,
            tmp_dir_abs_path=tmp_dir_abs_path,
        )

        # create the empty mmap array
        seg_masks = tempmmap.mmap_array_from_path(path_seg_masks)

        # load the data into the mmap array in chunks
        for i, seg in enumerate(seg_objects):
            if Z is not None:
                for z in range(Z):
                    seg_masks[i][z] = seg.data[z].compute()
            else:
                seg_masks[i] = seg.data.compute()

        # cleanup the cache
        del seg_masks, seg_objects, seg

        return path_seg_masks
