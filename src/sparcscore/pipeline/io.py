import os
import shutil

from spatialdata import SpatialData
from spatialdata.transformations.transformations import Identity
from spatialdata.models import PointsModel, Image2DModel

from sparcscore.pipeline.spatialdata_classes import spLabels2DModel
from sparcscore.utils.spatialdata_helper import (
    get_unique_cell_ids,
    generate_region_annotation_lookuptable,
    remap_region_annotation_table,
    rechunk_image,
    get_chunk_size,
    calculate_centroids,
)
from sparcscore.pipeline.base import Logable

class sdata_filehandler(Logable):
    def __init__(self, 
                 directory, 
                 sdata_path, 
                 input_image_name,
                 nuc_seg_name,
                 cyto_seg_name,
                 centers_name,
                 debug=False):
        
        super().__init__(directory=directory, 
                       debug=debug)
        
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
                    segmentation_object = spLabels2DModel().convert(
                        segmentation_object, classes=None
                    )

        return(_sdata)
    
    def get_sdata(self) -> SpatialData:
        return(self._read_sdata())
    
    def _force_delete_object(self, sdata, name:str, type:str):
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
        
        #define path 
        path = os.path.join(self.sdata_path, type, name)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def _check_sdata_status(self):
        _sdata = self._read_sdata()
        self.input_image_status = self.input_image_name in _sdata.images
        self.nuc_seg_status = self.nuc_seg_name in _sdata.labels
        self.cyto_seg_status = self.cyto_seg_name in _sdata.labels
        self.centers_status = self.centers_name in _sdata.points

    ### Write new objects to sdata ###
    
    def _write_segmentation_object_sdata(
        self, segmentation_object, segmentation_label: str, classes: set = None, overwrite = False
    ):
        _sdata = self._read_sdata()

        # ensure that the segmentation object is converted to the sparcspy Labels2DModel
        if not hasattr(segmentation_object.attrs, "cell_ids"):
            segmentation_object = spLabels2DModel().convert(
                segmentation_object, classes=classes
            )

        if overwrite:
            self._force_delete_object(_sdata, segmentation_label, "labels")

        _sdata.labels[segmentation_label] = segmentation_object
        _sdata.write_element(segmentation_label, overwrite=True)

        self.log(f"Segmentation {segmentation_label} written to sdata object.")

        self._check_sdata_status()
        print(self.nuc_seg_status, self.cyto_seg_status)

    def _write_segmentation_sdata(
        self,
        segmentation,
        segmentation_label: str,
        classes: set = None,
        chunks=(1000, 1000),
        overwrite = False
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

        self._write_segmentation_object_sdata(mask, segmentation_label, classes=classes, overwrite = overwrite)

    def _write_points_object_sdata(self, points, points_name: str, overwrite):
        
        _sdata = self._read_sdata()
        
        if overwrite:
            self._force_delete_object(_sdata, points_name, "points")

        _sdata.points[points_name] = points
        _sdata.write_element(points_name, overwrite=True)

        self.log(f"Points {points_name} written to sdata object.")

    ### Perform operations on sdata object ###
    def _get_centers(self, sdata, segmentation_label: str) -> PointsModel:
        if segmentation_label not in sdata.labels:
            raise ValueError(
                f"Segmentation {segmentation_label} not found in sdata object."
            )

        centers = calculate_centroids(sdata.labels[segmentation_label])

        return centers

    def _add_centers(self, segmentation_label: str, overwrite = False) -> None:
        _sdata = self._read_sdata()
        centroids_object = self._get_centers(_sdata, segmentation_label)
        self._write_points_object_sdata(centroids_object, self.centers_name, overwrite = overwrite)
