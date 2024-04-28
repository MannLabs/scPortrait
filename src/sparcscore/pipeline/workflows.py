from sparcscore.pipeline.segmentation import (
    Segmentation,
    ShardedSegmentation,
    TimecourseSegmentation,
    MultithreadedSegmentation,
)
from sparcscore.processing.preprocessing import percentile_normalization, downsample_img
from sparcscore.processing.utils import plot_image, visualize_class
from sparcscore.processing.segmentation import (
    segment_local_threshold,
    segment_global_threshold,
    numba_mask_centroid,
    contact_filter,
    size_filter,
    _class_size,
    global_otsu,
)

import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
import matplotlib.pyplot as plt
import skfmm
import time
from collections import defaultdict

from functools import partial
from multiprocessing import Pool
import multiprocessing

from skimage.filters import median
from skimage.morphology import binary_erosion, disk, dilation, erosion
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.transform import resize_local_mean

# for cellpose segmentation
from cellpose import models
from alphabase.io import tempmmap

class BaseSegmentation(Segmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalization(self, input_image):
        self.log("Starting with normalized map")
        if type(self.config["lower_quantile_normalization"]) == float:
            self.log("Normalizing each channel to the same range")
            self.maps["normalized"] = percentile_normalization(
                input_image,
                self.config["lower_quantile_normalization"],
                self.config["upper_quantile_normalization"],
            )
        else:
            if len(self.config["lower_quantile_normalization"])!= input_image.shape[0]:
                sys.exit("please specify normalization range for each input channel")
            self.log("Normalizing each channel individually.")
            normalized = []
            for i in range(input_image.shape[0]):
                lower = self.config["lower_quantile_normalization"][i]
                upper = self.config["upper_quantile_normalization"][i]
                normalized.append(percentile_normalization(
                input_image[i],
                lower,
                upper,
            ))
            self.maps["normalized"] = np.array(normalized)
        self.save_map("normalized")
        self.log("Normalized map created")

    def _median_calculation(self):
        self.log("Started with median map")
        self.maps["median"] = np.copy(self.maps["normalized"])

        for i, channel in enumerate(self.maps["median"]):
            self.maps["median"][i] = median(
                channel, disk(self.config["median_filter_size"])
            )

        self.save_map("median")
        self.log("Median map created")

    def _nucleus_thresholding(self):
        self.log("Generating thresholded nucleus map.")

        nucleus_map_tr = percentile_normalization(
            self.maps["median"][0],
            self.config["nucleus_segmentation"]["lower_quantile_normalization"],
            self.config["nucleus_segmentation"]["upper_quantile_normalization"],
        )

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if (
            "threshold" in self.config["nucleus_segmentation"]
            and "median_block" in self.config["nucleus_segmentation"]
        ):
            self.maps["nucleus_segmentation"] = segment_local_threshold(
                nucleus_map_tr,
                dilation=self.config["nucleus_segmentation"]["dilation"],
                thr=self.config["nucleus_segmentation"]["threshold"],
                median_block=self.config["nucleus_segmentation"]["median_block"],
                min_distance=self.config["nucleus_segmentation"]["min_distance"],
                peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"],
                speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"],
                median_step=self.config["nucleus_segmentation"]["median_step"],
                debug=self.debug,
            )
        else:
            self.log(
                "No threshold or median_block for nucleus segmentation defined, global otsu will be used."
            )
            self.maps["nucleus_segmentation"] = segment_global_threshold(
                nucleus_map_tr,
                dilation=self.config["nucleus_segmentation"]["dilation"],
                min_distance=self.config["nucleus_segmentation"]["min_distance"],
                peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"],
                speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"],
                debug=self.debug,
            )

        del nucleus_map_tr
        self.save_map("nucleus_segmentation")
        self.log("Thresholded nucleus map created.")

    def _nucleus_mask_generation(self):
        self.log("Started with nucleus mask map")
        self.maps["nucleus_mask"] = np.clip(self.maps["nucleus_segmentation"], 0, 1)
        self.save_map("nucleus_mask")
        self.log(
            "Nucleus mask map created with {} elements".format(
                np.max(self.maps["nucleus_segmentation"])
            )
        )

    def _filter_nuclei_classes(self):
        # filter nuclei based on size and contact
        center_nuclei, length = _class_size(
            self.maps["nucleus_segmentation"], debug=self.debug
        )
        all_classes = np.unique(self.maps["nucleus_segmentation"])

        # ids of all nuclei  s which are unconnected and can be used for further analysis
        labels_nuclei_unconnected = contact_filter(
            self.maps["nucleus_segmentation"],
            threshold=self.config["nucleus_segmentation"]["contact_filter"],
            reindex=False,
        )
        classes_nuclei_unconnected = np.unique(labels_nuclei_unconnected)

        self.log(
            "Filtered out due to contact limit: {} ".format(
                len(all_classes) - len(classes_nuclei_unconnected)
            )
        )

        labels_nuclei_filtered = size_filter(
            self.maps["nucleus_segmentation"],
            limits=[
                self.config["nucleus_segmentation"]["min_size"],
                self.config["nucleus_segmentation"]["max_size"],
            ],
        )

        classes_nuclei_filtered = np.unique(labels_nuclei_filtered)

        self.log(
            "Filtered out due to size limit: {} ".format(
                len(all_classes) - len(classes_nuclei_filtered)
            )
        )

        filtered_classes = set(classes_nuclei_unconnected).intersection(
            set(classes_nuclei_filtered)
        )
        self.log("Filtered out: {} ".format(len(all_classes) - len(filtered_classes)))

        if self.debug:
            self._plot_nucleus_size_distribution(length)
            self._visualize_nucleus_segmentation(
                classes_nuclei_unconnected, classes_nuclei_filtered
            )

        return (all_classes, filtered_classes, center_nuclei)

    def _cellmembrane_mask_calculation(self):
        self.log("Started with WGA mask map")

        if "wga_background_image" in self.config["wga_segmentation"]:
            if self.config["wga_segmentation"]["wga_background_image"]:
                # Perform percentile normalization
                wga_mask_comp = percentile_normalization(
                    self.maps["median"][-1],
                    self.config["wga_segmentation"]["lower_quantile_normalization"],
                    self.config["wga_segmentation"]["upper_quantile_normalization"],
                )
            else:
                # Perform percentile normalization
                wga_mask_comp = percentile_normalization(
                    self.maps["median"][1],
                    self.config["wga_segmentation"]["lower_quantile_normalization"],
                    self.config["wga_segmentation"]["upper_quantile_normalization"],
                )
        else:
            # Perform percentile normalization
            wga_mask_comp = percentile_normalization(
                self.maps["median"][1],
                self.config["wga_segmentation"]["lower_quantile_normalization"],
                self.config["wga_segmentation"]["upper_quantile_normalization"],
            )

        # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
        # If not, use global otsu
        if "threshold" in self.config["wga_segmentation"]:
            wga_mask = wga_mask_comp < self.config["wga_segmentation"]["threshold"]
        else:
            self.log(
                "No treshold for cytosol segmentation defined, global otsu will be used."
            )
            wga_mask = wga_mask_comp < global_otsu(wga_mask_comp)

        wga_mask = wga_mask.astype(float)
        wga_mask -= self.maps["nucleus_mask"]
        wga_mask = np.clip(wga_mask, 0, 1)

        # Apply dilation and erosion
        wga_mask = dilation(
            wga_mask, footprint=disk(self.config["wga_segmentation"]["erosion"])
        )
        self.maps["wga_mask"] = binary_erosion(
            wga_mask, footprint=disk(self.config["wga_segmentation"]["dilation"])
        )

        self.save_map("wga_mask")
        self.log("WGA mask map created")

    def _cellmembrane_potential_mask(self):
        self.log("Started with WGA potential map")

        wga_mask_comp = self.maps["median"][1] - np.quantile(
            self.maps["median"][1], 0.02
        )

        nn = np.quantile(self.maps["median"][1], 0.98)
        wga_mask_comp = wga_mask_comp / nn
        wga_mask_comp = np.clip(wga_mask_comp, 0, 1)

        # subtract golgi and dapi channel from wga
        diff = np.clip(wga_mask_comp - self.maps["median"][0], 0, 1)
        diff = np.clip(diff - self.maps["nucleus_mask"], 0, 1)
        diff = 1 - diff

        # enhance WGA map to generate speedmap
        # WGA 0.7-0.9
        min_clip = self.config["wga_segmentation"]["min_clip"]
        max_clip = self.config["wga_segmentation"]["max_clip"]
        diff = (np.clip(diff, min_clip, max_clip) - min_clip) / (max_clip - min_clip)

        diff = diff * 0.9 + 0.1
        diff = diff.astype(dtype=float)

        self.maps["wga_potential"] = diff

        self.save_map("wga_potential")
        self.log("WGA mask potential created")

    def _cellmembrane_fastmarching(self, center_nuclei):
        self.log("Started with fast marching")
        fmm_marker = np.ones_like(self.maps["median"][0])
        px_center = np.round(center_nuclei).astype(np.uint64)
        for center in px_center[1:]:
            fmm_marker[center[0], center[1]] = 0

        fmm_marker = np.ma.MaskedArray(fmm_marker, self.maps["wga_mask"])
        travel_time = skfmm.travel_time(fmm_marker, self.maps["wga_potential"])

        if not isinstance(travel_time, np.ma.core.MaskedArray):
            raise TypeError(
                "Travel_time for WGA based segmentation returned no MaskedArray. This is most likely due to missing WGA background determination."
            )

        self.maps["travel_time"] = travel_time.filled(fill_value=np.max(travel_time))

        self.save_map("travel_time")
        self.log("Fast marching finished")

    def _cellmembrane_watershed(self, center_nuclei):
        self.log("Started with watershed")

        marker = np.zeros_like(self.maps["median"][1])
        px_center = np.round(center_nuclei).astype(np.uint64)
        for i, center in enumerate(px_center[1:]):
            marker[center[0], center[1]] = i + 1
        wga_labels = watershed(
            self.maps["travel_time"], marker.astype(np.int64), mask=(self.maps["wga_mask"] == 0
        ).astype(np.int64))
        self.maps["watershed"] = np.where(self.maps["wga_mask"] > 0.5, 0, wga_labels)

        if self.debug:
            self._visualize_watershed_results(center_nuclei)

        self.save_map("watershed")
        self.log("watershed finished")

    def _filter_cells_cytosol_size(self, all_classes, filtered_classes):
        # filter cells based on cytosol size
        center_cell, length, coords = numba_mask_centroid(
            self.maps["watershed"], debug=self.debug
        )

        all_classes_wga = np.unique(self.maps["watershed"])

        labels_wga_filtered = size_filter(
            self.maps["watershed"],
            limits=[
                self.config["wga_segmentation"]["min_size"],
                self.config["wga_segmentation"]["max_size"],
            ],
        )

        classes_wga_filtered = np.unique(labels_wga_filtered)

        self.log(
            "Cells filtered out due to cytosol size limit: {} ".format(
                len(all_classes_wga) - len(classes_wga_filtered)
            )
        )

        filtered_classes_wga = set(classes_wga_filtered)
        filtered_classes = set(filtered_classes).intersection(filtered_classes_wga)
        self.log("Filtered out: {} ".format(len(all_classes) - len(filtered_classes)))
        self.log("Remaining: {} ".format(len(filtered_classes)))

        if self.debug:
            self._plot_cytosol_size_distribution(length)
            self._visualize_cytosol_filtering(classes_wga_filtered)

        return filtered_classes

    # functions to generate quality control plots
    def _dapi_median_intensity_plot(self):
        # generate plot of dapi median intensity
        plt.hist(self.maps["median"][0].flatten(), bins=100, log=False)
        plt.xlabel("intensity")
        plt.ylabel("frequency")
        plt.yscale("log")
        plt.title("DAPI intensity distribution")
        plt.savefig("dapi_intensity_dist.png")
        plt.show()

    def _cellmembrane_median_intensity_plot(self):
        # generate plot of median Cellmembrane Marker intensity
        plt.hist(self.maps["median"][1].flatten(), bins=100, log=False)
        plt.xlabel("intensity")
        plt.ylabel("frequency")
        plt.yscale("log")
        plt.title("WGA intensity distribution")
        plt.savefig("wga_intensity_dist.png")
        plt.show()

    def _visualize_nucleus_segmentation(
        self, classes_nuclei_unconnected, classes_nuclei_filtered
    ):
        um_p_px = 665 / 1024  # what is this!!?? @GWallmann
        um_2_px = um_p_px * um_p_px  # what is this!!?? @GWallmann

        visualize_class(
            classes_nuclei_unconnected,
            self.maps["nucleus_segmentation"],
            self.maps["normalized"][0],
        )
        visualize_class(
            classes_nuclei_filtered,
            self.maps["nucleus_segmentation"],
            self.maps["normalized"][0],
        )

    def _plot_nucleus_size_distribution(self, length):
        plt.hist(length, bins=50)
        plt.xlabel("px area")
        plt.ylabel("number")
        plt.title("Nucleus size distribution")
        plt.savefig("nucleus_size_dist.png")
        plt.show()

    def _plot_cytosol_size_distribution(self, length):
        plt.hist(length, bins=50)
        plt.xlabel("px area")
        plt.ylabel("number")
        plt.title("Cytosol size distribution")
        plt.savefig("cytosol_size_dist.png")
        plt.show()

    def _visualize_watershed_results(self, center_nuclei):
        image = label2rgb(
            self.maps["watershed"], self.maps["normalized"][0], bg_label=0, alpha=0.2
        )

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10, 10)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        plt.scatter(center_nuclei[:, 1], center_nuclei[:, 0], color="red")
        plt.savefig(os.path.join(self.directory, "watershed.png"))
        plt.show()

    def _visualize_cytosol_filtering(self, classes_wga_filtered):
        um_p_px = 665 / 1024  # what is this!!?? @GWallmann
        um_2_px = um_p_px * um_p_px  # what is this!!?? @GWallmann

        visualize_class(
            classes_wga_filtered, self.maps["watershed"], self.maps["normalized"][1]
        )
        visualize_class(
            classes_wga_filtered, self.maps["watershed"], self.maps["normalized"][0]
        )
    
    def _read_cellpose_model(self, modeltype, name, use_GPU, device):
        if modeltype == "pretrained":
            model = models.Cellpose(model_type=name, gpu=use_GPU, device = device)
        elif modeltype == "custom":
            model = models.CellposeModel(pretrained_model = name, gpu=use_GPU, device = device)
        return model
    
    def return_empty_mask(self, input_image):
        n_channels, x, y = input_image.shape
        self.save_segmentation(input_image, np.zeros((2, x, y)), [])

class WGASegmentation(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are the nucelus channel and a membrane marker channel like WGA
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if "wga_background_image" in self.config["wga_segmentation"]:
            if self.config["wga_segmentation"]["wga_background_image"]:
                # remove last channel since this is a pseudo channel to perform the WGA background calculation on
                feature_maps = [element for element in self.maps["normalized"][2:-1]]
            else:
                feature_maps = [element for element in self.maps["normalized"][2:]]
        else:
            feature_maps = [element for element in self.maps["normalized"][2:]]

        channels = np.stack(required_maps + feature_maps).astype(np.float64)

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["watershed"]]
        ).astype(np.uint64)

        return channels, segmentation

    def process(self, input_image):
        
        self.maps = {
            "normalized": None,
            "median": None,
            "nucleus_segmentation": None,
            "nucleus_mask": None,
            "wga_mask": None,
            "wga_potential": None,
            "travel_time": None,
            "watershed": None,
        }

        start_from = self.load_maps_from_disk()

        if self.identifier is not None:
            self.log(
                f"Segmentation started shard {self.identifier}, starting from checkpoint {start_from}"
            )

        else:
            self.log(f"Segmentation started, starting from checkpoint {start_from}")

        # Normalization
        if start_from <= 0:
            self._normalization(input_image)

        # Median calculation
        if start_from <= 1:
            self._median_calculation()

            if self.debug:
                self._dapi_median_intensity_plot()
                self._cellmembrane_median_intensity_plot()

        # segment dapi channels based on local tresholding
        if start_from <= 2:
            self.log("Started performing nucleus segmentation.")
            self._nucleus_thresholding()

        # Calc nucleus map
        if start_from <= 3:
            self._nucleus_mask_generation()

        all_classes, filtered_classes, center_nuclei = self._filter_nuclei_classes()

        # create background map based on WGA
        if start_from <= 4:
            self._cellmembrane_mask_calculation()

        # create WGA potential map
        if start_from <= 5:
            self._cellmembrane_potential_mask()

        # WGA cytosol segmentation by fast marching

        if start_from <= 6:
            self._cellmembrane_fastmarching(center_nuclei)

        if start_from <= 7:
            self._cellmembrane_watershed(center_nuclei)

        filtered_classes = self._filter_cells_cytosol_size(
            all_classes, filtered_classes
        )
        channels, segmentation = self._finalize_segmentation_results()

        results = self.save_segmentation(channels, segmentation, filtered_classes)

        # self.save_segmentation_zarr(channels, segmentation) #currently save both since we have not fully converted.
        return results


class ShardedWGASegmentation(ShardedSegmentation):
    method = WGASegmentation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DAPISegmentation(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        feature_maps = [element for element in self.maps["normalized"][1:]]

        channels = np.stack(required_maps + feature_maps).astype(np.float64)

        segmentation = np.stack([self.maps["nucleus_segmentation"]]).astype(np.uint64)
        return (channels, segmentation)

    def process(self, input_image):

        self.maps = {
            "normalized": None,
            "median": None,
            "nucleus_segmentation": None,
            "nucleus_mask": None,
            "travel_time": None,
        }

        start_from = self.load_maps_from_disk()

        if self.identifier is not None:
            self.log(
                f"Segmentation started shard {self.identifier}, starting from checkpoint {start_from}"
            )

        else:
            self.log(f"Segmentation started, starting from checkpoint {start_from}")

        # Normalization
        if start_from <= 0:
            self._normalization(input_image)

        # Median calculation
        if start_from <= 1:
            self._median_calculation()

            if self.debug:
                self._dapi_median_intensity_plot()

        # segment dapi channels based on local thresholding
        if start_from <= 2:
            self.log("Started performing nucleus segmentation.")
            self._nucleus_thresholding()

        # Calc nucleus map
        if start_from <= 3:
            self._nucleus_mask_generation()

        _, filtered_classes, _ = self._filter_nuclei_classes()
        channels, segmentation = self._finalize_segmentation_results()

        self.save_segmentation(channels, segmentation, filtered_classes)
        # self.save_segmentation_zarr(channels, segmentation) #currently save both since we have not fully converted.


class ShardedDAPISegmentation(ShardedSegmentation):
    method = DAPISegmentation

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)


class DAPISegmentationCellpose(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 1:
            feature_maps = [element for element in self.maps["normalized"][1:]]

            channels = np.stack(required_maps + feature_maps).astype(np.float64)
        else:
            channels = np.stack(required_maps).astype(np.float64)

        segmentation = np.stack([self.maps["nucleus_segmentation"]]).astype("uint64")
        return (channels, segmentation)

    def cellpose_segmentation(self, input_image):

        try:
            current = multiprocessing.current_process()
            cpu_name = current.name
            gpu_id_list = current.gpu_id_list
            cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
            gpu_id = gpu_id_list[cpu_id]
            self.log(f'starting process on GPU {gpu_id}')
            status = "multi_GPU"
        except:
            gpu_id = 0
            self.log(f'running on default GPU.')
            status = "single_GPU"
        
        gc.collect()
        torch.cuda.empty_cache() 
        
         # run this every once in a while to clean up cache and remove old variables

        # check that image is int
        input_image = input_image.astype("int64")

        # check if GPU is available
        if torch.cuda.is_available():
            if status == "multi_GPU":
                use_GPU = f"cuda:{gpu_id}"
                device = torch.device(use_GPU)
            else:
                use_GPU = True
                device = torch.device("cuda")
        #add M1 mac support
        elif torch.backends.mps.is_available():
            use_GPU = True
            device = torch.device("mps")
            self.log(f"Using MPS backend for segmentation.")
        else:
            use_GPU = False
            device = torch.device("cpu")

        self.log(f"GPU Status for segmentation: {use_GPU}")

        # load correct segmentation model
        model = models.Cellpose(model_type="nuclei", gpu=use_GPU)
        masks = model.eval([input_image], diameter=None, channels=[1, 0])[0]
        masks = np.array(masks)  # convert to array

        self.log(f"Segmented mask shape: {masks.shape}")
        self.maps["nucleus_segmentation"] = masks.reshape(
            masks.shape[1:]
        )  # need to add reshape so that hopefully saving works out

        #manually delete model and perform gc to free up memory on GPU
        del model
        gc.collect()
        torch.cuda.empty_cache()  

    def process(self, input_image):
        # initialize location to save masks to
        self.maps = {"normalized": None, "nucleus_segmentation": None}

        # could add a normalization step here if so desired
        self.maps["normalized"] = input_image

        self.log("Starting Cellpose DAPI Segmentation.")

        self.cellpose_segmentation(input_image)

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = np.unique(self.maps["nucleus_segmentation"])

        channels, segmentation = self._finalize_segmentation_results()

        results = self.save_segmentation(channels, segmentation, all_classes)
        return results


class ShardedDAPISegmentationCellpose(ShardedSegmentation):
    method = DAPISegmentationCellpose


class CytosolSegmentationCellpose(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(np.uint16)
        else:
            channels = np.stack(required_maps).astype(np.uint16)

        segmentation = np.stack(
            [self.maps["nucleus_segmentation"], self.maps["cytosol_segmentation"]]
        ).astype(np.uint32)
        
        return channels, segmentation

    def cellpose_segmentation(self, input_image):

        try:
            current = multiprocessing.current_process()
            cpu_name = current.name
            gpu_id_list = current.gpu_id_list
            cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
            lookup_id = cpu_id % len(gpu_id_list)
            gpu_id = gpu_id_list[lookup_id]
            if self.debug: 
                self.log(f"current process: {current}")
                self.log(f"cpu name: {cpu_name}")
                self.log(f"gpu id list: {gpu_id_list}")
                self.log(f"cpu id: {cpu_id}")
                self.log(f"gpu id: {gpu_id}")
            self.log(f'starting process on GPU {gpu_id}')
            status = "multi_GPU"
        
        except:
            gpu_id = 0
            self.log(f'running on default GPU.')
            status = "single_GPU"
            
        # clean up old cached variables to free up GPU memory
        gc.collect()
        torch.cuda.empty_cache()  

        # check that image is int
        input_image = input_image.astype(np.uint16)

        # check if GPU is available
        if torch.cuda.is_available():
            if status == "multi_GPU":
                use_GPU = f"cuda:{gpu_id}"
                device = torch.device(use_GPU)
            else:
                use_GPU = True
                device = torch.device("cuda")
        
        #add M1 mac support
        elif torch.backends.mps.is_available():
            use_GPU = True
            device = torch.device("mps")
            self.log(f"Using MPS backend for segmentation.")
        else:
            use_GPU = False
            device = torch.device("cpu")
        
        self.log(f"GPU Status for segmentation: {use_GPU}")

        #check to see if the cells should be filtered within the segmentation run
        if "filter_status" in self.config.keys():
            filter_status = self.config["filter_status"]
        else:
            filter_status = True

        # load correct segmentation model for nuclei
        if "model" in self.config["nucleus_segmentation"].keys():
            model_name = self.config["nucleus_segmentation"]["model"]
            model = self._read_cellpose_model("pretrained", model_name, use_GPU, device = device)
        elif "model_path" in self.config["nucleus_segmentation"].keys():
            model_name = self.config["nucleus_segmentation"]["model_path"]
            model = self._read_cellpose_model("custom", model_name, use_GPU, device = device)

        if "diameter" in self.config["nucleus_segmentation"].keys():
            diameter = self.config["nucleus_segmentation"]["diameter"]
        else:
            diameter = None

        ################################
        ### Perform Nucleus Segmentation
        ################################
            
        self.log(f"Segmenting nuclei using the following model: {model_name}")
        
        masks_nucleus = model.eval(
            [input_image], diameter=diameter, channels=[1, 0]
        )[0]
        masks_nucleus = np.array(masks_nucleus)  # convert to array

        #manually delete model and perform gc to free up memory on GPU
        del model
        gc.collect()
        torch.cuda.empty_cache()  

        # load correct segmentation model for cytosol
        if "model" in self.config["cytosol_segmentation"].keys():
            model_name = self.config["cytosol_segmentation"]["model"]
            model = self._read_cellpose_model("pretrained", model_name, use_GPU, device = device)
        elif "model_path" in self.config["cytosol_segmentation"].keys():
            model_name = self.config["cytosol_segmentation"]["model_path"]
            model = self._read_cellpose_model("custom", model_name, use_GPU, device = device)

        if "diameter" in self.config["cytosol_segmentation"].keys():
            diameter = self.config["cytosol_segmentation"]["diameter"]
        else:
            diameter = None

        self.log(f"Segmenting cytosol using the following model: {model_name}")
        
        #################################
        #### Perform Cytosol Segmentation
        #################################

        masks_cytosol = model.eval(
            [input_image], diameter=diameter, channels=[2, 1]
        )[0]
        masks_cytosol = np.array(masks_cytosol)  # convert to array

        #manually delete model and perform gc to free up memory on GPU
        del model
        gc.collect()
        torch.cuda.empty_cache() 

        if self.debug:
            # save unfiltered masks for visualization of filtering process
            masks_nucleus_unfiltered = masks_nucleus.copy()
            masks_cytosol_unfiltered = masks_cytosol.copy()

        if not filter_status:
            self.log("No filtering performed. Cytosol and Nucleus IDs in the two masks do not match. Before proceeding with extraction an additional filtering step needs to be performed")
        
        else:
            ##########################
            ### Perform Cell Filtering
            ##########################    

            #log start time of cell filtering to track
            timing_info = []
            
            start = time.time()
            timing_info.append(("start_time", "Time when started the segmentation run", start))

            all_nucleus_ids = np.unique(masks_nucleus)[1:]
            nucleus_cytosol_pairs = {}

            self.log(f"Number of nuclei to filter: {len(all_nucleus_ids)}")

            ### STEP 1: filter cells based on having a matching cytosol mask
            current_time = time.time()
            timing_info.append(("start_time", "Time when starting filtering cells (for nucleus_id in all_nucleus_ids) = STEP 1", current_time))
        
            for nucleus_id in all_nucleus_ids:
                
                ### STEP 1.1: lookup which image pixels belong to the nucleus
                time_in_the_loop = time.time()

                # get the nucleus and set the background to 0 and the nucleus to 1
                nucleus = (masks_nucleus == nucleus_id)
                
                # now get the coordinates of the nucleus
                nucleus_pixels = np.nonzero(nucleus)

                timing_info.append(("STEP 1.1", f"Time required for getting nucleus pixels in seconds for nucleus {nucleus_id}", time.time() - time_in_the_loop))

                ### Step 1.2: get the cytosol ids in the nucleus area
                time_in_the_loop = time.time()

                # check if those indices are not background in the cytosol mask
                potential_cytosol = masks_cytosol[nucleus_pixels]

                timing_info.append(("STEP 1.2", f"Time required for getting potential cytosol pixels in seconds for nucleus {nucleus_id}", time.time() - time_in_the_loop))

                if np.all(potential_cytosol != 0):
                    time_in_the_loop = time.time()

                    unique_cytosol, counts = np.unique(
                        potential_cytosol, return_counts=True
                    )
                    all_counts = np.sum(counts)
                    cytosol_proportions = counts / all_counts

                    timing_info.append(("STEP 1.3", f"Time required for getting unique cytosol pixels and calculating their proportions in seconds for nucleus {nucleus_id}", time.time() - time_in_the_loop))

                    if np.any(cytosol_proportions >= self.config["filtering_threshold"]):
                        time_in_the_loop = time.time()

                        # get the cytosol_id with max proportion
                        cytosol_id = unique_cytosol[
                            np.argmax(cytosol_proportions >= self.config["filtering_threshold"])
                        ]
                        nucleus_cytosol_pairs[nucleus_id] = cytosol_id
                    else:
                        nucleus_cytosol_pairs[nucleus_id] = 0

                    timing_info.append(("STEP 1.4", f"Time required for getting cytosol_id with max proportion in seconds for nucleus {nucleus_id}", time.time() - time_in_the_loop))
            
            timing_info.append(("STEP 1",  "Time required for filtering cells (for nucleus_id in all_nucleus_ids) in seconds", time.time() - current_time))
            
            #######################################################
            ### STEP 2: count the occurrences of each cytosol value
            #######################################################
            new_time = time.time()
            timing_info.append(("start_time", "Time when started counting the occurences of each cytosol id = STEP 2", new_time))

            # check if there are any cytosol masks that are assigned to multiple nuclei
            cytosol_count = defaultdict(int)

            # Count the occurrences of each cytosol value
            for cytosol in nucleus_cytosol_pairs.values():
                cytosol_count[cytosol] += 1
            
            timing_info.append(("STEP 2", "Time required for counting the occurences of each cytosol id in seconds", time.time() - new_time))
            
            #######################################################
            ### STEP 3: filter cytosol ids that are assigned to more than one nucleus
            #######################################################

            new_time = time.time()
            timing_info.append(("start_time", "Time when started finding cytosol ids assigned to more than one nucleus = STEP 3", new_time))

            # Find cytosol values assigned to more than one nucleus
            for nucleus, cytosol in nucleus_cytosol_pairs.items():
                if cytosol_count[cytosol] > 1:
                    nucleus_cytosol_pairs[nucleus] = 0
            
            timing_info.append(("STEP 3", "Time required for filtering cytosol ids that are assigned to more than one nucleus in seconds", time.time() - new_time))

            #######################################################
            ### STEP 4: filter cytosol masks that are not in the lookup table
            #######################################################

            new_time = time.time()
            timing_info.append(("start_time", "Time when started filtering cytosol masks that are not in the lookup table = STEP 4", new_time))

            # get unique cytosol ids that are not in the lookup table
            all_cytosol_ids = set(np.unique(masks_cytosol))
            all_cytosol_ids.discard(0)
            used_cytosol_ids = set(nucleus_cytosol_pairs.values())
            not_used_cytosol_ids = all_cytosol_ids - used_cytosol_ids

            # set all cytosol ids that are not present in lookup table to 0 in the cytosol mask
            ###speedup of 40X approximately in a small test case with an array of 10000x10000 and 400 cytosol ids to remove
            #masks_cytosol = np.where(np.isin(masks_cytosol, not_used_cytosol_ids), 0, masks_cytosol)
            for cytosol_id in not_used_cytosol_ids:
                masks_cytosol[masks_cytosol == cytosol_id] = 0
            
            timing_info.append(("STEP 4", "Time required for filtering cytosol masks that are not in the lookup table in seconds", time.time() - new_time))

            ### STEP 5: filter nucleus masks that are not in the lookup table
            new_time = time.time()
            timing_info.append(("start_time", "Time when started filtering nucleus masks that are not in the lookup table = STEP 5", new_time))

            # get unique nucleus ids that are not in the lookup table
            all_nucleus_ids = set(np.unique(masks_nucleus))
            all_nucleus_ids.discard(0)
            used_nucleus_ids = set(nucleus_cytosol_pairs.keys())
            not_used_nucleus_ids = all_nucleus_ids - used_nucleus_ids

            # set all nucleus ids that are not present in lookup table to 0 in the nucleus mask
            ###speedup of 40X approximately in a small test case with an array of 10000x10000 and 400 cytosol ids to remove
            #masks_nucleus = np.where(np.isin(masks_nucleus, not_used_nucleus_ids), 0, masks_nucleus)
            for nucleus_id in not_used_nucleus_ids:
                masks_nucleus[masks_nucleus == nucleus_id] = 0
            
            timing_info.append(("STEP 5", "Time required for filtering nucleus masks that are not in the lookup table in seconds", time.time() - new_time))

            #################################################################
            ### STEP 6: filter cytosol masks that are not in the lookup table
            #################################################################

            new_time = time.time()
            timing_info.append(("Time when started updating masks = STEP 6", new_time))

            # now we have all the nucleus cytosol pairs we can filter the masks
            updated_cytosol_mask = np.zeros_like(masks_cytosol, dtype=bool)
            for nucleus_id, cytosol_id in nucleus_cytosol_pairs.items():
                if cytosol_id == 0:
                    masks_nucleus[masks_nucleus == nucleus_id] = 0  # set the nucleus to 0
                else:
                    # set the cytosol pixels to the nucleus_id if not previously updated
                    condition = np.logical_and(
                        masks_cytosol == cytosol_id, ~updated_cytosol_mask
                    )
                    masks_cytosol[condition] = nucleus_id
                    updated_cytosol_mask = np.logical_or(updated_cytosol_mask, condition)
            
            timing_info.append(("STEP 6", "Time required for filtering cytosol masks that are not in the lookup table in seconds", time.time() - new_time))
            end = time.time()

            timing_info.append(("All STEPS", "Time required for filtering generated masks in seconds", end - start))
            self.log(f"Time required for filtering generated masks in seconds: {end - start}")

            #generate a dataframe with the time logging information and write out to file
            df_timing = pd.DataFrame(timing_info, columns=["Step", "description", "Time (s)"])
            df_timing.to_csv(f"{self.project_location}/segmentation/timing_info_{self.identifier}.csv", index=False)

            if self.debug:
                # plot nucleus and cytosol masks before and after filtering
                fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                axs[0, 0].imshow(masks_nucleus_unfiltered[0])
                axs[0, 0].axis("off")
                axs[0, 0].set_title("before filtering", fontsize=6)
                axs[0, 1].imshow(masks_nucleus[0])
                axs[0, 1].axis("off")
                axs[0, 1].set_title("after filtering", fontsize=6)

                axs[1, 0].imshow(masks_cytosol_unfiltered[0])
                axs[1, 0].axis("off")
                axs[1, 1].imshow(masks_cytosol[0])
                axs[1, 1].axis("off")
                fig.tight_layout()
                fig.show()
                del fig  # delete figure after showing to free up memory again

            #cleanup memory by deleting no longer required variables
            del updated_cytosol_mask, all_nucleus_ids, used_nucleus_ids

        # first when the masks are finalized save them to the maps
        self.maps["nucleus_segmentation"] = masks_nucleus.reshape(
            masks_nucleus.shape[1:]
        )  # need to add reshape to save in proper format for HDF5

        self.maps["cytosol_segmentation"] = masks_cytosol.reshape(
            masks_cytosol.shape[1:]
        )  # need to add reshape to save in proper format for HDF5

        #perform garbage collection to ensure memory is freedup
        del masks_nucleus, masks_cytosol
        gc.collect()
        torch.cuda.empty_cache() 

    def process(self, input_image):
        # initialize location to save masks to
        self.maps = {
            "normalized": tempmmap.array(shape = input_image.shape, dtype = float),
            "nucleus_segmentation": tempmmap.array(shape = input_image.shape, dtype = np.uint16),
            "cytosol_segmentation": tempmmap.array(shape = input_image.shape, dtype = np.uint16),
        }

        # could add a normalization step here if so desired
        self.maps["normalized"] = input_image
        
        del input_image
        gc.collect()

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(self.maps["normalized"])

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = np.unique(self.maps["nucleus_segmentation"])

        channels, segmentation = self._finalize_segmentation_results()
        results = self.save_segmentation(channels, segmentation, all_classes)
        
        #clean up memory
        del channels, segmentation, all_classes
        gc.collect()

        return results

class ShardedCytosolSegmentationCellpose(ShardedSegmentation):
    method = CytosolSegmentationCellpose

class CytosolSegmentationDownsamplingCellpose(CytosolSegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, size_padding):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(np.uint16)
        else:
            channels = np.stack(required_maps).astype(np.uint16)
        
        _seg_size = self.maps["nucleus_segmentation"].shape
        self.log(f"Segmentation size after downsampling before resize to original dimensions: {_seg_size}")
        
        _, x, y = size_padding
        segmentation_size = (x, y) #return to same size as original input image but adjust number of channels expected

        nuc_seg = self.maps["nucleus_segmentation"]
        nuc_seg = nuc_seg.repeat(self.config["downsampling_factor"], axis=0).repeat(self.config["downsampling_factor"], axis=1)
        
        cyto_seg = self.maps["cytosol_segmentation"]
        cyto_seg = cyto_seg.repeat(self.config["downsampling_factor"], axis=0).repeat(self.config["downsampling_factor"], axis=1)

        #perform erosion and dilation for smoothing
        nuc_seg = erosion(nuc_seg, footprint=disk(self.config["smoothing_kernel_size"]))
        nuc_seg  = dilation(nuc_seg, footprint=disk(self.config["smoothing_kernel_size"]))

        cyto_seg = erosion(cyto_seg, footprint=disk(self.config["smoothing_kernel_size"]))
        cyto_seg  = dilation(cyto_seg, footprint=disk(self.config["smoothing_kernel_size"]))

        #combine masks into one stack
        segmentation = np.stack([nuc_seg, cyto_seg]).astype(np.uint32)
        del cyto_seg, nuc_seg
        
        #rescale segmentation results to original size
        x_trim = x - channels.shape[1]
        y_trim = y - channels.shape[2]

        #if no padding was performed then we need to keep the same dimensions
        if x_trim > 0:
            if y_trim > 0:
                segmentation = segmentation[:, :-x_trim, :-y_trim]
            else:
                segmentation = segmentation[:, :-x_trim, :]
        else:
            if y_trim > 0:
                segmentation = segmentation[:, :, :-y_trim]
            else:
                segmentation = segmentation

        self.log(f"Segmentation size after resize to original dimensions: {segmentation.shape}")

        if segmentation.shape[1] != channels.shape[1]:
            sys.exit("Error. Segmentation mask and image have different shapes")
        if segmentation.shape[2] !=channels.shape[2]:
            sys.exit("Error. Segmentation mask and image have different shapes")

        return channels, segmentation 

    def process(self, input_image):

        _size = input_image.shape
        self.log(f"Input image size {_size} in position {self.window}")

        N = self.config["downsampling_factor"]
        self.log(f"Performing Cellpose Segmentation on Downsampled image. Downsampling input image by {N}X{N}")

        #check if N fits perfectly into image shape if not calculate how much we need to pad
        _, x, y = _size
        if x % N == 0:
            pad_x = (0, 0)
        else:
            pad_x = (0, N - x%N)
        
        if y % N == 0:
            pad_y = (0, 0)
        else:
            pad_y = (0, N - y%N)

        downsampled_image_size = (2, _size[1]+pad_x[1], _size[2]+pad_y[1]) 

        # initialize location to save masks to
        self.maps = {
            "normalized": tempmmap.array(shape = input_image.shape, dtype = float),
            "nucleus_segmentation": tempmmap.array(shape = downsampled_image_size, dtype = np.uint16),
            "cytosol_segmentation": tempmmap.array(shape = downsampled_image_size, dtype = np.uint16),
        }
        self.log("Created memory mapped temp arrays to store")

        # could add a normalization step here if so desired
        #perform downsampling after saving input image to ensure that we have a duplicate preserving the original dimensions
        self.maps["normalized"] = input_image.copy()
        _size = self.maps["normalized"].shape
        self.log(f"input image size: {input_image.shape}")

        input_image = input_image[:2, :, :] #only get the first 2 channels for segmentation (does not use excess space on the GPU this way)
        gc.collect()
        
        self.log(f"input image size after removing excess channels: {input_image.shape}")
        input_image = np.pad(input_image, ((0, 0), pad_x, pad_y))
        _size_padding = input_image.shape
        
        self.log(f"Performing image padding to ensure that image is compatible with downsample kernel size. Original image was {_size}, padded image is {_size_padding}")
        input_image = downsample_img(input_image, N= N)
        self.log(f"Downsampled image size {input_image.shape}")

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(input_image)

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = np.unique(self.maps["nucleus_segmentation"])

        channels, segmentation = self._finalize_segmentation_results(size_padding = _size_padding)
        results = self.save_segmentation(channels, segmentation, all_classes)

        return results

class ShardedCytosolSegmentationDownsamplingCellpose(ShardedSegmentation):
    method = CytosolSegmentationDownsamplingCellpose


class CytosolOnlySegmentationCellpose(BaseSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self):
        # The required maps are only nucleus channel
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(np.float64)
        else:
            channels = np.stack(required_maps).astype(np.float64)

        segmentation = np.stack(
            [self.maps["cytosol_segmentation"], self.maps["cytosol_segmentation"]]
        ).astype(np.uint64)
        return (channels, segmentation)

    def cellpose_segmentation(self, input_image):

        try:
            current = multiprocessing.current_process()
            cpu_name = current.name
            gpu_id_list = current.gpu_id_list
            cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
            gpu_id = gpu_id_list[cpu_id]
            self.log(f'starting process on GPU {gpu_id}')
            status = "multi_GPU"
        except:
            gpu_id = 0
            self.log(f'running on default GPU.')
            status = "single_GPU"
    
        gc.collect()
        torch.cuda.empty_cache()  # run this every once in a while to clean up cache and remove old variables

        # check that image is int
        input_image = input_image.astype(np.uint16)

        # check if GPU is available
        if torch.cuda.is_available():
            if status == "multi_GPU":
                use_GPU = f"cuda:{gpu_id}"
                device = torch.device(use_GPU)
            else:
                use_GPU = True
                device = torch.device("cuda")
        #add M1 mac support
        elif torch.backends.mps.is_available():
            use_GPU = True
            device = torch.device("mps")
            self.log(f"Using MPS backend for segmentation.")
        else:
            use_GPU = False
            device = torch.device("cpu")

        # currently no real acceleration through using GPU as we can't load batches
        self.log(f"GPU Status for segmentation: {use_GPU}")

        # load correct segmentation model for cytosol
        if "model" in self.config["cytosol_segmentation"].keys():
            model_name = self.config["cytosol_segmentation"]["model"]
            model = self._read_cellpose_model("pretrained", model_name, use_GPU, device = device)
        elif "model_path" in self.config["cytosol_segmentation"].keys():
            model_name = self.config["cytosol_segmentation"]["model_path"]
            model = self._read_cellpose_model("custom", model_name, use_GPU, device = device)
        
        if "model_channels" in self.config["cytosol_segmentation"].keys():
            model_channels = self.config["cytosol_segmentation"]["model_channels"]
        else:
            model_channels = [2, 1]

        if "diameter" in self.config["cytosol_segmentation"].keys():
            diameter = self.config["cytosol_segmentation"]["diameter"]
        else:
            diameter = None

        self.log(f"Segmenting cytosol using the following model: {model_name}")


        masks = model.eval(
            [input_image], diameter=diameter, channels=model_channels
            )[0]
        masks = np.array(masks)  # convert to array

        self.maps["cytosol_segmentation"] = masks.reshape(
            masks.shape[1:]
        )  # add reshape to match shape to HDF5 shape

        #manually delete model and perform gc to free up memory on GPU
        del model, masks
        gc.collect()
        torch.cuda.empty_cache()  

    def process(self, input_image):

        # initialize location to save masks to
        self.maps = {
            "normalized": tempmmap.array(shape = input_image.shape, dtype = float),
            "cytosol_segmentation": tempmmap.array(shape = input_image.shape, dtype = np.uint16),
        }

        # could add a normalization step here if so desired
        self.maps["normalized"] = input_image
        
        #delete input image to prevent overloading memory
        del input_image
        gc.collect()

        # self.log("Starting Cellpose DAPI Segmentation.")
        self.cellpose_segmentation(self.maps["normalized"])

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = np.unique(self.maps["cytosol_segmentation"])

        channels, segmentation = self._finalize_segmentation_results()
        results = self.save_segmentation(channels, segmentation, all_classes)
        
        #clean up memory
        del channels, segmentation, all_classes
        gc.collect()

        return results

class Sharded_CytosolOnly_Cellpose_Segmentation(ShardedSegmentation):
    method = CytosolOnlySegmentationCellpose

class CytosolOnly_Segmentation_Downsampling_Cellpose(CytosolOnlySegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, size_padding):
        required_maps = [self.maps["normalized"][0], self.maps["normalized"][1]]

        # Feature maps are all further channel which contain phenotypes needed for the classification
        if self.maps["normalized"].shape[0] > 2:
            feature_maps = [element for element in self.maps["normalized"][2:]]
            channels = np.stack(required_maps + feature_maps).astype(np.uint16)
        else:
            channels = np.stack(required_maps).astype(np.uint16)
        
        _seg_size = self.maps["cytosol_segmentation"].shape
        self.log(f"Segmentation size after downsampling before resize to original dimensions: {_seg_size}")
        
        _, x, y = size_padding
        segmentation_size = (x, y) #return to same size as original input image but adjust number of channels expected

        cyto_seg = self.maps["cytosol_segmentation"]
        cyto_seg = cyto_seg.repeat(self.config["downsampling_factor"], axis=0).repeat(self.config["downsampling_factor"], axis=1)

        #perform erosion and dilation for smoothing
        cyto_seg = erosion(cyto_seg, footprint=disk(self.config["smoothing_kernel_size"]))
        cyto_seg  = dilation(cyto_seg, footprint=disk(self.config["smoothing_kernel_size"]))

        #combine masks into one stack
        segmentation = np.stack([cyto_seg, cyto_seg]).astype(np.uint32)
        del cyto_seg
        
        #rescale segmentation results to original size
        x_trim = x - channels.shape[1]
        y_trim = y - channels.shape[2]

        #if no padding was performed then we need to keep the same dimensions
        if x_trim > 0:
            if y_trim > 0:
                segmentation = segmentation[:, :-x_trim, :-y_trim]
            else:
                segmentation = segmentation[:, :-x_trim, :]
        else:
            if y_trim > 0:
                segmentation = segmentation[:, :, :-y_trim]
            else:
                segmentation = segmentation

        self.log(f"Segmentation size after resize to original dimensions: {segmentation.shape}")

        if segmentation.shape[1] != channels.shape[1]:
            sys.exit("Error. Segmentation mask and image have different shapes")
        if segmentation.shape[2] !=channels.shape[2]:
            sys.exit("Error. Segmentation mask and image have different shapes")

        return channels, segmentation 

    def process(self, input_image):

        _size = input_image.shape
        self.log(f"Input image size {_size}")

        N = self.config["downsampling_factor"]
        self.log(f"Performing Cellpose Segmentation on Downsampled image. Downsampling input image by {N}X{N}")

        #check if N fits perfectly into image shape if not calculate how much we need to pad
        _, x, y = _size
        if x % N == 0:
            pad_x = (0, 0)
        else:
            pad_x = (0, N - x%N)
        
        if y % N == 0:
            pad_y = (0, 0)
        else:
            pad_y = (0, N - y%N)

        downsampled_image_size = (2, _size[1]+pad_x[1], _size[2]+pad_y[1]) 

        # initialize location to save masks to
        self.maps = {
            "normalized": tempmmap.array(shape = input_image.shape, dtype = float),
             "cytosol_segmentation": tempmmap.array(shape = downsampled_image_size, dtype = np.uint16),
        }
        self.log("Created memory mapped temp arrays to store")

        # could add a normalization step here if so desired
        #perform downsampling after saving input image to ensure that we have a duplicate preserving the original dimensions
        self.maps["normalized"] = input_image.copy()
        _size = self.maps["normalized"].shape
        self.log(f"input image size: {input_image.shape}")

        input_image = input_image[:2, :, :] #only get the first 2 channels for segmentation (does not use excess space on the GPU this way)
        gc.collect()
        
        self.log(f"input image size after removing excess channels: {input_image.shape}")
        input_image = np.pad(input_image, ((0, 0), pad_x, pad_y))
        _size_padding = input_image.shape
        
        self.log(f"Performing image padding to ensure that image is compatible with downsample kernel size. Original image was {_size}, padded image is {_size_padding}")
        input_image = downsample_img(input_image, N= N)
        self.log(f"Downsampled image size {input_image.shape}")

        self.cellpose_segmentation(input_image)

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = np.unique(self.maps["cytosol_segmentation"])

        channels, segmentation = self._finalize_segmentation_results(size_padding = _size_padding)
        results = self.save_segmentation(channels, segmentation, all_classes)

        return results

class Sharded_CytosolOnly_Segmentation_Downsampling_Cellpose(ShardedSegmentation):
    method = CytosolOnly_Segmentation_Downsampling_Cellpose


class WGA_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file.
    """

    class WGASegmentation_Timecourse(WGASegmentation, TimecourseSegmentation):
        method = WGASegmentation

    method = WGASegmentation_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Multithreaded_WGA_TimecourseSegmentation(MultithreadedSegmentation):
    class WGASegmentation_Timecourse(WGASegmentation, TimecourseSegmentation):
        method = WGASegmentation

    method = WGASegmentation_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Cytosol_Cellpose_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
    """

    class CytosolSegmentationCellpose_Timecourse(
        CytosolSegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationCellpose

    method = CytosolSegmentationCellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Cytosol_Cellpose_Downsampling_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
    """

    class Cytosol_Segmentation_Downsampling_Cellpose_Timecourse(
        CytosolSegmentationDownsamplingCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationDownsamplingCellpose

    method = Cytosol_Segmentation_Downsampling_Cellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CytosolOnly_Cellpose_TimecourseSegmentation(TimecourseSegmentation):
    """
    Specialized Processing for Timecourse segmentation (i.e. smaller tiles not stitched together from many different wells and or timepoints).
    No intermediate results are saved and everything is written to one .hdf5 file. Uses Cellpose segmentation models.
    """

    class CytosolOnly_Cellpose_TimecourseSegmentation(
        CytosolOnlySegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolOnlySegmentationCellpose

    method = CytosolOnly_Cellpose_TimecourseSegmentation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Multithreaded_Cytosol_Cellpose_TimecourseSegmentation(MultithreadedSegmentation):
    class CytosolSegmentationCellpose_Timecourse(
        CytosolSegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationCellpose

    method = CytosolSegmentationCellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Multithreaded_Cytosol_Cellpose_Downsampling_TimecourseSegmentation(MultithreadedSegmentation):
    
    class Cytosol_Segmentation_Downsampling_Cellpose_Timecourse(
        CytosolSegmentationDownsamplingCellpose, TimecourseSegmentation
    ):
        method = CytosolSegmentationDownsamplingCellpose

    method = Cytosol_Segmentation_Downsampling_Cellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class Multithreaded_CytosolOnly_Cellpose_TimecourseSegmentation(
    MultithreadedSegmentation
):
    class CytosolOnly_SegmentationCellpose_Timecourse(
        CytosolOnlySegmentationCellpose, TimecourseSegmentation
    ):
        method = CytosolOnlySegmentationCellpose

    method = CytosolOnly_SegmentationCellpose_Timecourse

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
