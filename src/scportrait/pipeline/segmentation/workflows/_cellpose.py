import multiprocessing
import os
import timeit
from pathlib import PosixPath

import numpy as np
import torch
from cellpose import models
from skfmm import travel_time as skfmm_travel_time
from skimage.morphology import dilation, disk
from skimage.segmentation import watershed

from scportrait.pipeline._utils.segmentation import (
    numba_mask_centroid,
    remove_edge_labels,
)
from scportrait.pipeline.segmentation.segmentation import (
    ShardedSegmentation,
)
from scportrait.pipeline.segmentation.workflows._base_segmentation_workflow import _BaseSegmentation


class _CellposeSegmentation(_BaseSegmentation):
    def _write_cellpose_seg_params_to_file(self, model_type: str, model_name: str) -> None:
        """
        Writes the cellpose segmentation parameters to a file for debugging/logging purposes
        """
        with open(os.path.join(self.directory, f"cellpose_params_{model_type}.txt"), "w") as f:
            f.write("Cellpose Parameters:\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Diameter: {self.diameter}\n")
            f.write(f"Resample: {self.resample}\n")
            f.write(f"Flow Threshold: {self.flow_threshold}\n")
            f.write(f"Cellprob Threshold: {self.cellprob_threshold}\n")
            f.write(f"Normalize: {self.normalize}\n")
            f.write(f"Rescale: {self.rescale}\n")

    def _read_cellpose_model(self, modeltype: str, name: str, gpu: str, device) -> models.Cellpose:
        """
        Reads cellpose model based on the modeltype and name. Will load to GPU if available as specified in self._use_gpu

        Parameters
        ----------
        modeltype
            either "pretrained" or "custom" depending on the model to load
        name
            name of the model to load

        Returns
        -------
        cellpose model

        """
        if modeltype == "pretrained":
            model = models.Cellpose(model_type=name, gpu=gpu, device=device)
        elif modeltype == "custom":
            model = models.CellposeModel(pretrained_model=name, gpu=gpu, device=device)
        return model

    def _load_model(self, model_type: str, gpu: str, device) -> models.Cellpose:
        """
        Loads cellpose model

        Parameters
        ----------
        model_type
            either "cytosol" or "nucleus" depending on the model to load

        Returns
        -------
        tuple of expected diameter and the cellpose model
        """

        # load correct segmentation model for cytosol
        if "model" in self.config[f"{model_type}_segmentation"].keys():
            model_name = self.config[f"{model_type}_segmentation"]["model"]
            model = self._read_cellpose_model("pretrained", model_name, gpu=gpu, device=device)

        elif "model_path" in self.config[f"{model_type}_segmentation"].keys():
            model_name = self.config[f"{model_type}_segmentation"]["model_path"]
            if isinstance(model_name, PosixPath):
                model_name = str(model_name)
            model = self._read_cellpose_model("custom", model_name, gpu=gpu, device=device)

        # get model parameters from config if not defined use default values
        if "diameter" in self.config[f"{model_type}_segmentation"].keys():
            self.diameter = self.config[f"{model_type}_segmentation"]["diameter"]
        else:
            self.diameter = None

        if "resample" in self.config[f"{model_type}_segmentation"].keys():
            self.resample = self.config[f"{model_type}_segmentation"]["resample"]
        else:
            self.resample = True

        if "flow_threshold" in self.config[f"{model_type}_segmentation"].keys():
            self.flow_threshold = self.config[f"{model_type}_segmentation"]["flow_threshold"]
        else:
            self.flow_threshold = 0.4

        if "cellprob_threshold" in self.config[f"{model_type}_segmentation"].keys():
            self.cellprob_threshold = self.config[f"{model_type}_segmentation"]["cellprob_threshold"]
        else:
            self.cellprob_threshold = 0.0

        if "normalize" in self.config[f"{model_type}_segmentation"].keys():
            self.normalize = self.config[f"{model_type}_segmentation"]["normalize"]
        else:
            self.normalize = True

        if "rescale" in self.config[f"{model_type}_segmentation"].keys():
            self.rescale = self.config[f"{model_type}_segmentation"]["rescale"]
        else:
            self.rescale = None

        self.log(f"Segmenting {model_type} using the following model: {model_name}")

        self._write_cellpose_seg_params_to_file(model_type=model_type, model_name=model_name)

        return model

    def _check_input_image_dtype(self, input_image: np.ndarray):
        if input_image.dtype != self.DEFAULT_IMAGE_DTYPE:
            if isinstance(input_image.dtype, int):
                ValueError(
                    "Default image dtype is no longer int. Cellpose expects int inputs. Please contact developers."
                )
            else:
                ValueError("Image is not of type uint16, cellpose segmentation expects int input images.")

    def _check_gpu_status(self):
        """
        Checks and updates the GPU status.
        If a multi-GPU setup is used, the function checks the current process and returns the GPU id to use for the segmentation.
        If no GPUs are available, the function defaults to CPU.
        """

        # get GPU information if run with workers
        try:
            current = multiprocessing.current_process()
            cpu_name = current.name
            gpu_id_list = current.gpu_id_list
            cpu_id = int(cpu_name[cpu_name.find("-") + 1 :]) - 1

            if cpu_id >= len(gpu_id_list):
                cpu_id = cpu_id % current.n_processes

            # track gpu_id and update GPU status
            self.gpu_id = gpu_id_list[cpu_id]
            self.status = "multi_GPU"

        except (AttributeError, ValueError):
            # default to single GPU
            self.gpu_id = 0
            self.status = "potentially_single_GPU"

        # check if cuda GPU is available
        if torch.cuda.is_available():
            if self.status == "multi_GPU":
                self.use_GPU = f"cuda:{self.gpu_id}"
                self.device = torch.device(self.use_GPU)
                self.nGPUs = torch.cuda.device_count()
            else:
                self.use_GPU = True
                self.device = torch.device(
                    "cuda"
                )  # dont need to specify id, saying cuda will default to the one thats avaialable
                self.nGPUs = 1

        # check if MPS is available
        elif torch.backends.mps.is_available():
            self.use_GPU = True
            self.device = torch.device("mps")
            self.nGPUs = 1

        # default to CPU
        else:
            self.use_GPU = False
            self.device = torch.device("cpu")
            self.nGPUs = 0

        self.log(
            f"GPU Status for segmentation is {self.use_GPU} and will segment using the following device {self.device}."
        )


class DAPISegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 1
    N_INPUT_CHANNELS = 1
    MASK_NAMES = ["nucleus"]
    DEFAULT_NUCLEI_CHANNEL_IDS = [0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=self.MASK_NAMES)

    def _finalize_segmentation_results(self, mask: np.ndarray) -> np.ndarray:
        # ensure correct dtype of the maps

        mask = self._check_seg_dtype(mask=mask, mask_name=self.MASK_NAMES[0])

        segmentation = np.stack([mask])

        return segmentation

    def cellpose_segmentation(self, input_image: np.ndarray) -> np.ndarray:
        self._check_gpu_status()
        self._clear_cache()  # ensure we start with an empty cache

        ################################
        ### Perform Nucleus Segmentation
        ################################

        model = self._load_model(model_type="nucleus", gpu=self.use_GPU, device=self.device)

        if self.normalize is False:
            input_image = (input_image - np.min(input_image)) / (
                np.max(input_image) - np.min(input_image)
            )  # min max normalize to 0-1 range as cellpose expects this

        masks = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[1, 0],
        )[0]
        masks = np.array(masks)

        # ensure all edge classes are removed
        masks = remove_edge_labels(masks)

        # check if filtering is required
        self._setup_filtering()

        if self.filter_size:
            masks = self._perform_size_filtering(
                mask=masks,
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="nucleus",
                log=True,
                input_image=input_image if self.debug else None,
            )

        masks = masks.reshape(masks.shape[1:])
        return masks

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()

        # check that the correct level of input image is used
        input_image = self._transform_input_image(input_image)

        self._check_input_image_dtype(input_image)

        start_segmentation = timeit.default_timer()
        nucleus_mask = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # finalize classes list
        all_classes = set(np.unique(nucleus_mask)) - {0}

        segmentation = self._finalize_segmentation_results(nucleus_mask=nucleus_mask)
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self.total_time = timeit.default_timer() - total_time_start


class ShardedDAPISegmentationCellpose(ShardedSegmentation):
    method = DAPISegmentationCellpose


class NuclearExpansionSegmentationCellpose(DAPISegmentationCellpose):
    N_MASKS = 1
    N_INPUT_CHANNELS = 1
    MASK_NAMES = ["nucleus"]
    DEFAULT_NUCLEUS_CHANNEL_IDS = [0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._get_kernel_config()

    def _get_kernel_config(self):
        if "kernel_size" in self.config.keys():
            self.kernel_size = self.config["kernel_size"]
        else:
            self.kernel_size = 20  # default value

        print(self.kernel_size)

    def _expand_nucleus_mask(self, nucleus_mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Expands the nucleus mask by a given kernel size using dilation

        Parameters
        ----------
        nucleus_mask
            nucleus mask to expand
        kernel_size
            size of the kernel to use for dilation

        Returns
        -------
        expanded nucleus mask
        """
        # get centers of segmented nuclei
        nucleus_centers, _, _ids = numba_mask_centroid(nucleus_mask)
        px_centers = np.round(nucleus_centers).astype(np.uint64)

        # convert to binary mask
        binary_mask = (nucleus_mask > 0).astype(np.uint16)

        # expand the mask using dilation with the given kernel size
        expanded_masks = dilation(binary_mask, disk(kernel_size))

        # initialize marker array with cell_ids
        marker = np.zeros_like(nucleus_mask)
        for center in px_centers:
            marker[center[0], center[1]] = nucleus_mask[center[0], center[1]]

        # perform fast marching to get travel times for watershed
        fmm_marker = np.ones_like(expanded_masks)
        for center in px_centers:
            fmm_marker[center[0], center[1]] = 0

        travel_time = skfmm_travel_time(fmm_marker, expanded_masks)

        # use watershed to get expanded cytosol masks
        cytosol_segmentation = watershed(
            travel_time,
            marker.astype(np.int64),
            mask=(expanded_masks).astype(np.int64),
        )

        # remove edge labels
        cytosol_segmentation = remove_edge_labels(cytosol_segmentation)

        return cytosol_segmentation

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()

        # check that the correct level of input image is used
        input_image = self._transform_input_image(input_image)

        self._check_input_image_dtype(input_image)

        start_segmentation = timeit.default_timer()
        nucleus_mask = self.cellpose_segmentation(input_image)
        cytosol_mask = self._expand_nucleus_mask(nucleus_mask, kernel_size=self.kernel_size)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # finalize classes list
        all_classes = set(np.unique(cytosol_mask)) - {0}

        segmentation = self._finalize_segmentation_results(mask=cytosol_mask)
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self.total_time = timeit.default_timer() - total_time_start


class ShardedNuclearExpansionSegmentationCellpose(ShardedSegmentation):
    method = NuclearExpansionSegmentationCellpose


class CytosolSegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 2
    N_INPUT_CHANNELS = 2
    MASK_NAMES = ["nucleus", "cytosol"]
    DEFAULT_NUCLEI_CHANNEL_IDS = [0]
    DEFAULT_CYTOSOL_CHANNEL_IDS = [1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, mask_nucleus: np.ndarray, mask_cytosol: np.ndarray) -> np.ndarray:
        # ensure correct dtype of maps

        mask_nucleus = self._check_seg_dtype(mask=mask_nucleus, mask_name="nucleus")
        mask_cytosol = self._check_seg_dtype(mask=mask_cytosol, mask_name="cytosol")

        segmentation = np.stack([mask_nucleus, mask_cytosol])
        return segmentation

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=self.MASK_NAMES)
        self._check_for_mask_matching_filtering()

    def cellpose_segmentation(self, input_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_gpu_status()
        self._clear_cache()  # ensure we start with an empty cache

        ################################
        ### Perform Nucleus Segmentation
        ################################

        model = self._load_model(model_type="nucleus", gpu=self.use_GPU, device=self.device)

        if self.normalize is False:
            input_image = (input_image - np.min(input_image)) / (
                np.max(input_image) - np.min(input_image)
            )  # min max normalize to 0-1 range as cellpose expects this

        masks_nucleus = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[1, 0],
        )[0]
        masks_nucleus = np.array(masks_nucleus)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model])

        # remove edge labels from masks
        masks_nucleus = remove_edge_labels(masks_nucleus)

        #################################
        #### Perform Cytosol Segmentation
        #################################

        model = self._load_model(model_type="cytosol", gpu=self.use_GPU, device=self.device)

        masks_cytosol = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[2, 1],
        )[0]
        masks_cytosol = np.array(masks_cytosol)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model])

        # remove edge labels from masks
        masks_cytosol = remove_edge_labels(masks_cytosol)

        ######################
        ### Perform Filtering to remove too small/too large masks if applicable
        ######################

        # check if filtering is required
        self._setup_filtering()

        if self.filter_size:
            masks_nucleus = self._perform_size_filtering(
                mask=masks_nucleus,
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="nucleus",
                log=True,
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

            masks_cytosol = self._perform_size_filtering(
                mask=masks_cytosol,
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="cytosol",
                log=True,
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

        ######################
        ### Perform Filtering match cytosol and nucleus IDs if applicable
        ######################
        self._setup_filtering()

        if self.filter_match_masks:
            masks_nucleus, masks_cytosol = self._perform_mask_matching_filtering(
                nucleus_mask=masks_nucleus,
                cytosol_mask=masks_cytosol,
                input_image=input_image if self.debug else None,
                filtering_threshold=self.mask_matching_filtering_threshold,
                debug=self.debug,
            )
        else:
            self.log(
                "No filtering performed. Cytosol and Nucleus IDs in the two masks do not match. Before proceeding with extraction an additional filtering step needs to be performed"
            )

        ######################
        ### Cleanup Generated Segmentation masks
        ######################

        masks_nucleus = masks_nucleus.reshape(masks_nucleus.shape[1:])
        masks_cytosol = masks_cytosol.reshape(masks_cytosol.shape[1:])

        return (masks_nucleus, masks_cytosol)

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()

        # ensure the correct level is selected for the input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        start_segmentation = timeit.default_timer()
        masks_nucleus, masks_cytosol = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # finalize segmentation classes ensuring that background is removed
        all_classes = set(np.unique(masks_nucleus)) - {0}

        segmentation = self._finalize_segmentation_results(mask_nucleus=masks_nucleus, mask_cytosol=masks_cytosol)
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)

        # clean up memory
        self._clear_cache(vars_to_delete=[segmentation, all_classes])
        self.total_time = timeit.default_timer() - total_time_start


class ShardedCytosolSegmentationCellpose(ShardedSegmentation):
    method = CytosolSegmentationCellpose


class CytosolSegmentationDownsamplingCellpose(CytosolSegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, mask_nucleus: np.ndarray, mask_cytosol: np.ndarray) -> np.ndarray:
        mask_nucleus = self._rescale_downsampled_mask(mask_nucleus, "nucleus_segmentation")
        mask_cytosol = self._rescale_downsampled_mask(mask_cytosol, "cytosol_segmentation")

        mask_nucleus = self._check_seg_dtype(mask=mask_nucleus, mask_name="nucleus")
        mask_cytosol = self._check_seg_dtype(mask=mask_cytosol, mask_name="cytosol")

        # combine masks into one stack
        segmentation = np.stack([mask_nucleus, mask_cytosol])

        return segmentation

    def _execute_segmentation(self, input_image):
        total_time_start = timeit.default_timer()

        # ensure the correct level is selected for the input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # only get the first two channels to save memory consumption
        input_image = input_image[:2, :, :]

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        # downsample the image
        input_image = self._downsample_image(input_image)

        start_segmentation = timeit.default_timer()
        mask_nucleus, mask_cytosol = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # finalize classes list
        all_classes = set(np.unique(mask_nucleus)) - {0}

        segmentation = self._finalize_segmentation_results(mask_nucleus=mask_nucleus, mask_cytosol=mask_cytosol)

        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self._clear_cache(vars_to_delete=[segmentation, all_classes])
        self.total_time = timeit.default_timer() - total_time_start


class ShardedCytosolSegmentationDownsamplingCellpose(ShardedSegmentation):
    method = CytosolSegmentationDownsamplingCellpose


class CytosolOnlySegmentationCellpose(_CellposeSegmentation):
    N_MASKS = 1
    N_INPUT_CHANNELS = 2
    MASK_NAMES = ["cytosol"]
    DEFAULT_CYTOSOL_CHANNEL_IDS = [0, 1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_filtering(self):
        self._check_for_size_filtering(mask_types=self.MASK_NAMES)

    def _finalize_segmentation_results(self, cytosol_mask: np.ndarray) -> np.ndarray:
        # ensure correct dtype of maps
        cytosol_mask = self._check_seg_dtype(mask=cytosol_mask, mask_name="cytosol")

        segmentation = np.stack([cytosol_mask])
        return segmentation

    def cellpose_segmentation(self, input_image: np.ndarray) -> np.ndarray:
        self._setup_processing()
        self._clear_cache()

        #####
        ### Perform Cytosol Segmentation
        #####

        model = self._load_model(model_type="cytosol", gpu=self.use_GPU, device=self.device)

        if self.normalize is False:
            input_image = (input_image - np.min(input_image)) / (
                np.max(input_image) - np.min(input_image)
            )  # min max normalize to 0-1 range as cellpose expects this

        masks_cytosol = model.eval(
            [input_image],
            rescale=self.rescale,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[2, 1],
        )[0]
        masks_cytosol = np.array(masks_cytosol)  # convert to array

        # manually delete model and perform gc to free up memory on GPU
        self._clear_cache(vars_to_delete=[model])

        # ensure edge classes are removed
        masks_cytosol = remove_edge_labels(masks_cytosol)

        #####
        ### Perform Filtering to remove too small/too large masks if applicable
        #####

        self._setup_filtering()

        if self.filter_size:
            masks_cytosol = self._perform_size_filtering(
                mask=masks_cytosol,
                thresholds=self.nucleus_thresholds,  # type: ignore
                confidence_interval=self.nucleus_confidence_interval,  # type: ignore
                mask_name="cytosol",
                log=True,
                debug=self.debug,
                input_image=input_image if self.debug else None,
            )

        masks_cytosol = masks_cytosol.reshape(masks_cytosol.shape[1:])  # add reshape to match shape to HDF5 shape

        return masks_cytosol

    def _execute_segmentation(self, input_image) -> None:
        total_time_start = timeit.default_timer()

        # transform input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # execute segmentation
        start_segmentation = timeit.default_timer()
        cytosol_mask = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # get final classes list
        all_classes = set(np.unique(cytosol_mask)) - {0}

        segmentation = self._finalize_segmentation_results(cytosol_mask)
        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)

        # clean up memory
        self._clear_cache(vars_to_delete=[segmentation, all_classes])

        self.total_time = timeit.default_timer() - total_time_start

        return None


class ShardedCytosolOnlySegmentationCellpose(ShardedSegmentation):
    method = CytosolOnlySegmentationCellpose


class CytosolOnlySegmentationDownsamplingCellpose(CytosolOnlySegmentationCellpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_segmentation_results(self, cytosol_mask: np.ndarray) -> np.ndarray:
        cytosol_mask = self._rescale_downsampled_mask(cytosol_mask, "cytosol_segmentation")

        cytosol_mask = self._check_seg_dtype(mask=cytosol_mask, mask_name="cytosol")

        # combine masks into one stack
        segmentation = np.stack([cytosol_mask])

        return segmentation

    def _execute_segmentation(self, input_image) -> None:
        total_time_start = timeit.default_timer()

        # ensure the correct level is selected for the input image
        input_image = self._transform_input_image(input_image)

        # check image dtype since cellpose expects int input images
        self._check_input_image_dtype(input_image)

        # get the relevant channels for processing
        input_image = input_image[:2, :, :]
        self.input_image_size = input_image.shape

        # setup downsampling
        self._get_downsampling_parameters()
        self._calculate_padded_image_size(input_image)

        # downsample the image
        input_image = self._downsample_image(input_image)

        start_segmentation = timeit.default_timer()
        cytosol_mask = self.cellpose_segmentation(input_image)
        stop_segmentation = timeit.default_timer()
        self.segmentation_time = stop_segmentation - start_segmentation

        # currently no implemented filtering steps to remove nuclei outside of specific thresholds
        all_classes = set(np.unique(cytosol_mask)) - {0}

        segmentation = self._finalize_segmentation_results(cytosol_mask=cytosol_mask)  # type: ignore

        self._save_segmentation_sdata(segmentation, all_classes, masks=self.MASK_NAMES)
        self.total_time = timeit.default_timer() - total_time_start


class ShardedCytosolOnlySegmentationDownsamplingCellpose(ShardedSegmentation):
    method = CytosolOnlySegmentationDownsamplingCellpose
