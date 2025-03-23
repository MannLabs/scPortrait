### Definition of default values used throughout the scPortrait pipeline ###
from typing import TypeAlias

import numpy as np

ChunkSize2D: TypeAlias = tuple[int, int]
ChunkSize3D: TypeAlias = tuple[int, int, int]

DEFAULT_LOG_NAME: str = "processing.log"
DEFAULT_FORMAT: str = "%d/%m/%Y %H:%M:%S"

DEFAULT_CONFIG_NAME: str = "config.yml"
DEFAULT_INPUT_IMAGE_NAME: str = "input_image"
DEFAULT_SDATA_FILE: str = "scportrait.sdata"

DEFAULT_PREFIX_MAIN_SEG: str = "seg_all"
DEFAULT_PREFIX_FILTERED_SEG: str = "seg_filtered"
DEFAULT_PREFIX_SELECTED_SEG: str = "seg_selected"

DEFAULT_SEG_NAME_0: str = "nucleus"
DEFAULT_SEG_NAME_1: str = "cytosol"

DEFAULT_CENTERS_NAME: str = "centers"
DEFAULT_CHUNK_SIZE_3D: ChunkSize3D = (1, 2000, 2000)
DEFAULT_CHUNK_SIZE_2D: ChunkSize2D = (2000, 2000)
DEFAULT_SCALE_FACTORS: list[int] = [2, 4, 8]

DEFAULT_SEGMENTATION_DIR_NAME: str = "segmentation"
DEFAULT_TILES_FOLDER: str = "tiles"
DEFAULT_FILTER_ADDTIONAL_FILE: str = "needs_additional_filtering.txt"

DEFAULT_EXTRACTION_DIR_NAME: str = "extraction"
DEFAULT_DATA_DIR: str = "data"
DEFAULT_NAME_SINGLE_CELL_IMAGES = "single_cell_images"
IMAGE_DATACONTAINER_NAME = f"obsm/{DEFAULT_NAME_SINGLE_CELL_IMAGES}"
DEFAULT_CELL_ID_NAME = "cell_id"
INDEX_DATACONTAINER_NAME = f"obs/{DEFAULT_CELL_ID_NAME}"

DEFAULT_IMAGE_DTYPE: np.dtype = np.uint16
DEFAULT_SEGMENTATION_DTYPE: np.dtype = np.uint64
DEFAULT_SINGLE_CELL_IMAGE_DTYPE: np.dtype = np.float16

DEFAULT_SEGMENTATION_FILE: str = "segmentation.h5"
DEFAULT_CLASSES_FILE: str = "classes.csv"
DEFAULT_REMOVED_CLASSES_FILE: str = "removed_classes.csv"
DEFAULT_EXTRACTION_FILE: str = "single_cells.h5sc"
DEFAULT_BENCHMARKING_FILE: str = "benchmarking.csv"

DEFAULT_SELECTION_DIR_NAME: str = "selection"
DEFAULT_FEATURIZATION_DIR_NAME: str = "featurization"

DEFAULT_CHANNELS_NAME: str = "channels"
DEFAULT_MASK_NAME: str = "labels"
