import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData
from anndata._io.h5ad import _clean_uns, _read_raw, read_dataframe, read_elem

from scportrait.pipeline._utils.constants import (
    DEFAULT_CELL_ID_NAME,
    DEFAULT_IDENTIFIER_FILENAME,
    DEFAULT_NAME_SINGLE_CELL_IMAGES,
    DEFAULT_SEGMENTATION_DTYPE,
    DEFAULT_SINGLE_CELL_IMAGE_DTYPE,
    IMAGE_DATACONTAINER_NAME,
)


def read_h5sc(filename: str | Path) -> AnnData:
    """Read scportrait's single-cell image dataset format.

    Args:
        filename: Path to the file to read.
        mode: Mode in which to open the file.

    Returns:
        An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
    """
    mode = "r"  # hard code as there is no benefit from letting the user set this manually as writing back and changing values is currently not supported
    d = {}

    # connect to h5py file
    f = h5py.File(filename, mode)

    attributes = ["varm", "obsp", "varp", "uns", "layers"]
    df_attributes = ["obs", "var"]

    if "encoding-type" in f.attrs:
        attributes.extend(df_attributes)
    else:
        for k in df_attributes:
            if k in f:  # Backwards compat
                d[k] = read_dataframe(f[k])

    d.update({k: read_elem(f[k]) for k in attributes if k in f})

    d["raw"] = _read_raw(f, attrs={"var", "varm"})
    adata = AnnData(**d)

    # Backwards compat to <0.7
    if isinstance(f["obs"], h5py.Dataset):
        _clean_uns(adata)

    adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES] = f.get(IMAGE_DATACONTAINER_NAME)
    adata.uns[DEFAULT_IDENTIFIER_FILENAME] = filename
    adata.uns["_h5sc_file_handle"] = f

    return adata


def numpy_to_h5sc(
    mask_names: Sequence[str],
    channel_names: Sequence[str],
    mask_imgs: npt.NDArray,
    channel_imgs: npt.NDArray,
    output_path: str | Path,
    cell_ids: npt.NDArray[np.integer[Any]],
    cell_metadata: pd.DataFrame | None = None,
    image_dtype=DEFAULT_SINGLE_CELL_IMAGE_DTYPE,
    compression_type: Literal["gzip", "lzf"] = "gzip",
) -> None:
    """Create and write an scPortrait-style `.h5sc` file from NumPy arrays of single-cell
    masks and image channels, with optional per-cell metadata.

    This function builds a valid AnnData-backed HDF5 container following the scPortrait
    “H5SC” convention. Internally, the file is a standard AnnData `.h5ad` structure whose
    filename ends in `.h5sc`, and which contains a 4D image tensor stored at:

        /obsm/single_cell_images

    with shape:

        (N, C, H, W)

    where:
        N = number of cells
        C = n_masks + n_image_channels
        H = image height
        W = image width

    The mask channels are stored first, followed by the image channels. All data are
    written as a single float16 HDF5 dataset, with mask values encoded as 0.0 and 1.0.

    Cell identifiers and optional per-cell metadata are written to `adata.obs`.

    Metadata are written redundantly:
        - At the AnnData level in `adata.uns[...]`
        - At the HDF5 level as attributes on `/obsm/single_cell_images`

    This allows the file to be read both via AnnData and as a standalone HDF5 image
    container.

    Args:
        mask_names: Names of the mask channels. Length must match `mask_imgs.shape[1]`.
        channel_names: Names of the image channels. Length must match `channel_imgs.shape[1]`.
        mask_imgs: Array of mask images with shape `(N, n_masks, H, W)`.
            Masks are expected to be binary (0 or 1) and will be stored as float16.
        channel_imgs: Array of image channels with shape `(N, n_image_channels, H, W)`.
            Images should already be normalized (e.g., to [0, 1]) before writing.
        output_path: Path of the `.h5sc` file to create, e.g. `"/path/to/file.h5sc"`.
            The file will be overwritten if it already exists.
        cell_ids: Array of segmentation cell identifiers with shape `(N,)`.
            These values are written into `adata.obs[DEFAULT_CELL_ID_NAME]` and define the
            mapping between row index and original segmentation label.
        cell_metadata: Optional per-cell metadata to be written into `adata.obs`.
            Must have exactly `N` rows. Columns will be merged into `obs` alongside the
            cell ID column. The index is ignored and replaced by AnnData’s internal index.
        compression_type: HDF5 compression algorithm used for the image tensor.
            - "gzip": better compression, slower I/O
            - "lzf" : faster I/O, lower compression ratio

    File layout created:
        The resulting file contains:

            /obs
                Per-cell metadata including cell IDs and optional user-provided metadata.
            /var
                Channel metadata (channel names and channel mapping).
            /uns
                scPortrait metadata describing the image container.
            /obsm/single_cell_images
                HDF5 dataset with shape (N, C, H, W), dtype float16, chunked as
                (1, 1, H, W), compressed.

    Notes:
        - The file is technically an AnnData `.h5ad` file with a `.h5sc` extension.
        - Masks and image channels share a single dataset and dtype (`float16`).
        - The function performs a single-threaded write; no file locking is used.
        - All input arrays are cast to the storage dtype before writing.

    Warnings:
        UserWarning: If `mask_imgs` or `channel_imgs` contain values outside [0, 1].
            Mask images  or channel images are outside the expected [0, 1] range. This does not align with
            scPortrait's convention and unscaled data can produce unexpected results in downstream
            functions or require additional preprocessing before passing images to deep learning models.

    Raises:
        Exception: If:
            - `mask_imgs` or `channel_imgs` do not have 4 dimensions `(N, C, H, W)`,
            - `mask_imgs` and `channel_imgs` have different numbers of cells,
            - `mask_imgs` and `channel_imgs` have different image sizes,
            - the number of provided channel names does not match the array shapes,
            - `cell_metadata` does not have `N` rows,
            - an unsupported compression type is requested.
    """
    if mask_imgs.ndim != 4 or channel_imgs.ndim != 4:
        raise Exception("mask_imgs and channel_imgs must have shape (N, C, H, W) with exactly 4 dimensions.")
    if mask_imgs.shape[0] != channel_imgs.shape[0]:
        raise Exception(
            "mask_imgs and channel_imgs do not contain the same number of cells. The expected shape is (N, C, H, W)."
        )
    if mask_imgs.shape[2:4] != channel_imgs.shape[2:4]:
        raise Exception(
            "mask_imgs and channel_imgs do not contain the same image size. The expected shape is (N, C, H, W)."
        )
    # check mask_names and channel_names fit to imgs shape-wise
    if len(mask_names) != mask_imgs.shape[1]:
        raise Exception(
            "mask_names needs to match mask_imgs.shape[1]. You need to pass the same number of masks and labels."
        )
    if len(channel_names) != channel_imgs.shape[1]:
        raise Exception(
            "channel_names needs to match channel_imgs.shape[1]. You need to pass the same number of image channels and labels."
        )
    if compression_type not in ["gzip", "lzf"]:
        raise Exception("Compression needs to be lzf or gzip.")

    # prepare metadata
    channels = np.concatenate([mask_names, channel_names])
    num_cells = channel_imgs.shape[0]
    img_size = channel_imgs.shape[2:4]
    cell_ids = cell_ids.astype(DEFAULT_SEGMENTATION_DTYPE, copy=False)
    channel_mapping = ["mask" for x in mask_names] + ["image_channel" for x in channel_names]

    # prepare images
    if mask_imgs.min() < 0 or mask_imgs.max() > 1:
        warnings.warn(
            "Mask images are outside the expected [0, 1] range. This does not align with "
            "scPortrait's convention and unscaled data can produce unexpected results in downstream "
            "functions or require additional preprocessing before passing images to "
            "deep learning models.",
            UserWarning,
            stacklevel=2,
        )

    if channel_imgs.min() < 0 or channel_imgs.max() > 1:
        warnings.warn(
            "   Image channels are outside the expected [0, 1] range. This does not align with"
            "scPortrait's convention and unscaled data can produce unexpected results in downstream "
            "functions or require additional preprocessing before passing images to "
            "deep learning models.",
            UserWarning,
            stacklevel=2,
        )
    all_imgs = np.concatenate([mask_imgs, channel_imgs], axis=1)
    all_imgs = all_imgs.astype(image_dtype, copy=False)

    # create var object with channel names and their mapping to mask or image channels
    vars = pd.DataFrame(index=np.arange(len(channels)).astype("str"))
    vars["channels"] = channels
    vars["channel_mapping"] = channel_mapping

    obs = pd.DataFrame({DEFAULT_CELL_ID_NAME: cell_ids})
    obs.index = obs.index.values.astype("str")
    if cell_metadata is not None:
        if len(cell_metadata) != num_cells:
            raise Exception(
                f"cell_metadata must have {num_cells} rows to match the number of cells, got {len(cell_metadata)}."
            )
        for col in cell_metadata.columns:
            obs[col] = cell_metadata[col].values

    # create anndata object
    adata = AnnData(obs=obs, var=vars)

    # add additional metadata to `uns`
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/n_cells"] = num_cells
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/n_channels"] = len(channels)
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/n_masks"] = mask_imgs.shape[1]
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/n_image_channels"] = channel_imgs.shape[1]
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/image_size_x"] = img_size[0]
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/image_size_y"] = img_size[1]
    # adata.uns[f"{self.DEFAULT_NAME_SINGLE_CELL_IMAGES}/normalization"] = self.normalization
    # adata.uns[f"{self.DEFAULT_NAME_SINGLE_CELL_IMAGES}/normalization_range_lower"] = self.normalization_range[0]
    # adata.uns[f"{self.DEFAULT_NAME_SINGLE_CELL_IMAGES}/normalization_range_upper"] = self.normalization_range[1]
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/channel_names"] = channels
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/channel_mapping"] = np.array(channel_mapping, dtype="<U15")
    adata.uns[f"{DEFAULT_NAME_SINGLE_CELL_IMAGES}/compression"] = compression_type

    # write to file
    adata.write(output_path)

    # add an empty HDF5 dataset to the obsm group of the anndata object
    with h5py.File(output_path, "a") as hf:
        hf.create_dataset(
            IMAGE_DATACONTAINER_NAME,
            shape=all_imgs.shape,
            chunks=(1, 1, img_size[0], img_size[1]),
            compression=compression_type,
            dtype=image_dtype,
        )

        # add required metadata from anndata package
        hf[IMAGE_DATACONTAINER_NAME].attrs["encoding-type"] = "array"
        hf[IMAGE_DATACONTAINER_NAME].attrs["encoding-version"] = "0.2.0"

        # add relevant metadata to the single-cell image container
        hf[IMAGE_DATACONTAINER_NAME].attrs["n_cells"] = num_cells
        hf[IMAGE_DATACONTAINER_NAME].attrs["n_channels"] = len(channels)
        hf[IMAGE_DATACONTAINER_NAME].attrs["n_masks"] = mask_imgs.shape[1]
        hf[IMAGE_DATACONTAINER_NAME].attrs["n_image_channels"] = channel_imgs.shape[1]
        hf[IMAGE_DATACONTAINER_NAME].attrs["image_size_x"] = img_size[0]
        hf[IMAGE_DATACONTAINER_NAME].attrs["image_size_y"] = img_size[1]
        # hf[IMAGE_DATACONTAINER_NAME].attrs["normalization"] = self.normalization
        # hf[IMAGE_DATACONTAINER_NAME].attrs["normalization_range"] = self.normalization_range
        hf[IMAGE_DATACONTAINER_NAME].attrs["channel_names"] = np.array([x.encode("utf-8") for x in channels])
        mapping_values = ["mask" for x in mask_names] + ["image_channel" for x in channel_names]
        hf[IMAGE_DATACONTAINER_NAME].attrs["channel_mapping"] = np.array([x.encode("utf-8") for x in mapping_values])
        hf[IMAGE_DATACONTAINER_NAME].attrs["compression"] = compression_type

        # Write images to .h5sc file, single thread
        single_cell_data_container: h5py.Dataset = hf[IMAGE_DATACONTAINER_NAME]

        for save_index, img in enumerate(all_imgs):
            single_cell_data_container[save_index] = img
