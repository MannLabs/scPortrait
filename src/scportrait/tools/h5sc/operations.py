"""
h5sc
=======

Functions to work with scPortrait's standardized single-cell data format.
"""

from pathlib import Path

import h5py
import numpy as np

from scportrait.pipeline._utils.constants import DEFAULT_CELL_ID_NAME, DEFAULT_NAME_SINGLE_CELL_IMAGES


def get_image_index(adata, cell_id: int | list[int]) -> int | list[int]:
    """
    Retrieve the image index (row index) of a specific cell id in a H5SC object.

    Args:
        adata: An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
        cell_id: A single cell ID or a list of cell IDs to retrieve indices for.

    Returns:
        The corresponding index or list of indices from `adata.obs.index`.

    """
    lookup = dict(zip(adata.obs[DEFAULT_CELL_ID_NAME], adata.obs.index.astype(int), strict=True))

    if isinstance(cell_id, int):
        return lookup[cell_id]

    return [lookup[_id] for _id in cell_id]


def get_image_with_cellid(adata, cell_id: list[int] | int, select_channel: int | list[int] | None = None) -> np.ndarray:
    """Get single cell images from the cells with the provided cell IDs. Images are returned in the order of the cell IDs.

    Args:
        adata: An AnnData object with obsm["single_cell_images"] containing a memory-backed array of the single-cell images.
        cell_id: The cell ID of the cell to retrieve the image for.
        select_channel: The channel to select from the image. If `None`, all channels are returned.

    Returns:
        The image(s) of the cell with the passed Cell IDs.
    """
    lookup = dict(zip(adata.obs[DEFAULT_CELL_ID_NAME], adata.obs.index.astype(int), strict=True))
    image_container = adata.obsm[DEFAULT_NAME_SINGLE_CELL_IMAGES]

    if isinstance(cell_id, int):
        cell_id = [cell_id]

    for x in cell_id:
        assert x in lookup.keys(), f"CellID {x} is not present in the AnnData object."

    images = []
    for _id in cell_id:
        idx = lookup[_id]
        if select_channel is None:
            image = image_container[idx][:]
        else:
            image = image_container[idx][select_channel]
        images.append(image)

    array = np.array(images)
    if array.shape[0] == 1:  # Check if the first dimension is 1
        return array.squeeze(axis=0)  # Remove the first dimension
    else:
        return array


def conforms(path: str | Path, *, raise_on_error: bool = False) -> bool:
    """
    Check whether a file conforms to scPortrait's current H5SC convention.

    Parameters
    ----------
    path
        Path to the file to validate.
    raise_on_error
        If True, raise ValueError describing the first validation failure.
        Otherwise return False.

    Returns
    -------
    bool
        True if the file conforms to the expected H5SC structure.
    """

    path = Path(path)

    def _decode_scalar(value) -> str:
        """Convert an HDF5 scalar string/bytes attribute to str."""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _decode_strings(values) -> list[str]:
        """Convert an HDF5 string array attribute to a list[str]."""
        array = np.asarray(values)
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in array]

    def fail(message: str) -> bool:
        if raise_on_error:
            raise ValueError(f"Invalid H5SC file: {message}")
        return False

    if not path.is_file():
        return fail(f"file does not exist: {path}")

    if path.suffix.lower() != ".h5sc":
        return fail("file extension must be '.h5sc'")

    try:
        with h5py.File(path, "r") as f:
            # H5SC is AnnData-backed, so these groups should exist.
            for group in ("obs", "var", "obsm", "uns"):
                if group not in f:
                    return fail(f"missing required AnnData group '/{group}'")

            dataset_path = "obsm/single_cell_images"

            if dataset_path not in f:
                return fail(f"missing required dataset '/{dataset_path}'")

            images = f[dataset_path]

            if not isinstance(images, h5py.Dataset):
                return fail(f"'/{dataset_path}' is not an HDF5 dataset")

            # scPortrait expects (cells, channels, height, width).
            if images.ndim != 4:
                return fail(f"'/{dataset_path}' must have shape (N, C, H, W); got {images.shape}")

            n_cells, n_channels, height, width = images.shape

            if n_cells == 0:
                return fail("image dataset contains zero cells")

            if n_channels == 0:
                return fail("image dataset contains zero channels")

            # Attributes written by scPortrait.
            required_attrs = {
                "encoding-type",
                "encoding-version",
                "n_cells",
                "n_channels",
                "n_masks",
                "n_image_channels",
                # "image_size_x",
                # "image_size_y",
                "channel_names",
                "channel_mapping",
                "compression",
            }

            missing_attrs = required_attrs - set(images.attrs)
            if missing_attrs:
                return fail("missing attributes on '/obsm/single_cell_images': " + ", ".join(sorted(missing_attrs)))

            attrs = images.attrs

            # AnnData array encoding used by the scPortrait writer.
            encoding_type = _decode_scalar(attrs["encoding-type"])
            encoding_version = _decode_scalar(attrs["encoding-version"])

            if encoding_type != "array":
                return fail(f"unexpected encoding-type {encoding_type!r}; expected 'array'")

            if encoding_version != "0.2.0":
                return fail(f"unexpected encoding-version {encoding_version!r}; expected '0.2.0'")

            # Validate shape metadata.
            expected_shape_metadata = {
                "n_cells": n_cells,
                "n_channels": n_channels,
                # "image_size_x": height,
                # "image_size_y": width,
            }

            for name, expected in expected_shape_metadata.items():
                actual = int(attrs[name])
                if actual != expected:
                    return fail(f"attribute {name!r} is {actual}, but dataset implies {expected}")

            channel_names = _decode_strings(attrs["channel_names"])
            channel_mapping = _decode_strings(attrs["channel_mapping"])

            if len(channel_names) != n_channels:
                return fail(
                    f"channel_names contains {len(channel_names)} entries"
                    f"but the image tensor contains {n_channels} channels"
                )

            if len(channel_mapping) != n_channels:
                return fail(
                    f"channel_mapping contains {len(channel_mapping)} entries"
                    f"but the image tensor contains {n_channels} channels"
                )

            valid_mappings = {"mask", "image_channel"}
            invalid = set(channel_mapping) - valid_mappings

            if invalid:
                return fail("channel_mapping contains unsupported value(s): " + ", ".join(sorted(invalid)))

            n_masks = sum(x == "mask" for x in channel_mapping)
            n_image_channels = sum(x == "image_channel" for x in channel_mapping)

            if int(attrs["n_masks"]) != n_masks:
                return fail(f"n_masks attribute is {attrs['n_masks']}, but channel_mapping contains {n_masks} masks")

            if int(attrs["n_image_channels"]) != n_image_channels:
                return fail(
                    f"n_image_channels attribute is {attrs['n_image_channels']}, "
                    f"but channel_mapping contains {n_image_channels} image channels"
                )

            if n_masks + n_image_channels != n_channels:
                return fail("mask and image-channel counts do not add up to n_channels")

            # Current scPortrait writer supports these compression methods.
            compression = _decode_scalar(attrs["compression"])
            if compression not in {"gzip", "lzf"}:
                return fail(f"unsupported compression {compression!r}; expected 'gzip' or 'lzf'")

            if images.compression != compression:
                return fail(f"compression metadata says {compression!r}, but dataset uses {images.compression!r}")

    except (OSError, ValueError, TypeError) as exc:
        return fail(str(exc))

    return True
