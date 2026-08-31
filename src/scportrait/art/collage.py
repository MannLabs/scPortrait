"""Create photomosaics from h5sc single-cell image datasets."""

from os import PathLike
from typing import Any

import numpy as np

from scportrait._utils.optional_dependencies import import_optional_dependency
from scportrait._utils.paths import normalize_path


def _get_phomo_module() -> Any:
    """Return phomo with a guided installation error when the art extra is absent."""
    return import_optional_dependency(
        "phomo",
        feature="h5sc photomosaic creation",
        install_hint="pip install 'scportrait[art]'",
    )


def _h5sc_collage_distance_matrix_mps(
    mosaic: Any,
    master_batch_size: int,
    tile_batch_size: int,
) -> np.ndarray:
    """Calculate phomo's Euclidean distance matrix on an Apple MPS device."""
    import torch

    resize_array = import_optional_dependency(
        "phomo.utils",
        attribute="resize_array",
        feature="h5sc photomosaic creation",
        install_hint="pip install 'scportrait[art]'",
    )
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available; use a Mac with an MPS-capable GPU.")

    device = torch.device("mps")
    tiles = np.asarray(mosaic.pool.array, dtype=np.float32)
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError("The collage tile pool must have shape (n_images, height, width, 3).")
    tiles = tiles.reshape(len(tiles), -1)
    distance_matrix = np.empty((len(mosaic.grid.arrays), len(tiles)), dtype=np.float32)

    for tile_start in range(0, len(tiles), tile_batch_size):
        tile_end = min(tile_start + tile_batch_size, len(tiles))
        tile_batch = torch.from_numpy(tiles[tile_start:tile_end]).to(device)
        tile_norm_sq = (tile_batch * tile_batch).sum(dim=1)
        for master_start in range(0, len(mosaic.grid.arrays), master_batch_size):
            master_end = min(master_start + master_batch_size, len(mosaic.grid.arrays))
            arrays = []
            for array in mosaic.grid.arrays[master_start:master_end]:
                if array.shape[:2] != mosaic.tile_shape:
                    array = resize_array(array, (mosaic.tile_shape[1], mosaic.tile_shape[0]))
                arrays.append(array)
            master_batch = torch.from_numpy(np.asarray(arrays, dtype=np.float32).reshape(len(arrays), -1)).to(device)
            distances_squared = (
                (master_batch * master_batch).sum(dim=1, keepdim=True)
                + tile_norm_sq.unsqueeze(0)
                - 2.0 * (master_batch @ tile_batch.T)
            ).clamp_(min=0)
            distance_matrix[master_start:master_end, tile_start:tile_end] = torch.sqrt(distances_squared).cpu().numpy()
    torch.mps.synchronize()
    return distance_matrix


def create_h5sc_collage(
    master_png: str | PathLike[str],
    h5sc_path: str | PathLike[str],
    colors: np.ndarray,
    *,
    grid_width: int,
    n_samples_per_color: int = 200,
    crop_size: int = 200,
    channel_indices: slice = slice(2, 6),
    dataset_key: str = "obsm/single_cell_images",
    edge_fraction_threshold: float | None = 0.20,
    random_seed: int | None = 492755,
    max_cell_uses: int = 1,
    master_batch_size: int = 128,
    tile_batch_size: int = 10_000,
    output_path: str | PathLike[str] | None = None,
) -> Any:
    """Create a photomosaic from colorized h5sc single-cell images.

    Args:
        master_png: PNG image to recreate as a collage.
        h5sc_path: An h5sc file or a directory containing ``*.h5sc`` or ``*.h5`` files.
        colors: RGB colors of shape ``(n_colors, 3)`` in the 0--255 range.
        grid_width: Number of mosaic tiles across the output.
        n_samples_per_color: Unique cells sampled per color and input h5 file.
        crop_size: Side length of each centered square crop in pixels.
        channel_indices: Channels combined into each colorized tile.
        dataset_key: HDF5 dataset containing images shaped ``(cell, channel, y, x)``.
            Defaults to the standardized scPortrait h5sc location.
        edge_fraction_threshold: Discard tiles with a greater nonzero edge fraction.
            Set to ``None`` to keep every tile.
        random_seed: Seed used for reproducible cell sampling; set to ``None`` for random sampling.
        max_cell_uses: Maximum number of times a colorized channel/cell tile may appear in the mosaic.
        master_batch_size: Number of master cells processed per MPS calculation batch.
        tile_batch_size: Number of source tiles processed per MPS calculation batch.
        output_path: Optional path where the completed collage is saved.

    Returns:
        The image result returned by ``phomo.Mosaic.build``.

    Example:
        ```python
        from scportrait.art import create_h5sc_collage

        result = create_h5sc_collage(
            master_png="master.png",
            h5sc_path="cells/",
            colors=[[179, 38, 42], [47, 85, 154]],
            grid_width=60,
            output_path="mosaic.png",
        )
        ```
    """
    import h5py
    from PIL import Image

    phomo = _get_phomo_module()
    source = normalize_path(h5sc_path)
    h5_paths = [source] if source.is_file() else sorted([*source.glob("*.h5sc"), *source.glob("*.h5")])
    if not h5_paths:
        raise FileNotFoundError(f"No .h5sc or .h5 h5sc files found at {source}.")
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3 or np.any((colors < 0) | (colors > 255)):
        raise ValueError("colors must have shape (n_colors, 3) with values from 0 to 255.")
    if grid_width < 1 or n_samples_per_color < 1 or crop_size < 1 or max_cell_uses < 1:
        raise ValueError("grid_width, n_samples_per_color, crop_size, and max_cell_uses must be positive.")

    rng = np.random.default_rng(random_seed)
    colored_images = []
    required_cells = n_samples_per_color * len(colors)
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as handle:
            if dataset_key not in handle:
                raise KeyError(
                    f"{h5_path.name} does not contain {dataset_key!r}. "
                    "Standard scPortrait h5sc files use 'obsm/single_cell_images'; "
                    "pass dataset_key='single_cell_data' only for legacy raw exports."
                )
            data = handle[dataset_key]
            if data.shape[0] < required_cells:
                raise ValueError(
                    f"{h5_path.name}: need {required_cells} unique cells, but only {data.shape[0]} are available."
                )
            height, width = data.shape[-2:]
            if crop_size > min(height, width):
                raise ValueError(f"{h5_path.name}: crop_size ({crop_size}) exceeds image size ({height}, {width}).")
            indices = rng.choice(data.shape[0], size=required_cells, replace=False).reshape(len(colors), -1)
            y0, x0 = (height - crop_size) // 2, (width - crop_size) // 2
            for color, color_indices in zip(colors, indices, strict=True):
                unique_indices, inverse_indices = np.unique(color_indices, return_inverse=True)
                images = data[unique_indices, channel_indices, y0 : y0 + crop_size, x0 : x0 + crop_size]
                images = images[inverse_indices]
                images = images.astype(np.float32).reshape(-1, crop_size, crop_size, 1)
                colored_images.append(images * (color / 255.0))

    tiles = np.concatenate(colored_images, axis=0)
    if edge_fraction_threshold is not None:
        edges = np.concatenate([tiles[:, 0], tiles[:, -1], tiles[:, 1:-1, 0], tiles[:, 1:-1, -1]], axis=1)
        tiles = tiles[np.mean(edges > 0, axis=(1, 2)) <= edge_fraction_threshold / tiles.shape[-1]]
    if not len(tiles):
        raise ValueError("No tiles remain after edge filtering.")
    tiles = np.clip(tiles * (255.0 if tiles.max() <= 1 else 1.0), 0, 255).astype(np.uint8)

    with Image.open(master_png) as image:
        master_width, master_height = image.size
        tile_height, tile_width = tiles.shape[1:3]
        grid_height = round(grid_width * (master_height / master_width) * (tile_width / tile_height))
        if grid_height < 1:
            raise ValueError("Calculated grid height is less than one.")
        if grid_width * grid_height > len(tiles) * max_cell_uses:
            raise ValueError("The requested grid requires more tile uses than are available.")
        master_array = np.asarray(
            image.convert("RGB").resize((grid_width * tile_width, grid_height * tile_height), Image.Resampling.LANCZOS)
        )

    mosaic = phomo.Mosaic(phomo.Master(master_array), phomo.Pool(tiles), n_appearances=max_cell_uses)
    result = mosaic.build(_h5sc_collage_distance_matrix_mps(mosaic, master_batch_size, tile_batch_size))
    if output_path is not None:
        result.save(output_path)
    return result
