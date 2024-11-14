import dask
import dask.array as da
import h5py
import numpy as np


def dask_array_from_path(file_path: str, container_name: str = "array") -> da.Array:
    """Create a Dask array from a HDF5 file, supporting both contiguous and chunked datasets.

    Args:
        file_path: Path pointing to the HDF5 file
        container_name: Name of the dataset in the HDF5 file

    Returns:
        Dask array representing the dataset
    """
    with h5py.File(file_path, "r") as hdf_file:
        array = hdf_file[container_name]
        shape = array.shape
        dtype = array.dtype

        if array.chunks is None:
            # Contiguous dataset
            offset = array.id.get_offset()
            chunks = calculate_chunk_sizes(shape, dtype)
            dask_array = mmap_dask_array_contigious(file_path, shape, dtype, offset=offset, chunks=chunks)
        else:
            # Chunked dataset
            chunks = calculate_chunk_sizes_chunks(shape, dtype, HDF5_chunk_size=array.chunks)
            dask_array = mmap_dask_array_chunked(file_path, shape, dtype, container_name, chunks)

    return dask_array


def calculate_chunk_sizes(shape: tuple[int, ...], dtype: np.dtype | str, target_size_gb: int = 5) -> tuple[int, ...]:
    """Calculate chunk sizes that result in chunks of approximately the target size in GB.

    Args:
        shape: Shape of the array
        dtype: Data type of the array
        target_size_gb: Target size of each chunk in gigabytes

    Returns:
        Calculated chunk sizes for the Dask array
    """
    element_size = np.dtype(dtype).itemsize
    target_size_bytes = target_size_gb * 1024**3
    total_elements_per_chunk = target_size_bytes // element_size

    chunk_sizes = list(shape)

    while np.prod(chunk_sizes) > total_elements_per_chunk:
        for i in range(len(chunk_sizes)):
            if chunk_sizes[i] > 1:
                chunk_sizes[i] = chunk_sizes[i] // 2
                if np.prod(chunk_sizes) <= total_elements_per_chunk:
                    break

    return tuple(chunk_sizes)


def calculate_chunk_sizes_chunks(
    shape: tuple[int, ...], dtype: np.dtype | str, HDF5_chunk_size: tuple[int, ...], target_size_gb: int = 5
) -> tuple[int, ...]:
    """Calculate chunk sizes that result in chunks of approximately the target size in GB.

    Args:
        shape: Shape of the array
        dtype: Data type of the array
        HDF5_chunk_size: Chunk sizes of the existing HDF5 data container
        target_size_gb: Target size of each chunk in gigabytes

    Returns:
        Calculated chunk sizes for the Dask array
    """
    element_size = np.dtype(dtype).itemsize
    target_size_bytes = target_size_gb * 1024**3
    total_elements_per_chunk = target_size_bytes // element_size

    HDF5_chunk_size = list(HDF5_chunk_size)  # type: ignore
    chunk_sizes = HDF5_chunk_size.copy()  # type: ignore

    while np.prod(chunk_sizes) < total_elements_per_chunk:
        for i in range(len(chunk_sizes)):
            if chunk_sizes[i] > 1:
                chunk_sizes[i] = chunk_sizes[i] + HDF5_chunk_size[i]
                if np.prod(chunk_sizes) >= total_elements_per_chunk:
                    break

    return tuple(chunk_sizes)


def mmap_dask_array_contigious(
    filename: str, shape: tuple[int, ...], dtype: np.dtype | str, offset: int = 0, chunks: tuple[int, ...] = (5,)
) -> da.Array:
    """Create a Dask array from raw binary data in `filename` by memory mapping.

    Args:
        filename: Path to the raw binary data file
        shape: Shape of the array
        dtype: Data type of the array
        offset: Offset in bytes from the beginning of the file
        chunks: Chunk sizes for the Dask array

    Returns:
        Dask array that is memory-mapped to disk
    """
    load = dask.delayed(load_hdf5_contigious)
    chunk_arrays = []

    for i in range(0, shape[0], chunks[0]):
        row_chunks = []
        for j in range(0, shape[1], chunks[1]):
            col_chunks = []
            for k in range(0, shape[2], chunks[2]):
                chunk_shape = (
                    min(chunks[0], shape[0] - i),
                    min(chunks[1], shape[1] - j),
                    min(chunks[2], shape[2] - k),
                )
                slices = (
                    slice(i, i + chunk_shape[0]),
                    slice(j, j + chunk_shape[1]),
                    slice(k, k + chunk_shape[2]),
                )
                chunk = da.from_delayed(
                    load(filename, shape, dtype, offset, slices),
                    shape=chunk_shape,
                    dtype=dtype,
                )
                col_chunks.append(chunk)
            row_chunks.append(da.concatenate(col_chunks, axis=2))
        chunk_arrays.append(da.concatenate(row_chunks, axis=1))
    return da.concatenate(chunk_arrays, axis=0)


def mmap_dask_array_chunked(
    filename: str, shape: tuple[int, ...], dtype: np.dtype | str, container_name: str, chunks: tuple[int, ...] = (5,)
) -> da.Array:
    """Create a Dask array from raw binary data in `filename` by memory mapping.

    Args:
        filename: Path to the raw binary data file
        shape: Shape of the array
        dtype: Data type of the array
        container_name: Name of the dataset in the HDF5 file
        chunks: Chunk sizes for the Dask array

    Returns:
        Dask array that is memory-mapped to disk
    """
    load = dask.delayed(load_hdf5_chunk)
    chunk_arrays = []

    for i in range(0, shape[0], chunks[0]):
        row_chunks = []
        for j in range(0, shape[1], chunks[1]):
            col_chunks = []
            for k in range(0, shape[2], chunks[2]):
                chunk_shape = (
                    min(chunks[0], shape[0] - i),
                    min(chunks[1], shape[1] - j),
                    min(chunks[2], shape[2] - k),
                )
                slices = (
                    slice(i, i + chunk_shape[0]),
                    slice(j, j + chunk_shape[1]),
                    slice(k, k + chunk_shape[2]),
                )
                chunk = da.from_delayed(
                    load(filename, container_name, slices),
                    shape=chunk_shape,
                    dtype=dtype,
                )
                col_chunks.append(chunk)
            row_chunks.append(da.concatenate(col_chunks, axis=2))
        chunk_arrays.append(da.concatenate(row_chunks, axis=1))
    return da.concatenate(chunk_arrays, axis=0)


def load_hdf5_contigious(
    filename: str, shape: tuple[int, ...], dtype: np.dtype | str, offset: int, slices: tuple[slice, ...]
) -> np.ndarray:
    """Memory map the given file with overall shape and dtype and return a slice.

    Args:
        filename: Path to the raw binary data file
        shape: Shape of the array
        dtype: Data type of the array
        offset: Offset in bytes from the beginning of the file
        slices: Tuple of slices specifying the chunk to load

    Returns:
        The sliced chunk from the memory-mapped array
    """
    data = np.memmap(filename, mode="r", shape=shape, dtype=dtype, offset=offset)
    return data[slices]


def load_hdf5_chunk(file_path: str, container_name: str, slices: tuple[slice, ...]) -> np.ndarray:
    """Load a chunk of data from a chunked HDF5 dataset.

    Args:
        file_path: Path to the HDF5 file
        container_name: Name of the dataset in the HDF5 file
        slices: Tuple of slices specifying the chunk to load

    Returns:
        The sliced chunk from the HDF5 dataset
    """
    with h5py.File(file_path, "r") as f:
        f[container_name][slices]
