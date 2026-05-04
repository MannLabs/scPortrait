import os
from collections.abc import Callable, Iterable
from pathlib import PosixPath
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from scportrait.pipeline._utils.constants import IMAGE_DATACONTAINER_NAME, INDEX_DATACONTAINER_NAME


def _check_type_input_list(var):
    """Return `True` when `var` is an iterable of iterables of integers."""
    return isinstance(var, Iterable) and all(
        isinstance(sublist, Iterable) and all(isinstance(item, int) for item in sublist) for sublist in var
    )


class _HDF5SingleCellDataset(Dataset):
    """Base class for loading scPortrait single-cell datasets stored in HDF5 files."""

    HDF_FILETYPES = ["hdf", "hf", "h5", "hdf5"]  # supported hdf5 filetypes
    _CLOSED_MESSAGE = "Dataset has been closed and cannot be reused."

    def __init__(
        self,
        dir_list: list[str],
        index_list: list[list[int]] | None = None,
        select_channel: list[int] | int | None = None,
        transform=None,
        return_id: bool = True,
        max_level: int = 5,
    ):
        """Initialize the dataset index and runtime state.

        Args:
            dir_list: Paths to HDF5 files or directories containing HDF5 files.
            index_list: Per-input lists of cell indices to include. If `None`, all cells are used.
            select_channel: Channel index or indices to load from each cell image. If `None`, all channels are used.
            transform: Optional transform applied to each returned tensor.
            return_id: Whether to return the unique cell id together with the cell data.
            max_level: Maximum directory depth to scan when an entry in `dir_list` is a directory.
        """
        self.dir_list = dir_list

        if isinstance(self.dir_list[0], PosixPath):
            self.dir_list = [str(x) for x in self.dir_list]

        # ensure select_channel is always a list
        if isinstance(select_channel, int):
            select_channel = [select_channel]

        self.select_channel = select_channel
        self.transform = transform
        self.return_id = return_id
        self.max_level = max_level

        # ensure index list is long enough for all directories
        if index_list is None:
            index_list = [[None]] * len(dir_list)
        else:
            if len(index_list) < len(dir_list):
                raise ValueError("index_list should be as long as dir_list")

            # type check index list to make sure its correctly constructed
            assert _check_type_input_list(index_list), (
                "The parameter index_list expects the following format [list_dataset1, list_dataset2, ...]. Please ensure that you provide an index list for each file listed in dir_list."
            )

        self.index_list: list[list[Any]] = index_list

        # check reading of an image element and ensure that the selected channels are valid

        # initialize placeholders to store dataset information
        self.paths: list[str] = []
        self._open_hdf: dict[str, h5py.File | None] = {}
        self.data_locator: list[list[int]] = []
        self._closed = False

        self.bulk_labels: list[int] | None = None
        self.label_column: int | None = None

    def _ensure_open(self) -> None:
        """Raise if the dataset has been explicitly closed."""
        if self._closed:
            raise RuntimeError(self._CLOSED_MESSAGE)

    def get_hdf(self, path: str) -> h5py.File:
        """
        Retrieve (or lazily create) a *process-local* file handle.
        """
        self._ensure_open()
        file = self._open_hdf.get(path)

        if file is None:  # first call in this process
            file = h5py.File(path, mode="r")
            self._open_hdf[path] = file

        return file

    def close(self) -> None:
        """Close open HDF5 handles and permanently disable further data access."""
        if self._closed:
            return

        for file in self._open_hdf.values():
            if file is not None:
                try:
                    file.close()
                except (OSError, RuntimeError, ValueError):
                    continue
        self._open_hdf.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass

    def _add_hdf_to_index(
        self,
        path: str,
        index_list: list[int] | None = None,
        label: int | None = None,
        label_column: int | None = None,
        dtype_label_column=float,
        label_column_transform=None,
        read_label: bool = False,
    ):
        """Add single-cell data from `path` to the dataset index."""

        # check to ensure that HDF5 file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found. Please ensure that the file exists.")

        input_hdf = None
        try:
            # connect to h5py file
            input_hdf = h5py.File(path, "r")

            # get single cell index handle
            if index_list != [None]:
                index_handle = np.zeros((len(index_list), 2), dtype=np.int64)

                # ensure that no out of bound elements are provided for the dataset
                max_elements = input_hdf.get("single_cell_index").shape[0]
                max_index = max(index_list)

                assert max_index < max_elements, (
                    f"Index {max_index} is out of bounds for file {path}. Only {max_elements} single cell records available in dataset."
                )

                for i, ix in enumerate(index_list):
                    index_handle[i] = input_hdf.get("single_cell_index")[ix]

            else:
                index_handle = input_hdf.get("single_cell_index")

            # ensure that selected channels are within range
            if self.select_channel is not None:
                max_channels = input_hdf.get("single_cell_data").shape[1]
                assert np.all([channel_ix < max_channels for channel_ix in self.select_channel]), (
                    f"Selected channels are out of bounds. Maximum available channelid is {max_channels}."
                )

            # Add the path for later lazy file access in `__getitem__`.
            handle_id = len(self.paths)
            self.paths.append(path)  # add path to new dataset to list of paths

            # add single-cell labelling
            if read_label:
                assert label_column is not None, "Label column must be provided if read_label is set to True."

                # get the column containing the labelling
                label_col = input_hdf.get("single_cell_index_labelled").asstr()[:, label_column]

                # Older datasets may store missing labels as empty strings.
                if len(label_col[label_col == ""]) > 0:
                    Warning(
                        "Empty strings found in label column. Replacing with nan. Please check your single-cell dataset."
                    )
                    label_col[label_col == ""] = np.nan

                # convert labels to desired dtype
                label_col = label_col.astype(dtype_label_column)

                # apply any mathematical transform to label column if specified (e.g. to change scale by dividing by 1000)
                if label_column_transform is not None:
                    label_col = label_column_transform(label_col)

                # generate identifiers for all single-cells
                # iterate over rows in index handle, i.e. over all cells
                for current_target, row in zip(label_col, index_handle, strict=False):
                    # append target, handle id, and row to data locator
                    self.data_locator.append([current_target, handle_id] + list(row))

            else:
                assert label is not None, "Label must be provided if read_label is set to False."

                # generate identifiers for all single-cells
                for row in index_handle:
                    self.data_locator.append([label, handle_id] + list(row))

        except (FileNotFoundError, KeyError, OSError) as e:
            print(f"Error: {e}")
            return
        finally:
            if input_hdf is not None:
                input_hdf.close()

    def _add_dataset(
        self,
        path: str,
        current_index_list: list[int],
        id: int,
        read_label_from_dataset: bool,
    ):
        """Adds a dataset to the index."""
        if read_label_from_dataset:
            assert self.label_column is not None, (
                "trying to read labels from dataset but no column to access information has been passed"
            )
            self._add_hdf_to_index(
                path=path,
                index_list=current_index_list,
                label=None,
                label_column=self.label_column,
                dtype_label_column=self.dtype_label_column,
                label_column_transform=self.label_column_transform,
                read_label=True,
            )
        else:
            assert self.bulk_labels is not None, (
                "trying to apply bulk labels to all cells from dataset but no label provided"
            )

            self._add_hdf_to_index(
                path=path,
                index_list=current_index_list,
                label=self.bulk_labels[id],
                label_column=None,
                dtype_label_column=None,
                label_column_transform=None,
                read_label=False,
            )

    def _scan_directory(
        self,
        path: str,
        levels_left: int,
        current_index_list: list[int] | None = None,
        read_label_from_dataset: bool = False,
    ) -> None:
        """Recursively scan `path` and add matching HDF5 files to the dataset index.

        Args:
            path: Directory to search for HDF5 files.
            levels_left: Remaining directory depth to scan recursively.
            current_index_list: Indices to select from the dataset, if any.
            read_label_from_dataset: Whether labels should be read from each dataset file.
        """

        # iterates over all files and folders in a directory
        # hdf5 files are added to the index
        # subfolders are recursively scanned

        if levels_left > 0:
            # get files and directories at current level

            current_level_directories = [
                os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
            ]

            current_level_files = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]

            for i, file in enumerate(current_level_files):
                filetype = file.split(".")[-1]

                if filetype in self.HDF_FILETYPES:
                    self._add_dataset(
                        path=file,
                        current_index_list=current_index_list,
                        id=i,
                        read_label_from_dataset=read_label_from_dataset,
                    )

            # recursively scan subdirectories
            for subdirectory in current_level_directories:
                self._scan_directory(
                    subdirectory,
                    levels_left - 1,
                    current_index_list,
                    read_label_from_dataset,
                )

        else:
            return

    def _add_all_datasets(
        self, read_label_from_dataset: bool = False, label_column_transform: None | Callable = None
    ) -> None:
        """
        iterate through all provided directories and add all found HDF5 files.

        Args:
            read_label_from_dataset: indicates if single-cell labels are read from file or provided in bulk for the entire dataset
            label_column_transform: Optional function to apply a mathematical transformation to the read labels.
        """
        # ensure that label_column_transform is not set to a value if read_label_from_dataset is False
        if not read_label_from_dataset:
            assert label_column_transform is None, (
                "label_column_transform should be None if read_label_from_dataset is False"
            )
        self.label_column_transform = label_column_transform

        # scan all directories provided
        for i, directory in enumerate(self.dir_list):
            path = os.path.abspath(directory)
            current_index_list = self.index_list[i]

            # get current label
            # self._get_dataset_label(i) has not been implemented yet

            # check if "directory" is a path to specific hdf5
            filetype = directory.split(".")[-1]

            if filetype in self.HDF_FILETYPES:
                self._add_dataset(
                    path=directory,
                    current_index_list=current_index_list,
                    id=i,
                    read_label_from_dataset=read_label_from_dataset,
                )

            else:
                # recursively scan for files
                self._scan_directory(
                    path,
                    self.max_level,
                    current_index_list=current_index_list,
                )

    def stats(self, detailed: bool = False):
        """Print dataset statistics.

        Args:
            detailed: Whether to print detailed statistics. Default is False.
        """
        labels = [el[0] for el in self.data_locator]

        print(f"Total single cell records: {len(labels)}")

        if detailed:
            for label in set(labels):
                print(f"single cell records with label {label} : {labels.count(label)}")

    def __len__(self) -> int:
        """Return the number of indexed cells."""
        return len(self.data_locator)

    def __getitem__(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Return the item at `idx`."""
        self._ensure_open()

        if torch.is_tensor(idx):
            idx = idx.tolist()  # convert tensor to list

        # get the label, filename and directory for the current dataset
        # data_info always consists of a list of the following information
        # [label, dataset id, index location of single cell, cell_id]

        data_info = self.data_locator[idx]
        label, dataset_id, index_loc, cell_id = data_info

        path = self.paths[dataset_id]
        sc_data = self.get_hdf(path).get("single_cell_data")

        if self.select_channel is not None:
            cell_tensor = sc_data[index_loc, self.select_channel]

            # convert to tensor
            t = torch.from_numpy(cell_tensor)

            if t.ndim == 2:
                # If t is 2D (Y, X), add a channel dimension
                t = torch.unsqueeze(t, 0)

            assert t.ndim == 3, f"Expected 3D tensor, got {t.ndim}D tensor"  # add check to ensure 3D tensor

        else:
            cell_tensor = sc_data[index_loc]

            # convert to tensor
            t = torch.from_numpy(cell_tensor)

            assert t.ndim == 3, f"Expected 3D tensor, got {t.ndim}D tensor"  # add check to ensure 3D tensor

        t = t.float()  # convert to float tensor

        if self.transform:
            t = self.transform(t)  # apply transformation

        if self.label_column_transform is not None:
            label = self.label_column_transform(label)

        label = float(label)  # ensure its a numeric dtype

        if self.return_id:
            # return data, label, and cell_id
            id = int(cell_id)  # ensure the cell_id is transformed to an int
            return (
                t,
                torch.tensor(label),
                torch.tensor(id),
            )

        else:
            # return data and label
            return (t, torch.tensor(label))


class HDF5SingleCellDataset(_HDF5SingleCellDataset):
    """Dataset reader for scPortrait single-cell datasets stored in HDF5 files.

    Examples:
        .. code-block:: python

            dataset = HDF5SingleCellDataset(
                dir_list=["path/to/data1.hdf5", "path/to/data2.hdf5"],
                dir_labels=[0, 1],
                return_id=True,
            )
            image, label, cell_id = dataset[0]
    """

    def __init__(
        self,
        dir_list: list[str],
        dir_labels: list[int],
        index_list: list[list[int]] | None = None,  # list of indices to select from the index
        transform=None,
        max_level: int = 5,
        return_id: bool = True,
        select_channel: int | list[int] | None = None,
    ):
        """Initialize a dataset with per-file labels stored outside the HDF5 payload.

        Args:
            dir_list: Paths to HDF5 files or directories containing HDF5 files.
            dir_labels: Bulk labels applied to all cells from each entry in `dir_list`.
            index_list: Per-input lists of cell indices to include. If `None`, all cells are used.
            transform: Optional transform applied to each returned tensor.
            max_level: Maximum directory depth to scan when an entry in `dir_list` is a directory.
            return_id: Whether to return the unique cell id together with the cell data.
            select_channel: Channel index or indices to load from each cell image. If `None`, all channels are used.
        """
        super().__init__(
            dir_list=dir_list,
            index_list=index_list,
            select_channel=select_channel,
            transform=transform,
            return_id=return_id,
            max_level=max_level,
        )

        self.bulk_labels: list[int] = dir_labels
        self.read_labels_from_dataset = False

        self._add_all_datasets(read_label_from_dataset=self.read_labels_from_dataset)

        # Use numpy instead of lists of objects
        self.data_locator = np.array(self.data_locator, dtype=np.uint32)

        self.stats()


class LabelledHDF5SingleCellDataset(_HDF5SingleCellDataset):
    """Dataset reader for HDF5 single-cell datasets with labels stored in the file.

    Examples:
        .. code-block:: python

            dataset = LabelledHDF5SingleCellDataset(
                dir_list=["path/to/data.hdf5"],
                label_colum=0,
                label_dtype=float,
            )
            image, label, cell_id = dataset[0]
    """

    def __init__(
        self,
        dir_list: list[str],
        label_colum: int,
        label_dtype: type,
        label_column_transform: Callable | None = None,
        index_list: list[list[int]] | None = None,  # list of indices to select from the index
        transform: Callable | None = None,
        max_level: int = 5,
        return_id: bool = True,
        select_channel: list[int] | None | int = None,
    ):
        """Initialize a dataset that reads per-cell labels from each HDF5 file.

        Args:
            dir_list: Paths to HDF5 files or directories containing HDF5 files.
            label_colum: Column index in `single_cell_index_labelled` used as the label source.
            label_dtype: Type used to convert the loaded labels.
            label_column_transform: Optional transform applied to labels after loading.
            index_list: Per-input lists of cell indices to include. If `None`, all cells are used.
            transform: Optional transform applied to each returned tensor.
            max_level: Maximum directory depth to scan when an entry in `dir_list` is a directory.
            return_id: Whether to return the unique cell id together with the cell data.
            select_channel: Channel index or indices to load from each cell image. If `None`, all channels are used.
        """
        super().__init__(
            dir_list=dir_list,
            index_list=index_list,  # list of indices to select from the index
            select_channel=select_channel,
            transform=transform,
            return_id=return_id,
            max_level=max_level,
        )

        self.label_column = label_colum
        self.dtype_label_column = label_dtype
        self.label_column_transform = label_column_transform
        self.read_labels_from_dataset = True

        self._add_all_datasets(read_label_from_dataset=self.read_labels_from_dataset)
        self.stats()


class _H5ScSingleCellDataset(Dataset):
    """Base class for loading scPortrait single-cell datasets stored in AnnData-backed HDF5 files."""

    HDF_FILETYPES = ["h5sc", "h5ad"]  # supported filetypes

    IMAGE_DATACONTAINTER_NAME = IMAGE_DATACONTAINER_NAME
    INDEX_DATACONTAINER_NAME = INDEX_DATACONTAINER_NAME
    _CLOSED_MESSAGE = "Dataset has been closed and cannot be reused."

    def __init__(
        self,
        dir_list: list[str],
        index_list: list[list[int]] | None = None,
        select_channel: list[int] | int | None = None,
        transform=None,
        return_id: bool = True,
        max_level: int = 5,
    ):
        """Initialize the dataset index and runtime state.

        Args:
            dir_list: Paths to H5SC/H5AD files or directories containing those files.
            index_list: Per-input lists of cell indices to include. If `None`, all cells are used.
            select_channel: Channel index or indices to load from each cell image. If `None`, all channels are used.
            transform: Optional transform applied to each returned tensor.
            return_id: Whether to return the unique cell id together with the cell data.
            max_level: Maximum directory depth to scan when an entry in `dir_list` is a directory.
        """
        self.dir_list = dir_list

        if isinstance(self.dir_list[0], PosixPath):
            self.dir_list = [str(x) for x in self.dir_list]

        # ensure select_channel is always a list
        if isinstance(select_channel, int):
            select_channel = [select_channel]

        self.select_channel = select_channel
        self.transform = transform
        self.return_id = return_id
        self.max_level = max_level

        # ensure index list is long enough for all directories
        if index_list is None:
            index_list = [[None]] * len(dir_list)
        else:
            if len(index_list) < len(dir_list):
                raise ValueError("index_list should be as long as dir_list")

            # type check index list to make sure its correctly constructed
            assert _check_type_input_list(index_list), (
                "The parameter index_list expects the following format [list_dataset1, list_dataset2, ...]. Please ensure that you provide an index list for each file listed in dir_list."
            )

        self.index_list: list[list[Any]] = index_list

        # check reading of an image element and ensure that the selected channels are valid

        # initialize placeholders to store dataset information
        self.handle_list: list[Any] = []
        self._open_hdf: list[h5py.File] = []
        self.data_locator: list[list[int]] = []
        self._closed = False

        self.bulk_labels: list[int] | None = None
        self.label_column: str | None = None

    def _ensure_open(self) -> None:
        """Raise if the dataset has been explicitly closed."""
        if self._closed:
            raise RuntimeError(self._CLOSED_MESSAGE)

    def close(self) -> None:
        """Close open HDF5 handles and permanently disable further data access."""
        if self._closed:
            return

        for file in self._open_hdf:
            try:
                file.close()
            except (OSError, RuntimeError, ValueError):
                continue
        self._open_hdf.clear()
        self.handle_list.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass

    def _add_hdf_to_index(
        self,
        path: str,
        index_list: list[int] | None = None,
        label: int | None = None,
        label_column: str | None = None,
        label_column_transform=None,
        read_label: bool = False,
    ):
        """Add single-cell data from `path` to the dataset index."""

        # check to ensure that HDF5 file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found. Please ensure that the file exists.")

        input_hdf = None
        keep_file_open = False
        try:
            # connect to h5py file
            input_hdf = h5py.File(path, "r")

            # ensure that the file is encoded with anndata
            assert input_hdf.attrs["encoding-type"] == "anndata", f"File {path} is not an anndata file. "

            # get single cell index handle
            if index_list != [None]:
                index_handle = np.zeros((len(index_list),), dtype=np.int64)
                cell_id_handle = np.zeros((len(index_list),), dtype=np.uint64)

                # ensure that no out of bound elements are provided for the dataset
                max_elements = input_hdf.get(self.IMAGE_DATACONTAINTER_NAME).shape[0]
                max_index = max(index_list)

                assert max_index < max_elements, (
                    f"Index {max_index} is out of bounds for file {path}. Only {max_elements} single cell records available in dataset."
                )

                for i, ix in enumerate(index_list):
                    index_handle[i] = ix
                    cell_id_handle[i] = input_hdf.get(self.INDEX_DATACONTAINER_NAME)[ix]

            else:
                max_elements = input_hdf.get(self.IMAGE_DATACONTAINTER_NAME).shape[0]
                index_handle = np.arange(max_elements)
                cell_id_handle = input_hdf.get(self.INDEX_DATACONTAINER_NAME)

            # ensure that selected channels are within range
            if self.select_channel is not None:
                max_channels = input_hdf.get(self.IMAGE_DATACONTAINTER_NAME).shape[1]
                assert np.all([channel_ix < max_channels for channel_ix in self.select_channel]), (
                    f"Selected channels are out of bounds. Maximum available channelid is {max_channels}."
                )

            # prepare entries for new dataset
            handle_id = len(self.handle_list)
            new_locator_entries: list[list[Any]] = []

            # add single-cell labelling
            if read_label:
                assert label_column is not None, "Label column must be provided if read_label is set to True."

                # get the column containing the labelling
                if index_list != [None]:
                    label_col = input_hdf.get(f"obs/{label_column}")[index_list]
                else:
                    label_col = input_hdf.get(f"obs/{label_column}")[:]

                # apply any mathematical transform to label column if specified (e.g. to change scale by dividing by 1000)
                if label_column_transform is not None:
                    label_col = label_column_transform(label_col)

                # generate identifiers for all single-cells
                # iterate over rows in index handle, i.e. over all cells
                for current_target, index, cell_id in zip(label_col, index_handle, cell_id_handle, strict=True):
                    # append target, handle id, and row to data locator
                    new_locator_entries.append([current_target, handle_id, index, cell_id])

            else:
                assert label is not None, "Label must be provided if read_label is set to False."

                # generate identifiers for all single-cells
                for index, cell_id in zip(index_handle, cell_id_handle, strict=True):
                    new_locator_entries.append([label, handle_id, index, cell_id])

            self._open_hdf.append(input_hdf)
            keep_file_open = True
            self.handle_list.append(
                input_hdf.get(self.IMAGE_DATACONTAINTER_NAME)
            )  # add new dataset to list of datasets
            self.data_locator.extend(new_locator_entries)

        except (FileNotFoundError, KeyError, OSError) as e:
            print(f"Error: {e}")
            return
        finally:
            if input_hdf is not None and not keep_file_open:
                input_hdf.close()

    def _add_dataset(
        self,
        path: str,
        current_index_list: list[int],
        id: int,
        read_label_from_dataset: bool,
    ):
        """Adds a dataset to the index."""
        if read_label_from_dataset:
            assert self.label_column is not None, (
                "trying to read labels from dataset but no column to access information has been passed"
            )
            self._add_hdf_to_index(
                path=path,
                index_list=current_index_list,
                label=None,
                label_column=self.label_column,
                label_column_transform=self.label_column_transform,
                read_label=True,
            )
        else:
            assert self.bulk_labels is not None, (
                "trying to apply bulk labels to all cells from dataset but no label provided"
            )

            self._add_hdf_to_index(
                path=path,
                index_list=current_index_list,
                label=self.bulk_labels[id],
                label_column=None,
                label_column_transform=None,
                read_label=False,
            )

    def _scan_directory(
        self,
        path: str,
        levels_left: int,
        current_index_list: list[int] | None = None,
        read_label_from_dataset: bool = False,
    ) -> None:
        """Recursively scan `path` and add matching HDF5 files to the dataset index.

        Args:
            path: Directory to search for HDF5 files.
            levels_left: Remaining directory depth to scan recursively.
            current_index_list: Indices to select from the dataset, if any.
            read_label_from_dataset: Whether labels should be read from each dataset file.
        """

        # iterates over all files and folders in a directory
        # hdf5 files are added to the index
        # subfolders are recursively scanned

        if levels_left > 0:
            # get files and directories at current level

            current_level_directories = [
                os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
            ]

            current_level_files = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]

            for i, file in enumerate(current_level_files):
                filetype = file.split(".")[-1]

                if filetype in self.HDF_FILETYPES:
                    self._add_dataset(
                        path=file,
                        current_index_list=current_index_list,
                        id=i,
                        read_label_from_dataset=read_label_from_dataset,
                    )

            # recursively scan subdirectories
            for subdirectory in current_level_directories:
                self._scan_directory(
                    subdirectory,
                    levels_left - 1,
                    current_index_list,
                    read_label_from_dataset,
                )

        else:
            return

    def _add_all_datasets(
        self, read_label_from_dataset: bool = False, label_column_transform: None | Callable = None
    ) -> None:
        """
        iterate through all provided directories and add all found HDF5 files.

        Args:
            read_label_from_dataset: indicates if single-cell labels are read from file or provided in bulk for the entire dataset
            label_column_transform: Optional function to apply a mathematical transformation to the read labels.
        """
        # ensure that label_column_transform is not set to a value if read_label_from_dataset is False
        if not read_label_from_dataset:
            assert label_column_transform is None, (
                "label_column_transform should be None if read_label_from_dataset is False"
            )
        self.label_column_transform = label_column_transform

        # scan all directories provided
        for i, directory in enumerate(self.dir_list):
            path = os.path.abspath(directory)
            current_index_list = self.index_list[i]

            # get current label
            # self._get_dataset_label(i) has not been implemented yet

            # check if "directory" is a path to specific hdf5
            filetype = directory.split(".")[-1]

            if filetype in self.HDF_FILETYPES:
                self._add_dataset(
                    path=directory,
                    current_index_list=current_index_list,
                    id=i,
                    read_label_from_dataset=read_label_from_dataset,
                )

            else:
                # recursively scan for files
                self._scan_directory(
                    path,
                    self.max_level,
                    current_index_list=current_index_list,
                )

    def stats(self, detailed: bool = False):
        """Print dataset statistics.

        Args:
            detailed: Whether to print detailed statistics. Default is False.
        """
        labels = [el[0] for el in self.data_locator]

        print(f"Total single cell records: {len(labels)}")

        if detailed:
            for label in set(labels):
                print(f"single cell records with label {label} : {labels.count(label)}")

    def __len__(self) -> int:
        """Return the number of indexed cells."""
        return len(self.data_locator)

    def __getitem__(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Return the item at `idx`."""
        self._ensure_open()

        if torch.is_tensor(idx):
            idx = idx.tolist()  # convert tensor to list

        # get the label, filename and directory for the current dataset
        # data_info always consists of a list of the following information
        # [label, dataset id, index location of single cell, cell_id]

        data_info = self.data_locator[idx]
        label, dataset_id, index_loc, cell_id = data_info

        if self.select_channel is not None:
            cell_tensor = self.handle_list[dataset_id][index_loc, self.select_channel]

            # convert to tensor
            t = torch.from_numpy(cell_tensor)

            if t.ndim == 2:
                # If t is 2D (Y, X), add a channel dimension
                t = torch.unsqueeze(t, 0)

            assert t.ndim == 3, f"Expected 3D tensor, got {t.ndim}D tensor"  # add check to ensure 3D tensor

        else:
            cell_tensor = self.handle_list[dataset_id][index_loc]

            # convert to tensor
            t = torch.from_numpy(cell_tensor)

            assert t.ndim == 3, f"Expected 3D tensor, got {t.ndim}D tensor"  # add check to ensure 3D tensor

        t = t.float()  # convert to float tensor

        if self.transform:
            t = self.transform(t)  # apply transformation

        if self.label_column_transform is not None:
            label = self.label_column_transform(label)
        label = float(label)  # ensure its a numeric dtype

        if self.return_id:
            # return data, label, and cell_id
            id = int(cell_id)  # ensure the cell_id is transformed to an int
            return (
                t,
                torch.tensor(label),
                torch.tensor(id),
            )

        else:
            # return data and label
            return (t, torch.tensor(label))


class H5ScSingleCellDataset(_H5ScSingleCellDataset):
    """Dataset reader for scPortrait single-cell datasets stored in H5SC or H5AD files.

    Examples:
        .. code-block:: python

            dataset = H5ScSingleCellDataset(
                dir_list=["path/to/data.h5sc"],
                dir_labels=[1],
                return_id=True,
            )
            image, label, cell_id = dataset[0]
    """

    def __init__(
        self,
        dir_list: list[str],
        dir_labels: list[int],
        index_list: list[list[int]] | None = None,  # list of indices to select from the index
        transform=None,
        max_level: int = 5,
        return_id: bool = True,
        select_channel: int | list[int] | None = None,
    ):
        """Initialize a dataset with per-file labels stored outside the H5SC payload.

        Args:
            dir_list: Paths to H5SC/H5AD files or directories containing those files.
            dir_labels: Bulk labels applied to all cells from each entry in `dir_list`.
            index_list: Per-input lists of cell indices to include. If `None`, all cells are used.
            transform: Optional transform applied to each returned tensor.
            max_level: Maximum directory depth to scan when an entry in `dir_list` is a directory.
            return_id: Whether to return the unique cell id together with the cell data.
            select_channel: Channel index or indices to load from each cell image. If `None`, all channels are used.
        """
        super().__init__(
            dir_list=dir_list,
            index_list=index_list,
            select_channel=select_channel,
            transform=transform,
            return_id=return_id,
            max_level=max_level,
        )

        self.bulk_labels: list[int] = dir_labels
        self.read_labels_from_dataset = False

        self._add_all_datasets(read_label_from_dataset=self.read_labels_from_dataset)
        self.stats()


class LabelledH5ScSingleCellDataset(_H5ScSingleCellDataset):
    """Dataset reader for H5SC/H5AD single-cell datasets with labels stored in the file.

    Examples:
        .. code-block:: python

            dataset = LabelledH5ScSingleCellDataset(
                dir_list=["path/to/data.h5sc"],
                label_colum="cell_type",
            )
            image, label, cell_id = dataset[0]
    """

    def __init__(
        self,
        dir_list: list[str],
        label_colum: str,
        label_column_transform: Callable | None = None,
        index_list: list[list[int]] | None = None,  # list of indices to select from the index
        transform: Callable | None = None,
        max_level: int = 5,
        return_id: bool = True,
        select_channel: list[int] | None | int = None,
    ):
        """Initialize a dataset that reads per-cell labels from each H5SC or H5AD file.

        Args:
            dir_list: Paths to H5SC/H5AD files or directories containing those files.
            label_colum: Column name in `obs` used as the label source.
            label_column_transform: Optional transform applied to labels after loading.
            index_list: Per-input lists of cell indices to include. If `None`, all cells are used.
            transform: Optional transform applied to each returned tensor.
            max_level: Maximum directory depth to scan when an entry in `dir_list` is a directory.
            return_id: Whether to return the unique cell id together with the cell data.
            select_channel: Channel index or indices to load from each cell image. If `None`, all channels are used.
        """
        super().__init__(
            dir_list=dir_list,
            index_list=index_list,  # list of indices to select from the index
            select_channel=select_channel,
            transform=transform,
            return_id=return_id,
            max_level=max_level,
        )

        self.label_column = label_colum
        self.label_column_transform = label_column_transform
        self.read_labels_from_dataset = True

        self._add_all_datasets(read_label_from_dataset=self.read_labels_from_dataset)
        self.stats()
