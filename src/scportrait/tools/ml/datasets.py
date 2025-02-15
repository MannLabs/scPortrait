import os
from collections.abc import Callable, Iterable
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _check_type_input_list(var):
    return isinstance(var, Iterable) and all(
        isinstance(sublist, Iterable) and all(isinstance(item, int) for item in sublist) for sublist in var
    )


class _HDF5SingleCellDataset(Dataset):
    """Base class with shared methods for loading scPortrait single cell datasets stored in HDF5 files."""

    HDF_FILETYPES = ["hdf", "hf", "h5", "hdf5"]  # supported hdf5 filetypes

    def __init__(
        self,
        dir_list,
        index_list=None,
        select_channel=None,
        transform=None,
        return_id=False,
        max_level=5,
    ):
        """
        Parameters
        ----------
        dir_list : list of str
            List of path(s) where the hdf5 files are stored. Supports specifying a path to a specific hdf5 file or directory
            containing hdf5 files.
        index_list : list of int, or None
            List of cell indices to select from the dataset. If set to None all cells are taken. Default is None.
        select_channel : int, optional
            Specify a specific channel or selection of channels to select from the data. Default is None, which returns all channels.
            Using this operation is more efficient than if this selection occurs via a passed transform.
        transform : callable, optional
            A optional user-defined function to apply transformations to the data. Default is None.
        return_id : bool, optional
            Whether to return the index of the cell with the data. Default is False.
        max_level : int, optional
            Maximum levels of directory to search for hdf5 files in the passed paths. Default is 5.
        """
        self.dir_list = dir_list
        self.select_channel = select_channel
        self.transform = transform
        self.return_id = return_id
        self.max_level = max_level

        # ensure index list is long enough for all directories
        if index_list is None:
            index_list = [None] * len(dir_list)
        else:
            if len(index_list) < len(dir_list):
                raise ValueError("index_list should be as long as dir_list")

            # type check index list to make sure its correctly constructed
            assert _check_type_input_list(
                index_list
            ), "The parameter index_list expects the following format [list_dataset1, list_dataset2, ...]. Please ensure that you provide an index list for each file listed in dir_list."

        self.index_list = index_list

        # initialize placeholders to store dataset information
        self.handle_list = []
        self.data_locator = []

        self.bulk_labels = None
        self.label_column = None

    def _add_hdf_to_index(
        self,
        path: str,
        index_list: list | None = None,
        label: int | None = None,
        label_column: int | None = None,
        dtype_label_column=float,
        label_column_transform=None,
        read_label: bool = False,
    ):
        """
        Adds single cell data from the hdf5 file located at `path` with the specified `current_label` to the index.
        """

        # check to ensure that HDF5 file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found. Please ensure that the file exists.")

        try:
            # connect to h5py file
            input_hdf = h5py.File(path, "r")

            # get single cell index handle
            if index_list is not None:
                index_handle = np.zeros((len(index_list), 2), dtype=np.int64)

                # ensure that no out of bound elements are provided for the dataset
                max_elements = input_hdf.get("single_cell_index").shape[0]
                max_index = max(index_list)

                assert (
                    max_index < max_elements
                ), f"Index {max_index} is out of bounds for file {path}. Only {max_elements} single cell records available in dataset."

                for i, ix in enumerate(index_list):
                    index_handle[i] = input_hdf.get("single_cell_index")[ix]

            else:
                index_handle = input_hdf.get("single_cell_index")

            # add connection to singe cell datasets
            handle_id = len(self.handle_list)
            self.handle_list.append(input_hdf.get("single_cell_data"))  # add new dataset to list of datasets

            # add single-cell labelling
            if read_label:
                assert label_column is not None, "Label column must be provided if read_label is set to True."

                # get the column containing the labelling
                label_col = input_hdf.get("single_cell_index_labelled").asstr()[:, label_column]

                # dirty fix that we have some old datasets that have empty strings instead of nan
                if len(label_col[label_col == ""]) > 0:
                    Warning(
                        "Empty strings found in label column. Replacing with nan. Please check your single-cell dataset."
                    )
                    label_col[label_col == ""] = np.nan

                # convert labels to desired dtype
                label_col = label_col.astype(dtype_label_column)

                # apply any mathematical transform to label column if specified (e.g. to change scale by dividing by 1000)
                if label_column is not None:
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

    def _add_dataset(
        self,
        path: str,
        current_index_list: list,
        id: int,
        read_label_from_dataset: bool,
    ):
        if read_label_from_dataset:
            assert (
                self.label_column is not None
            ), "trying to read labels from dataset but no column to access information has been passed"
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
            assert (
                self.bulk_labels is not None
            ), "trying to apply bulk labels to all cells from dataset but no label provided"
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
        current_index_list: list | None = None,
        read_label_from_dataset: bool = False,
    ):
        """
        iterates over all files and folders in the directory provided by path and adds all found hdf5 files to the index.
        Subfolders are recursively scanned.

        Parameters
        ----------
        path : str
            directory that should be searched for HDF5 files
        label : int
            label that should be attached to all cells found in any HDF5 files
        label_col : int
            column in the HDF5 file that should be used to read single-cell labels
        levels_left : int
            how many subfolder levels should be recurisively scanned for additional files
        current_index_list : Union[List, None], optional
            List of indices to select from the dataset. If set to None all cells are taken, by default None
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

        Parameters
        ----------
        read_label_from_dataset: bool
            boolean value indicating if single-cell labels are read from file or provided in bulk for the entire dataset
        """
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

    def stats(self):  # print dataset statistics
        """Print dataset statistics."""
        labels = [el[0] for el in self.data_locator]

        print(f"Total: {len(labels)}")

        for label in set(labels):
            print(f"{label}: {labels.count(label)}")

    def __len__(self):
        """get number of elements contained in the dataset"""
        return len(self.data_locator)

    def __getitem__(self, idx):
        "get item from dataset with the specified index `idx`"

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

        if self.return_id:
            # return data, label, and cell_id
            id = int(cell_id)  # ensure the cell_id is transformed to an int
            sample = (
                t,
                torch.tensor(label),
                torch.tensor(id),
            )

        else:
            # return data and label
            sample = (t, torch.tensor(label))

        return sample


class HDF5SingleCellDataset(_HDF5SingleCellDataset):
    """
    Class for handling scPortrait single cell datasets stored in HDF5 files.

    This class provides a convenient interface for scPortrait formatted hdf5 files containing single cell datasets. It supports loading data
    from multiple hdf5 files within specified directories, applying transformations on the data, and returning
    the required information, such as label or id, along with the single cell data.

    Attributes
    ----------
    dir_list : list of str
        List of path(s) where the hdf5 files are stored. Supports specifying a path to a specific hdf5 file or directory
        containing hdf5 files.
    dir_labels : list of int
        List of bulk labels that should be applied to all cells contained within each dataset in dir_list.
    index_list : list of int, or None
        List of indices to select from the dataset. If set to None all cells are taken. Default is None.
    select_channel : int, optional
        Specify a specific channel or selection of channels to select from the data. Default is None, which returns all channels.
        Using this operation is more efficient than if this selection occurs via a passed transform.
    transform : callable, optional
        A optional user-defined function to apply transformations to the data. Default is None.
    return_id : bool, optional
        Whether to return the index of the cell with the data. Default is False.
    max_level : int, optional
        Maximum levels of directory to search for hdf5 files in the passed paths. Default is 5.

    Methods
    -------
    stats()
        Prints dataset statistics including total count and count per label.
    len()
        Returns the total number of single cells in the dataset.
    getitem(idx)
        Returns the data, label, and optional id/fake_id of the single cell specified by the index `idx`.

    Examples
    --------
    >>> hdf5_data = HDF5SingleCellDataset(
    ...     dir_list=["path/to/data/data1.hdf5", "path/to/data2/data2.hdf5"],
    ...     dir_labels=[0, 1],
    ...     transform=None,
    ...     return_id=True,
    ... )
    >>> len(hdf5_data)
    2000
    """

    def __init__(
        self,
        dir_list,
        dir_labels,
        index_list=None,  # list of indices to select from the index
        transform=None,
        max_level=5,
        return_id=False,
        select_channel=None,
    ):
        super().__init__(
            dir_list=dir_list,
            index_list=index_list,
            select_channel=select_channel,
            transform=transform,
            return_id=return_id,
            max_level=max_level,
        )

        self.bulk_labels = dir_labels
        self.read_labels_from_dataset = False

        self._add_all_datasets(read_label_from_dataset=self.read_labels_from_dataset)
        self.stats()


class LabelledHDF5SingleCellDataset(_HDF5SingleCellDataset):
    """
    Class for handling scPortrait single cell datasets stored in HDF5 files.
    Single-cell labels are read directly from the HDF5 file.

    This class provides a convenient interface for scPortrait formatted hdf5 files containing single cell datasets. It supports loading data
    from multiple hdf5 files within specified directories, applying transformations on the data, and returning
    the required information, such as label or id, along with the single cell data.

    Attributes
    ----------
    dir_list : list of str
        List of path(s) where the hdf5 files are stored. Supports specifying a path to a specific hdf5 file or directory
        containing hdf5 files.
    label_colum: int
        index of column from single_cell_index_labelled from which single-cell labels should be read
    label_dtype: dtype | None
        dtype to which the read labels should be converted
    label_column_transform: function| None
        optional function that can define a mathematical transform on the read labels. E.g. if the labels are saved as seconds
        in the HDF5 dataset you can set this value to `lambda x: x/3600` to have the labels returned in hours instead.
    index_list : list of int, or None
        List of indices to select from the dataset. If set to None all cells are taken. Default is None.
    select_channel : int, optional
        Specify a specific channel or selection of channels to select from the data. Default is None, which returns all channels.
        Using this operation is more efficient than if this selection occurs via a passed transform.
    transform : callable, optional
        A optional user-defined function to apply transformations to the data. Default is None.
    return_id : bool, optional
        Whether to return the index of the cell with the data. Default is False.
    max_level : int, optional
        Maximum levels of directory to search for hdf5 files in the passed paths. Default is 5.

    Methods
    -------
    stats()
        Prints dataset statistics including total count and count per label.
    len()
        Returns the total number of single cells in the dataset.
    getitem(idx)
        Returns the data, label, and optional id/fake_id of the single cell specified by the index `idx`.

    Examples
    --------
    >>> hdf5_data = HDF5SingleCellDataset(
    ...     dir_list=["path/to/data/data1.hdf5", "path/to/data2/data2.hdf5"],
    ...     dir_labels=[0, 1],
    ...     transform=None,
    ...     return_id=True,
    ... )
    >>> len(hdf5_data)
    2000
    """

    def __init__(
        self,
        dir_list,
        label_colum,
        label_dtype,
        label_column_transform,
        index_list=None,  # list of indices to select from the index
        transform=None,
        max_level=5,
        return_id=False,
        select_channel=None,
    ):
        super().__init__(
            dir_list=dir_list,
            index_list=index_list,  # list of indices to select from the index
            select_channel=select_channel,
            transform=transform,
            return_id=return_id,
            max_level=max_level,
        )

        self.label_column = label_colum
        self.label_dtype = label_dtype
        self.label_column_transform = label_column_transform
        self.read_labels_from_dataset = True

        self._add_all_dataset(read_label_from_dataset=self.read_labels_from_dataset)
        self.stats()
