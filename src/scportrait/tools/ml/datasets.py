import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

#type checking functions
from typing import List, Union
from collections.abc import Iterable

def _check_type_input_list(var):
    return (isinstance(var, Iterable) and
            all(isinstance(sublist, Iterable) and all(isinstance(item, int) for item in sublist)
                for sublist in var))

class HDF5SingleCellDataset(Dataset):
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
        List of labels corresponding to the directories in dir_list.
    index_list : list of int, or None
        List of indices to select from the dataset. If set to None all cells are taken. Default is None.
    transform : callable, optional
        A optional user-defined function to apply transformations to the data. Default is None.
    max_level : int, optional
        Maximum levels of directory to search for hdf5 files. Default is 5.
    return_id : bool, optional
        Whether to return the index of the cell with the data. Default is False.
    return_fake_id : bool, optional
        Whether to return a fake index (0) with the data. Default is False.
    select_channel : int, optional
        Specify a specific channel to select from the data. Default is None, which returns all channels.

    Methods
    -------
    add_hdf_to_index(current_label, path)
        Adds single cell data from the hdf5 file located at `path` with the specified `current_label` to the index.
    stats()
        Prints dataset statistics including total count and count per label.
    len()
        Returns the total number of single cells in the dataset.
    getitem(idx)
        Returns the data, label, and optional id/fake_id of the single cell specified by the index `idx`.

    Examples
    --------
    >>> hdf5_data = HDF5SingleCellDataset(
    ...     dir_list=['path/to/data/data1.hdf5', 'path/to/data2/data2.hdf5'],
    ...     dir_labels=[0, 1],
    ...     transform=None,
    ...     return_id=True
    ... )
    >>> len(hdf5_data)
    2000
    >>> sample = hdf5_data[0]
    >>> sample[0].shape
    torch.Size([1, 128, 128])
    >>> sample[1]
    tensor(0)
    >>> sample[2]
    tensor(0)
    """

    HDF_FILETYPES = ["hdf", "hf", "h5", "hdf5"]  # supported hdf5 filetypes

    def __init__(
        self,
        dir_list,
        dir_labels,
        index_list=None, # list of indices to select from the index
        transform=None,
        max_level=5,

        return_id=False,
        return_fake_id=False,
        select_channel=None,
    ):
        """
        Parameters
        ----------
        dir_list : list of str
            List of path(s) where the hdf5 files are stored. Supports specifying a path to a specific hdf5 file or directory
            containing hdf5 files.
        dir_labels : list of int
            List of labels corresponding to the directories in dir_list.
        max_level : int, optional
            Maximum levels of directory to search for hdf5 files in the passed paths. Default is 5.
        transform : callable, optional
            A optional user-defined function to apply transformations to the data. Default is None.
        return_id : bool, optional
            Whether to return the index of the cell with the data. Default is False.
        return_fake_id : bool, optional
            Whether to return a fake index (0) with the data. Default is False.
        index_list : list of int, or None
            List of indices to select from the dataset. If set to None all cells are taken. Default is None.
        select_channel : int, optional
            Specify a specific channel to select from the data. Default is None, which returns all channels.
        """

        self.dir_labels = dir_labels
        self.dir_list = dir_list

        # ensure index list is long enough for all directories
        if index_list is None:
            index_list = [None] * len(dir_list)
        else:
            if len(index_list) < len(dir_list):
                raise ValueError("index_list should be as long as dir_list")
        
            #type check index list to make sure its correctly constructed
            assert (_check_type_input_list(index_list)), "The parameter index_list expects the following format [list_dataset1, list_dataset2, ...]. Please ensure that you provide an index list for each file listed in dir_list."

        self.index_list = index_list
        self.transform = transform

        self.handle_list = []
        self.data_locator = []

        self.select_channel = select_channel

        # scan all directories
        for i, directory in enumerate(dir_list):
            path = os.path.abspath(directory)
            current_label = self.dir_labels[i]
            current_index_list = self.index_list[i]

            # check if "directory" is a path to specific hdf5
            filetype = directory.split(".")[-1]

            if filetype in self.HDF_FILETYPES:
                self.add_hdf_to_index(
                    current_label, directory, current_index_list=current_index_list
                )
            else:
                # recursively scan for files
                self.scan_directory(
                    path,
                    current_label,
                    max_level,
                    current_index_list=current_index_list,
                )

        # print dataset stats at the end
        self.return_id = return_id
        self.return_fake_id = return_fake_id
        self.stats()

    def add_hdf_to_index(self, 
                          current_label: int, 
                          path: str, 
                          current_index_list: Union[List, None] = None):
        """
        Adds single cell data from the hdf5 file located at `path` with the specified `current_label` to the index.

        Parameters
        ----------
        current_label : int
            label which should be added to the dataset
        path : str
            path where the dataset that should be added is located.
        current_index_list : [int] | None 
            list of indices to select from the dataset. If set to None all cells are taken. Default is None.
        """
        try:
            input_hdf = h5py.File(path, "r")

            if current_index_list is not None:
                index_handle = np.zeros((len(current_index_list), 2), dtype=np.int64)
                for i, ix in enumerate(current_index_list):
                    index_handle[i] = input_hdf.get("single_cell_index")[ix]
            
            else:
                index_handle = input_hdf.get("single_cell_index")

            handle_id = len(self.handle_list)
            self.handle_list.append(input_hdf.get("single_cell_data"))

            for row in index_handle:
                self.data_locator.append([current_label, handle_id] + list(row))

        except Exception:
            return

    def _scan_directory(self, 
                        path:str, 
                        current_label:int, 
                        levels_left:int, 
                        current_index_list: Union[List, None]=None):
        """
        iterates over all files and folders in the directory provided by path and adds all found hdf5 files to the index.
        Subfolders are recursively scanned.

        Parameters
        ----------
        path : str
            directory that should be searched for HDF5 files
        current_label : int
            label that should be attached to any found HDF5 files
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
                os.path.join(path, name)
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]

            current_level_files = [
                name
                for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name))
            ]

            for i, file in enumerate(current_level_files):
                filetype = file.split(".")[-1]

                if filetype in self.HDF_FILETYPES:
                    self.add_hdf_to_index(
                        current_label,
                        os.path.join(path, file),
                        current_index_list=current_index_list,
                    )

            # recursively scan subdirectories
            for subdirectory in current_level_directories:
                self._scan_directory(subdirectory, current_label, levels_left - 1)

        else:
            return

    def stats(self):  # print dataset statistics
        """Print the dataset statistics."""
        labels = [el[0] for el in self.data_locator]

        print("Total: {}".format(len(labels)))

        for label in set(labels):
            print("{}: {}".format(label, labels.count(label)))

    def __len__(self):
        return len(self.data_locator)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() # convert tensor to list
        
        # get the label, filename and directory for the current dataset
        data_info = self.data_locator[idx]

        if self.select_channel is not None:
            cell_tensor = self.handle_list[data_info[1]][
                data_info[2], self.select_channel
            ]

            #convert to tensor
            t = torch.from_numpy(cell_tensor)
            
            if t.ndim == 2:
            # If t is 2D (Y, X), add a channel dimension
                t = torch.unsqueeze(t, 0)
            
            assert t.ndim == 3, f"Expected 3D tensor, got {t.ndim}D tensor" #add check to ensure 3D tensor

        else:
            cell_tensor = self.handle_list[data_info[1]][data_info[2]]

            #convert to tensor
            t = torch.from_numpy(cell_tensor)
            
            assert t.ndim == 3, f"Expected 3D tensor, got {t.ndim}D tensor" #add check to ensure 3D tensor

        t = t.float()  # convert to float tensor

        if self.transform:
            t = self.transform(t)  # apply transformation
        """  
        if not list(t.shape) == list(torch.Size([1,128,128])):
            t = torch.zeros((1,128,128))
        """
        if self.return_id and self.return_fake_id:
            raise ValueError("either return_id or return_fake_id should be set")

        if self.return_id:
            ids = int(data_info[3])
            sample = (
                t,
                torch.tensor(data_info[0]),
                torch.tensor(ids),
            )  # return data, label, and id
        elif self.return_fake_id:
            sample = (
                t,
                torch.tensor(data_info[0]),
                torch.tensor(0),
            )  # return data, label, and fake id
        else:
            sample = (t, torch.tensor(data_info[0]))  # return data and label

        return sample


class HDF5SingleCellDatasetRegression(Dataset):
    """
    Class for handling scPortrait single cell datasets stored in HDF5 files where the label should be read from the dataset itself.
    should be read from the dataset itself instead of being provided as an additional parameter.
    """

    HDF_FILETYPES = ["hdf", "hf", "h5", "hdf5"]  # supported hdf5 filetypes

    def __init__(
        self,
        dir_list: list[str],
        target_col: list[int],
        hours: False,
        max_level: int = 5,
        transform=None,
        return_id: bool = False,
        return_fake_id: bool = False,
        select_channel=None,
    ):
        self.dir_list = dir_list  # list of directories with hdf5 files
        self.target_col = target_col  # list of indices for target columns, maps 1 to 1 with dir_list, i.e. target_col[i] is the target column for dir_list[i]
        self.hours = hours  # convert target to hours
        self.transform = transform
        self.select_channel = select_channel

        self.handle_list = []
        self.data_locator = []

        # scan all directories in dir_list
        for i, directory in enumerate(dir_list):
            path = os.path.abspath(directory)  # get full path
            target_col = self.target_col[
                i
            ]  # get the target column for the current directory
            filetype = directory.split(".")[-1]  # get filetype

            if filetype in self.HDF_FILETYPES:  # check if filetype is supported
                self.add_hdf_to_index(path, target_col)  # add hdf5 files to index
            else:
                self._scan_directory(
                    path, target_col, max_level
                )  # recursively scan for files

        self.return_id = return_id
        self.return_fake_id = return_fake_id
        self.stats()

    def add_hdf_to_index(self, path, target_col):
        try:
            input_hdf = h5py.File(path, "r")  # read hdf5 file
            index_handle = input_hdf.get(
                "single_cell_index"
            )  # get single cell index handle

            current_target_col = input_hdf.get("single_cell_index_labelled").asstr()[
                :, target_col
            ]  # get target column
            current_target_col[current_target_col == ""] = (
                np.nan
            )  # replace empty values with nan
            current_target_col = current_target_col.astype(
                float
            )  # convert to float for regression

            handle_id = len(self.handle_list)  # get handle id
            self.handle_list.append(
                input_hdf.get("single_cell_data")
            )  # append data handle (i.e. extracted images)

            for current_target, row in zip(
                current_target_col, index_handle
            ):  # iterate over rows in index handle, i.e. over all cells
                if self.hours:
                    current_target = current_target / 3600  # convert seconds to hours
                self.data_locator.append(
                    [current_target, handle_id] + list(row)
                )  # append target, handle id, and row to data locator
        except Exception:
            return

    def _scan_directory(self, path, target_col, levels_left):
        if (
            levels_left > 0
        ):  # iterate over all files and folders in a directory if levels_left > 0
            current_level_directories = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]  # get directories
            current_level_files = [
                name
                for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name))
            ]  # get files

            for i, file in enumerate(
                current_level_files
            ):  # iterate over files from current level
                filetype = file.split(".")[-1]  # get filetypes

                if filetype in self.HDF_FILETYPES:
                    self.add_hdf_to_index(
                        os.path.join(path, file), target_col
                    )  # add hdf5 files to index if filetype is supported

            for (
                subdirectory
            ) in current_level_directories:  # recursively scan subdirectories
                self._scan_directory(subdirectory, target_col, levels_left - 1)
        else:
            return

    def stats(self):
        targets = [
            info[0] for info in self.data_locator
        ]  # get all targets from data locator
        targets = np.array(targets, dtype=float)  # convert to numpy array

        print(f"Total samples: {len(targets)}")

    def __len__(self):
        return len(self.data_locator)  # return length of data locator

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # convert tensor to list

        data_item = self.data_locator[
            idx
        ]  # get the data info for the current index, such as target, handle id, and row

        if self.select_channel is not None:  # select a specific channel
            cell_tensor = self.handle_list[data_item[1]][
                data_item[2], self.select_channel
            ]
            t = torch.from_numpy(cell_tensor).float()  # convert to float tensor
            t = torch.unsqueeze(t, 0)  # add channel dimension to tensor
        else:
            cell_tensor = self.handle_list[data_item[1]][data_item[2]]
            t = torch.from_numpy(cell_tensor).float()  # convert to float tensor

        if self.transform:
            t = self.transform(t)  # apply transformation to the data

        target = torch.tensor(data_item[0], dtype=torch.float)  # get target value

        if self.return_id:
            ids = int(data_item[3])
            sample = (t, target, torch.tensor(ids))  # return data, target, and id
        elif self.return_fake_id:
            sample = (t, target, torch.tensor(0))  # return data, target, and fake id
        else:
            sample = (t, target)  # return data and target

        return sample


class HDF5SingleCellDatasetRegressionSubset(Dataset):
    """
    Class for handling scPortrait single cell datasets stored in HDF5 files for regression tasks.
    Supports selecting a subset of the data based on given indices.
    """

    HDF_FILETYPES = ["hdf", "hf", "h5", "hdf5"]  # supported hdf5 filetypes

    def __init__(
        self,
        dir_list: list[str],
        target_col: list[int],
        index_list: list[int],  # list of indices to select from the index
        hours: False,
        max_level: int = 5,
        transform=None,
        return_id: bool = False,
        return_fake_id: bool = False,
        select_channel=None,
    ):
        self.dir_list = dir_list
        self.target_col = target_col
        self.index_list = index_list
        self.index_list = sorted(self.index_list)
        self.hours = hours
        self.transform = transform
        self.select_channel = select_channel

        self.handle_list = []
        self.data_locator = []

        # scan all directories in dir_list
        for i, directory in enumerate(dir_list):
            path = os.path.abspath(directory)  # get full path

            target_col = self.target_col[
                i
            ]  # get the target column for the current directory

            filetype = directory.split(".")[-1]  # get filetype

            if filetype in self.HDF_FILETYPES:  # check if filetype is supported
                self.add_hdf_to_index(path, target_col)  # add hdf5 files to index

            else:
                self._scan_directory(
                    path, target_col, max_level
                )  # recursively scan for files

        self.return_id = return_id  # return id
        self.return_fake_id = return_fake_id  # return fake id
        self.stats()  # print dataset stats at the end

    def add_hdf_to_index(self, path, target_col):
        try:
            input_hdf = h5py.File(path, "r")  # read hdf5 file

            index_handle = input_hdf.get("single_cell_index")[
                self.index_list
            ]  # get single cell index handle

            current_target_col = input_hdf.get("single_cell_index_labelled").asstr()[
                self.index_list, target_col
            ]  # get target column
            current_target_col[current_target_col == ""] = (
                np.nan
            )  # replace empty values with nan
            current_target_col = current_target_col.astype(
                float
            )  # convert to float for regression

            handle_id = len(self.handle_list)  # get handle id
            self.handle_list.append(
                input_hdf.get("single_cell_data")
            )  # append data handle (i.e. extracted images)

            for current_target, row in zip(
                current_target_col, index_handle
            ):  # iterate over rows in index handle, i.e. over all cells
                if self.hours:
                    current_target = current_target / 3600  # convert seconds to hours
                self.data_locator.append(
                    [current_target, handle_id] + list(row)
                )  # append target, handle id, and row to data locator
        except Exception:
            return

    def _scan_directory(self, path, target_col, levels_left):
        if (
            levels_left > 0
        ):  # iterate over all files and folders in a directory if levels_left > 0
            current_level_directories = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]  # get directories
            current_level_files = [
                name
                for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name))
            ]  # get files

            for i, file in enumerate(
                current_level_files
            ):  # iterate over files from current level
                filetype = file.split(".")[-1]  # get filetypes

                if filetype in self.HDF_FILETYPES:
                    self.add_hdf_to_index(
                        os.path.join(path, file), target_col
                    )  # add hdf5 files to index if filetype is supported

            for (
                subdirectory
            ) in current_level_directories:  # recursively scan subdirectories
                self._scan_directory(subdirectory, target_col, levels_left - 1)
        else:
            return

    def stats(self):
        targets = [
            info[0] for info in self.data_locator
        ]  # get all targets from data locator
        targets = np.array(targets, dtype=float)  # convert to numpy array

        print(f"Total samples: {len(targets)}")

    def __len__(self):
        return len(self.data_locator)  # return length of data locator

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # convert tensor to list

        data_item = self.data_locator[
            idx
        ]  # get the data info for the current index, such as target, handle id, and row

        if self.select_channel is not None:  # select a specific channel
            cell_tensor = self.handle_list[data_item[1]][
                data_item[2], self.select_channel
            ]
            t = torch.from_numpy(cell_tensor).float()  # convert to float tensor
            t = torch.unsqueeze(t, 0)  # add channel dimension to tensor
        else:
            cell_tensor = self.handle_list[data_item[1]][data_item[2]]
            t = torch.from_numpy(cell_tensor).float()  # convert to float tensor

        if self.transform:
            t = self.transform(t)  # apply transformation to the data

        target = torch.tensor(data_item[0], dtype=torch.float)  # get target value

        if self.return_id:
            ids = int(data_item[3])
            sample = (t, target, torch.tensor(ids))  # return data, target, and id
        elif self.return_fake_id:
            sample = (t, target, torch.tensor(0))  # return data, target, and fake id
        else:
            sample = (t, target)  # return data and target

        return sample
