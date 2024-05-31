from torch.utils.data import Dataset
import torch
import os
import h5py
   
class HDF5SingleCellDataset(Dataset):
    """
    Class for handling SPARCSpy single cell datasets stored in HDF5 files.

    This class provides a convenient interface for SPARCSpy formated hdf5 files containing single cell datasets. It supports loading data
    from multiple hdf5 files within specified directories, applying transformations on the data, and returning
    the required information, such as label or id, along with the single cell data.

    Attributes
    ----------
    root_dir : str
        Root directory where the hdf5 files are located.
    dir_labels : list of int
        List of labels corresponding to the directories in dir_list.
    dir_list : list of str
        List of path(s) where the hdf5 files are stored. Supports specifying a path to a specific hdf5 file or directory
        containing hdf5 files.
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
        Adds single cell data from the hdf5 file located at ‘path’ with the specified ‘current_label’ to the index.
    scan_directory(path, current_label, levels_left)
        Scans directories for hdf5 files and adds their data to the index with the specified ‘current_label’.
    stats()
        Prints dataset statistics including total count and count per label.
    len()
        Returns the total number of single cells in the dataset.
    getitem(idx)
        Returns the data, label, and optional id/fake_id of the single cell specified by the index ‘idx’.

    Examples
    --------
    >>> hdf5_data = HDF5SingleCellDataset(
    ...     dir_list=['data1.hdf5', 'data2.hdf5'],
    ...     dir_labels=[0, 1],
    ...     root_dir='/path/to/data',
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
    
    HDF_FILETYPES = ["hdf", "hf", "h5", "hdf5"]
    def __init__(self, dir_list, 
                 dir_labels, 
                 root_dir, 
                 max_level=5, 
                 transform=None, 
                 return_id=False, 
                 return_fake_id=False,
                 select_channel=None):
        
        self.root_dir = root_dir
        self.dir_labels = dir_labels
        self.dir_list = dir_list
        self.transform = transform
        
        self.handle_list = []
        self.data_locator = []
        
        self.select_channel = select_channel
        
        # scan all directoreis
        for i, directory in enumerate(dir_list):
            path = os.path.join(self.root_dir, directory)  
            current_label = self.dir_labels[i]

            #check if "directory" is a path to specific hdf5
            filetype = directory.split(".")[-1]
                
            if filetype in self.HDF_FILETYPES:
                self.add_hdf_to_index(current_label, directory)

            else:
                # recursively scan for files
                self.scan_directory(path, current_label, max_level)
        
        # print dataset stats at the end
        
        self.return_id = return_id
        self.return_fake_id = return_fake_id
        self.stats()
 
        
    def add_hdf_to_index(self, current_label, path):       
        try:
            input_hdf = h5py.File(path, 'r')
            index_handle = input_hdf.get('single_cell_index')

            handle_id = len(self.handle_list)
            self.handle_list.append(input_hdf.get('single_cell_data'))

            for row in index_handle:
                self.data_locator.append([current_label, handle_id]+list(row))      
        except Exception:
            return
        
    def scan_directory(self, path, current_label, levels_left):
        
        # iterates over all files and folders in a directory
        # hdf5 files are added to the index
        # subfolders are recursively scanned
        
        if levels_left > 0:
            
            # get files and directories at current level
            current_level_directories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

            current_level_files = [ name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
                        
            for i, file in enumerate(current_level_files):
                filetype = file.split(".")[-1]
                
                if filetype in self.HDF_FILETYPES:
                    
                    self.add_hdf_to_index(current_label, os.path.join(path, file))
                    
            # recursively scan subdirectories        
            for subdirectory in current_level_directories:
                self.scan_directory(subdirectory, current_label, levels_left-1)
            
        else:
            return
        
    def stats(self):
    
        labels = [el[0] for el in self.data_locator]
        
        print("Total: {}".format(len(labels)))
        
        for label in set(labels):
            print("{}: {}".format(label, labels.count(label)))
        
    def __len__(self):
        return len(self.data_locator)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get the label, filename and directory for the current dataset
        data_info = self.data_locator[idx]
        
        if self.select_channel is not None:
            cell_tensor = self.handle_list[data_info[1]][data_info[2], self.select_channel]
            t = torch.from_numpy(cell_tensor)
            t = torch.unsqueeze(t,0)
            
        else:
            
            cell_tensor = self.handle_list[data_info[1]][data_info[2]]
            t = torch.from_numpy(cell_tensor)
            
        t = t.float()     
        
        if self.transform:
            t = self.transform(t)
        """  
        if not list(t.shape) == list(torch.Size([1,128,128])):
            t = torch.zeros((1,128,128))
        """      
        if self.return_id and self.return_fake_id:
            raise ValueError("either return_id or return_fake_id should be set")
            
        if self.return_id:
            
            ids = int(data_info[3])
            sample = (t, torch.tensor(data_info[0]), torch.tensor(ids))
        elif self.return_fake_id:
            
            sample = (t, torch.tensor(data_info[0]), torch.tensor(0))
        else:
            sample = (t, torch.tensor(data_info[0]))
        
        return sample