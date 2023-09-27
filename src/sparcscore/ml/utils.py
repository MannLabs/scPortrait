from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import torch
from math import floor

def combine_datasets_balanced(list_of_datasets, class_labels, train_per_class, val_per_class, test_per_class,):
    """
    Combine multiple datasets to create a single balanced dataset with a specified number of samples per class for train, validation, and test set.
    A balanced dataset means that from each label source an equal number of data instances are used.

    Parameters
    ----------
    list_of_datasets : list of torch.utils.data.Dataset
        List of datasets to be combined.
    class_labels : list of str or int
        List of class labels present in the datasets.
    train_per_class : int
        Number of samples per class in the train set.
    val_per_class : int
        Number of samples per class in the validation set.
    test_per_class : int
        Number of samples per class in the test set.

    Returns
    -------
    train : torch.utils.data.Dataset
        Combined train dataset with balanced samples per class.
    val : torch.utils.data.Dataset
        Combined validation dataset with balanced samples per class.
    test : torch.utils.data.Dataset
        Combined test dataset with balanced samples per class.
    """
    
    elements = [len(el) for el in list_of_datasets]
    rows = np.arange(len(list_of_datasets))
    
    # create dataset fraction array of len(list_of_datasets)
    mat = csr_matrix((elements, (rows, class_labels))).toarray()
    cells_per_class = np.sum(mat,axis=0)
    normalized = mat / cells_per_class
    dataset_fraction = np.sum(normalized,axis=1)
    print(dataset_fraction)
    
    # Initialize empty lists to store the combined train, validation, and test datasets
    train_dataset = []
    test_dataset = []
    val_dataset = []
    
    #check to make sure we have more than one occurance of a dataset (otherwise it will throw an error)
    if np.sum(pd.Series(class_labels).value_counts() > 1) == 0:
        for dataset, label, fraction in zip(list_of_datasets, class_labels, dataset_fraction):
            print(dataset, label, 1)
            train_size = floor(train_per_class)
            test_size = floor(test_per_class)
            val_size = floor(val_per_class)
            
            residual_size = len(dataset) - train_size - test_size - val_size
            
            if(residual_size < 0):
                raise ValueError(f"Dataset with length {len(dataset)} is to small to be split into test set of size {test_size} and train set of size {train_size} and validation set of size {val_size}. Use a smaller test and trainset.")
            
            train, test, val, _ = torch.utils.data.random_split(dataset, [train_size, test_size, val_size, residual_size])
            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)
    else: 
    
        for dataset, label, fraction in zip(list_of_datasets, class_labels, dataset_fraction):
            print(dataset, label, fraction)
            train_size = floor(train_per_class*fraction)
            test_size = floor(test_per_class*fraction)
            val_size = floor(val_per_class*fraction)
            
            residual_size = len(dataset) - train_size - test_size - val_size
            
            if(residual_size < 0):
                raise ValueError(f"Dataset with length {len(dataset)} is to small to be split into test set of size {test_size} and train set of size {train_size} and validation set of size {val_size}. Use a smaller test and trainset.")
            
            train, test, val, _ = torch.utils.data.random_split(dataset, [train_size, test_size, val_size, residual_size])
            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)
    
    # Convert the combined datasets into torch.utils.data.Dataset objects
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)
    
    return train_dataset, val_dataset, test_dataset