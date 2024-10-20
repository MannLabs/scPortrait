from math import floor

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def combine_datasets_balanced(
    list_of_datasets, class_labels, train_per_class, val_per_class, test_per_class, seed=None
):
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
    cells_per_class = np.sum(mat, axis=0)
    normalized = mat / cells_per_class
    dataset_fraction = np.sum(normalized, axis=1)

    # Initialize empty lists to store the combined train, validation, and test datasets
    train_dataset = []
    test_dataset = []
    val_dataset = []

    # check to make sure we have more than one occurance of a dataset (otherwise it will throw an error)
    if np.sum(pd.Series(class_labels).value_counts() > 1) == 0:
        for dataset, label, fraction in zip(list_of_datasets, class_labels, dataset_fraction, strict=False):
            print(dataset, label, fraction)

            train_size = floor(train_per_class)
            test_size = floor(test_per_class)
            val_size = floor(val_per_class)

            residual_size = len(dataset) - train_size - test_size - val_size

            if residual_size < 0:
                raise ValueError(
                    f"Dataset with length {len(dataset)} is to small to be split into test set of size {test_size} and train set of size {train_size} and validation set of size {val_size}. Use a smaller test and trainset."
                )

            if seed is not None:
                print(f"Using seeded generator with seed {seed} to split dataset")
                gen = torch.Generator()
                gen.manual_seed(seed)
                train, test, val, _ = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size], generator=gen
                )
            else:
                train, test, val, _ = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size]
                )
            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)
    else:
        for dataset, fraction in zip(list_of_datasets, dataset_fraction, strict=False):
            train_size = int(np.round(train_per_class * fraction))
            test_size = int(np.round(test_per_class * fraction))
            val_size = int(np.round(val_per_class * fraction))

            residual_size = len(dataset) - train_size - test_size - val_size

            if residual_size < 0:
                raise ValueError(
                    f"Dataset with length {len(dataset)} is too small to be split into test set of size {test_size}, "
                    f"train set of size {train_size}, and validation set of size {val_size}. "
                    f"Use a smaller test and trainset."
                )
            if seed is not None:
                print(f"Using seeded generator with seed {seed} to split dataset")
                gen = torch.Generator()
                gen.manual_seed(seed)
                train, test, val, _ = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size], generator=gen
                )
            else:
                train, test, val, _ = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size]
                )

            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)

    # Convert the combined datasets into torch.utils.data.Dataset objects
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)

    return train_dataset, val_dataset, test_dataset


def split_dataset_fractions(
    list_of_datasets, train_size=None, test_size=None, val_size=None, fractions=None, seed=None
):
    """
    Split a dataset into train, test, and validation set based on the provided fractions or sizes.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to be split.
    train_size : int
        Number of samples in the train set.
    test_size : int
        Number of samples in the test set.
    val_size : int
        Number of samples in the validation set.
    fractions : list of float
        Fractions of the dataset to be used for train, test, and validation set. Should sum up to 1 (100%).
        For example, [0.8, 0.1, 0.1] will split the dataset into 80% train, 10% test, and 10% validation set.


    Returns
    -------
    train : torch.utils.data.Dataset
        Train dataset.
    val : torch.utils.data.Dataset
        Validation dataset.
    test : torch.utils.data.Dataset
        Test dataset.
    """
    train_dataset = []
    test_dataset = []
    val_dataset = []

    for dataset in list_of_datasets:
        if fractions is not None:
            if sum(fractions) != 1:
                raise ValueError("Provided fractions of the dataset should sum up to 1.")

            if seed is not None:
                gen = torch.Generator()
                gen.manual_seed(seed)
                train, test, val = torch.utils.data.random_split(dataset, fractions, generator=gen)
            else:
                train, test, val = torch.utils.data.random_split(dataset, fractions)

            print(
                f"Dataset {list_of_datasets.index(dataset)}:\n"
                f"Train: {len(train)}, \n"
                f"Test: {len(test)}, \n"
                f"Validation: {len(val)}"
            )

            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)

        if fractions is None:
            residual_size = len(dataset) - train_size - test_size - val_size
            if residual_size < 0:
                raise ValueError(
                    f"Dataset with length {len(dataset)} is too small to be split into test set of size {test_size}, "
                    f"train set of size {train_size}, and validation set of size {val_size}. "
                )

            if seed is not None:
                gen = torch.Generator()
                gen.manual_seed(seed)
                train, test, val, _ = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size], generator=gen
                )
            else:
                train, test, val, _ = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size]
                )

            print(
                f"Dataset {list_of_datasets.index(dataset)}:\n"
                f"Train: {len(train)}, \n"
                f"Test: {len(test)}, \n"
                f"Validation: {len(val)}"
            )

            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)

    # Convert the combined datasets into torch.utils.data.Dataset objects
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)

    print(
        f"Total size of the train dataset: {len(train_dataset)}, \n"
        f"Total size of the test dataset: {len(test_dataset)}, \n"
        f"Total size of the validation dataset: {len(val_dataset)}"
    )

    return train_dataset, test_dataset, val_dataset
