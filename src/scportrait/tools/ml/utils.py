from collections.abc import Sequence
from math import floor

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import ConcatDataset, Dataset


def combine_datasets_balanced(
    list_of_datasets: list[Dataset],
    class_labels: list[str | int],
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int | None = None,
    return_residual: bool = False,
    concat_datasets: bool = True,
) -> (
    tuple[ConcatDataset, ConcatDataset, ConcatDataset]
    | tuple[ConcatDataset, ConcatDataset, ConcatDataset, ConcatDataset]
    | tuple[list[Dataset], list[Dataset], list[Dataset]]
    | tuple[list[Dataset], list[Dataset], list[Dataset], list[Dataset]]
):
    """Combine multiple datasets to create a balanced dataset for train, val, and test sets.

    Args:
        list_of_datasets: List of datasets to be combined
        class_labels: List of class labels present in the datasets
        train_per_class: Number of samples per class in the train set
        val_per_class: Number of samples per class in the validation set
        test_per_class: Number of samples per class in the test set
        seed: Random seed for reproducibility
        return_residual: boolean value indicating of left over elements not allocated to one of the datasets should be returned as a seperate dataset
        concat_datasets: if the returned datasets should be concatenated or returned as a list

    Returns:
        Tuple containing:
            - train dataset(s) with balanced samples per class
            - validation dataset(s) with balanced samples per class
            - test dataset(s) with balanced samples per class
            - potentially: residual dataset(s) with all residual samples (unbalanced)

    Raises:
        ValueError: If dataset is too small to be split into requested sizes
    """
    elements = [len(el) for el in list_of_datasets]
    rows = np.arange(len(list_of_datasets))

    # Map labels to 0-based consecutive integers
    unique_labels = sorted(set(class_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indexed_labels = [label_to_index[label] for label in class_labels]

    # create dataset fraction array of len(list_of_datasets)
    # mat = csr_matrix((elements, (rows, class_labels))).toarray()
    mat = csr_matrix((elements, (rows, indexed_labels))).toarray()
    cells_per_class = np.sum(mat, axis=0)
    normalized = mat / cells_per_class
    dataset_fraction = np.sum(normalized, axis=1)

    train_dataset: list[Dataset] = []
    test_dataset: list[Dataset] = []
    val_dataset: list[Dataset] = []
    residual_dataset: list[Dataset] = []

    if np.sum(pd.Series(class_labels).value_counts() > 1) == 0:
        for dataset, label, fraction in zip(list_of_datasets, class_labels, dataset_fraction, strict=False):
            print(dataset, label, fraction)

            train_size = floor(train_per_class)
            test_size = floor(test_per_class)
            val_size = floor(val_per_class)

            residual_size = len(dataset) - train_size - test_size - val_size

            if residual_size < 0:
                raise ValueError(
                    f"Dataset with length {len(dataset)} is too small to be split into "
                    f"test set of size {test_size} and train set of size {train_size} "
                    f"and validation set of size {val_size}. Use smaller sizes."
                )

            if seed is not None:
                print(f"Using seeded generator with seed {seed} to split dataset")
                gen = torch.Generator()
                gen.manual_seed(seed)
                splits = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size], generator=gen
                )
            else:
                splits = torch.utils.data.random_split(dataset, [train_size, test_size, val_size, residual_size])
            train, test, val, residual = splits
            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)
            if return_residual:
                residual_dataset.append(residual)
    else:
        for dataset, fraction in zip(list_of_datasets, dataset_fraction, strict=False):
            train_size = int(np.round(train_per_class * fraction))
            test_size = int(np.round(test_per_class * fraction))
            val_size = int(np.round(val_per_class * fraction))

            residual_size = len(dataset) - train_size - test_size - val_size

            if residual_size < 0:
                raise ValueError(
                    f"Dataset with length {len(dataset)} is too small to be split into "
                    f"test set of size {test_size}, train set of size {train_size}, "
                    f"and validation set of size {val_size}. Use smaller sizes."
                )

            if seed is not None:
                print(f"Using seeded generator with seed {seed} to split dataset")
                gen = torch.Generator()
                gen.manual_seed(seed)
                splits = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size], generator=gen
                )
            else:
                splits = torch.utils.data.random_split(dataset, [train_size, test_size, val_size, residual_size])

            train, test, val, residual = splits
            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)

            if return_residual:
                residual_dataset.append(residual)

    if concat_datasets:
        train_dataset = ConcatDataset(train_dataset)
        val_dataset = ConcatDataset(val_dataset)
        test_dataset = ConcatDataset(test_dataset)
        if return_residual:
            residual_dataset = ConcatDataset(residual_dataset)

    if return_residual:
        return (train_dataset, val_dataset, test_dataset, residual_dataset)
    else:
        return (train_dataset, val_dataset, test_dataset)


def split_dataset_fractions(
    list_of_datasets: list[Dataset],
    train_size: int | None = None,
    test_size: int | None = None,
    val_size: int | None = None,
    fractions: Sequence[float] | None = None,
    seed: int | None = None,
) -> tuple[ConcatDataset, ConcatDataset, ConcatDataset]:
    """Split datasets into train, test, and validation sets based on sizes or fractions.

    Args:
        list_of_datasets: List of datasets to split
        train_size: Number of samples in the train set
        test_size: Number of samples in the test set
        val_size: Number of samples in the validation set
        fractions: List of fractions for train/test/val split, must sum to 1
        seed: Random seed for reproducibility

    Returns:
        Tuple containing:
            - Train dataset
            - Test dataset
            - Validation dataset

    Raises:
        ValueError: If fractions don't sum to 1 or if dataset is too small for requested sizes
    """
    train_dataset: list[Dataset] = []
    test_dataset: list[Dataset] = []
    val_dataset: list[Dataset] = []

    for dataset in list_of_datasets:
        if fractions is not None:
            if sum(fractions) != 1:
                raise ValueError("Provided fractions should sum to 1")

            if seed is not None:
                gen = torch.Generator()
                gen.manual_seed(seed)
                splits = torch.utils.data.random_split(dataset, fractions, generator=gen)
            else:
                splits = torch.utils.data.random_split(dataset, fractions)

            train, test, val = splits
            print(
                f"Dataset {list_of_datasets.index(dataset)}:\n"
                f"Train: {len(train)}\n"
                f"Test: {len(test)}\n"
                f"Validation: {len(val)}"
            )

            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)
        else:
            residual_size = len(dataset) - train_size - test_size - val_size
            if residual_size < 0:
                raise ValueError(f"Dataset with length {len(dataset)} is too small to be split into requested sizes")

            if seed is not None:
                gen = torch.Generator()
                gen.manual_seed(seed)
                splits = torch.utils.data.random_split(
                    dataset, [train_size, test_size, val_size, residual_size], generator=gen
                )
            else:
                splits = torch.utils.data.random_split(dataset, [train_size, test_size, val_size, residual_size])

            train, test, val, _ = splits
            print(
                f"Dataset {list_of_datasets.index(dataset)}:\n"
                f"Train: {len(train)}\n"
                f"Test: {len(test)}\n"
                f"Validation: {len(val)}"
            )

            train_dataset.append(train)
            test_dataset.append(test)
            val_dataset.append(val)

    # Convert to ConcatDataset objects
    train_concat = ConcatDataset(train_dataset)
    test_concat = ConcatDataset(test_dataset)
    val_concat = ConcatDataset(val_dataset)

    print(
        f"Total sizes:\n" f"Train: {len(train_concat)}\n" f"Test: {len(test_concat)}\n" f"Validation: {len(val_concat)}"
    )

    return train_concat, test_concat, val_concat
