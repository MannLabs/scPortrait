#######################################################
# Unit tests for ../tools/ml/datasets.py
#######################################################


import os
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from scportrait.tools.ml.datasets import (
    HDF5SingleCellDataset,
    LabelledHDF5SingleCellDataset,
    _check_type_input_list,
    _HDF5SingleCellDataset,
)  # Adjust import path as needed


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
    with h5py.File(temp_file.name, "w") as f:
        rng = np.random.default_rng()
        f.create_dataset("single_cell_data", data=rng.random((100, 3, 128, 128)))
        f.create_dataset("single_cell_index", data=np.array([[i, i] for i in range(100)]))
        labelled_index = np.char.encode(np.array([[i, i] for i in range(100)]).astype(str))
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("single_cell_index_labelled", data=labelled_index, chunks=None, dtype=dt)
    yield temp_file.name  # Provide the file path to the test
    os.remove(temp_file.name)  # Cleanup after the test


@pytest.fixture
def hdf5_dataset(temp_hdf5_file):
    """Fixture for an HDF5SingleCellDataset instance."""
    return HDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        dir_labels=[0],
        transform=None,
        return_id=True,
    )


@pytest.fixture
def labelled_hdf5_dataset(temp_hdf5_file):
    """Fixture for a LabelledHDF5SingleCellDataset instance."""
    return LabelledHDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        label_colum=0,
        label_dtype=int,
        label_column_transform=None,
        return_id=True,
    )


def test_dataset_initialization(hdf5_dataset):
    """Test dataset initializes correctly."""
    assert isinstance(hdf5_dataset, HDF5SingleCellDataset)
    assert len(hdf5_dataset.dir_list) == 1


def test_dataset_length(hdf5_dataset):
    """Test dataset length matches expected value."""
    assert len(hdf5_dataset) == 100


def test_get_item(hdf5_dataset):
    """Test retrieving an item from the dataset."""
    item = hdf5_dataset[0]
    assert len(item) == 3  # (data, label, id)
    assert isinstance(item[0], torch.Tensor)
    assert item[1].ndim == 0  # scalar
    assert item[2].ndim == 0  # scalar


def test_get_item_out_of_bounds(hdf5_dataset):
    """Test index out of range error handling."""
    with pytest.raises(IndexError):
        _ = hdf5_dataset[200]


def test_get_item_without_id(temp_hdf5_file):
    """Test retrieving an item when `return_id=False`."""
    dataset_no_id = HDF5SingleCellDataset(dir_list=[temp_hdf5_file], dir_labels=[0], return_id=False)
    item = dataset_no_id[0]
    assert len(item) == 2  # (data, label)
    assert isinstance(item[0], torch.Tensor)
    assert item[1].ndim == 0  # scalar


def test_labelled_dataset_initialization(labelled_hdf5_dataset):
    """Test labelled dataset initializes correctly."""
    assert isinstance(labelled_hdf5_dataset, LabelledHDF5SingleCellDataset)
    assert len(labelled_hdf5_dataset.dir_list) == 1


def test_labelled_dataset_length(labelled_hdf5_dataset):
    """Test labelled dataset length matches expected value."""
    assert len(labelled_hdf5_dataset) == 100


def test_labelled_get_item(labelled_hdf5_dataset):
    """Test retrieving an item from the labelled dataset."""
    item = labelled_hdf5_dataset[0]
    assert len(item) == 3  # (data, label, id)
    assert isinstance(item[0], torch.Tensor)
    assert item[1].ndim == 0  # scalar
    assert item[2].ndim == 0  # scalar


@pytest.mark.parametrize(
    "input_list, expected",
    [
        ([[1, 2, 3], [4, 5, 6]], True),
        ([[1, 2, "a"], [4, 5, 6]], False),
    ],
)
def test_check_type_input_list(input_list, expected):
    """Test `_check_type_input_list` with valid and invalid input."""
    assert _check_type_input_list(input_list) == expected


@patch("os.path.exists", return_value=False)
def test_add_hdf_to_index_file_not_found(mock_exists):
    """Test `_add_hdf_to_index` raises `FileNotFoundError` when file is missing."""
    dataset = _HDF5SingleCellDataset(dir_list=["nonexistent.hdf5"])
    with pytest.raises(FileNotFoundError):
        dataset._add_hdf_to_index("nonexistent.hdf5")


def test_index_list_subset(temp_hdf5_file):
    """Test that only specified indices are loaded via index_list."""
    # Only use first 10 entries
    index_list = [list(range(10))]
    dataset = HDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        dir_labels=[1],
        index_list=index_list,
        return_id=True,
    )
    assert len(dataset) == 10
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item[0], torch.Tensor)
        assert item[1].ndim == 0  # scalar
        assert item[2].ndim == 0  # scalar


def test_labelled_index_list_subset(temp_hdf5_file):
    """Test that only specified indices are loaded for labelled dataset via index_list."""
    index_list = [list(range(5, 15))]  # 10 elements
    dataset = LabelledHDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        label_colum=0,
        label_dtype=int,
        index_list=index_list,
        return_id=True,
    )
    assert len(dataset) == 10
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item[0], torch.Tensor)
        assert item[1].ndim == 0  # scalar
        assert item[2].ndim == 0  # scalar


# ### test h5sc


@pytest.fixture
def temp_h5sc_file():
    """Create a temporary H5SC (AnnData-like) HDF5 file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5sc")
    with h5py.File(temp_file.name, "w") as f:
        rng = np.random.default_rng()
        f.attrs["encoding-type"] = "anndata"
        f.create_dataset("obs/scportrait_cell_id", data=np.arange(100, dtype=np.uint64))  # cell ids
        f.create_dataset("obs/pseudo_label", data=np.arange(100, dtype=np.uint64))  # add a pseudo label
        f.create_dataset("obsm/single_cell_images", data=rng.random((100, 5, 64, 64)))  # single-cell images
    yield temp_file.name
    os.remove(temp_file.name)


from scportrait.tools.ml.datasets import (
    H5ScSingleCellDataset,
    LabelledH5ScSingleCellDataset,
)


def test_h5sc_dataset_initialization(temp_h5sc_file):
    dataset = H5ScSingleCellDataset(dir_list=[temp_h5sc_file], dir_labels=[1])
    assert isinstance(dataset, H5ScSingleCellDataset)
    assert len(dataset) == 100


def test_labelled_h5sc_dataset_initialization(temp_h5sc_file):
    dataset = LabelledH5ScSingleCellDataset(
        dir_list=[temp_h5sc_file], label_colum="pseudo_label", label_column_transform=None
    )
    assert isinstance(dataset, LabelledH5ScSingleCellDataset)
    assert len(dataset) == 100


def test_h5sc_get_item(temp_h5sc_file):
    dataset = H5ScSingleCellDataset(dir_list=[temp_h5sc_file], dir_labels=[1])
    item = dataset[0]
    assert len(item) == 3

    img, label, idx = item
    assert isinstance(img, torch.Tensor)


def test_labelled_h5sc_get_item(temp_h5sc_file):
    dataset = LabelledH5ScSingleCellDataset(
        dir_list=[temp_h5sc_file], label_colum="pseudo_label", label_column_transform=None
    )
    item = dataset[0]
    assert len(item) == 3

    img, label, idx = item
    assert isinstance(img, torch.Tensor)
    assert label.ndim == 0  # scalar
    assert idx.ndim == 0  # scalar


def test_h5sc_index_list_subset(temp_h5sc_file):
    dataset = H5ScSingleCellDataset(
        dir_list=[temp_h5sc_file],
        dir_labels=[0],
        index_list=[[0, 1, 2, 3, 4]],
    )
    assert len(dataset) == 5
    for i in range(len(dataset)):
        item = dataset[i]
        assert len(item) == 3

        img, label, idx = item
        assert isinstance(img, torch.Tensor)
        assert label.ndim == 0  # scalar
        assert idx.ndim == 0  # scalar


def test_labelled_h5sc_index_list_subset(temp_h5sc_file):
    dataset = LabelledH5ScSingleCellDataset(
        dir_list=[temp_h5sc_file],
        label_colum="pseudo_label",
        index_list=[[10, 11, 12]],
        label_column_transform=None,
    )
    assert len(dataset) == 3
    for i in range(len(dataset)):
        item = dataset[i]
        assert len(item) == 3
        img, label, idx = item
        assert isinstance(img, torch.Tensor)
        assert label.ndim == 0  # scalar
        assert idx.ndim == 0  # scalar
