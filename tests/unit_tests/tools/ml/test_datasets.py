#######################################################
# Unit tests for ../tools/ml/datasets.py
#######################################################


import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from scportrait.tools.ml.datasets import (
    H5ScSingleCellDataset,
    HDF5SingleCellDataset,
    LabelledH5ScSingleCellDataset,
    LabelledHDF5SingleCellDataset,
    _check_type_input_list,
    _HDF5SingleCellDataset,
)


def _create_hdf5_test_file(path: str) -> None:
    with h5py.File(path, "w") as f:
        rng = np.random.default_rng()
        f.create_dataset("single_cell_data", data=rng.random((100, 3, 128, 128)))
        f.create_dataset("single_cell_index", data=np.array([[i, i] for i in range(100)]))
        labelled_index = np.char.encode(np.array([[i, i] for i in range(100)]).astype(str))
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("single_cell_index_labelled", data=labelled_index, chunks=None, dtype=dt)


def _create_h5sc_test_file(path: str) -> None:
    with h5py.File(path, "w") as f:
        rng = np.random.default_rng()
        f.attrs["encoding-type"] = "anndata"
        f.create_dataset("obs/scportrait_cell_id", data=np.arange(100, dtype=np.uint64))  # cell ids
        f.create_dataset("obs/pseudo_label", data=np.arange(100, dtype=np.uint64))  # add a pseudo label
        f.create_dataset("obsm/single_cell_images", data=rng.random((100, 5, 64, 64)))  # single-cell images


@pytest.fixture
def temp_hdf5_dir():
    """Create a temporary directory containing multiple HDF5 files for testing."""
    temp_dir = tempfile.TemporaryDirectory()
    _create_hdf5_test_file(os.path.join(temp_dir.name, "part_a.hdf5"))
    _create_hdf5_test_file(os.path.join(temp_dir.name, "part_b.hdf5"))
    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
    temp_file.close()
    _create_hdf5_test_file(temp_file.name)
    yield temp_file.name  # Provide the file path to the test
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)  # Cleanup after the test


@pytest.fixture
def hdf5_dataset(temp_hdf5_file):
    """Fixture for an HDF5SingleCellDataset instance."""
    dataset = HDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        dir_labels=[0],
        transform=None,
        return_id=True,
    )
    try:
        yield dataset
    finally:
        dataset.close()


@pytest.fixture
def labelled_hdf5_dataset(temp_hdf5_file):
    """Fixture for a LabelledHDF5SingleCellDataset instance."""
    dataset = LabelledHDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        label_colum=0,
        label_dtype=int,
        label_column_transform=None,
        return_id=True,
    )
    try:
        yield dataset
    finally:
        dataset.close()


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
    with HDF5SingleCellDataset(dir_list=[temp_hdf5_file], dir_labels=[0], return_id=False) as dataset_no_id:
        item = dataset_no_id[0]
        assert len(item) == 2  # (data, label)
        assert isinstance(item[0], torch.Tensor)
        assert item[1].ndim == 0  # scalar


def test_hdf5_dataset_accepts_pathlike_inputs(temp_hdf5_file):
    """Test that HDF5 datasets accept pathlib.Path inputs."""
    with HDF5SingleCellDataset(dir_list=[Path(temp_hdf5_file)], dir_labels=[0], return_id=True) as dataset:
        item = dataset[0]
        assert len(item) == 3


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
    with HDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        dir_labels=[1],
        index_list=index_list,
        return_id=True,
    ) as dataset:
        assert len(dataset) == 10
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item[0], torch.Tensor)
            assert item[1].ndim == 0  # scalar
            assert item[2].ndim == 0  # scalar


def test_labelled_index_list_subset(temp_hdf5_file):
    """Test that only specified indices are loaded for labelled dataset via index_list."""
    index_list = [list(range(5, 15))]  # 10 elements
    with LabelledHDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        label_colum=0,
        label_dtype=int,
        index_list=index_list,
        return_id=True,
    ) as dataset:
        assert len(dataset) == 10
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item[0], torch.Tensor)
            assert item[1].ndim == 0  # scalar
            assert item[2].ndim == 0  # scalar


def test_labelled_hdf5_subset_uses_selected_row_labels(temp_hdf5_file):
    """Test that labelled HDF5 subsets read labels from the selected rows."""
    with LabelledHDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        label_colum=0,
        label_dtype=int,
        index_list=[[5, 6, 7]],
        return_id=True,
    ) as dataset:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
        assert labels == [5, 6, 7]


def test_labelled_hdf5_label_transform_applied_once(temp_hdf5_file):
    """Test that HDF5 label transforms are applied exactly once."""
    with LabelledHDF5SingleCellDataset(
        dir_list=[temp_hdf5_file],
        label_colum=0,
        label_dtype=float,
        label_column_transform=lambda x: x / 10,
        index_list=[[10]],
        return_id=True,
    ) as dataset:
        item = dataset[0]
        assert float(item[1]) == 1.0


def test_hdf5_directory_scan_uses_full_paths_and_directory_label(temp_hdf5_dir):
    """Test that directory inputs load all HDF5 files with the correct shared label."""
    with HDF5SingleCellDataset(dir_list=[temp_hdf5_dir], dir_labels=[7], return_id=True) as dataset:
        assert len(dataset) == 200
        assert all(os.path.dirname(path) == temp_hdf5_dir for path in dataset.paths)
        assert len(dataset.paths) == 2

        first_item = dataset[0]
        last_item = dataset[len(dataset) - 1]
        assert float(first_item[1]) == 7.0
        assert float(last_item[1]) == 7.0


def test_hdf5_close_is_idempotent(temp_hdf5_file):
    """Test that closing an HDF5 dataset is idempotent and terminal."""
    dataset = HDF5SingleCellDataset(dir_list=[temp_hdf5_file], dir_labels=[0], return_id=True)
    _ = dataset[0]
    dataset.close()
    dataset.close()
    assert dataset._closed is True
    assert dataset._open_hdf == {}
    with pytest.raises(RuntimeError, match="Dataset has been closed and cannot be reused."):
        dataset[0]


def test_hdf5_context_manager_releases_file():
    """Test that the HDF5 dataset context manager closes the dataset on exit."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
    temp_file.close()
    _create_hdf5_test_file(temp_file.name)

    try:
        with HDF5SingleCellDataset(dir_list=[temp_file.name], dir_labels=[0], return_id=True) as dataset:
            _ = dataset[0]
        assert dataset._closed is True
        assert dataset._open_hdf == {}
        with pytest.raises(RuntimeError, match="Dataset has been closed and cannot be reused."):
            dataset[0]
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


@pytest.fixture
def temp_h5sc_file():
    """Create a temporary H5SC (AnnData-like) HDF5 file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5sc")
    temp_file.close()
    _create_h5sc_test_file(temp_file.name)
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


@pytest.fixture
def temp_h5sc_dir():
    """Create a temporary directory containing multiple H5SC files for testing."""
    temp_dir = tempfile.TemporaryDirectory()
    _create_h5sc_test_file(os.path.join(temp_dir.name, "part_a.h5sc"))
    _create_h5sc_test_file(os.path.join(temp_dir.name, "part_b.h5sc"))
    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


def test_h5sc_dataset_initialization(temp_h5sc_file):
    """Test H5SC dataset initialization."""
    with H5ScSingleCellDataset(dir_list=[temp_h5sc_file], dir_labels=[1]) as dataset:
        assert isinstance(dataset, H5ScSingleCellDataset)
        assert len(dataset) == 100


def test_labelled_h5sc_dataset_initialization(temp_h5sc_file):
    """Test labelled H5SC dataset initialization."""
    with LabelledH5ScSingleCellDataset(
        dir_list=[temp_h5sc_file], label_colum="pseudo_label", label_column_transform=None
    ) as dataset:
        assert isinstance(dataset, LabelledH5ScSingleCellDataset)
        assert len(dataset) == 100


def test_h5sc_get_item(temp_h5sc_file):
    """Test retrieving an item from an H5SC dataset."""
    with H5ScSingleCellDataset(dir_list=[temp_h5sc_file], dir_labels=[1]) as dataset:
        item = dataset[0]
        assert len(item) == 3

        img, label, idx = item
        assert isinstance(img, torch.Tensor)


def test_h5sc_dataset_accepts_pathlike_inputs(temp_h5sc_file):
    """Test that H5SC datasets accept pathlib.Path inputs."""
    with H5ScSingleCellDataset(dir_list=[Path(temp_h5sc_file)], dir_labels=[1]) as dataset:
        item = dataset[0]
        assert len(item) == 3


def test_labelled_h5sc_get_item(temp_h5sc_file):
    """Test retrieving an item from a labelled H5SC dataset."""
    with LabelledH5ScSingleCellDataset(
        dir_list=[temp_h5sc_file], label_colum="pseudo_label", label_column_transform=None
    ) as dataset:
        item = dataset[0]
        assert len(item) == 3

        img, label, idx = item
        assert isinstance(img, torch.Tensor)
        assert label.ndim == 0  # scalar
        assert idx.ndim == 0  # scalar


def test_h5sc_index_list_subset(temp_h5sc_file):
    """Test that only specified indices are loaded for H5SC datasets."""
    with H5ScSingleCellDataset(
        dir_list=[temp_h5sc_file],
        dir_labels=[0],
        index_list=[[0, 1, 2, 3, 4]],
    ) as dataset:
        assert len(dataset) == 5
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item) == 3

            img, label, idx = item
            assert isinstance(img, torch.Tensor)
            assert label.ndim == 0  # scalar
            assert idx.ndim == 0  # scalar


def test_labelled_h5sc_index_list_subset(temp_h5sc_file):
    """Test that only specified indices are loaded for labelled H5SC datasets."""
    with LabelledH5ScSingleCellDataset(
        dir_list=[temp_h5sc_file],
        label_colum="pseudo_label",
        index_list=[[10, 11, 12]],
        label_column_transform=None,
    ) as dataset:
        assert len(dataset) == 3
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item) == 3
            img, label, idx = item
            assert isinstance(img, torch.Tensor)
            assert label.ndim == 0  # scalar
            assert idx.ndim == 0  # scalar


def test_labelled_h5sc_label_transform_applied_once(temp_h5sc_file):
    """Test that H5SC label transforms are applied exactly once."""
    with LabelledH5ScSingleCellDataset(
        dir_list=[temp_h5sc_file],
        label_colum="pseudo_label",
        label_column_transform=lambda x: x / 10,
        index_list=[[10]],
        return_id=True,
    ) as dataset:
        item = dataset[0]
        assert float(item[1]) == 1.0


def test_h5sc_directory_scan_uses_full_paths_and_directory_label(temp_h5sc_dir):
    """Test that directory inputs load all H5SC files with the correct shared label."""
    with H5ScSingleCellDataset(dir_list=[temp_h5sc_dir], dir_labels=[3], return_id=True) as dataset:
        assert len(dataset) == 200
        assert len(dataset.handle_list) == 2

        first_item = dataset[0]
        last_item = dataset[len(dataset) - 1]
        assert float(first_item[1]) == 3.0
        assert float(last_item[1]) == 3.0


def test_h5sc_close_is_idempotent(temp_h5sc_file):
    """Test that closing an H5SC dataset is idempotent and terminal."""
    dataset = H5ScSingleCellDataset(dir_list=[temp_h5sc_file], dir_labels=[1])
    _ = dataset[0]
    dataset.close()
    dataset.close()
    assert dataset._closed is True
    assert dataset.handle_list == []
    assert dataset._open_hdf == []
    with pytest.raises(RuntimeError, match="Dataset has been closed and cannot be reused."):
        dataset[0]


def test_h5sc_context_manager_releases_file():
    """Test that the H5SC dataset context manager closes the dataset on exit."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5sc")
    temp_file.close()
    _create_h5sc_test_file(temp_file.name)

    try:
        with H5ScSingleCellDataset(dir_list=[temp_file.name], dir_labels=[1]) as dataset:
            _ = dataset[0]
        assert dataset._closed is True
        assert dataset.handle_list == []
        assert dataset._open_hdf == []
        with pytest.raises(RuntimeError, match="Dataset has been closed and cannot be reused."):
            dataset[0]
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
