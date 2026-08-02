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


_BASIC_CASES = [
    pytest.param("temp_hdf5_file", HDF5SingleCellDataset, {"dir_labels": [0]}, id="hdf5"),
    pytest.param("temp_h5sc_file", H5ScSingleCellDataset, {"dir_labels": [1]}, id="h5sc"),
]

_LABELLED_CASES = [
    pytest.param(
        "temp_hdf5_file",
        LabelledHDF5SingleCellDataset,
        {"label_colum": 0, "label_dtype": int},
        id="labelled-hdf5",
    ),
    pytest.param(
        "temp_h5sc_file",
        LabelledH5ScSingleCellDataset,
        {"label_colum": "pseudo_label"},
        id="labelled-h5sc",
    ),
]

_DIRECTORY_CASES = [
    pytest.param("temp_hdf5_dir", HDF5SingleCellDataset, {"dir_labels": [7]}, 7.0, 200, id="hdf5"),
    pytest.param("temp_h5sc_dir", H5ScSingleCellDataset, {"dir_labels": [3]}, 3.0, 200, id="h5sc"),
]


def _assert_item_shapes(item: tuple[torch.Tensor, ...], expected_len: int) -> None:
    assert len(item) == expected_len
    assert isinstance(item[0], torch.Tensor)

    for value in item[1:]:
        assert value.ndim == 0


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _BASIC_CASES)
def test_dataset_initialization(request, fixture_name, dataset_cls, init_kwargs):
    """Test basic datasets initialize correctly."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(dir_list=[dataset_path], return_id=True, **init_kwargs) as dataset:
        assert isinstance(dataset, dataset_cls)
        assert len(dataset.dir_list) == 1
        assert len(dataset) == 100


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _BASIC_CASES)
def test_get_item(request, fixture_name, dataset_cls, init_kwargs):
    """Test retrieving an item from a basic dataset."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(dir_list=[dataset_path], return_id=True, **init_kwargs) as dataset:
        _assert_item_shapes(dataset[0], 3)


def test_get_item_out_of_bounds(temp_hdf5_file):
    """Test index out of range error handling."""
    with HDF5SingleCellDataset(dir_list=[temp_hdf5_file], dir_labels=[0], return_id=True) as dataset:
        with pytest.raises(IndexError):
            _ = dataset[200]


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _BASIC_CASES)
def test_dataset_accepts_pathlike_inputs(request, fixture_name, dataset_cls, init_kwargs):
    """Test that datasets accept pathlib.Path inputs."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(dir_list=[Path(dataset_path)], return_id=True, **init_kwargs) as dataset:
        _assert_item_shapes(dataset[0], 3)


def test_get_item_without_id(temp_hdf5_file):
    """Test retrieving an item when `return_id=False`."""
    with HDF5SingleCellDataset(dir_list=[temp_hdf5_file], dir_labels=[0], return_id=False) as dataset:
        _assert_item_shapes(dataset[0], 2)


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _LABELLED_CASES)
def test_labelled_dataset_initialization(request, fixture_name, dataset_cls, init_kwargs):
    """Test labelled datasets initialize correctly."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(dir_list=[dataset_path], return_id=True, **init_kwargs) as dataset:
        assert isinstance(dataset, dataset_cls)
        assert len(dataset.dir_list) == 1
        assert len(dataset) == 100


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _LABELLED_CASES)
def test_labelled_get_item(request, fixture_name, dataset_cls, init_kwargs):
    """Test retrieving an item from a labelled dataset."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(dir_list=[dataset_path], return_id=True, **init_kwargs) as dataset:
        _assert_item_shapes(dataset[0], 3)


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


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _BASIC_CASES)
def test_index_list_subset(request, fixture_name, dataset_cls, init_kwargs):
    """Test that only specified indices are loaded via index_list."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(
        dir_list=[dataset_path],
        index_list=[[0, 1, 2, 3, 4]],
        return_id=True,
        **init_kwargs,
    ) as dataset:
        assert len(dataset) == 5
        for i in range(len(dataset)):
            _assert_item_shapes(dataset[i], 3)


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _LABELLED_CASES)
def test_labelled_index_list_subset(request, fixture_name, dataset_cls, init_kwargs):
    """Test that only specified indices are loaded for labelled datasets via index_list."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(
        dir_list=[dataset_path],
        index_list=[[10, 11, 12]],
        return_id=True,
        **init_kwargs,
    ) as dataset:
        assert len(dataset) == 3
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
        assert labels == [10, 11, 12]
        for i in range(len(dataset)):
            _assert_item_shapes(dataset[i], 3)


@pytest.mark.parametrize(
    ("fixture_name", "dataset_cls", "init_kwargs"),
    [
        pytest.param(
            "temp_hdf5_file",
            LabelledHDF5SingleCellDataset,
            {"label_colum": 0, "label_dtype": float, "label_column_transform": lambda x: x / 10},
            id="hdf5",
        ),
        pytest.param(
            "temp_h5sc_file",
            LabelledH5ScSingleCellDataset,
            {"label_colum": "pseudo_label", "label_column_transform": lambda x: x / 10},
            id="h5sc",
        ),
    ],
)
def test_label_transform_applied_once(request, fixture_name, dataset_cls, init_kwargs):
    """Test that label transforms are applied exactly once."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(
        dir_list=[dataset_path],
        index_list=[[10]],
        return_id=True,
        **init_kwargs,
    ) as dataset:
        assert float(dataset[0][1]) == 1.0


@pytest.mark.parametrize(
    ("fixture_name", "dataset_cls", "init_kwargs", "expected_label", "expected_len"),
    _DIRECTORY_CASES,
)
def test_directory_scan_uses_full_paths_and_directory_label(
    request, fixture_name, dataset_cls, init_kwargs, expected_label, expected_len
):
    """Test that directory inputs load all files with the correct shared label."""
    dataset_dir = request.getfixturevalue(fixture_name)
    with dataset_cls(dir_list=[dataset_dir], return_id=True, **init_kwargs) as dataset:
        assert len(dataset) == expected_len

        if hasattr(dataset, "paths"):
            assert all(os.path.dirname(path) == dataset_dir for path in dataset.paths)
            assert len(dataset.paths) == 2
        else:
            assert len(dataset.handle_list) == 2

        assert float(dataset[0][1]) == expected_label
        assert float(dataset[len(dataset) - 1][1]) == expected_label


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _BASIC_CASES)
def test_close_is_idempotent(request, fixture_name, dataset_cls, init_kwargs):
    """Test that closing a dataset is idempotent and terminal."""
    dataset_path = request.getfixturevalue(fixture_name)
    dataset = dataset_cls(dir_list=[dataset_path], return_id=True, **init_kwargs)
    _ = dataset[0]
    dataset.close()
    dataset.close()

    assert dataset._closed is True
    if hasattr(dataset, "paths"):
        assert dataset._open_hdf == {}
    else:
        assert dataset.handle_list == []
        assert dataset._open_hdf == []

    with pytest.raises(RuntimeError, match="Dataset has been closed and cannot be reused."):
        dataset[0]


@pytest.mark.parametrize(("fixture_name", "dataset_cls", "init_kwargs"), _BASIC_CASES)
def test_context_manager_releases_file(request, fixture_name, dataset_cls, init_kwargs):
    """Test that dataset context managers close the dataset on exit."""
    dataset_path = request.getfixturevalue(fixture_name)
    with dataset_cls(dir_list=[dataset_path], return_id=True, **init_kwargs) as dataset:
        _ = dataset[0]

    assert dataset._closed is True
    if hasattr(dataset, "paths"):
        assert dataset._open_hdf == {}
    else:
        assert dataset.handle_list == []
        assert dataset._open_hdf == []

    with pytest.raises(RuntimeError, match="Dataset has been closed and cannot be reused."):
        dataset[0]


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
