import os
from pathlib import Path

from scportrait.data._dataloader import _download
from scportrait.ml.pretrained_models import _get_data_dir


def dataset_1():
    """
    Download and extract the example dataset 1 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "example_1_images")

    if not Path(save_path).exists():
        _download(
            url="https://zenodo.org/records/13385933/files/example_1_images.zip?download=1",
            output_path=save_path,
            archive_format="zip",
        )

    return save_path


def dataset_2():
    """
    Download and extract the example dataset 2 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "example_2_images")

    if not Path(save_path).exists():
        _download(
            url="https://zenodo.org/records/13742316/files/example_2_images.zip?download=1",
            output_path=save_path,
            archive_format="zip",
        )

    return save_path


def dataset_3():
    """
    Download and extract the example dataset 3 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "example_3_images")

    if not Path(save_path).exists():
        _download(
            url="https://zenodo.org/records/13742319/files/example_3_images.zip?download=1",
            output_path=save_path,
            archive_format="zip",
        )

    return save_path


def dataset_4():
    """
    Download and extract the example dataset 4 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "example_4_images")

    if not Path(save_path).exists():
        _download(
            url="https://zenodo.org/records/13742331/files/example_4_images.zip?download=1",
            output_path=save_path,
            archive_format="zip",
        )

    return save_path


def dataset_5():
    """
    Download and extract the example dataset 5 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "example_5_images")

    if not Path(save_path).exists():
        _download(
            url="https://zenodo.org/records/13742344/files/example_5_images.zip?download=1",
            output_path=save_path,
            archive_format="zip",
        )

    return save_path


def dataset_6():
    """
    Download and extract the example dataset 6 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "example_6_images")

    if not Path(save_path).exists():
        _download(
            url="https://zenodo.org/records/13742373/files/example_6_images.zip?download=1",
            output_path=save_path,
            archive_format="zip",
        )

    return save_path


def dataset_stitching_example():
    """
    Download and extract the example dataset for stitching images.

    Returns:
        Path to the downloaded and extracted images.
    """
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "stitching_example")

    if not Path(save_path).exists():
        _download(
            url="https://zenodo.org/records/13742379/files/example_stitching.zip?download=1",
            output_path=save_path,
            archive_format="zip",
        )

    return save_path