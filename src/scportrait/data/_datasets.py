from pathlib import Path

from scportrait.data._dataloader import _download
from scportrait.tools.ml.pretrained_models import get_data_dir


def _get_remote_dataset(dataset: str, url: str, name: str | None = None) -> Path:
    """Download and extract a dataset from a remote location.

    Args:
        dataset: Name of the folder to save the dataset
        url: URL to the dataset
        name: Name of a specific file in the dataset folder to return, if None the whole dataset folder is returned

    Returns:
        Path to the downloaded and extracted dataset or a specific file in the dataset
    """
    data_dir = get_data_dir()
    save_path = data_dir / dataset
    if not save_path.exists():
        _download(url=url, output_path=str(save_path), archive_format="zip")
    if name is None:
        return save_path
    else:
        return save_path / name


def dataset_1() -> Path:
    """Download and extract the example dataset 1 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "example_1_images"
    URL = "https://zenodo.org/records/13385933/files/example_1_images.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_1_omezarr() -> Path:
    """Download and extract the example dataset 1 images in ome.zarr format.

    Returns:
        Path to the downloaded ome.zarr file.
    """
    DATASET = "example_1_images_omezarr"
    URL = "https://zenodo.org/records/14841309/files/input_image.ome.zarr.zip?download=1"
    NAME = "input_image.ome.zarr"
    return _get_remote_dataset(DATASET, URL, NAME)


def dataset_2() -> Path:
    """Download and extract the example dataset 2 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "example_2_images"
    URL = "https://zenodo.org/records/13742316/files/example_2_images.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_3() -> Path:
    """Download and extract the example dataset 3 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "example_3_images"
    URL = "https://zenodo.org/records/13742319/files/example_3_images.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_4() -> Path:
    """Download and extract the example dataset 4 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "example_4_images"
    URL = "https://zenodo.org/records/13742331/files/example_4_images.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_5() -> Path:
    """Download and extract the example dataset 5 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "example_5_images"
    URL = "https://zenodo.org/records/13742344/files/example_5_images.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_6() -> Path:
    """Download and extract the example dataset 6 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "example_6_images"
    URL = "https://zenodo.org/records/13742373/files/example_6_images.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_stitching_example() -> Path:
    """Download and extract the example dataset for stitching images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "stitching_example"
    URL = "https://zenodo.org/records/13742379/files/example_stitching.zip?download=1"
    return _get_remote_dataset(DATASET, URL)
