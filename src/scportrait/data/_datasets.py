from pathlib import Path
from typing import Literal

from scportrait.data._dataloader import _download
from scportrait.tools.ml.pretrained_models import get_data_dir


def _get_remote_dataset(
    dataset: str,
    url: str,
    name: str | None = None,
    archive_format: Literal["zip", "tar", "tar.gz", "tgz"] | None = "zip",
    outfile_name: None | str = None,
    outdirectory: None | str = None,
    force_download: bool = False,
) -> Path:
    """Download and extract a dataset from a remote location.

    Args:
        dataset: Name of the folder to save the dataset
        url: URL to the dataset
        name: Name of a specific file in the dataset folder to return, if None the whole dataset folder is returned

    Returns:
        Path to the downloaded and extracted dataset or a specific file in the dataset
    """
    data_dir = get_data_dir()
    if outdirectory is not None:
        data_dir = data_dir / outdirectory
        data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / dataset

    dataset_exists = save_path.exists()
    expected_path = save_path / name if name is not None else None
    missing_expected_file = dataset_exists and expected_path is not None and not expected_path.exists()
    should_download = force_download or not dataset_exists or missing_expected_file

    if should_download:
        _download(
            url=url,
            output_path=str(save_path),
            output_file_name=outfile_name,
            archive_format=archive_format,
            overwrite=force_download or missing_expected_file,
        )

    if name is None:
        return save_path
    else:
        return save_path / name


def get_config_file(config_id, force_download: bool = False) -> Path:
    config_files = {
        "dataset_1_wga_config": "https://raw.githubusercontent.com/MannLabs/scPortrait-notebooks/main/example_projects/example_1/config_example1_WGASegmentation.yml",
        "dataset_1_config": "https://raw.githubusercontent.com/MannLabs/scPortrait-notebooks/main/example_projects/example_1/config_example1.yml",
        "dataset_1_custom_cellpost_config": "https://raw.githubusercontent.com/MannLabs/scPortrait-notebooks/main/example_projects/example_1/config_example1_custom_params.yml",
        "dataset_2_config": "https://raw.githubusercontent.com/MannLabs/scPortrait-notebooks/main/example_projects/example_2/config_example2.yml",
        "dataset_3_config": "https://raw.githubusercontent.com/MannLabs/scPortrait-notebooks/main/example_projects/example_3/config_example3.yml",
        "dataset_4_config": "https://raw.githubusercontent.com/MannLabs/scPortrait-notebooks/main/example_projects/example_4/config_example4.yml",
    }

    DATASET = config_id
    URL = config_files[config_id]
    NAME = "config.yml"
    return _get_remote_dataset(
        DATASET,
        URL,
        NAME,
        archive_format=None,
        outfile_name=NAME,
        outdirectory="example_configs",
        force_download=force_download,
    )


def custom_cellpose_model() -> Path:
    """Download the example custom cellpose model.

    Returns:
        Path to the downloaded and extracted custom cellpose model
    """
    DATASET = "custom_cellpose_model"
    URL = "https://zenodo.org/records/14931602/files/custom_cellpose_model.cpkt?download=1"
    NAME = "custom_cellpose_model.cpkt"
    return _get_remote_dataset(DATASET, URL, NAME, archive_format=None, outfile_name=NAME)


def autophagosome_h5sc() -> list[Path]:
    """Download the example autophagosome h5sc datasets.

    Consists of two h5sc files with a small subset of single-cell images of cells with and without autophagosomes.
    The first file is autophagy positive. The second file is autophagy negative.

    Returns:
        Path to the downloaded and extracted h5sc datasets
    """
    DATASET = "autophagosome_h5sc"
    URL = "https://zenodo.org/api/records/15105848/files-archive"
    NAMES = ["stimulated_small.h5sc", "unstimulated_small.h5sc"]
    save_path = _get_remote_dataset(DATASET, URL)
    return [save_path / name for name in NAMES]


def _test_dataset() -> Path:
    """Download and extract the test dataset.

    Returns:
        Path to the downloaded and extracted test dataset.
    """
    DATASET = "test_dataset"
    URL = "https://zenodo.org/api/records/17560340/files-archive"
    return _get_remote_dataset(DATASET, URL)


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


def dataset_7() -> Path:
    """Download and extract the example dataset 7 images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "example_7_images"
    URL = "https://zenodo.org/records/14951047/files/example_7_images.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_stitching_example() -> Path:
    """Download and extract the example dataset for stitching images.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "stitching_example"
    URL = "https://zenodo.org/records/13742379/files/example_stitching.zip?download=1"
    return _get_remote_dataset(DATASET, URL)


def dataset_parsing_example() -> Path:
    """Download and extract the example dataset for parsing Harmony exported imaging experiments.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "parsing_example_basic"
    URL = "https://zenodo.org/records/14193689/files/harmony_export_V7_basic.zip?download=1"
    NAME = "basic_export"
    return _get_remote_dataset(DATASET, URL, NAME)


def dataset_parsing_example_flatfield_corrected() -> Path:
    """Download and extract the example dataset for parsing Harmony exported imaging experiments.

    Returns:
        Path to the downloaded and extracted images.
    """
    DATASET = "parsing_example_flatfield"
    URL = "https://zenodo.org/records/14973295/files/harmony_export_V7_flatfield_corrected.zip?download=1"
    NAME = "flat_field_corrected"
    return _get_remote_dataset(DATASET, URL, NAME)
