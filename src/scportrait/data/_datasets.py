from scportrait.data._dataloader import _download
from scportrait.tools.ml.pretrained_models import _get_data_dir
from pathlib import Path
import os

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
    pass

def dataset_3():
    pass

def dataset_4():
    pass

def dataset_5():
    pass

def dataset_6():
    pass

