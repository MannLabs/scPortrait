from pathlib import Path

from anndata import AnnData

from scportrait.data._datasets import _get_remote_dataset
from scportrait.io.h5sc import read_h5sc


def dataset2_h5sc() -> AnnData:
    """Get example single-cell image dataset derived from dataset 2.

    Returns:
        Loaded h5sc dataset.
    """
    DATASET = "dataset2_h5sc"
    URL = "https://zenodo.org/records/15164629/files/single_cells.h5sc?download=1"
    NAME = "single_cells.h5sc"
    path = _get_remote_dataset(DATASET, URL, NAME, archive_format=None, outfile_name=NAME)
    return read_h5sc(path)
