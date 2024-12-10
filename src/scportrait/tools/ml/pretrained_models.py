"""Collection of functions to load pretrained models to use in the scPortrait environment."""

from pathlib import Path
from typing import Literal

import torch

from scportrait.data._dataloader import _download
from scportrait.tools.ml.plmodels import MultilabelSupervisedModel


def load_multilabelSupervised(
    checkpoint_path: str | Path,
    hparam_path: str | Path,
    model_type: str,
    eval: bool = True,
    device: Literal["cpu", "cuda", "mps"] = "cuda",
) -> MultilabelSupervisedModel:
    """Load a pretrained model uploaded to the github repository.

    Args:
        checkpoint_path: The path of the checkpoint file to load the pretrained model from
        hparam_path: The path of the hparams file containing the hyperparameters
        model_type: The type of the model, e.g., 'VGG1' or 'VGG2'
        eval: If True then the model will be returned in eval mode
        device: Device to load the model on, either "cuda" or "cpu"

    Returns:
        The pretrained multilabel classification model loaded from the checkpoint

    Examples:
        >>> model = load_multilabelSupervised("path/to/checkpoint.ckpt", "path/to/hparams.yaml", "resnet50")
        >>> print(model)
        MultilabelSupervisedModel(...)
    """
    model = MultilabelSupervisedModel.load_from_checkpoint(
        checkpoint_path, hparams_file=hparam_path, model_type=model_type, map_location=device
    )
    if eval:
        model.eval()
    return model


def get_data_dir() -> Path:
    """Get path to data that was packaged with scPortrait.

    Returns:
        Path to data directory
    """

    def find_root_by_file(marker_file: str, current_path: Path) -> Path | None:
        for parent in current_path.parents:
            print(parent)
            if (parent / marker_file).exists():
                return parent
        return None

    src_code_dir = find_root_by_file("README.md", Path(__file__))
    if src_code_dir is None:
        raise FileNotFoundError("Could not find scPortrait root directory")

    data_dir = src_code_dir / "scportrait_data"
    return data_dir.absolute()


def autophagy_classifier(device: Literal["cuda", "cpu"] = "cuda") -> MultilabelSupervisedModel:
    """Load binary autophagy classification model from original SPARCS publication.

    Args:
        device: Device to load the model on, either "cuda" or "cpu"

    Returns:
        Pretrained autophagy classification model

    References:
        Schmacke NA, Mädler SC, Wallmann G, Metousis A, Bérouti M, Harz H,
        Leonhardt H, Mann M, Hornung V. SPARCS, a platform for genome-scale
        CRISPR screening for spatial cellular phenotypes. bioRxiv. 2023 Jun
        1;542416. doi: 10.1101/2023.06.01.542416.
    """
    data_dir = get_data_dir()
    save_path = data_dir / "vgg_autophagy_classifier"

    if not save_path.exists():
        _download(
            url="https://zenodo.org/records/13385705/files/vgg_autophagy_classifier.zip?download=1",
            output_path=str(data_dir),
            archive_format="zip",
        )

    checkpoint_path = save_path / "VGG2_autophagy_classifier2.1.cpkt"
    hparam_path = save_path / "hparams.yaml"

    # check if cuda is available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Loading model on CPU.")
        device = "cpu"

    model = load_multilabelSupervised(
        checkpoint_path=str(checkpoint_path), hparam_path=str(hparam_path), model_type="VGG2_old", device=device
    )
    return model
