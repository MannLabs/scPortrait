"""
Collection of functions to load pretrained models to use in the scPortrait environment.
"""

from scportrait.tools.ml.plmodels import MultilabelSupervisedModel
from scportrait.data._dataloader import _download
from pathlib import Path
import torch
import os

def _load_multilabelSupervised(checkpoint_path, hparam_path, model_type, eval = True, device = "cuda"):
    """
    Load a pretrained model uploaded to the github repository.

    Parameters
    ----------
    checkpoint_path : str
        The path of the checkpoint file to load the pretrained model from.
    hparam_path : str
        The path of the hparams file containing the hyperparameters used in training the model.
    type : str
        The type of the model, e.g., 'VGG1' or 'VGG2'.
    eval : bool, optional
        If True then the model will be returned in eval mode. Default is set to True.
    device : str | "cuda" or "cpu"
        String indicating which device the model should be loaded to. Either "cuda" or  "cpu".

    Returns
    -------
    model : MultilabelSupervisedModel
        The pretrained multilabel classification model loaded from the checkpoint, and moved to the appropriate device.

    Examples
    --------
    >>> model = _load_multilabelSupervised(“path/to/checkpoint.ckpt”, “path/to/hparams.yaml”, “resnet50”)
    >>> print(model)
    MultilabelSupervisedModel(…)

    """

    # Load model
    model = MultilabelSupervisedModel.load_from_checkpoint(
        checkpoint_path, hparams_file=hparam_path, model_type=model_type, map_location=device
    )
    if eval:
        model.eval()

    return(model)

def _get_data_dir():
    """
    Helper Function to get path to data that was packaged with scPortrait

    Returns
    -------
        str: path to data directory
    """

    def find_root_by_file(marker_file, current_path):
        for parent in current_path.parents:
            if (parent / marker_file).exists():
                return parent
        return None
    
    src_code_dir = find_root_by_file("README.md", Path(__file__))                                   
    data_dir = os.path.join(src_code_dir, 'scportrait_data')
    data_dir = os.path.abspath(data_dir)

    return (data_dir)

def autophagy_classifier(device = "cuda"):
    """
    Load binary autophagy classification model published as Model 2.1 in original scPortrait publication.
    """

    # check if cuda is available
    if device == "cuda" and torch.cuda.is_available() is False:
        print("CUDA is not available. Loading model on CPU.")
        device = "cpu"
        
    data_dir = _get_data_dir()
    save_path = os.path.join(data_dir, "vgg_autophagy_classifier")

    if not Path(save_path).exists():
        _download(
            url="", # TODO: URL to download 
            output_path=data_dir,
            archive_format="zip",
        )
    
    checkpoint_path = os.path.join(save_path, "VGG2_autophagy_classifier2.1.cpkt")
    hparam_path = os.path.join(save_path, "hparams.yaml")
    
    model = _load_multilabelSupervised(checkpoint_path, hparam_path, model_type = "VGG2_old", device = device)

    return(model)