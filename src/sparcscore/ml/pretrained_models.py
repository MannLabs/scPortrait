"""
Collection of functions to load pretrained models to use in the SPARCSpy environment.
"""

from sparcscore.ml.plmodels import MultilabelSupervisedModel
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
    Helper Function to get path to data that was packaged with SPARCSpy

    Returns
    -------
        str: path to data directory
    """
    src_code_dir, _ = os.path.split(__file__)
    data_dir = src_code_dir.replace("sparcscore/ml", "pretrained_models/")
    return (data_dir)

def autophagy_classifier1_0(device = "cuda"):
    """
    Load binary autophagy classification model published as Model 1.0 in original SPARCSpy publication.
    """

    data_dir = _get_data_dir()

    checkpoint_path = os.path.join(data_dir, "autophagy/autophagy1.0/VGG1_autophagy_classifier1.0.cpkt")
    hparam_path = os.path.join(data_dir, "autophagy/autophagy1.0/hparams.yaml")

    model = _load_multilabelSupervised(checkpoint_path, hparam_path, model_type = "VGG1_old", device = device)
    return(model)

def autophagy_classifier2_0(device = "cuda"):
    """
    Load binary autophagy classification model published as Model 2.0 in original SPARCSpy publication.
    """

    data_dir = _get_data_dir()

    checkpoint_path = os.path.join(data_dir, "autophagy/autophagy2.0/VGG2_autophagy_classifier2.0.cpkt")
    hparam_path = os.path.join(data_dir, "autophagy/autophagy2.0/hparams.yaml")

    model = _load_multilabelSupervised(checkpoint_path, hparam_path, model_type = "VGG2_old", device = device)
    return(model)

def autophagy_classifier2_1(device = "cuda"):
    """
    Load binary autophagy classification model published as Model 2.1 in original SPARCSpy publication.
    """

    data_dir = _get_data_dir()

    checkpoint_path = os.path.join(data_dir, "autophagy/autophagy2.1/VGG2_autophagy_classifier2.1.cpkt")
    hparam_path = os.path.join(data_dir, "autophagy/autophagy2.1/hparams.yaml")

    model = _load_multilabelSupervised(checkpoint_path, hparam_path, model_type = "VGG2_old", device = device)
    return(model)