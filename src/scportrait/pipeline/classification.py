import os
import sys
import numpy as np

import torch
from torch.masked import masked_tensor

from scportrait.ml.datasets import HDF5SingleCellDataset
from scportrait.ml.transforms import ChannelSelector
from scportrait.ml.plmodels import MultilabelSupervisedModel
from scportrait.pipeline.base import ProcessingStep

from torchvision import transforms

import pandas as pd

import io
from contextlib import redirect_stdout
from pathlib import Path
from alphabase.io import tempmmap

from tqdm.auto import tqdm


class MLClusterClassifier(ProcessingStep):
    """
    Class for classifying single cells using a pre-trained machine learning model.

    This class takes a pre-trained model and uses it to classify single cells,
    using the model's forward function or encoder function, depending on the
    user's choice. The classification results are saved to a CSV file.

    Attributes
    ----------
    config : dict
        Config file which is passed by the Project class when called. It is loaded from the project based on the name of the class.
    directory : str
        Directory which should be used by the processing step. The directory will be newly created if it does not exist yet. When used with the :class:`sparcscore.pipeline.project.Project` class, a subdirectory of the project directory is passed.
    intermediate_output : bool, optional
        When set to True, intermediate outputs will be saved where applicable. Default is False.
    debug : bool, optional
        When set to True, debug outputs will be printed where applicable. Default is False.
    overwrite : bool, optional
        When set to True, the processing step directory will be completely deleted and newly created when called. Default is False.
    """

    CLEAN_LOG = True
    PRETRAINED_MODELS = [
        "autophagy_classifier1.0",
        "autophagy_classifier2.0",
        "autophagy_classifier2.1",
    ]
    CLASSIFIER_ARCHITECTURES = ["VGG1", "VGG2", "VGG1_old", "VGG2_old"]

    def __init__(
        self,
        config,
        path,
        project_location,
        debug=False,
        overwrite=False,
        intermediate_output=True,
    ):
        """
        Class is initiated to classify extracted single cells.

        Parameters
        ----------
        config : dict
            Configuration for the extraction passed over from the :class:`pipeline.Project`.

        path : str
            Directory for the extraction log and results. Will be created if not existing yet.

        debug : bool, optional, default=False
            Flag used to output debug information and map images.

        overwrite : bool, optional, default=False
            Flag used to overwrite existing results.
        """

        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        self.project_location = project_location

        if "filtered_dataset" in config.keys():
            self.filtered_dataset = self.config["filtered_dataset"]

        # Create classification directory
        self.directory = path
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # Set up log and clean old log
        if self.CLEAN_LOG:
            log_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)
            if os.path.isfile(log_path):
                os.remove(log_path)

        # check latest cluster run
        current_level_directories = [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
        runs = [int(i) for i in current_level_directories if self.is_Int(i)]

        self.current_run = max(runs) + 1 if len(runs) > 0 else 0

        if hasattr(self, "filtered_dataset"):
            if self.filtered_dataset is not None:
                self.run_path = os.path.join(
                    self.directory,
                    str(self.current_run)
                    + "_"
                    + self.config["inference_label"]
                    + "_"
                    + self.filtered_dataset,
                )
            else:
                self.run_path = os.path.join(
                    self.directory,
                    str(self.current_run) + "_" + self.config["inference_label"],
                )
        else:
            self.run_path = os.path.join(
                self.directory,
                str(self.current_run) + "_" + self.config["inference_label"],
            )  # to ensure that you can tell by directory name what is being classified

        if not os.path.isdir(self.run_path):
            os.makedirs(self.run_path)
            self.log("Created new directory " + self.run_path)

        self.log(f"current run: {self.current_run}")

    def is_Int(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def _get_config_parameters(self):
        # define base values if values are not specified in the config file
        # for those values that are essential raise warning if they are missing
        if "pretrained_model" in self.config.keys():
            self.pretrained_model = self.config["pretrained_model"]
        else:
            self.pretrained_model = False

        if "encoders" in self.config.keys():
            self.encoders = self.config["encoders"]
        else:
            self.encoders = ["forward"]

        if "resize" in self.config.keys():
            self.resize = True
            self.resize_value = self.config["resize"]
        else:
            self.resize = False

        assert (
            "network" in self.config.keys()
        ), "no network checkpoint specified in config file"
        assert (
            "channel_classification" in self.config.keys()
        ), "no channel_classification specified in config file"
        assert (
            "dataloader_worker_number" in self.config.keys()
        ), "no dataloader_worker_number specified in config file"
        assert (
            "batch_size" in self.config.keys()
        ), "no batch_size specified in config file"
        assert (
            "inference_device" in self.config.keys()
        ), "no inference_device specified in config file"
        assert (
            "inference_label" in self.config.keys()
        ), "no inference_label specified in config file"
        assert (
            "classifier_architecture" in self.config.keys()
        ), "no classifier_architecture specified in config file"

        self.network_dir = self.config["network"]

        if self.pretrained_model:
            assert (
                self.network_dir in self.PRETRAINED_MODELS
            ), f"the specified Pretrained model {self.networkdir} not available in scPortrait. Please choose one of the following available models: {self.PRETRAINED_MODELS}"
        else:
            assert os.path.exists(
                self.network_dir
            ), f"the specified network checkpoint {self.network_dir} does not exist"

            if "hparams_file" in self.config.keys():
                self.hparams_file = self.config["hparams_file"]
            else:
                hparam_path = Path(self.network_dir).parent / "hparams.yaml"
                if os.path.exists(hparam_path):
                    self.hparams_file = hparam_path
                else:
                    raise ValueError(
                        "no hparams file specified in config and could not be dynamically found based on checkpoint path."
                    )

        self.channel_classification = self.config["channel_classification"]
        # if multiple channels are selected ensure that they are converted to the proper format for passing to the pytorch dataset

        if isinstance(self.channel_classification, str):
            self.channel_classification = [
                int(x) for x in self.channel_classification.split(":")
            ]
        else:
            assert isinstance(
                self.channel_classification, int
            ), f"channel_classification should be an integer or a string of integers separated by colons. Provided value: {self.channel_classification}"
            self.channel_classification = [self.channel_classification]

        self.dataloader_worker_number = self.config["dataloader_worker_number"]
        self.batch_size = self.config["batch_size"]
        self.inference_device = self.config["inference_device"]
        self.inference_label = self.config["inference_label"]

        self.classifier_architecture = self.config["classifier_architecture"]
        assert (
            self.classifier_architecture in self.CLASSIFIER_ARCHITECTURES
        ), f"provided Classifier architecture {self.classifier_architecture} not implemented in scPortrait. Choose one of the following: {self.CLASSIFIER_ARCHITECTURES}"

    def _load_model(self):
        if self.pretrained_model:
            if self.network_dir == "autophagy_classifier1.0":
                from scportrait.ml.pretrained_models import autophagy_classifier1_0

                model = autophagy_classifier1_0(device=self.config["inference_device"])

            elif self.network_dir == "autophagy_classifier2.0":
                from scportrait.ml.pretrained_models import autophagy_classifier2_0

                model = autophagy_classifier2_0(device=self.config["inference_device"])

            elif self.network_dir == "autophagy_classifier2.1":
                from scportrait.ml.pretrained_models import autophagy_classifier2_1

                model = autophagy_classifier2_1(device=self.config["inference_device"])
            else:
                sys.exit("incorrect specification for pretrained model.")

        else:
            self.log(
                f"loading model from the following checkpoint file: {self.network_dir}"
            )
            self.log(
                f"loading model with the following hparams file: {self.hparams_file}"
            )

            model = MultilabelSupervisedModel.load_from_checkpoint(
                self.network_dir,
                hparams_file=self.hparams_file,
                type=self.classifier_architecture,
                map_location=self.inference_device,
            )

        model = model.eval()
        model.to(self.inference_device)

        return model

    def _load_dataset(self):
        # transforms like noise, random rotations, channel selection are still hardcoded
        if self.resize:
            t = transforms.Compose(
                [
                    ChannelSelector(self.channel_classification),
                    transforms.Resize(
                        (self.resize_value, self.resize_value), antialias=True
                    ),
                ]
            )
        else:
            t = transforms.Compose([ChannelSelector(self.channel_classification)])

        self.log(f"loading cells from {self.extraction_dir}")
        self.dataset = self.dataset_type(
            [f"{self.extraction_dir}"], [0], transform=t, return_id=True
        )

        self.dataset_size = len(
            self.dataset
        )  # save length of dataset for reaccess during inference
        self.log(f"Processing dataset with {self.dataset_size} cells")

    def __call__(  # type: ignore
        self,
        extraction_dir: str,
        partial: bool = False,
        dataset_type=HDF5SingleCellDataset,
    ):
        """
        Perform classification on the provided HDF5 dataset.

        Parameters
        ----------
        extraction_dir : str
            Directory containing the extracted HDF5 files from the project. If this class is used as part of
            a project processing workflow, this argument will be provided automatically.
        partial: bool, optional
            Flag to run on a selected subset of n_cells generated by a partial extraction. Default is False.
        dataset_type : HDF5SingleCellDataset, optional
            Pytorch Dataset to use for loading the dataset. Default is HDF5SingleCellDataset.

        Returns
        -------
        None
            Results are written to CSV files located in the project directory.

        Important
        ---------
        If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project``
        class based on the previous single-cell extraction. Therefore, only the second and third arguments need to be provided.
        The Project class will automatically provide the most recent extracted single-cell dataset together with the supplied parameters.

        Examples
        --------
        .. code-block:: python
            project.classify()

        Notes
        -----
        The following parameters are required in the config file:

        .. code-block:: yaml

            MLClusterClassifier:
                # Channel index on which the classification should be performed
                channel_classification: 4

                #boolean value indicating if a pretrained model availble in scPortrait should be used
                pretrained: False

                # Path to PyTorch checkpoint that should be used for inference or name of the pretrained model
                network: "path/to/model/"

                # Classifier architecture implemented in scPortrait
                # Choose one of VGG1, VGG2, VGG1_old, VGG2_old
                classifier_architecture: "VGG2_old"

                # Number of threads to use for dataloader
                dataloader_worker_number: 24

                # Batch size to pass to GPU
                batch_size: 900

                # Name under which the resulting CSV file will be saved
                inference_label: "Autophagy_15h_classifier1"

                # List of which inference methods should be performed
                # Available: "forward" and "encoder"
                # If "forward": images are passed through all layers of the model and the final inference results are written to file
                # If "encoder": activations at the end of the CNN are written to file
                encoders: ["forward", "encoder"]

                # On which device inference should be performed
                # For speed, should be "cuda"
                inference_device: "cuda"
        """

        self.create_temp_dir()  # setup directory for memory mapped temp file generation using alphabase.io
        self._get_config_parameters()

        self.extraction_dir = extraction_dir
        self.dataset_type = dataset_type

        self.log("Started classification")
        self.log(f"starting with run {self.current_run}")
        self.log(self.config)

        model = self._load_model()
        self._load_dataset()

        if partial is True:
            self.log("Running partial classification on selected cells.")

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dataloader_worker_number,
            shuffle=False,
        )

        if hasattr(self.config, "log_transform"):
            self.log(f"log transform: {self.config['log_transform']}")

        for encoder in self.encoders:
            if encoder == "forward":
                self.inference(dataloader, model.network.forward, partial=partial)
            if encoder == "encoder":
                self.inference(dataloader, model.network.encoder, partial=partial)

        # ensure all intermediate results are cleared after processing
        self.clear_temp_dir()

    def inference(self, dataloader, model_fun, partial=False):
        # 1. performs inference for a dataloader and a given network call
        # 2. saves the results to file

        data_iter = iter(dataloader)
        self.log(
            f"Start processing {len(data_iter)} batches with {model_fun.__name__} based inference"
        )

        with torch.no_grad():
            ix = 0
            batch_size = self.batch_size

            x, label, class_id = next(data_iter)
            r = model_fun(x.to(self.config["inference_device"]))
            result = r.cpu().detach()

            # initialize an empty memory mapped array for saving results into
            _, n_features = result.shape

            shape_features = (self.dataset_size, n_features)
            shape_labels = (self.dataset_size, 1)

            features_path = tempmmap.create_empty_mmap(shape_features, dtype=np.float32)
            cell_ids_path = tempmmap.create_empty_mmap(shape_labels, dtype=np.int64)
            labels_path = tempmmap.create_empty_mmap(shape_labels, dtype=np.int64)

            features = tempmmap.mmap_array_from_path(features_path)
            cell_ids = tempmmap.mmap_array_from_path(cell_ids_path)
            labels = tempmmap.mmap_array_from_path(labels_path)

            # save the results for each batch into the memory mapped array at the specified indices
            features[ix : (ix + batch_size)] = result.numpy()
            cell_ids[ix : (ix + batch_size)] = class_id.unsqueeze(1)
            labels[ix : (ix + batch_size)] = label.unsqueeze(1)
            ix += batch_size

            for i in range(len(dataloader) - 1):
                if i % 10 == 0:
                    self.log(f"processing batch {i}")
                x, label, class_id = next(data_iter)

                r = model_fun(x.to(self.config["inference_device"]))

                # save the results for each batch into the memory mapped array at the specified indices
                features[ix : (ix + r.shape[0])] = r.cpu().detach().numpy()
                cell_ids[ix : (ix + r.shape[0])] = class_id.unsqueeze(1)
                labels[ix : (ix + r.shape[0])] = label.unsqueeze(1)

                ix += r.shape[0]

        if hasattr(self.config, "log_transform"):
            if self.config["log_transform"]:
                sigma = 1e-9
                features = np.log(features + sigma)

        # save inferred activations / predictions
        result_labels = [f"result_{i}" for i in range(features.shape[1])]

        dataframe = pd.DataFrame(data=features, columns=result_labels)
        dataframe["label"] = labels
        dataframe["cell_id"] = cell_ids.astype("int")

        self.log("finished processing")

        if partial:
            path = os.path.join(
                self.run_path, f"partial_dimension_reduction_{model_fun.__name__}.csv"
            )
        else:
            path = os.path.join(
                self.run_path, f"dimension_reduction_{model_fun.__name__}.csv"
            )

        dataframe.to_csv(path)


class EnsembleClassifier(ProcessingStep):
    """
    This class takes a pre-trained ensemble of models and uses it to classify extracted single cell datasets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # create directory if it does not yet exist
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)

        self.ensemble_name = self.config["classification_label"]

        # generate directory where results should be saved
        self.directory = os.path.join(self.directory, self.ensemble_name)

    def load_model(self, ckpt_path):
        self.log(f"Loading model from checkpoing: {ckpt_path}")
        hparams_path = ckpt_path.replace(os.path.basename(ckpt_path), "hparams.yml")
        model = MultilabelSupervisedModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
            map_location=self.config["inference_device"],
        )
        model = model.eval()
        model.to(self.config["inference_device"])
        self.log(
            f"model loaded and transferred to device {self.config['inference_device']}"
        )

        return model

    def generate_dataloader(self, extraction_dir):
        # generate dataset
        self.log(f"Reading data from path: {extraction_dir}")
        px_size = self.config["input_image_px"]
        t = transforms.Compose([transforms.Resize((px_size, px_size), antialias=True)])
        self.log(f"Transforming input images to shape {px_size}x{px_size}")

        f = io.StringIO()
        with redirect_stdout(f):
            dataset = HDF5SingleCellDataset(
                [extraction_dir],
                [0],
                transform=t,
                return_id=True,
                select_channel=self.config["channel_classification"],
            )

        # generate dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["dataloader_worker_number"],
            drop_last=False,
        )

        self.log(
            f"Dataloader generated with a batchsize of {self.config['batch_size']} and {self.config['dataloader_worker_number']} workers. Dataloader contains {len(dataloader)} entries."
        )
        return dataloader

    def get_gpu_memory_usage(self):
        if self.config["inference_device"] == "cpu":
            return None
        else:
            try:
                memory_usage = []
                for i in range(torch.cuda.device_count()):
                    gpu_memory = (
                        torch.cuda.memory_reserved(i) / 1024**2
                    )  # Convert bytes to MiB
                    memory_usage.append(gpu_memory)
                results = {
                    f"GPU_{i}": f"{memory_usage[i]} MiB"
                    for i in range(len(memory_usage))
                }
                return results
            except Exception as e:
                print("Error:", e)
                return None

    def inference(self, dataloader, model_ensemble, partial=False):
        data_iter = iter(dataloader)
        self.log(
            f"Start processing {len(data_iter)} batches with {len(model_ensemble)} models from ensemble."
        )

        with torch.no_grad():
            x, label, class_id = next(data_iter)
            x = x.to(self.config["inference_device"])

            for ix, model_fun in enumerate(model_ensemble):
                r = model_fun(x)
                r = r.cpu().detach()

                if ix == 0:
                    _result = r
                else:
                    _result = torch.cat((_result, r), 1)

            result = _result

            for i in range(len(dataloader) - 1):
                if i % 10 == 0:
                    self.log(f"processing batch {i}")
                x, _label, id = next(data_iter)
                x = x.to(self.config["inference_device"])

                for ix, model_fun in enumerate(model_ensemble):
                    r = model_fun(x)
                    r = r.cpu().detach()

                    if ix == 0:
                        _result = r
                    else:
                        _result = torch.cat((_result, r), 1)

                result = torch.cat((result, _result), 0)
                label = torch.cat((label, _label), 0)
                class_id = torch.cat((class_id, id), 0)

        result = result.detach().numpy()
        label = label.numpy()
        class_id = class_id.numpy()

        # save inferred activations / predictions
        result_labels = []
        for model in self.model_names:
            _result_labels = [f"{model}_result_{i}" for i in range(r.shape[1])]
            result_labels = result_labels + _result_labels

        dataframe = pd.DataFrame(data=result, columns=result_labels)
        dataframe["cell_id"] = class_id.astype("int")

        # reorder columns to make it more readable
        columns_to_move = ["cell_id"]
        other_columns = [col for col in dataframe.columns if col not in columns_to_move]
        new_order = columns_to_move + other_columns
        dataframe = dataframe[new_order]

        if partial:
            path = os.path.join(
                self.directory, f"partial_ensemble_inference_{self.ensemble_name}.csv"
            )
        else:
            path = os.path.join(
                self.directory, f"ensemble_inference_{self.ensemble_name}.csv"
            )
        dataframe.to_csv(path, sep=",")

        self.log(f"Results saved to file: {path}")

    def __call__(self, extraction_dir, partial=False):
        """
        Function called to perform classification on the provided HDF5 dataset.

        Args:
            extraction_dir (str): Directory containing the extracted HDF5 files from the project. If this class is used as part of
            a project processing workflow this argument will be provided automatically.

        Returns:
            None: Results are written to csv files located in the project directory.

        Important:
            If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project``
            class based on the previous single-cell extraction. Therefore, no parameters need to be provided

        Example:

            .. code-block:: python

                project.classify()

        Note:

            The following parameters are required in the config file:

            .. code-block:: yaml

                EnsembleClassifier:
                    # channel number on which the classification should be performed
                    channel_classification: 4

                    #number of threads to use for dataloader
                    dataloader_worker_number: 24

                    #batch size to pass to GPU
                    batch_size: 900

                    #path to pytorch checkpoint that should be used for inference
                    networks:
                        model1: "path/to/model1/"
                        model2: "path/to/model2/"

                    #specify input size that the models expect, provided images will be rescaled to this size
                    input_image_px: 128

                    #label under which the results will be saved
                    classification_label: "Autophagy_15h_classifier1"

                    # on which device inference should be performed
                    # for speed should be "cuda"
                    inference_device: "cuda"
        """

        self.log("Starting Ensemble Classification")

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)
            self.log(
                f"Created new directory {self.directory} to save classification results to."
            )

        # load models and generate ensemble
        model_ensemble = []
        model_names = []

        for model_name, model_path in self.config["networks"].items():
            model = self.load_model(model_path)
            model_ensemble.append(model.network.forward)
            model_names.append(model_name)

        self.model_names = model_names

        self.log(
            f"Model Ensemble generated with a total of {len(model_ensemble)} models."
        )

        memory_usage = self.get_gpu_memory_usage()
        self.log(f"GPU memory usage after loading models: {memory_usage}")

        # generate dataloader
        dataloader = self.generate_dataloader(
            f"{extraction_dir}/{self.DEFAULT_DATA_FILE}"
        )

        # perform inference
        self.inference(
            dataloader=dataloader, model_ensemble=model_ensemble, partial=partial
        )


class CellFeaturizer(ProcessingStep):
    """
    Class for extracting general image features from SPARCS single-cell image datasets.
    The extracted features are saved to a CSV file. The features are calculated on the basis of a specified channel.

    The features which are calculated are:

    - Area of the nucleus in pixels
    - Area of the cytosol in pixels
    - Mean intensity of the chosen channel
    - Median intensity of the chosen channel
    - 75% quantile of the chosen channel
    - 25% quantile of the chosen channel
    - Summed intensity of the chosen channel in the region labeled as nucleus
    - Summed intensity of the chosen channel in the region labeled as cytosol
    - Summed intensity of the chosen channel in the region labeled as nucleus normalized to the nucleus area
    - Summed intensity of the chosen channel in the region labeled as cytosol normalized to the cytosol area

    The features are outputted in this order in the CSV file.
    """

    CLEAN_LOG = True

    def __init__(
        self,
        config,
        path,
        project_location,
        debug=False,
        overwrite=False,
        intermediate_output=True,
    ):
        """
        Class is initiated to featurize extracted single cells.

        Parameters
        ----------
        config : dict
            Configuration for the extraction passed over from the :class:`pipeline.Project`.
        path : str
            Directory for the extraction log and results. Will be created if not existing yet.
        project_location : str
            Location of the project directory.
        debug : bool, optional, default=False
            Flag used to output debug information and map images.
        overwrite : bool, optional, default=False
            Flag used to recalculate all images, not yet implemented.
        intermediate_output : bool, optional, default=True
            Flag to save intermediate outputs.
        """
        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        self.project_location = project_location

        if "filtered_dataset" in config.keys():
            self.filtered_dataset = self.config["filtered_dataset"]

        # Create classification directory
        self.directory = path
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # Set up log and clean old log
        if self.CLEAN_LOG:
            log_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)
            if os.path.isfile(log_path):
                os.remove(log_path)

        # check latest cluster run
        current_level_directories = [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
        runs = [int(i) for i in current_level_directories if self.is_Int(i)]

        self.current_run = max(runs) + 1 if len(runs) > 0 else 0

        if hasattr(self, "filtered_dataset"):
            self.run_path = os.path.join(
                self.directory,
                str(self.current_run)
                + "_"
                + self.config["screen_label"]
                + "_"
                + self.filtered_dataset,
            )
        else:
            self.run_path = os.path.join(
                self.directory,
                str(self.current_run) + "_" + self.config["screen_label"],
            )

        if not os.path.isdir(self.run_path):
            os.makedirs(self.run_path)
            self.log("Created new directory " + self.run_path)

        self.log(f"current run: {self.current_run}")

    def is_Int(self, s):
        """
        Check if a string represents an integer.

        Parameters
        ----------
        s : str
            String to check.

        Returns
        -------
        bool
            True if the string represents an integer, False otherwise.
        """
        try:
            int(s)
            return True
        except ValueError:
            return False

    def __call__(
        self,
        extraction_dir,
        accessory,
        size=0,
        partial=False,
        project_dataloader=HDF5SingleCellDataset,
        accessory_dataloader=HDF5SingleCellDataset,
    ):
        """
        Perform featurization on the provided HDF5 dataset.

        Parameters
        ----------
        extraction_dir : str
            Directory containing the extracted HDF5 files from the project. If this class is used as part of a project processing workflow this argument will be provided automatically.
        accessory : list
            List containing accessory datasets on which inference should be performed in addition to the cells contained within the current project.
        size : int, optional, default=0
            How many cells should be selected for inference. Default is 0, meaning all cells are selected.
        project_dataloader : HDF5SingleCellDataset, optional
            Dataloader for the project dataset. Default is HDF5SingleCellDataset.
        accessory_dataloader : HDF5SingleCellDataset, optional
            Dataloader for the accessory datasets. Default is HDF5SingleCellDataset.

        Returns
        -------
        None
            Results are written to CSV files located in the project directory.

        Important
        ---------
        If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project`` class based on the previous single-cell extraction. Therefore, only the second and third argument need to be provided. The Project class will automatically provide the most recent extraction results together with the supplied parameters.

        Examples
        --------
        .. code-block:: python

            # Define accessory dataset: additional HDF5 datasets that you want to perform an inference on
            # Leave empty if you only want to infer on all extracted cells in the current project

            accessory = ([], [], [])
            project.classify(accessory=accessory)

        Notes
        -----
        The following parameters are required in the config file:

        .. code-block:: yaml

            CellFeaturizer:
                # Channel number on which the featurization should be performed
                channel_classification: 4

                # Number of threads to use for dataloader
                dataloader_worker_number: 0 # needs to be 0 if using CPU

                # Batch size to pass to GPU
                batch_size: 900

                # On which device inference should be performed
                # For speed should be "cuda"
                inference_device: "cpu"

                # Label under which the results should be saved
                screen_label: "Ch3_Featurization"
        """
        self.log("Started classification")
        self.log(f"starting with run {self.current_run}")
        self.log(self.config)

        accessory_sizes, accessory_labels, accessory_paths = accessory

        self.log(f"{len(accessory_sizes)} different accessory datasets specified")

        # Generate project dataset dataloader
        t = transforms.Compose([])

        self.log(f"loading {extraction_dir}")

        # Redirect stdout to capture dataset size
        f = io.StringIO()
        with redirect_stdout(f):
            dataset = HDF5SingleCellDataset(
                [extraction_dir], [0], transform=t, return_id=True
            )

            if size == 0:
                size = len(dataset)
            residual = len(dataset) - size
            dataset, _ = torch.utils.data.random_split(dataset, [size, residual])

        # Load accessory dataset
        for i in range(len(accessory_sizes)):
            self.log(f"loading {accessory_paths[i]}")
            with redirect_stdout(f):
                local_dataset = HDF5SingleCellDataset(
                    [accessory_paths[i]], [i + 1], transform=t, return_fake_id=True
                )

            if len(local_dataset) > accessory_sizes[i]:
                residual = len(local_dataset) - accessory_sizes[i]
                local_dataset, _ = torch.utils.data.random_split(
                    local_dataset, [accessory_sizes[i], residual]
                )

            dataset = torch.utils.data.ConcatDataset([dataset, local_dataset])

        # Log stdout
        out = f.getvalue()
        self.log(out)

        # Classify samples
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["dataloader_worker_number"],
            shuffle=False,
        )
        self.inference(dataloader, partial=partial)

    def calculate_statistics(self, img, channel=-1):
        """
        Calculate statistics for an image batch.

        Parameters
        ----------
        img : torch.Tensor
            Tensor containing the image batch.
        channel : int, optional, default=-1
            Channel to calculate statistics over.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated statistics.
        """
        N, _, _, _ = img.shape

        # Calculate area statistics
        nucleus_mask = img[:, 0] > 0
        nucleus_area = nucleus_mask.view(N, -1).sum(1, keepdims=True)
        cytosol_mask = img[:, 1] > 0
        cytosol_area = cytosol_mask.view(N, -1).sum(1, keepdims=True)

        # Select channel to calculate summary statistics over
        img_selected = img[:, channel]

        # apply mask to channel to only compute the statistics over the pixels that are relevant
        mask = cytosol_mask
        mask[mask == 0] = torch.nan
        img_selected = (img_selected * mask).to(
            torch.float32
        )  # ensure we have correct dytpe for subsequent calculations

        mean = img_selected.view(N, -1).nanmean(1, keepdim=True)
        median = img_selected.view(N, -1).nanquantile(q=0.5, dim=1, keepdim=True)
        quant75 = img_selected.view(N, -1).nanquantile(q=0.75, dim=1, keepdim=True)
        quant25 = img_selected.view(N, -1).nanquantile(q=0.25, dim=1, keepdim=True)

        # Calculate more complex statistics
        summed_intensity_nucleus_area = (
            masked_tensor(img_selected, nucleus_mask)
            .view(N, -1)
            .sum(1)
            .reshape((N, 1))
            .to_tensor(0)
        )

        summed_intensity_nucleus_area_normalized = (
            summed_intensity_nucleus_area / nucleus_area
        )
        summed_intensity_cytosol_area = img_selected.view(N, -1).nansum(
            1, keepdims=True
        )
        summed_intensity_cytosol_area_normalized = (
            summed_intensity_cytosol_area / cytosol_area
        )

        # Generate results tensor with all values and return
        results = torch.concat(
            [
                nucleus_area,
                cytosol_area,
                mean,
                median,
                quant75,
                quant25,
                summed_intensity_nucleus_area,
                summed_intensity_cytosol_area,
                summed_intensity_nucleus_area_normalized,
                summed_intensity_cytosol_area_normalized,
            ],
            1,
        )
        return results

    def inference(self, dataloader, partial=False):
        """
        Perform inference for a dataloader and save the results to a file.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader to perform inference on.

        Returns
        -------
        None
        """
        data_iter = iter(dataloader)
        self.log(f"start processing {len(data_iter)} batches")
        with torch.no_grad():
            x, label, class_id = next(data_iter)
            r = self.calculate_statistics(
                x, channel=self.config["channel_classification"]
            )
            result = r

            for i in range(len(dataloader) - 1):
                if i % 10 == 0:
                    self.log(f"processing batch {i}")
                x, _label, id = next(data_iter)

                r = self.calculate_statistics(
                    x, channel=self.config["channel_classification"]
                )
                result = torch.cat((result, r), 0)
                label = torch.cat((label, _label), 0)
                class_id = torch.cat((class_id, id), 0)

        label = label.numpy()
        class_id = class_id.numpy()

        # Save inferred activations / predictions
        result_labels = [
            "nucleus_area",
            "cytosol_area",
            "mean",
            "median",
            "quant75",
            "quant25",
            "summed_intensity_nucleus_area",
            "summed_intensity_cytosol_area",
            "summed_intensity_nucleus_area_normalized",
            "summed_intensity_cytosol_area_normalized",
        ]

        dataframe = pd.DataFrame(data=result, columns=result_labels)
        dataframe["label"] = label
        dataframe["cell_id"] = class_id.astype("int")

        self.log("finished processing")

        if partial:
            path = os.path.join(self.run_path, "partial_calculated_features.csv")
        else:
            path = os.path.join(self.run_path, "calculated_features.csv")
        dataframe.to_csv(path)


class ConvNeXtFeaturizer(ProcessingStep):
    CLEAN_LOG = True

    def __init__(
        self,
        config,
        path,
        project_location,
        debug=False,
        overwrite=False,
        intermediate_output=True,
    ):
        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        self.project_location = project_location

        self._check_config()

        # assert that the correct transformers version is installed
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "transformers is not installed. Please install it via pip install 'transformers==4.26.0'"
            )

        assert (
            transformers.__version__ == "4.26.0"
        ), "Please install transformers version 4.26.0"

        if "filtered_dataset" in config.keys():
            self.filtered_dataset = self.config["filtered_dataset"]

        # Create classification directory
        self.directory = path
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # Set up log and clean old log
        if self.CLEAN_LOG:
            log_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)
            if os.path.isfile(log_path):
                os.remove(log_path)

        # create specific output directory
        if hasattr(self, "filtered_dataset"):
            if self.filtered_dataset is not None:
                self.run_path = os.path.join(
                    self.directory, f"ConvNeXt_{self.filtered_dataset}"
                )
            else:
                self.run_path = os.path.join(self.directory, "ConvNeXt")
        else:
            self.run_path = os.path.join(
                self.directory, "ConvNeXt"
            )  # to ensure that you can tell by directory name what is being classified

        if not os.path.isdir(self.run_path):
            os.makedirs(self.run_path)
            self.log("Created new directory " + self.run_path)

        self.log(f"current run: {self.current_run}")

        self.inference_device = self.config["inference_device"]
        self.dataloader_worker_number = self.config["dataloader_worker_number"]
        self.batch_size = self.config["batch_size"]

    def _check_config(self):
        assert (
            "inference_device" in self.config.keys()
        ), "no inference_device specified in config file"
        assert (
            "dataloader_worker_number" in self.config.keys()
        ), "no dataloader_worker_number specified in config file"
        assert (
            "batch_size" in self.config.keys()
        ), "no batch_size specified in config file"

        if "channel_selection" in self.config.keys():
            self.channel_selection = self.config["channel_selection"]

            assert isinstance(
                self.channel_selection, (int, list)
            ), "channel_selection should be an integer or a list of integers"

            if isinstance(self.channel_selection, int):
                self.channel_selection = [self.channel_selection]
            if isinstance(self.channel_selection, list):
                assert all(isinstance(i, int) for i in self.channel_selection)
                "channel_selection should be an integer or a list of integers"
                assert len(self.channel_selection) in [1, 3]
                "channel_selection should be either 1 or 3 channels"

    def _load_model(self):
        # lazy imports
        from transformers import ConvNextModel

        model = ConvNextModel.from_pretrained("facebook/convnext-xlarge-224-22k")
        model.eval()
        model.to(self.device)

        return model

    def _setup_transform(self):
        # lazy imports
        from transformers import AutoImageProcessor
        from scportrait.ml.transforms import ChannelSelector, ChannelMultiplier

        feature_extractor = AutoImageProcessor.from_pretrained(
            "facebook/convnext-xlarge-224-22k"
        )

        if len(self.channel_selection) == 1:
            self.transforms = transforms.Compose(
                [
                    ChannelSelector(self.channel_selection),
                    ChannelMultiplier(3),
                    feature_extractor,
                ]
            )
        elif len(self.channel_selection) == 3:
            self.transforms = transforms.Compose(
                [
                    ChannelSelector(self.channel_selection),
                    feature_extractor,
                ]
            )
        else:
            raise ValueError("channel_selection should be either 1 or 3 channels")

    def _load_dataset(self):
        # transforms like noise, random rotations, channel selection are still hardcoded

        self.log(f"loading cells from {self.extraction_dir}")
        self.dataset = self.dataset_type(
            [f"{self.extraction_dir}"], [0], transform=self.transforms, return_id=True
        )

        self.dataset_size = len(
            self.dataset
        )  # save length of dataset for reaccess during inference
        self.log(f"Processing dataset with {self.dataset_size} cells")

    def __call__(  # type: ignore
        self,
        extraction_dir: str,
        partial: bool = False,
        dataset_type=HDF5SingleCellDataset,
    ):
        """
        Perform ConvNeXt inference on the provided HDF5 dataset.

        Parameters
        ----------
        extraction_dir : str
            Directory containing the extracted HDF5 files from the project. If this class is used as part of
            a project processing workflow, this argument will be provided automatically.
        partial: bool, optional
            Flag to run on a selected subset of n_cells generated by a partial extraction. Default is False.
        dataset_type : HDF5SingleCellDataset, optional
            Pytorch Dataset to use for loading the dataset. Default is HDF5SingleCellDataset.

        Returns
        -------
        None
            Results are written to CSV files located in the project directory.

        Important
        ---------
        If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project``
        class based on the previous single-cell extraction. Therefore, only the second and third arguments need to be provided.
        The Project class will automatically provide the most recent extracted single-cell dataset together with the supplied parameters.

        Examples
        --------
        .. code-block:: python
            project.classify()

        Notes
        -----
        The following parameters are required in the config file:

        .. code-block:: yaml

            MLClusterClassifier:
                # Channel index on which the classification should be performed
                channel_selection: 4

                # Number of threads to use for dataloader
                dataloader_worker_number: 24

                # Batch size to pass to GPU
                batch_size: 100

                # On which device inference should be performed
                # For speed, should be "cuda"
                inference_device: "cuda"
        """

        self.create_temp_dir()  # setup directory for memory mapped temp file generation using alphabase.io

        self.extraction_dir = extraction_dir
        self.dataset_type = dataset_type

        self.log("Started Featurization")

        model = self._load_model()
        self._load_dataset()

        if partial is True:
            self.log("Running partial classification on selected cells.")

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dataloader_worker_number,
            shuffle=False,
        )

        self.inference(dataloader, model, partial=partial)

        # ensure all intermediate results are cleared after processing
        self.clear_temp_dir()

    def inference(self, dataloader, model_fun, partial=False):
        # 1. performs inference for a dataloader and a given network call
        # 2. saves the results to file

        data_iter = iter(dataloader)
        self.log(f"Start processing {len(data_iter)} batches with ConvNeXt model.")

        with torch.no_grad():
            ix = 0
            batch_size = self.batch_size

            images, label, class_id = next(data_iter)
            images["pixel_values"] = images["pixel_values"][0].to(self.device)
            o = model_fun(**images)
            result = o.pooler_output.cpu().detach()

            # initialize an empty memory mapped array for saving results into
            _, n_features = result.shape

            shape_features = (self.dataset_size, n_features)
            shape_labels = (self.dataset_size, 1)

            features_path = tempmmap.create_empty_mmap(shape_features, dtype=np.float32)
            cell_ids_path = tempmmap.create_empty_mmap(shape_labels, dtype=np.int64)
            labels_path = tempmmap.create_empty_mmap(shape_labels, dtype=np.int64)

            features = tempmmap.mmap_array_from_path(features_path)
            cell_ids = tempmmap.mmap_array_from_path(cell_ids_path)
            labels = tempmmap.mmap_array_from_path(labels_path)

            # save the results for each batch into the memory mapped array at the specified indices
            features[ix : (ix + batch_size)] = result.numpy()
            cell_ids[ix : (ix + batch_size)] = class_id.unsqueeze(1)
            labels[ix : (ix + batch_size)] = label.unsqueeze(1)
            ix += batch_size

            for i in tqdm(range(len(dataloader) - 1)):
                images, label, class_id = next(data_iter)
                images["pixel_values"] = images["pixel_values"][0].to(self.device)

                o = model_fun(**images)
                result = o.pooler_output.cpu().detach()

                # save the results for each batch into the memory mapped array at the specified indices
                features[ix : (ix + result.shape[0])] = result.numpy()
                cell_ids[ix : (ix + result.shape[0])] = class_id.unsqueeze(1)
                labels[ix : (ix + result.shape[0])] = label.unsqueeze(1)

                ix += result.shape[0]

            # save inferred activations / predictions
            result_labels = [f"convnext_{i}" for i in range(n_features)]
            dataframe = pd.DataFrame(data=features, columns=result_labels)
            dataframe["label"] = labels
            dataframe["cell_id"] = cell_ids.astype("int")

        self.log("finished processing")

        if partial:
            path = os.path.join(self.run_path, "partial_featurization_ConvNeXt.csv")
        else:
            path = os.path.join(self.run_path, "featurization_ConvNeXt.csv")

        dataframe.to_csv(path)
