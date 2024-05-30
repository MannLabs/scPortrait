from datetime import datetime
import os
import sys
import numpy as np

import torch
from torch.masked import masked_tensor

from sparcscore.ml.datasets import HDF5SingleCellDataset
from sparcscore.ml.transforms import ChannelSelector
from sparcscore.ml.plmodels import MultilabelSupervisedModel
from sparcscore.pipeline.base import ProcessingStep

from torchvision import transforms

import pandas as pd

import io
from contextlib import redirect_stdout


class MLClusterClassifier:
    """
    Class for classifying single cells using a pre-trained machine learning model.
    This class takes a pre-trained model and uses it to classify single_cells,
    using the model's forward function or encoder function, depending on the
    user's choice. The classification results are saved to a TSV file.

    Attributes:
        config (dict): Config file which is passed by the Project class when called. Is loaded from the project based on the name of the class.
        directory (str): Directory which should be used by the processing step. The directory will be newly created if it does not exist yet. When used with the :class:`vipercore.pipeline.project.Project` class, a subdirectory of the project directory is passed.
        intermediate_output (bool, default ``False``): When set to True intermediate outputs will be saved where applicable.
        debug (bool, default ``False``): When set to True debug outputs will be printed where applicable.
        overwrite (bool, default ``False``): When set to True, the processing step directory will be completely deleted and newly created when called.
    """

    DEFAULT_LOG_NAME = "processing.log"
    DEFAULT_DATA_DIR = "data"
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

        if self.filtered_dataset is not None:
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

    def get_timestamp(self):
        # Returns the current date and time as a formatted string.

        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        return "[" + dt_string + "] "

    def log(self, message):
        # Writes a message to a log file and prints it to the console if debug is True.

        log_path = os.path.join(self.run_path, self.DEFAULT_LOG_NAME)

        if isinstance(message, str):
            lines = message.split("\n")

        if isinstance(message, list):
            lines = message

        if isinstance(message, dict):
            lines = []
            for key, value in message.items():
                lines.append(f"{key}: {value}")

        for line in lines:
            with open(log_path, "a") as myfile:
                myfile.write(self.get_timestamp() + line + " \n")

            if self.debug:
                print(self.get_timestamp() + line)

    def __call__(
        self,
        extraction_dir,
        accessory,
        size=0,
        project_dataloader=HDF5SingleCellDataset,
        accessory_dataloader=HDF5SingleCellDataset,
    ):
        """
        Function called to perform classification on the provided HDF5 dataset.

        Args:
            extraction_dir (str): Directory containing the extracted HDF5 files from the project. If this class is used as part of
            a project processing workflow this argument will be provided automatically.
            accessory (list): List containing accessory datasets on which inference should be performed in addition to the cells
            contained within the current project.
            size (int, optional): How many cells should be selected for inference. Default is 0, which means all cells are selected.
            project_dataloader (HDF5SingleCellDataset, optional): Dataloader for the project dataset. Default is HDF5SingleCellDataset.
            accessory_dataloader (HDF5SingleCellDataset, optional): Dataloader for the accessory datasets. Default is HDF5SingleCellDataset.

        Returns:
            None: Results are written to tsv files located in the project directory.

        Important:
            If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project``
            class based on the previous single-cell extraction. Therefore, only the second and third argument need to be provided. The Project
            class will automaticly provide the most recent extracted single-cell dataset together with the supplied parameters.
            class based on the previous single-cell extraction. Therefore, only the second and third argument need to be provided. The Project
            class will automaticly provide the most recent extracted single-cell dataset together with the supplied parameters.



        Example:

            .. code-block:: python

                # define acceossory dataset: additional hdf5 datasets that you want to perform an inference on
                # leave empty if you only want to infere on all extracted cells in the current project

                accessory = ([], [], [])
                project.classify(accessory = accessory)

        Note:

            The following parameters are required in the config file:

            .. code-block:: yaml

                MLClusterClassifier:
                    # channel number on which the classification should be performed
                    channel_classification: 4

                    #number of threads to use for dataloader
                    threads: 24
                    dataloader_worker_number: 24

                    #batch size to pass to GPU
                    batch_size: 900

                    #path to pytorch checkpoint that should be used for inference
                    network: "path/to/model/"

                    #classifier architecture implemented in SPARCSpy
                    # choose one of VGG1, VGG2, VGG1_old, VGG2_old
                    classifier_architecture: "VGG2_old"

                    #if more than one checkpoint is provided in the network directory which checkpoint should be chosen
                    # should either be "max" or a numeric value indicating the epoch number
                    epoch: "max"

                    #name of the classifier used for saving the classification results to a directory
                    screen_label: "Autophagy_15h_classifier1"

                    # list of which inference methods should be performed
                    # available: "forward" and "encoder"
                    # if "forward": images are passed through all layers of the modela nd the final inference results are written to file
                    # if "encoder": activations at the end of the CNN is written to file
                    encoders: ["forward", "encoder"]

                    # on which device inference should be performed
                    # for speed should be "cuda"
                    inference_device: "cuda"
        """

        # is called with the path to the segmented image
        # Size: number of datapoints of the project dataset considered
        # ===== Dataloaders =====
        # should be HDF5SingleCellDataset for .h5 datasets
        # project_dataloader: dataloader for the project dataset
        # accessory_dataloader: dataloader for the accesssory datasets

        self.log("Started classification")
        self.log(f"starting with run {self.current_run}")
        self.log(self.config)

        accessory_sizes, accessory_labels, accessory_paths = accessory

        self.log(f"{len(accessory_sizes)} different accessory datasets specified")

        # Load model and parameters
        network_dir = self.config["network"]

        if network_dir in [
            "autophagy_classifier1.0",
            "autophagy_classifier2.0",
            "autophagy_classifier2.1",
        ]:
            if network_dir == "autophagy_classifier1.0":
                from sparcscore.ml.pretrained_models import autophagy_classifier1_0

                model = autophagy_classifier1_0(device=self.config["inference_device"])
            elif network_dir == "autophagy_classifier2.0":
                from sparcscore.ml.pretrained_models import autophagy_classifier2_0

                model = autophagy_classifier2_0(device=self.config["inference_device"])
            elif network_dir == "autophagy_classifier2.1":
                from sparcscore.ml.pretrained_models import autophagy_classifier2_1

                model = autophagy_classifier2_1(device=self.config["inference_device"])
            else:
                sys.exit("incorrect specification for pretrained model.")
        else:
            checkpoint_path = os.path.join(network_dir, "checkpoints")
            self.log(f"Checkpoints being read from path: {checkpoint_path}")
            checkpoints = [
                name
                for name in os.listdir(checkpoint_path)
                if os.path.isfile(os.path.join(checkpoint_path, name))
            ]
            checkpoints = [
                x for x in checkpoints if x.endswith(".ckpt")
            ]  # ensure we only have actualy checkpoint files
            checkpoints.sort()

            if len(checkpoints) < 1:
                raise ValueError(
                    f"No model parameters found at: {self.config['network']}"
                )

            # ensure that the most recent version is used if more than one is saved
            if len(checkpoints) > 1:
                # get max epoch number
                epochs = [int(x.split("epoch=")[1].split("-")[0]) for x in checkpoints]
                if self.config["epoch"] == "max":
                    max_value = max(epochs)
                    max_index = epochs.index(max_value)
                    self.log(f"Maximum epoch number found {max_value}")

                    # get checkpoint with the max epoch number
                    latest_checkpoint_path = os.path.join(
                        checkpoint_path, checkpoints[max_index]
                    )
                elif isinstance(self.config["epoch"], int):
                    _index = epochs.index(self.config["epoch"])
                    self.log(f"Using epoch number {self.config['epoch']}")

                    # get checkpoint with the max epoch number
                    latest_checkpoint_path = os.path.join(
                        checkpoint_path, checkpoints[_index]
                    )

            else:
                latest_checkpoint_path = os.path.join(checkpoint_path, checkpoints[0])

            # add log message to ensure that it is always 100% transparent which classifier is being used
            self.log(
                f"Using the following classifier checkpoint: {latest_checkpoint_path}"
            )
            hparam_path = os.path.join(network_dir, "hparams.yaml")

            model = MultilabelSupervisedModel.load_from_checkpoint(
                latest_checkpoint_path,
                hparams_file=hparam_path,
                type=self.config["classifier_architecture"],
                map_location=self.config["inference_device"],
            )

        model.eval()
        model.to(self.config["inference_device"])

        self.log(f"model parameters loaded from {self.config['network']}")

        # generate project dataset dataloader
        # transforms like noise, random rotations, channel selection are still hardcoded
        t = transforms.Compose(
            [ChannelSelector([self.config["channel_classification"]])]
        )

        self.log(f"loading {extraction_dir}")

        # redirect stdout to capture dataset size
        f = io.StringIO()
        with redirect_stdout(f):
            dataset = HDF5SingleCellDataset(
                [extraction_dir], [0], "/", transform=t, return_id=True
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
                    [accessory_paths[i]], [i + 1], "/", transform=t, return_fake_id=True
                )

            if len(local_dataset) > accessory_sizes[i]:
                residual = len(local_dataset) - accessory_sizes[i]
                local_dataset, _ = torch.utils.data.random_split(
                    local_dataset, [accessory_sizes[i], residual]
                )

            dataset = torch.utils.data.ConcatDataset([dataset, local_dataset])

        # log stdout
        out = f.getvalue()
        self.log(out)

        # classify samples
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["dataloader_worker_number"],
            shuffle=True,
        )

        self.log(f"log transform: {self.config['log_transform']}")

        # extract which inferences to make from config file
        encoders = self.config["encoders"]
        for encoder in encoders:
            if encoder == "forward":
                self.inference(dataloader, model.network.forward)
            if encoder == "encoder":
                self.inference(dataloader, model.network.encoder)

    def inference(self, dataloader, model_fun):
        # 1. performs inference for a dataloader and a given network call
        # 2. saves the results to file

        data_iter = iter(dataloader)
        self.log(
            f"Start processing {len(data_iter)} batches with {model_fun.__name__} based inference"
        )
        with torch.no_grad():
            x, label, class_id = next(data_iter)
            r = model_fun(x.to(self.config["inference_device"]))
            result = r.cpu().detach()

            for i in range(len(dataloader) - 1):
                if i % 10 == 0:
                    self.log(f"processing batch {i}")
                x, _label, id = next(data_iter)

                r = model_fun(x.to(self.config["inference_device"]))
                result = torch.cat((result, r.cpu().detach()), 0)
                label = torch.cat((label, _label), 0)
                class_id = torch.cat((class_id, id), 0)

        result = result.detach().numpy()

        if self.config["log_transform"]:
            sigma = 1e-9
            result = np.log(result + sigma)

        label = label.numpy()
        class_id = class_id.numpy()

        # save inferred activations / predictions
        result_labels = [f"result_{i}" for i in range(result.shape[1])]

        dataframe = pd.DataFrame(data=result, columns=result_labels)
        dataframe["label"] = label
        dataframe["cell_id"] = class_id.astype("int")

        self.log("finished processing")

        path = os.path.join(
            self.run_path, f"dimension_reduction_{model_fun.__name__}.tsv"
        )
        dataframe.to_csv(path)

class EnsembleClassifier(ProcessingStep):
    """
    This class takes a pre-trained ensemble of models and uses it to classify extracted single cell datasets.
    """

    DEFAULT_LOG_NAME = "processing.log"
    DEFAULT_FILE_NAME = "single_cells.h5"

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
                "/",
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

    def inference(self, dataloader, model_ensemble):
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

        path = os.path.join(
            self.directory, f"ensemble_inference_{self.ensemble_name}.csv"
        )
        dataframe.to_csv(path, sep=",")

        self.log(f"Results saved to file: {path}")

    def __call__(self, extraction_dir):
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
            f"{extraction_dir}/{self.DEFAULT_FILE_NAME}"
        )

        # perform inference
        self.inference(dataloader=dataloader, model_ensemble=model_ensemble)


class CellFeaturizer:
    """
    Class for extracting general image features from SPARCS single-cell image datasets.
    The extracted features are saved to a TSV file. The features are calculated on the basis of a specified channel.

    The features which are calculated are:

    - area of the nucleus in px,
    - area of the cytosol in px,
    - mean intensity of chosen channel
    - median intensity of chosen channel,
    - 75% quantile of chosen channel,
    - 25% quantile of chosen channel,
    - summed intensity of the chosen channel in the region labeled as nucleus,
    - summed intensity of the chosen channel in the region labeled as cyotosl,
    - summed intensity of the chosen channel in the region labelled as nucleus normalized to the nucleus area,
    - summed intensity of the chosen channel in the region labelled as cytosol normalized to the cytosol area, nucleus_area

    The features are outputed in this order in the tsv file.

    """

    DEFAULT_LOG_NAME = "processing.log"
    DEFAULT_DATA_DIR = "data"
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
        """Class is initiated to featurize extracted single cells.

        Parameters
        ----------
        config : dict
            Configuration for the extraction passed over from the :class:`pipeline.Project`.

        path : str
            Directory for the extraction log and results. Will be created if not existing yet.

        debug : bool, optional, default=False
            Flag used to output debug information and map images.

        overwrite : bool, optional, default=False
            Flag used to recalculate all images, not yet implemented.
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

    def get_timestamp(self):
        # Returns the current date and time as a formatted string.

        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        return "[" + dt_string + "] "

    def log(self, message):
        # Writes a message to a log file and prints it to the console if debug is True.

        log_path = os.path.join(self.run_path, self.DEFAULT_LOG_NAME)

        if isinstance(message, str):
            lines = message.split("\n")

        if isinstance(message, list):
            lines = message

        if isinstance(message, dict):
            lines = []
            for key, value in message.items():
                lines.append(f"{key}: {value}")

        for line in lines:
            with open(log_path, "a") as myfile:
                myfile.write(self.get_timestamp() + line + " \n")

            if self.debug:
                print(self.get_timestamp() + line)

    def __call__(
        self,
        extraction_dir,
        accessory,
        size=0,
        project_dataloader=HDF5SingleCellDataset,
        accessory_dataloader=HDF5SingleCellDataset,
    ):
        """
        Function called to perform featurization on the provided HDF5 dataset.

        Parameters
        ----------
            extraction_dir : str
                directory containing the extracted HDF5 files from the project. If this class is used as part of a project processing workflow this argument will be provided automatically.
            accessory : list
                list containing accessory datasets on which inference should be performed in addition to the cells contained within the current project
            size : int, default = 0
                How many cells should be selected for inference. Default is 0, then all cells are selected.

        Returns
        -------
            Writes results to tsv files located in the project directory.

        Important:

            If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project``
            class based on the previous single-cell extraction. Therefore, only the second and third argument need to be provided. The Project
            class will automaticly provide the most recent extraction results together with the supplied parameters.


        Example:

            .. code-block:: python

                # define acceossory dataset: additional hdf5 datasets that you want to perform an inference on
                # leave empty if you only want to infere on all extracted cells in the current project

                accessory = ([], [], [])
                project.classify(accessory = accessory)

        Note:

            The following parameters are required in the config file:

            .. code-block:: yaml

                CellFeaturizer:
                    # channel number on which the featurization should be performed
                    channel_classification: 4

                    #number of threads to use for dataloader
                    dataloader_worker_number: 0 #needs to be 0 if using cpu

                    #batch size to pass to GPU
                    batch_size: 900

                    # on which device inference should be performed
                    # for speed should be "cuda"
                    inference_device: "cpu"

                    #label under which the results should be saved
                    screen_label: "Ch3_Featurization"

        """

        # is called with the path to the segmented image
        # Size: number of datapoints of the project dataset considered
        # ===== Dataloaders =====
        # should be HDF5SingleCellDataset for .h5 datasets
        # project_dataloader: dataloader for the project dataset
        # accessory_dataloader: dataloader for the accesssory datasets

        self.log("Started classification")
        self.log(f"starting with run {self.current_run}")
        self.log(self.config)

        accessory_sizes, accessory_labels, accessory_paths = accessory

        self.log(f"{len(accessory_sizes)} different accessory datasets specified")

        # generate project dataset dataloader
        t = transforms.Compose([])

        self.log(f"loading {extraction_dir}")

        # redirect stdout to capture dataset size
        f = io.StringIO()
        with redirect_stdout(f):
            dataset = HDF5SingleCellDataset(
                [extraction_dir], [0], "/", transform=t, return_id=True
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
                    [accessory_paths[i]], [i + 1], "/", transform=t, return_fake_id=True
                )

            if len(local_dataset) > accessory_sizes[i]:
                residual = len(local_dataset) - accessory_sizes[i]
                local_dataset, _ = torch.utils.data.random_split(
                    local_dataset, [accessory_sizes[i], residual]
                )

            dataset = torch.utils.data.ConcatDataset([dataset, local_dataset])

        # log stdout
        out = f.getvalue()
        self.log(out)

        # classify samples
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["dataloader_worker_number"],
            shuffle=False,
        )
        self.inference(dataloader)

    def calculate_statistics(self, img, channel=-1):
        # get batch size
        N, _, _, _ = img.shape

        # calculate area statistics
        nucleus_mask = img[:, 0] > 0
        nucleus_area = nucleus_mask.view(N, -1).sum(1, keepdims=True)
        cytosol_mask = img[:, 1] > 0
        cytosol_area = cytosol_mask.view(N, -1).sum(1, keepdims=True)

        # select channel to calculate summary statistics over
        img_selected = img[:, channel]
        mean = img_selected.view(N, -1).mean(1, keepdim=True)
        median = img_selected.view(N, -1).quantile(q=0.5, dim=1, keepdim=True)
        quant75 = img_selected.view(N, -1).quantile(q=0.75, dim=1, keepdim=True)
        quant25 = img_selected.view(N, -1).quantile(q=0.25, dim=1, keepdim=True)

        # calculate more complex statistics
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
        summed_intensity_cytosol_area = img_selected.view(N, -1).sum(1, keepdims=True)
        summed_intensity_cytosol_area_normalized = (
            summed_intensity_cytosol_area / cytosol_area
        )

        # generate results tensor with all values and return
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

    def inference(self, dataloader):
        # 1. performs inference for a dataloader
        # 2. saves the results to file

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

        # save inferred activations / predictions
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

        path = os.path.join(self.run_path, "calculated_features.tsv")
        dataframe.to_csv(path)
