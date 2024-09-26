import io
import os
import platform
import shutil
from contextlib import redirect_stdout
from functools import partial as func_partial
from typing import List, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from anndata import AnnData
from spatialdata.models import TableModel
from torchvision import transforms

from scportrait.pipeline._base import ProcessingStep
from scportrait.tools.ml.datasets import HDF5SingleCellDataset
from scportrait.tools.ml.plmodels import MultilabelSupervisedModel


class _ClassificationBase(ProcessingStep):
    PRETRAINED_MODEL_NAMES = [
        "autophagy_classifier",
    ]
    MASK_NAMES = ["nucleus", "cytosol"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = self.config["label"]
        self.num_workers = self.config["dataloader_worker_number"]
        self.batch_size = self.config["batch_size"]

        self.model_class = None
        self.model = None
        self.transforms = None
        self.expected_imagesize = None

        self._setup_channel_classification()

        # setup deep debugging
        self.deep_debug = False

        if "overwrite_run_path" not in self.__dict__.keys():
            self.overwrite_run_path = self.overwrite

    def _setup_output(self):
        """Helper function to generate the output directory for the classification results."""

        # Create classification directory
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        self.run_path = os.path.join(self.directory, f"{self.data_type}_{self.label}")

        if not os.path.isdir(self.run_path):
            os.makedirs(self.run_path)
            self.log(f"Created new directory for classification results: {self.run_path}")
        else:
            if self.overwrite:
                self.log("Overwrite flag is set, deleting existing directory for classification results.")
                shutil.rmtree(self.run_path)
                os.makedirs(self.run_path)
                self.log(f"Created new directory for classification results: {self.run_path}")
            elif self.overwrite_run_path:
                self.log("Overwrite flag is set, deleting existing directory for classification results.")
                shutil.rmtree(self.run_path)
                os.makedirs(self.run_path)
                self.log(f"Created new directory for classification results: {self.run_path}")
            else:
                raise ValueError(
                    f"Directory for classification results already exists at {self.run_path}. Please set the overwrite flag to True if you wish to overwrite the existing directory."
                )

    def _setup_log_transform(self):
        if "log_transform" in self.config.keys():
            self.log_transform = self.config["log_transform"]
        else:
            self.log_transform = False  # default value

    def _setup_channel_classification(self):
        if "channel_classification" in self.config.keys():
            self.channel_classification = self.config["channel_classification"]
        else:
            self.channel_classification = None

    def _detect_automatic_inference_device(self):
        """Automatically detect the best inference device available on the system."""

        if torch.cuda.is_available():
            inference_device = "cuda"
        if torch.backends.mps.is_available():
            inference_device = torch.device("mps")
        else:
            inference_device = "cpu"

        return inference_device

    def _setup_inference_device(self):
        """
        Configure the classification run to use the specified inference device.
        If no device is specified, the device is automatically detected.
        """

        if "inference_device" in self.config.keys():
            self.inference_device = self.config["inference_device"]

            # check that the selected inference device is also available
            if self.inference_device in ("cuda", torch.device("cuda")):
                if not torch.cuda.is_available():
                    if torch.backends.mps.is_available():
                        self.log(
                            "CUDA specified in config file but CUDA not available on system, switching to MPS for inference."
                        )
                        self.inference_device = torch.device("mps")
                    else:
                        self.log(
                            "CUDA specified in config file but CUDA not available on system, switching to CPU for inference."
                        )
                        self.inference_device = "cpu"
                else:
                    self.inference_device = torch.device("cuda")  # ensure that the complete device is always specified

            elif self.inference_device in ("mps", torch.device("mps")):
                if not torch.backends.mps.is_available():
                    if torch.cuda.is_available():
                        self.log(
                            "MPS specified in config file but MPS not available on system, switching to CUDA for inference."
                        )
                        self.inference_device = "cuda"
                    else:
                        self.log(
                            "MPS specified in config file but MPIS not available on system, switching to CPU for inference."
                        )
                        self.inference_device = "cpu"
                else:
                    self.inference_device = torch.device("mps")  # ensure that the complete device is always specified

            elif self.inference_device in ("automatic", "auto"):
                self.inference_device = self._detect_automatic_inference_device()
                self.log(f"Automatically configured inference device to {self.inference_device}")

            elif self.inference_device == "cpu":
                if torch.backends.mps.is_available():
                    self.log(
                        "CPU specified in config file but MPS available on system. Consider changing the device for the next run."
                    )
                if torch.cuda.is_available():
                    self.log(
                        "CPU specified in config file but CUDA available on system. Consider changing the device for the next run."
                    )
            else:
                raise ValueError(
                    "Invalid inference device specified in config file. Please use one of ['cuda', 'mps', 'cpu', 'automatic', 'auto']."
                )

        else:
            self.inference_device = self._detect_automatic_inference_device()
            self.log(f"Automatically configured inferece device to {self.inference_device}")

    def _general_setup(self):
        """Helper function to execute all setup functions that are common to all classification steps."""

        self._setup_output()
        self._setup_log_transform()
        self._setup_inference_device()

    def _get_model_specs(self):
        # model location
        self.network_dir = self.config["network"]

        # hparams locatoin
        if "hparams_path" in self.config.keys():
            self.hparams_path = self.config["hparams_path"]
        else:
            self.hparams_path = None

        # model loading strategy
        if "model_loading_strategy" in self.config.keys():
            strategy = self.config["model_loading_strategy"]
            if strategy not in ("max", "min", "latest", "path"):
                raise ValueError(
                    f"Invalid model loading strategy {strategy} specified. Please use one of ['max', 'min', 'latest', 'path']"
                )

            self.model_loading_strategy = self.config["model_loading_strategy"]
        else:
            self.model_loading_strategy = "max"

        # modelclass
        if self.model_class is None:
            if "model_class" in self.config.keys():
                self.define_model_class(eval(self.config["model_class"]))
            else:
                self.define_model_class(self.DEFAULT_MODEL_CLASS)  # default model class
        else:
            self.log(
                f"Model class already defined as {self.model_class} will not overwrite. If this behaviour was unintended please set the model class to none by executing 'project.classification_f.model_class = None'"
            )

        if "model_type" in self.config.keys():
            self.model_type = self.config["model_type"]
        else:
            self.model_type = None

    def _get_gpu_memory_usage(self):
        if self.inference_device == "cpu":
            return None

        elif self.inference_device == torch.device("cuda"):
            try:
                memory_usage = []

                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_reserved(i) / 1024**2  # Convert bytes to MiB
                    memory_usage.append(gpu_memory)
                results = {f"GPU_{i}": f"{memory_usage[i]} MiB" for i in range(len(memory_usage))}
                return results
            except (RuntimeError, ValueError) as e:
                print("Error:", e)
                return None

        elif self.inference_device == torch.device("mps"):
            try:
                used_memory = torch.mps.driver_allocated_memory() + torch.mps.driver_allocated_memory()
                used_memory = used_memory / 1024**2  # Convert bytes to MiB
                return {"MPS": f"{memory_usage} MiB"}
            except (RuntimeError, ValueError) as e:
                print("Error:", e)
                return None

        else:
            raise ValueError("Invalid inference device specified.")

    ### Functions for model loading and setup

    def _assign_model(self, model):
        self.log("Model assigned to classification function.")
        self.model = model

        # check if the hparams specify an expected image size
        if "expected_imagesize" in model.hparams.keys():
            self.expected_imagesize = model.hparams["expected_imagesize"]

    def define_model_class(self, model_class, force_load=False):
        if isinstance(model_class, str):
            model_class = eval(model_class)  # convert string to class by evaluating it

        # check that it is a valid model class

        if force_load:
            if not issubclass(model_class, pl.LightningModule):
                Warning(
                    "Forcing the loading of the model class despite the provided model class not being a subclass of pl.LightningModule. This can lead to unexpected behaviour."
                )
            else:
                Warning(
                    "Forcing the loading of the model class is on but the provided model class is a subclass of pl.LightningModule. Consider setting the force_load parameter to False as this is not a recommended default setup."
                )
        else:
            if not issubclass(model_class, pl.LightningModule):
                raise ValueError(
                    "The provided model class is not a subclass of pl.LightningModule. Please provide a valid model class. If you are sure you wish to proceed with this, please reload the model class with the force_load parameter set to True."
                )

        self.model_class = model_class
        self.log(f"Model class defined as {model_class}")

    def _load_pretrained_model(self, model_name: str):
        """
        Load a pretrained model from the SPARCScore library.

        Parameters
        ----------
        model_name : str
            Name of the pretrained model to load.

        Returns
        -------
        MultilabelSupervisedModel
            The loaded model.
        """

        if model_name == "autophagy_classifier":
            from scportrait.tools.ml.pretrained_models import autophagy_classifier

            model = autophagy_classifier(device=self.config["inference_device"])
            self.expected_imagesize = (128, 128)
        else:
            raise ValueError(
                f"Invalid model name {model_name} specified for pretrained model. The available pretrained models are {self.PRETRAINED_MODEL_NAMES}"
            )

        # set to eval mode and move to device
        model.eval()
        model.to(self.inference_device)

        return model

    def _load_model(
        self,
        ckpt_path,
        hparams_path: Union[str, None] = None,
        model_type: Union[str, None] = None,
    ) -> pl.LightningModule:
        """Load a model from a checkpoint file and transfer it to the inference device.

        Parameters
        ----------
        ckpt_path : str
            Path to the checkpoint file.
        hparams_path : str, optional
            Path to the hparams file. If not provided, the hparams file is assumed to be in the same directory as the checkpoint file.
        model_type : str, optional
            Type of the model architecture to load. Default is None. For MultiLabelSupervisedModel, this can also be specified in the hparams file under the key model_type.

        Returns
        -------
        pl.LightningModule
            The loaded model.
        """

        if self.model_class is None:
            raise ValueError("Model class not defined. Please define a model class before loading a model.")

        self.log(f"Loading model from checkpoint path: {ckpt_path}")

        # get path to hparams file

        if hparams_path is None:
            if self.debug:
                self.log(
                    "No hparams file provided, assuming hparams file is in the same directory as the checkpoint file. Trying to load from default location."
                )
            hparams_path = ckpt_path.replace(os.path.basename(ckpt_path), "hparams.yml")

            if not os.path.isfile(hparams_path):
                hparams_path = ckpt_path.replace(os.path.basename(ckpt_path), "hparams.yaml")

                if not os.path.isfile(hparams_path):
                    raise ValueError(
                        f"No hparams file found at {hparams_path}. Please provide a valid hparams file path."
                    )

            if self.debug:
                self.log(f"Loading hparams file from {hparams_path}")

        if model_type is None:
            model = self.model_class.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                hparams_file=hparams_path,
                map_location=self.inference_device,
            )
        else:
            model = self.model_class.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                hparams_file=hparams_path,
                map_location=self.inference_device,
                model_type=model_type,
            )

        model = model.eval()
        model.to(self.inference_device)

        if self.debug:
            self.log(f"model loaded, set to eval mode and transferred to device {self.inference_device}")

        return model

    def load_model(
        self,
        ckpt_path,
        hparams_path: Union[str, None] = None,
        model_type: Union[str, None] = None,
    ):
        model = self._load_model(ckpt_path, hparams_path, model_type)
        self._assign_model(model)

    ### Functions regarding dataloading and transforms ####
    def configure_transforms(self, selected_transforms: List):
        self.transforms = transforms.Compose(selected_transforms)
        self.log(f"The following transforms were applied: {self.transforms}")

    def generate_dataloader(
        self,
        extraction_dir: str,
        selected_transforms: transforms.Compose = transforms.Compose([]),
        size: int = 0,
        seed: Union[int, None] = 42,
        dataset_class=HDF5SingleCellDataset,
    ) -> torch.utils.data.DataLoader:
        """Create a pytorch dataloader from the provided single-cell image dataset.

        Parameters
        ----------
        extraction_dir : str
            Path to the directory containing the extracted single-cell images.
        selected_transforms : list of torchvision.transforms
            List of transforms to apply to the images.
        size : int, optional
            Number of cells to select from the dataset. Default is 0, which means all samples are selected.
        seed : int, optional
            Seed for the random number generator if splitting the dataset and only using a subset. Default is 42.

        Returns
        -------
        torch.utils.data.DataLoader
            The generated dataloader.

        """
        # generate dataset
        self.log(f"Reading data from path: {extraction_dir}")

        assert isinstance(
            self.transforms, transforms.Compose
        ), f"Transforms should be a torchvision.transforms.Compose object but recieved {self.transforms.__class__} instead."
        t = self.transforms

        if self.expected_imagesize is not None:
            self.log(f"Expected image size is set to {self.expected_imagesize}. Resizing images to this size.")
            t = transforms.Compose([t, transforms.Resize(self.expected_imagesize)])

        f = io.StringIO()
        with redirect_stdout(f):
            dataset = dataset_class(
                [extraction_dir],
                [0],
                "/",
                transform=t,
                return_id=True,
                select_channel=self.channel_classification,
            )

        if size > 0:
            if size > len(dataset):
                raise ValueError(
                    f"Selected size {size} is larger than the dataset size {len(dataset)}. Please select a smaller size."
                )
            if size < len(dataset):
                residual_size = len(dataset) - size
                if seed is not None:
                    self.log(f"Using a seeded generator with seed {seed} to split dataset")
                    gen = torch.Generator()
                    gen.manual_seed(seed)
                    dataset, _ = torch.utils.data.random_split(dataset, [size, residual_size], generator=gen)
                else:
                    self.log("Using a random generator to split dataset.")
                    dataset, _ = torch.utils.data.random_split(dataset, [size, residual_size])
            # randomly select n elements from the dataset to process
            dataset = torch.utils.data.Subset(dataset, range(size))

        # check operating system
        if platform.system() == "Windows":
            context = "spawn"
            num_workers = 0  # need to disable multiprocessing otherwise it will throw an error
        elif platform.system() == "Darwin":
            context = "fork"
            num_workers = self.num_workers
        elif platform.system() == "Linux":
            context = "fork"
            num_workers = self.num_workers

        # generate dataloader

        if num_workers > 0:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                multiprocessing_context=context,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )

        self.log(
            f"Dataloader generated with a batchsize of {self.batch_size} and {self.num_workers} workers. Dataloader contains {len(dataloader)} entries."
        )

        return dataloader

    #### Inference functions ####
    def inference(self, dataloader, model_fun, column_names=None) -> pd.DataFrame:
        """
        # 1. performs inference for a dataloader and a given network call
        # 2. saves the results to file
        """
        self.log(f"Started processing of {len(dataloader)} batches.")

        data_iter = iter(dataloader)
        with torch.no_grad():
            x, label, class_id = next(data_iter)
            r = model_fun(x.to(self.inference_device))
            result = r.cpu().detach()

            # add check to ensure this only runs if we have more than one batch in the dataset
            if len(dataloader) > 1:
                for i in range(len(dataloader) - 1):
                    if i % 10 == 0:
                        self.log(f"processing batch {i}")
                    x, _label, id = next(data_iter)

                    r = model_fun(x.to(self.inference_device))
                    result = torch.cat((result, r.cpu().detach()), 0)
                    label = torch.cat((label, _label), 0)
                    class_id = torch.cat((class_id, id), 0)

        result = result.detach().numpy()

        if self.log_transform:
            self.log("Applying log transformation to results.")
            sigma = 1e-9  # to avoid log(0)
            result = np.log(result + sigma)

        label = label.numpy()
        class_id = class_id.numpy()

        # save inferred activations / predictions

        if column_names is None:
            column_names = [f"result_{i}" for i in range(result.shape[1])]

        dataframe = pd.DataFrame(data=result, columns=column_names)
        dataframe["label"] = label
        dataframe["cell_id"] = class_id.astype("int")

        self.log("finished processing.")

        return dataframe

    #### Results writing functions ####

    def _write_results_csv(self, results, path):
        results.to_csv(path, index=False)
        self.log(f"Results saved to file: {path}")

    def _write_results_sdata(self, results, label, mask_type="seg_all"):
        results.set_index("cell_id", inplace=True)
        results.drop(columns=["label"], inplace=True)

        feature_matrix = results.to_numpy()
        var_names = results.columns
        obs_indices = results.index.astype(str)

        if self.project.nuc_seg_status:
            # save nucleus segmentation
            obs = pd.DataFrame()
            obs.index = obs_indices
            obs["instance_id"] = obs_indices
            obs["region"] = f"{mask_type}_{self.MASK_NAMES[0]}"
            obs["region"] = obs["region"].astype("category")

            table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
            table = TableModel.parse(
                table,
                region=[f"{mask_type}_{self.MASK_NAMES[0]}"],
                region_key="region",
                instance_key="instance_id",
            )

            self.project._write_table_object_sdata(
                table,
                f"{self.__class__.__name__ }_{label}_{self.MASK_NAMES[0]}",
                overwrite=self.overwrite_run_path,
            )

        if self.project.cyto_seg_status:
            # save cytoplasm segmentation
            obs = pd.DataFrame()
            obs.index = obs_indices
            obs["instance_id"] = obs_indices
            obs["region"] = f"{mask_type}_{self.MASK_NAMES[1]}"
            obs["region"] = obs["region"].astype("category")

            table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
            table = TableModel.parse(
                table,
                region=[f"{mask_type}_{self.MASK_NAMES[1]}"],
                region_key="region",
                instance_key="instance_id",
            )

            self.project._write_table_object_sdata(
                table,
                f"{self.__class__.__name__ }_{label}_{self.MASK_NAMES[1]}",
                overwrite=self.overwrite_run_path,
            )

    #### Cleanup Functions ####

    def _post_processing_cleanup(self):
        if self.debug:
            memory_usage = self._get_gpu_memory_usage()
            self.log(f"GPU memory before performing cleanup: {memory_usage}")

        if "dataloader" in self.__dict__.keys():
            del self.dataloader

        if "models" in self.__dict__.keys():
            del self.models

        if "model" in self.__dict__.keys():
            del self.model

        if "overwrite_run_path" in self.__dict__.keys():
            del self.overwrite_run_path

        if "n_masks" in self.__dict__.keys():
            del self.n_masks

        if "data_type" in self.__dict__.keys():
            del self.data_type

        if "log_transform" in self.__dict__.keys():
            del self.log_transform

        if "channel_names" in self.__dict__.keys():
            del self.channel_names

        if "column_names" in self.__dict__.keys():
            del self.column_names

        # reset to init values to ensure that subsequent runs are not affected by previous runs
        self.model_class = None
        self.transforms = None
        self.channel_classification = None
        self.model = None

        self._clear_cache()

        if self.debug:
            memory_usage = self._get_gpu_memory_usage()
            self.log(f"GPU memory after performing cleanup: {memory_usage}")

        # this needs to be called after the memory usage has been assesed
        if "inference_device" in self.__dict__.keys():
            del self.inference_device


###############################################
###### DeepLearning based Classification ######
###############################################


class MLClusterClassifier(_ClassificationBase):
    """
    Class for classifying single cells using a pre-trained machine learning model.

    This class takes a pre-trained model and uses it to classify single cells,
    using the model's forward function or encoder function, depending on the
    user's choice. The classification results are saved to a CSV file.
    """

    CLEAN_LOG = True
    DEFAULT_LOG_NAME = "processing_MLClusterClassifier.log"
    DEFAULT_MODEL_CLASS = MultilabelSupervisedModel
    DEFAULT_DATA_LOADER = HDF5SingleCellDataset

    def __init__(self, *args, **kwargs):
        """
        Class is initiated to classify extracted single cells.

        Parameters
        ----------
        config : dict
            Configuration for the extraction passed over from the :class:`pipeline.Project`.

        directory : str
            Directory for the extraction log and results. Will be created if not existing yet.

        debug : bool, optional, default=False
            Flag used to output debug information and map images.

        overwrite : bool, optional, default=False
            Flag used to overwrite existing results.
        """
        super().__init__(*args, **kwargs)

        if self.CLEAN_LOG:
            self._clean_log_file()

    def _get_network_dir(self) -> pl.LightningModule:
        if self.network_dir in self.PRETRAINED_MODEL_NAMES:
            pass
        else:
            if self.model_loading_strategy == "path":
                pass

            elif self.network_dir.endswith(".ckpt"):
                Warning(
                    "Provided network ends in .ckpt, assuming this is a complete model path. To avoid this warning in the future please set the config parameter 'model_loading_strategy' to 'path'."
                )
                self.log(
                    "Provided network ends in .ckpt, assuming this is a complete model path. To avoid this warning in the future please set the config parameter 'model_loading_strategy' to 'path'."
                )
            else:
                # then we assume this is a directory containing multiple checkpoints and we want to load a specific one based on the model_loading_strategy

                model_path = self.network_dir
                checkpoints = [file for file in os.listdir(model_path) if file.endswith(".ckpt")]
                checkpoints.sort()

                if len(checkpoints) < 1:
                    raise ValueError(f"No checkpoint files found at {model_path}")

                if len(checkpoints) == 1:
                    self.log("Only one checkpoint found in network directory.")

                    # update network_dir to the path of the checkpoint
                    self.network_dir = os.path.join(model_path, checkpoints[0])
                else:
                    if self.model_loading_strategy == "max":
                        max_epoch = max([int(file.split("epoch=")[1].split("-")[0]) for file in checkpoints])
                        max_epoch_file = [file for file in checkpoints if f"epoch={max_epoch}" in file][0]
                        self.network_dir = os.path.join(model_path, max_epoch_file)
                        self.log(
                            f"Using model selection strategy 'max', selecting model with the maximum epoch {max_epoch} from path {self.network_dir}"
                        )
                    elif self.model_loading_strategy == "min":
                        min_epoch = min([int(file.split("epoch=")[1].split("-")[0]) for file in checkpoints])
                        min_epoch_file = [file for file in checkpoints if f"epoch={min_epoch}" in file][0]
                        self.network_dir = os.path.join(model_path, min_epoch_file)
                        self.log(
                            f"Using model selection strategy 'min', selecting model with the minimum epoch {min_epoch} from path {self.network_dir}"
                        )
                    elif self.model_loading_strategy == "latest":
                        self.network_dir = os.path.join(model_path, checkpoints[-1])
                        self.log(
                            f"Using model selection strategy 'latest', selecting the latest model from path {self.network_dir}"
                        )
                    else:
                        raise ValueError(
                            f"Invalid model loading strategy {self.model_loading_strategy} specified. Please use one of ['max', 'min', 'latest'] if not provding a path to a model cpkt."
                        )

    def _setup_encoders(self):
        # extract which inferences to make from config file
        if "encoders" in self.config.keys():
            encoders = self.config["encoders"]
        else:
            encoders = ["forward"]  # default parameter

        self.models = []

        for encoder in encoders:
            if encoder == "forward":
                self.models.append(self.model.network.forward)
            if encoder == "encoder":
                self.models.append(self.model.network.encoder)

    def _setup_transforms(self) -> None:
        if self.transforms is not None:
            self.log(
                "Transforms already configured manually. Will not overwrite. If this behaviour was unintended please set the transforms to None by executing 'project.classification_f.transforms = None'"
            )
            return

        if "transforms" in self.config.keys():
            self.transforms = eval(self.config["transforms"])
        else:
            self.transforms = transforms.Compose([])  # default is no transforms

        return

    def _setup(self):
        self._general_setup()
        self._get_model_specs()
        self._get_network_dir()

        if self.network_dir in self.PRETRAINED_MODEL_NAMES:
            model = self._load_pretrained_model(self.network_dir)
        else:
            model = self._load_model(
                ckpt_path=self.network_dir,
                hparams_path=self.hparams_path,
                model_type=self.model_type,
            )

        self._assign_model(model)

        self._setup_encoders()
        self._setup_transforms()

    def process(self, extraction_dir: str, size: int = 0):
        """
        Perform classification on the provided HDF5 dataset.

        Parameters
        ----------
        extraction_dir : str
            Directory containing the extracted HDF5 files from the project. If this class is used as part of
            a project processing workflow, this argument will be provided automatically.
        size : int, optional
            How many cells should be selected for inference. Default is 0, which means all cells are selected.

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
                # Channel number on which the classification should be performed
                channel_classification: 4

                # Number of threads to use for dataloader
                dataloader_worker_number: 24

                # Batch size to pass to GPU
                batch_size: 900

                # Path to PyTorch checkpoint that should be used for inference
                network: "path/to/model/"

                # Classifier architecture implemented in scPortrait
                # Choose one of VGG1, VGG2, VGG1_old, VGG2_old
                classifier_architecture: "VGG2_old"

                # If more than one checkpoint is provided in the network directory, which checkpoint should be chosen
                # Should either be "max" or a numeric value indicating the epoch number
                epoch: "max"

                # Name of the classifier used for saving the classification results to a directory
                label: "Autophagy_15h_classifier1"

                # List of which inference methods should be performed
                # Available: "forward" and "encoder"
                # If "forward": images are passed through all layers of the model and the final inference results are written to file
                # If "encoder": activations at the end of the CNN are written to file
                encoders: ["forward", "encoder"]

                # On which device inference should be performed
                # For speed, should be "cuda"
                inference_device: "cuda"

                #define dataset transforms
                transforms:
                    resize: 128

        """
        self.log("Started MLClusterClassifier classification.")

        # perform setup
        self._setup()

        self.dataloader = self.generate_dataloader(
            extraction_dir,
            selected_transforms=self.transforms,
            size=size,
            dataset_class=self.DEFAULT_DATA_LOADER,
        )

        # perform inference
        for model in self.models:
            self.log(f"Starting inference for model encoder {model.__name__}")
            results = self.inference(self.dataloader, model)

            output_name = f"inference_{model.__name__}"
            path = os.path.join(self.run_path, f"{output_name}.csv")

            self._write_results_csv(results, path)
            self._write_results_sdata(results, label=f"{self.label}_{model.__name__}")

        self.log(f"Results saved to file: {path}")

        # perform post processing cleanup
        if not self.deep_debug:
            self._post_processing_cleanup()


class EnsembleClassifier(_ClassificationBase):
    """
    This class takes a pre-trained ensemble of models and uses it to classify extracted single cell datasets.
    """

    CLEAN_LOG = True
    DEFAULT_LOG_NAME = "processing_EnsembleClassifier.log"
    DEFAULT_MODEL_CLASS = MultilabelSupervisedModel
    DEFAULT_DATA_LOADER = HDF5SingleCellDataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.CLEAN_LOG:
            self._clean_log_file()

    def _setup_transforms(self):
        if self.transforms is not None:
            self.log(
                "Transforms already configured manually. Will not overwrite. If this behaviour was unintended please set the transforms to None by executing 'project.classification_f.transforms = None'"
            )
            return
        else:
            self.transforms = transforms.Compose([])

    def _load_models(self):
        # load models and generate ensemble
        self.model = []
        self.model_names = []

        for model_name, model_path in self.network_dir.items():
            model = self._load_model(ckpt_path=model_path, hparams_path=self.hparams_path)

            # check for hparams expected_imagesize
            if self.expected_imagesize is None:
                if "expected_imagesize" in model.hparams.keys():
                    self.expected_imagesize = model.hparams["expected_imagesize"]
            else:
                if "expected_imagesize" in model.hparams.keys():
                    if self.expected_imagesize != model.hparams["expected_imagesize"]:
                        raise ValueError("Expected image sizes of models in ensemble do not match.")

            self.model.append(model)
            self.model_names.append(model_name)

        self.log(f"Model Ensemble generated with a total of {len(self.model)} models.")
        memory_usage = self._get_gpu_memory_usage()
        self.log(f"GPU memory usage after loading models: {memory_usage}")

    def _setup(self):
        self._general_setup()
        self._get_model_specs()
        self._setup_transforms()

        # ensure that the network_dir is a dictionary
        if not isinstance(self.network_dir, dict):
            raise ValueError(
                "network_dir should be a dictionary containing the model names and paths to the model checkpoints."
            )

        self._load_models()

    def process(self, extraction_dir, size=0):
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

        self._setup()

        self.dataloader = self.generate_dataloader(
            extraction_dir,
            selected_transforms=self.transforms,
            size=size,
            dataset_class=self.DEFAULT_DATA_LOADER,
        )

        # perform inference
        for model_name, model in zip(self.model_names, self.model):
            self.log(f"Starting inference for model {model_name}")
            results = self.inference(self.dataloader, model)

            output_name = f"ensemble_inference_{model_name}"
            path = os.path.join(self.run_path, f"{output_name}.csv")

            self._write_results_csv(results, path)
            self._write_results_sdata(results, label=model_name)

        # perform post processing cleanup
        if not self.deep_debug:
            self._post_processing_cleanup()


####### CellFeaturization based on Classic Featurecalculation #######
class _cellFeaturizerBase(_ClassificationBase):
    CLEAN_LOG = True
    DEFAULT_DATA_LOADER = HDF5SingleCellDataset

    # define the output column names
    MASK_NAMES = ["nucleus", "cytosol", "cytosol_only"]
    MASK_STATISTICS = ["area"]
    CHANNEL_STATISTICS = [
        "mean",
        "median",
        "quant75",
        "quant25",
        "summed_intensity",
        "summed_intensity_area_normalized",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.CLEAN_LOG:
            self._clean_log_file()

    def _setup_transforms(self):
        if self.transforms is not None:
            self.log(
                "Transforms already configured manually. Will not overwrite. If this behaviour was unintended please set the transforms to None by executing 'project.classification_f.transforms = None'"
            )
            return

        self.transforms = transforms.Compose([])
        return

    def _get_channel_specs(self):
        if "channel_names" in self.project.__dict__.keys():
            self.channel_names = self.project.channel_names
        else:
            self.channel_names = self.project.input_image.c.values

    def _generate_column_names(
        self,
        n_masks: int = 2,
        n_channels: int = 3,
        channel_names: Union[List, None] = None,
    ) -> None:
        column_names = []

        if n_masks == 1:
            self.project._check_sdata_status()

            if self.project.nuc_seg_status:
                mask_name = self.MASK_NAMES[0]
            elif self.project.cyto_seg_status:
                mask_name = self.MASK_NAMES[1]
            else:
                raise ValueError("no segmentation mask found in sdata object.")
            mask_names = [mask_name]

        elif n_masks == 2:
            mask_names = self.MASK_NAMES

        # get the mask names with the mask attributes
        for mask in mask_names:
            for mask_stat in self.MASK_STATISTICS:
                column_names.append(f"{mask}_{mask_stat}")

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(n_channels)]

        for channel_name in channel_names:
            for mask in mask_names:
                for channel_stat in self.CHANNEL_STATISTICS:
                    column_names.append(f"{channel_name}_{channel_stat}_{mask}")

        self.column_names = column_names

    def calculate_statistics(self, img, n_masks=2):
        """
        Calculate statistics for an image batch.

        Parameters
        ----------
        img : torch.Tensor
            Tensor containing the image batch.
        n_masks : int
            Number of masks in the image. Masks are always the first images in the image stack. Default is 2.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated statistics.
        """
        N, _, _, _ = img.shape

        mask_statistics = []
        masks = []

        for i in range(n_masks):
            mask = img[:, i] > 0
            area = mask.view(N, -1).sum(1, keepdims=True)

            masks.append(mask)
            mask_statistics.append(area)

        if n_masks == 2:
            mask = masks[1] ^ masks[0]
            area = mask.view(N, -1).sum(1, keepdims=True)

            masks.append(mask)
            mask_statistics.append(area)

        channel_statistics = []

        for channel in range(n_masks, img.shape[1]):
            img_selected = img[:, channel]

            for mask in masks:
                mask[mask == 0] = torch.nan

                # apply mask to channel to only compute the statistics over the pixels that are relevant
                _img_selected = (img_selected * mask).to(
                    torch.float32
                )  # ensure we have correct dytpe for subsequent calculations

                mean = _img_selected.view(N, -1).nanmean(1, keepdim=True)
                median = _img_selected.view(N, -1).nanquantile(q=0.5, dim=1, keepdim=True)
                quant75 = _img_selected.view(N, -1).nanquantile(q=0.75, dim=1, keepdim=True)
                quant25 = _img_selected.view(N, -1).nanquantile(q=0.25, dim=1, keepdim=True)
                summed_intensity = _img_selected.view(N, -1).sum(1, keepdim=True)
                summed_intensity_area_normalized = summed_intensity / mask_statistics[-1]

                # save results
                channel_statistics.extend(
                    [
                        mean,
                        median,
                        quant75,
                        quant25,
                        summed_intensity,
                        summed_intensity_area_normalized,
                    ]
                )

        # Generate results tensor with all values and return
        items = mask_statistics + channel_statistics
        results = torch.concat(
            items,
            1,
        )

        return results

    def _write_results_sdata(self, results, mask_type="seg_all"):
        if self.project.nuc_seg_status:
            # save nucleus segmentation
            columns_drop = [x for x in results.columns if self.MASK_NAMES[1] in x]

            _results = results.drop(columns=columns_drop)
            _results.set_index("cell_id", inplace=True)
            _results.drop(columns=["label"], inplace=True)

            feature_matrix = _results.to_numpy()
            var_names = _results.columns
            obs_indices = _results.index.astype(str)

            obs = pd.DataFrame()
            obs.index = obs_indices
            obs["instance_id"] = obs_indices
            obs["region"] = f"{mask_type}_{self.MASK_NAMES[0]}"
            obs["region"] = obs["region"].astype("category")

            table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
            table = TableModel.parse(
                table,
                region=[f"{mask_type}_{self.MASK_NAMES[0]}"],
                region_key="region",
                instance_key="instance_id",
            )

            # define name to save table under
            self.label.replace("CellFeaturizer_", "")  # remove class name from label to ensure we dont have duplicates

            if self.channel_classification is not None:
                table_name = f"{self.__class__.__name__ }_{self.config['channel_classification']}_{self.MASK_NAMES[0]}"
            else:
                table_name = f"{self.__class__.__name__ }_{self.MASK_NAMES[0]}"

            self.project._write_table_object_sdata(table, table_name, overwrite=self.overwrite_run_path)

        if self.project.cyto_seg_status:
            # save cytosol segmentation
            columns_drop = [x for x in results.columns if self.MASK_NAMES[0] in x]

            _results = results.drop(columns=columns_drop)
            _results.set_index("cell_id", inplace=True)
            _results.drop(columns=["label"], inplace=True)

            feature_matrix = _results.to_numpy()
            var_names = _results.columns
            obs_indices = _results.index.astype(str)

            obs = pd.DataFrame()
            obs.index = obs_indices
            obs["instance_id"] = obs_indices
            obs["region"] = f"{mask_type}_{self.MASK_NAMES[1]}"
            obs["region"] = obs["region"].astype("category")

            table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
            table = TableModel.parse(
                table,
                region=[f"{mask_type}_{self.MASK_NAMES[1]}"],
                region_key="region",
                instance_key="instance_id",
            )

            # define name to save table under
            if self.channel_classification is not None:
                table_name = f"{self.__class__.__name__ }_{self.config['channel_classification']}_{self.MASK_NAMES[1]}"
            else:
                table_name = f"{self.__class__.__name__ }_{self.MASK_NAMES[1]}"

            self.project._write_table_object_sdata(table, table_name, overwrite=self.overwrite_run_path)


class CellFeaturizer(_cellFeaturizerBase):
    """
    Class for extracting general image features from SPARCS single-cell image datasets.
    The extracted features are saved to a CSV file. The features are calculated on the basis of a specified channel.

    The features which are calculated are:

    - Area of the masks in pixels
    - Mean intensity of the chosen channel in the regions labelled by each of the masks
    - Median intensity of the chosen channel in the regions labelled by each of the masks
    - 75% quantile of the chosen channel in the regions labelled by each of the masks
    - 25% quantile of the chosen channel in the regions labelled by each of the masks
    - Summed intensity of the chosen channel in the regions labelled by each of the masks
    - Summed intensity of the chosen channel in the region labelled by each of the masks normalized for area

    The features are outputed in this order in the CSV file.
    """

    DEFAULT_LOG_NAME = "processing_CellFeaturizer.log"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_classification = None  # ensure that all images are passed to the function

    def _setup(self):
        self._general_setup()
        self._setup_transforms()
        self._get_channel_specs()

    def process(self, extraction_dir, size=0):
        """
        Perform featurization on the provided HDF5 dataset.

        Parameters
        ----------
        extraction_dir : str
            Directory containing the extracted HDF5 files from the project. If this class is used as part of a project processing workflow this argument will be provided automatically.
        size : int, optional, default=0
            How many cells should be selected for inference. Default is 0, meaning all cells are selected.

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

            project.classify()

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
        self.log("Started CellFeaturization of all available channels.")

        # perform setup
        self._setup()

        self.dataloader = self.generate_dataloader(
            extraction_dir,
            selected_transforms=self.transforms,
            size=size,
            dataset_class=self.DEFAULT_DATA_LOADER,
        )

        # get first example image from dataloader
        x, _, _ = next(iter(self.dataloader))
        N, c, x, y = x.shape

        # perform sanity check on the number of channels
        assert (
            (self.n_masks + len(self.channel_names)) == c
        ), f"Number of images in the dataset ({c}) does not match the number of masks ({self.n_masks}) and channel names ({len(self.channel_names)}) specified in the project."

        # generate column names
        self._generate_column_names(n_masks=self.n_masks, n_channels=c, channel_names=self.channel_names)

        # define inference function
        f = func_partial(self.calculate_statistics, n_masks=self.n_masks)

        results = self.inference(
            self.dataloader,
            f,
            column_names=self.column_names,
        )

        output_name = "calculated_image_features"
        path = os.path.join(self.run_path, f"{output_name}.csv")

        self._write_results_csv(results, path)
        self._write_results_sdata(results)

        # perform post processing cleanup
        if not self.deep_debug:
            self._post_processing_cleanup()


class CellFeaturizer_single_channel(_cellFeaturizerBase):
    DEFAULT_LOG_NAME = "processing_CellFeaturizer.log"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_channel_selection(self):
        if self.n_masks == 2:
            self.channel_classification = [0, 1, self.channel_classification]
        if self.n_masks == 1:
            self.channel_classification = [0, self.channel_classification]
        return

    def _setup(self):
        self._general_setup()
        self._setup_channel_selection()
        self._setup_transforms()
        self._get_channel_specs()

    def process(self, extraction_dir, size=0):
        self.log(f"Started CellFeaturization of selected channel {self.channel_classification}.")

        # perform setup
        self._setup()

        self.dataloader = self.generate_dataloader(
            extraction_dir,
            selected_transforms=self.transforms,
            size=size,
            dataset_class=self.DEFAULT_DATA_LOADER,
        )

        # generate column names
        channel_name = self.channel_names[self.channel_classification[-1] - self.n_masks]
        self._generate_column_names(n_masks=self.n_masks, n_channels=1, channel_names=[channel_name])

        # define inference function
        f = func_partial(self.calculate_statistics, n_masks=self.n_masks)

        results = self.inference(
            self.dataloader,
            f,
            column_names=self.column_names,
        )

        output_name = f"calculated_image_features_Channel_{channel_name}"
        path = os.path.join(self.run_path, f"{output_name}.csv")

        self._write_results_csv(results, path)
        self._write_results_sdata(results)

        # perform post processing cleanup
        if not self.deep_debug:
            self._post_processing_cleanup()
