import itertools

import numpy as np
import pandas as pd
import torch
import wandb

# Check if wandb is installed as optional dependency
try:
    import wandb
except ImportError as err:
    raise ImportError(
        "Wandb is not installed. Please install it via `pip install wandb` if you want to utilize the callback metrics."
    ) from err

# Check if Pytorch Lightning is installed as optional dependency
try:
    from pytorch_lightning import LightningModule
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.trainer import Trainer

except ImportError as err:
    raise ImportError(
        "Pytorch Lightning is not installed. Please install it via `pip install pytorch_lightning` "
        "if you want to utilize the callback metrics."
    ) from err

from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_curve,
)


class BatchAccumulatedMetricsCallback(Callback):
    """
    Callback to calculate metrics on the accumulated predictions of a batch.

    The following metrics are calculated:
    - F1-score
    - AUC precision-recall
    - AUC ROC curve
    - Precision-recall curve
    - ROC curve

    The precision-recall curve and ROC curve are only calculated after a complete epoch to reduce the amount of data that needs to be saved.
    This metric relies on wandb for logging the data.
    """

    def __init__(self, downsampling_factor: int = 4, n_epochs_for_big_calcs: int = 1) -> None:
        """
        Args:
            downsampling_factor: Factor to downsample the roc and precision-recall curve data to reduce the amount of data that needs to be saved. Defaults to 4.
            n_epochs_for_big_calcs: Number of epochs after which to calculate the roc and precision-recall curve data. All other data is calculated at each validation step. Defaults to 1.

        """
        self.iteration = 1
        self.n_epochs_big_calcs = n_epochs_for_big_calcs
        self.downsampling_factor = downsampling_factor

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_actual_labels = torch.tensor([])
        self.train_probabilities = torch.tensor([])

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: dict, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.train_actual_labels = torch.cat((self.train_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.train_probabilities = torch.cat((self.train_probabilities, outputs["probabilities"].detach().cpu()))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # calculate f1-score
        probabilities = self.train_probabilities.numpy()
        probs_1d = probabilities[:, 1]  # only get the predictions for the true-class
        preds_1d = (probs_1d >= 0.5).astype(int)
        labels = self.train_actual_labels.numpy()

        f1 = f1_score(y_true=labels, y_pred=preds_1d)
        self.log("f1_score/train_accumulated", f1, sync_dist=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_actual_labels = torch.tensor([])
        self.val_probabilities = torch.tensor([])

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: dict, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.val_actual_labels = torch.cat((self.val_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.val_probabilities = torch.cat((self.val_probabilities, outputs["probabilities"].detach().cpu()))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # calculate f1-score
        probabilities = self.val_probabilities.numpy()
        probs_1d = probabilities[:, 1]  # only get the predictions for the true-class
        preds_1d = (probs_1d >= 0.5).astype(int)
        labels = self.val_actual_labels.numpy()

        f1 = f1_score(y_true=labels, y_pred=preds_1d)
        self.log("f1_score/val_accumulated", f1, sync_dist=True)

        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, probs_1d, pos_label=1)
        auc_precision_recall = auc(recall, precision)

        self.log("auc_precision_recall/val_accumulated", auc_precision_recall, sync_dist=True)

        # calculate ROC
        fpr, tpr, thresholds = roc_curve(labels, probs_1d, pos_label=1)
        auc_roc = auc(fpr, tpr)

        self.log("auc_roc/val_accumulated", auc_roc, sync_dist=True)

        # we only want to log the roc curve and precision recall curve after a complete epoch (not for each validation step)
        # other metrics should be logged at all val checks though
        if self.iteration % (1 / trainer.val_check_interval) * self.n_epochs_big_calcs == 0:
            # plot roc curve
            data = pd.DataFrame([[x, y] for (x, y) in zip(tpr, fpr, strict=True)])
            data = data.iloc[
                :: self.downsampling_factor, :
            ]  # subsample to every second entry to make amount of data that needs to be saved smaller
            data.columns = ["tpr", "fpr"]
            table = wandb.Table(data=data, columns=["tpr", "fpr"])
            trainer.logger.experiment.log({"roc_curve/val_accumulated": table, "t_epoch": trainer.current_epoch})

            # plot precision-recall curve
            data = [[x, y] for (x, y) in zip(precision, recall, strict=True)]
            data = pd.DataFrame(data)
            data = data.iloc[
                :: self.downsampling_factor, :
            ]  # subsample to every second entry to make amount of data that needs to be saved smaller
            data.columns = ["precision", "recall"]
            table = wandb.Table(data=data, columns=["precision", "recall"])
            trainer.logger.experiment.log(
                {"precision_recall_curve/val_accumulated": table, "t_epoch": trainer.current_epoch}
            )

        self.iteration = self.iteration + 1

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.test_actual_labels = torch.tensor([])
        self.test_probabilities = torch.tensor([])

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: dict, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        self.test_actual_labels = torch.cat((self.test_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.test_probabilities = torch.cat((self.test_probabilities, outputs["probabilities"].detach().cpu()))

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # calculate f1-score
        probabilities = self.test_probabilities.numpy()
        probs_1d = probabilities[:, 1]  # only get the predictions for the true-class
        preds_1d = (probs_1d >= 0.5).astype(int)
        labels = self.test_actual_labels.numpy()

        f1 = f1_score(y_true=labels, y_pred=preds_1d)
        self.log("f1_score/test_accumulated", f1, sync_dist=True)

        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, probs_1d, pos_label=1)
        auc_precision_recall = auc(recall, precision)

        self.log("auc_precision_recall/test_accumulated", auc_precision_recall, sync_dist=True)

        # calculate ROC
        fpr, tpr, thresholds = roc_curve(labels, probs_1d, pos_label=1)
        auc_roc = auc(fpr, tpr)

        self.log("auc_roc/test_accumulated", auc_roc, sync_dist=True)

        # plot roc curve
        data = [[x, y] for (x, y) in zip(tpr, fpr, strict=True)]
        table = wandb.Table(data=data, columns=["tpr", "fpr"])
        trainer.logger.experiment.log({"roc_curve/test_accumulated": table, "t_epoch": trainer.current_epoch})

        # plot precision-recall curve
        data = [[x, y] for (x, y) in zip(precision, recall, strict=True)]
        table = wandb.Table(data=data, columns=["precision", "recall"])
        trainer.logger.experiment.log(
            {"precision_recall_curve/test_accumulated": table, "t_epoch": trainer.current_epoch}
        )


class MulticlassBatchAccumulatedMetricsCallback(Callback):
    """
    Multiclass variant of the callback to calculate metrics on the accumulated predictions of a batch.

    The following metrics are calculated:
    - Precision per class
    - Confusion matrix

    The confusion matrix is only calculated after a complete epoch to reduce the amount of data that needs to be saved.
    """

    def __init__(self, downsampling_factor: int = 4, n_epochs_for_big_calcs: int = 1) -> None:
        """
        Args:
            downsampling_factor (int, optional): Factor to downsample the roc and precision-recall curve data to reduce the amount of data that needs to be saved. Defaults to 4.
            n_epochs_for_big_calcs (int, optional): Number of epochs after which to calculate the roc and precision-recall curve data. All other data is calculated at each validation step. Defaults to 1.
        """
        self.iteration = 1
        self.n_epochs_big_calcs = n_epochs_for_big_calcs
        self.downsampling_factor = downsampling_factor
        pass

    def _calculate_confusion_table(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, labels: torch.Tensor | None = None, normalize: bool = False
    ) -> wandb.Table:
        """Calculate a confusion matrix using the given true and predicted labels.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: Labels to use for the confusion matrix. Defaults to None.
            normalize: Whether to normalize the confusion matrix. Defaults to False.

        Returns:
            Table containing the confusion matrix data.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm = confusion_matrix(y_true, y_pred)

        if labels is None:
            classes = np.unique((y_true, y_pred))
        else:
            classes = np.asarray(labels)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=2)
            cm[np.isnan(cm)] = 0.0

        np.isin(classes, y_true)
        true_classes = classes

        np.isin(classes, y_pred)
        pred_classes = classes

        data = []
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if labels is not None and (isinstance(pred_classes[i], int) or isinstance(pred_classes[0], np.integer)):
                pred_dict = labels[pred_classes[i]]
                true_dict = labels[true_classes[j]]
            else:
                pred_dict = pred_classes[j]
                true_dict = true_classes[i]

            data.append([pred_dict, true_dict, cm[i, j]])

        return wandb.Table(columns=["Predicted_Label", "Actual_Label", "Count"], data=data)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_actual_labels = torch.tensor([])
        self.train_probabilities = torch.tensor([])

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.train_actual_labels = torch.cat((self.train_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.train_probabilities = torch.cat((self.train_probabilities, outputs["probabilities"].detach().cpu()))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        probabilities = self.train_probabilities.numpy()
        predictions = np.argmax(probabilities, axis=1).astype(int)

        labels = self.train_actual_labels.numpy()
        np.sort(np.unique(labels))

        precision_per_class = precision_score(labels, predictions, average=None)
        precision_dict = {f"precision_class_{i}/train": p for i, p in enumerate(precision_per_class)}
        trainer.logger.experiment.log(precision_dict)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_actual_labels = torch.tensor([])
        self.val_probabilities = torch.tensor([])

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.val_actual_labels = torch.cat((self.val_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.val_probabilities = torch.cat((self.val_probabilities, outputs["probabilities"].detach().cpu()))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        probabilities = self.val_probabilities.numpy()
        predictions = np.argmax(probabilities, axis=1).astype(int)

        labels = self.val_actual_labels.numpy()
        label_overview = np.sort(np.unique(labels))

        precision_per_class = precision_score(labels, predictions, average=None)
        precision_dict = {f"precision_class_{i}/val": p for i, p in enumerate(precision_per_class)}
        trainer.logger.experiment.log(precision_dict)

        if self.iteration % (1 / trainer.val_check_interval) * self.n_epochs_big_calcs == 0:
            for label in label_overview:
                filtered = probabilities[labels == label]
                means = np.mean(filtered, axis=0)

                data = [[label, val] for (label, val) in zip(label_overview, means, strict=False)]
                table = wandb.Table(data=data, columns=["label", "mean prediction"])
                trainer.logger.experiment.log(
                    {f"prediction distribution true class {label}/val": table, "t_epoch": trainer.current_epoch}
                )

            table = self._calculate_confusion_table(y_true=labels, y_pred=predictions)
            trainer.logger.experiment.log({"confusion matrix/val": table, "t_epoch": trainer.current_epoch})

        self.iteration = self.iteration + 1

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.test_actual_labels = torch.tensor([])
        self.test_probabilities = torch.tensor([])

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: dict, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        self.test_actual_labels = torch.cat((self.test_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.test_probabilities = torch.cat((self.test_probabilities, outputs["probabilities"].detach().cpu()))

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        probabilities = self.test_probabilities.numpy()
        predictions = np.argmax(probabilities, axis=1).astype(int)

        labels = self.test_actual_labels.numpy()
        label_overview = np.sort(np.unique(labels))

        precision_per_class = precision_score(labels, predictions, average=None)
        precision_dict = {
            f"precision_class_{i}/val": p for i, p in zip(label_overview, precision_per_class, strict=False)
        }
        trainer.logger.experiment.log(precision_dict)

        if self.iteration % (1 / trainer.val_check_interval) * self.n_epochs_big_calcs == 0:
            for label in label_overview:
                filtered = probabilities[labels == label]
                means = np.mean(filtered, axis=0)

                data = [[label, val] for (label, val) in zip(label_overview, means, strict=False)]
                table = wandb.Table(data=data, columns=["label", "mean prediction"])
                trainer.logger.experiment.log(
                    {f"prediction distribution true class {label}/val": table, "t_epoch": trainer.current_epoch}
                )

            table = self._calculate_confusion_table(y_true=labels, y_pred=predictions)
            trainer.logger.experiment.log({"confusion matrix/test": table, "t_epoch": trainer.current_epoch})
