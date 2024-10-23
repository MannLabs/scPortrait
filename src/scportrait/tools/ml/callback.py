import pandas as pd
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve


class BatchAccumulatedMetricsCallback(Callback):
    """
    Callback to calculate metrics on the accumulated predictions of a batch.

    The following metrics are calculated:
    - F1-score
    - AUC precision-recall
    - AUC ROC curve
    - Precision-recall curve
    - ROC curve

    The Precision-recall curve and ROC curve are only calculated after a complete epoch to reduce the amount of data that needs to be saved.

    Parameters
    ----------
    downsampling_factor : int, optional, default = 4
        Factor to downsample the roc and precision-recall curve data to reduce the amount of data that needs to be saved.
    n_epochs_for_big_calcs : int, optional, default = 1
        Number of epochs after which to calculate the roc and precision-recall curve data. All other data is calculated at each validation step.
    """

    def __init__(self, downsampling_factor=4, n_epochs_for_big_calcs=1):
        self.iteration = 1
        self.n_epochs_big_calcs = n_epochs_for_big_calcs
        self.downsampling_factor = downsampling_factor

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_actual_labels = torch.tensor([])
        self.train_probabilities = torch.tensor([])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.train_actual_labels = torch.cat((self.train_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.train_probabilities = torch.cat((self.train_probabilities, outputs["probabilities"].detach().cpu()))

    def on_train_epoch_end(self, trainer, pl_module):
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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_actual_labels = torch.cat((self.val_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.val_probabilities = torch.cat((self.val_probabilities, outputs["probabilities"].detach().cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
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
            data = [[x, y] for (x, y) in zip(tpr, fpr, strict=True)]
            data = pd.DataFrame(data)
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

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_actual_labels = torch.tensor([])
        self.test_probabilities = torch.tensor([])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_actual_labels = torch.cat((self.test_actual_labels, outputs["actual_labels"].detach().cpu()))
        self.test_probabilities = torch.cat((self.test_probabilities, outputs["probabilities"].detach().cpu()))

    def on_test_epoch_end(self, trainer, pl_module):
        # calculate f1-score
        probabilities = self.test_probabilities.numpy()
        probs_1d = probabilities[:, 1]  # only get the predictionssfor the true-class
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
