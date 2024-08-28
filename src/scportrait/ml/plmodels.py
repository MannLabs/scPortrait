import gc
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from scportrait.ml.models import _VGG1, _VGG2, VGG1, VGG2, CAEBase, VGG2_regression


class MultilabelSupervisedModel(pl.LightningModule):
    """
    A pytorch lightning network module to use a multi-label supervised Model. 

    Parameters
    ----------
    type : str, optional, default = "VGG2"
        Network architecture to used in model. Architectures are defined in scPortrait.ml.models
        Valid options: "VGG1", "VGG2", "VGG1_old", "VGG2_old".
    kwargs : dict
        Additional parameters passed to the model.

    Attributes
    ----------
    network : torch.nn.Module
        The selected network architecture.
    train_metrics : torchmetrics.MetricCollection
        MetricCollection for evaluating model on training data.
    val_metrics : torchmetrics.MetricCollection
        MetricCollection for evaluating model on validation data.
    test_metrics : torchmetrics.MetricCollection
        MetricCollection for evaluating model on test data.
    
    Methods
    -------
    forward(x)
        perform forward pass of model.
    configure_optimizers()
        Optimization function
    on_train_epoch_end()
        Callback function after each training epoch
    on_validation_epoch_end()
        Callback function after each validation epoch
    confusion_plot(matrix)
        Generate confusion matrix plot
    training_step(batch, batch_idx)
        Perform a single training step
    validation_step(batch, batch_idx)
        Perform a single validation step
    test_step(batch, batch_idx)
        Perform a single test step
    test_epoch_end(outputs)
        Callback function after testing epochs
    """
    def __init__(self, model_type="VGG2", **kwargs):
        super().__init__()

        self.save_hyperparameters()

        #initialize metrics
        if self.hparams["num_classes"] == 2:
            task_type = "binary"
        elif self.hparams["num_classes"] > 2:
            task_type = "multiclass"
        else:
            raise ValueError("No num_classes specified in hparams")
        
        #initialize metrics to track
        self.accuracy = torchmetrics.Accuracy(task=task_type, num_classes=self.hparams["num_classes"])
        self.aucroc = torchmetrics.AUROC(task=task_type, thresholds=None, num_classes=self.hparams["num_classes"])
        
        if model_type == "VGG1":
            self.network = VGG1(in_channels=self.hparams["num_in_channels"],
                                cfg="B",
                                dimensions=128,
                                num_classes=self.hparams["num_classes"],
                                image_size_factor=self.hparams["image_size_factor"])
            
        elif model_type == "VGG2":
            self.network = VGG2(in_channels=self.hparams["num_in_channels"],
                                cfg="B",
                                dimensions=128,
                                num_classes=self.hparams["num_classes"], 
                                image_size_factor=self.hparams["image_size_factor"])
        
        ## add deprecated type for backward compatibility
        elif model_type == "VGG1_old":
            self.network = _VGG1(in_channels=self.hparams["num_in_channels"],
                                 cfg="B",
                                 dimensions=128,
                                 num_classes=self.hparams["num_classes"])
        
        ## add deprecated type for backward compatibility
        elif model_type == "VGG2_old":
            self.network = _VGG2(in_channels=self.hparams["num_in_channels"],
                                 cfg="B",
                                 dimensions=128,
                                 num_classes=self.hparams["num_classes"])
        else:
            sys.exit("Incorrect network architecture specified. Please check that MultilabelSupervisedModel type parameter is set to key present in method.")
        
        
    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        if self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["learning_rate"])

        elif self.hparams["optimizer"] == "Adam":
            #set weight decay to 0 if not specified in hparams 
            if self.hparams["weight_decay"] is None:
                self.hparams["weight_decay"] = 0
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])

        elif self.hparams["optimizer"] == "AdamW":
            #set weight decay to 0.01 if not specified in hparams 
            if self.hparams["weight_decay"] is None:
                self.hparams["weight_decay"] = 10 ** -2
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])

        else:
            raise ValueError("No optimizer specified in hparams")
        return optimizer
           
    def training_step(self, batch, batch_idx):
        data, label = batch

        #calculate loss
        output_softmax = self.network(data)
        loss = F.nll_loss(output_softmax, label)

        #calculate accuracy
        probabilities = torch.exp(output_softmax)
        pred_labels = torch.argmax(probabilities, dim=1)
        acc = self.accuracy(pred_labels, label)

        self.log('loss/train', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('acc/train', acc, prog_bar = True, on_epoch=True, sync_dist=True)
        
        return {'loss':loss, 'probabilities':probabilities, 'actual_labels':label}
    
    def validation_step(self, batch, batch_idx):
        data, label = batch

        output_softmax = self.network(data)
        loss = F.nll_loss(output_softmax, label)

        #calculate accuracy
        probabilities = torch.exp(output_softmax) #we use the logsoftmax so we need to take the exp to get the actual probabilities
        pred_labels = torch.argmax(probabilities, dim=1) #then we can select the predicted label taking a 0.5 threshold for binary classification, for multiclass problems it simply selects the most likely class
        acc = self.accuracy(pred_labels, label)

        self.log('loss/val', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('acc/val', acc, prog_bar = True, on_epoch=True, sync_dist=True)

        return {'loss':loss, 'probabilities':probabilities, 'actual_labels':label}

    def test_step(self, batch, batch_idx):
        data, label = batch

        output_softmax = self.network(data)
        loss = F.nll_loss(output_softmax, label)

        #calculate accuracy
        probabilities = torch.exp(output_softmax)
        pred_labels = torch.argmax(probabilities, dim=1)
        acc = self.accuracy(pred_labels, label)

        self.log('loss/test', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('acc/test', acc, prog_bar=True, on_epoch=True, sync_dist=True)

        return {'loss':loss, 'probabilities':probabilities, 'actual_labels':label}


class RegressionModel(pl.LightningModule):

    def __init__(self, model_type="VGG2_regression", **kwargs):
        super().__init__()
        self.save_hyperparameters()
    
        # Define the regression model
        if model_type == "VGG2_regression":
            self.network =  VGG2_regression(cfg="B",
                                            cfg_MLP="A", 
                                            in_channels=self.hparams["num_in_channels"],
                                            image_size_factor=self.hparams["image_size_factor"],
                                            )

        # Initialize metrics for regression model 
        self.mse = torchmetrics.MeanSquaredError() # MSE metric for regression
        self.mae = torchmetrics.MeanAbsoluteError() # MAE metric for regression

    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        if self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["learning_rate"])

        elif self.hparams["optimizer"] == "Adam":
            if self.hparams["weight_decay"] is None:
                self.hparams["weight_decay"] = 0
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])

        elif self.hparams["optimizer"] == "AdamW":
            if self.hparams["weight_decay"] is None:
                self.hparams["weight_decay"] = 10 ** -2
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])

        else:
            raise ValueError("No optimizer specified in hparams.")
        
        return optimizer
    
    def configure_loss(self):
        if self.hparams["loss"] == "mse":
            loss = F.mse_loss
        elif self.hparams["loss"] == "huber":
            if self.hparams["huber_delta"] is None:
                self.hparams["huber_delta"] = 1.0
            loss = F.huber_loss
        else:
            raise ValueError("No loss function specified in hparams.")
        
        return loss
    
    def training_step(self, batch):
        data, target = batch
        target = target.unsqueeze(1)
        output = self.network(data) # Forward pass, only one output

        loss_func = self.configure_loss()

        if self.hparams["loss"] == "huber":
            loss = loss_func(output, target, delta=self.hparams["huber_delta"], reduction='mean')
        else:
            loss = loss_func(output, target)

        self.log('loss/train', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mse/train', self.mse(output, target), on_epoch=True, prog_bar=True)
        self.log('mae/train', self.mae(output, target), on_epoch=True, prog_bar=True)

        return {'loss': loss, 'predictions': output, 'targets': target}
    
    def validation_step(self, batch):
        data, target = batch
        target = target.unsqueeze(1)
        output = self.network(data)

        loss_func = self.configure_loss()
        
        if self.hparams["loss"] == "huber":
            loss = loss_func(output, target, delta=self.hparams["huber_delta"], reduction='mean')
        else:
            loss = loss_func(output, target)

        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mse/val', self.mse(output, target), on_epoch=True, prog_bar=True)
        self.log('mae/val', self.mae(output, target), on_epoch=True, prog_bar=True)

        return {'loss': loss, 'predictions': output, 'targets': target}
    
    def test_step(self, batch):
        data, target = batch
        target = target.unsqueeze(1)
        output = self.network(data)

        loss_func = self.configure_loss()
        
        if self.hparams["loss"] == "huber":
            loss = loss_func(output, target, delta=self.hparams["huber_delta"], reduction='mean')
        else:
            loss = loss_func(output, target)

        self.log('loss/test', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mse/test', self.mse(output, target), on_epoch=True, prog_bar=True)
        self.log('mae/test', self.mae(output, target), on_epoch=True, prog_bar=True)

        return {'loss': loss, 'predictions': output, 'targets': target}

# implemented models for future use currently not applied to scPortrait

class GeneralModel(pl.LightningModule):

    def __init__(self, model, hparams):
        super().__init__()
            
        print(hparams)
        self.hp = hparams
        self.network = model
        
        self.train_metrics = torchmetrics.MetricCollection([torchmetrics.Precision("binary", average="none",num_classes=hparams["num_classes"]), 
                                                            torchmetrics.Recall("binary", average="none",num_classes=hparams["num_classes"]),
                                                            torchmetrics.AUROC(num_classes=hparams["num_classes"]),
                                                            torchmetrics.Accuracy("binary")]) 
        
        self.val_metrics = torchmetrics.MetricCollection([torchmetrics.Precision("binary", average="none",num_classes=hparams["num_classes"]), 
                                                          torchmetrics.Recall("binary", average="none",num_classes=hparams["num_classes"]),
                                                          torchmetrics.AUROC(num_classes=hparams["num_classes"]),
                                                          torchmetrics.Accuracy("binary")])
        
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hp, {"precision/train": 0,"recall/train": 0,"precision/val": 0,"recall/val": 0})
    
    def forward(self, x): 
        return self.network(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hp["lr"])
        return optimizer
    
    def on_train_epoch_start(self):
        pass
        
    def on_validation_epoch_start(self):
         pass
        
    def on_train_epoch_end(self,outputs):
        metrics = self.train_metrics.compute()
        
        self.log("precision/train", metrics["Precision"][0])
        self.log("recall/train", metrics["Recall"][0])
        self.log("auroc/train", metrics["AUROC"])
        self.log("accuracy/train", metrics["Accuracy"])
        
        # Reseting internal state such that metric ready for new data
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        
        metrics = self.val_metrics.compute()
        
        self.log("precision/val", metrics["Precision"][0])
        self.log("recall/val", metrics["Recall"][0])
        self.log("auroc/val", metrics["AUROC"])
        self.log("accuracy/val", metrics["Accuracy"])
        
        # Reseting internal state such that metric ready for new data
        self.val_metrics.reset()
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        
        data, label = data.cuda(), label.cuda()
        
        output = self.network(data)
        loss = F.nll_loss(output, label)
        
        # log accuracy metrics
        non_log = torch.exp(output)
        
        self.train_metrics(non_log, label)
        
        
        self.log('loss/train', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.cuda(), label.cuda()
        
        output = self.network(data)
        loss = F.nll_loss(output, label)
    

        #accuracy metrics
        non_log = torch.exp(output)    
        
        self.val_metrics(non_log, label)
        
        self.log('loss/val', loss, prog_bar=True)

class AutoEncoderModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
            
        print(self.hparams)
        self.save_hyperparameters()

        self.network = CAEBase(encoder_cfg = self.hparams["encoder_cfg"],
                decoder_cfg = self.hparams["decoder_cfg"],
                in_channels = self.hparams["num_in_channels"],
                out_channels = self.hparams["num_out_channels"])
        
        self.loss = torch.nn.BCELoss()
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        
        return optimizer
    
    def on_train_epoch_start(self):
        pass
        
    def on_validation_epoch_start(self):
         pass
        
    def on_train_epoch_end(self,outputs): 
        pass
        #tensorboard = self.logger.experiment
        #tensorboard.add_image()
        
    def training_step(self, batch, batch_idx):     
        
        opt = self.optimizers()
        opt.zero_grad()
    
        images, _ = batch

        images = images.cuda()
        input_golgi_channel = images[:,3:4,:,:]
        
        output = self.network(images)
        
        
        loss = self.loss(output, input_golgi_channel)    
        
        self.manual_backward(loss, retain_graph=False)
        
        opt.step()
        loss = loss.detach()
        
        self.log('loss/train', loss, prog_bar=True)
        
        if batch_idx > 118:
            torch.cuda.empty_cache()
            
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except Exception:
                    pass
        

        
    def validation_step(self, batch, batch_idx):     
        
        opt = self.optimizers()
        opt.zero_grad()
    
        images, labels = batch

        images = images.cuda()
        input_golgi_channel = images[:,3:4,:,:]
        
        output = self.network(images)
        
        
        loss = self.loss(output, input_golgi_channel)    

        loss = loss.detach()
        
        if batch_idx == 0:
            SAMPLE_NUM=10
            
            input_sample = images[:SAMPLE_NUM,:,:,:].detach().cpu()
            output_sample = output[:SAMPLE_NUM,:,:,:].detach().cpu()
            label_sample = labels[:SAMPLE_NUM].detach().cpu()
            
            img = self.sample_plot(input_sample,output_sample,label_sample)
            self.logger.experiment.add_image('prediction', img, self.current_epoch, dataformats="NHWC") 
        
        print(batch_idx, self.current_epoch)
        
        self.log('loss/val', loss.item(), prog_bar=True)
        return loss
    
    def sample_plot(self, input_sample, output_sample, label_sample):
        
        fig, axs = plt.subplots(6, len(label_sample), figsize=(15, 10))
        
        for i in range(len(label_sample)):
    
            label = int(label_sample[i].item())

           
            channels = input_sample[i]

            axs[0,i].set_title(self.hparams["class_labels"][label])

            for j in range(6):

                if j <2:
                    color = "gray"
                else:
                    color = "magma"

                if j < 5:
                    c = channels[j]

                else:
                    c = output_sample[i,0]

                axs[j,i].imshow(c, cmap=color)
                axs[j,i].xaxis.set_ticks([])
                axs[j,i].yaxis.set_ticks([])

        rows = ["Cell Mask", "Nucleus Mask", "Nucleus", "TGOLN2-mCherry","WGA","CAE"]
        
        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, size='large', ha="right")

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
        fig.tight_layout()
        
        fig.canvas.draw()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.expand_dims(data, axis=0)
        
        return data

class AEModel(pl.LightningModule):

    def __init__(self, model, hparams):
        super().__init__()
            
        print(hparams)
        self.hp = hparams
        self.network = model
        self.loss = torch.nn.BCELoss()
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        optimizer = optimizer = torch.optim.Adam(self.parameters(), lr=self.hp["lr"])
        
        return optimizer
    
    def on_train_epoch_start(self):
        pass
        
    def on_validation_epoch_start(self):
         pass
        
    def on_train_epoch_end(self,outputs):
        
        pass
        #tensorboard = self.logger.experiment
        #tensorboard.add_image()
        
    def training_step(self, batch, batch_idx):     
        
        opt = self.optimizers()
        opt.zero_grad()
    
        images, _ = batch

        images = images.cuda()
        input_golgi_channel = images[:,3:4,:,:]
        
        output = self.network(images)
        
        
        loss = self.loss(output, input_golgi_channel)    
        self.manual_backward(loss)
        
        opt.step()
        loss = loss.detach()
        
        self.log('loss/train', loss, prog_bar=True)
        
