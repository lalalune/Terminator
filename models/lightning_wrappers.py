import os
import glob
import hydra
import wandb
import numpy as np

import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from utils import no_param
from models.modules.loss import LnLoss, LabelSmoothingCrossEntropy
from models.optim import construct_optimizer, construct_scheduler, ChainedScheduler

from omegaconf import OmegaConf


class LightningWrapperBase(pl.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
    ):
        super().__init__()
        self.cfg = cfg
        
        # Define network
        self.network = network
        # Save optimizer & scheduler parameters
        self.optim_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        # Regularization metrics
        if self.optim_cfg.l2_reg != 0.0:
            self.weight_regularizer = LnLoss(
                weight_loss=self.optim_cfg.l2_reg,
                norm_type=2,
            )
        else:
            self.weight_regularizer = None
        # Placeholders for logging of best train & validation values
        self.no_params = -1
        # Explicitly define whether we are in distributed mode.
        self.distributed = cfg.train.distributed and cfg.train.avail_gpus != 1
        
    def forward(self, x, train_mode=False):
        return self.network(x, train_mode=train_mode)

    def configure_optimizers(self):
        # Construct optimizer & scheduler
        self.optimizer = construct_optimizer(
            model=self,
            optim_cfg=self.optim_cfg,
        )
        self.scheduler, self.warmup_iterations = construct_scheduler(
            optimizer=self.optimizer,
            scheduler_cfg=self.scheduler_cfg,
        )
        
        # Construct output dictionary
        output_dict = {"optimizer": self.optimizer}
        if self.scheduler is not None:
            output_dict["lr_scheduler"] = {}
            output_dict["lr_scheduler"]["scheduler"] = self.scheduler
            output_dict["lr_scheduler"]["interval"] = "step"

            # If we use a ReduceLROnPlateu scheduler, we must monitor val/acc
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.scheduler_cfg.mode == "min":
                    output_dict["lr_scheduler"]["monitor"] = "val/loss"
                else:
                    output_dict["lr_scheduler"]["monitor"] = "val/acc"
                    # output_dict["lr_scheduler"]["monitor"] = 'train/loss_epoch'
                output_dict["lr_scheduler"]["reduce_on_plateau"] = True
                output_dict["lr_scheduler"]["interval"] = "epoch"

            # TODO: ReduceLROnPlateau with warmup
            if isinstance(
                self.scheduler, ChainedScheduler
            ) and isinstance(
                self.scheduler._schedulers[-1], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                raise NotImplementedError("cannot use ReduceLROnPlateau with warmup")
        # Return
        return output_dict

    def on_train_start(self):
        if self.global_rank == 0:
            # Calculate and log the size of the model
            if self.no_params == -1:
                with torch.no_grad():
                    # Log parameters
                    no_params = no_param(self.network)
                    self.logger.experiment.summary["no_params"] = no_params
                    self.no_params = no_params

                    # Log code
                    code = wandb.Artifact(
                        f"source-code-{self.logger.experiment.name}", type="code"
                    )
                    # Get paths
                    paths = glob.glob(
                        hydra.utils.get_original_cwd() + "/**/*.py",
                        recursive=True,
                    )
                    paths += glob.glob(
                        hydra.utils.get_original_cwd() + "/**/*.yaml",
                        recursive=True,
                    )
                    # Filter paths
                    paths = list(filter(lambda x: "outputs" not in x, paths))
                    paths = list(filter(lambda x: "wandb" not in x, paths))
                    # Get all source files
                    for path in paths:
                        code.add_file(
                            path,
                            name=path.replace(f"{hydra.utils.get_original_cwd()}/", ""),
                        )
                    # Use the artifact
                    if not self.logger.experiment.offline:
                        wandb.run.use_artifact(code)


class ClassificationWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
        **kwargs,
    ):
        super().__init__(
            network=network,
            cfg=cfg,
        )
        # Other metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        
        # Binary problem?
        self.multiclass = True
        
        # Loss metric
        if self.multiclass:
            self.loss_metric = LabelSmoothingCrossEntropy()
        else:
            self.loss_metric = torch.nn.BCEWithLogitsLoss()
        
        # Function to get predictions:
        if self.multiclass:
            self.get_predictions = self.multiclass_prediction
        else:
            self.get_predictions = self.binary_prediction
            
        # Placeholders for logging of best train & validation values
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0
    
    def _step(self, batch, accuracy_calculator, train_mode=False):
        
        x, labels = batch
        logits, slow_neural_loss = self(x, train_mode=train_mode)
        
        # Predictions
        predictions = self.get_predictions(logits)
        
        # Calculate accuracy and loss
        accuracy_calculator(predictions, labels)
        
        # For binary classification, the labels must be float
        if not self.multiclass:
            labels = labels.float()
            logits = logits.view(-1)
        
        loss = self.loss_metric(logits, labels) + slow_neural_loss
        
        # Return predictions and loss
        return predictions, logits, loss, slow_neural_loss
        
    def training_step(self, batch, batch_idx):
        
        # Perform step
        predictions, logits, loss, slow_neural_loss = self._step(batch, self.train_acc, train_mode=True)
        
        # Add regularization
        if self.weight_regularizer is not None:
            reg_loss = self.weight_regularizer(self.network)
        else:
            reg_loss = 0.0
        
        # Log and return loss (Required in training step)
        self.log(
            "train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "train/slow_loss",
            slow_neural_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "train/reg_loss",
            reg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        
        loss = loss + reg_loss
        
        self.network.slow_neural_loss = 0
        
        return {"loss": loss, "logits": logits.detach()}

    # @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        
        # Perform step
        redictions, logits, loss, slow_neural_loss = self._step(batch, self.val_acc, train_mode=False)
        
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        
        self.network.slow_neural_loss = 0
        
        return logits.detach()
    
    def test_step(self, batch, batch_idx):
        
        # Perform step
        predictions, _, loss, slow_neural_loss = self._step(batch, self.test_acc, train_mode=False)
        
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )

    def training_epoch_end(self, train_step_outputs):
        flattened_logits = torch.flatten(
            torch.cat([step_output["logits"] for step_output in train_step_outputs])
        )
        self.logger.experiment.log(
            {
                "global_step": self.global_step,
            }
        )
        # Log best accuracy
        train_acc = self.trainer.callback_metrics["train/acc_epoch"]
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc.item()
            self.logger.experiment.log(
                {
                    "train/best_acc": self.best_train_acc,
                    "global_step": self.global_step,
                }
            )

    def validation_epoch_end(self, validation_step_outputs):
        # Gather logits from validation set and construct a histogram of them.
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {
                "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                "global_step": self.global_step,
            }
        )
        # Log best accuracy
        val_acc = self.trainer.callback_metrics["val/acc"]
        
        log_dir = self.trainer.logger.experiment.dir
        if log_dir is not None:
            file_name = self.cfg.net.type + "_" + "val_acc.txt"
            file_path = os.path.join(log_dir, file_name)
            with open(file_path, "a") as file:
                file.write(str(val_acc.cpu().numpy()) + "\n")
                
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            self.logger.experiment.log(
                {
                    "val/best_acc": self.best_val_acc,
                    "global_step": self.global_step,
                }
            )

    @staticmethod
    def multiclass_prediction(logits):
        return torch.argmax(logits, 1)

    @staticmethod
    def binary_prediction(logits):
        return (logits > 0.0).squeeze().long()
