import os

import torch
import pytorch_lightning as pl

from datamodules.cifar10 import CIFAR10DataModule, CIFAR100DataModule

from omegaconf import OmegaConf


def construct_datamodule(
    cfg: OmegaConf,
) -> pl.LightningDataModule:

    # Define num_workers
    if cfg.no_workers == -1:
        cfg.no_workers = int(os.cpu_count() / 4)

    # Define pin_memory
    if torch.cuda.is_available() and cfg.device == "cuda":
        pin_memory = True
    else:
        pin_memory = False

    # Gather module from datamodules, create instance and return
    if cfg.dataset.name == "CIFAR10":
        dataset_name = f"{cfg.dataset.name}DataModule"
        dataset = CIFAR10DataModule
    elif cfg.dataset.name == "CIFAR100":
        dataset_name = f"{cfg.dataset.name}DataModule"
        dataset = CIFAR100DataModule
    else:
        dataset_name = f"{cfg.dataset.name}DataModule"
        dataset = CIFAR100DataModule
    
    datamodule = dataset(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size // cfg.train.accumulate_grad_steps,
        test_batch_size=cfg.test.batch_size_multiplier * cfg.train.batch_size,
        data_type=cfg.dataset.data_type,
        num_workers=cfg.no_workers,
        pin_memory=pin_memory,
        augment=cfg.dataset.augment,
    )
    
    # Assert if the datamodule has the parameters needed for the model creation
    assert hasattr(datamodule, "data_dim")
    assert hasattr(datamodule, "input_channels")
    assert hasattr(datamodule, "output_channels")
    assert hasattr(datamodule, "data_type")
    return datamodule
