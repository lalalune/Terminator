import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from hydra import utils


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        data_type,
        num_workers,
        pin_memory,
        **kwargs,
    ):
        super().__init__()

        # Save parameters to self
        self.data_dir = utils.get_original_cwd() + data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Determine data_type
        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        elif data_type == "sequence":
            self.data_type = data_type
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Determine sizes of dataset
        self.input_channels = 3
        self.output_channels = 10
        
        # Transform
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                (0.2470, 0.2435, 0.2616),),
        ])
        
        val_test_transform = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2470, 0.2435, 0.2616),
            ),
        ]
        self.val_test_transform = transforms.Compose(val_test_transform)
        
    def setup(self, stage=None):
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=self.train_transform,
                download=True,
            )
  
            self.val_dataset = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=self.val_test_transform,
                download=True,
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=self.val_test_transform,
                download=True,
            )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_dataloader
