import math

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.modules import magnet


def linspace_grid(grid_sizes):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    tensors = []
    for size in grid_sizes:
        tensors.append(torch.linspace(-1, 1, steps=size))
    grid = torch.stack(torch.meshgrid(*tensors), dim=0)
    return grid


class SlowNetBase(torch.nn.Module):
    """
    The base of slow net for the following SlowNet_G, SlowNet_GC, SlowNet_L, 
    """
    def __init__(
        self,
        in_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        **kwargs,
    ):
        super().__init__(
        )
        
        self.data_dim = data_dim
        self.kernel_size = kernel_cfg.size
        
        kernel_type = kernel_cfg.type
        hidden_channels = kernel_cfg.no_hidden
        num_layers = kernel_cfg.num_layers
        
        # Variable placeholders
        self.register_buffer("train_length", torch.zeros(1).int(), persistent=True)
        self.register_buffer("kernel_coords", torch.zeros(1), persistent=False)
        self.register_buffer("initialized", torch.zeros(1).bool(), persistent=False)
        
        # Create the kernel
        KernelClass = getattr(magnet, kernel_type)
        self.Kernel = KernelClass(
            data_dim=data_dim,
            kernel_size=self.kernel_size,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_layers=num_layers,
        )
    
    def construct_kernel(self, x):
        # kernel coordinations
        self.kernel_coords = self.get_kernel_coords(x)
        
        # initialization
        self.chang_initialization(self.kernel_coords)
        
        x_shape = x.shape
        hyper_kernel = self.Kernel(self.kernel_coords, x).view(
            -1, x_shape[1], *self.kernel_coords.shape[2:]
        )
        
        return hyper_kernel
    
    def get_kernel_coords(self, x):
        self.train_length[0] = int(self.kernel_size)
        
        kernel_coords = linspace_grid(
            grid_sizes=self.train_length.repeat(self.data_dim)
        )
        kernel_coords = kernel_coords.unsqueeze(0).type_as(self.kernel_coords)

        return kernel_coords
    
    def chang_initialization(self, kernel_coords):
        if not self.initialized[0]:
            # Initialize the last layer of self.Kernel as in Chang et al. (2020)
            with torch.no_grad():
                kernel_size = kernel_coords.reshape(
                    *kernel_coords.shape[: -self.data_dim], -1
                )
                normalization_factor = kernel_size.shape[-1]
                    
                self.Kernel.output_linear.weight.data *= math.sqrt(
                    1.0 / normalization_factor
                )
            # Set the initialization flag to true
            self.initialized[0] = True

    def forward(self, x):
        raise NotImplementedError


class SlowNet_G(SlowNetBase):
    """
    The slow net to generate hyper-kernel for global branch & hyper-spatial interaction
    """
    def forward(self, x):
        global_hk = self.construct_kernel(x)
        
        return global_hk
        
        
class SlowNet_GC(SlowNetBase):
    """
    The slow net to generate hyper-kernel for hyper-channel interaction
    """
    def forward(self, x):
        global_hk_c = self.construct_kernel(x)
        
        return global_hk_c
    

class SlowNet_L(SlowNetBase):
    """
    The slow net to generate context-dependent hyper-kernel for local branch
    """
    def __init__(
        self,
        in_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        kernel_size: int,
    ):
        super().__init__(
            in_channels=in_channels,
            data_dim=data_dim,
            kernel_cfg=kernel_cfg,
        )
        
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        
        kernel_type = kernel_cfg.type
        hidden_channels = kernel_cfg.no_hidden
        num_layers = kernel_cfg.num_layers
        
        # Variable placeholders
        self.register_buffer("train_length", torch.zeros(1).int(), persistent=True)
        self.register_buffer("kernel_coords", torch.zeros(1), persistent=False)
        self.register_buffer("initialized", torch.zeros(1).bool(), persistent=False)
        
        # Create the kernel
        KernelClass = getattr(magnet, kernel_type)
        self.Kernel = KernelClass(
            data_dim=data_dim,
            kernel_size=self.kernel_size,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_layers=num_layers,
        )
    
    def forward(self, x):
        local_hk = self.construct_kernel(x)
        
        local_hk = local_hk.view(local_hk.shape[1], 1, *local_hk.shape[2:])
        
        return local_hk
