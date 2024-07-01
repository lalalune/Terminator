import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import linear


def normalize(out):
    """Normalizes hidden ouputs of slow net"""
    return F.normalize(out.view(out.shape[0], -1)).view_as(out)


class MAGNetLayer(torch.nn.Module):
    """
    Gabor-like filter as used in GaborNet for hyper kernel.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        omega_0: float,
        alpha: float,
        beta: float,
    ):
        super().__init__()

        # Define type of linear to use
        Linear = getattr(linear, f"Linear{data_dim}d")

        self.gamma = torch.distributions.gamma.Gamma(alpha, beta).sample(
                (hidden_channels, data_dim)
            )
        
        self.linear = Linear(data_dim, hidden_channels, bias=True)
        self.linear.weight.data *= (
            2 * np.pi * omega_0 * self.gamma.view(*self.gamma.shape, *((1,) * data_dim))
        )
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        return torch.sin(self.linear(x))


class MAGNet_G(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        kernel_size: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        omega_0: float = 2000.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        **kwargs,
    ):
        # call super class
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Define type of linear to use
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        # Hidden layers
        self.linears = torch.nn.ModuleList(
            [
                Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer
        self.output_linear = Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        
        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,   
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                )
                for layer in range(num_layers + 1)
            ]
        )
        
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)
        
        self.bias_p = torch.nn.Parameter(
            torch.zeros(1, 1, kernel_size, kernel_size), requires_grad=True
        )
        
    def forward(self, coords, x):
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_p.repeat(1, self.hidden_channels, 1, 1)
        out = self.output_linear(out)
        return out


class MAGNet_GC(torch.nn.Module):
    """
    The slow net to generate hyper-kernel for hyper-channel interaction.
    """
    def __init__(
        self,
        data_dim: int,
        kernel_size: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        omega_0: float = 2000.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        **kwargs,
    ):
        # call super class
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Define type of linear to use
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        # Hidden layers
        self.linears = torch.nn.ModuleList(
            [
                Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer
        self.output_linear = Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        
        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,   
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                )
                for layer in range(num_layers + 1)
            ]
        )
        
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)
        
        # Bias
        self.bias_p = torch.nn.Parameter(
            torch.zeros(1, 1, kernel_size), requires_grad=True
        )
        
    def forward(self, coords, x):
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_p.repeat(1, self.hidden_channels, 1)
        out = self.output_linear(out)
        return out
    

class MAGNet_L(torch.nn.Module):
    """
    The slow net to generate hyper-kernel for local branch.
    Embedding context-dependency in the generation process.
    """
    def __init__(
        self,
        data_dim: int,
        kernel_size: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        omega_0: float = 2000.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        **kwargs,
    ):
        # call super class
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Define type of linear to use
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        # Hidden layers
        self.linears = torch.nn.ModuleList(
            [
                Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer
        self.output_linear = Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        
        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,   
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                )
                for layer in range(num_layers + 1)
            ]
        )
        
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)
        
        # Bias
        self.bias_pk = torch.nn.Parameter(
            torch.zeros(1, 1, kernel_size, kernel_size), requires_grad=True
        )
        
        # Transform input to add context-dependency
        NonlinearType = getattr(torch.nn, 'GELU')
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        
        # Define type of linear to use
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        self.fast_reduce = torch.nn.Conv2d(self.out_channels, self.hidden_channels, 1, 1, 0, bias=True)
        self.fast_fc1 = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=True)
        self.fast_gelu1 = NonlinearType()
        self.fast_linears_x = torch.nn.ModuleList(
            [
                Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        # Initialize
        for idx, lin in enumerate(self.fast_linears_x):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
                
        self.alphas = [nn.Parameter(torch.Tensor(hidden_channels).fill_(1)) for _ in range(num_layers)]
        self.betas = [nn.Parameter(torch.Tensor(hidden_channels).fill_(0.1)) for _ in range(num_layers)]
        
    def s_renormalization(self, out, alpha, beta):
        out = out.transpose(0, 1)
        
        delta = out.data.clone()
        assert delta.shape == out.shape

        v = (-1,) + (1,) * (out.dim() - 1)
        out_t = alpha.view(*v) * delta + beta.view(*v) * normalize(out)
        
        return out_t.transpose(0, 1)
    
    def forward(self, coords, x):
        x_gap = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (self.kernel_size,) * self.data_dim,
        )
        x_gap = self.fast_reduce(x_gap)
        x_gap = x_gap.mean(axis=0, keepdims=True)
        x_gap = self.fast_fc1(x_gap)
        x_gap = self.fast_gelu1(x_gap)
        
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_pk.repeat(1, self.hidden_channels, 1, 1)
            out = self.s_renormalization(out, self.alphas[i-1].cuda(), self.betas[i-1].cuda())
            out = out * self.fast_linears_x[i - 1](x_gap)

        out = self.output_linear(out)
        
        return out
