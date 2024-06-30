import math
import torch
import torch.nn as nn
from models.modules import linear

def normalize(out):
    """
    Normalizes the hidden outputs of the slow network.
    
    Args:
        out (torch.Tensor): Hidden output tensor.
        
    Returns:
        torch.Tensor: Normalized hidden output tensor.
    """
    return nn.functional.normalize(out.view(out.shape[0], -1)).view_as(out)


class MAGNetLayer(torch.nn.Module):
    """
    MAGNet layer for generating hyper-kernels.
    
    This layer is a building block of the MAGNet (Multiplicative Anisotropic Gabor Network) slow network.
    It generates Gabor-like filters using sinusoidal activations and linear transformations.
    
    Args:
        data_dim (int): Dimensionality of the input data.
        hidden_channels (int): Number of hidden channels.
        omega_0 (float): Initial frequency of the sinusoidal activations.
        alpha (float): Parameter for the gamma distribution.
        beta (float): Parameter for the gamma distribution.
    """
    def __init__(self, data_dim: int, hidden_channels: int, omega_0: float, alpha: float, beta: float):
        super().__init__()
        
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        self.gamma = torch.distributions.gamma.Gamma(alpha, beta).sample((hidden_channels, data_dim))
        
        self.linear = Linear(data_dim, hidden_channels, bias=True)
        self.linear.weight.data *= (2 * math.pi * omega_0 * self.gamma.view(*self.gamma.shape, *((1,) * data_dim)))
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Forward pass of the MAGNet layer.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying the Gabor-like filters.
        """
        return torch.sin(self.linear(x))


class MAGNetBase(torch.nn.Module):
    """
    Base class for MAGNet slow networks.
    
    This class serves as a base for different variations of MAGNet slow networks, such as MAGNet_G, MAGNet_GC, and MAGNet_L.
    It defines the common structure and initialization of the slow network.
    
    Args:
        data_dim (int): Dimensionality of the input data.
        kernel_size (int): Size of the generated hyper-kernels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of layers in the slow network.
        omega_0 (float): Initial frequency of the sinusoidal activations.
        alpha (float): Parameter for the gamma distribution.
        beta (float): Parameter for the gamma distribution.
    """
    def __init__(self, data_dim: int, kernel_size: int, hidden_channels: int, out_channels: int, num_layers: int,
                 omega_0: float, alpha: float = 6.0, beta: float = 1.0, **kwargs):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        self.linears = torch.nn.ModuleList([
            Linear(in_channels=hidden_channels, out_channels=hidden_channels, bias=True)
            for _ in range(num_layers)
        ])
        
        self.output_linear = Linear(in_channels=hidden_channels, out_channels=out_channels, bias=True)
        
        self.filters = torch.nn.ModuleList([
            MAGNetLayer(data_dim=data_dim, hidden_channels=hidden_channels, omega_0=omega_0,
                        alpha=alpha / (layer + 1), beta=beta)
            for layer in range(num_layers + 1)
        ])
        
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)
        
    def forward(self, coords, x):
        """
        Forward pass of the MAGNet slow network.
        
        This method should be overridden by the specific MAGNet slow network implementations.
        
        Args:
            coords (torch.Tensor): Coordinate tensor for generating hyper-kernels.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Generated hyper-kernels.
        """
        raise NotImplementedError
    

class MAGNet_G(MAGNetBase):
    """
    MAGNet slow network for generating global hyper-kernels.
    
    This class extends the MAGNetBase and implements the specific forward pass for generating global hyper-kernels.
    
    Args:
        data_dim (int): Dimensionality of the input data.
        kernel_size (int): Size of the generated hyper-kernels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of layers in the slow network.
        omega_0 (float): Initial frequency of the sinusoidal activations.
        alpha (float): Parameter for the gamma distribution.
        beta (float): Parameter for the gamma distribution.
    """
    def __init__(self, data_dim: int, kernel_size: int, hidden_channels: int, out_channels: int, num_layers: int,
                 omega_0: float, alpha: float = 6.0, beta: float = 1.0, **kwargs):
        super().__init__(data_dim=data_dim, kernel_size=kernel_size, hidden_channels=hidden_channels,
                         out_channels=out_channels, num_layers=num_layers, omega_0=omega_0, alpha=alpha, beta=beta)
        
        self.bias_p = torch.nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size), requires_grad=True)
        
    def forward(self, coords, x):
        """
        Forward pass of the MAGNet_G slow network.
        
        Args:
            coords (torch.Tensor): Coordinate tensor for generating hyper-kernels.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Generated global hyper-kernels.
        """
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_p.repeat(1, self.hidden_channels, 1, 1)
        out = self.output_linear(out)
        return out


class MAGNet_GC(MAGNetBase):
    """
    MAGNet slow network for generating hyper-kernels for hyper-channel interaction.
    
    This class extends the MAGNetBase and implements the specific forward pass for generating hyper-kernels
    for hyper-channel interaction.
    
    Args:
        data_dim (int): Dimensionality of the input data.
        kernel_size (int): Size of the generated hyper-kernels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of layers in the slow network.
        omega_0 (float): Initial frequency of the sinusoidal activations.
        alpha (float): Parameter for the gamma distribution.
        beta (float): Parameter for the gamma distribution.
    """
    def __init__(self, data_dim: int, kernel_size: int, hidden_channels: int, out_channels: int, num_layers: int,
                 omega_0: float, alpha: float = 6.0, beta: float = 1.0, **kwargs):
        super().__init__(data_dim=data_dim, kernel_size=kernel_size, hidden_channels=hidden_channels,
                         out_channels=out_channels, num_layers=num_layers, omega_0=omega_0, alpha=alpha, beta=beta)
        
        self.bias_p = torch.nn.Parameter(torch.zeros(1, 1, kernel_size), requires_grad=True)
        
    def forward(self, coords, x):
        """
        Forward pass of the MAGNet_GC slow network.
        
        Args:
            coords (torch.Tensor): Coordinate tensor for generating hyper-kernels.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Generated hyper-kernels for hyper-channel interaction.
        """
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_p.repeat(1, self.hidden_channels, 1)
        out = self.output_linear(out)
        return out
    

class MAGNet_L(MAGNetBase):
    """
    MAGNet slow network for generating local hyper-kernels with context-dependency.
    
    This class extends the MAGNetBase and implements the specific forward pass for generating local hyper-kernels
    with context-dependency.
    
    Args:
        data_dim (int): Dimensionality of the input data.
        kernel_size (int): Size of the generated hyper-kernels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of layers in the slow network.
        omega_0 (float): Initial frequency of the sinusoidal activations.
        alpha (float): Parameter for the gamma distribution.
        beta (float): Parameter for the gamma distribution.
    """
    def __init__(self, data_dim: int, kernel_size: int, hidden_channels: int, out_channels: int, num_layers: int,
                 omega_0: float, alpha: float = 6.0, beta: float = 1.0, **kwargs):
        super().__init__(data_dim=data_dim, kernel_size=kernel_size, hidden_channels=hidden_channels,
                         out_channels=out_channels, num_layers=num_layers, omega_0=omega_0, alpha=alpha, beta=beta)
        
        self.bias_pk = torch.nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size), requires_grad=True)
        
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        self.fast_reduce = torch.nn.Conv2d(self.out_channels, self.hidden_channels, 1, 1, 0, bias=True)
        self.fast_fc1 = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=True)
        self.fast_gelu1 = nn.GELU()
        self.fast_linears_x = torch.nn.ModuleList([
            Linear(in_channels=hidden_channels, out_channels=hidden_channels, bias=True)
            for _ in range(num_layers)
        ])
        
        for idx, lin in enumerate(self.fast_linears_x):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
                
        self.alphas = [nn.Parameter(torch.Tensor(hidden_channels).fill_(1)) for _ in range(num_layers)]
        self.betas = [nn.Parameter(torch.Tensor(hidden_channels).fill_(0.1)) for _ in range(num_layers)]
        
    def s_renormalization(self, out, alpha, beta):
        """
        S-renormalization of the hidden outputs.
        
        Args:
            out (torch.Tensor): Hidden output tensor.
            alpha (torch.Tensor): Alpha scaling factor.
            beta (torch.Tensor): Beta scaling factor.
            
        Returns:
            torch.Tensor: S-renormalized hidden output tensor.
        """
        out = out.transpose(0, 1)
        
        delta = out.data.clone()
        assert delta.shape == out.shape

        v = (-1,) + (1,) * (out.dim() - 1)
        out_t = alpha.view(*v) * delta + beta.view(*v) * normalize(out)
        
        return out_t.transpose(0, 1)
    
    def forward(self, coords, x):
        """
        Forward pass of the MAGNet_L slow network.
        
        Args:
            coords (torch.Tensor): Coordinate tensor for generating hyper-kernels.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Generated local hyper-kernels with context-dependency.
        """
        x_gap = torch.nn.functional.adaptive_avg_pool2d(x, (self.kernel_size,) * self.data_dim)
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