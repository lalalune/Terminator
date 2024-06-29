import numpy as np

import torch

from models.modules import linear


class RGU(torch.nn.Module):
    """
    Recursive Gated Unit: incorporates recursion in the gated linear unit (GLU).
    """
    def __init__(
        self,
        data_dim: int,
        in_channels: int,
        bias_size: int,
        **kwargs,
    ):
        super().__init__(
        )
        
        self.bias_size = bias_size
        self.hidden_channels = in_channels
        
        # Define type of linear
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        # W_K
        self.fast_linear_k = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        # W_V
        self.fast_linear_v = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        # W_Q &
        self.fast_linear_q = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        # W_Y
        self.fast_linear_y = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        
        # Initialize
        torch.nn.init.kaiming_uniform_(self.fast_linear_k.weight, nonlinearity="linear")
        torch.nn.init.kaiming_uniform_(self.fast_linear_v.weight, nonlinearity="linear")
        torch.nn.init.kaiming_uniform_(self.fast_linear_q.weight, nonlinearity="linear")
        torch.nn.init.kaiming_uniform_(self.fast_linear_y.weight, nonlinearity="linear")
        self.fast_linear_k.bias.data.fill_(0.0)
        self.fast_linear_v.bias.data.fill_(0.0)
        self.fast_linear_q.bias.data.fill_(0.0)
        self.fast_linear_y.bias.data.fill_(0.0)
        
        # Instance Standardization
        norm_name = f"InstanceNorm2d" 
        NormType = getattr(torch.nn, norm_name)
        self.ins_std = NormType(
            in_channels, 
            eps=1e-05, 
            momentum=1.0, 
            affine=False, 
            track_running_stats=False
        )
        # Nonlinear
        NonlinearType = getattr(torch.nn, 'GELU')
        self.nonlinear = NonlinearType()
        
        # Bias
        self.bias_p = torch.nn.Parameter(
            torch.zeros(1, 1, bias_size, bias_size), requires_grad=True
        )
        
    def forward(self, x):
        out_k = self.fast_linear_k(x)
        out_v = self.fast_linear_v(x)
        
        out_q = self.nonlinear(self.ins_std(out_k * self.fast_linear_q(out_v))) + self.bias_p.repeat(x.shape[0], self.hidden_channels, 1, 1)
        
        out_y = self.fast_linear_y(out_q)
        
        return out_y
