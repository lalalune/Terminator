import torch
from models.modules import linear

class RGU(torch.nn.Module):
    """
    Recursive Gated Unit (RGU) module.
    
    This module incorporates recursion into the gated linear unit (GLU) to enhance the representation
    learning capability of the network. It applies linear transformations to the input tensor and
    performs elementwise multiplication and addition operations to generate the output.
    
    Args:
        data_dim (int): Dimensionality of the input data.
        in_channels (int): Number of input channels.
        bias_size (int): Size of the bias tensor.
    """
    def __init__(self, data_dim: int, in_channels: int, bias_size: int, **kwargs):
        super().__init__()
        
        self.bias_size = bias_size
        self.hidden_channels = in_channels
        
        Linear = getattr(linear, f"Linear{data_dim}d")
        
        self.fast_linear_k = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        self.fast_linear_v = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        self.fast_linear_q = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        self.fast_linear_y = Linear(in_channels=in_channels, out_channels=in_channels, bias=True)
        
        torch.nn.init.kaiming_uniform_(self.fast_linear_k.weight, nonlinearity="linear")
        torch.nn.init.kaiming_uniform_(self.fast_linear_v.weight, nonlinearity="linear")
        torch.nn.init.kaiming_uniform_(self.fast_linear_q.weight, nonlinearity="linear")
        torch.nn.init.kaiming_uniform_(self.fast_linear_y.weight, nonlinearity="linear")
        self.fast_linear_k.bias.data.fill_(0.0)
        self.fast_linear_v.bias.data.fill_(0.0)
        self.fast_linear_q.bias.data.fill_(0.0)
        self.fast_linear_y.bias.data.fill_(0.0)
        
        norm_name = f"InstanceNorm2d" 
        NormType = getattr(torch.nn, norm_name)
        self.ins_std = NormType(in_channels, eps=1e-05, momentum=1.0, affine=False, track_running_stats=False)
        
        NonlinearType = getattr(torch.nn, 'GELU')
        self.nonlinear = NonlinearType()
        
        self.bias_p = torch.nn.Parameter(torch.zeros(1, 1, bias_size, bias_size), requires_grad=True)
        
    def forward(self, x):
        """
        Forward pass of the RGU module.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying the RGU operations.
        """
        out_k = self.fast_linear_k(x)
        out_v = self.fast_linear_v(x)
        
        out_q = self.nonlinear(self.ins_std(out_k * self.fast_linear_q(out_v))) + self.bias_p.repeat(x.shape[0], self.hidden_channels, 1, 1)
        
        out_y = self.fast_linear_y(out_q)
        
        return out_y