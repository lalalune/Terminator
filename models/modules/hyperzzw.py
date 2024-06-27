import torch

from models.modules import LocalConv


def HyperZZW_G(global_hk, x):
    """
    Global hyperzzw by dot product
    """
    global_ctx_hk = x.mul(global_hk)
        
    global_feat = torch.matmul(global_ctx_hk, x)
        
    return global_feat


class HyperZZW_L(torch.nn.Module):
    """
    Local hyperzzw by sliding window-based convolution
    """
    def __init__(
        self,
        SlowConvType: torch.nn.Module,
        in_channels: int,
        kernel_size: int,
    ):
        super().__init__(
        )
        
        self.slow_net = SlowConvType(in_channels=in_channels, kernel_size=kernel_size)
        
        self.local_conv = LocalConv(in_channels=in_channels, kernel_size=kernel_size)
        self.bias = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, x1, x2):
        local_ctx_hk = self.slow_net(x1)
        
        local_feat = self.local_conv(local_ctx_hk, x2) + self.bias
        
        return local_feat


def HyperZZW_2E(hk, x):
    """
    Hyperzzw with two elementwise multiplication
    """
    ctx_hk = x.mul(hk)
        
    feat = torch.matmul(ctx_hk, x)
        
    return feat
