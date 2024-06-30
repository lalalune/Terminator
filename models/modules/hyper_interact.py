from functools import partial
import torch
import torch.nn as nn
from models.modules.hyperzzw import HyperZZW_2E
from models.modules.slow_net import SlowNet_GC, SlowNet_G

class HyperChannelInteract(nn.Module):
    """
    Hyper-channel interaction module for global cross-channel interaction.
    
    This module performs channel-wise interaction by generating channel-wise hyper-weights
    using a slow network and applying them to the input tensor through the HyperZZW_2E operator.
    It achieves channel mixing by capturing global cross-channel dependencies.
    
    Args:
        channel (int): Number of channels in the input tensor.
        kernel_cfg (dict): Configuration for the slow network kernel.
    """
    def __init__(self, channel: int, kernel_cfg: dict):
        super(HyperChannelInteract, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        kernel_cfg.size = channel
        SlowNetType_GC = partial(SlowNet_GC, data_dim=1, kernel_cfg=kernel_cfg)
        self.slow_net_gc = SlowNetType_GC(in_channels=1)

    def forward(self, x):
        """
        Forward pass of the hyper-channel interaction module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor after channel-wise interaction.
        """
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        
        hk = self.slow_net_gc(y)  # (1, 1, C)
        score = HyperZZW_2E(hk, y)  # (B, 1, C)
        score = score.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)

        score = self.sigmoid(score)

        return x * score.expand_as(x)  # (B, C, H, W)


class HyperInteract(nn.Module):
    """
    Hyper interaction module for combining hyper-channel and hyper-spatial interactions.
    
    This module integrates both hyper-channel and hyper-spatial interactions to enhance the
    feature extraction capability of the network. It generates channel-wise and spatial-wise
    hyper-weights using slow networks and applies them to the input tensor through the HyperZZW_2E
    operator. The resulting weights are used for pixel-level filtering of the input tensor.
    
    Args:
        channel (int): Number of channels in the input tensor.
        kernel_cfg_c (dict): Configuration for the channel-based slow network kernel.
        kernel_cfg_s (dict): Configuration for the spatial-based slow network kernel.
    """
    def __init__(self, channel: int, kernel_cfg_c: dict, kernel_cfg_s: dict):
        super(HyperInteract, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        kernel_cfg_c.size = channel
        SlowNetType_GC = partial(SlowNet_GC, data_dim=1, kernel_cfg=kernel_cfg_c)
        self.slow_net_gc = SlowNetType_GC(in_channels=1)
        
        SlowNetType_GS = partial(SlowNet_G, data_dim=2, kernel_cfg=kernel_cfg_s)
        self.slow_net_gs = SlowNetType_GS(in_channels=1)

    def forward(self, y, x=None):
        """
        Forward pass of the hyper interaction module.
        
        Args:
            y (torch.Tensor): High-level output tensor of shape (B, C, H, W).
            x (torch.Tensor, optional): Input tensor to perform pixel-level filtering. If not provided, y is used.
            
        Returns:
            torch.Tensor: Output tensor after hyper-channel and hyper-spatial interactions.
        """
        y_c = self.avg_pool(y)  # (B, C, 1, 1)
        y_c = y_c.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        hk_gc = self.slow_net_gc(y_c)  # (1, 1, C)
        score_c = HyperZZW_2E(hk_gc, y_c)
        score_c = score_c.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        
        y_s = y.mean(axis=1, keepdims=True)  # (B, 1, H, W)
        hk_gs = self.slow_net_gs(y_s)  # (1, H, W)
        score_s = HyperZZW_2E(hk_gs, y_s)  # (B, 1, H, W)

        score = score_s.mul(score_c)  # (B, 1, H, W) * (B, C, 1, 1) -> (B, C, H, W)
        score = self.sigmoid(score)

        if x is not None:
            return x * score
        else:
            return y * score