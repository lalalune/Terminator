from functools import partial

import torch
import torch.nn as nn

from models.modules.hyperzzw import HyperZZW_2E
from models.modules.slow_net import SlowNet_GC, SlowNet_G

from omegaconf import OmegaConf
    

class HyperChannelInteract(nn.Module):
    """
    Args:
        channel: number of channels of the input feature map
        kernel_cfg: parameters of slow net
    """
    def __init__(
        self, 
        channel: int, 
        kernel_cfg: OmegaConf
    ):
        super(HyperChannelInteract, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        kernel_cfg.size = channel
        SlowNetType_GC = partial(
            SlowNet_GC,
            data_dim=1,
            kernel_cfg=kernel_cfg,
            )
        self.slow_net_gc = SlowNetType_GC(in_channels=1)

    def forward(self, x):   # x: NCHW
        y = self.avg_pool(x)   # NC11
        y = y.squeeze(-1).transpose(-1, -2)   # N1C
        
        hk = self.slow_net_gc(y)   # 11C
        # **hyperzzw_2e**
        score = HyperZZW_2E(hk, y)   # N1C
        score = score.transpose(-1, -2).unsqueeze(-1)   # NC11

        score = self.sigmoid(score)

        return x * score.expand_as(x)   # NCHW


class HyperInteract(nn.Module):
    """.
    Args:
        channel: number of channels of the input feature map
        kernel_cfg_c: parameters of channel-based slow net
        kernel_cfg_s: parameters of spatial-based slow net
    """
    def __init__(
        self, 
        channel: int, 
        kernel_cfg_c: OmegaConf, 
        kernel_cfg_s: OmegaConf
    ):
        super(HyperInteract, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        # slow net for channel
        kernel_cfg_c.size = channel
        SlowNetType_GC = partial(
            SlowNet_GC,
            data_dim=1,
            kernel_cfg=kernel_cfg_c,
            )
        self.slow_net_gc = SlowNetType_GC(in_channels=1)
        
        # slow net for spatial
        SlowNetType_GS = partial(
            SlowNet_G,
            data_dim=2,
            kernel_cfg=kernel_cfg_s,
            )
        self.slow_net_gs = SlowNetType_GS(in_channels=1)

    def forward(self, y, x=None):   # high-level output y: NCHW
        
        # hyper-channel interaction
        y_c = self.avg_pool(y)   # NC11
        y_c = y_c.squeeze(-1).transpose(-1, -2)   # N1C
        hk_gc = self.slow_net_gc(y_c)   # 11C
        # **hyperzzw_2e**
        score_c = HyperZZW_2E(hk_gc, y_c)
        score_c = score_c.transpose(-1, -2).unsqueeze(-1)   # NC11
        
        # hyper-spatial interaction
        y_s = y.mean(axis=1, keepdims=True)  # N1HW
        hk_gs = self.slow_net_gs(y_s)   # 1HW
        # **hyperzzw_2e**
        score_s = HyperZZW_2E(hk_gs, y_s)   # N1HW

        # channel-spatial
        score = score_s.mul(score_c)   # N1HW*NC11 -> NCHW
        score = self.sigmoid(score)

        if x is not None:
            return x * score
        else:
            return y * score
