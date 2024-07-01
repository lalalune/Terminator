import torch
from torch.nn.functional import dropout

from models.modules.g_ibs import G_IBS
from models.modules.fast_multi_branch import FastMultiBranchLayer

from omegaconf import OmegaConf


class SFNEBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        FastMultiBranchLayerType: torch.nn.Module,
        NonlinearType: torch.nn.Module,
        DropoutType: torch.nn.Module,
        dropout: float,
        bottleneck_factor: int,
    ):
        super().__init__()
        
        self.fast_multi_branch = FastMultiBranchLayerType(
            in_channels=in_channels, 
            out_channels=out_channels, 
            bottleneck_factor=bottleneck_factor
        )
        
        self.norm = G_IBS(int(in_channels*bottleneck_factor), group_num=12)
        self.nonlinear = NonlinearType()
        self.dp = DropoutType(dropout)
        
        self.bias = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, x, x_pre=None, x_pre_pre=None):
        
        out, global_hk = self.fast_multi_branch(x, x_pre, x_pre_pre)
        out = self.nonlinear(self.norm(out))
        out_ = self.dp(out + self.bias)
        
        return out_, global_hk
