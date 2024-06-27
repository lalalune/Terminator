import math

import torch
import torch.nn as nn


class G_IBS(nn.Module):
    def __init__(
        self,
        channels: int,
        group_num: int,
    ):
        super(G_IBS, self).__init__()
        
        g_channels = int(math.floor(channels / group_num))
        self.groups = [g_channels] * group_num
        
        self.batch_std = nn.BatchNorm2d(
            num_features=g_channels, 
            eps=1e-05, 
            momentum=1.0, 
            affine=False, 
            track_running_stats=False
        )
        self.ins_std = nn.InstanceNorm2d(
            num_features=g_channels,
            momentum=1.0,
            affine=False
        )

    def forward(self, x):
        splits = torch.split(x, split_size_or_sections=self.groups, dim=1)
        
        x_gibs = []
        for i, split in enumerate(splits):
            if i % 2 == 0:
                x_bs= self.batch_std(split.contiguous())
                x_gibs.append(x_bs)
            else:
                x_is = self.ins_std(split.contiguous())
                x_gibs.append(x_is)

        out = torch.cat(x_gibs, dim=1)
        
        return out
