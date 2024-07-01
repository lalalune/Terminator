import numpy as np
from functools import partial

import torch
import torch.nn as nn

from . import modules
from models.modules import linear
from models.modules.g_ibs import G_IBS
from models.modules.slow_net import SlowNet_G, SlowNet_L
from models.modules.fast_multi_branch import FastMultiBranchLayer
from models.modules.loss import cal_slow_loss, cal_slow_loss_channel_sum

from omegaconf import OmegaConf


class TerminatorBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_cfg: OmegaConf,
        kernel_cfg_g: OmegaConf,
        kernel_cfg_l: OmegaConf,
        kernel_cfg_gc: OmegaConf,
        kernel_cfg_gc_hi: OmegaConf,
        kernel_cfg_gs_hi: OmegaConf,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_cfg.no_hidden
        no_blocks = net_cfg.no_blocks
        data_dim = net_cfg.data_dim
        dropout_type = net_cfg.dropout_type   

        # Define NonlinearType
        nonlinearity = "GELU"
        NonlinearType = getattr(torch.nn, nonlinearity)

        # Define Dropout layer type
        DropoutType = getattr(torch.nn, dropout_type)
        
        # Define BlockType
        block_type = net_cfg.block.type
        
        # Create Blocks
        BlockType = getattr(modules, f"{block_type}Block")
        blocks = []
        for i in range(no_blocks):
            print(f"Block {i}/{no_blocks}")
            
            input_ch = net_cfg.block.in_channel[i]
            hidden_ch = net_cfg.block.in_channel[i]
            
            kernel_cfg_g.num_layers = kernel_cfg_g.num_layers_list[i]
            SlowNetType_G = partial(
                SlowNet_G,
                data_dim=data_dim,
                kernel_cfg=kernel_cfg_g,
                )
            kernel_cfg_l.size = kernel_cfg_l.sizes[0]
            SlowNetType_L1 = partial(
                SlowNet_L,
                data_dim=data_dim,
                kernel_cfg=kernel_cfg_l,
                kernel_size=kernel_cfg_l.sizes[0],
                )
            kernel_cfg_l.size = kernel_cfg_l.sizes[1]
            SlowNetType_L2 = partial(
                SlowNet_L,
                data_dim=data_dim,
                kernel_cfg=kernel_cfg_l,
                kernel_size=kernel_cfg_l.sizes[1],
                )
            kernel_cfg_l.size = kernel_cfg_l.sizes[2]
            SlowNetType_L3 = partial(
                SlowNet_L,
                data_dim=data_dim,
                kernel_cfg=kernel_cfg_l,
                kernel_size=kernel_cfg_l.sizes[2],
                )
            kernel_cfg_l.size = kernel_cfg_l.sizes[3]
            SlowNetType_L4 = partial(
                SlowNet_L,
                data_dim=data_dim,
                kernel_cfg=kernel_cfg_l,
                kernel_size=kernel_cfg_l.sizes[3],
                )
            FastMultiBranchLayerType = partial(
                FastMultiBranchLayer,
                num_branch=net_cfg.block.num_branch,
                data_dim=data_dim,
                SlowNetType_G=SlowNetType_G,
                SlowNetType_L1=SlowNetType_L1,
                SlowNetType_L2=SlowNetType_L2,
                SlowNetType_L3=SlowNetType_L3,
                SlowNetType_L4=SlowNetType_L4,
                kernel_cfg=kernel_cfg_g,
                kernel_cfg_gc=kernel_cfg_gc,
                kernel_cfg_gc_hi=kernel_cfg_gc_hi,
                kernel_cfg_gs_hi=kernel_cfg_gs_hi,
                local_kernel_sizes=kernel_cfg_l.sizes,
                num_concat_pre=net_cfg.block.num_concat_pre[i],
                )

            blocks.append(
                BlockType(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    FastMultiBranchLayerType=FastMultiBranchLayerType,
                    NonlinearType=NonlinearType,
                    DropoutType=DropoutType,
                    dropout=net_cfg.dropout[i],
                    bottleneck_factor=net_cfg.block.bottleneck_factors[i],)
                )

        self.blocks = torch.nn.Sequential(*blocks)

        # Define Output Layers:
        final_no_hidden = net_cfg.final_no_hidden
        self.out_layer = nn.Linear(final_no_hidden, out_channels)
        torch.nn.init.kaiming_normal_(self.out_layer.weight)
        self.out_layer.bias.data.fill_(value=0.0)
        
        # Save variables in self
        self.data_dim = data_dim
        self.fast_bias = [torch.nn.Parameter(torch.zeros(1))] * no_blocks
        self.factors = net_cfg.block.bottleneck_factors
        self.slow_neural_loss = 0
        
    def forward(self, x):
        raise NotImplementedError


class Terminator_image(TerminatorBase):
    def forward(self, x, train_mode=False):
            
        # ------------- block training ------------- #
        out, ghk_b_0 = self.blocks[0](x)
        out_pre = out

        out, ghk_b_1 = self.blocks[1](out + self.fast_bias[0].cuda())
        out_pre_pre = out
        
        out, ghk_b_2 = self.blocks[2](out + self.fast_bias[1].cuda(), x_pre=out_pre)
        out_pre = out
        
        out, ghk_b_3 = self.blocks[3](out + self.fast_bias[2].cuda(), x_pre=out_pre_pre)

        out, ghk_b_4 = self.blocks[4](out + self.fast_bias[3].cuda(), x_pre=out_pre, x_pre_pre=out_pre_pre)

        if train_mode:                
            # self.slow_neural_loss += cal_slow_loss([ghk_b_0, ghk_b_1, ghk_b_2, ghk_b_3, ghk_b_4], self.factors)
            self.slow_neural_loss = cal_slow_loss_channel_sum([ghk_b_0, ghk_b_1, ghk_b_2, ghk_b_3, ghk_b_4])

        out = torch.nn.functional.adaptive_avg_pool2d(
            out,
            (1,) * self.data_dim,
        )

        out = self.out_layer(out.squeeze() + self.fast_bias[-1].cuda())
           
        return out.squeeze(), self.slow_neural_loss
