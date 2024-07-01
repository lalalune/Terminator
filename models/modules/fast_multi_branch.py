import torch

from models.modules import rgu
from models.modules import linear
from models.modules.local_conv import LocalConv
from models.modules.hyperzzw import HyperZZW_G, HyperZZW_L
from models.modules.hyper_interact import HyperChannelInteract, HyperInteract

from omegaconf import OmegaConf


class FastMultiBranchLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_branch: int,
        data_dim: int,
        SlowNetType_G: torch.nn.Module,
        SlowNetType_L1: torch.nn.Module,
        SlowNetType_L2: torch.nn.Module,
        SlowNetType_L3: torch.nn.Module,
        SlowNetType_L4: torch.nn.Module,
        kernel_cfg: OmegaConf,
        kernel_cfg_gc: OmegaConf,
        kernel_cfg_gc_hi: OmegaConf,
        kernel_cfg_gs_hi: OmegaConf,
        local_kernel_sizes: OmegaConf,
        bottleneck_factor=2,
        num_concat_pre=0,
    ):
        super().__init__(
        )
        
        self.kernel_size = kernel_cfg.size
        
        # Slow net for a global branch
        self.slow_net_g = SlowNetType_G(in_channels=in_channels)
        
        # Slow nets for four global branches - HyperZZW in the hyper-kernel generation process
        self.slow_net_l1 = SlowNetType_L1(in_channels=in_channels, kernel_size=local_kernel_sizes[0])
        self.slow_net_l2 = SlowNetType_L2(in_channels=in_channels, kernel_size=local_kernel_sizes[1])
        self.slow_net_l3 = SlowNetType_L3(in_channels=in_channels, kernel_size=local_kernel_sizes[2])
        self.slow_net_l4 = SlowNetType_L4(in_channels=in_channels, kernel_size=local_kernel_sizes[3])
        
        self.local_interact_conv_1 = LocalConv(in_channels=in_channels, kernel_size=local_kernel_sizes[0])
        self.local_interact_conv_2 = LocalConv(in_channels=in_channels, kernel_size=local_kernel_sizes[1])
        self.local_interact_conv_3 = LocalConv(in_channels=in_channels, kernel_size=local_kernel_sizes[2])
        self.local_interact_conv_4 = LocalConv(in_channels=in_channels, kernel_size=local_kernel_sizes[3])
        
        self.bias1 = torch.nn.Parameter(torch.zeros(1))
        self.bias2 = torch.nn.Parameter(torch.zeros(1))
        self.bias3 = torch.nn.Parameter(torch.zeros(1))
        self.bias4 = torch.nn.Parameter(torch.zeros(1))
        
        # ---------------------- Channel Mixers -------------------- #
        # MLP
        ChannelMixerClass = getattr(linear, f"Linear{data_dim}d")
        self.channel_mixer_mlp = ChannelMixerClass(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
        )
        torch.nn.init.kaiming_normal_(self.channel_mixer_mlp.weight)
        torch.nn.init._no_grad_fill_(self.channel_mixer_mlp.bias, 0.0)
        
        # ---------------------- Channel Bottleneck Layer -------------------- #
        num_concat = 10 + num_concat_pre
        BottleneckLayerClass = getattr(linear, f"Linear{data_dim}d")
        self.bottleneck_layer = BottleneckLayerClass(
            in_channels=in_channels*num_concat,
            out_channels=out_channels*bottleneck_factor,
            bias=True,
        )
        torch.nn.init.kaiming_normal_(self.bottleneck_layer.weight)
        torch.nn.init._no_grad_fill_(self.bottleneck_layer.bias, 0.0)
        
        # RGU
        FastKernelClass = getattr(rgu, 'RGU')
        self.Fast_Kernel_rgu = FastKernelClass(
            data_dim=data_dim,
            in_channels=in_channels,
            bias_size=self.kernel_size,
        )
        FastKernelClass_2 = getattr(rgu, 'RGU')
        self.Fast_Kernel_rgu_2 = FastKernelClass_2(
            data_dim=data_dim,
            in_channels=in_channels,
            bias_size=self.kernel_size,
        )
        
        # Hyper-Channel Interaction
        self.channel_mixer_hci = HyperChannelInteract(
            channel=in_channels, 
            kernel_cfg=kernel_cfg_gc
        )
        
        # ---------------------- Hyper Interaction -------------------- #
        self.hyper_interact = HyperInteract(
            channel=in_channels, 
            kernel_cfg_c=kernel_cfg_gc_hi, 
            kernel_cfg_s=kernel_cfg_gs_hi,
        )

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

        # Sigmoid
        NonlinearType = getattr(torch.nn, 'Sigmoid')
        self.nonlinear = NonlinearType()
        
    def construct_fast_kernel_rgu(self, x):
        conv_kernel = self.Fast_Kernel_rgu(x)
        return conv_kernel
    def construct_fast_kernel_rgu_slow(self, x):
        conv_kernel = self.Fast_Kernel_rgu_2(x)
        return conv_kernel
    
    def forward(self, x, x_pre=None, x_pre_pre=None):
        
        # --------------- Global --------------- #
        global_hk = self.slow_net_g(x)
        global_ctx_hk = x.mul(global_hk)
        
        # mlp
        slow_mlp = self.construct_fast_kernel_rgu_slow(global_ctx_hk)
        fast_mlp = self.construct_fast_kernel_rgu(x)
        
        # fast
        fast = slow_mlp + fast_mlp
        fast = self.ins_std(fast)

        # mixer mlp
        mixer_x = self.channel_mixer_mlp(x)
        mixer_x = self.ins_std(mixer_x)
        
        # hyper-channel interaction
        x_eca = self.channel_mixer_hci(x)
        x_eca = self.ins_std(x_eca)

        # global branches
        out = torch.matmul(global_ctx_hk, fast)
        out_x = torch.matmul(global_ctx_hk, mixer_x)
        out_eca = torch.matmul(global_ctx_hk, x_eca)
        
        # si-glu
        out_glu = self.nonlinear(out) * out
        
        # hyper interaction
        x_eca_hyper = self.hyper_interact(out, x)
        
        # local branches
        local_kernel_f1 = self.slow_net_l1(x) 
        local_kernel_f2 = self.slow_net_l2(x_eca)
        local_kernel_f3 = self.slow_net_l3(mixer_x)
        local_kernel_f4 = self.slow_net_l4(x)
        
        local_feat_1 = self.local_interact_conv_1(local_kernel_f1, x_eca) + self.bias1
        local_feat_2 = self.local_interact_conv_2(local_kernel_f2, x) + self.bias2
        local_feat_3 = self.local_interact_conv_3(local_kernel_f3, x) + self.bias3
        local_feat_4 = self.local_interact_conv_4(local_kernel_f4, mixer_x) + self.bias4
        
        # --------------- channel concat --------------- #
        if x_pre is not None and x_pre_pre is None:
            scale = x.shape[1] // x_pre.shape[1]
            x_pre_1 = x_pre.repeat(1, scale, 1, 1)
            
            out_concat = torch.cat([out, local_feat_2, x_pre_1, fast, out_x, local_feat_1, out_glu, local_feat_3, out_eca, local_feat_4, x_eca_hyper], 1)
            
        elif x_pre_pre is not None:
            scale = x.shape[1] // x_pre.shape[1]
            x_pre_1 = x_pre.repeat(1, scale, 1, 1)
            
            scale = x.shape[1] // x_pre_pre.shape[1]
            x_pre_pre_1 = x_pre_pre.repeat(1, scale, 1, 1)
            
            out_concat = torch.cat([out, local_feat_2, x_pre_1, fast, out_x, local_feat_1, out_glu, local_feat_3, out_eca, local_feat_4, x_pre_pre_1, x_eca_hyper], 1)
            
        else:
            out_concat = torch.cat([out, local_feat_2, fast, out_x, local_feat_1, out_glu, local_feat_3, out_eca, local_feat_4, x_eca_hyper], 1)
        
        out_o = self.bottleneck_layer(out_concat)
        
        return out_o, global_hk
