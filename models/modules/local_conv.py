import torch

from models.modules import linear


class LocalConv(torch.nn.Module):
    """
    A local branch
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
    ):
        super().__init__(
        )
        
        self.padding = (kernel_size-1) // 2
        self.local_conv = getattr(torch.nn.functional, f"conv2d")
        
        ChannelMixerClass = getattr(linear, f"Linear2d")
        self.channel_mixer = ChannelMixerClass(
            in_channels=in_channels,
            out_channels=in_channels,
            bias=True,
        )
        torch.nn.init.kaiming_normal_(self.channel_mixer.weight)
        torch.nn.init._no_grad_fill_(self.channel_mixer.bias, 0.0)
        
        # Batch Standardization
        norm_name = f"BatchNorm2d"
        NormType = getattr(torch.nn, norm_name)
        self.batch_std = NormType(
            in_channels, 
            eps=1e-05, 
            momentum=1.0,
            affine=False, 
            track_running_stats=False
        )
        
        # Nonlinear
        NonlinearType = getattr(torch.nn, 'GELU')
        self.nonlinear = NonlinearType()
        
    def forward(self, local_hk, x):
        groups = local_hk.shape[0]
        
        local_feat = self.local_conv(x, local_hk, stride=1, padding=self.padding, dilation=1, groups=groups)
        local_feat = self.channel_mixer(local_feat)
        local_feat = self.nonlinear(self.batch_std(local_feat))
            
        return local_feat
