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


