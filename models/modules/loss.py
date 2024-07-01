import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)


def calculate_product_between_positions(x, start_idx, end_idx):
    product = 1
    for i in range(start_idx, end_idx + 1):
        product *= x[i]
    return product


def cal_slow_loss(ghk_bs, factors):
    """
    Computes the distance of global hyper-kernels between blocks
    Args:
        ghk_bs: global hyper kernels of all blocks are in a list
        factors: the bottleneck factors in all blocks
    Returns:
        L2 distance between them
    """
    num_block = len(ghk_bs)
    slow_loss = 0.0
    for i in range(num_block-1, 0, -1):
        ghk_deep = ghk_bs[i]
        for j in range(0, i, 1):
            ghk_shallow = ghk_bs[j]
            factor = calculate_product_between_positions(factors, j, i-1)
            
            slow_loss += l2_loss(torch.tile(ghk_bs[j], (1, factor, 1, 1)), ghk_bs[i])
            
    return slow_loss


def cal_slow_loss_channel_sum(ghk_bs):
    """
    Computes the distance of global hyper-kernels between blocks
    Args:
        ghk_bs: global hyper kernels of all blocks are in a list
    Returns:
        L2 distance between them
    """
    num_block = len(ghk_bs)
    slow_loss = 0.0
    for i in range(num_block-1, 0, -1):
        ghk_deep = ghk_bs[i]
        for j in range(0, i, 1):
            ghk_shallow = ghk_bs[j]
            
            slow_loss += l2_loss(ghk_bs[j].sum(1), ghk_bs[i].sum(1))
            
    return slow_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)
    

class LnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on slow net and fast net
        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def _calculate_loss_weights(self, model):
        loss = 0.0
        
        # loss on fast net
        params_outside_kernelnets = filter(
            lambda x: "fast" in x[0] and "Kernel" not in x[0], model.named_parameters()
        )
        for named_param in params_outside_kernelnets:
            loss += named_param[1].norm(self.norm_type)

        # loss on slow net
        for n, m in model.named_modules():
            if 'Kernel' in n and isinstance(m, nn.Conv2d):
                loss += m.weight.norm(self.norm_type) * 1e-6
    
        return loss

    def forward(
        self,
        model: torch.nn.Module,
    ):
        
        loss = self._calculate_loss_weights(model)
        loss = self.weight_loss * loss
        
        return loss
