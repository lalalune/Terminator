import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.09, reduction='mean'):
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


def slow_neural_loss(global_hk_1, global_hk_2, scale_coeff):
    """
    Computes the distance of global hyper-kernels between blocks
    Args:
        global_hk_1: in shallow block with 1*C*H*W
        global_hk_2: in deep block with 1*(C*scale_coeff)*H*W
        scale_coeff: the increase of channel
    Returns:
        L2 distance between them
    """
    loss = l2_loss(torch.tile(global_hk_1, (1, scale_coeff, 1, 1)), global_hk_2)
    
    return loss
