import torch
from torch.optim.optimizer import Optimizer


class Kar3(Optimizer):
  r"""Implements Kar3 algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.5, 0.95)):
    """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.5, 0.95))
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    
    defaults = dict(lr=lr, betas=betas)

    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs **three** optimization steps.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        grad = p.grad
        state = self.state[p]
        
        # State initialization
        if len(state) == 0:
            # Exponential moving average of gradient values
            state['exp_avg_sq'] = torch.zeros_like(p)
            state['exp_avg_sq_2'] = torch.zeros_like(p)

        exp_avg_sq, exp_avg_sq_2 = state['exp_avg_sq'], state['exp_avg_sq_2']
        beta1, beta2 = group['betas']

        sign_grad = torch.sign(grad)
        sign_avg_sq = torch.sign(exp_avg_sq)
        sign_avg_sq_2 = torch.sign(exp_avg_sq_2)
        
        p.add_(sign_grad, alpha=-group['lr'])
        p.add_(sign_avg_sq, alpha=-group['lr'])
        p.add_(sign_avg_sq_2, alpha=-group['lr'])
        
        exp_avg_sq.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq_2.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss
