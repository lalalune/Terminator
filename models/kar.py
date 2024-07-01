"""
This module implements two variations of custom optimization algorithms, Kar2 and Kar3.
These optimizers introduce modified update rules that uniquely utilize the sign of gradients
and running averages to adjust model parameters during training. Here's a detailed breakdown
of each optimizer:
1. Kar3 Optimizer:
   - Kar3 utilizes two sets of beta coefficients for running averages of the gradients.
   - During each update step, Kar3 adjusts model parameters based on the sign of the current
     gradient and the signs of these two exponential moving averages. This approach allows 
     Kar3 to moderate parameter updates more effectively, 
     considering both short-term and longer-term trends in gradient changes.
2. Kar2 Optimizer:
   - A simplified variant of Kar3, Kar2 uses a single set of beta coefficients for maintaining a
     running average of the gradients.
   - It modifies parameters using their gradient's sign combined with the sign of the single
     running average, providing a more straightforward but robust mechanism for parameter updates.
     
Both optimizers utilize sign-based updates, focusing on the directionality of the gradient rather than 
its magnitude. This approach differs from typical gradient descent algorithms, which adjust parameters 
based on the magnitude of the gradient. 
The differences from the previous Sign-SGD include two points: first, we remove weight decay from the optimizer; 
second, the gradient is no longer updated by addition with the momentum term.
"""

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


class Kar2(Optimizer):
  r"""Implements Kar2 algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.95)):
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
    
    defaults = dict(lr=lr, betas=betas)

    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs **two** optimization steps.
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

        exp_avg_sq = state['exp_avg_sq']
        beta = group['betas']

        sign_grad = torch.sign(grad)
        sign_avg_sq = torch.sign(exp_avg_sq)
        
        p.add_(sign_grad, alpha=-group['lr'])
        p.add_(sign_avg_sq, alpha=-group['lr'])
        
        exp_avg_sq.mul_(beta).add_(grad, alpha=1 - beta)

    return loss
