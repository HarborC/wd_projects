"""
Per-point Adam optimizer for Gaussian Splatting.

This module implements a modified Adam optimizer that supports per-point
learning rates, which is useful for confidence-weighted optimization.
"""

import torch
from torch.optim import Optimizer
import math


class PerPointAdam(Optimizer):
    """
    Adam optimizer with per-point learning rate support.

    This optimizer allows each parameter to have a different learning rate
    for each point in the point cloud, which is useful for confidence-weighted
    optimization in sparse-view 3D reconstruction.
    """

    def __init__(self, params, lr=0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        """Initialize PerPointAdam optimizer."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(PerPointAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PerPointAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Apply per-point learning rate if available
                if 'per_point_lr' in group:
                    per_point_lr = group['per_point_lr']
                    # Ensure per_point_lr has the right shape
                    if per_point_lr.dim() == 1 and p.data.dim() == 2:
                        # Broadcasting: (N,) -> (N, 1) for (N, D) tensor
                        per_point_lr = per_point_lr.unsqueeze(-1)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    # Apply per-point learning rate
                    p.data.addcdiv_(exp_avg * per_point_lr, denom, value=-step_size)
                else:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'])

        return loss
