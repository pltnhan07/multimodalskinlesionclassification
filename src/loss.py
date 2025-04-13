# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', device='cpu'):
        """
        Focal Loss implementation.
        Args:
            gamma (float): Focusing parameter.
            alpha (list or np.ndarray or torch.Tensor): Class balancing weight.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
            device (str): Device to put the alpha tensor.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float).to(device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                raise TypeError("Alpha must be a list, np.ndarray, or torch.Tensor")
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.view(-1, 1)
        log_p = log_probs.gather(1, targets).squeeze(1)
        p = probs.gather(1, targets).squeeze(1)
        loss = -((1 - p) ** self.gamma) * log_p
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.squeeze(1))
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
