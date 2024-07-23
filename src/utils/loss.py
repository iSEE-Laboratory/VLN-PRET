import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(probs, target, ignore_mask=None, reduction='mean'):
    """
    nn.CrossEntropyLoss take non-normalized logits as input,
    which is not suitable here.

    Args:
        probs: tensor of shape (batch_size, N)
        target: tensor of shape (batch_size,) or (batch_size, 1)
        ignore_mask: a bool mask indicate which samples should be ignored.
        reduction: 'mean', 'none'
    """
    target = target.view(-1, 1)
    if ignore_mask is not None:
        ignore_mask = ignore_mask.view(-1, 1)
        target[ignore_mask] = 0  # ignore_id may be invalid

    eps = torch.tensor(1e-5, device=probs.device)

    loss = probs.gather(dim=1, index=target)  # (B, 1)
    loss = - torch.max(loss, eps).log()  # take max to avoid log(0)

    if ignore_mask is not None:
        loss[ignore_mask] = 0
    
    if reduction == 'mean':
        if ignore_mask is None or not ignore_mask.any():
            return loss.mean()
        else:
            return loss.sum() / ignore_mask.sum()
    elif reduction == 'none':
        return loss
    else:
        raise NotImplementedError()
