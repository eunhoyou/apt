import torch.nn.functional as F
import torch
import math

def cross_entropy(pred, target, reduction='none'):
    # print(pred.shape, target.shape)
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction=reduction)
    loss = loss.reshape(target.shape)
    return loss

def masked_loss(pred, target, mask, loss_func):
    """
    Compute masked loss for sequences.
    
    Args:
        pred: predicted values
        target: target values  
        mask: mask tensor where 1 indicates valid positions
        loss_func: loss function to use (should return unreduced loss)
    """
    if pred is None:
        return torch.tensor(0.0, device=target.device)
    
    # Apply loss function with reduction='none' to get per-element loss
    if loss_func == F.smooth_l1_loss:
        loss = F.smooth_l1_loss(pred, target, reduction='none')
    elif loss_func == F.binary_cross_entropy_with_logits:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    elif loss_func == cross_entropy:
        loss = cross_entropy(pred, target, reduction='none')
    else:
        # For other loss functions, try to call with reduction='none'
        try:
            loss = loss_func(pred, target, reduction='none')
        except TypeError:
            # If loss function doesn't accept reduction parameter
            loss = loss_func(pred, target)
    
    # Apply mask
    masked_loss_tensor = loss * mask
    
    # Reduce over masked dimensions
    total_loss = masked_loss_tensor.sum()
    valid_count = mask.sum()
    
    # Avoid division by zero
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=target.device)