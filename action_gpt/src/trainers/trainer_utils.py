import torch.nn.functional as F
import math

def cross_entropy(pred, target, reduction):
    # print(pred.shape, target.shape)
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction=reduction)
    loss = loss.reshape(target.shape)
    return loss