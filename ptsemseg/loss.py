import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pdb

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n,h,w ,1).repeat(1, 1, 1, c) < 11]
    log_p = log_p.view(-1, c)

    mask = target < 11  # remove unlabelled class
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    loss = F.nll_loss(log_p, target, weight=weight, reduce=False)
    topk_loss, _ = loss.topk(K)
    reduced_topk_loss = topk_loss.sum() / K

    return reduced_topk_loss





