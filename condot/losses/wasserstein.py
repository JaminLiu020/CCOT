#!/usr/bin/python3

# imports
from geomloss import SamplesLoss
import torch


def wasserstein_loss(x, y, epsilon=0.1):
    """Computes transport between x and y via Sinkhorn algorithm."""
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=epsilon)

    try:
        return loss(x, y)
    except ValueError as e:
        print(f"Error in computing Wasserstein loss: {e}")
        return torch.tensor(float('nan'))  # 返回 NaN 表示计算失败
