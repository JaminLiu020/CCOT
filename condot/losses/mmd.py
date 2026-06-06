#!/usr/bin/python3

# imports
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

import torch
from geomloss import SamplesLoss


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


def compute_scalar_mmd_gpu(target, transport):
    gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = []
    for gamma in gammas:
        mmd = SamplesLoss("gaussian", blur=1/(2*gamma))(target, transport)
        mmds.append(mmd)
    return torch.stack(mmds).mean()
