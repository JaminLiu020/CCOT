from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel


def compute_mmd(x: np.ndarray, y: np.ndarray, gammas=None) -> float:
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    values = []
    for gamma in gammas:
        xx = rbf_kernel(x, x, gamma)
        xy = rbf_kernel(x, y, gamma)
        yy = rbf_kernel(y, y, gamma)
        values.append(float(xx.mean() + yy.mean() - 2 * xy.mean()))
    return float(np.mean(values))


def compute_l2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0), ord=2))


def compute_r2(x: np.ndarray, y: np.ndarray) -> float:
    return float(r2_score(x.mean(axis=0), y.mean(axis=0)))


def _cov_with_eps(x: np.ndarray, eps: float) -> np.ndarray:
    n = x.shape[0]
    if n < 2:
        d = x.shape[1] if x.ndim > 1 else 1
        return np.eye(d) * eps
    xc = x - x.mean(axis=0, keepdims=True)
    c = (xc.T @ xc) / (n - 1)
    c = (c + c.T) * 0.5
    c = c + np.eye(c.shape[0]) * eps
    return c


def compute_fid(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    mu1 = x.mean(axis=0)
    mu2 = y.mean(axis=0)

    sigma1 = _cov_with_eps(x, eps)
    sigma2 = _cov_with_eps(y, eps)

    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[sigma2]])

    np.fill_diagonal(sigma1, sigma1.diagonal() + eps)
    np.fill_diagonal(sigma2, sigma2.diagonal() + eps)

    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    mean_diff_term = float(diff @ diff)
    trace_term = float(np.trace(sigma1 + sigma2 - 2.0 * covmean))
    trace_term = max(trace_term, 0.0)

    fid = mean_diff_term + trace_term
    if np.isnan(fid) or np.isinf(fid):
        return float("nan")
    return fid if fid >= 0.0 else 0.0


def compute_energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    xx = cdist(x, x, metric="euclidean")
    yy = cdist(y, y, metric="euclidean")
    xy = cdist(x, y, metric="euclidean")

    exx = xx[np.triu_indices(xx.shape[0], 1)].mean() if x.shape[0] > 1 else 0.0
    eyy = yy[np.triu_indices(yy.shape[0], 1)].mean() if y.shape[0] > 1 else 0.0
    exy = xy.mean()
    return float(2 * exy - exx - eyy)


def compute_metrics(real: np.ndarray, pred: np.ndarray) -> dict:
    real = np.asarray(real, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if real.shape != pred.shape:
        raise ValueError(f"Shape mismatch: real {real.shape} vs pred {pred.shape}")

    return {
        "mmd": compute_mmd(real, pred),
        "energy_distance": compute_energy_distance(real, pred),
        "r2": compute_r2(real, pred),
        "l2_loss": compute_l2(real, pred),
        "fid": compute_fid(real, pred),
    }
