from __future__ import annotations

import numpy as np
import torch
from scipy import linalg

try:
    from torchmetrics import R2Score
except Exception:
    R2Score = None


def _compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.float()
    y_pred = y_pred.float()
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    if R2Score is not None:
        metric = R2Score().to(y_true.device)
        metric.update(y_pred, y_true)
        return float(metric.compute().item())

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if torch.abs(ss_tot) < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _compute_l2_distance(target: torch.Tensor, transport: torch.Tensor) -> float:
    return float(torch.dist(torch.mean(target, dim=0), torch.mean(transport, dim=0), p=2).item())


def _compute_scalar_mmd(
    target: np.ndarray,
    transport: np.ndarray,
    gammas=(2.0, 1.0, 0.5, 0.1, 0.01, 0.005),
) -> float:
    x = torch.tensor(target, dtype=torch.float32)
    y = torch.tensor(transport, dtype=torch.float32)

    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    mmd_vals = []
    for g in gammas:
        k_xx = torch.exp(-g * xx)
        k_yy = torch.exp(-g * yy)
        k_xy = torch.exp(-g * xy)
        mmd_vals.append(float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()))
    return float(np.mean(mmd_vals))


def _compute_fid_optimized(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> float:
    """FID aligned with condot's `compute_fid_optimized` numerics.

    Differences from the naive FID recipe:
      - Covariance diagonal receives eps regularization TWICE (once inside the
        unbiased-cov call, once via np.fill_diagonal) to match condot exactly.
      - Trace term is clamped to >= 0 to avoid tiny negative values that arise
        when sqrtm produces a slightly-off matrix square root.
      - Final FID is clamped to >= 0.
    These only matter when predicted covariance is near-singular (e.g. biolord
    underestimates per-gene variance); on well-conditioned data the two
    formulations agree to ~1e-5.
    """
    real = y_true.detach().cpu().numpy().astype(np.float64)
    fake = y_pred.detach().cpu().numpy().astype(np.float64)

    mu1 = np.mean(real, axis=0)
    mu2 = np.mean(fake, axis=0)

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

    sigma1 = _cov_with_eps(real, eps)
    sigma2 = _cov_with_eps(fake, eps)

    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[sigma2]])

    np.fill_diagonal(sigma1, sigma1.diagonal() + eps)
    np.fill_diagonal(sigma2, sigma2.diagonal() + eps)

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

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


def _energy_distance_gpu(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    n = x.shape[0]
    m = y.shape[0]

    d_xy = torch.cdist(x, y, p=2).sum() / (n * m)
    if n > 1:
        d_xx = torch.cdist(x, x, p=2).sum() / (n * (n - 1))
    else:
        d_xx = torch.tensor(0.0, device=x.device)
    if m > 1:
        d_yy = torch.cdist(y, y, p=2).sum() / (m * (m - 1))
    else:
        d_yy = torch.tensor(0.0, device=y.device)

    ed = 2.0 * d_xy - d_xx - d_yy
    return float(ed.item())


def compute_metrics_bundle(real: torch.Tensor, pred: torch.Tensor) -> dict:
    real = real.float()
    pred = pred.float()

    mmd = _compute_scalar_mmd(real.detach().cpu().numpy(), pred.detach().cpu().numpy())
    l2 = _compute_l2_distance(real.detach(), pred.detach())
    fid = _compute_fid_optimized(real.detach(), pred.detach())
    r2 = _compute_r2(real.detach().mean(dim=0), pred.detach().mean(dim=0))
    ed = _energy_distance_gpu(real.detach(), pred.detach())

    return {
        "mmd": float(mmd),
        "l2_loss": l2,
        "fid": fid,
        "r2": r2,
        "energy_distance": ed,
    }
