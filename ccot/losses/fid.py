import torch
import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """计算 FID 值"""
    # 计算均值之间的平方差
    diff = mu1 - mu2
    # 将张量转换为 NumPy 数组
    cov1 = sigma1.cpu().detach().numpy()
    cov2 = sigma2.cpu().detach().numpy()

    # 检查 cov1 和 cov2 是否包含 NaN 或 inf
    if np.any(np.isnan(cov1)) or np.any(np.isnan(cov2)):
        print("NaN detected in covariance matrix.")
        return float('nan')
    if np.any(np.isinf(cov1)) or np.any(np.isinf(cov2)):
        print("Inf detected in covariance matrix.")
        return float('inf')

    # 正则化，保持矩阵正定
    cov1 += np.eye(cov1.shape[0]) * 1e-4  # 给协方差矩阵添加微小对角噪声
    cov2 += np.eye(cov2.shape[0]) * 1e-4

    try:
        # 计算协方差的平方根
        cov_sqrt = sqrtm(cov1 @ cov2)

        # 如果协方差矩阵不正定，进行调整
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real

        # 计算 FID
        fid = diff.dot(diff) + np.trace(cov1 + cov2 - 2 * cov_sqrt)
        return fid  # 返回标量值

    except ValueError as e:
        print(f"Error in sqrtm calculation: {e}")
        return float('nan')

def compute_fid(y_true, y_pred):
    """计算两个分布的 FID"""
    mu1, sigma1 = torch.mean(y_true, dim=0), torch.cov(y_true.T)
    mu2, sigma2 = torch.mean(y_pred, dim=0), torch.cov(y_pred.T)

    fid = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid


def compute_fid_optimized(y_true, y_pred, eps=1e-6):
    """优化的FID计算函数，支持数值稳定性"""
    # 确保输入是浮点类型（避免整数计算）
    y_true = y_true.float()
    y_pred = y_pred.float()

    # 计算均值和协方差（修正协方差计算）
    mu1 = torch.mean(y_true, dim=0)
    mu2 = torch.mean(y_pred, dim=0)
    sigma1 = _covariance(y_true, eps)  # 自定义协方差计算函数
    sigma2 = _covariance(y_pred, eps)

    return calculate_fid_optimized(mu1, sigma1, mu2, sigma2, eps)


def _covariance(x, eps):
    """数值稳定的协方差计算"""
    n = x.shape[0]
    x_centered = x - torch.mean(x, dim=0)
    cov = (x_centered.T @ x_centered) / (n - 1)  # 无偏估计
    cov = (cov + cov.T) * 0.5  # 强制对称
    cov += torch.eye(cov.shape[0], device=x.device) * eps  # 正则化
    return cov


def calculate_fid_optimized(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """优化后的FID计算核心逻辑"""
    # 计算均值差项（数值稳定）
    diff = mu1 - mu2
    mean_diff_term = torch.sum(diff ** 2)

    # 始终在CPU上使用SciPy的sqrtm（更稳定）
    sigma1_np = sigma1.cpu().detach().numpy().astype(np.float64)
    sigma2_np = sigma2.cpu().detach().numpy().astype(np.float64)

    # 正则化协方差矩阵
    np.fill_diagonal(sigma1_np, sigma1_np.diagonal() + eps)
    np.fill_diagonal(sigma2_np, sigma2_np.diagonal() + eps)

    # 计算矩阵平方根（强制实部）
    sqrt_product = sqrtm(sigma1_np @ sigma2_np)
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    # 计算Trace项（避免负值）
    trace_term = np.trace(sigma1_np + sigma2_np - 2 * sqrt_product)
    trace_term = max(trace_term, 0)  # 强制非负

    fid = mean_diff_term.item() + trace_term
    return fid if fid >= 0 else 0.0  # 最终结果非负