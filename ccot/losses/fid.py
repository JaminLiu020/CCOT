import torch
import numpy as np
from scipy.linalg import sqrtm
# from torch.linalg import matrix_norm

#
# def calculate_fid_optimized(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """优化后的FID计算函数
#
#     Args:
#         mu1, mu2: 两个分布的均值向量
#         sigma1, sigma2: 两个分布的协方差矩阵
#         eps: 添加到协方差矩阵对角线的小值，用于数值稳定性
#
#     Returns:
#         fid: 计算得到的FID值
#     """
#     # 直接在GPU上计算均值差的平方和，避免不必要的CPU转移
#     if mu1.is_cuda and mu2.is_cuda:
#         diff = mu1 - mu2
#         mean_diff_term = torch.sum(diff * diff).item()
#     else:
#         diff = mu1 - mu2
#         mean_diff_term = diff.dot(diff).item()
#
#     # 检查是否可以直接在GPU上进行后续计算
#     if torch.cuda.is_available() and sigma1.is_cuda and sigma2.is_cuda:
#         try:
#             # 直接在GPU上添加对角线正则化
#             identity = torch.eye(sigma1.shape[0], device=sigma1.device) * eps
#             sigma1_reg = sigma1 + identity
#             sigma2_reg = sigma2 + identity
#
#             # 尝试使用PyTorch的矩阵平方根计算（如果可用）
#             product = torch.mm(sigma1_reg, sigma2_reg)
#
#             # 对于较小的矩阵，将数据移回CPU并使用scipy计算可能更快
#             if product.shape[0] <= 512:  # 阈值可以根据实际情况调整
#                 cov_sqrt = torch.tensor(
#                     sqrtm(product.cpu().detach().numpy()),
#                     device=sigma1.device
#                 )
#             else:
#                 # 对于大型矩阵，使用特征值分解在GPU上计算平方根
#                 # 注意：这是近似解，但对于FID计算通常足够
#                 eigenvalues, eigenvectors = torch.linalg.eigh(product)
#                 eigenvalues = torch.clamp(eigenvalues, min=0)  # 避免数值不稳定
#                 sqrt_eigenvalues = torch.sqrt(eigenvalues)
#                 cov_sqrt = torch.mm(
#                     torch.mm(eigenvectors, torch.diag(sqrt_eigenvalues)),
#                     eigenvectors.transpose(-2, -1)
#                 )
#
#             # 计算trace项
#             trace_term = torch.trace(sigma1_reg + sigma2_reg - 2 * cov_sqrt).item()
#
#             return mean_diff_term + trace_term
#
#         except Exception as e:
#             print(f"GPU计算失败，回退到CPU: {e}")
#             # 如果GPU计算失败，回退到CPU版本
#             pass
#
#     # CPU版本计算（优化后）
#     cov1 = sigma1.cpu().detach().numpy()
#     cov2 = sigma2.cpu().detach().numpy()
#
#     # 快速检查是否有NaN或Inf
#     if not (np.isfinite(cov1).all() and np.isfinite(cov2).all()):
#         print("检测到非有限值，返回NaN")
#         return float('nan')
#
#     # 添加正则化
#     cov1 += np.eye(cov1.shape[0]) * eps
#     cov2 += np.eye(cov2.shape[0]) * eps
#
#     # 使用scipy的sqrtm函数
#     sqrt_product = sqrtm(cov1 @ cov2)
#
#     # 如果结果包含虚部（由于数值不稳定），取实部
#     if np.iscomplexobj(sqrt_product):
#         if np.max(np.abs(sqrt_product.imag)) > 1e-3:
#             print(f"警告：平方根计算存在较大虚部: {np.max(np.abs(sqrt_product.imag))}")
#         sqrt_product = sqrt_product.real
#
#     # 计算trace项
#     trace_term = np.trace(cov1 + cov2 - 2 * sqrt_product)
#
#     # 返回最终结果
#     return mean_diff_term + trace_term
#
#
# def compute_fid_optimized(y_true, y_pred, batch_size=None):
#     """优化的FID计算函数，支持批处理模式
#
#     Args:
#         y_true: 真实数据分布
#         y_pred: 预测数据分布
#         batch_size: 批处理大小，用于大型数据集
#
#     Returns:
#         fid: 计算得到的FID值
#     """
#     # # 对于大数据集，使用批处理计算均值和协方差
#     # if batch_size is not None and (len(y_true) > batch_size or len(y_pred) > batch_size):
#     #     # 计算均值 - 使用批处理以减少内存使用
#     #     mu1 = torch.zeros(y_true.shape[1], device=y_true.device)
#     #     mu2 = torch.zeros(y_pred.shape[1], device=y_pred.device)
#     #
#     #     # 批处理计算均值
#     #     for i in range(0, len(y_true), batch_size):
#     #         batch = y_true[i:i + batch_size]
#     #         mu1 += torch.sum(batch, dim=0)
#     #     mu1 /= len(y_true)
#     #
#     #     for i in range(0, len(y_pred), batch_size):
#     #         batch = y_pred[i:i + batch_size]
#     #         mu2 += torch.sum(batch, dim=0)
#     #     mu2 /= len(y_pred)
#     #
#     #     # 计算协方差 - 批处理
#     #     sigma1 = torch.zeros((y_true.shape[1], y_true.shape[1]), device=y_true.device)
#     #     for i in range(0, len(y_true), batch_size):
#     #         batch = y_true[i:i + batch_size]
#     #         batch_centered = batch - mu1
#     #         sigma1 += torch.mm(batch_centered.T, batch_centered)
#     #     sigma1 /= (len(y_true) - 1)
#     #
#     #     sigma2 = torch.zeros((y_pred.shape[1], y_pred.shape[1]), device=y_pred.device)
#     #     for i in range(0, len(y_pred), batch_size):
#     #         batch = y_pred[i:i + batch_size]
#     #         batch_centered = batch - mu2
#     #         sigma2 += torch.mm(batch_centered.T, batch_centered)
#     #     sigma2 /= (len(y_pred) - 1)
#     # else:
#     # 对于小型数据集，直接使用PyTorch内置函数
#     mu1 = torch.mean(y_true, dim=0)
#     mu2 = torch.mean(y_pred, dim=0)
#     sigma1 = torch.cov(y_true.T)  # PyTorch 需要转置来获得正确的协方差
#     sigma2 = torch.cov(y_pred.T)
#
#     # 计算FID
#     return calculate_fid_optimized(mu1, sigma1, mu2, sigma2)

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