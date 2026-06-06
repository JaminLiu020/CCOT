import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import numba
import torch

@numba.njit(parallel=True)
def energy_distance_optimized(X, Y):
    """
    计算两个样本集之间的能量距离（Energy Distance）
    使用numba加速的实现

    参数:
    X: 形状为(n, d)的numpy数组，第一个样本集
    Y: 形状为(m, d)的numpy数组，第二个样本集

    返回:
    能量距离值（标量）
    """
    n = X.shape[0]
    m = Y.shape[0]

    # 计算X内部的平均距离
    XX_sum = 0.0
    for i in numba.prange(n):
        for j in range(i + 1, n):
            XX_sum += np.sqrt(np.sum((X[i] - X[j]) ** 2))
    XX_mean = 2 * XX_sum / (n * (n - 1)) if n > 1 else 0.0

    # 计算Y内部的平均距离
    YY_sum = 0.0
    for i in numba.prange(m):
        for j in range(i + 1, m):
            YY_sum += np.sqrt(np.sum((Y[i] - Y[j]) ** 2))
    YY_mean = 2 * YY_sum / (m * (m - 1)) if m > 1 else 0.0

    # 计算X和Y之间的平均距离
    XY_sum = 0.0
    for i in numba.prange(n):
        for j in range(m):
            XY_sum += np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    XY_mean = XY_sum / (n * m)

    # 计算能量距离
    energy_dist = 2 * XY_mean - XX_mean - YY_mean
    return energy_dist


def energy_distance_scipy(X, Y):
    """
    使用scipy的pdist加速的能量距离计算
    适用于中小规模数据

    参数:
    X: 形状为(n, d)的numpy数组，第一个样本集
    Y: 形状为(m, d)的numpy数组，第二个样本集

    返回:
    能量距离值（标量）
    """
    n = X.shape[0]
    m = Y.shape[0]

    # 合并数据以使用pdist
    combined = np.vstack((X, Y))

    # 计算成对距离
    dists = squareform(pdist(combined, metric='euclidean'))

    # 提取子矩阵
    XX_dists = dists[:n, :n]
    YY_dists = dists[n:, n:]
    XY_dists = dists[:n, n:]

    # 计算平均距离
    XX_mean = np.sum(XX_dists) / (n * (n - 1)) if n > 1 else 0.0
    YY_mean = np.sum(YY_dists) / (m * (m - 1)) if m > 1 else 0.0
    XY_mean = np.sum(XY_dists) / (n * m)

    # 计算能量距离
    energy_dist = 2 * XY_mean - XX_mean - YY_mean
    return energy_dist


def energy_distance_gpu(X, Y, device=None):
    """
    使用GPU加速的能量距离计算
    需要PyTorch库

    参数:
    X: 形状为(n, d)的numpy数组，第一个样本集
    Y: 形状为(m, d)的numpy数组，第二个样本集
    device: torch设备，默认为None（使用可用的GPU或CPU）

    返回:
    能量距离值（标量）
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转换为torch张量
    if X.dtype != torch.float32 or Y.dtype != torch.float32:
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
        Y_torch = torch.tensor(Y, dtype=torch.float32, device=device)
    else:
        X_torch = X
        Y_torch = Y

    n = X_torch.shape[0]
    m = Y_torch.shape[0]

    # 计算X内部的平均距离（使用广播）
    XX_dists = torch.cdist(X_torch, X_torch)
    XX_mean = XX_dists.sum() / (n * (n - 1)) if n > 1 else torch.tensor(0.0, device=device)

    # 计算Y内部的平均距离
    YY_dists = torch.cdist(Y_torch, Y_torch)
    YY_mean = YY_dists.sum() / (m * (m - 1)) if m > 1 else torch.tensor(0.0, device=device)

    # 计算X和Y之间的平均距离
    XY_dists = torch.cdist(X_torch, Y_torch)
    XY_mean = XY_dists.sum() / (n * m)

    # 计算能量距离
    energy_dist = 2 * XY_mean - XX_mean - YY_mean

    return energy_dist.cpu().numpy()


def energy_distance_batched(X, Y, batch_size=1000):
    """
    分批计算能量距离，适用于非常大的数据集
    避免内存溢出

    参数:
    X: 形状为(n, d)的numpy数组，第一个样本集
    Y: 形状为(m, d)的numpy数组，第二个样本集
    batch_size: 每批处理的样本数

    返回:
    能量距离值（标量）
    """
    n = X.shape[0]
    m = Y.shape[0]

    # 计算X内部的平均距离（分批）
    XX_sum = 0.0
    n_batches_X = (n + batch_size - 1) // batch_size
    count_XX = 0

    for i in range(n_batches_X):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        X_batch_i = X[start_i:end_i]

        for j in range(i, n_batches_X):
            start_j = j * batch_size
            end_j = min((j + 1) * batch_size, n)
            X_batch_j = X[start_j:end_j]

            if i == j:
                # 避免计算自身与自身的距离
                dists = pdist(X_batch_i, 'euclidean')
                XX_sum += np.sum(dists)
                count_XX += (end_i - start_i) * (end_i - start_i - 1) // 2
            else:
                dists = cdist(X_batch_i, X_batch_j, 'euclidean')
                XX_sum += np.sum(dists)
                count_XX += (end_i - start_i) * (end_j - start_j)

    XX_mean = XX_sum / count_XX if count_XX > 0 else 0.0

    # 计算Y内部的平均距离（分批）
    YY_sum = 0.0
    n_batches_Y = (m + batch_size - 1) // batch_size
    count_YY = 0

    for i in range(n_batches_Y):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, m)
        Y_batch_i = Y[start_i:end_i]

        for j in range(i, n_batches_Y):
            start_j = j * batch_size
            end_j = min((j + 1) * batch_size, m)
            Y_batch_j = Y[start_j:end_j]

            if i == j:
                # 避免计算自身与自身的距离
                dists = pdist(Y_batch_i, 'euclidean')
                YY_sum += np.sum(dists)
                count_YY += (end_i - start_i) * (end_i - start_i - 1) // 2
            else:
                dists = cdist(Y_batch_i, Y_batch_j, 'euclidean')
                YY_sum += np.sum(dists)
                count_YY += (end_i - start_i) * (end_j - start_j)

    YY_mean = YY_sum / count_YY if count_YY > 0 else 0.0

    # 计算X和Y之间的平均距离（分批）
    XY_sum = 0.0
    for i in range(n_batches_X):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        X_batch = X[start_i:end_i]

        for j in range(n_batches_Y):
            start_j = j * batch_size
            end_j = min((j + 1) * batch_size, m)
            Y_batch = Y[start_j:end_j]

            dists = cdist(X_batch, Y_batch, 'euclidean')
            XY_sum += np.sum(dists)

    XY_mean = XY_sum / (n * m)

    # 计算能量距离
    energy_dist = 2 * XY_mean - XX_mean - YY_mean
    return energy_dist


def choose_best_energy_distance_method(X, Y):
    """
    根据数据大小选择最适合的能量距离计算方法

    参数:
    X: 形状为(n, d)的numpy数组，第一个样本集
    Y: 形状为(m, d)的numpy数组，第二个样本集

    返回:
    能量距离函数
    """
    n, d_x = X.shape
    m, d_y = Y.shape

    if d_x != d_y:
        raise ValueError("X和Y的维度必须相同")

    total_size = n + m
    dimension = d_x

    try:
        import torch
        has_torch = torch.cuda.is_available()
    except ImportError:
        has_torch = False

    try:
        import numba
        has_numba = True
    except ImportError:
        has_numba = False

    # 选择最佳方法
    if has_torch and (total_size > 10000 or dimension > 100):
        print("使用GPU加速计算")
        return energy_distance_gpu
    elif has_numba and total_size <= 10000:
        print("使用Numba加速计算")
        return energy_distance_optimized
    elif total_size <= 5000:
        print("使用SciPy计算")
        return energy_distance_scipy
    else:
        print("使用分批计算")
        return energy_distance_batched


