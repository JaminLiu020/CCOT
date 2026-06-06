import torch
import torch.nn.functional as F


def compute_kl_divergence(p, q, epsilon=1e-8, normalize=True):
    """
    计算真实分布P和预测分布Q之间的KL散度：D_KL(P || Q)

    参数:
        p (Tensor): 真实分布，形状为 [n, 977]
        q (Tensor): 预测分布，形状与p相同
        epsilon (float): 数值稳定项，防止log(0)
        normalize (bool): 是否对输入进行softmax归一化

    返回:
        float: 平均KL散度（按样本维度平均）
    """
    if normalize:
        # 将输入转换为概率分布（假设输入为logits）
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

    # 数值稳定处理：确保概率非零
    p = p + epsilon
    q = q + epsilon

    # 计算KL散度：sum(p * (log(p) - log(q)))
    kl = p * (torch.log(p) - torch.log(q))
    kl = kl.sum(dim=-1)  # 对每个样本的特征维度求和

    # 返回所有样本的平均KL散度
    return kl.mean()