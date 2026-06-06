#!/usr/bin/python3

# imports
import pandas as pd
import torch
from sklearn import preprocessing
# import torch.nn.init as init
import torch.nn as nn
# from chemCPA.model import MLP

class EmbeddingSMACOF():
    """Queries precomputed SMACOF embedding of conditions."""

    def __init__(self, path, **kwargs):
        self.embedding = self.setup(path)

    def setup(self, emb_config):
        # load pickled embedding
        return pd.read_hdf(emb_config.path, f'smacof_{emb_config.dim}d')

    def forward(self, y):
        return torch.Tensor(self.embedding[y].values)


class EmbeddingFingerprint():
    """Queries precomputed molecular fingerprint of conditions."""

    def __init__(self, path, **kwargs):
        self.embedding = self.setup(path)

    def setup(self, emb_config):
        # load precomputed fingerprints
        embedding = pd.read_csv(
          emb_config.path,
          converters={f"{emb_config.method} Fingerprint": lambda x: x.strip(
              "[]").split(", ")})
        return dict(zip(embedding.Name,
                        embedding[f"{emb_config.method} Fingerprint"]))

    def forward(self, y):
        return torch.Tensor(list(map(float, self.embedding[y])))


class EmbeddingOneHot(nn.Module):
    """Returns one-hot of conditions."""

    def __init__(self, labels, **kwargs):
        super(EmbeddingOneHot, self).__init__()
        self.setup(labels)

    def setup(self, labels):
        multi_labels = [i.split("+") for i in labels]
        self.lb = preprocessing.MultiLabelBinarizer()
        self.lb.fit(multi_labels)

    def forward(self, y):
        return torch.Tensor(self.lb.transform([y.split("+")])[0])

class EmbeddingIdx2OneHot(nn.Module):
    """Returns one-hot of conditions."""

    def __init__(self, labels, **kwargs):
        super(EmbeddingIdx2OneHot, self).__init__()
        self.setup(labels)

    def setup(self, labels):
        self.lb = preprocessing.MultiLabelBinarizer()
        self.lb.fit(labels)

    def forward(self, y):
        return torch.Tensor(self.lb.transform(y))


class EmbeddingValue(nn.Module):
    """Returns value to use as condition."""
    def __init__(self, factor):
        super(EmbeddingValue, self).__init__()
        self.factor = factor

    def forward(self, y):
        dose = y / self.factor
        return torch.Tensor(dose)

class EmbeddingSMILES(nn.Module):
    # def __init__(self, hparams, emb_pretrained):
    def __init__(self, emb_pretrained):
        super(EmbeddingSMILES, self).__init__()  # 调用父类的初始化方法
        # self.hparams = hparams

        # self.emb_pretrained = torch.nn.Embedding.from_pretrained(emb_pretrained, freeze=True)
        # self.emb_pretrained = emb_pretrained

        # 注册为buffer避免重复拷贝
        self.register_buffer("emb_pretrained", emb_pretrained)

        # self.drug_embedding_encoder = MLP(
        #     [self.emb_pretrained.shape[1]]
        #     + [self.hparams["embedding_encoder_width"]]
        #     * self.hparams["embedding_encoder_depth"]
        #     + [self.hparams["dim"]],
        #     last_layer_act="linear",
        # )
        #
        # self.dosers = MLP(
        #     [self.emb_pretrained.shape[1] + 1]
        #     + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
        #     + [1],
        # )
        #
        # # 权重初始化
        # self.dosers.apply(kaiming_init)
        # self.drug_embedding_encoder.apply(kaiming_init)

    def to(self, device):
        # 将所有模型组件移到指定设备
        self.emb_pretrained = self.emb_pretrained.to(device)
        # self.drug_embedding_encoder = self.drug_embedding_encoder.to(device)
        # self.dosers = self.dosers.to(device)
        return self  # 返回自身，以便链式调用

    def forward(self, indexes):
        # drug_latent = self.emb_pretrained(indexes)
        # indexes = indexes
        # drug_latent = torch.index_select(self.emb_pretrained, 0, indexes)

        # 对输入进行标准化
        #drug_latent = drug_latent / (torch.norm(drug_latent, dim=1, keepdim=True) + 1e-8)

        # dosage_emb = self.dosers(torch.concat([drug_latent, torch.unsqueeze(dosages, dim=-1)], dim=1)).squeeze()
        # drug_emb = self.drug_embedding_encoder(drug_latent)
        # drug_emb = drug_latent

        # return torch.einsum("b,be->be", [dosage_emb, drug_emb])
        # return drug_latent
        return self.emb_pretrained[indexes]
#
# def xavier_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#
# def kaiming_init(module):
#     """
#     Kaiming Initialization (He Initialization)
#     """
#     if isinstance(module, torch.nn.Linear):
#         init.kaiming_uniform_(module.weight, nonlinearity='relu')  # 使用 Kaiming 初始化
#         if module.bias is not None:
#             init.zeros_(module.bias)  # 偏置初始化为零
#
#     elif isinstance(module, torch.nn.BatchNorm1d):
#         init.ones_(module.weight)  # BatchNorm 权重初始化为 1
#         init.zeros_(module.bias)  # BatchNorm 偏置初始化为 0


# class EmbeddingSMILES(nn.Module):
#     def __init__(self, emb_pretrained, padding_idx=None, cache_size=None):
#         """
#         优化的嵌入查询类
#
#         Args:
#             emb_pretrained: 预训练的嵌入向量矩阵，形状为(vocab_size, embed_dim)
#             padding_idx: 可选的填充索引，如果提供则在该位置初始化为零向量
#             cache_size: LRU缓存大小，用于存储最近访问的嵌入向量
#         """
#         super(EmbeddingSMILES, self).__init__()
#
#         # 使用register_buffer注册嵌入矩阵，避免被视为模型参数
#         self.register_buffer("emb_pretrained", emb_pretrained)
#
#         # 记录嵌入维度信息
#         self.vocab_size, self.embedding_dim = emb_pretrained.shape
#
#         # 如果提供了padding_idx，将该位置的嵌入向量初始化为零
#         if padding_idx is not None:
#             with torch.no_grad():
#                 self.emb_pretrained[padding_idx].fill_(0)
#
#         # 使用LRU缓存存储常用查询结果
#         if cache_size is None:
#             cache_size = self.vocab_size
#         self.use_cache = cache_size > 0
#         if self.use_cache:
#             self.cache = {}
#             self.cache_size = cache_size
#             self.cache_hits = 0
#             self.cache_misses = 0
#             self.cache_keys_queue = []
#
#     def clear_cache_stats(self):
#         """重置缓存统计信息"""
#         self.cache_hits = 0
#         self.cache_misses = 0
#
#     def get_cache_stats(self):
#         """获取缓存命中率统计信息"""
#         total = self.cache_hits + self.cache_misses
#         hit_rate = self.cache_hits / total if total > 0 else 0
#         return {
#             "hits": self.cache_hits,
#             "misses": self.cache_misses,
#             "total": total,
#             "hit_rate": hit_rate
#         }
#
#     def to(self, device):
#         """将模块移动到指定设备上"""
#         # 调用父类的to方法
#         super(EmbeddingSMILES, self).to(device)
#
#         # 确保嵌入矩阵也移到正确的设备
#         if hasattr(self, "emb_pretrained"):
#             self.emb_pretrained = self.emb_pretrained.to(device)
#
#         # 清空缓存，因为缓存中的张量可能在不同设备上
#         if self.use_cache:
#             self.cache = {}
#             self.cache_keys_queue = []
#
#         return self
#
#     def forward(self, indexes):
#         """
#         查询嵌入向量
#
#         Args:
#             indexes: 索引张量，可以是单个索引、一维或多维张量
#
#         Returns:
#             查询到的嵌入向量
#         """
#         # 如果indexes是标量，转换为张量
#         if not isinstance(indexes, torch.Tensor):
#             indexes = torch.tensor(indexes, device=self.emb_pretrained.device)
#
#         # 优化：检测是否有重复索引，提取唯一索引以减少查询次数
#         original_shape = indexes.shape
#         indexes_flat = indexes.reshape(-1)
#
#         # 使用缓存的情况
#         if self.use_cache:
#             # 转换为可哈希类型作为缓存键
#             cache_key = tuple(indexes_flat.cpu().numpy().tolist())
#
#             # 检查是否在缓存中
#             if cache_key in self.cache:
#                 self.cache_hits += 1
#                 return self.cache[cache_key].reshape(*original_shape, -1)
#
#             self.cache_misses += 1
#
#         # 优化：对于大批量查询，使用唯一索引查询再重组
#         # 仅当有大量索引且存在重复时有效
#         if indexes_flat.numel() > 100:
#             unique_indexes, inverse_indices = torch.unique(indexes_flat, return_inverse=True)
#
#             # 如果唯一索引数量明显少于总索引数，使用唯一索引查询再重组
#             if len(unique_indexes) < len(indexes_flat) * 0.8:
#                 # 查询唯一索引的嵌入向量
#                 unique_embeddings = self.emb_pretrained[unique_indexes]
#
#                 # 使用inverse_indices重组原始形状
#                 result = unique_embeddings[inverse_indices].reshape(*original_shape, -1)
#
#                 # 更新缓存
#                 if self.use_cache:
#                     self._update_cache(cache_key, result)
#
#                 return result
#
#         # 常规查询：直接使用索引获取嵌入向量
#         result = self.emb_pretrained[indexes]
#
#         # 更新缓存
#         if self.use_cache:
#             self._update_cache(cache_key, result)
#
#         return result
#
#     def _update_cache(self, key, value):
#         """更新LRU缓存"""
#         # 如果缓存已满，移除最老的条目
#         if len(self.cache) >= self.cache_size:
#             oldest_key = self.cache_keys_queue.pop(0)
#             self.cache.pop(oldest_key)
#
#         # 添加新条目到缓存
#         self.cache[key] = value
#         self.cache_keys_queue.append(key)
#
#         # 确保队列长度不超过缓存大小
#         while len(self.cache_keys_queue) > self.cache_size:
#             self.cache_keys_queue.pop(0)
