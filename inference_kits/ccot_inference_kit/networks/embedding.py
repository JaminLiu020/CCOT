#!/usr/bin/python3
"""
精简版 EmbeddingSMILES - 仅用于CCOT推理

用途：通过索引快速查询预训练SMILES嵌入
"""

import torch
import torch.nn as nn


class EmbeddingSMILES(nn.Module):
    """
    SMILES嵌入查询类（精简版）
    
    用途：通过索引快速查询预训练SMILES嵌入
    """
    
    def __init__(self, emb_pretrained):
        """
        Args:
            emb_pretrained: 预训练嵌入张量 [num_drugs, embed_dim]
        """
        super(EmbeddingSMILES, self).__init__()
        
        # 注册为buffer避免重复拷贝
        self.register_buffer("emb_pretrained", emb_pretrained)

    def to(self, device):
        """将模型移到指定设备"""
        self.emb_pretrained = self.emb_pretrained.to(device)
        return self

    def forward(self, indexes):
        """
        通过索引查询嵌入向量
        
        Args:
            indexes: 药物索引张量 [batch_size] 或 [batch_size, ...]
        
        Returns:
            嵌入向量 [batch_size, embed_dim]
        """
        return self.emb_pretrained[indexes]
