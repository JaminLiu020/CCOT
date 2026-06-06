"""
药物嵌入加载工具。
从 parquet 文件加载预计算的 rdkit2D 化学表征。
"""

from pathlib import Path
from typing import List

import pandas as pd
import torch


def load_drug_embedding(
    smiles: List[str],
    embedding_path: str,
    device: str = "cpu",
) -> torch.nn.Embedding:
    """
    从 parquet 文件加载药物嵌入。

    Args:
        smiles: SMILES 字符串列表（按排序顺序传入，与训练时一致）
        embedding_path: parquet 文件路径
        device: 设备

    Returns:
        torch.nn.Embedding: 冻结的药物嵌入层
    """
    embedding_path = Path(embedding_path)
    assert embedding_path.exists(), f"Embedding file not found: {embedding_path}"

    df = pd.read_parquet(embedding_path)

    # 确保所有 SMILES 都在嵌入中
    missing = set(smiles) - set(df.index)
    if missing:
        raise ValueError(f"以下 SMILES 在嵌入文件中未找到 ({len(missing)} 个): {list(missing)[:5]}...")

    emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
    assert emb.shape[0] == len(smiles)
    return torch.nn.Embedding.from_pretrained(emb, freeze=True)


def get_embedding_dim(embedding_path: str) -> int:
    """获取嵌入维度"""
    df = pd.read_parquet(embedding_path)
    return df.shape[1]


def get_all_smiles(embedding_path: str) -> List[str]:
    """获取嵌入文件中所有可用的 SMILES"""
    df = pd.read_parquet(embedding_path)
    return df.index.tolist()
