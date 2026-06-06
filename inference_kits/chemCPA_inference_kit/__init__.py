"""
chemCPA Inference Kit - 独立的 chemCPA 推理模块

将训练好的 chemCPA 模型参数冻结后，用于推理预测。
输入：对照组基因表达数据 + 药物嵌入(或SMILES) + 剂量
输出：扰动后的基因表达预测值（均值）
"""

from .inference import ChemCPAPredictor

__all__ = ["ChemCPAPredictor"]
