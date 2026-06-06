#!/usr/bin/env python3
"""
CCOT推理包装器 - 用于快速推理任意新数据

功能：
1. 从checkpoint自动推断并加载预训练的f、g网络
2. 接收 control cells + SMILES → 预测 treated cells
3. 完全冻结参数，仅推理
4. 支持beta、路径等参数覆盖
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 兼容独立运行和作为包导入
try:
    from .networks.picnn import PICNN
    from .networks.embedding import EmbeddingSMILES
except ImportError:
    from networks.picnn import PICNN
    from networks.embedding import EmbeddingSMILES


class CCOTInferenceWrapper:
    """
    CCOT推理包装器
    
    优势：
    1. 自动从checkpoint推断网络架构
    2. 保留SMILES索引优化方案（unique_drug_list过滤）
    3. 完全冻结参数
    4. 支持beta、路径等参数覆盖
    """
    
    def __init__(
        self,
        model_path='assets/CCOT_beta_10.0/best_model.pt',
        embedding_path='assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet',
        unique_drug_list=None,       # 用于过滤嵌入表
        beta=10.0,                   # 可覆盖
        device='cuda:0',
        skip_embedding=False         # 新增：beta=0时可跳过embedding加载
    ):
        """
        初始化推理器
        
        Args:
            model_path: CCOT权重文件路径
            embedding_path: 预训练SMILES嵌入parquet路径
            unique_drug_list: 数据集中包含的所有SMILES去重列表（用于过滤嵌入表）
            beta: Classifier-free guidance参数（默认10.0）
            device: 推理设备
            skip_embedding: 当beta=0时可设为True，跳过embedding加载（无条件推理）
        """
        self.device = device
        self.beta = beta
        self.skip_embedding = skip_embedding
        
        print(f"初始化CCOT推理器...")
        print(f"  - 设备: {device}")
        print(f"  - Beta: {beta}")
        
        # 1. 加载并过滤预训练嵌入（如果需要）
        if not skip_embedding:
            print(f"  - 加载SMILES嵌入: {embedding_path}")
            self.emb_pretrained, self.smiles_to_index = self._load_and_filter_embedding(
                embedding_path, unique_drug_list
            )
            print(f"    ✓ 嵌入表大小: {len(self.smiles_to_index)} 个药物")
        else:
            print(f"  - 跳过embedding加载（beta=0无条件推理模式）")
            self.emb_pretrained = None
            self.smiles_to_index = None
        
        # 2. 从checkpoint自动推断并初始化f、g网络
        print(f"  - 加载模型权重: {model_path}")
        self._load_networks(model_path)
        print(f"    ✓ 模型架构已加载")
        
        # 3. 冻结所有参数
        self._freeze_model()
        print(f"    ✓ 模型参数已冻结")
        print("初始化完成！")
    
    
    def _load_and_filter_embedding(self, embedding_path, unique_drug_list):
        """
        加载并过滤预训练SMILES嵌入（遵循原项目优化方案）
        
        Returns:
            emb_pretrained: torch.Tensor [num_drugs, 32]
            smiles_to_index: Dict[str, int]
        """
        # 读取完整嵌入表
        df = pd.read_parquet(embedding_path)
        
        # 关键优化：用unique_drug_list过滤
        if unique_drug_list is not None:
            df = df.loc[unique_drug_list]
        
        # 构建smiles_to_index（遵循原项目方式）
        smiles_to_index = {smiles: idx for idx, smiles in enumerate(df.index)}
        
        # 转换为张量并标准化
        emb_array = df.to_numpy(dtype=np.float32)
        emb_tensor = torch.tensor(emb_array, device=self.device)
        emb_tensor = emb_tensor / (torch.norm(emb_tensor, dim=1, keepdim=True) + 1e-8)
        emb_tensor.requires_grad = False
        
        return emb_tensor, smiles_to_index
    
    
    def _load_networks(self, model_path):
        """
        从checkpoint加载f、g、emb_encoder网络（自动推断架构）
        """
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从checkpoint自动推断网络架构参数
        # checkpoint结构: {'f_state': ..., 'g_state': ..., 'opt_f_state': ..., 'opt_g_state': ...}
        f_state = checkpoint['f_state']
        g_state = checkpoint['g_state']
        
        # 从state_dict推断架构（以f为例）
        input_dim = f_state['wx.0.weight'].shape[1]  # 基因数（977）
        input_dim_label = f_state['w.0.weight'].shape[1]  # embedding维度（32）
        
        # 推断hidden_units（从wx层的权重维度推断）
        hidden_units = []
        # wx包含多层：wx.0, wx.1, ..., wx.n
        # 每层的输出维度就是hidden_units
        for i in range(len([k for k in f_state.keys() if k.startswith('wx.') and k.endswith('.weight')]) - 1):
            hidden_units.append(f_state[f'wx.{i}.weight'].shape[0])
        
        # 初始化网络（f和g都使用PICNN）
        self.f = PICNN(
            input_dim=input_dim,
            input_dim_label=input_dim_label,
            hidden_units=hidden_units,
            activation='leakyrelu',
            softplus_wz_kernels=False,  # 从config.yaml硬编码
            fnorm_penalty=0,            # f网络不使用
            embedding=False             # 不使用embedding模块（由外部emb_encoder提供）
        ).to(self.device)
        
        self.g = PICNN(
            input_dim=input_dim,
            input_dim_label=input_dim_label,
            hidden_units=hidden_units,
            activation='leakyrelu',
            softplus_wz_kernels=False,
            fnorm_penalty=1,            # g网络使用fnorm_penalty=1
            embedding=False
        ).to(self.device)
        
        # 初始化嵌入编码器（如果需要）
        if self.emb_pretrained is not None:
            self.emb_encoder = EmbeddingSMILES(self.emb_pretrained).to(self.device)
        else:
            self.emb_encoder = None
        
        # 加载权重
        self.f.load_state_dict(f_state)
        self.g.load_state_dict(g_state)
    
    
    def _freeze_model(self):
        """冻结所有参数"""
        self.f.eval()
        self.g.eval()
        for param in self.f.parameters():
            param.requires_grad = False
        for param in self.g.parameters():
            param.requires_grad = False
    
    
    def get_smiles_to_index(self):
        """
        返回smiles_to_index供外部使用（用于构造DataLoader）
        
        Returns:
            smiles_to_index: Dict[str, int] 或 None（如果skip_embedding=True）
        """
        return self.smiles_to_index
    
    def is_unconditional(self):
        """
        检查是否为无条件推理模式（beta=0）
        
        Returns:
            bool: True if beta=0
        """
        return abs(self.beta) < 1e-5
    
    
    def transport(
        self,
        source_data,        # [N, 977] numpy或tensor
        smiles_indices=None,     # [N] 已转为index的列表/tensor (beta=0时可为None)
        beta=None,          # 可选覆盖
        return_numpy=False
    ):
        """
        执行转运映射
        
        Args:
            source_data: 对照组基因表达 [N, 977]
            smiles_indices: 已转为index的药物索引 [N]（beta=0时可为None）
            beta: 覆盖默认beta值（可选）
            return_numpy: 是否返回numpy数组
        
        Returns:
            transported_data: 预测的扰动后基因表达 [N, 977]
        
        Raises:
            ValueError: 当数据维度不匹配时
        """
        # 确定使用的beta值
        beta = beta if beta is not None else self.beta
        
        # 1. 验证输入
        if abs(beta) > 1e-5 and smiles_indices is None:
            raise ValueError("smiles_indices is required when beta != 0")
        
        if smiles_indices is not None and len(smiles_indices) != len(source_data):
            raise ValueError(
                f"Length mismatch: source_data ({len(source_data)}) "
                f"!= smiles_indices ({len(smiles_indices)})"
            )
        
        # 2. 转换为张量
        if not isinstance(source_data, torch.Tensor):
            source_data = torch.tensor(source_data, dtype=torch.float32)
        source_data = source_data.to(self.device)
        
        # 3. 设置requires_grad（g.transport需要）
        source_data.requires_grad_(True)
        
        # 4. 准备条件输入
        if abs(beta) < 1e-5:
            # beta=0: 无条件推理，使用dummy条件向量（不会被使用）
            batch_size = source_data.shape[0]
            emb_condition = torch.zeros(batch_size, self.g.input_dim_label, device=self.device)
        else:
            # beta!=0: 需要真实的条件嵌入
            if smiles_indices is None:
                raise ValueError("smiles_indices cannot be None when beta != 0")
            if not isinstance(smiles_indices, torch.Tensor):
                smiles_indices = torch.tensor(smiles_indices, dtype=torch.long)
            smiles_indices = smiles_indices.to(self.device)
            emb_condition = self.emb_encoder(smiles_indices)
        
        # 5. 执行转运（遵循原始CCOT逻辑，移除transform_matrix）
        
        # 注意：g.transport()内部需要计算梯度，所以不能在no_grad块内
        if abs(beta - 1.0) < 1e-5:
            # 无classifier-free guidance
            transport = self.g.transport(
                source_data, emb_condition, is_conditional_generation=True
            )
        else:
            # 有classifier-free guidance
            transport_cond = self.g.transport(
                source_data, emb_condition, is_conditional_generation=True
            )
            transport_uncond = self.g.transport(
                source_data, emb_condition, is_conditional_generation=False
            )
            transport = beta * transport_cond - (beta - 1) * transport_uncond
        
        # 分离梯度，返回推理结果
        transport = transport.detach()
        
        # 6. 返回结果
        if return_numpy:
            return transport.cpu().numpy()
        return transport
