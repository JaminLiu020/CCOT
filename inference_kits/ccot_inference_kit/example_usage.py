#!/usr/bin/env python3
"""
CCOT推理包装器使用示例

演示如何使用CCOTInferenceWrapper进行药物扰动效应预测
"""

import sys
from pathlib import Path

# 添加到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from ccot_inference_kit.ccot_wrapper import CCOTInferenceWrapper


def example1_basic_inference():
    """
    场景1: 基本推理（提供unique_drug_list）
    """
    print("=" * 60)
    print("场景1: 基本推理")
    print("=" * 60)
    
    # 假设你的数据集包含这些药物（从CSV文件读取并去重）
    # 例如：df = pd.read_csv('your_data.csv')
    #      unique_drugs = df['SMILES'].drop_duplicates().tolist()
    unique_drugs = ['CC(C)O', 'CCO', 'C1=CC=CC=C1']  # 示例药物
    
    # 获取ccot_inference_kit包所在目录
    kit_dir = Path(__file__).parent
    
    # 初始化CCOT
    model = CCOTInferenceWrapper(
        model_path=str(kit_dir / 'assets/CCOT_beta_10.0/best_model.pt'),
        embedding_path=str(kit_dir / 'assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet'),
        unique_drug_list=unique_drugs,  # 过滤嵌入表
        beta=10.0,
        device='cuda:0'
    )
    
    # 获取smiles_to_index供DataLoader使用
    smiles_to_index = model.get_smiles_to_index()
    print(f"\n过滤后的嵌入表大小: {len(smiles_to_index)} 个药物")
    print(f"SMILES索引映射: {smiles_to_index}")
    
    # 准备推理数据（假设已从DataLoader获取）
    # 注意：在实际使用中，source_data来自你的对照组数据
    control_cells = np.random.randn(100, 977)  # [N, 977] 对照组基因表达
    drug_indices = [smiles_to_index['CC(C)O']] * 100  # 已转为index
    
    # 执行推理
    print("\n执行推理...")
    predicted_treated = model.transport(control_cells, drug_indices)
    print(f"预测结果形状: {predicted_treated.shape}")  # (100, 977)
    print(f"预测值范围: [{predicted_treated.min():.3f}, {predicted_treated.max():.3f}]")
    
    return model


def example2_override_beta():
    """
    场景2: 覆盖beta参数
    """
    print("\n" + "=" * 60)
    print("场景2: 覆盖beta参数")
    print("=" * 60)
    
    # 使用场景1的模型
    unique_drugs = ['CC(C)O', 'CCO', 'C1=CC=CC=C1']
    kit_dir = Path(__file__).parent
    model = CCOTInferenceWrapper(
        model_path=str(kit_dir / 'assets/CCOT_beta_10.0/best_model.pt'),
        embedding_path=str(kit_dir / 'assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet'),
        unique_drug_list=unique_drugs,
        beta=10.0,  # 默认值
        device='cuda:0'
    )
    
    smiles_to_index = model.get_smiles_to_index()
    control_cells = np.random.randn(50, 977)
    drug_indices = [smiles_to_index['CCO']] * 50
    
    # 使用默认beta=10.0
    print("\n使用默认beta=10.0:")
    result_beta10 = model.transport(control_cells, drug_indices)
    print(f"  预测值范围: [{result_beta10.min():.3f}, {result_beta10.max():.3f}]")
    
    # 覆盖为beta=5.0
    print("\n覆盖为beta=5.0:")
    result_beta5 = model.transport(control_cells, drug_indices, beta=5.0)
    print(f"  预测值范围: [{result_beta5.min():.3f}, {result_beta5.max():.3f}]")
    
    # 无classifier-free guidance (beta=1.0)
    print("\n无guidance (beta=1.0):")
    result_beta1 = model.transport(control_cells, drug_indices, beta=1.0)
    print(f"  预测值范围: [{result_beta1.min():.3f}, {result_beta1.max():.3f}]")


def example3_batch_inference():
    """
    场景3: 批量推理（不同药物）
    """
    print("\n" + "=" * 60)
    print("场景3: 批量推理（多个药物）")
    print("=" * 60)
    
    unique_drugs = ['CC(C)O', 'CCO', 'C1=CC=CC=C1']
    kit_dir = Path(__file__).parent
    model = CCOTInferenceWrapper(
        model_path=str(kit_dir / 'assets/CCOT_beta_10.0/best_model.pt'),
        embedding_path=str(kit_dir / 'assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet'),
        unique_drug_list=unique_drugs,
        beta=10.0,
        device='cuda:0'
    )
    
    smiles_to_index = model.get_smiles_to_index()
    
    # 不同药物作用于不同细胞
    batch_smiles = ['CC(C)O', 'CCO', 'C1=CC=CC=C1']
    batch_indices = [smiles_to_index[s] for s in batch_smiles]
    batch_controls = np.random.randn(3, 977)
    
    print(f"\n批量推理 {len(batch_smiles)} 个药物:")
    batch_results = model.transport(batch_controls, batch_indices)
    print(f"  结果形状: {batch_results.shape}")
    
    for i, smiles in enumerate(batch_smiles):
        print(f"  药物 {i+1} ({smiles}): 预测值范围 [{batch_results[i].min():.3f}, {batch_results[i].max():.3f}]")


def example4_error_handling():
    """
    场景4: 错误处理示例
    """
    print("\n" + "=" * 60)
    print("场景4: 错误处理")
    print("=" * 60)
    
    unique_drugs = ['CC(C)O', 'CCO']
    kit_dir = Path(__file__).parent
    model = CCOTInferenceWrapper(
        model_path=str(kit_dir / 'assets/CCOT_beta_10.0/best_model.pt'),
        embedding_path=str(kit_dir / 'assets/rdkit2D_embedding_lincs_trapnell_chemCPA.parquet'),
        unique_drug_list=unique_drugs,
        beta=10.0,
        device='cuda:0'
    )
    
    smiles_to_index = model.get_smiles_to_index()
    
    # 错误1: 长度不匹配
    print("\n测试错误1: source_data和smiles_indices长度不匹配")
    try:
        control_cells = np.random.randn(10, 977)
        drug_indices = [smiles_to_index['CC(C)O']] * 5  # 只有5个
        model.transport(control_cells, drug_indices)
    except ValueError as e:
        print(f"  ✓ 成功捕获错误: {e}")
    
    # 错误2: SMILES不在嵌入表中
    print("\n测试错误2: SMILES不在嵌入表中")
    try:
        # 尝试使用一个不在unique_drugs中的药物
        invalid_index = 999  # 假设这个索引不存在
        control_cells = np.random.randn(1, 977)
        model.transport(control_cells, [invalid_index])
    except (KeyError, IndexError) as e:
        print(f"  ✓ 成功捕获错误: {type(e).__name__}")


if __name__ == '__main__':
    # 运行所有示例
    print("\n" + "=" * 60)
    print("CCOT推理包装器使用示例")
    print("=" * 60)
    
    # 示例1: 基本推理
    model = example1_basic_inference()
    
    # 示例2: 覆盖beta参数
    example2_override_beta()
    
    # 示例3: 批量推理
    example3_batch_inference()
    
    # 示例4: 错误处理
    example4_error_handling()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
