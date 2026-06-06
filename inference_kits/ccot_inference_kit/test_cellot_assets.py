#!/usr/bin/env python3
"""
快速测试：验证CellOT模型（beta=0）可以从assets加载
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from ccot_wrapper import CCOTInferenceWrapper


def test_cellot_from_assets():
    """测试从assets/CellOT加载模型"""
    
    print("=" * 80)
    print("测试从assets/CellOT加载beta=0模型")
    print("=" * 80)
    
    # 使用assets中的模型
    print("\n【步骤1】从assets/CellOT加载模型")
    wrapper = CCOTInferenceWrapper(
        model_path='assets/CellOT/best_model.pt',
        beta=0.0,
        device='cuda:0',
        skip_embedding=True
    )
    
    print(f"\n【步骤2】模型信息")
    print(f"  - 无条件模式: {wrapper.is_unconditional()}")
    print(f"  - input_dim: {wrapper.g.input_dim}")
    print(f"  - input_dim_label: {wrapper.g.input_dim_label}")
    
    # 创建测试数据
    print(f"\n【步骤3】执行推理测试")
    batch_size = 5
    n_genes = 977
    
    source_data = np.random.randn(batch_size, n_genes).astype(np.float32)
    
    transported = wrapper.transport(
        source_data=source_data,
        smiles_indices=None,  # beta=0不需要
        return_numpy=True
    )
    
    print(f"  ✓ 推理成功")
    print(f"  - 输入形状: {source_data.shape}")
    print(f"  - 输出形状: {transported.shape}")
    print(f"  - 输出统计: mean={transported.mean():.4f}, std={transported.std():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ 测试通过！CellOT模型已成功部署到assets/")
    print("=" * 80)


def show_assets_structure():
    """显示assets目录结构"""
    
    print("\n" + "=" * 80)
    print("Assets目录结构")
    print("=" * 80)
    
    import os
    
    assets_dir = Path(__file__).parent / 'assets'
    
    print(f"\n{assets_dir}/")
    for item in sorted(assets_dir.iterdir()):
        if item.is_dir():
            print(f"├── {item.name}/")
            for subitem in sorted(item.iterdir()):
                is_last = subitem == sorted(item.iterdir())[-1]
                prefix = "└──" if is_last else "├──"
                size = ""
                if subitem.is_file():
                    size_mb = subitem.stat().st_size / (1024 * 1024)
                    size = f" ({size_mb:.1f}MB)" if size_mb > 1 else f" ({subitem.stat().st_size}B)"
                print(f"│   {prefix} {subitem.name}{size}")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            size = f" ({size_mb:.1f}MB)" if size_mb > 1 else ""
            print(f"├── {item.name}{size}")
    
    print("\n可用模型:")
    print("  1. assets/CCOT_beta_10.0/best_model.pt - 条件模型（SMILES, beta=10.0）")
    print("  2. assets/CellOT/best_model.pt         - 无条件模型（cell_type, beta=0.0）")


if __name__ == "__main__":
    show_assets_structure()
    test_cellot_from_assets()
