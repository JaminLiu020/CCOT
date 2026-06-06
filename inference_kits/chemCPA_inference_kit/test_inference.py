"""
测试脚本：验证 ChemCPAPredictor 的加载和推理功能。

运行方法:
    cd inference_kits   (chemCPA_inference_kit 的父目录)
    python chemCPA_inference_kit/test_inference.py
"""

import sys
import torch
import numpy as np
from pathlib import Path


def _setup():
    """公共初始化：路径检查 + 加载 predictor"""
    model_path = Path(__file__).parent / "pretrained" / "manual_2025-03-08_03-24-40.pt"
    embedding_path = Path(__file__).parent / "pretrained" / "rdkit2D_embedding_lincs_trapnell.parquet"

    if not model_path.exists() or not embedding_path.exists():
        print("[跳过] 模型或嵌入文件不存在，请先运行 setup_pretrained.py")
        return None

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chemCPA_inference_kit import ChemCPAPredictor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = ChemCPAPredictor.from_pretrained(
        model_path=str(model_path),
        embedding_path=str(embedding_path),
        device=device,
    )
    return predictor


def test_load_and_predict():
    """测试 1: 加载模型 + 通过 drug_idx 基本推理"""
    print("=" * 60)
    print("测试 1: 加载模型 + 通过 drug_idx 基本推理")
    print("=" * 60)

    predictor = _setup()
    if predictor is None:
        return False

    num_genes = predictor.get_num_genes()
    num_drugs = predictor.get_num_drugs()
    emb_dim = predictor.get_drug_embedding_dim()
    print(f"\n基因数: {num_genes}")
    print(f"嵌入文件药物总数: {num_drugs}")
    print(f"嵌入维度: {emb_dim}")

    batch_size = 32
    control_genes = np.random.randn(batch_size, num_genes).astype(np.float32)

    # drug_idx = 0 指 parquet 第 0 行的药物
    prediction = predictor.predict(
        control_expression=control_genes,
        drug_idx=0,
        dosage=1.0,
    )

    print(f"\n预测结果:")
    print(f"  输入形状: {control_genes.shape}")
    print(f"  输出形状: {prediction.shape}")
    print(f"  均值范围: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print(f"  均值的均值: {prediction.mean():.4f}")

    assert prediction.shape == (batch_size, num_genes)
    assert not np.isnan(prediction).any(), "输出包含 NaN"
    print("  ✓ 基本推理通过")
    return True


def test_batch_predict():
    """测试 2: 批量多药物预测"""
    print("\n" + "=" * 60)
    print("测试 2: 批量多药物预测 (idx)")
    print("=" * 60)

    predictor = _setup()
    if predictor is None:
        return False

    num_genes = predictor.get_num_genes()
    batch_size = 16
    control_genes = np.random.randn(batch_size, num_genes).astype(np.float32)

    # 取 3 个间隔较远的索引，验证不同药物给出不同结果
    results = predictor.predict_batch(
        control_expression=control_genes,
        drug_indices=[0, 100, 5000],
        dosages=[1.0, 0.5, 2.0],
    )

    print(f"批量预测结果 ({len(results)} 种药物):")
    for drug_idx, pred in results.items():
        print(f"  drug_idx={drug_idx} -> shape={pred.shape}, mean={pred.mean():.4f}")

    assert len(results) == 3
    preds = list(results.values())
    for pred in preds:
        assert pred.shape == (batch_size, num_genes)
        assert not np.isnan(pred).any()

    # 不同药物结果应不同
    diff = np.abs(preds[0] - preds[1]).max()
    print(f"  drug 0 vs drug 100 最大差异: {diff:.6f}")
    assert diff > 1e-6, "不同药物预测结果相同"
    print("  ✓ 批量预测通过")
    return True


def test_smiles_predict():
    """测试 3: 通过 SMILES 推理 + 与 idx 的一致性"""
    print("\n" + "=" * 60)
    print("测试 3: 通过 SMILES 推理 + 与 idx 的一致性")
    print("=" * 60)

    predictor = _setup()
    if predictor is None:
        return False

    num_genes = predictor.get_num_genes()
    batch_size = 8
    control_genes = np.random.randn(batch_size, num_genes).astype(np.float32)

    # 取第 42 号药物的 SMILES
    test_idx = 42
    smiles = predictor.idx_to_smiles(test_idx)
    print(f"  药物索引 {test_idx} 的 SMILES: {smiles[:60]}...")

    # 两种方式推理应完全一致
    pred_via_idx = predictor.predict(control_genes, drug_idx=test_idx, dosage=1.0)
    pred_via_smi = predictor.predict_by_smiles(control_genes, smiles=smiles, dosage=1.0)

    diff = np.abs(pred_via_idx - pred_via_smi).max()
    print(f"  idx 推理 shape: {pred_via_idx.shape}")
    print(f"  SMILES 推理 shape: {pred_via_smi.shape}")
    print(f"  两种方式最大差异: {diff:.8f}")

    assert diff < 1e-6, f"idx 和 SMILES 推理结果不一致: max_diff={diff}"
    print("  ✓ SMILES 推理一致性通过")
    return True


def test_raw_embedding_predict():
    """测试 4: 直接使用嵌入向量推理 + 与 idx 的一致性"""
    print("\n" + "=" * 60)
    print("测试 4: 直接使用嵌入向量推理")
    print("=" * 60)

    predictor = _setup()
    if predictor is None:
        return False

    num_genes = predictor.get_num_genes()
    batch_size = 8
    control_genes = np.random.randn(batch_size, num_genes).astype(np.float32)

    # 从查找表中取第 42 号嵌入
    test_idx = 42
    drug_emb = predictor.get_drug_embedding(test_idx)  # (194,)
    print(f"  嵌入维度: {drug_emb.shape}")

    pred_via_idx = predictor.predict(control_genes, drug_idx=test_idx, dosage=1.0)
    pred_via_emb = predictor.predict_with_raw_embedding(
        control_genes, drug_embedding=drug_emb, dosage=1.0,
    )

    diff = np.abs(pred_via_idx - pred_via_emb).max()
    print(f"  idx 推理 shape: {pred_via_idx.shape}")
    print(f"  raw_embedding 推理 shape: {pred_via_emb.shape}")
    print(f"  两种方式最大差异: {diff:.8f}")

    assert diff < 1e-4, f"两种推理方式结果不一致: max_diff={diff}"
    print("  ✓ 嵌入推理一致性通过")
    return True


def test_different_batch_sizes():
    """测试 5: 不同 batch size 鲁棒性"""
    print("\n" + "=" * 60)
    print("测试 5: 不同 batch size 鲁棒性")
    print("=" * 60)

    predictor = _setup()
    if predictor is None:
        return False

    num_genes = predictor.get_num_genes()

    for bs in [1, 2, 16, 64, 256]:
        control = np.random.randn(bs, num_genes).astype(np.float32)
        pred = predictor.predict(control, drug_idx=0, dosage=1.0)
        assert pred.shape == (bs, num_genes), f"batch_size={bs} 时形状错误"
        assert not np.isnan(pred).any(), f"batch_size={bs} 时包含 NaN"
        print(f"  batch_size={bs}: ✓")

    # 测试单个样本（1D 输入）
    single = np.random.randn(num_genes).astype(np.float32)
    pred = predictor.predict(single, drug_idx=0, dosage=1.0)
    assert pred.shape == (1, num_genes)
    print(f"  single sample (1D input): ✓")

    print("  ✓ 鲁棒性测试通过")
    return True


def test_batch_by_smiles():
    """测试 6: 批量 SMILES 推理"""
    print("\n" + "=" * 60)
    print("测试 6: 批量 SMILES 推理")
    print("=" * 60)

    predictor = _setup()
    if predictor is None:
        return False

    num_genes = predictor.get_num_genes()
    batch_size = 16
    control_genes = np.random.randn(batch_size, num_genes).astype(np.float32)

    # 取 3 个不同药物的 SMILES
    smiles_list = [predictor.idx_to_smiles(i) for i in [10, 500, 10000]]
    results = predictor.predict_batch_by_smiles(
        control_expression=control_genes,
        smiles_list=smiles_list,
        dosages=[1.0, 0.5, 2.0],
    )

    print(f"批量 SMILES 预测结果 ({len(results)} 种药物):")
    for smi, pred in results.items():
        print(f"  {smi[:40]}... -> shape={pred.shape}, mean={pred.mean():.4f}")
        assert pred.shape == (batch_size, num_genes)
        assert not np.isnan(pred).any()

    print("  ✓ 批量 SMILES 推理通过")
    return True


if __name__ == "__main__":
    tests = [
        ("加载与基本推理 (idx)", test_load_and_predict),
        ("批量多药物预测 (idx)", test_batch_predict),
        ("SMILES 推理一致性", test_smiles_predict),
        ("嵌入向量推理一致性", test_raw_embedding_predict),
        ("不同 batch size 鲁棒性", test_different_batch_sizes),
        ("批量 SMILES 推理", test_batch_by_smiles),
    ]

    results = []
    for name, fn in tests:
        passed = fn()
        results.append((name, passed))

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 跳过/失败"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")

    if all_pass:
        print("\n所有测试通过！chemCPA_inference_kit 可以正常使用。")
    else:
        print("\n部分测试未通过，请检查预训练文件是否就绪。")
