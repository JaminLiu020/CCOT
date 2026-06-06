# SMILES Reference Files

本目录保存小型、可复用的 SMILES 参考表和其规范化结果。

- `label.csv`: 原始标签表
- `label_canonical_unique.csv`: 对 `label.csv` 做 canonical SMILES 规范化并按 SMILES 去重后的结果
- `sciplex3_drugs.csv`: sciplex3 药物原始 SMILES 表
- `sciplex3_drugs_canonical.csv`: sciplex3 药物 canonical SMILES 结果
- `sciplex3_genes.csv`: 与该整理过程一起使用的基因列表

重新生成规范化结果：

```bash
python scripts/data/convert_smiles.py
```

依赖说明：

- 该脚本依赖 `pandas` 和 `rdkit`
- 当前仓库主 `requirements.txt` 不包含 `rdkit`，如需重跑请额外安装对应环境依赖
