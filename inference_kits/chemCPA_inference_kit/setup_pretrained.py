"""
设置脚本：将预训练模型和嵌入文件复制到 pretrained/ 目录。

使用方法:
    python setup_pretrained.py /path/to/condot
    
    或在 condot 项目目录下直接运行:
    python setup_pretrained.py
"""

import shutil
from pathlib import Path


def setup(project_root: str = None):
    """
    从 condot 项目中复制必要的预训练文件。

    Args:
        project_root: condot 项目根目录路径。如果为 None，则尝试自动检测。
    """
    kit_dir = Path(__file__).parent
    pretrained_dir = kit_dir / "pretrained"
    pretrained_dir.mkdir(exist_ok=True)

    if project_root is None:
        # 尝试自动检测: kit 在 inference_kits/chemCPA_inference_kit/, 根在 ../../
        possible_root = kit_dir.parent.parent
        if (possible_root / "notebooks").exists():
            project_root = str(possible_root)
        else:
            print("无法自动检测项目根目录，请手动指定: python setup_pretrained.py /path/to/condot")
            return

    project_root = Path(project_root)

    files_to_copy = [
        # (源文件路径, 目标文件名)
        (
            project_root / "notebooks" / "evaluate_chemCPA" / "manual_2025-03-08_03-24-40.pt",
            "manual_2025-03-08_03-24-40.pt",
        ),
        (
            project_root / "embeddings" / "rdkit2D_embedding_lincs_trapnell.parquet",
            "rdkit2D_embedding_lincs_trapnell.parquet",
        ),
    ]

    for src, dst_name in files_to_copy:
        dst = pretrained_dir / dst_name
        if dst.exists():
            print(f"[跳过] 已存在: {dst}")
            continue

        if not src.exists():
            print(f"[警告] 源文件不存在: {src}")
            print(f"  请手动复制到: {dst}")
            continue

        print(f"[复制] {src.name} -> {dst}")
        shutil.copy2(src, dst)

    print(f"\n设置完成！预训练文件位于: {pretrained_dir}")
    print(f"文件列表:")
    for f in sorted(pretrained_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else None
    setup(root)
