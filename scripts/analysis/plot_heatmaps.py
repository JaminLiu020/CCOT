#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


DEFAULT_MODELS = ["CCOT", "CELLOT", "CondOT_cell_type", "pretrained_chemCPA"]
DEFAULT_CELL_TYPES = ["A549", "K562", "MCF7"]
DEFAULT_DRUGS = [
    "Hesperadin",
    "TAK-901",
    "Dacinostat",
    "Givinostat",
    "Belinostat",
    "Quisinostat",
    "Alvespimycin",
    "Tanespimycin",
    "Flavopiridol",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def existing_input_root() -> Path:
    candidates = [
        repo_root() / "old_result" / "nine_drugs" / "977genes" / "drug_pert" / "visualization_exp",
        repo_root() / "results" / "nine_drugs" / "977genes" / "drug_pert" / "visualization_exp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def pick_font(candidates: list[str], fallback: str | None = None) -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return fallback


ENGLISH_FONT = pick_font(["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "DejaVu Serif")
CHINESE_FONT = pick_font(["SimSun", "Noto Serif CJK SC", "AR PL UMing CN", "WenQuanYi Zen Hei"])
CHINESE_PROPS = font_manager.FontProperties(family=CHINESE_FONT) if CHINESE_FONT else None
plt.rcParams["font.family"] = ENGLISH_FONT
plt.rcParams["font.serif"] = [ENGLISH_FONT]
plt.rcParams["axes.unicode_minus"] = False


def load_data_for_drug(input_dir: Path, model: str, test_set: str, drug: str, cell_types: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[None, None, None]:
    all_real: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    labels: list[str] = []
    for cell_type in cell_types:
        file_path = input_dir / model / test_set / drug / f"{cell_type}.npz"
        if not file_path.exists():
            continue
        data = np.load(file_path)
        real = np.asarray(data["real"])
        pred = np.asarray(data["pred"])
        all_real.append(real)
        all_pred.append(pred)
        labels.extend([cell_type] * len(real))
    if not all_real:
        return None, None, None
    return np.vstack(all_real), np.vstack(all_pred), labels


def create_violin_plots(input_dir: Path, output_dir: Path, models: list[str], test_set: str, drugs: list[str], cell_types: list[str], top_n_genes: int) -> None:
    violin_dir = output_dir / "violin"
    violin_dir.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        for model in models:
            loaded = load_data_for_drug(input_dir, model, test_set, drug, cell_types)
            if loaded[0] is None:
                continue
            real_data, pred_data, labels = loaded
            gene_var = np.var(real_data, axis=0)
            top_gene_indices = np.argsort(gene_var)[-top_n_genes:]
            plt.figure(figsize=(4 * top_n_genes, 6))
            for i, gene_idx in enumerate(top_gene_indices, start=1):
                plt.subplot(1, top_n_genes, i)
                frame = pd.DataFrame(
                    {
                        "Expression": np.concatenate([real_data[:, gene_idx], pred_data[:, gene_idx]]),
                        "Type": ["Real"] * len(real_data) + ["Predicted"] * len(pred_data),
                        "Cell Type": labels * 2,
                    }
                )
                sns.violinplot(x="Cell Type", y="Expression", hue="Type", data=frame, split=True)
                plt.title(f"Gene {gene_idx}")
                plt.xticks(rotation=45)
                plt.ylabel("Expression" if i == 1 else "")
            plt.tight_layout()
            output_path = violin_dir / f"{model}_{drug}_violin_plot.pdf"
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()


def create_heatmaps(input_dir: Path, output_dir: Path, models: list[str], test_set: str, drugs: list[str], cell_types: list[str]) -> None:
    heatmap_dir = output_dir / "heatmap"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        all_model_data: dict[str, dict[str, np.ndarray]] = {}
        for model in models:
            loaded = load_data_for_drug(input_dir, model, test_set, drug, cell_types)
            if loaded[0] is None:
                continue
            real_data, pred_data, labels = loaded
            label_array = np.asarray(labels)
            unique_types = np.unique(label_array)
            real_means = [np.mean(real_data[label_array == cell_type], axis=0) for cell_type in unique_types]
            pred_means = [np.mean(pred_data[label_array == cell_type], axis=0) for cell_type in unique_types]
            all_model_data[model] = {
                "real": np.asarray(real_means),
                "pred": np.asarray(pred_means),
                "diff": np.asarray(pred_means) - np.asarray(real_means),
                "cell_types": unique_types,
            }
        if not all_model_data:
            continue
        expr_min = min(min(data["real"].min(), data["pred"].min()) for data in all_model_data.values())
        expr_max = max(max(data["real"].max(), data["pred"].max()) for data in all_model_data.values())
        diff_max = max(
            max(abs(data["diff"].min()), abs(data["diff"].max()))
            for data in all_model_data.values()
        )
        for model, data in all_model_data.items():
            for kind, cmap, vmin, vmax in (
                ("real", "viridis", expr_min, expr_max),
                ("pred", "viridis", expr_min, expr_max),
                ("diff", "RdBu_r", -diff_max, diff_max),
            ):
                plt.figure(figsize=(10, 6))
                sns.heatmap(
                    data[kind],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    center=0 if kind == "diff" else None,
                    xticklabels=False,
                    yticklabels=data["cell_types"],
                )
                plt.xlabel("基因", fontsize=18, fontproperties=CHINESE_PROPS)
                plt.ylabel("细胞类型", fontsize=18, fontproperties=CHINESE_PROPS)
                output_path = heatmap_dir / f"{model}_{drug}_{kind}_heatmap_cn.pdf"
                plt.tight_layout()
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot heatmaps and violin plots from visualization_exp NPZ files.")
    parser.add_argument("--input-dir", type=Path, default=existing_input_root(), help="Root directory containing model/test_set/drug/*.npz files.")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "scripts" / "visualization_results", help="Directory for generated figures.")
    parser.add_argument("--test-set", default="ood_DEGs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--cell-types", nargs="+", default=DEFAULT_CELL_TYPES)
    parser.add_argument("--drugs", nargs="+", default=DEFAULT_DRUGS)
    parser.add_argument("--include-violin", action="store_true", help="Also render violin plots.")
    parser.add_argument("--top-n-genes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_heatmaps(args.input_dir, args.output_dir, args.models, args.test_set, args.drugs, args.cell_types)
    if args.include_violin:
        create_violin_plots(args.input_dir, args.output_dir, args.models, args.test_set, args.drugs, args.cell_types, args.top_n_genes)


if __name__ == "__main__":
    main()
