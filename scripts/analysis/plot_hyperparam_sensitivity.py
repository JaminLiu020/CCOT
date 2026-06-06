#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


METRIC_COLUMNS = {
    "MMD": ("MMD_mean", "MMD_std", "MMD"),
    "L2PS": ("L2PS_mean", "L2PS_std", r"$\ell_2(PS)$"),
    "R2": ("R2_mean", "R2_std", r"$R^2$"),
    "Ed": ("Ed_mean", "Ed_std", r"$E_d$"),
    "FID": ("FID_mean", "FID_std", "FID"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_file() -> Path:
    candidates = [
        repo_root() / "old_result" / "nine_drugs" / "977genes" / "drug_pert" / "hyperpara_sensitivity_analysis" / "DEGs_id_test.xlsx",
        repo_root() / "results" / "nine_drugs" / "977genes" / "drug_pert" / "hyperpara_sensitivity_analysis" / "DEGs_id_test.xlsx",
        repo_root() / "scripts" / "visualization_results" / "hyperpara_sensitivity_analysis" / "legacy" / "id_test.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_output_dir() -> Path:
    return repo_root() / "scripts" / "visualization_results" / "hyperpara_sensitivity_analysis" / "DEGs_id_test"


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
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] = False
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hyperparameter sensitivity curves from an Excel summary.")
    parser.add_argument("--input-file", type=Path, default=default_input_file())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_file.exists():
        raise SystemExit(f"Input file does not exist: {args.input_file}")
    frame = pd.read_excel(args.input_file)
    beta_col = frame.columns[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, (mean_col, std_col, latex_label) in METRIC_COLUMNS.items():
        if mean_col not in frame.columns or std_col not in frame.columns:
            print(f"Skipping {metric_name}: missing columns {mean_col}/{std_col}")
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(frame[beta_col], frame[mean_col], marker="o", linestyle="-", label=metric_name)
        lower = frame[mean_col] - frame[std_col]
        upper = frame[mean_col] + frame[std_col]
        ax.fill_between(frame[beta_col], lower, upper, color=ax.lines[-1].get_color(), alpha=0.15)
        ax.set_xlabel(r"超参数 $\beta$", fontsize=18, fontproperties=CHINESE_PROPS)
        ax.set_ylabel(latex_label, fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.6)

        x_numeric = pd.to_numeric(frame[beta_col], errors="coerce").dropna()
        if not x_numeric.empty:
            tick_min = np.floor(x_numeric.min())
            tick_max = min(np.ceil(x_numeric.max()), 15)
            xticks = np.arange(tick_min, tick_max + 1.0, 1.0)
            ax.set_xticks(xticks)
            ax.set_xlim(tick_min, 15)

        output_path = args.output_dir / f"{metric_name}_sensitivity_cn.pdf"
        plt.tight_layout()
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
