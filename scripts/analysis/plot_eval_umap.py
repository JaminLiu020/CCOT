#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from umap import UMAP

matplotlib.use("Agg")


CELL_TYPE_TO_LABEL = {"A549": 0, "K562": 1, "MCF7": 2}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def embed_pair(target: np.ndarray, transport: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reducer = UMAP(n_components=2, random_state=42, spread=2.0, min_dist=0.4, n_neighbors=15, n_jobs=1)
    reducer.fit(target)
    return reducer.transform(target), reducer.transform(transport)


def plot_combined_scatter(
    target: np.ndarray,
    target_labels: np.ndarray,
    transport: np.ndarray,
    transport_labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    colors = ["#3b82f6", "#ef4444", "#f59e0b"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(transport[:, 0], transport[:, 1], c=transport_labels, cmap=cmap, s=30, alpha=0.8, edgecolor="k")
    ax.scatter(target[:, 0], target[:, 1], c=target_labels, cmap=cmap, s=30, alpha=0.12, edgecolor="k")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label)
        for label, color in zip(CELL_TYPE_TO_LABEL, colors)
    ]
    ax.legend(handles=legend, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def collect_drug_arrays(drug_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    real_data = []
    pred_data = []
    labels = []
    for cell_type, label in CELL_TYPE_TO_LABEL.items():
        cell_file = drug_dir / f"{cell_type}.npz"
        if not cell_file.exists():
            continue
        loaded = np.load(cell_file)
        real = to_numpy(loaded["real"])
        pred = to_numpy(loaded["pred"])
        real_data.append(real)
        pred_data.append(pred)
        labels.append(np.full(real.shape[0], label))
    if not real_data or not pred_data:
        return None
    return np.vstack(real_data), np.vstack(pred_data), np.concatenate(labels)


def process_eval_dir(eval_dir: Path, output_dir: Path, test_set: str) -> None:
    data_dir = eval_dir / "data" / test_set
    if not data_dir.exists():
        raise SystemExit(f"Missing evaluation data directory: {data_dir}")
    for drug_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        arrays = collect_drug_arrays(drug_dir)
        if arrays is None:
            continue
        real, pred, labels = arrays
        embedded_real, embedded_pred = embed_pair(real, pred)
        output_path = output_dir / test_set / f"local_{drug_dir.name}_combined_umap_embedding.pdf"
        plot_combined_scatter(embedded_real, labels, embedded_pred, labels, drug_dir.name, output_path)
        print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create UMAP overlays from evaluation NPZ files.")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Evaluation directory containing a data/ tree.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "scripts" / "visualization_results" / "eval_umap",
        help="Directory where plots will be written.",
    )
    parser.add_argument("--test-set", default="ood_DEGs", help="Evaluation split name under eval_dir/data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_eval_dir(args.eval_dir, args.output_dir, args.test_set)


if __name__ == "__main__":
    main()
