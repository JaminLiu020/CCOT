from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from umap import UMAP


CELL_TYPE_TO_LABEL = {
    "A549": 0,
    "K562": 1,
    "MCF7": 2,
}


def _plot_combined(real, pred, real_labels, pred_labels, out_path: Path) -> None:
    reducer = UMAP(n_components=2, random_state=42, spread=2.0, min_dist=0.4, n_neighbors=15, n_jobs=1)
    reducer.fit(real)
    real_2d = reducer.transform(real)
    pred_2d = reducer.transform(pred)

    cmap = ListedColormap(["blue", "red", "orange"])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c=pred_labels, cmap=cmap, s=30, alpha=0.8, edgecolor="k")
    ax.scatter(real_2d[:, 0], real_2d[:, 1], c=real_labels, cmap=cmap, s=30, alpha=0.1, edgecolor="k")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=16, label="A549"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=16, label="K562"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=16, label="MCF7"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=18)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def generate_umap_from_eval_data(eval_dir: str, test_sets=("test_DEGs", "ood_DEGs")) -> None:
    eval_path = Path(eval_dir)
    data_dir = eval_path / "data"
    plot_dir = eval_path / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for split in test_sets:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        split_plot_dir = plot_dir / split
        split_plot_dir.mkdir(parents=True, exist_ok=True)

        for pert_dir in sorted(split_dir.iterdir()):
            if not pert_dir.is_dir():
                continue

            real_data = []
            pred_data = []
            real_labels = []
            pred_labels = []

            for cell_type, label in CELL_TYPE_TO_LABEL.items():
                npz_path = pert_dir / f"{cell_type}.npz"
                if not npz_path.exists():
                    continue

                arr = np.load(npz_path)
                real = arr["real"]
                pred = arr["pred"]
                real_data.append(real)
                pred_data.append(pred)
                real_labels.append(np.full(real.shape[0], label))
                pred_labels.append(np.full(pred.shape[0], label))

            if not real_data or not pred_data:
                continue

            real_data = np.vstack(real_data)
            pred_data = np.vstack(pred_data)
            real_labels = np.concatenate(real_labels)
            pred_labels = np.concatenate(pred_labels)

            out_file = split_plot_dir / f"local_{pert_dir.name}_combined_umap_embedding.pdf"
            _plot_combined(real_data, pred_data, real_labels, pred_labels, out_file)
