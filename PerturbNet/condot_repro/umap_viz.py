from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from umap import UMAP


def plot_condot_style_umap(
    *,
    real: np.ndarray,
    pred: np.ndarray,
    real_labels: np.ndarray,
    pred_labels: np.ndarray,
    condition: str,
    stage: str,
    save_dir: Path,
    random_seed: int = 42,
) -> Path:
    """Create CondOT-like combined UMAP with real faint points + pred highlighted."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # UMAP requires n_neighbors < n_samples; many perturbation groups are tiny.
    # Adapt to group size to avoid noisy runtime warnings.
    n_real = int(np.asarray(real).shape[0])
    if n_real < 2:
        raise ValueError("UMAP requires at least 2 real samples.")
    n_neighbors = max(2, min(15, n_real - 1))

    umap_kwargs = {
        "n_components": 2,
        "random_state": random_seed,
        "init": "random",
        "spread": 2.0,
        "min_dist": 0.4,
        "n_neighbors": n_neighbors,
    }
    # Compatibility across umap-learn versions: older versions do not accept n_jobs.
    try:
        reducer = UMAP(**{**umap_kwargs, "n_jobs": 1})
    except TypeError:
        reducer = UMAP(**umap_kwargs)
    reducer.fit(real)
    real_2d = reducer.transform(real)
    pred_2d = reducer.transform(pred)

    cmap = ListedColormap(["blue", "red", "orange"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        real_2d[:, 0],
        real_2d[:, 1],
        c=real_labels,
        cmap=cmap,
        s=20,
        alpha=0.12,
        edgecolor="none",
    )
    ax.scatter(
        pred_2d[:, 0],
        pred_2d[:, 1],
        c=pred_labels,
        cmap=cmap,
        s=24,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.2,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{stage} | {condition}")

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="A549"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="K562"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="MCF7"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    outpath = save_dir / f"{stage}_{condition}_combined_umap_embedding.pdf"
    fig.tight_layout()
    fig.savefig(outpath, format="pdf", dpi=300)
    plt.close(fig)
    return outpath
