#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.patches import FancyBboxPatch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_dir() -> Path:
    candidates = [
        repo_root() / "results" / "nine_drugs" / "977genes" / "drug_pert" / "visualization_exp" / "CCOT" / "ood_DEGs",
        repo_root() / "old_result" / "nine_drugs" / "977genes" / "drug_pert" / "visualization_exp" / "CCOT" / "ood_DEGs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_output_dir() -> Path:
    return repo_root() / "scripts" / "visualization_results" / "transport_map"


def load_drug_arrays(drug_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    source_arrays: list[np.ndarray] = []
    real_arrays: list[np.ndarray] = []
    cell_types: list[str] = []
    for file_path in sorted(drug_dir.glob("*.npz")):
        cell_type = file_path.stem
        data = np.load(file_path)
        if "real" not in data or "source" not in data:
            continue
        real = np.asarray(data["real"])
        source = np.asarray(data["source"])
        real_arrays.append(real)
        source_arrays.append(source)
        cell_types.extend([cell_type] * len(real))
    if not source_arrays or not real_arrays:
        return None
    return np.vstack(source_arrays), np.vstack(real_arrays), cell_types


def embed_pair(source_data: np.ndarray, real_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reducer = umap.UMAP(spread=3.0, n_neighbors=10, min_dist=0.5, n_components=2, random_state=42, n_jobs=1)
    return reducer.fit_transform(source_data), reducer.fit_transform(real_data)


def get_figure_coordinate(ax: plt.Axes, bbox, x: float, y: float) -> tuple[float, float]:
    x_rel = (x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_rel = (y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    return bbox.x0 + x_rel * bbox.width, bbox.y0 + y_rel * bbox.height


def save_transport_map(source_embedding: np.ndarray, target_embedding: np.ndarray, cell_types: list[str], drug_name: str, output_dir: Path, file_format: str) -> None:
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1_pos = [0.09, 0.05, 0.38, 0.85]
    ax2_pos = [0.56, 0.05, 0.38, 0.85]
    ax1.set_position(ax1_pos)
    ax2.set_position(ax2_pos)

    unique_cell_types = sorted(set(cell_types))
    custom_colors = [
        (230 / 255, 111 / 255, 81 / 255),
        (42 / 255, 157 / 255, 140 / 255),
        (82 / 255, 143 / 255, 173 / 255),
    ]
    color_map = {cell_type: custom_colors[i % len(custom_colors)] for i, cell_type in enumerate(unique_cell_types)}
    indices_by_type = {cell_type: [] for cell_type in unique_cell_types}
    for idx, cell_type in enumerate(cell_types):
        indices_by_type[cell_type].append(idx)

    for axis, embedding in ((ax1, source_embedding), (ax2, target_embedding)):
        for cell_type, indices in indices_by_type.items():
            axis.scatter(
                embedding[indices, 0],
                embedding[indices, 1],
                c=[color_map[cell_type]],
                alpha=0.8,
                s=40,
                edgecolors="white",
                linewidths=0.5,
            )
        axis.axis("off")
        axis.add_patch(
            FancyBboxPatch(
                (-0.05, -0.05),
                width=1.1,
                height=1.1,
                boxstyle="round,pad=0.05",
                ec="black",
                fc="none",
                linewidth=2.0,
                linestyle="--",
                clip_on=False,
                transform=axis.transAxes,
            )
        )

    for axis, embedding in ((ax1, source_embedding), (ax2, target_embedding)):
        min_x, max_x = embedding[:, 0].min(), embedding[:, 0].max()
        min_y, max_y = embedding[:, 1].min(), embedding[:, 1].max()
        margin = 0.1
        axis.set_xlim(min_x - margin * (max_x - min_x), max_x + margin * (max_x - min_x))
        axis.set_ylim(min_y - margin * (max_y - min_y), max_y + margin * (max_y - min_y))

    ax1_bbox = ax1.get_position()
    ax2_bbox = ax2.get_position()
    for cell_type, indices in indices_by_type.items():
        for idx in indices:
            source_x_fig, source_y_fig = get_figure_coordinate(ax1, ax1_bbox, *source_embedding[idx])
            target_x_fig, target_y_fig = get_figure_coordinate(ax2, ax2_bbox, *target_embedding[idx])
            line = plt.Line2D(
                [source_x_fig, target_x_fig],
                [source_y_fig, target_y_fig],
                transform=fig.transFigure,
                color=color_map[cell_type],
                alpha=0.2,
                linewidth=0.5,
                zorder=0,
            )
            fig.add_artist(line)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{drug_name}_combined_umap_embedding.{file_format}"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def process_all_drugs(input_dir: Path, output_dir: Path, file_format: str) -> None:
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    for drug_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        arrays = load_drug_arrays(drug_dir)
        if arrays is None:
            continue
        source_data, real_data, cell_types = arrays
        source_embedding, target_embedding = embed_pair(source_data, real_data)
        save_transport_map(source_embedding, target_embedding, cell_types, drug_dir.name, output_dir / drug_dir.name, file_format)
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create transport-map visualizations from evaluation NPZ files.")
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--format", default="svg", choices=["pdf", "svg", "eps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_all_drugs(args.input_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
