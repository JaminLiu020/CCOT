#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def generate_cluster_data(centers, params, n_samples=1000):
    data = []
    labels = []
    for i, (center, (a, b, angle)) in enumerate(zip(centers, params)):
        r = np.sqrt(np.random.uniform(0.0, 1.0, n_samples))
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        x_ellipse = a * x_circle
        y_ellipse = b * y_circle
        x = center[0] + (x_ellipse * np.cos(angle) - y_ellipse * np.sin(angle))
        y = center[1] + (x_ellipse * np.sin(angle) + y_ellipse * np.cos(angle))
        x += np.random.normal(0, a * 0.05, n_samples)
        y += np.random.normal(0, b * 0.05, n_samples)
        data.append(np.column_stack((x, y)))
        labels.extend([f"Cell type {chr(65 + i)}"] * n_samples)
    return np.vstack(data), labels


def build_demo_data():
    np.random.seed(42)
    source_data, source_labels = generate_cluster_data(
        [(-8, 6), (3, -5), (8, 7)],
        [(3.5, 2.2, np.pi / 4), (3.0, 1.8, np.pi / 2), (4.0, 2.5, np.pi / 6)],
    )
    target_data, _ = generate_cluster_data(
        [(5, 5), (-7, -6), (0, 8)],
        [(3.2, 2.0, np.pi / 3), (3.8, 2.3, np.pi), (3.0, 1.5, np.pi / 2.5)],
    )
    return source_data, target_data, source_labels


def visualize_transport_map(source_data, target_data, labels, output_path: Path):
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_position([0.08, 0.05, 0.38, 0.85])
    ax2.set_position([0.58, 0.05, 0.38, 0.85])

    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    source_by_label = {label: [] for label in unique_labels}
    target_by_label = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        source_by_label[label].append(source_data[idx])
        target_by_label[label].append(target_data[idx])

    for label in unique_labels:
        source_points = np.asarray(source_by_label[label])
        target_points = np.asarray(target_by_label[label])
        ax1.scatter(source_points[:, 0], source_points[:, 1], c=[color_map[label]], alpha=0.8, s=40, edgecolors="white", linewidths=0.5)
        ax2.scatter(target_points[:, 0], target_points[:, 1], c=[color_map[label]], alpha=0.8, s=40, edgecolors="white", linewidths=0.5)

    legend_elements = [plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color_map[label], markersize=10) for label in unique_labels]
    ax1.legend(handles=legend_elements, loc="best")
    ax1.axis("off")
    ax2.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a synthetic transport-map demo figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "scripts" / "visualization_results" / "examples" / "synthetic_transport_map.svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_data, target_data, labels = build_demo_data()
    visualize_transport_map(source_data, target_data, labels, args.output)


if __name__ == "__main__":
    main()
