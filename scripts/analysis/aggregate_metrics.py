#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment


SUMMARY_FILES = {
    "summary_results.csv": {
        "sheet_name": "Metrics Summary",
        "combined_name": "combined_all_folders.xlsx",
        "drop_columns": ["Loss_f", "Loss_g", "dist"],
        "suffix": "_combined_metrics_summary.xlsx",
    },
    "DEGs_summary_results.csv": {
        "sheet_name": "DEGs Metrics Summary",
        "combined_name": "DEGs_combined_all_folders.xlsx",
        "drop_columns": [],
        "suffix": "_combined_deg_metrics_summary.xlsx",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_metrics(csv_path: Path, drop_columns: list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path, index_col=0)
    if drop_columns:
        frame = frame.drop(columns=drop_columns, errors="ignore")
    return frame.dropna(axis=1, how="all")


def save_excel(frame: pd.DataFrame, output_path: Path, sheet_name: str) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=True, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for column in worksheet.columns:
            max_length = 0
            for cell in column:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))
            worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 40)


def process_metric_tree(root_dir: Path, eval_subdir: str, filename: str) -> None:
    config = SUMMARY_FILES[filename]
    combined_frames: list[pd.DataFrame] = []
    for run_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        csv_path = run_dir / eval_subdir / filename
        frame = load_metrics(csv_path, config["drop_columns"])
        if frame.empty:
            continue
        output_path = run_dir / f"{run_dir.name}{config['suffix']}"
        save_excel(frame, output_path, config["sheet_name"])
        frame.insert(0, "Folder", run_dir.name)
        combined_frames.append(frame)

    if not combined_frames:
        print(f"No '{filename}' files found under {root_dir}")
        return

    combined = pd.concat(combined_frames, ignore_index=False)
    save_excel(combined, root_dir / config["combined_name"], config["sheet_name"])
    print(f"Wrote {root_dir / config['combined_name']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment metric CSV files into Excel summaries.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=repo_root() / "results",
        help="Directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--eval-subdir",
        default="eval",
        help="Subdirectory below each run directory that stores summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root_dir.exists():
        raise SystemExit(f"Root directory does not exist: {args.root_dir}")
    for filename in SUMMARY_FILES:
        process_metric_tree(args.root_dir, args.eval_subdir, filename)


if __name__ == "__main__":
    main()
