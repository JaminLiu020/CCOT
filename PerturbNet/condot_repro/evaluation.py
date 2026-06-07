from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def ensure_results_dirs(output_root: str, experiment_name: str) -> Dict[str, Path]:
    root = Path(output_root) / experiment_name
    eval_dir = root / "eval"
    data_dir = eval_dir / "data"
    plot_dir = eval_dir / "plot"

    for p in [root, eval_dir, data_dir, plot_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "eval": eval_dir,
        "data": data_dir,
        "plot": plot_dir,
    }


def save_group_arrays(
    *,
    data_dir: Path,
    stage: str,
    condition: str,
    cell_type: str,
    real: np.ndarray,
    pred: np.ndarray,
    source: np.ndarray,
    subset_name: str,
) -> Path:
    outdir = data_dir / f"{stage}_{subset_name}" / condition
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{cell_type}.npz"
    np.savez(outpath, real=real, pred=pred, source=source)
    return outpath


def save_stage_metrics(
    *,
    eval_dir: Path,
    stage: str,
    stage_metrics: pd.DataFrame,
    subset_name: str,
) -> Path:
    outpath = eval_dir / f"{stage}_{subset_name}_results.csv"
    stage_metrics.to_csv(outpath)
    return outpath


def save_summary_metrics(
    *,
    eval_dir: Path,
    summary_df: pd.DataFrame,
    subset_name: str,
) -> Path:
    outpath = eval_dir / f"summary_{subset_name}_results.csv"
    summary_df.to_csv(outpath)
    return outpath
