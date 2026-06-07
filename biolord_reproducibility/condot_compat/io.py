from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def build_experiment_dir(base_dir: str, tag: str, seed: int, batch_size: int, epochs: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = Path(base_dir) / f"{tag}_seed{seed}_bs{batch_size}_ep{epochs}_{ts}"
    exp.mkdir(parents=True, exist_ok=True)
    return exp


def save_config_snapshot(config: dict, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_df(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
